import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class NodeResource:
    node_id: str
    node_type: str  # 'cloud' or 'edge'
    cpu_cores: int
    memory_gb: float
    gpu_memory_gb: float
    bandwidth_mbps: float
    
    def can_handle_layer(self, layer_size_mb: float, compute_flops: float) -> bool:
        """Check if node can handle a layer"""
        memory_available = self.gpu_memory_gb * 1024 if self.node_type == 'cloud' else self.memory_gb * 1024
        return layer_size_mb < memory_available * 0.8  # 80% utilization threshold

@dataclass
class LayerInfo:
    layer_idx: int
    layer_name: str
    size_mb: float
    compute_flops: float
    depends_on: List[int]  # Previous layer dependencies

class DynamicPartitioner:
    """
    Layer partitioning system for distributed inference
    """
    
    def __init__(self, nodes: List[NodeResource]):
        self.nodes = {node.node_id: node for node in nodes}
        self.placement_cache = {}
        
    def analyze_model(self, model: nn.Module) -> List[LayerInfo]:
        """Analyze model to extract layer information"""
        layers = []
        
        for idx, (name, module) in enumerate(model.named_modules()):
            if len(list(module.children())) == 0:  # Leaf modules only
                size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024 * 1024)
                
                # Estimate compute requirements
                if isinstance(module, nn.Linear):
                    compute_flops = module.in_features * module.out_features * 2  # Basic estimation
                elif isinstance(module, nn.MultiheadAttention):
                    compute_flops = module.embed_dim * module.embed_dim * 4  # Attention computation
                else:
                    compute_flops = size_mb * 1e6  # Fallback estimation
                
                layers.append(LayerInfo(
                    layer_idx=idx,
                    layer_name=name,
                    size_mb=size_mb,
                    compute_flops=compute_flops,
                    depends_on=[idx-1] if idx > 0 else []
                ))
        
        return layers
    
    def partition_layers(self, layers: List[LayerInfo]) -> Dict[str, List[int]]:
        """
        Partition layers across available nodes using dynamic programming
        """
        if not layers:
            return {}
        
        n_layers = len(layers)
        n_nodes = len(self.nodes)
        node_list = list(self.nodes.keys())
        
        # DP table: dp[i][j] = min cost to place layers 0..i-1 with layer i-1 on node j
        dp = np.full((n_layers + 1, n_nodes), float('inf'))
        parent = np.full((n_layers + 1, n_nodes), -1, dtype=int)
        
        # Initialize: can place first layer on any capable node
        for j, node_id in enumerate(node_list):
            node = self.nodes[node_id]
            if node.can_handle_layer(layers[0].size_mb, layers[0].compute_flops):
                dp[1][j] = self._compute_placement_cost(layers[0], node)
        
        # Fill DP table
        for i in range(2, n_layers + 1):
            layer = layers[i - 1]
            
            for j, curr_node_id in enumerate(node_list):
                curr_node = self.nodes[curr_node_id]
                
                if not curr_node.can_handle_layer(layer.size_mb, layer.compute_flops):
                    continue
                
                placement_cost = self._compute_placement_cost(layer, curr_node)
                
                for k, prev_node_id in enumerate(node_list):
                    if dp[i - 1][k] == float('inf'):
                        continue
                    
                    comm_cost = 0
                    if j != k:  # Different nodes
                        comm_cost = self._compute_communication_cost(
                            layers[i - 2], layer, 
                            self.nodes[prev_node_id], curr_node
                        )
                    
                    total_cost = dp[i - 1][k] + placement_cost + comm_cost
                    
                    if total_cost < dp[i][j]:
                        dp[i][j] = total_cost
                        parent[i][j] = k
        
        # Find optimal solution
        min_cost = float('inf')
        best_final_node = -1
        for j in range(n_nodes):
            if dp[n_layers][j] < min_cost:
                min_cost = dp[n_layers][j]
                best_final_node = j
        
        if best_final_node == -1:
            logger.error("No valid partitioning found")
            return {}
        
        # Reconstruct solution
        placement = {node_id: [] for node_id in self.nodes.keys()}
        curr_node_idx = best_final_node
        
        for i in range(n_layers, 0, -1):
            node_id = node_list[curr_node_idx]
            placement[node_id].append(i - 1)
            curr_node_idx = parent[i][curr_node_idx]
        
        # Reverse lists to maintain order
        for node_id in placement:
            placement[node_id].reverse()
        
        logger.info(f"Partitioning completed with cost: {min_cost:.2f}")
        return placement
    
    def _compute_placement_cost(self, layer: LayerInfo, node: NodeResource) -> float:
        """Compute cost of placing layer on node"""
        # Memory pressure cost
        memory_usage = layer.size_mb / (node.memory_gb * 1024)
        memory_cost = memory_usage ** 2  # Quadratic penalty for high usage
        
        # Compute cost (inverse of node capability)
        if node.node_type == 'cloud':
            compute_cost = layer.compute_flops / 1e12  # Normalize for cloud
        else:
            compute_cost = layer.compute_flops / 1e10  # Edge has less compute
        
        return memory_cost + compute_cost
    
    def _compute_communication_cost(self, prev_layer: LayerInfo, curr_layer: LayerInfo,
                                  prev_node: NodeResource, curr_node: NodeResource) -> float:
        """Compute communication cost between layers on different nodes"""
        if prev_node.node_id == curr_node.node_id:
            return 0
        
        # Data transfer size (simplified)
        transfer_size_mb = min(prev_layer.size_mb, curr_layer.size_mb) * 0.1  # Activation size estimation
        
        # Bandwidth bottleneck
        effective_bandwidth = min(prev_node.bandwidth_mbps, curr_node.bandwidth_mbps)
        transfer_time = transfer_size_mb * 8 / effective_bandwidth  # Convert MB to Mbits
        
        # Inter-tier penalty (cloud-edge communication is more expensive)
        tier_penalty = 1.0
        if (prev_node.node_type != curr_node.node_type):
            tier_penalty = 2.0
        
        return transfer_time * tier_penalty
    
    def get_partitioning_stats(self, placement: Dict[str, List[int]]) -> Dict:
        """Get statistics about the partitioning"""
        stats = {
            'total_layers': sum(len(layers) for layers in placement.values()),
            'nodes_used': len([k for k, v in placement.items() if v]),
            'cloud_layers': 0,
            'edge_layers': 0
        }
        
        for node_id, layer_indices in placement.items():
            if self.nodes[node_id].node_type == 'cloud':
                stats['cloud_layers'] += len(layer_indices)
            else:
                stats['edge_layers'] += len(layer_indices)
        
        return stats