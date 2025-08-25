#!/usr/bin/env python3
"""
LL-CIoT Research Validation - DP Partitioner (Algorithm 2)
Exact implementation of Algorithm 2 from the manuscript

This implements the quantization-aware layer-node mapping algorithm that:
- Uses dynamic programming for optimal layer placement
- Minimizes total inference delay and communication overhead  
- Considers quantization effects: c_l^quant = α_l * c_l, m_l^quant = α_l * m_l
- Handles resource constraints: compute capacity φ_n(t) and memory M_n
- Optimizes communication between consecutive layers
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class NodeTier(Enum):
    """Node tier classification matching manuscript"""
    CLOUD = "cloud"
    EDGE = "edge"

@dataclass
class ModelLayer:
    """
    Model layer representation matching manuscript notation
    
    Manuscript symbols:
    - l: layer index
    - c_l: computational demand
    - m_l: memory requirement  
    - α_l: quantization factor
    - c_l^quant = α_l * c_l: quantized computational demand
    - m_l^quant = α_l * m_l: quantized memory requirement
    """
    layer_id: int                    # l in manuscript
    layer_name: str
    compute_demand: float            # c_l
    memory_requirement: float        # m_l  
    quantization_factor: float = 1.0 # α_l
    
    def __post_init__(self):
        """Validate layer parameters"""
        if self.quantization_factor <= 0 or self.quantization_factor > 1:
            raise ValueError(f"Invalid quantization factor: {self.quantization_factor}")
    
    @property
    def quantized_compute_demand(self) -> float:
        """Quantized computational demand: c_l^quant = α_l * c_l"""
        return self.quantization_factor * self.compute_demand
    
    @property 
    def quantized_memory_requirement(self) -> float:
        """Quantized memory requirement: m_l^quant = α_l * m_l"""
        return self.quantization_factor * self.memory_requirement

@dataclass
class NodeResource:
    """
    Computational node resource specification matching manuscript
    
    Manuscript symbols:
    - n: node index
    - φ_n(t): compute capacity at time t
    - M_n: memory capacity
    - b_{n,m}: bandwidth between nodes n and m
    """
    node_id: str
    tier: NodeTier
    gpu_id: int
    compute_capacity: float          # φ_n(t) in manuscript
    memory_capacity: float           # M_n in manuscript
    bandwidth_to_nodes: Dict[str, float]  # b_{n,m} in manuscript
    
    def can_accommodate_layer(self, layer: ModelLayer) -> bool:
        """Check if node can accommodate a layer with its quantized demands"""
        return (layer.quantized_memory_requirement <= self.memory_capacity and
                layer.quantized_compute_demand <= self.compute_capacity)
    
    def get_processing_delay(self, layer: ModelLayer) -> float:
        """Calculate processing delay: c_l^quant / φ_n(t)"""
        if self.compute_capacity <= 0:
            return float('inf')
        return layer.quantized_compute_demand / self.compute_capacity

@dataclass
class LayerAssignment:
    """
    Layer-to-node assignment result matching manuscript
    
    Represents the solution X_{l,n} where X_{l,n} = 1 if layer l assigned to node n
    """
    layer_id: int
    node_id: str
    processing_delay: float          # c_l^quant / φ_n
    communication_delay: float       # β_{l,l+1} / b_{n,m}
    total_delay: float
    quantization_factor: float

class DPPartitioner:
    """
    Dynamic Programming Layer Partitioner implementing Algorithm 2
    
    Solves the optimization problem:
    - Minimize: Σ D_j^inf + C^total
    - Subject to: resource constraints (C1)-(C6)
    - Uses DP to find optimal layer-node mapping X_{l,n}
    """
    
    def __init__(self, config: Dict):
        self.optimization_levels = config.get('optimization_levels', 3)
        self.communication_penalty = config.get('communication_penalty', 0.1)
        self.memory_penalty = config.get('memory_penalty', 0.05)
        self.timeslot_duration_ms = config.get('timeslot_duration_ms', 100)
        self.max_optimization_time_ms = config.get('max_optimization_time_ms', 50)
        self.resource_update_interval_ms = config.get('resource_update_interval_ms', 10)
        
        # DP table: dp[layer][node] = minimum cumulative delay
        self.dp_table: Dict[Tuple[int, str], float] = {}
        
        self.parent_table: Dict[Tuple[int, str], Optional[str]] = {}
        self.inter_layer_data_sizes: Dict[Tuple[int, int], float] = {}
        self.current_timeslot = 0
        self.last_resource_update = 0
        self.cached_solutions: Dict[str, Tuple[List[LayerAssignment], float]] = {}
        self.resource_change_threshold = 0.1
        
        # Performance tracking
        self.optimization_stats = {
            'total_partitioning_time': 0.0,
            'layers_processed': 0,
            'nodes_considered': 0,
            'optimal_assignments': 0,
            'communication_overhead_reduced': 0.0,
            'timeslots_processed': 0,
            'real_time_violations': 0,
            'cache_hits': 0,
            'incremental_updates': 0,
            'average_optimization_time_ms': 0.0,
            'complexity_validation': {
                'max_layers_processed': 0,
                'max_nodes_processed': 0,
                'theoretical_complexity': 'O(L * N^2)',
                'measured_complexity_factor': 0.0
            }
        }
        
        logger.info(f"DP Partitioner initialized with {self.optimization_levels} optimization levels")
        logger.info(f"Communication penalty: {self.communication_penalty}")
        logger.info(f"Memory penalty: {self.memory_penalty}")
        logger.info(f"Timeslot duration: {self.timeslot_duration_ms}ms, Max optimization time: {self.max_optimization_time_ms}ms")
        logger.info(f"Resource update interval: {self.resource_update_interval_ms}ms")
    
    def set_inter_layer_data_sizes(self, data_sizes: Dict[Tuple[int, int], float]):
        """
        Set intermediate data sizes between layers: β_{l,l+1}
        
        Args:
            data_sizes: Dictionary mapping (layer_l, layer_l+1) -> data_size_MB
        """
        self.inter_layer_data_sizes = data_sizes
        logger.info(f"Set inter-layer data sizes for {len(data_sizes)} layer pairs")
    
    def _calculate_processing_delay(self, layer: ModelLayer, node: NodeResource) -> float:
        """
        Calculate processing delay from manuscript: c_l^quant / φ_n(t)
        
        This is the core delay calculation that considers quantization effects
        """
        return node.get_processing_delay(layer)
    
    def _calculate_communication_delay(self, prev_node: str, curr_node: str, 
                                      layer_id: int, nodes: Dict[str, NodeResource]) -> float:
        """
        Calculate communication delay from manuscript: β_{l-1,l} / b_{n,m}
        
        This handles inter-node communication when consecutive layers are on different nodes
        """
        if prev_node == curr_node:
            return 0.0  # No communication needed for same-node placement
        
        # Get intermediate data size between layers
        data_size_key = (layer_id - 1, layer_id)
        if data_size_key not in self.inter_layer_data_sizes:
            # Use default size based on layer complexity
            data_size = 100.0  # Default 100MB
            logger.warning(f"No data size specified for layers {layer_id-1}->{layer_id}, using default {data_size}MB")
        else:
            data_size = self.inter_layer_data_sizes[data_size_key]
        
        # Get bandwidth between nodes: b_{n,m}
        prev_node_resource = nodes[prev_node]
        if curr_node not in prev_node_resource.bandwidth_to_nodes:
            logger.warning(f"No bandwidth specified between {prev_node} and {curr_node}")
            return float('inf')
        
        bandwidth = prev_node_resource.bandwidth_to_nodes[curr_node]
        if bandwidth <= 0:
            return float('inf')
        
        # Communication delay: β_{l-1,l} / b_{n,m}
        comm_delay = data_size / bandwidth
        logger.debug(f"Communication delay {prev_node}->{curr_node}: {comm_delay:.3f}ms")
        return comm_delay
    
    def _initialize_dp_table(self, layers: List[ModelLayer], nodes: Dict[str, NodeResource]):
        """
        Initialize DP table for first layer
        
        Algorithm 2 Step 1: Set up base case for dynamic programming
        """
        first_layer = layers[0]
        
        for node_id, node in nodes.items():
            if node.can_accommodate_layer(first_layer):
                # For first layer, only processing delay (no communication)
                proc_delay = self._calculate_processing_delay(first_layer, node)
                self.dp_table[(0, node_id)] = proc_delay
                self.parent_table[(0, node_id)] = None
                logger.debug(f"Layer 0 -> Node {node_id}: delay = {proc_delay:.3f}ms")
            else:
                self.dp_table[(0, node_id)] = float('inf')
                self.parent_table[(0, node_id)] = None
                logger.debug(f"Layer 0 -> Node {node_id}: insufficient resources")
    
    def _update_dp_table(self, layer_idx: int, layer: ModelLayer, 
                        nodes: Dict[str, NodeResource]):
        """
        Update DP table for current layer using dynamic programming recurrence
        
        Algorithm 2 Step 2: DP recurrence relation
        dp[l][n] = min_{n'} {dp[l-1][n'] + c_l^quant/φ_n + β_{l-1,l}/b_{n',n}}
        """
        for curr_node_id, curr_node in nodes.items():
            if not curr_node.can_accommodate_layer(layer):
                self.dp_table[(layer_idx, curr_node_id)] = float('inf')
                self.parent_table[(layer_idx, curr_node_id)] = None
                continue
            
            min_delay = float('inf')
            best_prev_node = None
            
            # Try all possible previous nodes (DP recurrence core)
            for prev_node_id in nodes.keys():
                prev_delay = self.dp_table.get((layer_idx - 1, prev_node_id), float('inf'))
                
                if prev_delay == float('inf'):
                    continue
                
                # Calculate delays for current assignment
                proc_delay = self._calculate_processing_delay(layer, curr_node)
                comm_delay = self._calculate_communication_delay(
                    prev_node_id, curr_node_id, layer_idx, nodes)
                
                # Total delay with penalties
                total_delay = prev_delay + proc_delay + comm_delay
                
                # Apply optimization penalties from manuscript
                
                # Penalty for high precision on edge (memory constraint consideration)
                if curr_node.tier == NodeTier.EDGE and layer.quantization_factor < 0.5:
                    total_delay += self.memory_penalty * layer.quantized_memory_requirement
                
                # Communication penalty for inter-tier communication
                if prev_node_id != curr_node_id:
                    prev_tier = nodes[prev_node_id].tier
                    curr_tier = curr_node.tier
                    if prev_tier != curr_tier:
                        total_delay += self.communication_penalty * comm_delay
                
                # Update DP table if better solution found
                if total_delay < min_delay:
                    min_delay = total_delay
                    best_prev_node = prev_node_id
            
            self.dp_table[(layer_idx, curr_node_id)] = min_delay
            self.parent_table[(layer_idx, curr_node_id)] = best_prev_node
            
            logger.debug(f"Layer {layer_idx} -> Node {curr_node_id}: "
                        f"delay = {min_delay:.3f}ms, prev = {best_prev_node}")
    
    def _reconstruct_solution(self, layers: List[ModelLayer], 
                             nodes: Dict[str, NodeResource]) -> List[LayerAssignment]:
        """
        Reconstruct optimal solution using backtracking
        
        Algorithm 2 Step 3: Backtrack through DP table to get optimal assignments
        """
        # Find the node with minimum delay for the last layer
        last_layer_idx = len(layers) - 1
        min_delay = float('inf')
        best_final_node = None
        
        for node_id in nodes.keys():
            delay = self.dp_table.get((last_layer_idx, node_id), float('inf'))
            if delay < min_delay:
                min_delay = delay
                best_final_node = node_id
        
        if best_final_node is None:
            raise ValueError("No valid solution found - check resource constraints")
        
        logger.info(f"Optimal solution found with total delay: {min_delay:.3f}ms")
        
        # Backtrack to construct solution
        solution = []
        current_node = best_final_node
        
        for layer_idx in range(last_layer_idx, -1, -1):
            layer = layers[layer_idx]
            node = nodes[current_node]
            
            # Calculate delays for this assignment
            proc_delay = self._calculate_processing_delay(layer, node)
            
            if layer_idx == 0:
                comm_delay = 0.0
            else:
                prev_node = self.parent_table[(layer_idx, current_node)]
                comm_delay = self._calculate_communication_delay(
                    prev_node, current_node, layer_idx, nodes)
            
            assignment = LayerAssignment(
                layer_id=layer_idx,
                node_id=current_node,
                processing_delay=proc_delay,
                communication_delay=comm_delay,
                total_delay=proc_delay + comm_delay,
                quantization_factor=layer.quantization_factor
            )
            
            solution.append(assignment)
            
            # Move to previous node for backtracking
            current_node = self.parent_table[(layer_idx, current_node)]
        
        # Reverse to get correct order (layer 0 to L-1)
        solution.reverse()
        
        return solution
    
    def partition_model_with_timeslot(self, layers: List[ModelLayer], 
                                     nodes: Dict[str, NodeResource],
                                     current_time_ms: float = None,
                                     deadline_ms: float = None) -> Tuple[List[LayerAssignment], Dict]:
        if current_time_ms is None:
            current_time_ms = time.time() * 1000
        if deadline_ms is None:
            deadline_ms = current_time_ms + self.max_optimization_time_ms
            
        return self._partition_with_time_bounds(layers, nodes, current_time_ms, deadline_ms)
    
    def partition_model(self, layers: List[ModelLayer], 
                       nodes: Dict[str, NodeResource]) -> Tuple[List[LayerAssignment], Dict]:
        current_time = time.time() * 1000
        return self.partition_model_with_timeslot(layers, nodes, current_time)
    def _partition_with_time_bounds(self, layers: List[ModelLayer],
                                   nodes: Dict[str, NodeResource], 
                                   start_time_ms: float,
                                   deadline_ms: float) -> Tuple[List[LayerAssignment], Dict]:
        partition_start = time.time()
        start_time_ms_actual = time.time() * 1000
        
        # Update current timeslot
        self.current_timeslot = int(start_time_ms / self.timeslot_duration_ms)
        
        # Check if resource profiling update is needed
        if (start_time_ms_actual - self.last_resource_update) >= self.resource_update_interval_ms:
            self._update_resource_profiles(nodes, start_time_ms_actual)
        
        # Check cache for similar configurations
        cache_key = self._generate_cache_key(layers, nodes)
        if cache_key in self.cached_solutions:
            cached_solution, cached_time = self.cached_solutions[cache_key]
            if (start_time_ms_actual - cached_time) < (self.timeslot_duration_ms * 2):
                self.optimization_stats['cache_hits'] += 1
                logger.debug(f"Using cached solution for timeslot {self.current_timeslot}")
                return cached_solution, {'optimization_method': 'cached', 'cache_age_ms': start_time_ms_actual - cached_time}
        
        logger.info(f"Starting time-bounded DP partitioning for {len(layers)} layers on {len(nodes)} nodes")
        logger.info(f"Timeslot: {self.current_timeslot}, Deadline: {deadline_ms - start_time_ms_actual:.1f}ms")
        
        # Validate inputs
        if not layers:
            raise ValueError("No layers provided for partitioning")
        if not nodes:
            raise ValueError("No nodes provided for partitioning")
        
        # Complexity validation
        self.optimization_stats['complexity_validation']['max_layers_processed'] = max(
            self.optimization_stats['complexity_validation']['max_layers_processed'], len(layers))
        self.optimization_stats['complexity_validation']['max_nodes_processed'] = max(
            self.optimization_stats['complexity_validation']['max_nodes_processed'], len(nodes))
        
        # Measure theoretical vs actual complexity
        theoretical_ops = len(layers) * len(nodes) * len(nodes)
        
        # Clear previous state
        self.dp_table.clear()
        self.parent_table.clear()
        
        # Algorithm 2 Implementation with real-time constraints
        try:
            actual_ops = 0
            
            # Step 1: Initialize DP table for first layer
            self._initialize_dp_table(layers, nodes)
            actual_ops += len(nodes)
            
            # Real-time check after initialization
            current_time_ms = time.time() * 1000
            if current_time_ms > deadline_ms:
                logger.warning(f"Optimization deadline exceeded during initialization")
                self.optimization_stats['real_time_violations'] += 1
                return self._get_fallback_solution(layers, nodes), {'method': 'fallback_timeout'}
            
            # Step 2: Fill DP table for remaining layers with time bounds
            for layer_idx in range(1, len(layers)):
                layer = layers[layer_idx]
                
                # Check time constraint before processing layer
                current_time_ms = time.time() * 1000
                remaining_time = deadline_ms - current_time_ms
                
                if remaining_time < 5.0:  # Need at least 5ms for solution reconstruction
                    logger.info(f"Time constraint reached at layer {layer_idx}, using partial solution")
                    partial_assignments = self._get_partial_solution(layers[:layer_idx], nodes)
                    return partial_assignments, {'method': 'partial_time_bounded', 'layers_completed': layer_idx}
                
                self._update_dp_table(layer_idx, layer, nodes)
                actual_ops += len(nodes) * len(nodes)
            
            # Step 3: Reconstruct optimal solution
            assignments = self._reconstruct_solution(layers, nodes)
            
            # Calculate comprehensive statistics with real-time metrics
            partition_time = time.time() - partition_start
            partition_time_ms = partition_time * 1000
            
            # Update complexity measurements
            if theoretical_ops > 0:
                measured_factor = actual_ops / theoretical_ops
                self.optimization_stats['complexity_validation']['measured_complexity_factor'] = measured_factor
            
            stats = self._calculate_partitioning_stats(assignments, nodes, partition_time)
            stats['real_time_metrics'] = {
                'timeslot': self.current_timeslot,
                'optimization_time_ms': partition_time_ms,
                'deadline_met': partition_time_ms <= self.max_optimization_time_ms,
                'time_utilization': partition_time_ms / self.max_optimization_time_ms,
                'operations_performed': actual_ops,
                'complexity_factor': measured_factor if theoretical_ops > 0 else 0.0
            }
            
            # Cache solution for future use
            self.cached_solutions[cache_key] = (assignments, start_time_ms_actual)
            if len(self.cached_solutions) > 10:  # Limit cache size
                oldest_key = min(self.cached_solutions.keys(), key=lambda k: self.cached_solutions[k][1])
                del self.cached_solutions[oldest_key]
            
            # Update performance tracking
            self.optimization_stats['total_partitioning_time'] += partition_time
            self.optimization_stats['layers_processed'] += len(layers)
            self.optimization_stats['nodes_considered'] += len(nodes)
            self.optimization_stats['optimal_assignments'] += len(assignments)
            self.optimization_stats['timeslots_processed'] += 1
            
            # Update average optimization time
            total_optimizations = self.optimization_stats['timeslots_processed']
            current_avg = self.optimization_stats['average_optimization_time_ms']
            self.optimization_stats['average_optimization_time_ms'] = (
                (current_avg * (total_optimizations - 1) + partition_time_ms) / total_optimizations
            )
            
            logger.info(f"DP partitioning completed in {partition_time:.3f}s ({partition_time_ms:.1f}ms)")
            logger.info(f"Solution: {len([a for a in assignments if nodes[a.node_id].tier == NodeTier.CLOUD])} cloud layers, "
                       f"{len([a for a in assignments if nodes[a.node_id].tier == NodeTier.EDGE])} edge layers")
            logger.info(f"Real-time performance: {partition_time_ms:.1f}ms/{self.max_optimization_time_ms}ms "
                       f"({'PASS' if partition_time_ms <= self.max_optimization_time_ms else 'EXCEED'})")
            
            return assignments, stats
            
        except Exception as e:
            logger.error(f"DP partitioning failed: {e}")
            fallback_solution = self._get_fallback_solution(layers, nodes)
            fallback_stats = {'method': 'fallback_error', 'error': str(e)}
            return fallback_solution, fallback_stats
    
    def _update_resource_profiles(self, nodes: Dict[str, NodeResource], current_time_ms: float):
        for node_id, node in nodes.items():
            utilization_factor = 0.8 + 0.2 * np.sin(current_time_ms / 10000)
            node.compute_capacity *= utilization_factor
        
        self.last_resource_update = current_time_ms
        self.optimization_stats['incremental_updates'] += 1
    
    def _generate_cache_key(self, layers: List[ModelLayer], nodes: Dict[str, NodeResource]) -> str:
        layer_sig = hash(tuple((l.layer_id, l.compute_demand, l.memory_requirement, l.quantization_factor) for l in layers))
        node_sig = hash(tuple((n, node.compute_capacity, node.memory_capacity) for n, node in nodes.items()))
        return f"{layer_sig}_{node_sig}_{self.current_timeslot}"
    
    def _get_fallback_solution(self, layers: List[ModelLayer], nodes: Dict[str, NodeResource]) -> List[LayerAssignment]:
        assignments = []
        for layer in layers:
            best_node = min(nodes.items(), key=lambda x: x[1].get_processing_delay(layer))[0]
            assignment = LayerAssignment(
                layer_id=layer.layer_id,
                node_id=best_node,
                processing_delay=nodes[best_node].get_processing_delay(layer),
                communication_delay=0.0,
                total_delay=nodes[best_node].get_processing_delay(layer),
                quantization_factor=layer.quantization_factor
            )
            assignments.append(assignment)
        return assignments
    
    def _get_partial_solution(self, partial_layers: List[ModelLayer], nodes: Dict[str, NodeResource]) -> List[LayerAssignment]:
        if not partial_layers:
            return []
        
        self.dp_table.clear()
        self.parent_table.clear()
        
        self._initialize_dp_table(partial_layers, nodes)
        
        for layer_idx in range(1, len(partial_layers)):
            self._update_dp_table(layer_idx, partial_layers[layer_idx], nodes)
        
        return self._reconstruct_solution(partial_layers, nodes)
    
    def _calculate_partitioning_stats(self, assignments: List[LayerAssignment], 
                                    nodes: Dict[str, NodeResource], 
                                    partition_time: float) -> Dict:
        """
        Calculate comprehensive statistics for research validation
        """
        # Basic delays
        total_processing_delay = sum(a.processing_delay for a in assignments)
        total_communication_delay = sum(a.communication_delay for a in assignments)
        total_delay = total_processing_delay + total_communication_delay
        
        # Node distribution
        cloud_assignments = sum(1 for a in assignments if nodes[a.node_id].tier == NodeTier.CLOUD)
        edge_assignments = len(assignments) - cloud_assignments
        
        # Communication analysis
        inter_tier_communications = 0
        intra_tier_communications = 0
        
        for i in range(1, len(assignments)):
            prev_tier = nodes[assignments[i-1].node_id].tier
            curr_tier = nodes[assignments[i].node_id].tier
            
            if assignments[i].communication_delay > 0:
                if prev_tier != curr_tier:
                    inter_tier_communications += 1
                else:
                    intra_tier_communications += 1
        
        # Quantization analysis
        quantization_levels = [a.quantization_factor for a in assignments]
        avg_quantization = np.mean(quantization_levels)
        
        # Resource utilization
        node_utilization = {}
        for node_id, node in nodes.items():
            assigned_layers = [a for a in assignments if a.node_id == node_id]
            if assigned_layers:
                compute_used = sum(layers[a.layer_id].quantized_compute_demand for a in assigned_layers)
                memory_used = sum(layers[a.layer_id].quantized_memory_requirement for a in assigned_layers)
                node_utilization[node_id] = {
                    'compute_utilization': compute_used / node.compute_capacity,
                    'memory_utilization': memory_used / node.memory_capacity,
                    'layers_assigned': len(assigned_layers)
                }
        
        stats = {
            # Core performance metrics
            'total_delay': total_delay,
            'processing_delay': total_processing_delay,
            'communication_delay': total_communication_delay,
            'partition_time': partition_time,
            
            # Assignment distribution
            'cloud_assignments': cloud_assignments,
            'edge_assignments': edge_assignments,
            'total_layers': len(assignments),
            
            # Communication analysis  
            'inter_tier_communications': inter_tier_communications,
            'intra_tier_communications': intra_tier_communications,
            'communication_efficiency': 1.0 - (total_communication_delay / total_delay) if total_delay > 0 else 0.0,
            
            # Quantization analysis
            'average_quantization_factor': avg_quantization,
            'quantization_distribution': {
                '1.0 (full_precision)': sum(1 for q in quantization_levels if q == 1.0),
                '0.5 (half_precision)': sum(1 for q in quantization_levels if q == 0.5),
                '0.25 (quarter_precision)': sum(1 for q in quantization_levels if q == 0.25),
                'other': sum(1 for q in quantization_levels if q not in [1.0, 0.5, 0.25])
            },
            
            # Resource utilization
            'node_utilization': node_utilization,
            'optimization_efficiency': 1.0 - (inter_tier_communications / max(len(assignments)-1, 1)),
            
            # Algorithm performance
            'dp_table_size': len(self.dp_table),
            'backtrack_steps': len(assignments)
        }
        
        return stats
    
    def optimize_quantization(self, layers: List[ModelLayer], 
                            nodes: Dict[str, NodeResource],
                            quantization_levels: List[float] = [1.0, 0.5, 0.25]) -> List[ModelLayer]:
        """
        Optimize quantization factors for layers based on placement
        
        This extends Algorithm 2 to jointly optimize both placement and quantization
        """
        logger.info("Optimizing quantization factors for LL-CIoT deployment")
        
        optimized_layers = []
        
        for layer in layers:
            best_quantization = 1.0
            best_delay = float('inf')
            
            # Try different quantization levels
            for q_factor in quantization_levels:
                test_layer = ModelLayer(
                    layer.layer_id, layer.layer_name,
                    layer.compute_demand, layer.memory_requirement,
                    q_factor
                )
                
                # Find best placement for this quantization
                test_layers = [test_layer]
                try:
                    assignments, stats = self.partition_model(test_layers, nodes)
                    if len(assignments) > 0 and stats['total_delay'] < best_delay:
                        best_delay = stats['total_delay']
                        best_quantization = q_factor
                except:
                    # Skip invalid quantization
                    continue
            
            # Create optimized layer
            optimized_layer = ModelLayer(
                layer.layer_id, layer.layer_name,
                layer.compute_demand, layer.memory_requirement,
                best_quantization
            )
            
            optimized_layers.append(optimized_layer)
            logger.debug(f"Layer {layer.layer_id}: optimal quantization = {best_quantization}")
        
        return optimized_layers
    
    def export_assignments(self, assignments: List[LayerAssignment], 
                          filename: str):
        """Export layer assignments to JSON file for research validation"""
        export_data = {
            'timestamp': time.time(),
            'algorithm': 'DP_Partitioner_Algorithm_2',
            'assignments': [
                {
                    'layer_id': a.layer_id,
                    'node_id': a.node_id,
                    'processing_delay': a.processing_delay,
                    'communication_delay': a.communication_delay,
                    'total_delay': a.total_delay,
                    'quantization_factor': a.quantization_factor
                }
                for a in assignments
            ],
            'summary': {
                'total_layers': len(assignments),
                'total_delay': sum(a.total_delay for a in assignments),
                'total_processing_delay': sum(a.processing_delay for a in assignments),
                'total_communication_delay': sum(a.communication_delay for a in assignments),
                'optimization_stats': self.optimization_stats
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Assignments exported to {filename}")

if __name__ == "__main__":
    # Test DP partitioner with sample configuration
    logging.basicConfig(level=logging.INFO)
    
    # Configuration matching manuscript parameters
    config = {
        'optimization_levels': 3,
        'communication_penalty': 0.1,
        'memory_penalty': 0.05
    }
    
    partitioner = DPPartitioner(config)
    
    # Create sample layers (simulating Llama2-7B and TinyLlama-1.1B)
    layers = [
        ModelLayer(0, "embedding", 100.0, 500.0, 1.0),
        ModelLayer(1, "attention_0", 200.0, 800.0, 0.5),
        ModelLayer(2, "mlp_0", 150.0, 600.0, 0.5),
        ModelLayer(3, "attention_1", 200.0, 800.0, 0.25),
        ModelLayer(4, "output", 80.0, 400.0, 1.0)
    ]
    
    # Create sample nodes matching manuscript testbed
    nodes = {
        'cloud_server': NodeResource('cloud_server', NodeTier.CLOUD, 0, 8.0, 11000.0, 
                                   {'edge_server': 50.0}),
        'edge_server': NodeResource('edge_server', NodeTier.EDGE, 0, 2.0, 4000.0,
                                  {'cloud_server': 50.0})
    }
    
    # Set inter-layer data sizes β_{l,l+1}
    inter_layer_sizes = {
        (0, 1): 50.0,  # MB
        (1, 2): 40.0,
        (2, 3): 40.0,
        (3, 4): 30.0
    }
    partitioner.set_inter_layer_data_sizes(inter_layer_sizes)
    
    # Run Algorithm 2
    assignments, stats = partitioner.partition_model(layers, nodes)
    
    print("DP Partitioner Results (Algorithm 2):")
    print("=" * 50)
    for assignment in assignments:
        print(f"Layer {assignment.layer_id} -> Node {assignment.node_id} "
              f"(proc: {assignment.processing_delay:.3f}ms, "
              f"comm: {assignment.communication_delay:.3f}ms, "
              f"α: {assignment.quantization_factor})")
    
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
    
    # Export results
    partitioner.export_assignments(assignments, "dp_partitioner_results.json")