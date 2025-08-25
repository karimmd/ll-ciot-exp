#!/usr/bin/env python3
"""
LL-CIoT Research Validation - VDF Scheduler (Algorithm 1)
Exact implementation of Algorithm 1 from the manuscript

This implements the Value Density First scheduling algorithm with:
- Value density calculation: ρⱼ = vⱼ/(tⱼᵈˡ - tⱼᵃʳʳ)
- Preemption mechanism with threshold γ > 1  
- Batch processing for inference and fine-tuning tasks
- Task prioritization based on real-time requirements
"""

import time
import logging
import heapq
import threading
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Task types as defined in manuscript"""
    INFERENCE = "INF"  # I_j = 1 in manuscript
    FINE_TUNING = "FT"  # I_j = 0 in manuscript

@dataclass
class IoTTask:
    """
    IoT Task representation matching manuscript notation
    
    Manuscript symbols:
    - j: task index
    - t_j^arr: arrival time  
    - t_j^dl: deadline
    - v_j: priority value
    - c_j: computational demand
    - ρ_j: value density = v_j/(t_j^dl - t_j^arr)
    - I_j: task type indicator (1=INF, 0=FT)
    """
    task_id: str
    task_type: TaskType
    arrival_time: float  # t_j^arr
    deadline: float      # t_j^dl  
    priority_value: float  # v_j
    compute_demand: float  # c_j
    memory_demand: float   # m_j
    data_size: float      # β_j
    device_id: str = ""
    
    def __post_init__(self):
        """Calculate derived properties after initialization"""
        self.time_window = self.deadline - self.arrival_time
        if self.time_window <= 0:
            raise ValueError(f"Invalid time window for task {self.task_id}: {self.time_window}")
    
    @property
    def value_density(self) -> float:
        """
        Value density calculation from manuscript: ρⱼ = vⱼ/(tⱼᵈˡ - tⱼᵃʳʳ)
        """
        return self.priority_value / self.time_window
    
    @property
    def task_indicator(self) -> int:
        """Task type indicator I_j from manuscript (1=INF, 0=FT)"""
        return 1 if self.task_type == TaskType.INFERENCE else 0
    
    def __lt__(self, other):
        """Comparison for priority queue (higher value density = higher priority)"""
        return self.value_density > other.value_density

@dataclass  
class TaskBatch:
    """
    Task batch representation matching manuscript notation B_i
    
    Batch contains tasks sorted by value density: ρ_j1 ≥ ρ_j2 ≥ ... ≥ ρ_jb
    """
    batch_id: str
    tasks: List[IoTTask] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    
    @property
    def batch_size(self) -> int:
        """Batch size b in manuscript"""
        return len(self.tasks)
    
    @property
    def cumulative_value_density(self) -> float:
        """Cumulative value density for preemption decisions"""
        return sum(task.value_density for task in self.tasks)
    
    @property
    def total_compute_demand(self) -> float:
        """Total computational demand for resource allocation"""
        return sum(task.compute_demand for task in self.tasks)
    
    @property
    def processing_delay(self, node_capacity: float) -> float:
        """Processing delay D_B^proc = Σ(c_j/φ_n) from manuscript"""
        if node_capacity <= 0:
            return float('inf')
        return self.total_compute_demand / node_capacity

class VDFScheduler:
    """
    Value Density First Scheduler implementing Algorithm 1 from manuscript
    
    Key features:
    1. Task prioritization by value density ρⱼ = vⱼ/(tⱼᵈˡ - tⱼᵃʳʳ)
    2. Preemption mechanism with threshold γ > 1
    3. Batch processing with size constraints
    4. Support for both inference and fine-tuning tasks
    5. Resource-aware scheduling decisions
    """
    
    def __init__(self, config: Dict):
        self.preemption_threshold = config.get('preemption_threshold', 1.5)  # γ parameter
        self.batch_size_max = config.get('batch_size_max', 16)
        self.batch_size_min = config.get('batch_size_min', 4)
        self.priority_weights = config.get('priority_weights', {
            'high': 3.0, 'medium': 2.0, 'low': 1.0
        })
        
        # Task management
        self.task_queue = []  # Priority queue for incoming tasks
        self.current_batches = {}  # Currently executing batches per node
        self.waiting_batches = []  # Batches waiting for execution
        self.completed_tasks = []  # Task execution history
        
        # Thread safety
        self.queue_lock = threading.Lock()
        self.batch_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_tasks_processed': 0,
            'total_batches_created': 0,
            'preemptions_performed': 0,
            'average_value_density': 0.0,
            'inference_task_count': 0,
            'fine_tuning_task_count': 0
        }
        
        logger.info(f"VDF Scheduler initialized with γ={self.preemption_threshold}")
        logger.info(f"Batch size range: {self.batch_size_min}-{self.batch_size_max}")
    
    def submit_task(self, task: IoTTask) -> bool:
        """
        Submit new task to scheduler
        
        Algorithm 1 Step 1: Insert task into priority queue based on value density
        """
        try:
            with self.queue_lock:
                # Validate task timing constraints
                current_time = time.time()
                if task.deadline <= current_time:
                    logger.warning(f"Task {task.task_id} deadline already passed")
                    return False
                
                # Insert into priority queue (heapq maintains min-heap, but our __lt__ reverses order)
                heapq.heappush(self.task_queue, task)
                
                # Update metrics
                self.metrics['total_tasks_processed'] += 1
                if task.task_type == TaskType.INFERENCE:
                    self.metrics['inference_task_count'] += 1
                else:
                    self.metrics['fine_tuning_task_count'] += 1
                
                logger.debug(f"Task {task.task_id} submitted: ρ={task.value_density:.4f}, type={task.task_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def create_batch(self, target_size: Optional[int] = None) -> Optional[TaskBatch]:
        """
        Create task batch from priority queue
        
        Algorithm 1 Step 2: Group tasks into batch B_i sorted by value density
        """
        if target_size is None:
            target_size = self.batch_size_max
        
        try:
            with self.queue_lock:
                if len(self.task_queue) < self.batch_size_min:
                    return None
                
                # Extract tasks with highest value density
                batch_tasks = []
                extracted_tasks = []
                
                # Get up to target_size tasks
                for _ in range(min(target_size, len(self.task_queue))):
                    if self.task_queue:
                        task = heapq.heappop(self.task_queue)
                        batch_tasks.append(task)
                        extracted_tasks.append(task)
                
                if not batch_tasks:
                    return None
                
                # Sort by value density (descending) as per manuscript: ρ_j1 ≥ ρ_j2 ≥ ... ≥ ρ_jb
                batch_tasks.sort(key=lambda t: t.value_density, reverse=True)
                
                # Create batch
                batch_id = f"batch_{int(time.time()*1000)}_{len(batch_tasks)}"
                batch = TaskBatch(batch_id=batch_id, tasks=batch_tasks)
                
                # Update metrics
                self.metrics['total_batches_created'] += 1
                avg_density = np.mean([task.value_density for task in batch_tasks])
                self.metrics['average_value_density'] = avg_density
                
                logger.info(f"Created batch {batch_id}: {len(batch_tasks)} tasks, ρ_avg={avg_density:.4f}")
                return batch
                
        except Exception as e:
            logger.error(f"Failed to create batch: {e}")
            return None
    
    def check_preemption(self, new_batch: TaskBatch, current_batch: TaskBatch) -> bool:
        """
        Check if new batch can preempt current batch
        
        Algorithm 1 Step 3: Preemption condition from manuscript:
        - For FT preempting INF: ρ_j^FT ≥ γ · ρ_k^INF
        - For batch preemption: Σ ρ_j ≥ γ · Σ ρ_k
        """
        try:
            # Calculate cumulative value densities
            new_cumulative_density = new_batch.cumulative_value_density
            current_cumulative_density = current_batch.cumulative_value_density
            
            # Preemption condition: new_density ≥ γ * current_density
            preemption_threshold_met = (
                new_cumulative_density >= self.preemption_threshold * current_cumulative_density
            )
            
            # Additional checks for task type priorities
            new_has_high_priority_ft = any(
                task.task_type == TaskType.FINE_TUNING and task.priority_value > 2.5 
                for task in new_batch.tasks
            )
            
            current_has_inf_tasks = any(
                task.task_type == TaskType.INFERENCE 
                for task in current_batch.tasks
            )
            
            # Enhanced preemption for high-priority FT tasks interrupting INF tasks
            enhanced_preemption = (
                new_has_high_priority_ft and current_has_inf_tasks and
                new_cumulative_density >= (self.preemption_threshold * 0.8) * current_cumulative_density
            )
            
            preemption_decision = preemption_threshold_met or enhanced_preemption
            
            if preemption_decision:
                logger.info(f"Preemption triggered: new_ρ={new_cumulative_density:.4f} vs current_ρ={current_cumulative_density:.4f} (γ={self.preemption_threshold})")
                self.metrics['preemptions_performed'] += 1
            
            return preemption_decision
            
        except Exception as e:
            logger.error(f"Preemption check failed: {e}")
            return False
    
    def schedule_batch(self, batch: TaskBatch, node_id: str, node_capacity: float) -> bool:
        """
        Schedule batch for execution on specified node
        
        Algorithm 1 Step 4: Assign batch to node with resource checking
        """
        try:
            with self.batch_lock:
                current_time = time.time()
                
                # Check if node has current batch
                current_batch = self.current_batches.get(node_id)
                
                if current_batch is not None:
                    # Check preemption condition
                    if self.check_preemption(batch, current_batch):
                        # Preempt current batch
                        logger.info(f"Preempting batch {current_batch.batch_id} with {batch.batch_id}")
                        
                        # Move preempted batch back to waiting queue
                        self.waiting_batches.append(current_batch)
                        
                        # Assign new batch
                        self.current_batches[node_id] = batch
                        batch.assigned_node = node_id
                        
                        return True
                    else:
                        # Cannot preempt, add to waiting queue
                        logger.debug(f"Batch {batch.batch_id} added to waiting queue")
                        self.waiting_batches.append(batch)
                        return False
                else:
                    # Node is free, assign batch directly
                    self.current_batches[node_id] = batch
                    batch.assigned_node = node_id
                    
                    logger.info(f"Batch {batch.batch_id} scheduled on {node_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to schedule batch {batch.batch_id}: {e}")
            return False
    
    def complete_batch(self, node_id: str, execution_results: Dict) -> bool:
        """
        Mark batch as completed and schedule next waiting batch
        """
        try:
            with self.batch_lock:
                completed_batch = self.current_batches.pop(node_id, None)
                if completed_batch is None:
                    logger.warning(f"No batch found for node {node_id}")
                    return False
                
                # Record completion
                completion_time = time.time()
                for task in completed_batch.tasks:
                    task_result = {
                        'task_id': task.task_id,
                        'completion_time': completion_time,
                        'execution_time': execution_results.get('execution_time', 0),
                        'node_id': node_id,
                        'value_density': task.value_density,
                        'task_type': task.task_type.value
                    }
                    self.completed_tasks.append(task_result)
                
                logger.info(f"Completed batch {completed_batch.batch_id} on {node_id}")
                
                # Schedule next waiting batch if available
                if self.waiting_batches:
                    next_batch = self.waiting_batches.pop(0)
                    self.current_batches[node_id] = next_batch
                    next_batch.assigned_node = node_id
                    logger.info(f"Scheduled waiting batch {next_batch.batch_id} on {node_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete batch on {node_id}: {e}")
            return False
    
    def get_schedule_status(self) -> Dict:
        """Get current scheduling status and metrics"""
        with self.queue_lock, self.batch_lock:
            return {
                'queued_tasks': len(self.task_queue),
                'active_batches': len(self.current_batches),
                'waiting_batches': len(self.waiting_batches),
                'completed_tasks': len(self.completed_tasks),
                'metrics': self.metrics.copy(),
                'current_assignments': {
                    node: batch.batch_id for node, batch in self.current_batches.items()
                }
            }
    
    def get_performance_analysis(self) -> Dict:
        """Generate performance analysis for research validation"""
        try:
            total_tasks = len(self.completed_tasks)
            if total_tasks == 0:
                return {"error": "No completed tasks for analysis"}
            
            # Task type analysis
            inf_tasks = [t for t in self.completed_tasks if t['task_type'] == 'INF']
            ft_tasks = [t for t in self.completed_tasks if t['task_type'] == 'FT']
            
            # Value density analysis
            value_densities = [t['value_density'] for t in self.completed_tasks]
            
            # Execution time analysis
            execution_times = [t['execution_time'] for t in self.completed_tasks]
            
            analysis = {
                'total_tasks_processed': total_tasks,
                'task_type_distribution': {
                    'inference': len(inf_tasks),
                    'fine_tuning': len(ft_tasks)
                },
                'value_density_stats': {
                    'mean': np.mean(value_densities),
                    'std': np.std(value_densities),
                    'min': np.min(value_densities),
                    'max': np.max(value_densities)
                },
                'execution_time_stats': {
                    'mean': np.mean(execution_times),
                    'std': np.std(execution_times),
                    'min': np.min(execution_times),
                    'max': np.max(execution_times)
                },
                'scheduling_efficiency': {
                    'preemption_rate': self.metrics['preemptions_performed'] / max(self.metrics['total_batches_created'], 1),
                    'average_batch_utilization': self.metrics['total_tasks_processed'] / max(self.metrics['total_batches_created'], 1),
                    'inf_task_percentage': len(inf_tasks) / total_tasks * 100,
                    'ft_task_percentage': len(ft_tasks) / total_tasks * 100
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}

def create_sample_tasks() -> List[IoTTask]:
    """Create sample tasks for testing VDF scheduler"""
    tasks = []
    current_time = time.time()
    
    # Sample IoT analysis tasks with varying priorities and deadlines
    task_configs = [
        {"type": TaskType.INFERENCE, "priority": 3.0, "deadline_offset": 2.0, "compute": 100},
        {"type": TaskType.FINE_TUNING, "priority": 2.0, "deadline_offset": 10.0, "compute": 300},
        {"type": TaskType.INFERENCE, "priority": 2.5, "deadline_offset": 1.5, "compute": 80},
        {"type": TaskType.INFERENCE, "priority": 1.8, "deadline_offset": 3.0, "compute": 120},
        {"type": TaskType.FINE_TUNING, "priority": 3.5, "deadline_offset": 8.0, "compute": 250},
    ]
    
    for i, config in enumerate(task_configs):
        task = IoTTask(
            task_id=f"task_{i+1:03d}",
            task_type=config["type"],
            arrival_time=current_time,
            deadline=current_time + config["deadline_offset"],
            priority_value=config["priority"],
            compute_demand=config["compute"],
            memory_demand=50.0,
            data_size=20.0,
            device_id=f"device_{i%3 + 1}"
        )
        tasks.append(task)
    
    return tasks

if __name__ == "__main__":
    # Test VDF scheduler with sample tasks
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'preemption_threshold': 1.5,
        'batch_size_max': 4,
        'batch_size_min': 2
    }
    
    scheduler = VDFScheduler(config)
    
    # Create and submit sample tasks
    sample_tasks = create_sample_tasks()
    for task in sample_tasks:
        scheduler.submit_task(task)
        print(f"Task {task.task_id}: ρ = {task.value_density:.4f}, type = {task.task_type.value}")
    
    # Create batch
    batch = scheduler.create_batch()
    if batch:
        print(f"\nCreated batch: {batch.batch_id}")
        print(f"Tasks in batch: {[t.task_id for t in batch.tasks]}")
        print(f"Cumulative value density: {batch.cumulative_value_density:.4f}")
    
    # Display scheduler status
    status = scheduler.get_schedule_status()
    print(f"\nScheduler Status: {status}")