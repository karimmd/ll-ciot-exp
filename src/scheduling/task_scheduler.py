import heapq
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TaskType(Enum):
    INFERENCE = "inference"
    FINE_TUNING = "fine_tuning"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class IoTTask:
    task_id: str
    task_type: TaskType
    arrival_time: float
    deadline: float
    priority: float
    input_size: int
    expected_output_size: int
    device_id: str
    model_requirements: Dict = field(default_factory=dict)
    
    @property
    def value_density(self) -> float:
        """Value density function as per manuscript"""
        time_window = max(0.1, self.deadline - self.arrival_time)
        return self.priority / time_window
    
    @property
    def urgency(self) -> float:
        """Current urgency based on remaining time"""
        remaining_time = max(0.1, self.deadline - time.time())
        return 1.0 / remaining_time
    
    def is_expired(self) -> bool:
        """Check if task has exceeded deadline"""
        return time.time() > self.deadline

@dataclass
class BatchInfo:
    batch_id: str
    tasks: List[IoTTask]
    target_node: str
    creation_time: float
    estimated_completion_time: float
    cumulative_value_density: float
    
    def should_preempt(self, new_batch_density: float, threshold: float) -> bool:
        """Check if this batch should be preempted"""
        return new_batch_density > (self.cumulative_value_density * threshold)

class ValueDensityScheduler:
    """
    Task scheduler implementing value density function (VDF) algorithm
    """
    
    def __init__(self, preemption_threshold: float = 1.5, 
                 max_batch_size: int = 16, min_batch_size: int = 4):
        self.preemption_threshold = preemption_threshold
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        
        # Task queues
        self.pending_tasks = []  # Priority queue
        self.active_batches = {}  # node_id -> BatchInfo
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Node information
        self.node_capacities = {}
        self.node_loads = defaultdict(float)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'batches_created': 0,
            'preemptions': 0
        }
    
    def register_node(self, node_id: str, capacity: Dict):
        """Register a compute node with the scheduler"""
        with self.lock:
            self.node_capacities[node_id] = capacity
            logger.info(f"Registered node {node_id} with capacity: {capacity}")
    
    def submit_task(self, task: IoTTask) -> bool:
        """Submit a task to the scheduler"""
        with self.lock:
            if task.is_expired():
                logger.warning(f"Task {task.task_id} already expired")
                return False
            
            # Calculate priority score combining value density and urgency
            priority_score = -(task.value_density * task.urgency)  # Negative for min-heap
            
            heapq.heappush(self.pending_tasks, (priority_score, task.arrival_time, task))
            self.stats['tasks_submitted'] += 1
            
            logger.debug(f"Task {task.task_id} submitted with priority {-priority_score:.4f}")
            return True
    
    def create_batch(self, target_node: Optional[str] = None) -> Optional[BatchInfo]:
        """Create a batch of tasks for execution"""
        with self.lock:
            if not self.pending_tasks:
                return None
            
            batch_tasks = []
            cumulative_density = 0.0
            current_time = time.time()
            
            # Select tasks for batch
            temp_tasks = []
            while (len(batch_tasks) < self.max_batch_size and 
                   self.pending_tasks and 
                   len(temp_tasks) < len(self.pending_tasks)):
                
                priority_score, arrival_time, task = heapq.heappop(self.pending_tasks)
                
                if task.is_expired():
                    self.failed_tasks.append(task)
                    self.stats['tasks_failed'] += 1
                    continue
                
                batch_tasks.append(task)
                cumulative_density += task.value_density
                
            # Restore remaining tasks
            for task in temp_tasks:
                if task not in batch_tasks:
                    priority_score = -(task.value_density * task.urgency)
                    heapq.heappush(self.pending_tasks, (priority_score, task.arrival_time, task))
            
            if len(batch_tasks) < self.min_batch_size:
                # Restore tasks to queue
                for task in batch_tasks:
                    priority_score = -(task.value_density * task.urgency)
                    heapq.heappush(self.pending_tasks, (priority_score, task.arrival_time, task))
                return None
            
            # Create batch
            batch_id = f"batch_{int(current_time * 1000) % 1000000}"
            estimated_completion = current_time + self._estimate_batch_time(batch_tasks, target_node)
            
            batch = BatchInfo(
                batch_id=batch_id,
                tasks=batch_tasks,
                target_node=target_node or self._select_best_node(batch_tasks),
                creation_time=current_time,
                estimated_completion_time=estimated_completion,
                cumulative_value_density=cumulative_density
            )
            
            self.stats['batches_created'] += 1
            logger.info(f"Created batch {batch_id} with {len(batch_tasks)} tasks, "
                       f"density={cumulative_density:.4f}")
            
            return batch
    
    def schedule_batch(self, batch: BatchInfo) -> bool:
        """Schedule a batch on target node with preemption check"""
        with self.lock:
            node_id = batch.target_node
            
            # Check for preemption
            if node_id in self.active_batches:
                current_batch = self.active_batches[node_id]
                if batch.should_preempt(current_batch.cumulative_value_density, 
                                      self.preemption_threshold):
                    logger.info(f"Preempting batch {current_batch.batch_id} with {batch.batch_id}")
                    self._preempt_batch(current_batch)
                    self.stats['preemptions'] += 1
                else:
                    # Queue batch for later
                    return False
            
            # Schedule batch
            self.active_batches[node_id] = batch
            self.node_loads[node_id] += len(batch.tasks)
            
            logger.info(f"Scheduled batch {batch.batch_id} on {node_id}")
            return True
    
    def complete_batch(self, batch_id: str, node_id: str, results: List[Dict]) -> bool:
        """Mark batch as completed and process results"""
        with self.lock:
            if node_id not in self.active_batches:
                logger.warning(f"No active batch on node {node_id}")
                return False
            
            batch = self.active_batches[node_id]
            if batch.batch_id != batch_id:
                logger.warning(f"Batch ID mismatch on {node_id}: expected {batch.batch_id}, got {batch_id}")
                return False
            
            # Process results
            for i, (task, result) in enumerate(zip(batch.tasks, results)):
                if result.get('success', False):
                    self.completed_tasks.append((task, result))
                    self.stats['tasks_completed'] += 1
                else:
                    self.failed_tasks.append(task)
                    self.stats['tasks_failed'] += 1
            
            # Remove from active batches
            del self.active_batches[node_id]
            self.node_loads[node_id] -= len(batch.tasks)
            
            logger.info(f"Completed batch {batch_id} on {node_id}")
            return True
    
    def _select_best_node(self, tasks: List[IoTTask]) -> str:
        """Select best node for task batch"""
        if not self.node_capacities:
            return "default_node"
        
        best_node = None
        best_score = float('inf')
        
        for node_id, capacity in self.node_capacities.items():
            # Calculate load score
            load_score = self.node_loads[node_id] / capacity.get('max_concurrent_tasks', 10)
            
            # Prefer edge nodes for small tasks, cloud for large ones
            avg_input_size = sum(task.input_size for task in tasks) / len(tasks)
            if capacity.get('node_type') == 'edge' and avg_input_size < 1024:
                load_score *= 0.8  # Prefer edge for small tasks
            elif capacity.get('node_type') == 'cloud' and avg_input_size >= 1024:
                load_score *= 0.8  # Prefer cloud for large tasks
            
            if load_score < best_score:
                best_score = load_score
                best_node = node_id
        
        return best_node or list(self.node_capacities.keys())[0]
    
    def _estimate_batch_time(self, tasks: List[IoTTask], node_id: Optional[str]) -> float:
        """Estimate batch execution time"""
        if not node_id or node_id not in self.node_capacities:
            return len(tasks) * 1.0  # Default 1 second per task
        
        capacity = self.node_capacities[node_id]
        processing_rate = capacity.get('tasks_per_second', 1.0)
        
        return len(tasks) / processing_rate
    
    def _preempt_batch(self, batch: BatchInfo):
        """Preempt a running batch"""
        # Return tasks to pending queue
        for task in batch.tasks:
            priority_score = -(task.value_density * task.urgency)
            heapq.heappush(self.pending_tasks, (priority_score, task.arrival_time, task))
        
        # Remove from active batches
        if batch.target_node in self.active_batches:
            del self.active_batches[batch.target_node]
            self.node_loads[batch.target_node] -= len(batch.tasks)
    
    def get_queue_status(self) -> Dict:
        """Get current scheduler status"""
        with self.lock:
            return {
                'pending_tasks': len(self.pending_tasks),
                'active_batches': len(self.active_batches),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'statistics': self.stats.copy(),
                'node_loads': dict(self.node_loads)
            }
    
    def cleanup_expired_tasks(self):
        """Remove expired tasks from queues"""
        with self.lock:
            # Clean pending tasks
            valid_tasks = []
            expired_count = 0
            
            while self.pending_tasks:
                priority_score, arrival_time, task = heapq.heappop(self.pending_tasks)
                if not task.is_expired():
                    valid_tasks.append((priority_score, arrival_time, task))
                else:
                    self.failed_tasks.append(task)
                    self.stats['tasks_failed'] += 1
                    expired_count += 1
            
            # Restore valid tasks
            for task_tuple in valid_tasks:
                heapq.heappush(self.pending_tasks, task_tuple)
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired tasks")