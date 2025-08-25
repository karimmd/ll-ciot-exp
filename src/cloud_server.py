#!/usr/bin/env python3

import asyncio
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from quantization.quantizer import EdgeModelWrapper
from scheduling.task_scheduler import ValueDensityScheduler, IoTTask, TaskType

logger = logging.getLogger(__name__)

class CloudServer:
    """
    Cloud server component for LL-CIoT system
    Handles Mistral-7B inference and coordinates with edge servers
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.scheduler = ValueDensityScheduler(
            preemption_threshold=self.config.get('preemption_threshold', 1.5),
            max_batch_size=self.config.get('max_batch_size', 16),
            min_batch_size=self.config.get('min_batch_size', 4)
        )
        
        # Register this node with scheduler
        self.scheduler.register_node('cloud_server', {
            'node_type': 'cloud',
            'max_concurrent_tasks': self.config.get('max_concurrent_tasks', 32),
            'tasks_per_second': self.config.get('tasks_per_second', 8.0),
            'memory_gb': self.config.get('memory_gb', 256),
            'gpu_memory_gb': self.config.get('gpu_memory_gb', 11)
        })
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0.0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load server configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for cloud server"""
        return {
            'model_path': 'models/mistral/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe',
            'max_concurrent_tasks': 32,
            'tasks_per_second': 8.0,
            'memory_gb': 256,
            'gpu_memory_gb': 11,
            'preemption_threshold': 1.5,
            'max_batch_size': 16,
            'min_batch_size': 4,
            'max_new_tokens': 100,
            'temperature': 0.7
        }
    
    async def initialize(self):
        """Initialize model and tokenizer"""
        logger.info("Initializing cloud server...")
        
        model_path = self.config['model_path']
        if not Path(model_path).exists():
            raise RuntimeError(f"Model not found at {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info("Loading Mistral-7B model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: f"{self.config['gpu_memory_gb']-1}GB"}
        )
        
        logger.info("Cloud server initialized successfully")
    
    def create_task_from_request(self, request_data: Dict) -> IoTTask:
        """Create IoTTask from incoming request"""
        current_time = time.time()
        
        return IoTTask(
            task_id=request_data.get('task_id', f"task_{int(current_time*1000)%1000000}"),
            task_type=TaskType.INFERENCE,
            arrival_time=current_time,
            deadline=current_time + request_data.get('timeout', 30.0),
            priority=request_data.get('priority', 1.0),
            input_size=len(request_data.get('prompt', '')),
            expected_output_size=request_data.get('max_tokens', 100),
            device_id=request_data.get('device_id', 'unknown'),
            model_requirements=request_data.get('model_requirements', {})
        )
    
    async def process_inference_request(self, request_data: Dict) -> Dict:
        """Process a single inference request"""
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Create task
            task = self.create_task_from_request(request_data)
            
            # Submit to scheduler
            if not self.scheduler.submit_task(task):
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': 'Task rejected by scheduler',
                    'task_id': task.task_id
                }
            
            # Create and schedule batch
            batch = self.scheduler.create_batch(target_node='cloud_server')
            if not batch:
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': 'Failed to create batch',
                    'task_id': task.task_id
                }
            
            if not self.scheduler.schedule_batch(batch):
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': 'Failed to schedule batch',
                    'task_id': task.task_id
                }
            
            # Execute batch
            results = await self.execute_batch(batch)
            
            # Complete batch
            self.scheduler.complete_batch(batch.batch_id, 'cloud_server', results)
            
            # Find our task result
            task_result = None
            for task_in_batch, result in zip(batch.tasks, results):
                if task_in_batch.task_id == task.task_id:
                    task_result = result
                    break
            
            if task_result and task_result.get('success', False):
                self.stats['successful_requests'] += 1
                inference_time = time.time() - start_time
                self.stats['avg_latency'] = (
                    (self.stats['avg_latency'] * (self.stats['successful_requests'] - 1) + inference_time) /
                    self.stats['successful_requests']
                )
                
                return {
                    'success': True,
                    'response': task_result['response'],
                    'inference_time': inference_time,
                    'tokens_generated': task_result.get('tokens_generated', 0),
                    'task_id': task.task_id
                }
            else:
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': task_result.get('error', 'Unknown error'),
                    'task_id': task.task_id
                }
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Inference request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_id': request_data.get('task_id', 'unknown')
            }
    
    async def execute_batch(self, batch) -> List[Dict]:
        """Execute a batch of tasks"""
        if not self.model or not self.tokenizer:
            return [{'success': False, 'error': 'Model not initialized'} for _ in batch.tasks]
        
        results = []
        
        for task in batch.tasks:
            try:
                # Get prompt from task (in real deployment, this would come from request)
                prompt = f"Process IoT request from device {task.device_id}"
                
                start_time = time.time()
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=min(self.config['max_new_tokens'], task.expected_output_size),
                        temperature=self.config['temperature'],
                        do_sample=True,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        attention_mask=inputs.attention_mask
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                inference_time = time.time() - start_time
                tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
                
                results.append({
                    'success': True,
                    'response': response,
                    'inference_time': inference_time,
                    'tokens_generated': tokens_generated
                })
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_server_stats(self) -> Dict:
        """Get server statistics"""
        scheduler_stats = self.scheduler.get_queue_status()
        
        return {
            'server_type': 'cloud',
            'model_loaded': self.model is not None,
            'requests': self.stats,
            'scheduler': scheduler_stats,
            'gpu_info': self._get_gpu_info() if torch.cuda.is_available() else None
        }
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return None
        
        return {
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3,
            'max_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }

async def main():
    parser = argparse.ArgumentParser(description='Cloud Server for LL-CIoT')
    parser.add_argument('--config', default='configs/cloud_config.json', 
                       help='Configuration file path')
    parser.add_argument('--port', type=int, default=8000, 
                       help='Server port')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize cloud server
    server = CloudServer(args.config)
    await server.initialize()
    
    logger.info(f"Cloud server started on port {args.port}")
    logger.info("Server statistics:")
    logger.info(json.dumps(server.get_server_stats(), indent=2))
    
    # In real deployment, this would run a web server (FastAPI, Flask, etc.)
    # For now, demonstrate with a test request
    test_request = {
        'task_id': 'test_001',
        'prompt': 'Analyze IoT sensor data',
        'priority': 2.0,
        'timeout': 30.0,
        'max_tokens': 50,
        'device_id': 'sensor_001'
    }
    
    result = await server.process_inference_request(test_request)
    logger.info(f"Test result: {result}")

if __name__ == "__main__":
    asyncio.run(main())