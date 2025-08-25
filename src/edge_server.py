#!/usr/bin/env python3

import asyncio
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from quantization.quantizer import EdgeModelWrapper, quantize_model_for_edge
from scheduling.task_scheduler import ValueDensityScheduler, IoTTask, TaskType

logger = logging.getLogger(__name__)

class EdgeServer:
    """
    Edge server component for LL-CIoT system
    Handles small model inference on CPU for edge deployment environments
    """
    
    def __init__(self, server_id: str, config_path: str):
        self.server_id = server_id
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        
        # Initialize scheduler for this edge node
        self.scheduler = ValueDensityScheduler(
            preemption_threshold=self.config.get('preemption_threshold', 1.2),
            max_batch_size=self.config.get('max_batch_size', 8),
            min_batch_size=self.config.get('min_batch_size', 2)
        )
        
        # Register this node with scheduler
        self.scheduler.register_node(self.server_id, {
            'node_type': 'edge',
            'max_concurrent_tasks': self.config.get('max_concurrent_tasks', 8),
            'tasks_per_second': self.config.get('tasks_per_second', 2.0),
            'memory_gb': self.config.get('memory_gb', 16),
            'gpu_memory_gb': self.config.get('gpu_memory_gb', 2),  # Edge GPU
            'cpu_cores': self.config.get('cpu_cores', 4)
        })
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency': 0.0,
            'cpu_inference_count': 0
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
        """Default configuration for edge server"""
        return {
            'model_path': 'models/tinyllama/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6',
            'use_cpu_inference': True,  # Edge deployment constraint
            'quantization_bits': 8,
            'max_concurrent_tasks': 8,
            'tasks_per_second': 2.0,
            'memory_gb': 16,
            'gpu_memory_gb': 2,
            'cpu_cores': 4,
            'preemption_threshold': 1.2,
            'max_batch_size': 8,
            'min_batch_size': 2,
            'max_new_tokens': 50,
            'temperature': 0.8
        }
    
    async def initialize(self):
        """Initialize model and tokenizer with CPU inference"""
        logger.info(f"Initializing edge server {self.server_id}...")
        
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
        
        # Load model on CPU (edge deployment constraint)
        logger.info("Loading edge model on CPU (edge deployment constraint)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU inference
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Wrap with edge optimization
        self.model = EdgeModelWrapper(
            base_model, 
            quantization_bits=self.config['quantization_bits']
        )
        
        # Apply quantization for edge deployment
        if self.config.get('quantization_bits', 8) < 16:
            logger.info(f"Applying {self.config['quantization_bits']}-bit quantization...")
            self.model.quantize_for_deployment()
        
        logger.info(f"Edge server {self.server_id} initialized successfully")
        logger.info(f"Model memory usage: {self.model.get_memory_usage():.2f} MB")
    
    def create_task_from_request(self, request_data: Dict) -> IoTTask:
        """Create IoTTask from incoming request"""
        current_time = time.time()
        
        return IoTTask(
            task_id=request_data.get('task_id', f"edge_task_{int(current_time*1000)%1000000}"),
            task_type=TaskType.INFERENCE,
            arrival_time=current_time,
            deadline=current_time + request_data.get('timeout', 15.0),  # Shorter timeout for edge
            priority=request_data.get('priority', 1.0),
            input_size=len(request_data.get('prompt', '')),
            expected_output_size=request_data.get('max_tokens', 30),
            device_id=request_data.get('device_id', 'unknown'),
            model_requirements={'prefer_edge': True}
        )
    
    async def process_inference_request(self, request_data: Dict) -> Dict:
        """Process a single inference request on edge"""
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
                    'task_id': task.task_id,
                    'server_id': self.server_id
                }
            
            # Create and schedule batch
            batch = self.scheduler.create_batch(target_node=self.server_id)
            if not batch:
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': 'Failed to create batch',
                    'task_id': task.task_id,
                    'server_id': self.server_id
                }
            
            if not self.scheduler.schedule_batch(batch):
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': 'Failed to schedule batch',
                    'task_id': task.task_id,
                    'server_id': self.server_id
                }
            
            # Execute batch
            results = await self.execute_batch(batch)
            
            # Complete batch
            self.scheduler.complete_batch(batch.batch_id, self.server_id, results)
            
            # Find our task result
            task_result = None
            for task_in_batch, result in zip(batch.tasks, results):
                if task_in_batch.task_id == task.task_id:
                    task_result = result
                    break
            
            if task_result and task_result.get('success', False):
                self.stats['successful_requests'] += 1
                self.stats['cpu_inference_count'] += 1
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
                    'task_id': task.task_id,
                    'server_id': self.server_id,
                    'inference_location': 'cpu'
                }
            else:
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': task_result.get('error', 'Unknown error'),
                    'task_id': task.task_id,
                    'server_id': self.server_id
                }
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Edge inference request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_id': request_data.get('task_id', 'unknown'),
                'server_id': self.server_id
            }
    
    async def execute_batch(self, batch) -> List[Dict]:
        """Execute a batch of tasks on CPU"""
        if not self.model or not self.tokenizer:
            return [{'success': False, 'error': 'Model not initialized'} for _ in batch.tasks]
        
        results = []
        
        for task in batch.tasks:
            try:
                # Get prompt from task
                prompt = f"Process IoT request from device {task.device_id}"
                
                start_time = time.time()
                
                # Tokenize input (shorter max length for edge)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256  # Shorter for edge
                )
                
                # CPU inference
                with torch.no_grad():
                    outputs = self.model.base_model.generate(
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
                    'tokens_generated': tokens_generated,
                    'inference_location': 'cpu'
                })
                
            except Exception as e:
                logger.error(f"Edge task {task.task_id} failed: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_server_stats(self) -> Dict:
        """Get server statistics"""
        scheduler_stats = self.scheduler.get_queue_status()
        system_stats = self._get_system_info()
        
        return {
            'server_id': self.server_id,
            'server_type': 'edge',
            'model_loaded': self.model is not None,
            'inference_location': 'cpu',
            'quantization_applied': self.model.is_quantized if self.model else False,
            'requests': self.stats,
            'scheduler': scheduler_stats,
            'system': system_stats
        }
    
    def _get_system_info(self) -> Dict:
        """Get system resource information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        system_info = {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory.percent,
            'memory_available_gb': memory.available / 1024**3,
            'cpu_cores': psutil.cpu_count()
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            system_info['gpu'] = {
                'device_name': torch.cuda.get_device_name(0),
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'note': 'Low-end GPU - insufficient for LLM inference, using CPU'
            }
        
        return system_info

async def main():
    parser = argparse.ArgumentParser(description='Edge Server for LL-CIoT')
    parser.add_argument('--server-id', default='edge_server_1', 
                       help='Edge server identifier')
    parser.add_argument('--config', default='configs/edge_config.json', 
                       help='Configuration file path')
    parser.add_argument('--port', type=int, default=8001, 
                       help='Server port')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize edge server
    server = EdgeServer(args.server_id, args.config)
    await server.initialize()
    
    logger.info(f"Edge server {args.server_id} started on port {args.port}")
    logger.info("Server statistics:")
    logger.info(json.dumps(server.get_server_stats(), indent=2))
    
    # In real deployment, this would run a web server
    # For now, demonstrate with a test request
    test_request = {
        'task_id': 'edge_test_001',
        'prompt': 'Process sensor reading',
        'priority': 1.5,
        'timeout': 15.0,
        'max_tokens': 30,
        'device_id': 'iot_sensor_001'
    }
    
    result = await server.process_inference_request(test_request)
    logger.info(f"Test result: {result}")

if __name__ == "__main__":
    asyncio.run(main())