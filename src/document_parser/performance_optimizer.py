#!/usr/bin/env python3
"""
Performance Optimizer for Enhanced Document Parser

This module provides performance optimization capabilities including:
- Batch processing optimization
- Memory management
- Parallel processing
- Caching mechanisms
- Performance monitoring
- Resource optimization
"""

import os
import time
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import queue
import psutil
import numpy as np
import cv2
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pickle
import hashlib
import json
from pathlib import Path
import gc
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    cpu_usage_avg: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_docs_per_second: float = 0.0
    error_rate: float = 0.0
    
    def update_metrics(self, processing_time: float, success: bool, memory_usage: float = None):
        """Update metrics with new processing result"""
        self.total_documents += 1
        self.total_processing_time += processing_time
        
        if success:
            self.successful_documents += 1
        else:
            self.failed_documents += 1
        
        self.average_processing_time = self.total_processing_time / self.total_documents
        self.error_rate = self.failed_documents / self.total_documents
        self.throughput_docs_per_second = self.total_documents / self.total_processing_time if self.total_processing_time > 0 else 0
        
        if memory_usage:
            self.peak_memory_usage = max(self.peak_memory_usage, memory_usage)

@dataclass
class ProcessingTask:
    """Individual processing task"""
    task_id: str
    file_path: str
    document_type: str
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority

class DocumentCache:
    """Intelligent caching system for document processing results"""
    
    def __init__(self, max_size: int = 1000, cache_dir: str = None):
        self.max_size = max_size
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), '.document_cache')
        self.memory_cache = {}
        self.access_times = deque()
        self.hit_count = 0
        self.miss_count = 0
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load persistent cache index
        self.index_file = os.path.join(self.cache_dir, 'cache_index.json')
        self.persistent_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.persistent_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _generate_cache_key(self, file_path: str, processing_config: Dict[str, Any] = None) -> str:
        """Generate cache key for a file and processing configuration"""
        # Get file stats for cache invalidation
        try:
            stat = os.stat(file_path)
            file_info = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        except OSError:
            file_info = file_path
        
        # Include processing configuration in key
        config_str = json.dumps(processing_config or {}, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.md5(f"{file_info}_{config_str}".encode()).hexdigest()
        return cache_key
    
    def get(self, file_path: str, processing_config: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result"""
        cache_key = self._generate_cache_key(file_path, processing_config)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.hit_count += 1
            self.access_times.append((cache_key, time.time()))
            return self.memory_cache[cache_key]
        
        # Check persistent cache
        if cache_key in self.persistent_index:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Move to memory cache
                    self.memory_cache[cache_key] = result
                    self.hit_count += 1
                    self.access_times.append((cache_key, time.time()))
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cached result: {e}")
                    # Remove invalid cache entry
                    del self.persistent_index[cache_key]
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
        
        self.miss_count += 1
        return None
    
    def put(self, file_path: str, result: Any, processing_config: Dict[str, Any] = None):
        """Cache processing result"""
        cache_key = self._generate_cache_key(file_path, processing_config)
        
        # Store in memory cache
        self.memory_cache[cache_key] = result
        self.access_times.append((cache_key, time.time()))
        
        # Store in persistent cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            self.persistent_index[cache_key] = {
                'file_path': file_path,
                'created_at': time.time(),
                'config': processing_config
            }
            self._save_cache_index()
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
        
        # Cleanup if cache is too large
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up cache when it exceeds max size"""
        if len(self.memory_cache) > self.max_size:
            # Remove oldest accessed items
            while len(self.access_times) > self.max_size:
                old_key, _ = self.access_times.popleft()
                if old_key in self.memory_cache:
                    del self.memory_cache[old_key]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self.access_times.clear()
        self.persistent_index.clear()
        
        # Remove cache files
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
        
        self._save_cache_index()

class ResourceMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Store metrics
                timestamp = time.time()
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['disk_io'].append({
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                })
                self.metrics['timestamps'].append(timestamp)
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average resource usage metrics"""
        if not self.metrics['cpu_usage']:
            return {'cpu_avg': 0.0, 'memory_avg': 0.0}
        
        return {
            'cpu_avg': np.mean(self.metrics['cpu_usage']),
            'memory_avg': np.mean(self.metrics['memory_usage']),
            'cpu_max': np.max(self.metrics['cpu_usage']),
            'memory_max': np.max(self.metrics['memory_usage'])
        }
    
    def reset_metrics(self):
        """Reset collected metrics"""
        for key in self.metrics:
            self.metrics[key].clear()

class BatchProcessor:
    """Optimized batch processing for documents"""
    
    def __init__(self, 
                 processor_func: Callable,
                 max_workers: int = None,
                 use_multiprocessing: bool = False,
                 batch_size: int = 10,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 enable_monitoring: bool = True):
        
        self.processor_func = processor_func
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_multiprocessing = use_multiprocessing
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.cache = DocumentCache(max_size=cache_size) if enable_caching else None
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        self.metrics = PerformanceMetrics()
        
        # Task queue for priority processing
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        
        logger.info(f"BatchProcessor initialized with {self.max_workers} workers")
    
    def add_task(self, file_path: str, document_type: str, priority: int = 1, metadata: Dict[str, Any] = None):
        """Add a processing task to the queue"""
        task_id = f"{file_path}_{int(time.time() * 1000)}"
        task = ProcessingTask(
            task_id=task_id,
            file_path=file_path,
            document_type=document_type,
            priority=priority,
            metadata=metadata or {}
        )
        self.task_queue.put(task)
        return task_id
    
    def process_single_document(self, task: ProcessingTask) -> Tuple[str, Any]:
        """Process a single document with caching and monitoring"""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(task.file_path, {'document_type': task.document_type})
                if cached_result is not None:
                    processing_time = time.time() - start_time
                    self.metrics.update_metrics(processing_time, True)
                    return task.task_id, cached_result
            
            # Process document
            result = self.processor_func(task.file_path, task.document_type)
            
            # Cache result
            if self.cache and result.success:
                self.cache.put(task.file_path, result, {'document_type': task.document_type})
            
            # Update metrics
            processing_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.metrics.update_metrics(processing_time, result.success, memory_usage)
            
            return task.task_id, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_metrics(processing_time, False)
            logger.error(f"Error processing {task.file_path}: {e}")
            return task.task_id, None
    
    def process_batch_parallel(self, tasks: List[ProcessingTask]) -> Dict[str, Any]:
        """Process batch of tasks in parallel"""
        results = {}
        
        # Start monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        try:
            # Choose executor based on configuration
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.process_single_document, task): task 
                    for task in tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        task_id, result = future.result()
                        results[task_id] = result
                    except Exception as e:
                        logger.error(f"Task {task.task_id} failed: {e}")
                        results[task.task_id] = None
        
        finally:
            # Stop monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
        
        return results
    
    def process_queue(self) -> Dict[str, Any]:
        """Process all tasks in the queue"""
        tasks = []
        
        # Collect tasks from queue
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                tasks.append(task)
            except queue.Empty:
                break
        
        if not tasks:
            return {}
        
        logger.info(f"Processing {len(tasks)} tasks in batches of {self.batch_size}")
        
        all_results = {}
        
        # Process in batches
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_results = self.process_batch_parallel(batch)
            all_results.update(batch_results)
            
            # Memory cleanup between batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results
    
    def process_files(self, file_paths: List[str], document_types: List[str] = None) -> Dict[str, Any]:
        """Process a list of files"""
        if document_types is None:
            document_types = ['auto'] * len(file_paths)
        
        # Add tasks to queue
        task_ids = []
        for file_path, doc_type in zip(file_paths, document_types):
            task_id = self.add_task(file_path, doc_type)
            task_ids.append(task_id)
        
        # Process queue
        results = self.process_queue()
        
        # Return results mapped by file path
        file_results = {}
        for task_id in task_ids:
            if task_id in results:
                # Find corresponding file path
                for file_path in file_paths:
                    if task_id.startswith(file_path):
                        file_results[file_path] = results[task_id]
                        break
        
        return file_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'processing_metrics': {
                'total_documents': self.metrics.total_documents,
                'successful_documents': self.metrics.successful_documents,
                'failed_documents': self.metrics.failed_documents,
                'success_rate': (self.metrics.successful_documents / self.metrics.total_documents) if self.metrics.total_documents > 0 else 0,
                'average_processing_time': self.metrics.average_processing_time,
                'throughput_docs_per_second': self.metrics.throughput_docs_per_second,
                'error_rate': self.metrics.error_rate
            },
            'resource_usage': {},
            'cache_performance': {},
            'configuration': {
                'max_workers': self.max_workers,
                'use_multiprocessing': self.use_multiprocessing,
                'batch_size': self.batch_size,
                'enable_caching': self.enable_caching,
                'enable_monitoring': self.enable_monitoring
            }
        }
        
        # Add resource usage metrics
        if self.resource_monitor:
            resource_metrics = self.resource_monitor.get_average_metrics()
            report['resource_usage'] = resource_metrics
        
        # Add cache performance metrics
        if self.cache:
            report['cache_performance'] = {
                'hit_rate': self.cache.get_hit_rate(),
                'total_hits': self.cache.hit_count,
                'total_misses': self.cache.miss_count,
                'cache_size': len(self.cache.memory_cache)
            }
        
        return report
    
    def optimize_configuration(self, sample_files: List[str]) -> Dict[str, Any]:
        """Automatically optimize configuration based on sample processing"""
        logger.info("Optimizing batch processor configuration...")
        
        # Test different configurations
        configurations = [
            {'max_workers': 2, 'batch_size': 5, 'use_multiprocessing': False},
            {'max_workers': 4, 'batch_size': 10, 'use_multiprocessing': False},
            {'max_workers': 8, 'batch_size': 20, 'use_multiprocessing': False},
            {'max_workers': 2, 'batch_size': 5, 'use_multiprocessing': True},
            {'max_workers': 4, 'batch_size': 10, 'use_multiprocessing': True},
        ]
        
        best_config = None
        best_throughput = 0
        
        # Test each configuration with sample files
        for config in configurations:
            try:
                # Create temporary processor with test config
                test_processor = BatchProcessor(
                    processor_func=self.processor_func,
                    max_workers=config['max_workers'],
                    use_multiprocessing=config['use_multiprocessing'],
                    batch_size=config['batch_size'],
                    enable_caching=False,  # Disable caching for fair comparison
                    enable_monitoring=True
                )
                
                # Process sample files
                start_time = time.time()
                test_processor.process_files(sample_files[:min(5, len(sample_files))])
                end_time = time.time()
                
                # Calculate throughput
                processing_time = end_time - start_time
                throughput = len(sample_files) / processing_time if processing_time > 0 else 0
                
                logger.info(f"Config {config}: {throughput:.2f} docs/sec")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
                    
            except Exception as e:
                logger.warning(f"Failed to test configuration {config}: {e}")
        
        if best_config:
            logger.info(f"Optimal configuration found: {best_config}")
            # Apply best configuration
            self.max_workers = best_config['max_workers']
            self.use_multiprocessing = best_config['use_multiprocessing']
            self.batch_size = best_config['batch_size']
        
        return best_config or {}

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self, processor_func: Callable):
        self.processor_func = processor_func
        self.batch_processor = None
        self.optimization_history = []
    
    def create_optimized_processor(self, 
                                 sample_files: List[str] = None,
                                 target_throughput: float = None,
                                 memory_limit_mb: int = None) -> BatchProcessor:
        """Create an optimized batch processor"""
        
        # Create initial processor
        self.batch_processor = BatchProcessor(
            processor_func=self.processor_func,
            enable_caching=True,
            enable_monitoring=True
        )
        
        # Optimize configuration if sample files provided
        if sample_files:
            optimal_config = self.batch_processor.optimize_configuration(sample_files)
            self.optimization_history.append({
                'timestamp': time.time(),
                'config': optimal_config,
                'sample_size': len(sample_files)
            })
        
        return self.batch_processor
    
    def benchmark_processing(self, 
                           test_files: List[str],
                           iterations: int = 3) -> Dict[str, Any]:
        """Benchmark processing performance"""
        if not self.batch_processor:
            self.batch_processor = BatchProcessor(self.processor_func)
        
        results = []
        
        for i in range(iterations):
            logger.info(f"Benchmark iteration {i+1}/{iterations}")
            
            # Reset metrics
            self.batch_processor.metrics = PerformanceMetrics()
            if self.batch_processor.resource_monitor:
                self.batch_processor.resource_monitor.reset_metrics()
            
            # Process files
            start_time = time.time()
            file_results = self.batch_processor.process_files(test_files)
            end_time = time.time()
            
            # Collect metrics
            iteration_result = {
                'iteration': i + 1,
                'total_time': end_time - start_time,
                'throughput': len(test_files) / (end_time - start_time),
                'success_rate': sum(1 for r in file_results.values() if r and r.success) / len(test_files),
                'performance_report': self.batch_processor.get_performance_report()
            }
            
            results.append(iteration_result)
        
        # Calculate averages
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_success_rate = np.mean([r['success_rate'] for r in results])
        
        benchmark_summary = {
            'iterations': iterations,
            'test_files_count': len(test_files),
            'average_throughput': avg_throughput,
            'average_success_rate': avg_success_rate,
            'detailed_results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        return benchmark_summary
    
    def _generate_recommendations(self, benchmark_results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze results
        avg_throughput = np.mean([r['throughput'] for r in benchmark_results])
        avg_cpu = np.mean([r['performance_report']['resource_usage'].get('cpu_avg', 0) for r in benchmark_results])
        avg_memory = np.mean([r['performance_report']['resource_usage'].get('memory_avg', 0) for r in benchmark_results])
        
        # Generate recommendations based on metrics
        if avg_throughput < 1.0:  # Less than 1 doc/sec
            recommendations.append("Consider increasing batch size or number of workers for better throughput")
        
        if avg_cpu < 50:
            recommendations.append("CPU utilization is low - consider increasing worker count")
        elif avg_cpu > 90:
            recommendations.append("CPU utilization is high - consider reducing worker count or using multiprocessing")
        
        if avg_memory > 80:
            recommendations.append("Memory usage is high - consider reducing batch size or enabling more aggressive caching")
        
        cache_hit_rates = [r['performance_report']['cache_performance'].get('hit_rate', 0) for r in benchmark_results]
        if cache_hit_rates and np.mean(cache_hit_rates) < 0.3:
            recommendations.append("Cache hit rate is low - consider increasing cache size or reviewing caching strategy")
        
        return recommendations
    
    def export_performance_report(self, 
                                benchmark_results: Dict[str, Any],
                                output_path: str):
        """Export detailed performance report"""
        report = {
            'timestamp': time.time(),
            'benchmark_results': benchmark_results,
            'optimization_history': self.optimization_history,
            'system_info': {
                'cpu_count': os.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
                'python_version': os.sys.version
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Mock processor function for testing
    def mock_processor(file_path: str, document_type: str):
        """Mock document processor for testing"""
        import time
        import random
        from dataclasses import dataclass
        
        @dataclass
        class MockResult:
            success: bool
            confidence_score: float
            processing_time: float
            
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        return MockResult(
            success=random.random() > 0.1,  # 90% success rate
            confidence_score=random.uniform(0.7, 0.95),
            processing_time=random.uniform(0.1, 0.5)
        )
    
    # Test performance optimizer
    optimizer = PerformanceOptimizer(mock_processor)
    
    # Create test files (mock)
    test_files = [f"test_file_{i}.jpg" for i in range(10)]
    
    # Create optimized processor
    batch_processor = optimizer.create_optimized_processor(test_files[:5])
    
    # Run benchmark
    benchmark_results = optimizer.benchmark_processing(test_files, iterations=2)
    
    print("Benchmark Results:")
    print(f"Average Throughput: {benchmark_results['average_throughput']:.2f} docs/sec")
    print(f"Average Success Rate: {benchmark_results['average_success_rate']:.2%}")
    print("\nRecommendations:")
    for rec in benchmark_results['recommendations']:
        print(f"- {rec}")