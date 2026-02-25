# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD (Attention-FFN Disaggregation) Metrics and Monitoring.

This module provides metrics collection and monitoring for AFD workers,
including:
- Transfer latency and throughput
- Attention/FFN compute time breakdown
- Queue lengths and utilization
- Memory usage per worker type

Reference: https://arxiv.org/abs/2601.21351
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

# AFD-specific metrics
AFD_TRANSFER_LATENCY = Histogram(
    'afd_transfer_latency_seconds',
    'Latency of activation transfers between Attention and FFN workers',
    ['worker_type', 'transfer_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

AFD_TRANSFER_BYTES = Counter(
    'afd_transfer_bytes_total',
    'Total bytes transferred between Attention and FFN workers',
    ['worker_type', 'transfer_type'],
)

AFD_QUEUE_LENGTH = Gauge(
    'afd_queue_length',
    'Current queue length for AFD workers',
    ['worker_type'],
)

AFD_COMPUTE_TIME = Histogram(
    'afd_compute_time_seconds',
    'Time spent in computation for AFD workers',
    ['worker_type', 'operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

AFD_ACTIVE_REQUESTS = Gauge(
    'afd_active_requests',
    'Number of active requests being processed',
    ['worker_type'],
)

AFD_MEMORY_USAGE = Gauge(
    'afd_memory_usage_bytes',
    'GPU memory usage for AFD workers',
    ['worker_type', 'memory_type'],
)

AFD_ATTENTION_RATIO = Gauge(
    'afd_attention_ratio',
    'Configured attention ratio (r in r:1 topology)',
    ['worker_id'],
)

AFD_BATCH_SIZE = Histogram(
    'afd_batch_size',
    'Batch size distribution for AFD workers',
    ['worker_type'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
)


@dataclass
class AFDWorkerMetrics:
    """Metrics for a single AFD worker.
    
    Tracks performance metrics for either an Attention or FFN worker.
    """
    
    worker_type: str  # "attention" or "ffn"
    worker_id: str
    
    # Request metrics
    total_requests: int = 0
    active_requests: int = 0
    
    # Timing metrics
    total_compute_time_ms: float = 0.0
    total_transfer_time_ms: float = 0.0
    
    # Transfer metrics
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    # Queue metrics
    current_queue_length: int = 0
    max_queue_length: int = 0
    
    # Memory metrics (in bytes)
    kv_cache_memory: int = 0
    activation_memory: int = 0
    
    # Latency tracking
    latency_samples: List[float] = field(default_factory=list)
    max_latency_samples: int = 1000
    
    def record_request_start(self) -> None:
        """Record the start of a new request."""
        self.total_requests += 1
        self.active_requests += 1
        AFD_ACTIVE_REQUESTS.labels(worker_type=self.worker_type).inc()
    
    def record_request_end(self, compute_time_ms: float) -> None:
        """Record the end of a request."""
        self.active_requests = max(0, self.active_requests - 1)
        AFD_ACTIVE_REQUESTS.labels(worker_type=self.worker_type).dec()
        
        self.total_compute_time_ms += compute_time_ms
        AFD_COMPUTE_TIME.labels(
            worker_type=self.worker_type,
            operation='total'
        ).observe(compute_time_ms / 1000)
    
    def record_transfer(
        self,
        bytes_transferred: int,
        latency_ms: float,
        transfer_type: str = "activation",
    ) -> None:
        """Record a transfer operation."""
        if self.worker_type == "attention":
            self.total_bytes_sent += bytes_transferred
        else:
            self.total_bytes_received += bytes_transferred
        
        self.total_transfer_time_ms += latency_ms
        
        # Update Prometheus metrics
        AFD_TRANSFER_BYTES.labels(
            worker_type=self.worker_type,
            transfer_type=transfer_type,
        ).inc(bytes_transferred)
        
        AFD_TRANSFER_LATENCY.labels(
            worker_type=self.worker_type,
            transfer_type=transfer_type,
        ).observe(latency_ms / 1000)
        
        # Track latency samples
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > self.max_latency_samples:
            self.latency_samples.pop(0)
    
    def update_queue_length(self, length: int) -> None:
        """Update the current queue length."""
        self.current_queue_length = length
        self.max_queue_length = max(self.max_queue_length, length)
        AFD_QUEUE_LENGTH.labels(worker_type=self.worker_type).set(length)
    
    def update_memory_usage(
        self,
        kv_cache: Optional[int] = None,
        activation: Optional[int] = None,
    ) -> None:
        """Update memory usage metrics."""
        if kv_cache is not None:
            self.kv_cache_memory = kv_cache
            AFD_MEMORY_USAGE.labels(
                worker_type=self.worker_type,
                memory_type='kv_cache'
            ).set(kv_cache)
        
        if activation is not None:
            self.activation_memory = activation
            AFD_MEMORY_USAGE.labels(
                worker_type=self.worker_type,
                memory_type='activation'
            ).set(activation)
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency from samples."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)
    
    @property
    def p99_latency_ms(self) -> float:
        """Calculate P99 latency."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        p99_index = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(p99_index, len(sorted_samples) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "worker_type": self.worker_type,
            "worker_id": self.worker_id,
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "avg_compute_time_ms": (
                self.total_compute_time_ms / max(1, self.total_requests)
            ),
            "avg_transfer_time_ms": (
                self.total_transfer_time_ms / max(1, self.total_requests)
            ),
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "current_queue_length": self.current_queue_length,
            "max_queue_length": self.max_queue_length,
            "kv_cache_memory": self.kv_cache_memory,
            "activation_memory": self.activation_memory,
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }


class AFDMetricsCollector:
    """Collects and aggregates metrics from all AFD workers.
    
    This class provides:
    1. Centralized metrics collection from Attention and FFN workers
    2. Aggregate statistics calculation
    3. Prometheus export for monitoring systems
    """
    
    def __init__(
        self,
        attention_ratio: int = 1,
        registry: Optional[CollectorRegistry] = None,
    ):
        """Initialize the metrics collector.
        
        Args:
            attention_ratio: The r in r:1 AFD topology
            registry: Optional Prometheus registry
        """
        self.attention_ratio = attention_ratio
        self.registry = registry
        
        # Worker metrics
        self._attention_workers: Dict[str, AFDWorkerMetrics] = {}
        self._ffn_workers: Dict[str, AFDWorkerMetrics] = {}
        
        # Set attention ratio gauge
        AFD_ATTENTION_RATIO.labels(worker_id='global').set(attention_ratio)
        
        logging.info(
            f"AFD Metrics Collector initialized - attention_ratio={attention_ratio}"
        )
    
    def register_attention_worker(self, worker_id: str) -> AFDWorkerMetrics:
        """Register a new Attention worker.
        
        Args:
            worker_id: Unique worker identifier
            
        Returns:
            AFDWorkerMetrics instance for the worker
        """
        metrics = AFDWorkerMetrics(
            worker_type="attention",
            worker_id=worker_id,
        )
        self._attention_workers[worker_id] = metrics
        logging.info(f"Registered Attention worker: {worker_id}")
        return metrics
    
    def register_ffn_worker(self, worker_id: str) -> AFDWorkerMetrics:
        """Register a new FFN worker.
        
        Args:
            worker_id: Unique worker identifier
            
        Returns:
            AFDWorkerMetrics instance for the worker
        """
        metrics = AFDWorkerMetrics(
            worker_type="ffn",
            worker_id=worker_id,
        )
        self._ffn_workers[worker_id] = metrics
        logging.info(f"Registered FFN worker: {worker_id}")
        return metrics
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all workers.
        
        Returns:
            Dictionary of aggregate statistics
        """
        # Attention worker aggregates
        attention_metrics = list(self._attention_workers.values())
        ffv_metrics = list(self._ffn_workers.values())
        
        total_attention_requests = sum(m.total_requests for m in attention_metrics)
        total_ffn_requests = sum(m.total_requests for m in ffv_metrics)
        
        total_attention_bytes = sum(m.total_bytes_sent for m in attention_metrics)
        total_ffn_bytes = sum(m.total_bytes_received for m in ffv_metrics)
        
        # Calculate effective throughput
        total_attention_time = sum(
            m.total_compute_time_ms + m.total_transfer_time_ms
            for m in attention_metrics
        )
        total_ffn_time = sum(
            m.total_compute_time_ms + m.total_transfer_time_ms
            for m in ffv_metrics
        )
        
        # Average latencies
        all_latencies = []
        for m in attention_metrics + ffv_metrics:
            all_latencies.extend(m.latency_samples)
        
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        sorted_latencies = sorted(all_latencies)
        p99_latency = (
            sorted_latencies[int(len(sorted_latencies) * 0.99)]
            if sorted_latencies else 0
        )
        
        return {
            "attention_ratio": self.attention_ratio,
            "num_attention_workers": len(self._attention_workers),
            "num_ffn_workers": len(self._ffn_workers),
            "total_attention_requests": total_attention_requests,
            "total_ffn_requests": total_ffn_requests,
            "total_bytes_transferred": total_attention_bytes,
            "total_attention_time_ms": total_attention_time,
            "total_ffn_time_ms": total_ffn_time,
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "attention_workers": {
                wid: m.to_dict() for wid, m in self._attention_workers.items()
            },
            "ffn_workers": {
                wid: m.to_dict() for wid, m in self._ffn_workers.items()
            },
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        from prometheus_client import generate_latest
        
        # Update aggregate gauges
        agg = self.get_aggregate_metrics()
        AFD_QUEUE_LENGTH.labels(worker_type='attention_all').set(
            sum(m.current_queue_length for m in self._attention_workers.values())
        )
        AFD_QUEUE_LENGTH.labels(worker_type='ffn_all').set(
            sum(m.current_queue_length for m in self._ffn_workers.values())
        )
        
        return generate_latest(self.registry).decode('utf-8')


class AFDPerformanceAnalyzer:
    """Analyzes AFD performance and recommends optimizations.
    
    This class provides:
    1. Performance bottleneck detection
    2. Optimal attention ratio calculation
    3. Resource utilization analysis
    """
    
    def __init__(self, metrics_collector: AFDMetricsCollector):
        """Initialize the analyzer.
        
        Args:
            metrics_collector: The metrics collector to analyze
        """
        self.collector = metrics_collector
    
    def detect_bottleneck(self) -> Dict[str, Any]:
        """Detect performance bottleneck (attention vs FFN).
        
        Returns:
            Analysis results with bottleneck identification
        """
        metrics = self.collector.get_aggregate_metrics()
        
        attention_time = metrics.get("total_attention_time_ms", 0)
        ffn_time = metrics.get("total_ffn_time_ms", 0)
        
        if attention_time == 0 and ffn_time == 0:
            return {
                "bottleneck": "unknown",
                "reason": "No requests processed yet",
                "recommendation": "Wait for workload data",
            }
        
        # Calculate time ratio
        time_ratio = attention_time / max(1, ffn_time)
        
        # Determine bottleneck
        if time_ratio > 1.5:
            bottleneck = "attention"
            reason = f"Attention time ({attention_time:.1f}ms) >> FFN time ({ffn_time:.1f}ms)"
            recommendation = "Consider increasing attention ratio (more FFN workers per attention group)"
        elif time_ratio < 0.67:
            bottleneck = "ffn"
            reason = f"FFN time ({ffn_time:.1f}ms) >> Attention time ({attention_time:.1f}ms)"
            recommendation = "Consider decreasing attention ratio (fewer attention workers per FFN)"
        else:
            bottleneck = "balanced"
            reason = f"Attention ({attention_time:.1f}ms) ≈ FFN ({ffn_time:.1f}ms)"
            recommendation = "Current configuration is well balanced"
        
        return {
            "bottleneck": bottleneck,
            "time_ratio": time_ratio,
            "attention_time_ms": attention_time,
            "ffn_time_ms": ffn_time,
            "reason": reason,
            "recommendation": recommendation,
            "current_ratio": self.collector.attention_ratio,
        }
    
    def calculate_optimal_ratio(
        self,
        target_latency_ms: Optional[float] = None,
    ) -> int:
        """Calculate optimal attention ratio based on observed performance.
        
        Uses the formula from the AFD paper:
        r* = ceil(T_attn / T_ffn)
        
        Args:
            target_latency_ms: Optional target latency constraint
            
        Returns:
            Recommended attention ratio
        """
        metrics = self.collector.get_aggregate_metrics()
        
        attention_time = metrics.get("total_attention_time_ms", 0)
        ffn_time = metrics.get("total_ffn_time_ms", 0)
        
        if attention_time == 0 or ffn_time == 0:
            return self.collector.attention_ratio
        
        # Calculate optimal ratio
        avg_attention_time = attention_time / max(1, metrics["total_attention_requests"])
        avg_ffn_time = ffn_time / max(1, metrics["total_ffn_requests"])
        
        optimal_ratio = max(1, int(avg_attention_time / avg_ffn_time))
        
        # Adjust for target latency if specified
        if target_latency_ms is not None:
            current_latency = metrics.get("avg_latency_ms", 0)
            if current_latency > target_latency_ms:
                # Need to reduce ratio for lower latency
                optimal_ratio = max(1, optimal_ratio - 1)
        
        return optimal_ratio
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization across workers.
        
        Returns:
            Utilization analysis
        """
        metrics = self.collector.get_aggregate_metrics()
        
        attention_workers = list(self.collector._attention_workers.values())
        ffn_workers = list(self.collector._ffn_workers.values())
        
        # Memory utilization
        attention_memory = sum(m.kv_cache_memory + m.activation_memory for m in attention_workers)
        ffn_memory = sum(m.kv_cache_memory + m.activation_memory for m in ffn_workers)
        
        # Queue utilization
        attention_queue = sum(m.current_queue_length for m in attention_workers)
        ffn_queue = sum(m.current_queue_length for m in ffn_workers)
        
        return {
            "attention": {
                "num_workers": len(attention_workers),
                "total_memory_bytes": attention_memory,
                "total_queue_length": attention_queue,
                "avg_queue_length": attention_queue / max(1, len(attention_workers)),
            },
            "ffn": {
                "num_workers": len(ffn_workers),
                "total_memory_bytes": ffn_memory,
                "total_queue_length": ffn_queue,
                "avg_queue_length": ffn_queue / max(1, len(ffn_workers)),
            },
            "ratio_actual": len(attention_workers) / max(1, len(ffn_workers)),
            "ratio_configured": self.collector.attention_ratio,
        }
