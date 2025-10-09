# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file defines Python type stubs for Prometheus metric types.
# It should be kept in sync with:
# - lib/bindings/python/rust/metrics.rs (Rust implementations)
# - lib/runtime/src/metrics.rs (MetricsRegistry trait and Prometheus types)

from typing import Callable, Dict, List, Optional, Tuple

# Specific metric type classes
class Counter:
    """Prometheus Counter metric (float)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def inc(self) -> None:
        """Increment counter by 1"""
        ...
    def inc_by(self, value: float) -> None:
        """Increment counter by value"""
        ...
    def get(self) -> float:
        """Get counter value"""
        ...

class CounterVec:
    """Prometheus CounterVec metric with labels (float)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def variable_labels(self) -> List[str]:
        """Get the variable label names"""
        ...
    def inc(self, labels: Dict[str, str]) -> None:
        """Increment counter by 1 with labels"""
        ...
    def inc_by(self, labels: Dict[str, str], value: float) -> None:
        """Increment counter by value with labels"""
        ...
    def get(self, labels: Dict[str, str]) -> float:
        """Get counter value with labels"""
        ...

class Gauge:
    """Prometheus Gauge metric (float)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def set(self, value: float) -> None:
        """Set gauge value"""
        ...
    def get(self) -> float:
        """Get gauge value"""
        ...
    def inc(self) -> None:
        """Increment gauge by 1"""
        ...
    def inc_by(self, value: float) -> None:
        """Increment gauge by value"""
        ...
    def dec(self) -> None:
        """Decrement gauge by 1"""
        ...
    def dec_by(self, value: float) -> None:
        """Decrement gauge by value"""
        ...
    def add(self, value: float) -> None:
        """Add value to gauge"""
        ...
    def sub(self, value: float) -> None:
        """Subtract value from gauge"""
        ...

class GaugeVec:
    """Prometheus GaugeVec metric with labels (float)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def variable_labels(self) -> List[str]:
        """Get the variable label names"""
        ...
    def set(self, value: float, labels: Dict[str, str]) -> None:
        """Set gauge value with labels"""
        ...
    def get(self, labels: Dict[str, str]) -> float:
        """Get gauge value with labels"""
        ...
    def inc(self, labels: Dict[str, str]) -> None:
        """Increment gauge by 1 with labels"""
        ...
    def dec(self, labels: Dict[str, str]) -> None:
        """Decrement gauge by 1 with labels"""
        ...
    def add(self, labels: Dict[str, str], value: float) -> None:
        """Add value to gauge with labels"""
        ...
    def sub(self, labels: Dict[str, str], value: float) -> None:
        """Subtract value from gauge with labels"""
        ...

class Histogram:
    """Prometheus Histogram metric"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def observe(self, value: float) -> None:
        """Observe a value"""
        ...

class IntCounter:
    """Prometheus IntCounter metric (integer)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def inc(self) -> None:
        """Increment counter by 1"""
        ...
    def inc_by(self, value: int) -> None:
        """Increment counter by value"""
        ...
    def get(self) -> int:
        """Get counter value"""
        ...

class IntCounterVec:
    """Prometheus IntCounterVec metric with labels (integer)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def variable_labels(self) -> List[str]:
        """Get the variable label names"""
        ...
    def inc(self, labels: Dict[str, str]) -> None:
        """Increment counter by 1 with labels"""
        ...
    def inc_by(self, labels: Dict[str, str], value: int) -> None:
        """Increment counter by value with labels"""
        ...
    def get(self, labels: Dict[str, str]) -> int:
        """Get counter value with labels"""
        ...

class IntGauge:
    """Prometheus IntGauge metric (integer)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def set(self, value: int) -> None:
        """Set gauge value"""
        ...
    def get(self) -> int:
        """Get gauge value"""
        ...
    def inc(self) -> None:
        """Increment gauge by 1"""
        ...
    def dec(self) -> None:
        """Decrement gauge by 1"""
        ...
    def add(self, value: int) -> None:
        """Add value to gauge"""
        ...
    def sub(self, value: int) -> None:
        """Subtract value from gauge"""
        ...

class IntGaugeVec:
    """Prometheus IntGaugeVec metric with labels (integer)"""
    def name(self) -> str:
        """Get the metric name"""
        ...
    def const_labels(self) -> Dict[str, str]:
        """Get the constant labels"""
        ...
    def variable_labels(self) -> List[str]:
        """Get the variable label names"""
        ...
    def set(self, value: int, labels: Dict[str, str]) -> None:
        """Set gauge value with labels"""
        ...
    def get(self, labels: Dict[str, str]) -> int:
        """Get gauge value with labels"""
        ...
    def inc(self, labels: Dict[str, str]) -> None:
        """Increment gauge by 1 with labels"""
        ...
    def dec(self, labels: Dict[str, str]) -> None:
        """Decrement gauge by 1 with labels"""
        ...
    def add(self, labels: Dict[str, str], value: int) -> None:
        """Add value to gauge with labels"""
        ...
    def sub(self, labels: Dict[str, str], value: int) -> None:
        """Subtract value from gauge with labels"""
        ...

class RuntimeMetrics:
    """
    Helper class for creating Prometheus metrics on an Endpoint.

    Provides factory methods to create various Prometheus metric types
    that are automatically registered with the endpoint's Prometheus registry.
    Also provides utilities for registering metrics callbacks.
    """

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a Python callback to be invoked before metrics are scraped.

        This allows you to update metric values dynamically when the /metrics endpoint
        is accessed. The callback will be executed synchronously before serving metrics.

        Args:
            callback: A callable that takes no arguments and returns None.
                     This function will be called each time metrics are scraped.

        Example:
            ```python
            metrics = endpoint.metrics
            counter = metrics.create_intcounter("request_count", "Total requests")

            def update_metrics():
                counter.inc()

            metrics.register_update_callback(update_metrics)
            ```
        """
        ...

    def create_counter(self, name: str, description: str, const_labels: Optional[List[Tuple[str, str]]] = None) -> Counter:
        """Create a Counter metric (float) with optional static labels"""
        ...

    def create_countervec(self, name: str, description: str, label_names: List[str], const_labels: Optional[List[Tuple[str, str]]] = None) -> CounterVec:
        """Create a CounterVec metric with labels (float)"""
        ...

    def create_gauge(self, name: str, description: str, const_labels: Optional[List[Tuple[str, str]]] = None) -> Gauge:
        """Create a Gauge metric (float) with optional static labels"""
        ...

    def create_gaugevec(self, name: str, description: str, label_names: List[str], const_labels: Optional[List[Tuple[str, str]]] = None) -> GaugeVec:
        """Create a GaugeVec metric with labels (float)"""
        ...

    def create_histogram(self, name: str, description: str, const_labels: Optional[List[Tuple[str, str]]] = None) -> Histogram:
        """Create a Histogram metric with optional static labels"""
        ...

    def create_intcounter(self, name: str, description: str, const_labels: Optional[List[Tuple[str, str]]] = None) -> IntCounter:
        """Create an IntCounter metric (integer) with optional static labels"""
        ...

    def create_intcountervec(self, name: str, description: str, label_names: List[str], const_labels: Optional[List[Tuple[str, str]]] = None) -> IntCounterVec:
        """Create an IntCounterVec metric with labels (integer)"""
        ...

    def create_intgauge(self, name: str, description: str, const_labels: Optional[List[Tuple[str, str]]] = None) -> IntGauge:
        """Create an IntGauge metric (integer) with optional static labels"""
        ...

    def create_intgaugevec(self, name: str, description: str, label_names: List[str], const_labels: Optional[List[Tuple[str, str]]] = None) -> IntGaugeVec:
        """Create an IntGaugeVec metric with labels (integer)"""
        ...

__all__ = [
    "Counter",
    "CounterVec",
    "Gauge",
    "GaugeVec",
    "Histogram",
    "IntCounter",
    "IntCounterVec",
    "IntGauge",
    "IntGaugeVec",
    "RuntimeMetrics",
]
