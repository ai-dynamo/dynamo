# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official public API for prometheus metrics types.

This module provides the official public API for Prometheus metrics in Dynamo.
The metric types are implemented in Rust and exposed via the _core extension module.
"""

# Import directly from the Rust extension module
# Note: IDEs/type checkers may complain about this import because _core is a compiled
# extension module (.so file). However, this import is valid at runtime because the
# Rust code (lib.rs) creates and registers the prometheus_metrics submodule.
from dynamo._core import PyRuntimeMetrics  # type: ignore[attr-defined]
from dynamo._core import prometheus_metrics  # type: ignore[attr-defined]

# Re-export metric type classes from the prometheus_metrics submodule
Counter = prometheus_metrics.Counter
CounterVec = prometheus_metrics.CounterVec
Gauge = prometheus_metrics.Gauge
GaugeVec = prometheus_metrics.GaugeVec
Histogram = prometheus_metrics.Histogram
IntCounter = prometheus_metrics.IntCounter
IntCounterVec = prometheus_metrics.IntCounterVec
IntGauge = prometheus_metrics.IntGauge
IntGaugeVec = prometheus_metrics.IntGaugeVec

# RuntimeMetrics is in the main _core module (as PyRuntimeMetrics), not the submodule
RuntimeMetrics = PyRuntimeMetrics

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
