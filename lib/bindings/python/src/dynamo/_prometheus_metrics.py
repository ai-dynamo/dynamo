# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export prometheus_metrics types from _core for convenience.

This module exists alongside _prometheus_metrics.pyi but serves a different purpose:
- This .py file: Executed at runtime to re-export metric types from _core.prometheus_metrics,
  providing a clean public API (dynamo._prometheus_metrics.Counter) instead of exposing
  internal implementation details (dynamo._core.prometheus_metrics.Counter).
- The .pyi file: Used only by type checkers/IDEs for type hints and autocomplete. Never executed.
  Provides signatures and docstrings for the Rust-implemented classes.

Both files are needed: .py provides the runtime imports, .pyi provides static type information.
"""

# Note: IDEs/type checkers may complain about this import because _core is a compiled
# extension module (.so file), not a Python module. The type stub (_core.pyi) doesn't
# declare the prometheus_metrics submodule. However, this import is valid at runtime
# because the Rust code (lib.rs) creates and registers the prometheus_metrics submodule
# via PyModule::new() and add_submodule(). The type: ignore suppresses the false warning.
from dynamo._core import (  # type: ignore[attr-defined]
    PyRuntimeMetrics,
    prometheus_metrics,
)

# Import metric type classes from the prometheus_metrics submodule
Counter = prometheus_metrics.Counter
CounterVec = prometheus_metrics.CounterVec
Gauge = prometheus_metrics.Gauge
GaugeVec = prometheus_metrics.GaugeVec
Histogram = prometheus_metrics.Histogram
IntCounter = prometheus_metrics.IntCounter
IntCounterVec = prometheus_metrics.IntCounterVec
IntGauge = prometheus_metrics.IntGauge
IntGaugeVec = prometheus_metrics.IntGaugeVec

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
    "PyRuntimeMetrics",
]
