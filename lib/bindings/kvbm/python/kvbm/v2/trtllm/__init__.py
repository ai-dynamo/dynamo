# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM v2 connector for TensorRT-LLM.

Usage via TRT-LLM's KvCacheConnectorConfig::

    kv_connector_config:
        connector_module: "kvbm.v2.trtllm"
        connector_scheduler_class: "TrtllmConnectorScheduler"
        connector_worker_class: "TrtllmConnectorWorker"

Classes are imported lazily to avoid pulling in TensorRT-LLM at
package-scan time (e.g. when only ``sched_output`` is needed).
"""


def __getattr__(name: str):
    if name == "TrtllmConnectorScheduler":
        from .scheduler import TrtllmConnectorScheduler

        return TrtllmConnectorScheduler
    if name == "TrtllmConnectorWorker":
        from .worker import TrtllmConnectorWorker

        return TrtllmConnectorWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrtllmConnectorScheduler",
    "TrtllmConnectorWorker",
]
