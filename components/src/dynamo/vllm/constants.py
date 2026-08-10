# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants for vLLM backend.

DisaggregationMode is defined in dynamo.common.constants and re-exported here
so that existing imports from dynamo.vllm.constants continue to work.
"""

from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode

GMS_LOAD_FORMAT = "gms"
GMS_V0_WORKER_CLASSES = frozenset(
    {
        "auto",
        "gpu_memory_service.integrations.vllm.worker.GMSWorker",
        "gpu_memory_service.integrations.vllm.worker:GMSWorker",
    }
)
GMS_V1_WORKER_CLASS = "gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker"


def has_gms_failover_load_profile(load_format: str, worker_cls: str) -> bool:
    """Return whether load format and worker class select GMS V0 or V1."""
    if worker_cls == GMS_V1_WORKER_CLASS:
        return load_format == "auto"
    return load_format == GMS_LOAD_FORMAT and worker_cls in GMS_V0_WORKER_CLASSES


__all__ = [
    "DisaggregationMode",
    "EmbeddingTransferMode",
    "GMS_LOAD_FORMAT",
    "GMS_V0_WORKER_CLASSES",
    "GMS_V1_WORKER_CLASS",
    "has_gms_failover_load_profile",
]
