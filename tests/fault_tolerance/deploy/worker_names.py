# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend worker service names used by fault-tolerance deployment tests."""

WORKER_MAP = {
    "vllm": {
        "decode": "decode",
        "decode_agg": "worker",
        "prefill": "prefill",
    },
    "sglang": {
        "decode": "decode",
        "prefill": "prefill",
    },
    "trtllm": {
        "decode": "decode",
        "decode_agg": "TRTLLMWorker",
        "prefill": "prefill",
    },
}


def get_worker_service_name(backend: str, deploy_type: str) -> str:
    """Return the primary worker service name for a deployment topology."""
    workers = WORKER_MAP[backend]
    if deploy_type == "agg" and "decode_agg" in workers:
        return workers["decode_agg"]
    return workers["decode"]
