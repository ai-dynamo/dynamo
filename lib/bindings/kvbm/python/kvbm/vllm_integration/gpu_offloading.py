# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in retained-GPU cache with isolated CUDA IPC or MNNVL delivery.

The primary cache uses vLLM's existing ref-counted cache policy and scheduling
contract, but its storage is a CUDA allocation, never a host mmap. Local stores
and loads are D2D copies; the remote leg is a GPU-only KVCR instance. This raw
layout requires an explicit identical model/layout compatibility domain.
"""

from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec


class GPURetainedLoadStoreSpec(CPULoadStoreSpec):
    @staticmethod
    def medium():
        return "GPU_CACHE"
