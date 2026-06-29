# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM engine bridge for gms_kv_ring."""

from gms_kv_ring.engines.trtllm.install_kv_ring import install_for_trtllm

__all__ = ["install_for_trtllm"]
