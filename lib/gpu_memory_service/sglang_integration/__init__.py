# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang integration for GPU Memory Service.

This module provides SGLang integration for weight sharing via GPU Memory Service:
- Custom model loader that loads weights via GMS
- torch_memory_saver integration for hybrid weights/KV cache management
- Utility patches for empty_cache and memory accounting

Usage:
    Set --load-format gpu_memory_service when launching SGLang.
"""
