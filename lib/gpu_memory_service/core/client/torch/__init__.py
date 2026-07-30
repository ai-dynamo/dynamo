# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch storage rebinding primitives used by GMS integrations."""

from gpu_memory_service.core.client.torch.storage_rebinding import (
    clone_storage_spans_and_rebind_tensors,
    tensor_storage_byte_bounds,
)

__all__ = [
    "clone_storage_spans_and_rebind_tensors",
    "tensor_storage_byte_bounds",
]
