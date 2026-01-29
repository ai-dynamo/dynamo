# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

from gpu_memory_service.integrations.common.patches import patch_empty_cache
from gpu_memory_service.integrations.common.utils import (
    finalize_gms_write,
    setup_meta_tensor_workaround,
)

__all__ = [
    "patch_empty_cache",
    "setup_meta_tensor_workaround",
    "finalize_gms_write",
]
