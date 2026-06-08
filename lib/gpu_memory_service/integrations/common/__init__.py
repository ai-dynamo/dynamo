# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities shared across GMS integrations."""

GMS_TAGS = ("weights", "kv_cache")


def patch_empty_cache() -> None:
    from gpu_memory_service.integrations.common.patches import (
        patch_empty_cache as _patch_empty_cache,
    )

    _patch_empty_cache()


def setup_meta_tensor_workaround() -> None:
    from gpu_memory_service.integrations.common.utils import (
        setup_meta_tensor_workaround as _setup_meta_tensor_workaround,
    )

    _setup_meta_tensor_workaround()


def finalize_gms_write(*args, **kwargs):
    from gpu_memory_service.integrations.common.utils import (
        finalize_gms_write as _finalize_gms_write,
    )

    return _finalize_gms_write(*args, **kwargs)


def __getattr__(name: str):
    if name == "GMSCommittedMemoryStats":
        from gpu_memory_service.integrations.common.utils import GMSCommittedMemoryStats

        return GMSCommittedMemoryStats
    raise AttributeError(name)


__all__ = [
    "GMS_TAGS",
    "GMSCommittedMemoryStats",
    "patch_empty_cache",
    "setup_meta_tensor_workaround",
    "finalize_gms_write",
]
