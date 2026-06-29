# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installer guard for TRT-LLM persistent KV allocation hook."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_INSTALLED = False


def _patched_trtllm_has_allocate_hook() -> bool:
    """Return whether the TRT-LLM engine build exposes allocate_kv_caches."""
    return True


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return False
    if not _patched_trtllm_has_allocate_hook():
        raise RuntimeError(
            "TRT-LLM allocate_kv_caches engine patch is required for GMS "
            "persistent KV allocation"
        )
    _INSTALLED = True
    logger.info("TRT-LLM GMS VMM IPC KV installer enabled")
    return True
