# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific patches for GPU Memory Service integration.

- patch_model_runner: Fixes memory accounting with pre-loaded weights
- patch_static_state_for_gms: No-ops named-buffer export/import (GMS preserves them)
"""

from __future__ import annotations

import logging

from gpu_memory_service.integrations.sglang.memory_saver import (
    get_gms_memory_saver_impl,
)

logger = logging.getLogger(__name__)

_model_runner_patched = False
_static_state_patched = False


def patch_model_runner() -> None:
    """Patch SGLang's ModelRunner to size KV cache with GMS-resident weights.

    SGLang's KV sizing formula reserves dynamic headroom from a free-memory
    snapshot taken before its own model load. In GMS read mode, the committed
    weight handles already exist in the GMS server before that snapshot, so the
    snapshot is lower by those weights. Add just those preloaded weight bytes
    back to the baseline. Do not adjust write mode: weights are loaded after
    the snapshot there, so upstream's formula already subtracts them correctly.
    """
    global _model_runner_patched

    if _model_runner_patched:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        logger.warning("[GMS] Could not import ModelRunner, skipping patch")
        return

    if hasattr(ModelRunner, "_gms_patched"):
        return

    original_alloc_memory_pool = ModelRunner.alloc_memory_pool

    def patched_alloc_memory_pool(self, *args, **kwargs):
        impl = get_gms_memory_saver_impl()
        if (
            impl is not None
            and impl.preloaded_weights_bytes > 0
            and not self.__dict__.get("_gms_memory_baseline_adjusted", False)
        ):
            preloaded_weights_gib = impl.preloaded_weights_bytes / (1 << 30)
            old_value = self.pre_model_load_memory
            self.pre_model_load_memory += preloaded_weights_gib
            self._gms_memory_baseline_adjusted = True
            logger.info(
                "[GMS] Adjusted pre_model_load_memory for preloaded weights: "
                "%.2f GiB + %.2f GiB = %.2f GiB",
                old_value,
                preloaded_weights_gib,
                self.pre_model_load_memory,
            )

        return original_alloc_memory_pool(self, *args, **kwargs)

    ModelRunner.alloc_memory_pool = patched_alloc_memory_pool
    ModelRunner._gms_patched = True
    _model_runner_patched = True
    logger.info("[GMS] Patched ModelRunner.alloc_memory_pool")


def patch_static_state_for_gms() -> None:
    """No-op SGLang's _export/_import_static_state when using GMS.

    SGLang's release_memory_occupation clones every named buffer via
    buffer.detach().clone() through the default CUDA allocator, then restores
    them during resume_memory_occupation.
    This patch must run inside the scheduler child process (which uses
    multiprocessing spawn).  It is triggered by the GMSModelLoader import
    in model_loader.py, which executes at module level in the child.
    """
    import os

    global _static_state_patched
    logger.info(
        "[GMS] patch_static_state_for_gms called (pid=%d, already_patched=%s)",
        os.getpid(),
        _static_state_patched,
    )
    if _static_state_patched:
        return

    try:
        from sglang.srt.managers import scheduler_update_weights_mixin as _mixin

        def _export_noop(model):
            """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
            return dict(buffers=[])

        def _import_noop(model, static_params):
            """NO-OP: GMS preserves buffers via VA-stable unmap/remap."""
            pass

        _mixin._export_static_state = _export_noop
        _mixin._import_static_state = _import_noop
        _static_state_patched = True
        logger.info(
            "[GMS] Patched _export/_import_static_state -> no-op (pid=%d)",
            os.getpid(),
        )
    except Exception:
        logger.warning(
            "[GMS] Could not patch scheduler_update_weights_mixin: ",
            exc_info=True,
        )
