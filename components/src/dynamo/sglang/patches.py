# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_snapshot_patches_applied = False


def _configure_snapshot_dist_init() -> None:
    pod_uid = os.environ.get("POD_UID", "checkpoint")
    store_path = Path("/tmp") / f"dynamo-sglang-dist-init-{pod_uid}"
    try:
        store_path.unlink()
    except FileNotFoundError:
        pass

    override = f"file://{store_path}"
    previous = os.environ.get("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE")
    if previous and previous != override:
        logger.warning(
            "Overriding SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=%r with %r for checkpoint mode "
            "to avoid TCPStore listeners that CRIU rejects under tcp-loopback-only",
            previous,
            override,
        )
    os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] = override
    logger.info(
        "Using file-based torch distributed init for checkpoint mode: %s",
        store_path,
    )


def run_scheduler_process_with_snapshot_patches(*args, **kwargs):
    apply_snapshot_patches()

    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


run_scheduler_process_with_snapshot_patches._dynamo_snapshot_patch = True


def apply_snapshot_patches() -> None:
    global _snapshot_patches_applied

    if _snapshot_patches_applied:
        return

    _configure_snapshot_dist_init()

    from sglang.srt.distributed import parallel_state as parallel_state_module
    from sglang.srt.distributed.device_communicators import pynccl as pynccl_module
    from sglang.srt.entrypoints import engine as engine_module

    original_init = pynccl_module.PyNcclCommunicator.__init__
    original_init_group = parallel_state_module.init_model_parallel_group
    current_scheduler_entrypoint = engine_module.Engine.run_scheduler_process_func
    if (
        getattr(original_init, "_dynamo_snapshot_patch", False)
        and getattr(original_init_group, "_dynamo_snapshot_patch", False)
        and getattr(current_scheduler_entrypoint, "_dynamo_snapshot_patch", False)
    ):
        _snapshot_patches_applied = True
        return

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self, "available", False) and getattr(self, "world_size", 1) > 1:
            self.disabled = False

    def patched_init_model_parallel_group(*args, **kwargs):
        group = original_init_group(*args, **kwargs)
        pynccl_comm = getattr(group, "pynccl_comm", None)
        if pynccl_comm is not None and getattr(group, "world_size", 1) > 1:
            pynccl_comm.disabled = False
        return group

    patched_init._dynamo_snapshot_patch = True
    patched_init_model_parallel_group._dynamo_snapshot_patch = True
    pynccl_module.PyNcclCommunicator.__init__ = patched_init
    parallel_state_module.init_model_parallel_group = patched_init_model_parallel_group
    engine_module.Engine.run_scheduler_process_func = staticmethod(
        run_scheduler_process_with_snapshot_patches
    )
    _snapshot_patches_applied = True
    logger.info(
        "Forced SGLang PyNccl communicators on in checkpoint mode, including spawned schedulers"
    )
