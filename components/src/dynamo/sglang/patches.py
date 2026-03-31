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


def apply_snapshot_patches() -> None:
    global _snapshot_patches_applied

    if _snapshot_patches_applied:
        return

    _configure_snapshot_dist_init()

    from sglang.srt.distributed.device_communicators import pynccl as pynccl_module

    original_init = pynccl_module.PyNcclCommunicator.__init__
    if getattr(original_init, "_dynamo_snapshot_patch", False):
        _snapshot_patches_applied = True
        return

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self, "available", False) and getattr(self, "world_size", 1) > 1:
            self.disabled = False

    patched_init._dynamo_snapshot_patch = True
    pynccl_module.PyNcclCommunicator.__init__ = patched_init
    _snapshot_patches_applied = True
    logger.info("Enabled SGLang PyNccl communicator by default for checkpoint mode")
