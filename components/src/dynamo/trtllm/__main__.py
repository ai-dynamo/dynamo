# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # In restore mode (DYN_SNAPSHOT_RESTORE_STANDBY=1), before importing TRT-LLM,
    # write selected restore-time env vars to snapshot-control/restore-context.json
    # and exec `sleep infinity` without initializing CUDA or backend/runtime state.
    maybe_run_restore_standby_mode()

    from dynamo.common.snapshot.lifecycle import is_snapshot_enabled

    if is_snapshot_enabled():
        from dynamo.trtllm.snapshot import _configure_trtllm_snapshot_capture_env

        # TRT-LLM caches this setting while importing its distributed ops, so
        # configure it before importing main and the TRT-LLM worker modules.
        _configure_trtllm_snapshot_capture_env()

    from dynamo.trtllm.main import main

    main()
