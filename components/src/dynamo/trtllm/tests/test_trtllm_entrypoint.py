# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_trtllm_entrypoint_bootstraps_snapshot_before_main_import():
    entrypoint = Path(__file__).parents[1] / "__main__.py"
    source = entrypoint.read_text(encoding="utf-8")

    standby_import = (
        "from dynamo.common.snapshot.restore_context import "
        "maybe_run_restore_standby_mode"
    )
    standby_call = "maybe_run_restore_standby_mode()"
    snapshot_env_import = (
        "from dynamo.trtllm.snapshot import _configure_trtllm_snapshot_capture_env"
    )
    snapshot_env_call = "_configure_trtllm_snapshot_capture_env()"
    main_import = "from dynamo.trtllm.main import main"

    assert standby_import in source
    assert standby_call in source
    assert snapshot_env_import in source
    assert snapshot_env_call in source
    assert main_import in source
    assert source.index(standby_import) < source.index(main_import)
    assert source.index(standby_call) < source.index(main_import)
    assert source.index(snapshot_env_import) < source.index(main_import)
    assert source.index(snapshot_env_call) < source.index(main_import)
