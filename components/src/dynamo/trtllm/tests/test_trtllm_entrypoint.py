# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_trtllm_entrypoint_checks_snapshot_standby_before_main_import():
    entrypoint = Path(__file__).parents[1] / "__main__.py"
    source = entrypoint.read_text(encoding="utf-8")

    standby_import = (
        "from dynamo.common.snapshot.restore_context import "
        "maybe_run_restore_standby_mode"
    )
    standby_call = "maybe_run_restore_standby_mode()"
    main_import = "from dynamo.trtllm.main import main"

    assert standby_import in source
    assert standby_call in source
    assert main_import in source
    assert source.index(standby_import) < source.index(main_import)
    assert source.index(standby_call) < source.index(main_import)
