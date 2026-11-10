# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental CPU replay, selected explicitly outside required CI lanes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.frontend.trace_replay.runner import RunOptions, run

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skip_in_nightly,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.skipif(
        os.environ.get("DYN_RUN_FRONTEND_REPLAY") != "1",
        reason="experimental replay is opt-in: set DYN_RUN_FRONTEND_REPLAY=1",
    ),
]


@pytest.mark.timeout(300)
def test_deepseek_v41_five_captures_through_real_frontend(tmp_path: Path) -> None:
    configured = os.environ.get("DYN_REPLAY_MODEL_DIR")
    if configured is None:
        pytest.fail(
            "set DYN_REPLAY_MODEL_DIR to the hash-verified native model bundle; see trace_replay/README.md"
        )
    corpus = Path(__file__).parent / "trace_replay" / "fixtures" / "deepseek_v41"
    code, summary = run(
        RunOptions(
            corpus=corpus, model_dir=Path(configured), output_dir=tmp_path / "replay"
        )
    )
    assert summary["coverage"]["selected_cases"] == 5
    assert summary["coverage"]["executions_completed"] == 20
    assert code == 0, summary
    assert summary["counts"]["pass"] == 20
