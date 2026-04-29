# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_aggregate_launch_script_keeps_tokenizer_init_enabled():
    """SGLang grammar generation requires tokenizer initialization."""
    agg_launch_script = (
        REPO_ROOT / "examples" / "backends" / "sglang" / "launch" / "agg.sh"
    )

    assert "--skip-tokenizer-init" not in agg_launch_script.read_text(encoding="utf-8")
