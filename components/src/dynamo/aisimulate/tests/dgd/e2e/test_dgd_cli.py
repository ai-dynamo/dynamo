# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dynamo.aisimulate.tests.dgd.cli import (
    assert_dgd_outputs,
    assert_recommend_succeeded,
    run_dgd_recommend,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]

_E2E_ROOT = Path(__file__).resolve().parent
_CASES = tuple(sorted((_E2E_ROOT / "configs").glob("*.yaml")))
assert _CASES, f"no Recommend-to-DGD cases found under {_E2E_ROOT / 'configs'}"
_DGD_GOLDEN_CASES = {"aggregated", "disaggregated"}


@pytest.mark.parametrize("config_path", _CASES, ids=lambda path: path.stem)
def test_dgd_cli_cases(config_path: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = tmp_path / config_path.stem
    command, result = run_dgd_recommend(config_path, output)
    assert_recommend_succeeded(command, result)
    golden_path = (
        _E2E_ROOT / "goldens" / config_path.name
        if config_path.stem in _DGD_GOLDEN_CASES
        else None
    )
    assert_dgd_outputs(
        case_name=config_path.stem,
        input_config=config,
        options=config["dgd"],
        output_dir=output,
        golden_path=golden_path,
    )
