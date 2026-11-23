# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest
import yaml
from comparison import run_cases
from golden import repository_root

from dynamo.aisimulate.tests.dgd.cli import (
    assert_dgd_outputs,
    assert_recommend_succeeded,
    run_dgd_recommend,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]

_SUITE = (
    repository_root()
    / "components/src/dynamo/aisimulate/tests/dgd/comparison/testsuite-issue-8469.yaml"
)
_GOLDEN_ROOT = run_cases.default_suite_output_root(_SUITE)
_CASES = tuple(run_cases.load_suite(_SUITE))
assert _CASES, f"no comparison cases found in {_SUITE}"


def _write_cli_config(case: run_cases.ComparisonCase, path: Path) -> dict:
    config = run_cases.recommendation_config(case)
    options = case.generation_options
    dgd = {
        "name": run_cases._dgd_name(case.name),
        "output_file": "deployment.yaml",
        "runtime_image": options.runtime_image,
        "num_gpus_per_node": options.num_gpus_per_node,
    }
    if options.runtime_version_override is not None:
        dgd["runtime_version_override"] = options.runtime_version_override
    config["dgd"] = dgd
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return dgd


@pytest.mark.parametrize(
    "entry", _CASES, ids=lambda entry: f"{entry.hardware}-{entry.case}"
)
def test_model_corpus_dgd_goldens(entry: run_cases.SuiteEntry, tmp_path: Path) -> None:
    hardware = run_cases.load_hardware(entry.hardware)
    case = run_cases.load_case(entry.case, hardware, output_root=tmp_path)
    config_path = tmp_path / entry.hardware / entry.case / "config.yaml"
    config_path.parent.mkdir(parents=True)
    options = _write_cli_config(case, config_path)
    output = tmp_path / entry.hardware / entry.case / "output"
    command, result = run_dgd_recommend(
        config_path,
        output,
        environment={
            "HF_HOME": str(case.cache_dir / "huggingface"),
            "MPLCONFIGDIR": str(case.cache_dir / "matplotlib"),
            "XDG_CACHE_HOME": str(case.cache_dir),
        },
    )
    expected_failure = entry.exception_for("recommend", "aic")
    if expected_failure is not None:
        assert expected_failure.status == "broken"
        assert expected_failure.match is not None
        assert result.returncode != 0, (
            f"{entry.hardware}/{entry.case} unexpectedly passed: "
            f"{expected_failure.describe()}"
        )
        diagnostic = f"{result.stdout}\n{result.stderr}"
        assert re.search(expected_failure.match, diagnostic), (
            f"{entry.hardware}/{entry.case} failed with an unexpected diagnostic\n"
            f"expected: {expected_failure.match}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return

    assert_recommend_succeeded(command, result)
    golden_dir = _GOLDEN_ROOT / entry.hardware / entry.case
    assert_dgd_outputs(
        case_name=entry.case,
        input_config=config,
        options=options,
        output_dir=output,
        golden_path=golden_dir / "dgd-sweeper-aic.yaml",
    )
