# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from dynamo.aisimulate.tests.dgd.comparison import run_cases
from dynamo.aisimulate.tests.dgd.golden import assert_yaml_golden, repository_root

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
_CASES = tuple(entry for entry in run_cases.load_suite(_SUITE) if entry.automatic)
assert _CASES, f"no automatic comparison cases found in {_SUITE}"


@pytest.mark.parametrize(
    "entry", _CASES, ids=lambda entry: f"{entry.hardware}-{entry.case}"
)
def test_model_corpus_dgd_goldens(entry: run_cases.SuiteEntry, tmp_path: Path) -> None:
    hardware = run_cases.load_hardware(entry.hardware)
    case = run_cases.load_case(entry.case, hardware, output_root=tmp_path)

    failures = run_cases.run_case(case, entry)

    assert failures == [], "\n".join(failures)
    golden_dir = _GOLDEN_ROOT / entry.hardware / entry.case
    golden_paths = {path.name: path for path in sorted(golden_dir.glob("dgd-*.yaml"))}
    assert golden_paths, f"{golden_dir} has no checked-in DGD goldens"
    actual_paths = {
        path.name: path for path in sorted(case.generated_dir.glob("dgd-*.yaml"))
    }
    missing = sorted(golden_paths.keys() - actual_paths.keys())
    assert not missing, (
        f"{entry.hardware}/{entry.case} did not generate checked-in variant(s): "
        + ", ".join(missing)
    )
    for name, actual_path in actual_paths.items():
        assert_yaml_golden(golden_dir / name, actual_path.read_text())
