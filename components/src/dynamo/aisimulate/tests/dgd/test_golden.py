# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from golden import assert_yaml_golden, repository_root

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]

_EXPECTED = """\
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: expected
"""
_CHANGED = _EXPECTED.replace("expected", "changed")


def test_repository_root_prefers_github_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "components/src/dynamo/aisimulate").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(workspace))

    assert repository_root() == workspace.resolve()


def test_yaml_golden_accepts_parsed_yaml_equivalence(tmp_path: Path) -> None:
    golden = tmp_path / "expected.yaml"
    golden.write_text(_EXPECTED)

    assert_yaml_golden(golden, _EXPECTED.replace("name: expected", "name: expected\n"))

    assert not golden.with_name("expected.new.yaml").exists()


@pytest.mark.parametrize("existing", [True, False], ids=["changed", "missing"])
def test_yaml_golden_writes_proposal_and_reports_diff(
    tmp_path: Path, existing: bool
) -> None:
    golden = tmp_path / "expected.yaml"
    if existing:
        golden.write_text(_EXPECTED)

    with pytest.raises(pytest.fail.Exception) as failure:
        assert_yaml_golden(golden, _CHANGED)

    proposal = tmp_path / "expected.new.yaml"
    assert proposal.read_text() == _CHANGED
    if existing:
        assert golden.read_text() == _EXPECTED
    else:
        assert not golden.exists()
    message = str(failure.value)
    assert "expected.yaml" in message
    assert "expected.new.yaml" in message
    assert "mv" in message
    if existing:
        assert "-  name: expected" in message
    else:
        assert "--- /dev/null" in message
    assert "+  name: changed" in message
