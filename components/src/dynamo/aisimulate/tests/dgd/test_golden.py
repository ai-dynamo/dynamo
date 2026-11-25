# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from golden import assert_structured_goldens, compare_structured_golden, repository_root

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


@pytest.mark.parametrize(
    ("suffix", "expected", "actual"),
    [
        (".yaml", _EXPECTED, _EXPECTED.replace("name: expected", "name: expected\n")),
        (".json", '{"value": 1, "other": 2}\n', '{"other": 2, "value": 1}\n'),
    ],
)
def test_structured_golden_accepts_parsed_equivalence(
    tmp_path: Path, suffix: str, expected: str, actual: str
) -> None:
    golden = tmp_path / f"expected{suffix}"
    output = tmp_path / f"actual{suffix}"
    golden.write_text(expected)
    output.write_text(actual)

    assert compare_structured_golden(golden, output) is None
    assert not golden.with_name(f"expected.new{suffix}").exists()


@pytest.mark.parametrize("existing", [True, False], ids=["changed", "missing"])
def test_structured_golden_writes_proposal_and_reports_diff(
    tmp_path: Path, existing: bool
) -> None:
    golden = tmp_path / "expected.yaml"
    actual = tmp_path / "actual.yaml"
    if existing:
        golden.write_text(_EXPECTED)
    actual.write_text(_CHANGED)

    message = compare_structured_golden(golden, actual)

    assert message is not None
    proposal = tmp_path / "expected.new.yaml"
    assert proposal.read_text() == _CHANGED
    if existing:
        assert golden.read_text() == _EXPECTED
    else:
        assert not golden.exists()
    assert "expected.yaml" in message
    assert "expected.new.yaml" in message
    assert "mv" in message
    if existing:
        assert "-  name: expected" in message
    else:
        assert "--- /dev/null" in message
    assert "+  name: changed" in message


def test_structured_goldens_report_every_mismatch(tmp_path: Path) -> None:
    pairs = []
    for name in ("first", "second"):
        golden = tmp_path / f"{name}.json"
        actual = tmp_path / f"{name}.actual.json"
        golden.write_text('{"value": 1}\n')
        actual.write_text('{"value": 2}\n')
        pairs.append((golden, actual))

    with pytest.raises(pytest.fail.Exception) as failure:
        assert_structured_goldens(pairs)

    message = str(failure.value)
    assert "first.new.json" in message
    assert "second.new.json" in message
