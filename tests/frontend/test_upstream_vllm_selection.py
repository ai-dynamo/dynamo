# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep infrastructure errors outside the known-failure allowance."""

import xml.etree.ElementTree as ET

import pytest

from tests.frontend.test_upstream_vllm import UpstreamTestFailure, check_result

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


@pytest.mark.parametrize("tag", ["error", "skipped", "failure"])
def test_upstream_result_failure_types(tag):
    """Only test-body failures may match the expected-failure marker."""
    case = ET.fromstring(f'<testcase><{tag} message="detail"/></testcase>')
    error = UpstreamTestFailure if tag == "failure" else RuntimeError
    with pytest.raises(error, match="detail"):
        check_result(case)


def test_teardown_error_overrides_test_failure():
    """A known assertion failure must not hide broken server cleanup."""
    case = ET.fromstring(
        "<testcase><failure>known</failure><error>cleanup failed</error></testcase>"
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        check_result(case)


def test_upstream_pass():
    """A successful upstream case needs no exception or special handling."""
    check_result(ET.fromstring("<testcase/>"))


def test_discovery_uses_full_node_ids(tmp_path):
    """Keep same-named and parametrized tests distinct and label actual server use."""
    import json
    from types import SimpleNamespace

    from tests.utils.upstream_vllm.run import Report

    class Server:
        """Stand in for an adapted server fixture without starting a process."""

    nodes = ["a.py::test_same[x]", "b.py::test_same[x]"]
    items = [
        SimpleNamespace(nodeid=node, user_properties=[], funcargs=fixtures)
        for node, fixtures in zip(nodes, [{"server": Server()}, {}])
    ]
    report = Report(tmp_path, Server)
    report.pytest_collection_finish(SimpleNamespace(items=items))
    assert json.loads((tmp_path / "collection.json").read_text()) == nodes
    for item in items:
        report.pytest_runtest_call(item)
    assert dict(items[0].user_properties) == {
        "upstream_nodeid": nodes[0],
        "upstream_target": "dynamo",
    }
    assert dict(items[1].user_properties) == {
        "upstream_nodeid": nodes[1],
        "upstream_target": "vllm-only",
    }


def test_allowances_do_not_select_tests(tmp_path, monkeypatch):
    """Newly discovered tests run even when absent from the known-failure list."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tests.frontend import test_upstream_vllm as suite

    nodes = ["a.py::test_known", "b.py::test_new[x]"]
    (tmp_path / "collection.json").write_text(json.dumps(nodes))
    monkeypatch.setenv("VLLM_TESTS_DIR", str(tmp_path))
    monkeypatch.setattr(suite, "run_upstream", lambda *args: (tmp_path, 0))
    monkeypatch.setattr(suite, "ALLOW_FAILURES", {nodes[0]: "known"})
    metafunc = SimpleNamespace(fixturenames=["nodeid"], parametrize=Mock())
    suite.pytest_generate_tests(metafunc)
    parameters = metafunc.parametrize.call_args.args[1]
    assert [parameter.values[0] for parameter in parameters] == nodes
    assert parameters[0].marks[0].name == "xfail"
    assert not parameters[1].marks
