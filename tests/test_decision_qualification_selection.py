# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification collection must require an explicitly provisioned profile."""

import ast
import re
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.mark.parametrize(
    "release",
    [
        "0.5.19",
        "0.5.19+cu128",
        "0.5.22",
        "0.5.22+cu130",
        "0.5.23",
        "0.5.23+cu130",
    ],
)
def test_native_parity_fixture_uses_version_specific_prefill_graph_flag(release):
    source = Path(__file__).parent / "serve" / "test_systemone_sglang.py"
    tree = ast.parse(source.read_text())
    fixture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "native_sglang_server"
    )
    command = next(
        keyword.value
        for node in ast.walk(fixture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ManagedProcess"
        for keyword in node.keywords
        if keyword.arg == "command"
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_native_prefill_graph_args"
    ]
    namespace = {
        "sys": sys,
        "QWEN": "model",
        "ports": [12345],
        "version": lambda _: release,
    }
    exec(
        compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    arguments = eval(compile(ast.Expression(command), str(source), "eval"), namespace)
    if release.startswith(("0.5.22", "0.5.23")):
        index = arguments.index("--cuda-graph-backend-prefill")
        assert arguments[index + 1] == "disabled"
        assert "--disable-piecewise-cuda-graph" not in arguments
    else:
        assert "--disable-piecewise-cuda-graph" in arguments
        assert "--cuda-graph-backend-prefill" not in arguments


@pytest.mark.parametrize(
    "release,is_target",
    [
        ("0.5.19", True),
        ("0.5.22", True),
        ("0.5.19+cu128", True),
        ("0.5.22+cu130", True),
        ("0.5.19+debug", False),
        ("0.5.22+debug", False),
        ("0.5.22+cu130.debug", False),
        ("0.5.22rc1", False),
        ("0.5.22.dev1", False),
        ("0.5.22.post1", False),
        ("0.5.23", True),
        ("0.5.23+cu130", True),
        ("0.5.23+debug", False),
        ("0.5.24", False),
    ],
)
def test_native_parity_fixture_accepts_only_target_release_versions(release, is_target):
    source = Path(__file__).parent / "serve" / "test_systemone_sglang.py"
    fixture = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "systemone_sglang_server"
    )
    # Execute the fixture's real version assertion without starting any services.
    assert isinstance(fixture.body[0], ast.Assert)
    gate = compile(
        ast.Module(body=[fixture.body[0]], type_ignores=[]), str(source), "exec"
    )
    namespace = {"version": lambda _: release, "re": re}
    if is_target:
        exec(gate, namespace)
    else:
        with pytest.raises(AssertionError):
            exec(gate, namespace)


@pytest.mark.parametrize(
    "environment,expected",
    [
        ({}, (1, 2)),
        ({"DYNAMO_DECISION_RUN_PINNED_SGLANG": "0"}, (1, 2)),
        ({"DYNAMO_DECISION_TEST_URL": "   "}, (1, 2)),
        ({"DYNAMO_DECISION_RUN_PINNED_SGLANG": "1"}, (2, 1)),
        ({"DYNAMO_DECISION_TEST_URL": "http://localhost"}, (2, 1)),
        (
            {
                "DYNAMO_DECISION_RUN_PINNED_SGLANG": "1",
                "DYNAMO_DECISION_TEST_URL": "http://localhost",
            },
            (3, 0),
        ),
    ],
)
@pytest.mark.parametrize("explicit_files", [False, True])
def test_decision_qualification_collection(
    pytester, monkeypatch, environment, expected, explicit_files
):
    source = Path(__file__).parent / "serve" / "conftest.py"
    hook = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pytest_collection_modifyitems"
    )
    pytester.makeconftest("import os\nfrom pathlib import Path\n" + ast.unparse(hook))
    pytester.makepyfile(
        test_systemone_sglang="def test_pinned(): pass",
        test_decisions_external="def test_external(): pass",
        test_other="def test_other(): pass",
    )
    for variable in (
        "DYNAMO_DECISION_RUN_PINNED_SGLANG",
        "DYNAMO_DECISION_TEST_URL",
    ):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    targets = sorted(pytester.path.glob("test_*.py")) if explicit_files else []
    result = pytester.runpytest("-o", "addopts=", *targets)
    result.assert_outcomes(passed=expected[0], deselected=expected[1], skipped=0)
    if environment.get("DYNAMO_DECISION_RUN_PINNED_SGLANG") != "1":
        result.stdout.fnmatch_lines(["*Set DYNAMO_DECISION_RUN_PINNED_SGLANG=1*"])
    if not environment.get("DYNAMO_DECISION_TEST_URL", "").strip():
        result.stdout.fnmatch_lines(["*Set DYNAMO_DECISION_TEST_URL*"])


def test_native_parity_timeout_covers_both_worker_startups():
    source = Path(__file__).parent / "serve" / "test_systemone_sglang.py"
    tree = ast.parse(source.read_text())
    timeouts = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "timeout"
    ]
    # Two sequential 360s starts and six 60s HTTP requests need over 1080s.
    assert timeouts and min(timeouts) > 2 * 360 + 6 * 60


def test_external_native_parity_timeout_covers_all_http_requests():
    source = Path(__file__).parent / "serve" / "test_decisions_external.py"
    test = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_external_native_sglang_numeric_parity"
    )
    timeouts = [
        node.args[0].value
        for node in test.decorator_list
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "timeout"
    ]
    assert timeouts and min(timeouts) > 6 * 60
