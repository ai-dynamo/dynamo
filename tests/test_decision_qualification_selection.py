# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification collection must require an explicitly provisioned profile."""

import ast
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


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
