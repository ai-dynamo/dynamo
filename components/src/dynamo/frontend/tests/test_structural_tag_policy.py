# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.frontend.structural_tag_policy import (
    effective_tool_strict,
    runtime_structural_tag_options,
    should_attempt_structural_tag,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("mode", "scope", "choice", "strict", "single", "expected"),
    [
        ("off", "always", "auto", False, False, False),
        ("on", "always", "auto", False, False, True),
        ("on", "auto", "auto", True, False, True),
        ("on", "auto", "auto", False, True, True),
        ("on", "auto", "auto", False, False, False),
        ("on", "auto", "required", False, False, True),
        ("on", "auto", "named", False, False, True),
        ("on", "always", "none", True, True, False),
    ],
)
def test_structural_tag_activation_policy(
    mode, scope, choice, strict, single, expected
):
    assert (
        should_attempt_structural_tag(
            mode=mode,
            scope=scope,
            tool_choice_kind=choice,
            has_tools=True,
            any_explicit_strict=strict,
            parallel_tool_calls_explicitly_false=single,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("request_strict", "schema_mode", "expected"),
    [
        (None, "auto", True),
        (True, "auto", True),
        (False, "auto", False),
        (False, "strict", True),
    ],
)
def test_effective_tool_strict(request_strict, schema_mode, expected):
    assert effective_tool_strict(request_strict, schema_mode) is expected


def test_runtime_structural_tag_options_are_conservative_when_absent():
    assert runtime_structural_tag_options(None) == ("off", "auto", "auto")
    assert runtime_structural_tag_options({}) == ("off", "auto", "auto")


def test_runtime_structural_tag_options_read_model_card_policy():
    assert runtime_structural_tag_options(
        {
            "structural_tag_mode": "on",
            "structural_tag_scope": "always",
            "structural_tag_schema": "strict",
        }
    ) == ("on", "always", "strict")
