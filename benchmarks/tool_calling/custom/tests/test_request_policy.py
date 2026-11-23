# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request_policy import effective_tool_policy  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("mode", ("auto", "required"))
@pytest.mark.parametrize("endpoint", ("chat", "responses"))
def test_allowed_subset_preserves_mode_and_single_call_policy(mode, endpoint):
    selection = {"mode": mode, "tools": [{"type": "function", "name": "lookup"}]}
    if endpoint == "chat":
        selection["tools"] = [{"type": "function", "function": {"name": "lookup"}}]
        choice = {"type": "allowed_tools", "allowed_tools": selection}
    else:
        choice = {"type": "allowed_tools", **selection}
    payload = {"tool_choice": choice, "parallel_tool_calls": False}

    policy = effective_tool_policy(payload)

    assert policy.mode == mode
    assert policy.requires_call is (mode == "required")
    assert policy.allowed_names == {"lookup"}
    assert policy.max_calls == 1


@pytest.mark.parametrize(
    "choice,expected_mode,expected_names,required",
    (
        (None, "auto", None, False),
        ("auto", "auto", None, False),
        ("required", "required", None, True),
        ("none", "none", frozenset(), False),
        (
            {"type": "function", "function": {"name": "get_today"}},
            "named",
            {"get_today"},
            True,
        ),
        (
            {"type": "function", "name": "get_today"},
            "named",
            {"get_today"},
            True,
        ),
    ),
)
def test_choices_have_explicit_call_requirements(
    choice, expected_mode, expected_names, required
):
    policy = effective_tool_policy({"tool_choice": choice})
    assert policy.mode == expected_mode
    assert policy.allowed_names == expected_names
    assert policy.requires_call is required
    assert policy.max_calls is None


def test_policy_reads_the_final_payload_without_mutating_it():
    payload = {"tool_choice": "required", "parallel_tool_calls": False}
    payload.update({"tool_choice": "auto", "parallel_tool_calls": True})
    policy = effective_tool_policy(payload)
    assert policy.requires_call is False
    assert policy.max_calls is None
    assert payload == {"tool_choice": "auto", "parallel_tool_calls": True}


@pytest.mark.parametrize(
    "choice",
    (
        "surprise",
        {"type": "function", "function": {}},
        {"type": "allowed_tools", "mode": "none", "tools": []},
        {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "custom"}]},
    ),
)
def test_unsupported_policy_shapes_do_not_silently_become_auto(choice):
    with pytest.raises(ValueError):
        effective_tool_policy({"tool_choice": choice})


@pytest.mark.parametrize("endpoint", ("chat", "responses"))
def test_named_choice_keeps_namespace_in_function_identity(endpoint):
    function = {"name": "lookup", "namespace": "crm"}
    choice = (
        {"type": "function", "function": function}
        if endpoint == "chat"
        else {"type": "function", **function}
    )
    policy = effective_tool_policy({"tool_choice": choice})

    assert policy.allowed_names == {"lookup"}
    assert policy.allowed_identities == {("crm", "lookup")}
    assert policy.allows("lookup", "crm")
    assert not policy.allows("lookup", "inventory")
    assert not policy.allows("lookup")


@pytest.mark.parametrize("endpoint", ("chat", "responses"))
def test_allowed_subset_does_not_admit_same_leaf_in_another_namespace(endpoint):
    function = {"name": "lookup", "namespace": "crm"}
    if endpoint == "chat":
        selection = {
            "mode": "auto",
            "tools": [{"type": "function", "function": function}],
        }
        choice = {"type": "allowed_tools", "allowed_tools": selection}
    else:
        choice = {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", **function}],
        }
    policy = effective_tool_policy({"tool_choice": choice})

    assert policy.allowed_names == {"lookup"}
    assert policy.allowed_identities == {("crm", "lookup")}
    assert policy.allows("lookup", "crm")
    assert not policy.allows("lookup", "inventory")
    assert not policy.allows("lookup")
    assert not policy.requires_call
