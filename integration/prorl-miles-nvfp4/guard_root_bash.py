#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deny root Bash while allowing one exact Bash command in a subagent."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

EXPECTED_COMMAND = "/opt/m6-integration/apply_moto_fix.sh"


def hook_output(payload: dict[str, Any]) -> dict[str, Any]:
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    agent_id = payload.get("agent_id")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if tool_name != "Bash":
        reason = "M6 Bash guard was invoked for a non-Bash tool"
        decision = "deny"
    elif not agent_id:
        reason = "Root Bash is forbidden; delegate once to the general-purpose child"
        decision = "deny"
    elif command != EXPECTED_COMMAND:
        reason = f"Child Bash permits exactly {EXPECTED_COMMAND}"
        decision = "deny"
    else:
        reason = "Exact M6 child Bash command allowed"
        decision = "allow"
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def self_test() -> None:
    base = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": EXPECTED_COMMAND},
    }
    assert hook_output(base)["hookSpecificOutput"]["permissionDecision"] == "deny"
    child = {**base, "agent_id": "child-1"}
    assert hook_output(child)["hookSpecificOutput"]["permissionDecision"] == "allow"
    wrong = {**child, "tool_input": {"command": "git status"}}
    assert hook_output(wrong)["hookSpecificOutput"]["permissionDecision"] == "deny"
    print("__M6_ROOT_BASH_GUARD_SELF_TEST_PASS__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    payload = json.load(sys.stdin)
    print(json.dumps(hook_output(payload), separators=(",", ":")))


if __name__ == "__main__":
    main()
