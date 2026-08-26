#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inject the exact root final-response contract after a validated child."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from prepare_swegym_task import CHILD_SUCCESS_HANDOFF, ROOT_SUCCESS_FINAL

POST_AGENT_CONTEXT = (
    "M6_POST_AGENT_FINALIZATION. Validation is complete. Do not reason about or "
    "verify the task and do not call any tool. Your next response must contain "
    f"exactly: {ROOT_SUCCESS_FINAL}"
)


def hook_output(payload: dict[str, Any]) -> dict[str, Any]:
    event_name = payload.get("hook_event_name")
    tool_name = payload.get("tool_name")
    agent_id = payload.get("agent_id")
    response_blob = json.dumps(payload.get("tool_response"), sort_keys=True)
    if event_name != "PostToolUse":
        reason = "M6 root finalizer requires a PostToolUse event"
    elif tool_name != "Agent":
        reason = "M6 root finalizer requires the Agent tool"
    elif agent_id:
        reason = "M6 root finalizer may run only for the root Agent result"
    elif CHILD_SUCCESS_HANDOFF not in response_blob:
        reason = "M6 child success handoff is missing from the Agent result"
    else:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": POST_AGENT_CONTEXT,
            }
        }
    return {"continue": False, "stopReason": reason}


def self_test() -> None:
    valid = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Agent",
        "tool_response": {"content": CHILD_SUCCESS_HANDOFF},
    }
    output = hook_output(valid)
    assert output["hookSpecificOutput"]["additionalContext"] == POST_AGENT_CONTEXT
    assert hook_output({**valid, "agent_id": "child-1"})["continue"] is False
    assert hook_output({**valid, "tool_response": {"content": "failed"}})[
        "continue"
    ] is False
    assert hook_output({**valid, "tool_name": "Bash"})["continue"] is False
    print("__M6_ROOT_FINALIZER_SELF_TEST_PASS__")


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
