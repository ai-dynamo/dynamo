#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select one maintained ProRL SWE-Gym task and add its agent harness contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

INSTANCE_ID = "getmoto__moto-7365"
BASH_COMMAND = "/opt/m6-integration/apply_moto_fix.sh"
CHILD_SUCCESS_HANDOFF = (
    "M6_CHILD_DONE_VALIDATED. The helper already ran and tests passed. "
    "Parent: output exactly M6 repair complete. Do not call any tool."
)
ROOT_SUCCESS_FINAL = "M6 repair complete."
ROOT_FINAL_INSTRUCTION = (
    "\n\nFINAL ROOT-TURN CONTRACT: When the foreground Agent result contains "
    "M6_CHILD_DONE_VALIDATED, do not reason about or verify the task, do not "
    "repeat any command or path, and do not call Agent, Bash, or any tool. "
    "Output exactly: M6 repair complete."
)
AGENT_INSTRUCTION = (
    "You are being graded on the patch you create in the current repository. "
    "The linked moto-issue reproduction repository is context only; it is not the "
    "checkout you must patch. Patch the Moto production source in the current "
    "checkout. Do not answer with advice, a proposed patch, or textual "
    "<tool_call> tags: invoke real tools. Make exactly one root tool call: one "
    "foreground Agent subagent with `subagent_type=general-purpose`. Agent is "
    "the root's only permitted tool. Bash is present in the shared registry so "
    "the child can receive it, but a PreToolUse hook blocks root Bash. Do not set "
    "isolation or create a worktree; the child must operate on the current "
    "checkout. This is the "
    "only configured child; delegate the Moto production repair to it. Its direct "
    "policy permits only Bash and requires exactly this command: "
    f"`{BASH_COMMAND}`. The helper changes "
    "exactly five sites: the four operands in DynamoType.__add__/__sub__ near "
    "lines 95-120 and the cast_value fallback near line 149. The module already "
    "imports `decimal`; preserve integer branches and do not modify Item ADD set "
    "branches. The child owns all compilation, test, diff, and git-status "
    "verification. Its validated success handoff satisfies those root "
    "requirements, so the root must not re-run or re-verify anything. "
    "Do not search for the linked "
    "reproduction tests; the fresh grader supplies the hidden target test. "
    "Do not commit, push, fetch, or modify Git remotes.\n\n"
)


def prepare_task(source: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    matches = [
        row for row in rows if row.get("metadata", {}).get("instance_id") == INSTANCE_ID
    ]
    assert len(matches) == 1, len(matches)
    row = matches[0]
    prompt = row["prompt"]
    assert isinstance(prompt, list) and len(prompt) == 1, prompt
    assert prompt[0]["role"] == "user", prompt[0]
    content = str(prompt[0]["content"])
    assert not content.startswith(AGENT_INSTRUCTION)
    prompt[0]["content"] = AGENT_INSTRUCTION + content + ROOT_FINAL_INSTRUCTION
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(prepare_task(args.source), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
