# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static prompts and config writers for frontend coding-agent smoke tests."""

import json
from pathlib import Path

LIST_DIRECTORY_PROMPT = (
    "What files exist in the current working directory? Use your shell tool to run "
    "ls and report each filename verbatim from the output."
)


def write_codex_config(codex_home: Path, frontend_port: int) -> None:
    """Emit a minimal ~/.codex/config.toml pointing Codex at Dynamo."""
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        f"""
model_max_output_tokens = 4096

[model_providers.local]
name = "local-dynamo"
base_url = "http://localhost:{frontend_port}/v1"
wire_api = "responses"
env_key = "LOCAL_API_KEY"
        """.lstrip()
    )


def write_claude_subagent_config(cwd: Path, subagent_name: str) -> None:
    """Register a project-local Claude Code subagent for best-effort CI signal."""
    agents_dir = cwd / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{subagent_name}.md").write_text(
        f"""
---
name: {subagent_name}
description: Dynamo CI smoke subagent that lists the current directory.
tools: Bash
---

Use Bash to run `ls -1` in the current working directory. Return only the exact
filenames from the command output, one per line. Do not explain anything.
        """.lstrip()
    )


def claude_subagent_prompt(subagent_name: str, marker_filename: str) -> str:
    return (
        f"@agent-{subagent_name}\n"
        f"Invoke the {subagent_name} subagent exactly once. Do not use Bash yourself. "
        "The subagent must use Bash to run `ls -1` in the current working directory "
        f"and return the exact filenames. The output must include {marker_filename}."
    )


def write_opencode_config(
    cwd: Path,
    frontend_port: int,
    model: str,
    subtask_command: str,
) -> None:
    config = {
        "provider": {
            "dynamo": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Dynamo",
                "env": ["DYNAMO_API_KEY"],
                "models": {
                    model: {
                        "id": model,
                        "name": model,
                        "limit": {"context": 8192, "output": 512},
                        "cost": {"input": 0, "output": 0},
                    }
                },
                "options": {"baseURL": f"http://localhost:{frontend_port}/v1"},
            }
        },
        "permission": {"task": "allow"},
        "command": {
            subtask_command: {
                "template": "Reply with exactly OK.",
                "subtask": True,
            }
        },
    }

    project_config = cwd / ".opencode"
    project_config.mkdir(parents=True, exist_ok=True)
    (project_config / "opencode.jsonc").write_text(json.dumps(config, indent=2))
