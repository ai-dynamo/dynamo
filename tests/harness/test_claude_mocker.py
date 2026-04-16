# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real `claude` CLI driving Dynamo's `/v1/messages` endpoint over the mocker.

These tests invoke the actual `claude` binary (Claude Code) in `--print`
non-interactive mode with `ANTHROPIC_BASE_URL` pointed at the harness
frontend. The mocker emits deterministic fake tokens, so the assertions
focus on exit code, non-empty output, and JSON shape when `--output-format
json` is used — not on specific token content.

`tool_call` exercises the tool-definition translation path: Claude is given
`Bash` as an allowed tool, so the outgoing request carries a `tools` block
which the Anthropic-to-ChatCompletion converter has to handle. The mocker
will not produce a `tool_use` response, which is fine — the assertion is
that the request reaches the worker and a valid reply comes back.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid

import pytest

from tests.harness.conftest import cli_env, has_cli

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e_agent,
    pytest.mark.gpu_0,
    pytest.mark.skipif(not has_cli("claude"), reason="`claude` CLI not on PATH"),
]

# `claude` spins up a Node process; give it a generous but bounded budget.
CLAUDE_TIMEOUT = 120


def _run_claude(
    service,
    prompt: str,
    *,
    extra_args: list[str] | None = None,
    session_id: str | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        "claude",
        "--bare",
        "--print",
        "--output-format",
        "json",
        "--model",
        service.model,
        "--max-budget-usd",
        "1",
    ]
    if session_id is not None:
        cmd.extend(["--session-id", session_id])
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(prompt)

    logger.info("invoking: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        env=cli_env(service, provider="claude"),
        capture_output=True,
        text=True,
        timeout=CLAUDE_TIMEOUT,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode == 0, (
        f"claude exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip(), "claude produced no stdout"
    payload = json.loads(proc.stdout)
    # `--output-format json` wraps the turn; the exact field names evolve
    # with Claude Code versions, so just confirm the shape is an object and
    # carries *some* non-empty field.
    assert isinstance(payload, dict), f"expected JSON object, got {type(payload)}"
    assert any(payload.values()), f"claude JSON payload is empty: {payload}"
    return payload


# ---------------------------------------------------------------------------
# C1 — plain one-shot completion
# ---------------------------------------------------------------------------
def test_hello(harness_service) -> None:
    proc = _run_claude(harness_service, "Say hi in one word.")
    _assert_ok(proc)


# ---------------------------------------------------------------------------
# C2 — tool_call: request carries a tool block (Bash), mocker answers in text
# ---------------------------------------------------------------------------
def test_tool_call(harness_service) -> None:
    proc = _run_claude(
        harness_service,
        "If you need a shell, use Bash. Otherwise, answer directly.",
        extra_args=["--allowedTools", "Bash"],
    )
    _assert_ok(proc)


# ---------------------------------------------------------------------------
# C3 — multiturn: same session_id resumed for a follow-up turn
# ---------------------------------------------------------------------------
def test_multiturn(harness_service) -> None:
    session_id = str(uuid.uuid4())

    first = _run_claude(
        harness_service,
        "Remember the number 7.",
        session_id=session_id,
    )
    _assert_ok(first)

    second = _run_claude(
        harness_service,
        "What number did I ask you to remember?",
        session_id=session_id,
        extra_args=["--resume", session_id],
    )
    _assert_ok(second)
