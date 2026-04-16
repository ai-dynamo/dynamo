# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real `codex` CLI driving Dynamo's `/v1/responses` endpoint over the mocker.

These tests invoke the actual OpenAI Codex CLI (`codex exec`) in headless
mode with `OPENAI_BASE_URL` pointed at the harness frontend. The mocker
emits deterministic fake tokens, so the assertions focus on exit code and
that *some* output was produced — not on specific token content.

`tool_call` exercises the tool-definition translation path: Codex is
pointed at a prompt that would ordinarily invoke a shell tool, so the
outgoing request carries a `tools` block which the Responses-to-
ChatCompletion converter has to handle. The mocker will not emit a
`function_call` item in its reply, which is fine — the assertion is that
the request reaches the worker and a valid reply comes back.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from tests.harness.conftest import cli_env, has_cli

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e_agent,
    pytest.mark.gpu_0,
    pytest.mark.skipif(not has_cli("codex"), reason="`codex` CLI not on PATH"),
]

CODEX_TIMEOUT = 120


def _run_codex(
    service,
    prompt: str,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    # `codex exec` is the non-interactive one-shot mode. We pin the model
    # to the name the mocker registered, and disable sandboxing so the CLI
    # doesn't try to spawn its seatbelt/landlock wrapper.
    cmd = [
        "codex",
        "exec",
        "--model",
        service.model,
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(prompt)

    logger.info("invoking: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        env=cli_env(service, provider="codex"),
        capture_output=True,
        text=True,
        timeout=CODEX_TIMEOUT,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, (
        f"codex exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip() or proc.stderr.strip(), "codex produced no output"


# ---------------------------------------------------------------------------
# X1 — plain one-shot completion
# ---------------------------------------------------------------------------
def test_hello(harness_service) -> None:
    proc = _run_codex(harness_service, "Say hi in one word.")
    _assert_ok(proc)


# ---------------------------------------------------------------------------
# X2 — tool_call: `codex exec` sends shell-tool definitions by default
# ---------------------------------------------------------------------------
def test_tool_call(harness_service) -> None:
    # Codex's `exec` mode always sends a shell tool definition, so any
    # prompt phrased as a shell task exercises the tool path. The mocker
    # won't issue a function_call; we assert the CLI completes cleanly.
    proc = _run_codex(
        harness_service,
        "If you need a shell, use it. Otherwise reply in plain text.",
    )
    _assert_ok(proc)


# ---------------------------------------------------------------------------
# X3 — multiturn: two consecutive `codex exec` invocations against the same
#       service. `codex exec` doesn't persist sessions, so each call is a
#       fresh conversation — this catches any state leakage between calls.
# ---------------------------------------------------------------------------
def test_multiturn(harness_service) -> None:
    first = _run_codex(harness_service, "What is 1+1?")
    _assert_ok(first)

    second = _run_codex(harness_service, "What is 2+2?")
    _assert_ok(second)
