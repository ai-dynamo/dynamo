# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend API-surface compliance suite against a live Dynamo frontend.

Subject under test is Dynamo's HTTP surface (`/v1/responses` and
`/v1/messages` wire shapes, tool-call routing through both); sglang is
just the backend vehicle for producing real traffic. Runs three suites
sequentially against one server:

1. Upstream OpenResponses compliance-test.ts harness (bun/TypeScript
   validator against zod schemas generated from the OpenAPI spec).
2. `codex exec` smoke — forces the shell tool-call path through
   `/v1/responses`.
3. `claude -p` smoke — forces the Bash tool-call path through
   `/v1/messages` (Anthropic Messages API).

The OpenResponses suite is pre-cloned and `bun install`ed into
`/opt/openresponses` by `container/Dockerfile.test` at a pinned SHA;
bumps go through that file in lockstep.

Non-gating by design: carries the `frontend_api_surface_compliance`
marker which routes to a dedicated CI job kept out of
`backend-status-check.needs` until the signal is trustworthy.
"""

import logging
import os
import subprocess
import time

import pytest
import requests

from tests.serve.common import WORKSPACE_DIR
from tests.utils.engine_process import EngineConfig, EngineProcess

logger = logging.getLogger(__name__)

sglang_dir = os.environ.get("SGLANG_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang"
)

# OpenResponses suite pre-installed here by Dockerfile.test; SHA pin lives
# with the git clone in that file. This path must stay in sync with it.
OPENRESPONSES_DIR = "/opt/openresponses"

COMPLIANCE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(COMPLIANCE_MODEL)
@pytest.mark.profiled_vram_gib(6.0)
@pytest.mark.requested_sglang_kv_tokens(512)
# Budget: sglang cold start (30-60s) + bun compliance (up to 180s) +
# codex exec (up to 180s) + claude exec (up to 180s) + two inter-suite
# health checks + teardown. 750s leaves headroom for CI variance
# without masking real hangs.
@pytest.mark.timeout(750)
# Routed to the dedicated `frontend-api-surface-compliance-check` CI job
# on PRs, deliberately *not* listed in `backend-status-check.needs` —
# failures surface on the PR but don't block merge. Promote by adding
# the job to that list once the suite's signal is trustworthy. Marker
# is outside `pre_merge` so the main sglang gate job's filter doesn't
# pick it up; `post_merge` satisfies the repo's required Lifecycle
# marker check and runs the suite on main after merge for trailing
# signal via post-merge-ci.yml.
@pytest.mark.frontend_api_surface_compliance
@pytest.mark.post_merge
def test_frontend_api_surface_compliance(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    tmp_path,
):
    """Assert the frontend passes the upstream OpenResponses compliance suite."""

    frontend_port = int(dynamo_dynamic_ports.frontend_port)
    system_port = int(dynamo_dynamic_ports.system_ports[0])

    if not os.path.isdir(OPENRESPONSES_DIR):
        pytest.skip(
            f"OpenResponses suite not pre-installed at {OPENRESPONSES_DIR} "
            "— rebuild the test image from container/Dockerfile.test"
        )

    config = EngineConfig(
        name="responses_compliance",
        directory=sglang_dir,
        marks=[],
        request_payloads=[],
        model=COMPLIANCE_MODEL,
        script_name="agg.sh",
        # Qwen3-VL-2B-specific flags: vision-model CUDA graph workaround +
        # model-aware reasoning/tool-call parsers. Forwarded verbatim to
        # `dynamo.sglang` by agg.sh's pass-through loop.
        #
        # Tool-call parser is `hermes`, not `qwen3_coder`: Qwen3-VL-Instruct
        # emits `<tool_call>{"name":..., "arguments":...}</tool_call>` (JSON
        # inside the tags — Hermes-style), while `qwen3_coder` expects the
        # XML-structured `<tool_call><function=name><parameter=k>v</parameter>
        # </function></tool_call>` that Qwen3-Coder models emit. Using the
        # wrong parser leaves tool calls as raw text in the response and
        # breaks end-to-end agent flows (codex exec, etc.).
        script_args=[
            "--model-path",
            COMPLIANCE_MODEL,
            "--disable-piecewise-cuda-graph",
            "--dyn-reasoning-parser",
            "qwen3",
            "--dyn-tool-call-parser",
            "hermes",
        ],
        timeout=360,
        env={},
        frontend_port=frontend_port,
    )

    merged_env = {
        "DYN_HTTP_PORT": str(frontend_port),
        "DYN_SYSTEM_PORT": str(system_port),
        # agg.sh doesn't forward frontend args, but the frontend reads this
        # env var directly. Enables /v1/messages for the claude smoke step.
        "DYN_ENABLE_ANTHROPIC_API": "1",
    }

    codex_home = tmp_path / "codex_home"
    _write_codex_config(codex_home, frontend_port)

    # Marker file that the agents can only "see" by invoking their shell/Bash
    # tool; if a model answers from its prior without actually running `ls`,
    # the marker won't appear in stdout and the assertion fails. Proves the
    # tool-call paths through the frontend end-to-end (both /v1/responses
    # for codex and /v1/messages for claude), not just text generation.
    agent_cwd = tmp_path / "agent_cwd"
    agent_cwd.mkdir()
    marker_filename = "dynamo_compliance_marker.txt"
    (agent_cwd / marker_filename).write_text("compliance-smoke")

    # Isolated HOME so claude doesn't write session state into the runner's
    # ~/.claude during CI / local invocation.
    claude_home = tmp_path / "claude_home"
    claude_home.mkdir()

    with EngineProcess.from_script(config, request, extra_env=merged_env):
        _run_bun_compliance(frontend_port)
        _wait_for_frontend_healthy(frontend_port)
        _run_codex_exec_smoke(codex_home, agent_cwd, marker_filename)
        _wait_for_frontend_healthy(frontend_port)
        _run_claude_exec_smoke(claude_home, agent_cwd, marker_filename, frontend_port)


def _wait_for_frontend_healthy(
    frontend_port: int, timeout_s: float = 15.0, model: str = COMPLIANCE_MODEL
) -> None:
    """Confirm the frontend is still serving before the next subprocess fires.

    Without this check, if bun compliance accidentally destabilized the
    server (e.g. a hang that the bun timeout cut short) a codex exec
    failure looks identical to "codex is broken" in CI logs. The health
    probe collapses that ambiguity: if the frontend has crashed or the
    worker has deregistered, fail here with a clear message rather than
    letting codex run and time out.
    """
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(
                f"http://localhost:{frontend_port}/v1/models", timeout=2
            )
            if resp.ok and any(
                m.get("id") == model for m in resp.json().get("data", [])
            ):
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.5)
    pytest.fail(
        f"frontend unhealthy after bun compliance — /v1/models did not list "
        f"{model!r} within {timeout_s}s (last error: {last_err})"
    )


def _run_bun_compliance(frontend_port: int) -> None:
    """Invoke compliance-test.ts against the running frontend."""
    base_url = f"http://localhost:{frontend_port}/v1"
    logger.info("Running OpenResponses compliance suite against %s", base_url)

    result = subprocess.run(
        [
            "bun",
            "run",
            "bin/compliance-test.ts",
            "--base-url",
            base_url,
            "--api-key",
            "sk-compliance-dummy",
            "--model",
            COMPLIANCE_MODEL,
            "--verbose",
        ],
        cwd=OPENRESPONSES_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )

    # Surface the suite output whether it passed or failed so CI logs are
    # self-contained; pytest will attach these to the failure diff.
    if result.stdout:
        logger.info("compliance stdout:\n%s", result.stdout)
    if result.stderr:
        logger.info("compliance stderr:\n%s", result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"OpenResponses compliance suite failed (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def _write_codex_config(codex_home, frontend_port: int) -> None:
    """Emit a minimal ~/.codex/config.toml pointing Codex at Dynamo.

    Using a per-test CODEX_HOME keeps the runner's global Codex state
    (if any) untouched.
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    config_path.write_text(
        f"""
[model_providers.local]
name = "local-dynamo"
base_url = "http://localhost:{frontend_port}/v1"
wire_api = "responses"
env_key = "LOCAL_API_KEY"
""".lstrip()
    )


def _run_codex_exec_smoke(codex_home, cwd, marker_filename: str) -> None:
    """Run `codex exec` against the Dynamo Responses endpoint and assert the
    tool-call path actually fires.

    We prompt codex to list `cwd`; `cwd` contains `marker_filename` and nothing
    else the model could pattern-match from prior knowledge. If codex answers
    without invoking its shell tool, the marker won't appear in stdout and the
    assertion fails — which proves we're testing the full Responses API
    tool-calling chain, not just text generation.
    """
    logger.info("Running codex exec smoke test against CODEX_HOME=%s", codex_home)

    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "LOCAL_API_KEY": "sk-none",
    }

    result = subprocess.run(
        [
            "codex",
            "-m",
            COMPLIANCE_MODEL,
            "-c",
            "model_provider=local",
            "exec",
            "Run `ls` in the current directory and tell me the filenames.",
            "--dangerously-bypass-approvals-and-sandbox",
        ],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.stdout:
        logger.info("codex stdout:\n%s", result.stdout)
    if result.stderr:
        logger.info("codex stderr:\n%s", result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"codex exec failed (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if marker_filename not in result.stdout:
        pytest.fail(
            "codex exec did not report the marker file — expected stdout to "
            f"contain {marker_filename!r} (implies the shell tool was invoked "
            f"and actually ran `ls` in {cwd}). Got:\n{result.stdout}"
        )


def _run_claude_exec_smoke(
    claude_home, cwd, marker_filename: str, frontend_port: int
) -> None:
    """Run `claude -p` against the Dynamo Anthropic Messages endpoint and
    assert the Bash tool-call path actually fires.

    Same marker-file pattern as the codex step but hitting /v1/messages:
    if claude answers without invoking its Bash tool, the marker won't
    appear in stdout and the assertion fails — which proves the full
    Anthropic Messages + tool-calling chain, not just text generation.

    Isolated HOME so claude doesn't write session state into the runner's
    `~/.claude`. An `ANTHROPIC_AUTH_TOKEN` is required even though Dynamo
    ignores the value: on a fresh HOME with no cached OAuth, the CLI
    aborts with "Not logged in" unless a bearer is supplied.
    """
    base_url = f"http://localhost:{frontend_port}"
    logger.info("Running claude exec smoke test against %s", base_url)

    env = {
        **os.environ,
        "HOME": str(claude_home),
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": "sk-none",
    }

    result = subprocess.run(
        [
            "claude",
            "--model",
            COMPLIANCE_MODEL,
            "--dangerously-skip-permissions",
            "-p",
            "Run `ls` in the current directory and tell me the filenames.",
        ],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.stdout:
        logger.info("claude stdout:\n%s", result.stdout)
    if result.stderr:
        logger.info("claude stderr:\n%s", result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"claude -p failed (exit={result.returncode}).\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    if marker_filename not in result.stdout:
        pytest.fail(
            "claude -p did not report the marker file — expected stdout to "
            f"contain {marker_filename!r} (implies the Bash tool was invoked "
            f"and actually ran `ls` in {cwd}). Got:\n{result.stdout}"
        )
