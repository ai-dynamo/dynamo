# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenResponses compliance suite against a live Dynamo frontend.

Spins up an aggregated sglang worker behind a frontend, then runs the
upstream OpenResponses compliance-test.ts suite (a bun/TypeScript harness
that validates the response wire shape against zod schemas generated from
the OpenAPI spec) plus a codex exec smoke test. The suite is pre-cloned
and `bun install`ed into `/opt/openresponses` by `container/Dockerfile.test`
at a pinned SHA; bumps go through that file in lockstep.
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
# inter-suite health check + codex exec (up to 180s) + teardown. 600s
# leaves headroom for CI variance without masking real hangs.
@pytest.mark.timeout(600)
@pytest.mark.pre_merge
def test_responses_openresponses_compliance(
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
        script_args=[
            "--model-path",
            COMPLIANCE_MODEL,
            "--disable-piecewise-cuda-graph",
            "--dyn-reasoning-parser",
            "qwen3",
            "--dyn-tool-call-parser",
            "qwen3_coder",
        ],
        timeout=360,
        env={},
        frontend_port=frontend_port,
    )

    merged_env = {
        "DYN_HTTP_PORT": str(frontend_port),
        "DYN_SYSTEM_PORT": str(system_port),
    }

    codex_home = tmp_path / "codex_home"
    _write_codex_config(codex_home, frontend_port)

    with EngineProcess.from_script(config, request, extra_env=merged_env):
        _run_bun_compliance(frontend_port)
        _wait_for_frontend_healthy(frontend_port)
        _run_codex_exec_smoke(codex_home)


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
            if resp.ok and any(m.get("id") == model for m in resp.json().get("data", [])):
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


def _run_codex_exec_smoke(codex_home) -> None:
    """Run `codex exec` against the Dynamo Responses endpoint.

    Loose assertion: exit 0 and model produced a non-empty answer. The
    agent may invoke tool calls, so we don't pin the exact output.
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
            "what is 10+10",
            "--dangerously-bypass-approvals-and-sandbox",
        ],
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

    if "20" not in result.stdout:
        pytest.fail(
            f"codex exec produced unexpected output — expected the answer '20' "
            f"for 'what is 10+10', got:\n{result.stdout}"
        )
