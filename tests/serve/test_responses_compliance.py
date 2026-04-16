# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenResponses compliance suite against a live Dynamo frontend.

Spins up an aggregated sglang worker behind a frontend with Responses and
Anthropic Messages APIs enabled, then runs the upstream OpenResponses
compliance-test.ts suite (a bun/TypeScript harness that validates the
response wire shape against zod schemas generated from the OpenAPI spec).

The suite is pinned to a specific commit so flakes are locked to our
version bumps rather than upstream changes.
"""

import logging
import os
import subprocess

import pytest

from tests.serve.common import WORKSPACE_DIR
from tests.utils.engine_process import EngineConfig, EngineProcess

logger = logging.getLogger(__name__)

sglang_dir = os.environ.get("SGLANG_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/sglang"
)

# Pin the OpenResponses checkout. Update in lockstep with intentional bumps
# — bumps should include a CI run showing the suite still passes.
OPENRESPONSES_REPO = "https://github.com/openresponses/openresponses.git"
OPENRESPONSES_SHA = "fa29df5"

COMPLIANCE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.mark.sglang
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(COMPLIANCE_MODEL)
@pytest.mark.profiled_vram_gib(6.0)
@pytest.mark.requested_sglang_kv_tokens(512)
@pytest.mark.timeout(420)
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

    config = EngineConfig(
        name="responses_compliance",
        directory=sglang_dir,
        marks=[],
        request_payloads=[],
        model=COMPLIANCE_MODEL,
        script_name="agg_compliance.sh",
        script_args=[],
        timeout=360,
        env={"MODEL": COMPLIANCE_MODEL},
        frontend_port=frontend_port,
    )

    merged_env = {
        "MODEL": COMPLIANCE_MODEL,
        "DYN_HTTP_PORT": str(frontend_port),
        "DYN_SYSTEM_PORT": str(system_port),
    }

    repo_dir = tmp_path / "openresponses"
    _clone_openresponses(repo_dir)

    with EngineProcess.from_script(config, request, extra_env=merged_env):
        _run_bun_compliance(repo_dir, frontend_port)


def _clone_openresponses(repo_dir) -> None:
    """Shallow-clone OpenResponses at the pinned SHA into `repo_dir`."""
    subprocess.run(
        ["git", "clone", "--filter=blob:none", OPENRESPONSES_REPO, str(repo_dir)],
        check=True,
    )
    subprocess.run(["git", "checkout", OPENRESPONSES_SHA], cwd=repo_dir, check=True)
    # `bun install` fetches the suite's runtime deps (zod, etc.).
    subprocess.run(["bun", "install", "--frozen-lockfile"], cwd=repo_dir, check=True)


def _run_bun_compliance(repo_dir, frontend_port: int) -> None:
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
        cwd=repo_dir,
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
