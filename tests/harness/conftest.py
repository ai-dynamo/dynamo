# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness fixtures for real-CLI e2e tests against the mocker backend.

Boots `dynamo.frontend` with `--enable-anthropic-api` plus a `dynamo.mocker`
worker on free ports, waits until the HTTP completions route can serve the
model, and yields a `HarnessService` whose `openai_base` / `anthropic_base`
are pointed at the live frontend.

Consumer tests drive the CLI under test (`codex exec`, `claude --print`) at
that base URL, so the assertions exercise the real agent loop end-to-end.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Generator

import pytest

from tests.frontend.conftest import (
    MockerWorkerProcess,
    wait_for_http_completions_ready,
)
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

logger = logging.getLogger(__name__)

HARNESS_MODEL = QWEN


@dataclass(frozen=True)
class HarnessService:
    """Handle to a running frontend+mocker pair."""

    frontend_port: int
    model: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.frontend_port}"

    @property
    def openai_base(self) -> str:
        # Codex / OpenAI-compatible clients expect the `/v1` suffix.
        return f"{self.base_url}/v1"

    @property
    def anthropic_base(self) -> str:
        # Claude Code's ANTHROPIC_BASE_URL is the host root; the client
        # appends `/v1/messages` itself.
        return self.base_url


@pytest.fixture(scope="function")
def harness_service(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
) -> Generator[HarnessService, None, None]:
    """Start frontend (Anthropic API enabled) + mocker for a single test."""
    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--enable-anthropic-api"],
        extra_env={"DYN_ENABLE_ANTHROPIC_API": "1"},
        terminate_all_matching_process_names=False,
    ):
        with MockerWorkerProcess(
            request,
            model=HARNESS_MODEL,
            frontend_port=frontend_port,
            system_port=system_port,
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=HARNESS_MODEL
            )
            logger.info(
                "harness_service ready on port %d (anthropic enabled)", frontend_port
            )
            yield HarnessService(frontend_port=frontend_port, model=HARNESS_MODEL)


def cli_env(service: HarnessService, *, provider: str) -> dict[str, str]:
    """Build a minimal env for invoking the CLI under test.

    `provider` is "codex" (OpenAI-shaped) or "claude" (Anthropic-shaped).
    Copies the current process env to inherit PATH/HF_HOME and then overrides
    the URL + API-key vars the CLI reads.
    """
    env = os.environ.copy()
    if provider == "codex":
        env["OPENAI_BASE_URL"] = service.openai_base
        env["OPENAI_API_KEY"] = "harness-dummy"
    elif provider == "claude":
        env["ANTHROPIC_BASE_URL"] = service.anthropic_base
        env["ANTHROPIC_API_KEY"] = "harness-dummy"
        # Opt out of OAuth / session caching so the harness never touches
        # real Anthropic auth state. `claude --bare` reinforces this too.
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return env


def has_cli(name: str) -> bool:
    return shutil.which(name) is not None
