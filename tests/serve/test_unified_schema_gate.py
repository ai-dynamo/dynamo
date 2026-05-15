# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the unified backend's schema gate.

Launches the CPU-only sample backend with ``unsupported_field_policy=reject``
and sends an OpenAI request that populates the ``agent_context``
forwarded field via ``nvext``. The sample engine does not declare
``Capability::AgentContext`` (capabilities are hard-coded per backend),
so the gate must reject the request — covering the full HTTP →
frontend → preprocessor → worker → engine path.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import requests

from tests.serve.common import WORKSPACE_DIR
from tests.utils.engine_process import EngineConfig, EngineProcess

sample_dir = os.path.join(WORKSPACE_DIR, "examples/backends/sample")
MODEL = "Qwen/Qwen3-0.6B"

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.timeout(300),
    pytest.mark.unified,
    # Piggyback on the vLLM CI image (sample backend is CPU-only and has no
    # vLLM dependency, but the `pre_merge and vllm and gpu_0` marker filter
    # in .github/workflows/pr.yaml is what selects the test into a runner).
    pytest.mark.vllm,
]


def _agent_context_chat_request(model: str) -> dict[str, Any]:
    """OpenAI chat request that populates `PreprocessedRequest.agent_context`."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": False,
        "nvext": {
            "agent_context": {
                "session_type_id": "test-session-type",
                "session_id": "test-session",
                "trajectory_id": "test-trajectory",
            }
        },
    }


def test_schema_gate_rejects_forwarded_field_without_capability(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    """Reject policy + sample engine (no declared capabilities) + forwarded
    field set → 4xx with a typed error mentioning the missing capability."""
    config = EngineConfig(
        name="schema-gate-reject",
        directory=sample_dir,
        script_name="agg.sh",
        script_args=[
            "--model-name",
            MODEL,
            "--unsupported-field-policy",
            "reject",
        ],
        marks=[],
        model=MODEL,
        frontend_port=dynamo_dynamic_ports.frontend_port,
        request_payloads=[],
    )
    with EngineProcess.from_config(config, request):
        resp = requests.post(
            f"http://localhost:{config.frontend_port}/v1/chat/completions",
            json=_agent_context_chat_request(MODEL),
            timeout=30,
        )
    assert (
        400 <= resp.status_code < 500
    ), f"expected 4xx, got {resp.status_code}: {resp.text}"
    body_lower = resp.text.lower()
    assert (
        "agent_context" in body_lower or "capability" in body_lower
    ), f"error body should name the failing capability; got: {resp.text}"
