# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

import pytest
import requests
from transformers import AutoTokenizer

from tests.serve.common import WORKSPACE_DIR, managed_serve_deployment
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.post_merge
@pytest.mark.model(MODEL)
@pytest.mark.requested_sglang_kv_tokens(2048)
@pytest.mark.timeout(360)  # Measured 34s aggregated and 58s disaggregated.
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
@pytest.mark.parametrize(
    "topology",
    [
        pytest.param("agg", marks=pytest.mark.profiled_vram_gib(4.2)),
        pytest.param("disagg_same_gpu", marks=pytest.mark.profiled_vram_gib(7.5)),
    ],
)
@pytest.mark.parametrize("processor", ["dynamo", "sglang"])
def test_sglang_thinking_budget_enforcement(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    tmp_path,
    topology,
    processor,
):
    """Check real reasoning output, including the disaggregated first token."""
    payload = chat_payload(
        "Prove that there are infinitely many primes.",
        repeat_count=1,
        expected_response=[],
        max_tokens=512,
        temperature=0,
    )
    config = EngineConfig(
        name=f"thinking_budget_{topology}_{processor}",
        directory=str(Path(WORKSPACE_DIR) / "examples/backends/sglang"),
        script_name=f"{topology}.sh",
        script_args=[
            "--enable-strict-thinking",
            "--reasoning-parser",
            "qwen3",
            "--dyn-reasoning-parser",
            "qwen3",
            "--grammar-backend",
            "xgrammar",
            "--disable-cuda-graph",
        ],
        marks=[],
        model=MODEL,
        request_payloads=[payload],
        env={"DYN_CHAT_PROCESSOR": processor},
        health_check_workers=topology == "disagg_same_gpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    evidence = []
    with managed_serve_deployment(config, request, ports=dynamo_dynamic_ports):
        for budget in [None, 0, 16]:
            body = {**payload.body, "model": MODEL}
            if budget is not None:
                body["thinking_token_budget"] = budget
            response = requests.post(
                f"http://localhost:{dynamo_dynamic_ports.frontend_port}/v1/chat/completions",
                json=body,
                timeout=120,
            )
            assert response.status_code == 200, response.text
            result = response.json()
            message = result["choices"][0]["message"]
            reasoning = message.get("reasoning_content") or ""
            retokenized_count = len(
                tokenizer.encode(reasoning, add_special_tokens=False)
            )
            # The SGLang chat processor does not expose reasoning usage details.
            count = (
                result["usage"]["completion_tokens_details"]["reasoning_tokens"]
                if processor == "dynamo"
                else retokenized_count
            )
            evidence.append(
                {
                    "budget": budget,
                    "reasoning_tokens": count,
                    "count_source": "usage" if processor == "dynamo" else "tokenizer",
                    "retokenized_tokens": retokenized_count,
                    "response": result,
                }
            )
            (tmp_path / "thinking_budget_results.json").write_text(
                json.dumps(evidence, indent=2)
            )
            logger.info("Budget %s: %d reasoning tokens", budget, count)
            if budget is None:
                assert (
                    count > 16
                ), "Control prompt must exercise reasoning beyond the tested budget"
            else:
                assert (
                    count <= budget
                ), f"Budget {budget} produced {count} reasoning tokens: {reasoning!r}"
                assert message.get(
                    "content"
                ), "Budget closure must allow a final answer"
                assert "</think>" not in message["content"]
