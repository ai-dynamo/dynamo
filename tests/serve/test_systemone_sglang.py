# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native parity qualification for SGLang 0.5.19 and 0.5.22 targets.

Set DYNAMO_DECISION_RUN_PINNED_SGLANG=1 in an explicitly provisioned target
environment. Version acceptance enables the campaign; it is not qualification
evidence. Ordinary CI deselects these tests before worker startup. This
small-model campaign does not establish Qwen3.8-27B qualification or physical
GPU cancellation settlement.
"""

import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import pytest
import requests
from transformers import AutoTokenizer

from tests.utils.constants import QWEN
from tests.utils.decision_api import assert_chat_decode, assert_native_sglang_parity
from tests.utils.http_checks import check_health_ready, check_http_ok, models_available
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import reserved_ports

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.model(QWEN),
    # Two sequential 360s worker starts, six 60s parity requests, and teardown.
    pytest.mark.timeout(1800),
    # NVML peak includes Dynamo and standalone SGLang, each with 2048 KV tokens.
    pytest.mark.profiled_vram_gib(6.4),
    pytest.mark.requested_sglang_kv_tokens(2048),
]


@pytest.fixture
def systemone_sglang_server(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    assert re.fullmatch(
        r"0\.5\.(?:19|22)(?:\+cu[0-9]+)?", version("sglang")
    ), "Parity targets SGLang 0.5.19 or 0.5.22 with an optional CUDA build suffix"
    ports = dynamo_dynamic_ports
    base_url = f"http://localhost:{ports.frontend_port}"
    env = {**os.environ, "DYN_SYSTEM_PORT": str(ports.system_ports[0])}
    with (
        DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            extra_args=["--enable-systemone-api"],
            extra_env={"DYN_SGLANG_ENABLE_GENERATE": "1"},
            terminate_all_matching_process_names=False,
        ),
        ManagedProcess(
            command=[
                sys.executable,
                "-m",
                "dynamo.sglang",
                "--model-path",
                QWEN,
                "--served-model-name",
                QWEN,
                "--context-length",
                "2048",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.9",
                "--page-size",
                "16",
                "--tp",
                "1",
                "--disable-piecewise-cuda-graph",
                "--enable-metrics",
                "--disable-overlap-schedule",
                "--max-running-requests",
                "1",
                "--pp-size",
                "1",
            ],
            env=env,
            health_check_urls=[
                (f"{base_url}/v1/models", models_available),
                (
                    f"http://localhost:{ports.system_ports[0]}/health",
                    check_health_ready,
                ),
            ],
            timeout=360,
            display_output=True,
            display_name="sglang",
            terminate_all_matching_process_names=False,
            log_dir=f"{request.node.name}_sglang",
        ),
    ):
        yield base_url


def _native_prefill_graph_args():
    if version("sglang").split("+", 1)[0] == "0.5.22":
        return ["--cuda-graph-backend-prefill", "disabled"]
    return ["--disable-piecewise-cuda-graph"]


@pytest.fixture
def native_sglang_server(request, dynamo_dynamic_ports, predownload_models):
    with reserved_ports(1, dynamo_dynamic_ports.frontend_port) as ports:
        base_url = f"http://localhost:{ports[0]}"
        with ManagedProcess(
            command=[
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                QWEN,
                "--served-model-name",
                QWEN,
                "--host",
                "127.0.0.1",
                "--port",
                str(ports[0]),
                "--context-length",
                "2048",
                "--max-total-tokens",
                "2048",
                "--mem-fraction-static",
                "0.2",
                *_native_prefill_graph_args(),
                "--disable-overlap-schedule",
                "--max-running-requests",
                "1",
            ],
            env=os.environ.copy(),
            health_check_urls=[(f"{base_url}/health", check_http_ok)],
            timeout=360,
            display_output=True,
            display_name="native-sglang",
            terminate_all_matching_process_names=False,
            log_dir=f"{request.node.name}_native_sglang",
        ):
            yield base_url


def test_systemone_matches_native_zero_decode_scores(
    systemone_sglang_server, native_sglang_server
):
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    assert_native_sglang_parity(
        systemone_sglang_server, native_sglang_server, QWEN, tokenizer
    )


def test_systemone_scoring_and_chat_overlap(systemone_sglang_server):
    scoring = {
        "model": QWEN,
        "state": "A customer requests a refund today.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Action is needed today."}
        },
    }
    scoring_oai = {
        "model": QWEN,
        "input": scoring["state"],
        "questions": [
            {
                "type": "predicate",
                "name": "urgent",
                "instructions": "Action is needed today.",
            }
        ],
    }
    chat = {
        "model": QWEN,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def invoke(index):
        route, payload = (
            ("systemone", scoring),
            ("decisions", scoring_oai),
            ("chat/completions", chat),
        )[index % 3]
        response = requests.post(
            f"{systemone_sglang_server}/v1/{route}", json=payload, timeout=60
        )
        assert response.status_code == 200, response.text
        body = response.json()
        if route != "chat/completions":
            assert body["usage"]["output_tokens"] == 0
            if route == "systemone":
                answer = body["answers"]["urgent"]
                probability = answer["noul"]
            else:
                assert len(body["answers"]) == 1
                answer = body["answers"][0]
                assert answer["name"] == "urgent" and answer["type"] == "predicate"
                probability = answer["probability"]
            assert math.isfinite(probability) and 0 <= probability <= 1
            if "x_label_mass" in answer:
                assert (
                    math.isfinite(answer["x_label_mass"])
                    and 0 <= answer["x_label_mass"] <= 1
                )
        else:
            assert_chat_decode(body)

    # The serialized worker admits one active request and one queued request.
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(invoke, range(24)))
