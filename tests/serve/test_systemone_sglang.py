# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-backed System One parity against native SGLang candidate scoring."""

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import pytest
import requests
from transformers import AutoTokenizer

from tests.utils.constants import QWEN
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
    pytest.mark.timeout(600),
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
    assert (
        version("sglang").split("+")[0] == "0.5.19"
    ), "Parity is pinned to SGLang 0.5.19"
    ports = dynamo_dynamic_ports
    base_url = f"http://localhost:{ports.frontend_port}"
    env = {**os.environ, "DYN_SYSTEM_PORT": str(ports.system_ports[0])}
    with (
        DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            extra_args=["--enable-systemone-api"],
            extra_env={"DYN_SGLANG_ENABLE_GENERATE": "1"},
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
                "--disable-piecewise-cuda-graph",
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


def _native_probabilities(base_url, tokenizer, content, labels):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt)
    label_ids = []
    for label in labels:
        encoded = tokenizer.encode(prompt + label)
        assert encoded[:-1] == prompt_ids and len(encoded) == len(prompt_ids) + 1
        label_ids.append(encoded[-1])
    assert len(set(label_ids)) == len(labels)
    payload = {
        "model": QWEN,
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "token_ids_logprob": label_ids,
        "cache_salt": "systemone-parity-reference",
        "stream": True,
    }
    frames = []
    with requests.post(
        f"{base_url}/generate", json=payload, stream=True, timeout=60
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                frames.append(json.loads(line[6:]))
    assert frames, "Native Generate returned no score frame"
    final = frames[-1]
    assert "error" not in final, final
    assert final.get("output_ids", []) == []
    assert final["meta_info"]["completion_tokens"] == 0
    rows = final["meta_info"]["output_token_ids_logprobs"]
    assert len(rows) == 1
    assert [row[1] for row in rows[0]] == label_ids
    logprobs = [row[0] for row in rows[0]]
    assert all(math.isfinite(value) for value in logprobs)
    weights = [math.exp(value - max(logprobs)) for value in logprobs]
    return (
        [weight / sum(weights) for weight in weights],
        sum(map(math.exp, logprobs)),
        len(prompt_ids),
    )


def test_systemone_matches_native_zero_decode_scores(
    systemone_sglang_server, native_sglang_server
):
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    state = "A customer was charged twice and requests a refund today."
    payload = {
        "model": QWEN,
        "state": state,
        "chat_template_kwargs": {"enable_thinking": False},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose the responsible team.",
                "criteria": {"billing": "Payments", "technical": "Software"},
            },
            "urgent": {"type": "noul", "instructions": "Action is needed today."},
            "severity": {
                "type": "score",
                "instructions": "Rate the impact.",
                "criteria": ["Low", "Moderate", "High"],
            },
        },
    }
    response = requests.post(
        f"{systemone_sglang_server}/v1/systemone", json=payload, timeout=60
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-dynamo-systemone-version"] == "1"
    body = response.json()
    assert list(body["answers"]) == ["route", "urgent", "severity"]
    references = (
        (
            "route",
            f"{state}\n\nQuestion: Choose the responsible team.\nA: billing - Payments\nB: technical - Software\nAnswer with the letter of one option only.",
            ["A", "B"],
        ),
        (
            "urgent",
            f"{state}\n\nIs the following true? Action is needed today.\nAnswer with yes or no only.",
            ["yes", "no"],
        ),
        (
            "severity",
            f"{state}\n\nQuestion: Rate the impact.\n0: Low\n1: Moderate\n2: High\nAnswer with the number of one level only.",
            ["0", "1", "2"],
        ),
    )
    total_input_tokens = 0
    for question_id, content, labels in references:
        probabilities, label_mass, input_tokens = _native_probabilities(
            native_sglang_server, tokenizer, content, labels
        )
        total_input_tokens += input_tokens
        answer = body["answers"][question_id]
        assert answer["x_label_mass"] == pytest.approx(label_mass, abs=5e-4, rel=1e-3)
        if question_id == "urgent":
            assert answer["noul"] == pytest.approx(probabilities[0], abs=5e-4, rel=1e-3)
        else:
            assert list(answer["probabilities"].values()) == pytest.approx(
                probabilities, abs=5e-4, rel=1e-3
            )
            assert sum(answer["probabilities"].values()) == pytest.approx(
                1.0, abs=1e-12
            )
            if question_id == "severity":
                expected_score = sum(
                    index * value for index, value in enumerate(probabilities)
                )
                assert answer["score"] == pytest.approx(
                    expected_score, abs=5e-4, rel=1e-3
                )
    assert body["usage"] == {"input_tokens": total_input_tokens, "output_tokens": 0}
    chat = requests.post(
        f"{systemone_sglang_server}/v1/chat/completions",
        json={
            "model": QWEN,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    assert chat.status_code == 200, chat.text
    assert len(chat.json()["choices"]) == 1


def _post_with_capacity_retry(url, payload):
    for attempt in range(3):
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code not in (503, 529) or attempt == 2:
            return response
        assert response.headers.get("retry-after") == "1", response.text
        time.sleep(1)


def test_systemone_scoring_and_chat_overlap(systemone_sglang_server):
    scoring = {
        "model": QWEN,
        "state": "A customer requests a refund today.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Action is needed today."}
        },
    }
    chat = {
        "model": QWEN,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def invoke(index):
        route, payload = (
            ("systemone", scoring) if index % 2 == 0 else ("chat/completions", chat)
        )
        response = _post_with_capacity_retry(
            f"{systemone_sglang_server}/v1/{route}", payload
        )
        assert response.status_code == 200, response.text
        body = response.json()
        if index % 2 == 0:
            assert body["usage"]["output_tokens"] == 0
            answer = body["answers"]["urgent"]
            assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1
            assert 0 <= answer["x_label_mass"] <= 1
        else:
            assert len(body["choices"]) == 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(invoke, range(24)))
