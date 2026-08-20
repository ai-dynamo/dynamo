# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end happy-path coverage for token-in/token-out RL rollouts."""

from __future__ import annotations

import json
import math
from functools import partial
from typing import Any, Generator

import pytest

from tests.rl.utils import (
    check_model_registered,
    check_ready,
    prepare_log_dir,
    process_env,
    vllm_gpu_mem_args,
)
from tests.utils.client import send_request
from tests.utils.constants import QWEN
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import ServicePorts

TEST_MODEL = QWEN
TEST_PROMPT = "The three primary colors are red,"
SGLANG_INPUT_IDS = (151644, 8948, 198)
LOGPROB_ATOL = 1e-6

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.model(TEST_MODEL),
]


def _sglang_gpu_mem_args(env: dict[str, str]) -> list[str]:
    return build_gpu_mem_args("build_sglang_gpu_mem_args", env=env) or [
        "--max-total-tokens",
        "2048",
        "--mem-fraction-static",
        "0.9",
    ]


class VllmTitoWorkerProcess(ManagedProcess):
    def __init__(
        self,
        request: pytest.FixtureRequest,
        *,
        frontend_port: int,
        system_port: int,
    ):
        env = process_env(
            DYN_SYSTEM_PORT=str(system_port),
            DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]',
        )

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                TEST_MODEL,
                "--enforce-eager",
                *vllm_gpu_mem_args(),
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "2",
                "--enable-rl",
            ],
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_ready),
                (
                    f"http://localhost:{frontend_port}/v1/models",
                    partial(check_model_registered, model=TEST_MODEL),
                ),
            ],
            timeout=600,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=prepare_log_dir(request, "tito-vllm-worker"),
            display_name="tito-vllm-worker",
        )


class SglangTitoWorkerProcess(ManagedProcess):
    def __init__(
        self,
        request: pytest.FixtureRequest,
        *,
        frontend_port: int,
        system_port: int,
    ):
        env = process_env(
            DYN_SYSTEM_PORT=str(system_port),
            DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]',
        )

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                TEST_MODEL,
                "--served-model-name",
                TEST_MODEL,
                "--page-size",
                "16",
                "--tp",
                "1",
                "--trust-remote-code",
                "--disable-piecewise-cuda-graph",
                *_sglang_gpu_mem_args(env),
            ],
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_ready),
                (
                    f"http://localhost:{frontend_port}/v1/models",
                    partial(check_model_registered, model=TEST_MODEL),
                ),
            ],
            timeout=600,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["SGLANG:EngineCore"],
            straggler_commands=["-m dynamo.sglang"],
            log_dir=prepare_log_dir(request, "tito-sglang-worker"),
            display_name="tito-sglang-worker",
        )


@pytest.fixture(scope="function")
def start_vllm_tito_services(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    predownload_models: object,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports, predownload_models
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env=process_env(),
        terminate_all_matching_process_names=False,
        display_name="tito-vllm-frontend",
    ):
        with VllmTitoWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
        ):
            yield frontend_port


@pytest.fixture(scope="function")
def start_sglang_tito_services(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    predownload_models: object,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports, predownload_models
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env=process_env(DYN_SGLANG_ENABLE_GENERATE="1"),
        terminate_all_matching_process_names=False,
        display_name="tito-sglang-frontend",
    ):
        with SglangTitoWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
        ):
            yield frontend_port


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = send_request(url, payload, timeout=60)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, dict), data
    return data


def _get_nvext(payload: dict[str, Any]) -> dict[str, Any]:
    nvext = payload.get("nvext")
    return nvext if isinstance(nvext, dict) else {}


def _request_vllm_completion(
    *, frontend_port: int, prompt: str | list[int]
) -> dict[str, Any]:
    return _post_json(
        f"http://localhost:{frontend_port}/v1/completions",
        {
            "model": TEST_MODEL,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
            "logprobs": 0,
            "nvext": {"extra_fields": ["completion_token_ids", "engine_data"]},
        },
    )


@pytest.mark.vllm
@pytest.mark.profiled_vram_gib(6.9)
@pytest.mark.requested_vllm_kv_cache_bytes(331_801_000)
@pytest.mark.timeout(600)
def test_vllm_token_in_token_out(start_vllm_tito_services: int) -> None:
    frontend_port = start_vllm_tito_services
    text_response = _request_vllm_completion(
        frontend_port=frontend_port,
        prompt=TEST_PROMPT,
    )
    text_nvext = _get_nvext(text_response)
    text_completion_ids = text_nvext.get("completion_token_ids")
    assert (
        isinstance(text_completion_ids, list)
        and text_completion_ids
        and all(isinstance(token_id, int) for token_id in text_completion_ids)
    ), text_response

    text_engine_data = text_nvext.get("engine_data")
    assert isinstance(text_engine_data, dict), text_response
    prompt_token_ids = text_engine_data.get("prompt_token_ids")
    assert (
        isinstance(prompt_token_ids, list)
        and prompt_token_ids
        and all(isinstance(token_id, int) for token_id in prompt_token_ids)
    ), text_response
    assert (text_response.get("usage") or {}).get("prompt_tokens") == len(
        prompt_token_ids
    ), text_response

    token_response = _request_vllm_completion(
        frontend_port=frontend_port,
        prompt=prompt_token_ids,
    )
    token_nvext = _get_nvext(token_response)
    token_completion_ids = token_nvext.get("completion_token_ids")
    assert token_completion_ids == text_completion_ids, {
        "text_completion_ids": text_completion_ids,
        "token_completion_ids": token_completion_ids,
    }
    assert (token_response.get("usage") or {}).get("prompt_tokens") == len(
        prompt_token_ids
    ), token_response

    token_engine_data = token_nvext.get("engine_data")
    assert isinstance(token_engine_data, dict), token_response
    assert (
        token_engine_data.get("completion_token_ids") == token_completion_ids
    ), token_response
    rl_logprobs = token_engine_data.get("completion_logprobs")
    assert (
        isinstance(rl_logprobs, list)
        and len(rl_logprobs) == len(token_completion_ids)
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in rl_logprobs
        )
    ), token_response

    choices = token_response.get("choices") or []
    assert choices and isinstance(choices[0], dict), token_response
    openai_logprobs = (choices[0].get("logprobs") or {}).get("token_logprobs")
    assert (
        isinstance(openai_logprobs, list)
        and len(openai_logprobs) == len(rl_logprobs)
        and all(
            left is not None
            and math.isclose(left, right, rel_tol=0.0, abs_tol=LOGPROB_ATOL)
            for left, right in zip(openai_logprobs, rl_logprobs, strict=True)
        )
    ), {
        "openai_logprobs": openai_logprobs,
        "rl_logprobs": rl_logprobs,
    }


def _request_sglang_generation_events(*, frontend_port: int) -> list[dict[str, Any]]:
    with send_request(
        f"http://localhost:{frontend_port}/generate",
        {
            "input_ids": SGLANG_INPUT_IDS,
            "sampling_params": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "n": 1,
            },
            "stream": True,
            "return_logprob": True,
            "top_logprobs_num": 0,
        },
        timeout=120,
        stream=True,
    ) as response:
        assert response.status_code == 200, response.text
        events: list[dict[str, Any]] = []
        stream_finished = False
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                stream_finished = True
                continue
            event = json.loads(data)
            assert isinstance(event, dict), event
            assert not event.get("error"), event
            events.append(event)

    assert stream_finished, events
    assert events, "SGLang /generate returned no SSE data events"
    return events


@pytest.mark.sglang
@pytest.mark.profiled_vram_gib(3.7)
@pytest.mark.requested_sglang_kv_tokens(2048)
@pytest.mark.timeout(600)
def test_sglang_token_in_token_out(start_sglang_tito_services: int) -> None:
    events = _request_sglang_generation_events(frontend_port=start_sglang_tito_services)
    all_output_ids: list[int] = []
    for event in events:
        output_ids = event.get("output_ids")
        assert isinstance(output_ids, list) and all(
            isinstance(token_id, int) for token_id in output_ids
        ), event

        meta_info = event.get("meta_info")
        assert isinstance(meta_info, dict), event
        if not output_ids:
            continue

        output_logprobs = meta_info.get("output_token_logprobs")
        assert isinstance(output_logprobs, list), event
        assert len(output_logprobs) == len(output_ids), event
        for token_id, logprob_entry in zip(output_ids, output_logprobs, strict=True):
            assert isinstance(logprob_entry, list) and len(logprob_entry) >= 2, event
            logprob, logprob_token_id = logprob_entry[:2]
            assert isinstance(logprob, (int, float)) and math.isfinite(logprob), event
            assert logprob_token_id == token_id, event

        all_output_ids.extend(output_ids)

    assert all_output_ids, events
    final = events[-1]
    meta_info = final.get("meta_info")
    assert isinstance(meta_info, dict), final
    assert meta_info.get("prompt_tokens") == len(SGLANG_INPUT_IDS), final
    assert meta_info.get("completion_tokens") == len(all_output_ids), {
        "completion_tokens": meta_info.get("completion_tokens"),
        "all_output_ids": all_output_ids,
    }
