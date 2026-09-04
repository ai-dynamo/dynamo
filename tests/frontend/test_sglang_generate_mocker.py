# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for native SGLang streaming `/generate` through the mocker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.client import send_request
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import ServicePorts
from tests.utils.sglang_generate import assert_native_stream, stream_generate

TEST_MODEL = QWEN
INPUT_IDS = [11, 12, 13]
OUTPUT_IDS = [101, 202, 303]
REQUEST_ID = "native-replay"

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.e2e,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.sglang,
    pytest.mark.model(TEST_MODEL),
]


@pytest.fixture(scope="function")
def sglang_generate_mocker(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    predownload_tokenizers: object,
    tmp_path: Path,
) -> Generator[int, None, None]:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]
    replay_trace = tmp_path / "sglang-response-replay.jsonl"
    replay_trace.write_text(
        json.dumps(
            {
                "request_id": REQUEST_ID,
                "output_length": len(OUTPUT_IDS),
                "output_token_ids": OUTPUT_IDS,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env={"DYN_SGLANG_ENABLE_GENERATE": "1"},
        terminate_all_matching_process_names=False,
    ):
        with MockerWorkerProcess(
            request,
            TEST_MODEL,
            frontend_port,
            system_port,
            extra_args=[
                "--engine-type",
                "sglang",
                "--sglang-generate",
                "--response-replay-trace-path",
                str(replay_trace),
            ],
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port,
                model=TEST_MODEL,
            )
            yield frontend_port


@pytest.mark.timeout(120)
def test_native_generate_replays_incremental_ids_and_keeps_openai_canonical(
    sglang_generate_mocker: int,
) -> None:
    frontend_port = sglang_generate_mocker
    events = stream_generate(
        frontend_port=frontend_port,
        body={
            "rid": REQUEST_ID,
            "input_ids": INPUT_IDS,
            "sampling_params": {
                "max_new_tokens": len(OUTPUT_IDS),
                "ignore_eos": True,
                "n": 1,
            },
            "return_logprob": True,
            "top_logprobs_num": 2,
            "logprob_start_len": 1,
        },
        timeout=60,
    )

    # The mocker owes callers the same stream contract as a real SGLang worker.
    replayed = assert_native_stream(events, prompt_tokens=len(INPUT_IDS))
    assert replayed == OUTPUT_IDS

    completion_tokens = 0
    for event in events:
        meta_info = event["meta_info"]
        completion_tokens += len(event["output_ids"])
        assert meta_info["id"] == REQUEST_ID, event
        assert meta_info["completion_tokens"] == completion_tokens, event
        top_logprobs = meta_info["output_top_logprobs"]
        assert len(top_logprobs) == len(event["output_ids"]), event
        assert all(len(candidates) == 2 for candidates in top_logprobs), event

    terminal = [event for event in events if event["meta_info"]["finish_reason"]]
    assert terminal == [events[-1]]
    terminal_meta = events[-1]["meta_info"]
    assert terminal_meta["finish_reason"] == {"type": "length"}
    assert terminal_meta["input_token_logprobs"][0] == [None, INPUT_IDS[1], None]
    assert terminal_meta["input_top_logprobs"][0] is None

    # OpenAI requests carry no sglang_tito payload, so they skip the adapter.
    completion = send_request(
        f"http://localhost:{frontend_port}/v1/completions",
        {"model": TEST_MODEL, "prompt": "ping", "max_tokens": 2},
        timeout=60,
    )
    assert completion.status_code == 200, completion.text
    completion_body = completion.json()
    assert completion_body.get("object") == "text_completion"
    assert completion_body.get("choices")
    assert "sglang_response" not in completion.text
