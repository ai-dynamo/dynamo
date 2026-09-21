# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed mock workers leave response decoding to the frontend."""

import asyncio
import json

import aiohttp
import pytest
from dynamo.prometheus_names import frontend_perf, name_prefix
from dynamo.runtime import DistributedRuntime
from transformers import AutoTokenizer

from tests.frontend.conftest import (
    MockerWorkerProcess,
    wait_for_http_completions_ready,
)
from tests.utils.constants import QWEN
from tests.utils.prometheus import find_metric_samples

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(QWEN),
    pytest.mark.timeout(120),
]


async def _detokenize_count(
    session: aiohttp.ClientSession, frontend_port: int
) -> float:
    async with session.get(f"http://localhost:{frontend_port}/metrics") as response:
        response.raise_for_status()
        content = await response.text()
    samples = find_metric_samples(
        content,
        f"{name_prefix.FRONTEND}_{frontend_perf.DETOKENIZE_TOKEN_COUNT}",
    )
    assert samples, "frontend detokenization counter is missing"
    return sum(samples)


@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_mocker_returns_tokens_and_frontend_decodes(
    request,
    start_services_with_http,
    predownload_tokenizers,
    discovery_backend,
    request_plane,
    event_plane,
    tmp_path,
):
    frontend_port, system_port = start_services_with_http
    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    prompt_ids = tokenizer.encode("Hello", add_special_tokens=False)
    output_ids = tokenizer.encode(
        " Mock workers return tokens: café 🌍.", add_special_tokens=False
    )
    expected_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    replay_id = "detokenization"
    trace = tmp_path / "response-replay.jsonl"
    trace.write_text(
        json.dumps(
            {
                "request_id": replay_id,
                "output_length": len(output_ids),
                "output_token_ids": output_ids,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    endpoint = "detokenization.backend.generate"
    annotations = [f"output_replay_id:{replay_id}"]

    with MockerWorkerProcess(
        request,
        QWEN,
        frontend_port,
        system_port,
        extra_args=[
            "--endpoint",
            f"dyn://{endpoint}",
            "--response-replay-trace-path",
            str(trace),
        ],
    ):
        wait_for_http_completions_ready(frontend_port=frontend_port, model=QWEN)
        runtime = DistributedRuntime(
            asyncio.get_running_loop(),
            discovery_backend=discovery_backend,
            request_plane=request_plane,
            event_plane=event_plane,
        )
        try:
            client = await runtime.endpoint(endpoint).client()
            await asyncio.wait_for(client.wait_for_instances(), timeout=30)
            stream = await client.round_robin(
                {
                    "model": QWEN,
                    "token_ids": prompt_ids,
                    "stop_conditions": {
                        "max_tokens": len(output_ids),
                        "ignore_eos": True,
                    },
                    "sampling_options": {},
                    "output_options": {},
                    "eos_token_ids": [],
                    "annotations": annotations,
                },
                annotated=False,
            )
            chunks = [chunk async for chunk in stream]
            assert [t for c in chunks for t in c["token_ids"]] == output_ids
            assert all(c.get("text") is None for c in chunks)
            assert all(c.get("tokens") is None for c in chunks)
            assert chunks[-1]["finish_reason"] == "length"
        finally:
            runtime.shutdown()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            for streaming in (False, True):
                before = await _detokenize_count(session, frontend_port)
                async with session.post(
                    f"http://localhost:{frontend_port}/v1/completions",
                    json={
                        "model": QWEN,
                        "prompt": prompt_ids,
                        "max_tokens": len(output_ids),
                        "stream": streaming,
                        "ignore_eos": True,
                        "nvext": {"annotations": annotations},
                        **(
                            {"stream_options": {"include_usage": True}}
                            if streaming
                            else {}
                        ),
                    },
                ) as response:
                    assert response.status == 200, await response.text()
                    if streaming:
                        chunks = [
                            json.loads(line[6:])
                            async for line in response.content
                            if line.startswith(b"data: ")
                            and line.strip() != b"data: [DONE]"
                        ]
                    else:
                        chunks = [await response.json()]
                choices = [choice for chunk in chunks for choice in chunk["choices"]]
                assert "".join(choice["text"] for choice in choices) == expected_text
                assert [
                    c["finish_reason"] for c in choices if c.get("finish_reason")
                ] == ["length"]
                usage = next(c["usage"] for c in reversed(chunks) if c.get("usage"))
                assert usage["prompt_tokens"] == len(prompt_ids)
                assert usage["completion_tokens"] == len(output_ids)
                for _ in range(50):
                    if await _detokenize_count(session, frontend_port) - before >= len(
                        output_ids
                    ):
                        break
                    await asyncio.sleep(0.1)
                assert await _detokenize_count(session, frontend_port) - before >= len(
                    output_ids
                )
