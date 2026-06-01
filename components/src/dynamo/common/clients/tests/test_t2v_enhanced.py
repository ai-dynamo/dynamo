# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for T2VEnhancedClient.

The tests use ``aioresponses`` to mock both the LLM and T2V HTTP endpoints
so the chained orchestration logic can be exercised without launching any
Dynamo workers.
"""

from __future__ import annotations

import pytest
from aioresponses import aioresponses

from dynamo.common.clients import (
    EnhancerMode,
    T2VEnhancedClient,
    T2VEnhancedClientError,
)


LLM_URL = "http://llm.test"
T2V_URL = "http://t2v.test"
LLM_MODEL = "Qwen/Qwen3-0.6B"
T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _llm_ok_payload(rewritten: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": rewritten}}
        ],
    }


def _t2v_ok_payload(url: str) -> dict:
    return {"data": [{"url": url}]}


@pytest.mark.asyncio
async def test_generate_runs_enhancer_then_t2v():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{LLM_URL}/v1/chat/completions",
                payload=_llm_ok_payload("a cinematic bear by a campfire"),
            )
            mock.post(
                f"{T2V_URL}/v1/videos",
                payload=_t2v_ok_payload("file:///tmp/result.mp4"),
            )

            result = await client.generate("a bear by a campfire")

            assert result.url == "file:///tmp/result.mp4"
            assert result.rewritten_prompt == "a cinematic bear by a campfire"
            assert result.timings.enhance_ms > 0
            assert result.timings.t2v_ms > 0
            assert result.timings.e2e_ms >= result.timings.enhance_ms

            # T2V was called with the rewritten prompt, not the user input.
            requests = mock.requests
            t2v_calls = [k for k in requests if k[1].path == "/v1/videos"]
            assert t2v_calls, "expected /v1/videos call"
            body = requests[t2v_calls[0]][0].kwargs["json"]
            assert body["prompt"] == "a cinematic bear by a campfire"


@pytest.mark.asyncio
async def test_enhancer_off_skips_llm_call():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{T2V_URL}/v1/videos",
                payload=_t2v_ok_payload("file:///tmp/skipped.mp4"),
            )

            result = await client.generate(
                "raw user prompt", enhancer=EnhancerMode.OFF
            )

            assert result.url == "file:///tmp/skipped.mp4"
            assert result.rewritten_prompt is None
            assert result.timings.enhance_ms == 0

            calls = list(mock.requests.keys())
            paths = [c[1].path for c in calls]
            assert "/v1/chat/completions" not in paths


@pytest.mark.asyncio
async def test_enhancer_off_string_alias():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{T2V_URL}/v1/videos",
                payload=_t2v_ok_payload("file:///tmp/off.mp4"),
            )

            result = await client.generate("anything", enhancer="off")
            assert result.rewritten_prompt is None


@pytest.mark.asyncio
async def test_default_enhancer_off_applies_when_no_override():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
        default_enhancer=EnhancerMode.OFF,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{T2V_URL}/v1/videos",
                payload=_t2v_ok_payload("file:///tmp/default-off.mp4"),
            )

            result = await client.generate("anything")
            assert result.rewritten_prompt is None


@pytest.mark.asyncio
async def test_llm_5xx_raises_client_error():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{LLM_URL}/v1/chat/completions",
                status=500,
                body="internal error",
            )

            with pytest.raises(T2VEnhancedClientError) as exc_info:
                await client.generate("anything")

            assert "HTTP 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_t2v_5xx_raises_client_error():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{LLM_URL}/v1/chat/completions",
                payload=_llm_ok_payload("rewritten"),
            )
            mock.post(f"{T2V_URL}/v1/videos", status=502, body="bad gateway")

            with pytest.raises(T2VEnhancedClientError) as exc_info:
                await client.generate("anything")

            assert "HTTP 502" in str(exc_info.value)


@pytest.mark.asyncio
async def test_t2v_extra_kwargs_are_forwarded():
    async with T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    ) as client:
        with aioresponses() as mock:
            mock.post(
                f"{T2V_URL}/v1/videos",
                payload=_t2v_ok_payload("file:///tmp/forward.mp4"),
            )
            await client.generate(
                "anything",
                enhancer=EnhancerMode.OFF,
                size="832x480",
                num_inference_steps=50,
                num_frames=33,
            )

            requests = mock.requests
            t2v_calls = [k for k in requests if k[1].path == "/v1/videos"]
            body = requests[t2v_calls[0]][0].kwargs["json"]
            assert body["size"] == "832x480"
            assert body["num_inference_steps"] == 50
            assert body["num_frames"] == 33


@pytest.mark.asyncio
async def test_client_must_be_entered_when_session_not_provided():
    client = T2VEnhancedClient(
        llm_url=LLM_URL,
        t2v_url=T2V_URL,
        llm_model=LLM_MODEL,
        t2v_model=T2V_MODEL,
    )
    with pytest.raises(T2VEnhancedClientError):
        await client.generate("anything", enhancer=EnhancerMode.OFF)
