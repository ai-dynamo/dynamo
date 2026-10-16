# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic test (xdist-safe via dynamic ports).
# GPU Requirement: gpu_0 (CPU-only, mocker does not use GPU)
#
# Acceptance test for per-chunk streaming token usage, ported from the manual
# Qwen3.6-27B acceptance script (dynamo-issues/test_qwen36_messages_stream_usage.py)
# into a hermetic mocker-backed pytest.
#
# Goal (regression guard for the /v1/messages per-chunk usage fix in
# lib/llm/src/protocols/anthropic/stream_converter.rs): in stream mode, 100% of
# the token-bearing chunks must carry a token-usage triple — input, output and
# total tokens — not only the final chunk.
#
#   /v1/messages (Anthropic SSE)
#     Anthropic's native protocol reports usage only on `message_start`
#     (input_tokens) and the terminal `message_delta` (output_tokens). The
#     Dynamo frontend additionally stamps the full triple onto every
#     `content_block_delta`. token chunk = a `content_block_delta` event.
#
#   /v1/chat/completions + {"include_usage": true, "continuous_usage_stats": true}
#     Every `chat.completion.chunk` repeats usage.{prompt,completion,total}_tokens.
#     token chunk = a chunk whose delta carries content/reasoning_content.
#
# Acceptance criterion (per endpoint): usage_chunks == token_chunks (100%), and
# each usage exposes input, output and total tokens (all > 0), where
# total = input + output for the Anthropic shape.

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN
MAX_TOKENS = 64
PROMPT = "In one short sentence, what is the capital of France?"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,  # Mocker is CPU-only (no GPU required)
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
]


@pytest.fixture(scope="function")
def anthropic_frontend_with_mocker(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """Start an HTTP frontend with the Anthropic Messages API enabled, backed by
    a mocker worker.

    The stock ``start_services_with_mocker`` fixture does not enable
    ``/v1/messages``; this variant sets ``DYN_ENABLE_ANTHROPIC_API`` on the
    frontend so the Anthropic SSE path is exercised. Function-scoped for
    xdist-parallel execution — each test gets its own frontend + mocker on
    unique ports.

    Yields:
        frontend_port: Port where the frontend is listening.
    """
    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        terminate_all_matching_process_names=False,
        extra_env={"DYN_ENABLE_ANTHROPIC_API": "1"},
    ):
        with MockerWorkerProcess(request, TEST_MODEL, frontend_port, system_port):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )
            logger.info(
                "Anthropic-enabled frontend + mocker ready on port %s", frontend_port
            )
            yield frontend_port


def _iter_sse(frontend_port: int, path: str, payload: dict[str, Any]):
    """Yield decoded JSON objects from an SSE stream; skips the [DONE] sentinel."""
    url = f"http://localhost:{frontend_port}{path}"
    headers = {"Content-Type": "application/json"}
    with requests.post(
        url, json=payload, headers=headers, stream=True, timeout=180
    ) as r:
        assert (
            r.status_code == 200
        ), f"{path} returned HTTP {r.status_code}: {r.text[:500]}"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _triple_ok(u: dict) -> bool:
    """True if a usage dict exposes input, output and total, all > 0."""
    if not isinstance(u, dict):
        return False
    return all(
        isinstance(u.get(k), int) and u[k] > 0
        for k in ("input_tokens", "output_tokens", "total_tokens")
    )


def _check_messages(frontend_port: int) -> tuple[int, int, list]:
    """Return (token_chunks, chunks_with_usage_triple, sample_usages) for /v1/messages.

    A token chunk is a ``content_block_delta``. Its usage triple must ride on
    that same chunk (the Dynamo per-chunk usage extension).
    """
    payload = {
        "model": TEST_MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    token_chunks = 0
    with_usage = 0
    samples: list = []
    input_seen = None
    for obj in _iter_sse(frontend_port, "/v1/messages", payload):
        etype = obj.get("type")
        if etype == "message_start":
            input_seen = ((obj.get("message") or {}).get("usage") or {}).get(
                "input_tokens"
            )
        elif etype == "content_block_delta":
            token_chunks += 1
            u = obj.get("usage") or (obj.get("delta") or {}).get("usage")
            triple = None
            if isinstance(u, dict) and "input_tokens" in u and "output_tokens" in u:
                in_tokens = u.get("input_tokens", input_seen)
                out_tokens = u.get("output_tokens")
                triple = {
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": u.get(
                        "total_tokens", (in_tokens or 0) + (out_tokens or 0)
                    ),
                }
            if triple and _triple_ok(triple):
                with_usage += 1
                if len(samples) < 3:
                    samples.append(triple)
    return token_chunks, with_usage, samples


def _check_chat(frontend_port: int) -> tuple[int, int, list]:
    """Return (token_chunks, chunks_with_usage_triple, sample_usages) for chat.

    ``continuous_usage_stats`` makes every chat.completion.chunk repeat usage.
    A token chunk is any chunk carrying generated content in its delta.
    """
    payload = {
        "model": TEST_MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "messages": [{"role": "user", "content": PROMPT}],
    }
    token_chunks = 0
    with_usage = 0
    samples: list = []
    for obj in _iter_sse(frontend_port, "/v1/chat/completions", payload):
        choices = obj.get("choices") or []
        delta = (choices[0].get("delta") if choices else {}) or {}
        is_token_chunk = bool(delta.get("content") or delta.get("reasoning_content"))
        if not is_token_chunk:
            continue
        token_chunks += 1
        u = obj.get("usage") or {}
        triple = {
            "input_tokens": u.get("prompt_tokens"),
            "output_tokens": u.get("completion_tokens"),
            "total_tokens": u.get("total_tokens"),
        }
        if _triple_ok(triple):
            with_usage += 1
            if len(samples) < 3:
                samples.append(triple)
    return token_chunks, with_usage, samples


def test_messages_stream_carries_usage_on_every_chunk(
    anthropic_frontend_with_mocker,
) -> None:
    """TC1 — /v1/messages: every ``content_block_delta`` carries the usage triple.

    This is the direct regression guard for the per-chunk usage fix: without it,
    Anthropic SSE only reports usage on ``message_start`` / ``message_delta`` and
    a proxy reading the stream for live per-token accounting gets nothing until
    the stream ends.
    """
    frontend_port = anthropic_frontend_with_mocker
    total, ok, samples = _check_messages(frontend_port)

    assert total > 0, "no content_block_delta (token) chunks were streamed"
    assert ok == total, (
        f"/v1/messages: only {ok}/{total} content_block_delta chunks carry the "
        f"usage triple — requirement is 100% (every token chunk). "
        f"sample usages: {samples[:3]}"
    )
    logger.info(
        "TC1 /v1/messages: %d/%d token chunks carry usage triple; e.g. %s",
        ok,
        total,
        samples[:2],
    )


def test_chat_stream_continuous_usage_on_every_chunk(
    anthropic_frontend_with_mocker,
) -> None:
    """TC2 — /v1/chat/completions with ``continuous_usage_stats``: every content
    chunk repeats the usage triple.

    Baseline that the Anthropic per-chunk behavior mirrors; kept in the same
    file so both wire shapes are asserted against one running frontend.
    """
    frontend_port = anthropic_frontend_with_mocker
    total, ok, samples = _check_chat(frontend_port)

    assert total > 0, "no content-bearing chat.completion.chunk chunks were streamed"
    assert ok == total, (
        f"/v1/chat/completions: only {ok}/{total} content chunks carry the usage "
        f"triple — requirement is 100% (continuous_usage_stats). "
        f"sample usages: {samples[:3]}"
    )
    logger.info(
        "TC2 /v1/chat/completions: %d/%d token chunks carry usage triple; e.g. %s",
        ok,
        total,
        samples[:2],
    )
