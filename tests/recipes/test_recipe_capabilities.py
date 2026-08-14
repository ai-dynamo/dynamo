# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability checks against an already-deployed recipe.

Each of these was verified to be reachable on a real deployment before being
written, so a failure means a regression rather than an unsupported feature.

Deliberately NOT here:
  * logprobs -- measured to come back ``null`` on a Kimi-K2.5 + Eagle3
    deployment. A test would pass while asserting nothing, because
    ``logprobs`` has no ``skip_serializing_if`` and every structural assertion
    in payloads.py is guarded behind ``is not None``.
  * ``n>1`` -- rejected with HTTP 400 unless the worker sets
    ``TLLM_ALLOW_N_GREEDY_DECODING``.
  * embeddings / multimodal -- not served by an LLM-only, text-only worker.
  * structured output (``response_format: json_schema``) -- ACCEPTED with HTTP
    200 but NOT enforced: the reply comes back as markdown prose, not schema-
    conforming JSON. The recipe sets no ``guided_decoding_backend`` and does not
    enable structural tags. Accepting a request is not the same as honouring it,
    so a status-code check is not evidence of support.
"""

import json

import pytest

from tests.recipes.helpers import assert_natural_language
from tests.recipes.helpers import post as _post
from tests.recipes.helpers import stream as _stream

pytestmark = [
    pytest.mark.endpoint_only,
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.gpu_0,
]


def test_reasoning_content_is_emitted(attached_endpoint):
    """A reasoning-parser deployment separates thinking from the answer.

    ``max_tokens`` is generous on purpose: the parser spends its first tokens
    inside ``<think>``, and a short budget never closes the block, leaving
    ``content`` empty through no fault of the deployment.
    """
    body = _post(
        attached_endpoint,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 512,
        },
    )
    message = body["choices"][0]["message"]
    assert message.get("reasoning_content"), (
        "no reasoning_content: the deployment is not running a reasoning parser, "
        "or the model did not emit a think block"
    )
    assert message.get("content"), (
        "reasoning_content present but content empty -- the think block never "
        "closed within max_tokens"
    )
    # Presence is not enough: a replica with a stuck decoder fills
    # reasoning_content with one repeated token, which is truthy.
    assert_natural_language(message["reasoning_content"], "reasoning_content")
    assert_natural_language(message["content"], "content")


def test_thinking_false_suppresses_reasoning(attached_endpoint):
    """``chat_template_args {"thinking": false}`` turns reasoning off per request.

    This is the cheap escape for any payload that asserts on ``content``: it
    avoids both a large token budget and a deployment-level flag change.
    """
    body = _post(
        attached_endpoint,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 64,
            "chat_template_args": {"thinking": False},
        },
    )
    message = body["choices"][0]["message"]
    assert not message.get(
        "reasoning_content"
    ), "thinking:false did not suppress reasoning_content"
    assert message.get("content"), "no content with reasoning suppressed"
    assert_natural_language(message["content"], "content")


def test_streaming_emits_chunks(attached_endpoint):
    """SSE streaming produces well-formed chunks and a terminal finish_reason."""
    chunks = _stream(
        attached_endpoint,
        {
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 64,
            "stream": True,
        },
    )
    assert chunks, "no SSE chunks received"
    assert any(
        c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")
    ), "stream never reported a finish_reason"
    # Reassemble the stream: a finish_reason alone says the stream terminated,
    # not that it carried anything meaningful. A stuck decoder still terminates
    # -- on `length` -- after emitting one repeated token.
    streamed = "".join(
        (delta.get("reasoning_content") or "") + (delta.get("content") or "")
        for chunk in chunks
        for delta in [(chunk.get("choices") or [{}])[0].get("delta") or {}]
    )
    assert_natural_language(streamed, "streamed text")


def test_continuous_usage_stats_on_every_chunk(attached_endpoint):
    """``continuous_usage_stats`` puts usage on each chunk, not just the last.

    Note it requires ``include_usage`` alongside it -- without that the frontend
    rejects the request with 400 ("missing field `include_usage`").
    """
    chunks = _stream(
        attached_endpoint,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        },
    )
    assert chunks, "no SSE chunks received"
    without = [i for i, c in enumerate(chunks) if not c.get("usage")]
    assert not without, (
        f"{len(without)}/{len(chunks)} chunks carried no usage "
        f"(indices {without[:5]}) despite continuous_usage_stats"
    )


def test_router_reports_the_worker_that_served(attached_endpoint):
    """``nvext.extra_fields=["worker_id"]`` surfaces the routing decision.

    The only KV-router assertion observable from a response alone -- everything
    else about routing needs per-worker visibility and is topology_dependent.
    With a single worker this proves the plumbing, not the choice; it becomes a
    real routing assertion at replicas >= 2.
    """
    body = _post(
        attached_endpoint,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "nvext": {"extra_fields": ["worker_id"]},
        },
    )
    assert "worker_id" in json.dumps(body), (
        "no worker_id in the response: nvext extra_fields is disabled, or the "
        "frontend is not running a KV router"
    )
