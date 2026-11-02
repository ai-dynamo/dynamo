# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read advertisement follows bounded CPU preparation, without engine inference."""

import asyncio
from types import SimpleNamespace

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm import decision_native_registry as registry
from dynamo.vllm.tests.decision_head_test_support import monitor
from dynamo.vllm.tests.decision_read_test_support import Engine, Tokenizer

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def _config():
    return SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        enable_rl=False,
        route_to_encoder=False,
        custom_encoder_class=None,
        enable_multimodal=False,
        realtime=False,
    )


class TailTokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        if len(text) > 1 and text.startswith("x"):
            return [4096 + ord("x"), *super().encode(text[1:], **kwargs)]
        return super().encode(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        return [*super().apply_chat_template(messages, **kwargs), 4096 + ord("x")]


@pytest.mark.asyncio
@pytest.mark.parametrize("diffusion", [False, True])
@pytest.mark.parametrize("fault", [None, "salt", "tail"])
async def test_read_startup_prepares_real_pinned_strategy_without_engine_calls(
    native_reads, monkeypatch, diffusion, fault
):
    engine = Engine(diffusion=diffusion)
    if fault == "tail":
        tokenizer = TailTokenizer()
        engine.renderer.get_tokenizer = lambda: tokenizer
    original = native_reads.online_renderer.preprocess_chat

    async def render(self, *args, **kwargs):
        conversation, inputs = await original(self, *args, **kwargs)
        if fault == "salt":
            inputs[0].pop("cache_salt")
        elif fault == "tail":
            inputs[0]["prompt_token_ids"][-1] = 123
        return conversation, inputs

    monkeypatch.setattr(native_reads.online_renderer, "preprocess_chat", render)
    surface = await registry.initialize_native_decisions(
        "read", engine, engine.model_config, _config(), abort_monitor=monitor
    )
    assert engine.rendered
    assert (surface.spec is not None) == (fault is None)
    assert (surface.failed_profile is None) == (fault is None)
    assert surface.maybe_handle({"input": "ordinary"}, object()) is None
    assert engine.calls == engine.admissions == engine.aborted == []


@pytest.mark.asyncio
async def test_read_preparation_finishes_before_advertisement(
    native_reads, monkeypatch
):
    engine = Engine()
    entered, released = asyncio.Event(), asyncio.Event()
    original = native_reads.online_renderer.preprocess_chat

    async def render(self, *args, **kwargs):
        entered.set()
        await released.wait()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(native_reads.online_renderer, "preprocess_chat", render)
    pending = asyncio.create_task(
        registry.initialize_native_decisions(
            "read", engine, engine.model_config, _config(), abort_monitor=monitor
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 0.5)
        assert not pending.done() and engine.calls == []
        released.set()
        surface = await pending
        assert surface.spec is not None
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
