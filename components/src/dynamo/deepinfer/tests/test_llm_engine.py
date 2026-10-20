# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from dynamo.deepinfer.llm_engine import DeepInferLLMEngine, build_sampling_params
from dynamo.deepinfer.mock_engine import MockDeepInfer, mock_response
from dynamo.llm.exceptions import InvalidArgument

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def model(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello from the DeepInfer mock."],
        trainers.BpeTrainer(
            vocab_size=256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    return tmp_path


def test_sampling_bounds_and_native_names():
    assert build_sampling_params(
        {
            "token_ids": [1, 2],
            "sampling_options": {"temperature": 0.2, "seed": 7, "n": None},
            "stop_conditions": {"max_tokens": 30},
        },
        10,
    ) == {"temperature": 0.2, "seq_seed": 7, "max_tokens": 8, "ignore_eos": False}
    for request in (
        {"token_ids": [1], "sampling_options": {"n": 2}},
        {"token_ids": [1], "sampling_options": {"guided_decoding": {"json": {}}}},
        {"token_ids": [1], "multi_modal_data": {"image": "unsupported"}},
        {"token_ids": [1], "stop_conditions": {"max_tokens": 0}},
        {"token_ids": list(range(10))},
    ):
        with pytest.raises(InvalidArgument):
            build_sampling_params(request, 10)


def test_fixture_selection_uses_latest_turn():
    prompt = "<｜User｜>DEEPINFER_TOOL<｜Assistant｜>previous<｜User｜>hello"
    assert mock_response(prompt) == "Hello from the DeepInfer mock."


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_shared_owner_demux_limits_and_cancel(model, monkeypatch):
    owners = set()
    for method in ("add_request", "step", "abort_request"):
        original = getattr(MockDeepInfer, method)

        def record(self, *args, _original=original, **kwargs):
            owners.add(threading.get_ident())
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(MockDeepInfer, method, record)
    engine, worker = await DeepInferLLMEngine.from_args(
        ["--model", str(model), "--mock-token-delay", "0"]
    )
    assert worker.tool_call_parser == worker.reasoning_parser == "deepseek_v41"
    await engine.start(1)
    try:

        def context(rid):
            return SimpleNamespace(id=lambda: rid)

        canceled = engine.generate({"token_ids": [1]}, context("canceled"))
        first = await anext(canceled)
        assert len(first["token_ids"]) == 1
        await engine.abort(context("canceled"))
        async for _ in canceled:
            pass

        async def collect(rid, maximum):
            return [
                chunk
                async for chunk in engine.generate(
                    {
                        "token_ids": [1, 2],
                        "stop_conditions": {"max_tokens": maximum},
                    },
                    context(rid),
                )
            ]

        full, limited = await asyncio.gather(
            collect("full", 100), collect("limited", 2)
        )
        assert full[-1]["finish_reason"] == "stop"
        assert limited[-1]["finish_reason"] == "length"
        assert len(limited) == limited[-1]["completion_usage"]["completion_tokens"] == 2
        assert (
            engine.engine.tokenizer.decode([c["token_ids"][0] for c in full])
            == "Hello from the DeepInfer mock."
        )
        assert not engine.streams
        assert not await engine._on_owner(lambda: engine.engine.requests)
        assert len(owners) == 1 and threading.get_ident() not in owners
    finally:
        await engine.cleanup()
        await engine.cleanup()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_driver_failure_reaches_request_and_cleanup(model, monkeypatch):
    def broken_step(self):
        raise RuntimeError("engine failed")

    monkeypatch.setattr(MockDeepInfer, "step", broken_step)
    engine, _ = await DeepInferLLMEngine.from_args(["--model", str(model)])
    await engine.start(1)
    stream = engine.generate({"token_ids": [1]}, SimpleNamespace(id=lambda: "failed"))
    with pytest.raises(RuntimeError, match="engine failed"):
        await anext(stream)
    with pytest.raises(RuntimeError, match="engine failed"):
        await engine.cleanup()
    assert engine.executor is None
