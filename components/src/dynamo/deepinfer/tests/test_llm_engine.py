# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine contract tests; DeepInfer itself is replaced by a scripted fake."""

import asyncio
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from dynamo.deepinfer import llm_engine
from dynamo.deepinfer.llm_engine import DeepInferLLMEngine, read_bos_token_id
from dynamo.deepinfer.tests.fake_native import FakeEngine
from dynamo.llm.exceptions import EngineShutdown, InvalidArgument, Unknown

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

EOS = [2, 3]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        if kwargs.get("max_tokens", 1) < 1:
            raise ValueError("max_tokens must be >= 1")
        self.__dict__.update(kwargs)


class FakeContext:
    def __init__(self, context_id="r1"):
        self._id = context_id

    def id(self):
        return self._id


@pytest.fixture
def native(monkeypatch):
    """Install a fake ``deepinfer`` package whose LLM wraps a FakeEngine."""
    fake = FakeEngine()
    fake.construct_threads = set()
    llm_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            llm_kwargs.update(kwargs)
            fake.construct_threads.add(threading.current_thread().name)
            self.llm_engine = fake

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    module("deepinfer")
    module("deepinfer._C")
    module("deepinfer._C.engine", RequestOutput=SimpleNamespace(clone=fake.clone))
    module("deepinfer.entrypoints")
    module("deepinfer.entrypoints.llm", LLM=FakeLLM)
    module("deepinfer.sampling_params", SamplingParams=FakeSamplingParams)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unset")
    fake.llm_kwargs = llm_kwargs
    return fake


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"eos_token_id": EOS, "bos_token_id": 7})
    )
    return tmp_path


async def make_engine(model_dir, *extra):
    engine, worker_config = await DeepInferLLMEngine.from_args(
        [
            "--model",
            str(model_dir),
            "--expert-parallel-size",
            "2",
            "--context-parallel-size",
            "2",
            "--max-model-len",
            "64",
            *extra,
        ]
    )
    return engine, worker_config


@pytest.fixture
async def engine(native, model_dir):
    engine, _ = await make_engine(model_dir)
    await engine.start(worker_id=1)
    yield engine
    await engine.cleanup()


def request(max_tokens=3, **extra):
    return {
        "token_ids": [1, 2, 3, 4],
        "stop_conditions": {"max_tokens": max_tokens},
        **extra,
    }


async def wait_until(predicate, timeout=5):
    """The runner thread applies aborts asynchronously."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.005)


async def run(engine, req, context_id="r1"):
    return [chunk async for chunk in engine.generate(req, FakeContext(context_id))]


async def test_from_args_builds_worker_config(model_dir):
    _, worker_config = await make_engine(model_dir, "--served-model-name", "alias")

    assert worker_config.model_name == str(model_dir)
    assert worker_config.served_model_name == "alias"
    assert worker_config.enable_kv_routing is False
    assert worker_config.endpoint == "generate"


async def test_start_builds_the_native_engine_and_registers(native, model_dir):
    engine, _ = await make_engine(model_dir, "--device-offset", "4")

    config = await engine.start(worker_id=1)
    try:
        assert config.model == str(model_dir)
        assert config.llm.context_length == 64
        assert config.llm.data_parallel_size == 1
        assert native.llm_kwargs["model"] == str(model_dir)
        assert native.llm_kwargs["engine_rank"] == 0
        assert native.llm_kwargs["data_parallel_size"] == 1
        assert native.llm_kwargs["context_parallel_size"] == 2
        assert native.llm_kwargs["expert_parallel_size"] == 2
        assert sys.modules["os"].environ["CUDA_VISIBLE_DEVICES"] == "4,5"
    finally:
        await engine.cleanup()


async def test_native_engine_is_built_and_stepped_on_one_thread(engine, native):
    await run(engine, request(max_tokens=2))

    assert len(native.construct_threads) == 1
    assert native.construct_threads == native.step_threads
    assert threading.current_thread().name not in native.step_threads


async def test_generate_streams_tokens_then_a_terminal_chunk(engine):
    chunks = await run(engine, request(max_tokens=3))

    assert [t for c in chunks for t in c["token_ids"]] == [100, 101, 102]
    assert all(c["index"] == 0 for c in chunks)
    assert "finish_reason" not in chunks[0]
    assert chunks[-1]["finish_reason"] == "length"
    assert chunks[-1]["completion_usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }


async def test_prompt_is_sent_as_int32_token_ids(engine, native):
    await run(engine, request(max_tokens=1))

    request_id, params, prompt = native.added[0]
    assert request_id == "dyn-r1"
    assert prompt == [1, 2, 3, 4]
    assert params.max_tokens == 1
    assert not hasattr(params, "eos_id")  # DeepInfer fills its own


async def test_eos_token_is_passed_through_for_the_decoder_to_hide(engine, native):
    native.finish_reason = "stop"
    native.final_token = EOS[0]

    chunks = await run(engine, request(max_tokens=3))

    assert [t for c in chunks for t in c["token_ids"]] == [100, 101, EOS[0]]
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 3


async def test_stop_token_ids_from_the_frontend_are_accepted(engine, native):
    req = request(max_tokens=2)
    req["stop_conditions"]["stop_token_ids_hidden"] = [EOS[0], 9, 128009]

    chunks = await run(engine, req)

    assert chunks[-1]["finish_reason"] == "length"


async def test_logprobs_are_returned_when_requested(engine, native):
    native.top_k = 2

    chunks = await run(engine, request(max_tokens=1, output_options={"logprobs": 2}))

    assert chunks[0]["log_probs"] == [-0.5]
    assert [e["rank"] for e in chunks[0]["top_logprobs"][0]] == [1, 2]
    assert chunks[0]["top_logprobs"][0][0]["token_id"] == 100


async def test_logprobs_are_omitted_when_not_requested(engine):
    chunks = await run(engine, request(max_tokens=1))

    assert "log_probs" not in chunks[0]
    assert "top_logprobs" not in chunks[0]


async def test_partial_logprobs_are_dropped_not_misaligned(engine, native):
    native.logprob = None

    chunks = await run(engine, request(max_tokens=2, output_options={"logprobs": 0}))

    assert all("log_probs" not in c for c in chunks)


async def test_concurrent_requests_do_not_cross_streams(engine):
    first, second = await asyncio.gather(
        run(engine, request(max_tokens=2), "a"),
        run(engine, request(max_tokens=5), "b"),
    )

    assert sum(len(c["token_ids"]) for c in first) == 2
    assert sum(len(c["token_ids"]) for c in second) == 5


async def test_abort_ends_the_stream_with_a_cancelled_chunk(engine, native):
    context = FakeContext("r1")
    chunks = []
    async for chunk in engine.generate(request(max_tokens=10**9), context):
        chunks.append(chunk)
        if len(chunks) == 2:
            await engine.abort(context)

    assert chunks[-1]["finish_reason"] == "cancelled"
    assert chunks[-1]["token_ids"] == []
    await wait_until(lambda: native.aborted)
    assert native.aborted == ["dyn-r1"]


async def test_closing_the_stream_early_frees_the_native_request(engine, native):
    stream = engine.generate(request(max_tokens=10**9), FakeContext("r1"))
    await anext(stream)

    await stream.aclose()

    await wait_until(lambda: native.aborted)
    assert native.aborted == ["dyn-r1"]


async def test_abort_of_unknown_request_is_a_noop(engine, native):
    await engine.abort(FakeContext("nope"))
    await engine.abort(FakeContext(None))

    assert native.aborted == []


@pytest.mark.parametrize(
    "bad",
    [
        {"sampling_options": {"frequency_penalty": 0.5}},
        {"sampling_options": {"n": 2}},
    ],
)
async def test_unsupported_sampling_is_invalid_argument(engine, native, bad):
    with pytest.raises(InvalidArgument):
        await run(engine, {"token_ids": [1], **bad})

    assert native.added == []


async def test_empty_prompt_is_invalid_argument(engine):
    with pytest.raises(InvalidArgument, match="token_ids"):
        await run(engine, {"token_ids": []})


@pytest.mark.parametrize("token_ids", [[-1], [1000], [1, 2**40]])
async def test_out_of_vocabulary_prompt_never_reaches_the_engine(
    engine, native, token_ids
):
    with pytest.raises(InvalidArgument, match="token ids"):
        await run(engine, {"token_ids": token_ids})

    assert native.added == []


async def test_prompt_longer_than_the_context_never_reaches_the_engine(engine, native):
    with pytest.raises(InvalidArgument, match="context length"):
        await run(engine, {"token_ids": [1] * 65})  # --max-model-len is 64

    assert native.added == []


async def test_request_deepinfer_drops_fails_instead_of_hanging(engine, native):
    native.drop.add("dyn-bad")

    with pytest.raises(InvalidArgument, match="rejected"):
        await asyncio.wait_for(run(engine, request(), "bad"), 5)


async def test_prompt_too_long_is_invalid_argument(engine, native):
    native.finish_reason = "prompt_length_exceeded"

    with pytest.raises(InvalidArgument, match="context length"):
        await run(engine, request(max_tokens=1))


async def test_other_native_finish_reasons_are_errors(engine, native):
    native.finish_reason = "ve_error"

    with pytest.raises(Unknown, match="ve_error"):
        await run(engine, request(max_tokens=1))


async def test_native_abort_finish_reason_is_cancelled(engine, native):
    native.finish_reason = "abort"

    chunks = await run(engine, request(max_tokens=1))

    assert chunks[-1]["finish_reason"] == "cancelled"


async def test_rejected_native_request_surfaces_invalid_argument(engine, native):
    native.reject.add("dyn-bad")

    with pytest.raises(InvalidArgument, match="too long"):
        await run(engine, request(), "bad")


async def test_native_failure_fails_the_stream_then_the_engine(engine, native):
    native.fail_on_step = 1

    with pytest.raises(Unknown, match="cuda error"):
        await run(engine, request(max_tokens=5))
    with pytest.raises(EngineShutdown):
        await run(engine, request(), "r2")


async def test_generate_before_start_is_engine_shutdown(model_dir):
    engine, _ = await make_engine(model_dir)

    with pytest.raises(EngineShutdown):
        await run(engine, request())


async def test_cleanup_stops_the_runner_and_is_idempotent(native, model_dir):
    engine, _ = await make_engine(model_dir)
    await engine.start(worker_id=1)

    await engine.cleanup()
    await engine.cleanup()

    assert native.closed.is_set()
    with pytest.raises(EngineShutdown):
        await run(engine, request())


async def test_cleanup_without_start_is_safe(model_dir):
    engine, _ = await make_engine(model_dir)

    await engine.cleanup()
    await engine.cleanup()


async def test_health_check_payload_uses_the_model_bos(engine):
    payload = await engine.health_check_payload()

    assert payload["token_ids"] == [7]
    assert payload["stop_conditions"]["max_tokens"] == 1


@pytest.mark.parametrize(
    "config, expected",
    [({"bos_token_id": 0}, 0), ({"bos_token_id": 7, "eos_token_id": 1}, 7), ({}, 1)],
)
def test_read_bos_token_id(tmp_path, config, expected):
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert read_bos_token_id(str(tmp_path)) == expected


def test_resolve_model_path_requires_config_json(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        llm_engine.resolve_model_path(str(tmp_path))
