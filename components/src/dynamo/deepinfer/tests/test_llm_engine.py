# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run directly with Python and NumPy, without CUDA or Dynamo bindings."""

import asyncio
import contextlib
import copy
import importlib.util
import io
import struct
import sys
import threading
import time
import types
import unittest
import pytest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

def load_adapter():
    stubs = {}

    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        stubs[name] = value

    class Mode(Enum):
        AGGREGATED = "agg"

    class InvalidArgument(ValueError):
        pass

    class EngineShutdown(RuntimeError):
        pass

    class Cancelled(RuntimeError):
        pass

    def normalize(request):
        if isinstance(request.get("token_ids"), bytes):
            data = request["token_ids"]
            request["token_ids"] = list(struct.unpack(f"<{len(data)//4}I", data))

    module("dynamo._core", Context=object)
    module(
        "dynamo.common.backend.engine",
        EngineConfig=SimpleNamespace,
        GenerateChunk=dict,
        GenerateRequest=dict,
        LLMEngine=object,
        LlmRegistration=SimpleNamespace,
    )
    module("dynamo.common.backend.worker", WorkerConfig=SimpleNamespace)
    module("dynamo.common.constants", DisaggregationMode=Mode)
    module("dynamo.common.utils.token_ids", normalize_request_token_ids=normalize)
    module(
        "dynamo.llm.exceptions",
        InvalidArgument=InvalidArgument,
        EngineShutdown=EngineShutdown,
        Cancelled=Cancelled,
    )
    path = Path(__file__).resolve().parents[1] / "llm_engine.py"
    spec = importlib.util.spec_from_file_location("_deepinfer_adapter_test", path)
    adapter = importlib.util.module_from_spec(spec)
    stubs[spec.name] = adapter
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(adapter)
    return adapter


a = load_adapter()


class FakeEngine:
    def __init__(self):
        self.model_config = SimpleNamespace(
            max_model_len=128, get_vocab_size=lambda: 100
        )
        self.scheduler_config = SimpleNamespace(
            max_num_seqs=16, max_num_batched_tokens=128
        )
        self.tokenizer = SimpleNamespace(bos_token_id=1)
        self.eos_id = 2
        self.active = {}
        self.pointers = {}
        self.calls = []
        self.aborted = []
        self.closed = 0
        self.threads = set()
        self.max_active = 0
        self.reject = False
        self.crash = False
        self.defer = False
        self.pending = []

    def record(self):
        self.threads.add(threading.get_ident())

    def add_request(self, **kwargs):
        self.record()
        self.calls.append(kwargs)
        if self.reject:
            return None
        self.active[kwargs["request_id"]] = [kwargs, 0]
        self.max_active = max(self.max_active, len(self.active))
        return object()

    def has_unfinished_requests(self):
        self.record()
        return bool(self.active)

    def step(self):
        self.record()
        if self.crash:
            raise RuntimeError("fake step failure")
        time.sleep(0.002)
        for output in self.pointers.values():
            output.token_id = -999
        self.pointers = {}
        for rid, (request, count) in list(self.active.items()):
            count += 1
            finished = count >= request["sampling_params"]["max_tokens"]
            out = SimpleNamespace(
                request_id=rid,
                token_id=10 + count,
                finished=finished,
                finish_reason="length" if finished else "",
                prompt_len=len(request["prompt_token_ids"]),
                num_output_tokens=count,
            )
            if self.defer and finished:
                self.pending.append(out)
            else:
                self.pointers[len(self.pointers) + 1] = out
            if finished:
                del self.active[rid]
            else:
                self.active[rid][1] = count
        return list(self.pointers)

    def flush_inflight_reqs(self):
        self.record()
        self.pointers = {i + 1: out for i, out in enumerate(self.pending)}
        self.pending = []
        return list(self.pointers)

    def clone(self, pointer):
        self.record()
        return copy.deepcopy(self.pointers[pointer])

    def abort_request(self, rid):
        self.record()
        self.aborted.append(rid)
        self.active.pop(rid, None)

    def close(self):
        self.record()
        self.closed += 1


def ctx(rid="r", stopped=False):
    return SimpleNamespace(id=lambda: rid, is_stopped=lambda: stopped)


def request(count=3):
    return {"token_ids": [1, 2], "stop_conditions": {"max_tokens": count}}


async def collect(engine, req=None, context=None):
    return [x async for x in engine.generate(req or request(), context or ctx())]


class MappingTests(unittest.TestCase):
    def test_sampling_zero_and_seed(self):
        self.assertEqual(
            a._sampling(
                {
                    "sampling_options": {"temperature": 0, "seed": 0},
                    "stop_conditions": {"max_tokens": 3, "ignore_eos": False},
                }
            ),
            {"temperature": 0, "seq_seed": 0, "max_tokens": 3, "ignore_eos": False},
        )

    def test_invalid_sampling(self):
        for value in [0, True]:
            with self.assertRaises(a.InvalidArgument):
                a._sampling({"stop_conditions": {"max_tokens": value}})
        for key, value in [("temperature", float("nan")), ("top_p", 0), ("top_k", 0)]:
            with self.assertRaises(a.InvalidArgument):
                a._sampling({"sampling_options": {key: value}})

    def test_default_and_frontend_owned_fields(self):
        a._unsupported(
            {
                "sampling_options": {"n": 1, "repetition_penalty": 1},
                "stop_conditions": {"stop_token_ids_hidden": [2], "stop": ["end"]},
                "output_options": {"skip_special_tokens": True},
            }
        )

    def test_unsupported_including_zero_logprobs(self):
        for req in [
            {"output_options": {"logprobs": 0}},
            {"sampling_options": {"n": 2}},
            {"sampling_options": {"guided_decoding": {"json": {}}}},
            {"extra_args": {"sampling_options": {"bad_words_token_ids": [[2]]}}},
            {"multi_modal_data": {"image": [1]}},
        ]:
            with self.assertRaises(a.InvalidArgument):
                a._unsupported(req)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeEngine()
        self.patches = [
            patch.object(a, "_load_engine_args", return_value=SimpleNamespace),
            patch.object(
                a,
                "_load_llm_engine",
                return_value=SimpleNamespace(from_engine_args=lambda args: self.fake),
            ),
            patch.object(
                a,
                "_load_request_output",
                return_value=SimpleNamespace(clone=self.fake.clone),
            ),
        ]
        for p in self.patches:
            p.start()
        self.engine, self.config = await a.DeepInferLLMEngine.from_args(
            ["--model", "local-model"]
        )
        await self.engine.start(1)

    async def asyncTearDown(self):
        await asyncio.wait_for(self.engine.cleanup(), 3)
        for p in reversed(self.patches):
            p.stop()

    async def test_stream_usage_clone_and_native_thread(self):
        chunks = await asyncio.wait_for(collect(self.engine), 3)
        self.assertEqual([c["token_ids"] for c in chunks], [[11], [12], [13]])
        self.assertEqual(chunks[-1]["finish_reason"], "length")
        self.assertEqual(
            chunks[-1]["completion_usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )
        self.assertEqual(len(self.fake.threads), 1)
        self.assertNotIn(threading.get_ident(), self.fake.threads)

    async def test_concurrent_admission_during_generation(self):
        stream = self.engine.generate(request(20), ctx("first"))
        await asyncio.wait_for(anext(stream), 3)
        second = asyncio.create_task(collect(self.engine, request(2), ctx("second")))
        remainder = [x async for x in stream]
        await asyncio.wait_for(second, 3)
        self.assertEqual(len(remainder), 19)
        self.assertGreaterEqual(self.fake.max_active, 2)

    async def test_none_rejection_does_not_kill_driver(self):
        self.fake.reject = True
        with self.assertRaises(a.InvalidArgument):
            await asyncio.wait_for(collect(self.engine), 3)
        self.fake.reject = False
        self.assertEqual(len(await collect(self.engine)), 3)

    async def test_flush_delivers_last_output(self):
        self.fake.defer = True
        chunks = await asyncio.wait_for(collect(self.engine), 3)
        self.assertEqual(chunks[-1]["finish_reason"], "length")

    async def test_abort_and_generator_close(self):
        for rid in ("abort", "close"):
            stream = self.engine.generate(request(30), ctx(rid))
            await asyncio.wait_for(anext(stream), 3)
            if rid == "abort":
                await self.engine.abort(ctx(rid))
                with self.assertRaises(a.Cancelled):
                    await anext(stream)
            else:
                await stream.aclose()
            for _ in range(100):
                if rid not in self.engine._requests:
                    break
                await asyncio.sleep(0.005)
            self.assertNotIn(rid, self.engine._requests)
        self.assertEqual(len(self.fake.aborted), 2)

    async def test_slow_consumer_fails_explicitly(self):
        with patch.object(a, "_QUEUE_SIZE", 1):
            stream = self.engine.generate(request(30), ctx())
            await anext(stream)
            await asyncio.sleep(0.1)
            with self.assertRaisesRegex(a.EngineShutdown, "too slow"):
                await anext(stream)
        self.assertTrue(self.fake.aborted)

    async def test_cleanup_wakes_stream_and_is_idempotent(self):
        stream = self.engine.generate(request(30), ctx())
        await anext(stream)
        await self.engine.cleanup()
        with self.assertRaises(a.EngineShutdown):
            await anext(stream)
        await self.engine.cleanup()
        self.assertEqual(self.fake.closed, 1)

    async def test_native_failure_fails_future_requests(self):
        self.fake.crash = True
        with self.assertRaises(a.EngineShutdown):
            await asyncio.wait_for(collect(self.engine), 3)
        with self.assertRaises(a.EngineShutdown):
            await asyncio.wait_for(collect(self.engine, context=ctx("next")), 3)

    async def test_input_validation_before_submission(self):
        for tokens in ([], [-1], [2**32], [True], [100]):
            with self.assertRaises(a.InvalidArgument):
                await collect(self.engine, {"token_ids": tokens})
        self.assertEqual(self.fake.calls, [])

    async def test_cli_lazy_import_and_disabled_kv(self):
        with patch.object(
            a, "_load_engine_args", side_effect=AssertionError("early import")
        ):
            e, config = await a.DeepInferLLMEngine.from_args(
                ["--model", "m", "--response-plane", "quic"]
            )
            self.assertFalse(config.enable_kv_routing)
            self.assertEqual(config.response_plane, "quic")
            await e.cleanup()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            await a.DeepInferLLMEngine.from_args(
                ["--model", "m", "--data-parallel-size", "2"]
            )

    async def test_partial_start_failure(self):
        e, _ = await a.DeepInferLLMEngine.from_args(["--model", "m"])
        with patch.object(
            self.fake.model_config,
            "get_vocab_size",
            side_effect=RuntimeError("metadata"),
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata"):
                await e.start(2)
        self.assertTrue(e._closed)
        self.assertEqual(self.fake.closed, 1)


if __name__ == "__main__":
    unittest.main()
