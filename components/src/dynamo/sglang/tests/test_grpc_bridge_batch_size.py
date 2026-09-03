# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the SGLang native gRPC bridge batch_size shim.

SGLang 0.5.17 and 0.5.18 read ``obj.batch_size`` in the streaming branch of the
native gRPC bridge's ``_run_generate``, one statement after building the
``generate_request`` async generator -- and therefore before anything has run
the normalization that assigns it. The fakes below reproduce that ordering, so
these tests run without SGLang installed.
"""

import asyncio

import pytest

from dynamo.sglang._compat import (
    _normalize_generate_request_once,
    ensure_sglang_grpc_bridge_batch_size,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


class FakeGenerateReqInput:
    """Stand-in for SGLang's GenerateReqInput normalization behavior.

    Mirrors the parts of ``normalize_batch_and_arguments()`` that matter here:
    ``batch_size`` exists only after it runs, and running it twice re-expands
    parallel-sampling inputs from an already-expanded list.
    """

    def __init__(self, text: str, n: int = 1) -> None:
        self.rid = "test-request"
        self.text: object = text
        self.n = n
        self.parallel_sample_num = 1
        self.normalize_calls = 0
        # batch_size is deliberately not set here: SGLang assigns it only in
        # _determine_batch_size(), reached through the normalization below.

    def normalize_batch_and_arguments(self) -> None:
        self.normalize_calls += 1
        if isinstance(self.text, str):
            self.is_single = True
            self.batch_size = 1
        else:
            self.is_single = False
            self.batch_size = len(self.text)
        self.parallel_sample_num = self.n
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            self.text = [self.text]
        if not self.is_single:
            assert isinstance(self.text, list)
            self.text = self.text * self.parallel_sample_num


class FakeRuntimeHandle:
    """Stand-in for the SGLang 0.5.18 grpc_bridge.RuntimeHandle call ordering."""

    async def generate_request(self, obj):
        # TokenizerManager.generate_request normalizes inside the generator
        # body, so nothing here runs until the first anext() below.
        obj.normalize_batch_and_arguments()
        for index in range(obj.batch_size * obj.parallel_sample_num):
            yield {"index": index, "text": "hi", "finish_reason": "stop"}

    async def _run_generate(self, obj, chunk_callback, stream, request=None):
        gen = self.generate_request(obj)
        if stream:
            # The defect: obj.batch_size is read before the generator body runs.
            # Upstream wraps this in `except Exception` and forwards the message
            # to the client, which is how it surfaces as an HTTP 500 body.
            expected_choices = obj.batch_size * obj.parallel_sample_num
            completed = set()
            async for chunk in gen:
                chunk_callback(chunk)
                if chunk["finish_reason"] is not None:
                    completed.add(chunk["index"])
                if len(completed) >= expected_choices:
                    return


def _drive(handle, obj, stream=True):
    """Run one request through the bridge and return the streamed chunks."""
    chunks: list[dict] = []

    async def run():
        await handle._run_generate(obj, chunks.append, stream)

    asyncio.run(run())
    return chunks


def test_unpatched_bridge_reproduces_the_defect():
    """Without the shim, a streaming request dies on the missing attribute."""

    class Bridge(FakeRuntimeHandle):
        pass

    with pytest.raises(AttributeError) as excinfo:
        _drive(Bridge(), FakeGenerateReqInput("hello"))

    assert "batch_size" in str(excinfo.value)


def test_shim_lets_a_streaming_request_complete():
    """With the shim installed, the request normalizes first and streams."""

    class Bridge(FakeRuntimeHandle):
        pass

    ensure_sglang_grpc_bridge_batch_size(Bridge)

    obj = FakeGenerateReqInput("hello")
    chunks = _drive(Bridge(), obj)

    assert [chunk["index"] for chunk in chunks] == [0]
    assert obj.batch_size == 1


def test_shim_does_not_double_expand_parallel_sampling():
    """Normalization must happen exactly once, even though SGLang calls it too.

    ``normalize_batch_and_arguments()`` is not idempotent: a second pass derives
    the batch size from the already-expanded input list, so ``n`` sequences
    would become ``n * n``. Normalizing early is only safe because the shim
    neutralizes SGLang's own later call on that request object.
    """

    class Bridge(FakeRuntimeHandle):
        pass

    ensure_sglang_grpc_bridge_batch_size(Bridge)

    obj = FakeGenerateReqInput("hello", n=3)
    chunks = _drive(Bridge(), obj)

    assert len(chunks) == 3, "parallel sampling produced the wrong choice count"
    assert obj.text == ["hello", "hello", "hello"]
    assert obj.normalize_calls == 1


def test_shim_installation_is_idempotent():
    """Installing twice must not stack wrappers or re-normalize."""

    class Bridge(FakeRuntimeHandle):
        pass

    ensure_sglang_grpc_bridge_batch_size(Bridge)
    wrapped = Bridge._run_generate
    ensure_sglang_grpc_bridge_batch_size(Bridge)

    assert Bridge._run_generate is wrapped

    obj = FakeGenerateReqInput("hello", n=2)
    assert len(_drive(Bridge(), obj)) == 2
    assert obj.normalize_calls == 1


def test_shim_leaves_an_already_normalized_request_alone():
    """On a release that assigns batch_size first, the shim must do nothing."""
    obj = FakeGenerateReqInput("hello", n=2)
    obj.normalize_batch_and_arguments()
    calls_before = obj.normalize_calls

    _normalize_generate_request_once(obj)

    assert obj.normalize_calls == calls_before
    assert obj.text == ["hello", "hello"]


def test_missing_sglang_bridge_is_not_an_error():
    """A stock engine without the native gRPC bridge must still start."""
    try:
        import sglang.srt.entrypoints.grpc_bridge  # noqa: F401
    except Exception:
        pass
    else:
        # Skip rather than patch the real class process-wide for other tests.
        pytest.skip("SGLang's native gRPC bridge is importable here")

    ensure_sglang_grpc_bridge_batch_size()
