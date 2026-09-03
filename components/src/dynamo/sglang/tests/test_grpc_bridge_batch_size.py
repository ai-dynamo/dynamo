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
    _is_absent_grpc_bridge,
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
            # Upstream forwards the resulting error to the client as HTTP 500.
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


class RejectedGenerateReqInput(FakeGenerateReqInput):
    """A request SGLang validates and rejects, after normalization mutated it."""

    def normalize_batch_and_arguments(self) -> None:
        super().normalize_batch_and_arguments()
        raise ValueError("the rids length mismatches the batch size")


def test_a_rejected_request_is_reported_not_left_to_time_out():
    """A rejected request must come back as an error rather than a hang.

    The bridge normally answers one from inside ``_run_generate``'s own handler,
    via ``_send_native_error``. Normalizing early moves the failure outside that
    handler, where an escaping exception would only reach the scheduler's
    logger and the client would wait out its deadline.
    """

    class Bridge(FakeRuntimeHandle):
        def __init__(self) -> None:
            self.reached_generate = 0

        async def generate_request(self, obj):
            self.reached_generate += 1
            async for chunk in super().generate_request(obj):
                yield chunk

        def _send_native_error(self, chunk_callback, message: str) -> None:
            # Upstream shape: an empty payload, marked finished, carrying the error.
            chunk_callback({}, finished=True, error=message)

    ensure_sglang_grpc_bridge_batch_size(Bridge)

    bridge = Bridge()
    obj = RejectedGenerateReqInput("hello", n=2)
    sent: list[tuple] = []

    asyncio.run(
        bridge._run_generate(
            obj, lambda payload, **kw: sent.append((payload, kw)), True
        )
    )

    assert sent == [
        ({}, {"finished": True, "error": "the rids length mismatches the batch size"})
    ]
    assert bridge.reached_generate == 0, "the rejected request still ran"
    assert obj.normalize_calls == 1, "SGLang re-normalized a half-expanded request"
    assert obj.text == ["hello", "hello"]


def test_a_rejected_request_raises_when_there_is_no_error_callback():
    """Without a native error callback, the failure must at least escape.

    Staying silent would lose it entirely; raising leaves it for the scheduler
    to log. Either way the half-expanded request never reaches the bridge.
    """

    class Bridge(FakeRuntimeHandle):
        pass

    ensure_sglang_grpc_bridge_batch_size(Bridge)

    obj = RejectedGenerateReqInput("hello", n=2)
    with pytest.raises(ValueError, match="rids length"):
        _drive(Bridge(), obj)

    assert obj.normalize_calls == 1, "SGLang re-normalized a half-expanded request"
    assert obj.text == ["hello", "hello"]


@pytest.mark.parametrize(
    "missing, absent",
    [
        ("sglang", True),
        ("sglang.srt.entrypoints", True),
        ("sglang.srt.entrypoints.grpc_bridge", True),
        ("grpc", False),
        ("sglang.srt.entrypoints.grpc_bridge_helpers", False),
        (None, False),
    ],
)
def test_an_absent_bridge_is_distinguished_from_a_broken_install(missing, absent):
    """Only the bridge's own absence is tolerable; a broken dependency is not."""
    assert _is_absent_grpc_bridge(ImportError("boom", name=missing)) is absent


def test_missing_sglang_bridge_is_not_an_error():
    """A stock engine without the native gRPC bridge must still start."""
    try:
        import sglang.srt.entrypoints.grpc_bridge  # noqa: F401
    except ImportError as exc:
        if not _is_absent_grpc_bridge(exc):
            raise
    else:
        # Skip rather than patch the real class process-wide for other tests.
        pytest.skip("SGLang's native gRPC bridge is importable here")

    ensure_sglang_grpc_bridge_batch_size()
