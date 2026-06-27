# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.batched_custom_encoder.

Pin the runtime contract: build and every forward run on one dedicated thread
(so CUDA-graph capture/replay share a thread), concurrent encode() calls coalesce
into fewer forward_batch calls, a forward error reaches every awaiting caller,
pad_to_max_batch yields a static batch size, and the Qwen placeholder mixin.
"""

import asyncio
import threading

import pytest
import torch

from dynamo.vllm.multimodal_utils.batched_custom_encoder import (
    QWEN_IMAGE_PLACEHOLDER_TOKEN,
    BatchedCustomEncoder,
    QwenPlaceholderMixin,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_HIDDEN = 4


class _RecordingEncoder(BatchedCustomEncoder):
    """Records the threads and batch sizes forward_batch is invoked with."""

    max_wait_ms = 200.0  # wide window so concurrent calls coalesce deterministically

    def __init__(self) -> None:
        self.build_thread: int | None = None
        self.forward_threads: list[int] = []
        self.batch_sizes: list[int] = []

    def build(self, model_id: str, device: str) -> None:
        self.build_thread = threading.get_ident()

    def get_image_placeholder_token_id(self) -> int:
        return 7

    def forward_batch(self, image_urls: list[str]) -> list[torch.Tensor]:
        self.forward_threads.append(threading.get_ident())
        self.batch_sizes.append(len(image_urls))
        # One tensor per URL, value = its index, so callers can verify ordering.
        return [torch.full((1, _HIDDEN), float(i)) for i in range(len(image_urls))]


async def test_encode_returns_one_tensor_per_url():
    enc = _RecordingEncoder()
    enc.load("model", "cpu")
    try:
        out = await enc.encode(["a", "b", "c"])
        assert len(out) == 3
        assert all(t.shape == (1, _HIDDEN) for t in out)
    finally:
        enc.shutdown()


async def test_build_and_forward_share_one_non_main_thread():
    """build() and every forward_batch() run on the same dedicated thread, which
    is not the event-loop thread — the affinity a CUDA graph needs."""
    enc = _RecordingEncoder()
    enc.load("model", "cpu")
    try:
        await asyncio.gather(enc.encode(["x"]), enc.encode(["y"]))
        assert enc.build_thread is not None
        assert enc.build_thread != threading.get_ident()  # not the main thread
        assert enc.forward_threads  # ran at least once
        assert set(enc.forward_threads) == {enc.build_thread}  # all on the batcher
    finally:
        enc.shutdown()


async def test_concurrent_calls_coalesce():
    """Concurrent encode() calls merge into fewer forward_batch invocations, and
    every image is still accounted for exactly once."""
    enc = _RecordingEncoder()
    enc.load("model", "cpu")
    try:
        results = await asyncio.gather(*(enc.encode(["u"]) for _ in range(5)))
        assert all(len(r) == 1 for r in results)
        assert sum(enc.batch_sizes) == 5  # nothing dropped or duplicated
        assert max(enc.batch_sizes) >= 2  # coalescing happened
        assert len(enc.batch_sizes) < 5  # fewer forwards than requests
    finally:
        enc.shutdown()


async def test_forward_error_reaches_every_caller():
    class _Failing(_RecordingEncoder):
        def forward_batch(self, image_urls: list[str]) -> list[torch.Tensor]:
            raise ValueError("boom")

    enc = _Failing()
    enc.load("model", "cpu")
    try:
        results = await asyncio.gather(
            *(enc.encode(["u"]) for _ in range(3)), return_exceptions=True
        )
        assert len(results) == 3
        assert all(isinstance(r, ValueError) and str(r) == "boom" for r in results)
    finally:
        enc.shutdown()


async def test_wrong_output_count_raises():
    class _BadCount(_RecordingEncoder):
        def forward_batch(self, image_urls: list[str]) -> list[torch.Tensor]:
            return []  # should be one per URL

    enc = _BadCount()
    enc.load("model", "cpu")
    try:
        with pytest.raises(RuntimeError, match="one tensor per URL"):
            await enc.encode(["a", "b"])
    finally:
        enc.shutdown()


async def test_oversized_request_is_chunked_to_max_batch():
    """A single request larger than max_batch_size is split so forward_batch never
    exceeds the cap, and the outputs stay in order."""

    class _Small(_RecordingEncoder):
        max_batch_size = 3

    enc = _Small()
    enc.load("model", "cpu")
    try:
        out = await enc.encode(["a", "b", "c", "d", "e", "f", "g"])  # 7 > 3
        assert len(out) == 7
        assert enc.batch_sizes == [3, 3, 1]  # capped per forward
    finally:
        enc.shutdown()


async def test_encode_before_load_raises():
    enc = _RecordingEncoder()
    with pytest.raises(RuntimeError, match="before load"):
        await enc.encode(["a"])


async def test_encode_after_shutdown_raises():
    enc = _RecordingEncoder()
    enc.load("model", "cpu")
    enc.shutdown()
    with pytest.raises(RuntimeError, match="after shutdown"):
        await enc.encode(["a"])


async def test_build_error_marks_closed_so_encode_raises():
    """A build() failure exits the thread; load() re-raises and marks the encoder
    closed, so a later encode() raises instead of hanging on a dead consumer."""

    class _BadBuild(_RecordingEncoder):
        def build(self, model_id: str, device: str) -> None:
            raise RuntimeError("build failed")

    enc = _BadBuild()
    with pytest.raises(RuntimeError, match="build failed"):
        enc.load("model", "cpu")
    with pytest.raises(RuntimeError, match="after shutdown"):
        await enc.encode(["a"])


async def test_shutdown_defers_reap_while_forward_in_flight():
    """If a forward is still running when the join times out, shutdown() leaves
    the sentinel for the thread (does not drain it) and a later shutdown() reaps
    the thread once the forward completes."""
    entered = threading.Event()
    release = threading.Event()

    class _Blocking(_RecordingEncoder):
        max_wait_ms = 10.0
        _join_timeout_s = 0.3

        def forward_batch(self, image_urls: list[str]) -> list[torch.Tensor]:
            entered.set()
            release.wait(timeout=5.0)  # hold the batcher thread inside the forward
            return [torch.zeros(1, _HIDDEN) for _ in image_urls]

    enc = _Blocking()
    enc.load("model", "cpu")
    task = asyncio.ensure_future(enc.encode(["a"]))
    for _ in range(200):  # wait until the batcher is actually inside forward_batch
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()

    enc.shutdown()  # join times out (forward blocked) → thread left alive
    assert enc._thread.is_alive()

    release.set()  # forward returns → request resolves, thread reads the sentinel
    assert len(await task) == 1
    enc.shutdown()  # retry join → reaps the now-exited thread
    assert not enc._thread.is_alive()


async def test_shutdown_fails_queued_not_yet_started_requests():
    """On shutdown, requests still queued (not yet picked up) are failed fast
    instead of run as a backlog; the in-flight batch still completes."""
    entered = threading.Event()
    release = threading.Event()

    class _Blocking(_RecordingEncoder):
        max_wait_ms = 0.0  # dispatch the first request alone, immediately
        _join_timeout_s = 0.2

        def forward_batch(self, image_urls: list[str]) -> list[torch.Tensor]:
            entered.set()
            release.wait(timeout=5.0)  # hold the consumer inside "a"'s forward
            return [torch.zeros(1, _HIDDEN) for _ in image_urls]

    enc = _Blocking()
    enc.load("model", "cpu")
    in_flight = asyncio.ensure_future(enc.encode(["a"]))
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()  # "a" is blocked in forward; consumer cannot pull more

    queued = [
        asyncio.ensure_future(enc.encode(["b"])),
        asyncio.ensure_future(enc.encode(["c"])),
    ]
    await asyncio.sleep(0.05)  # let b, c enqueue behind the blocked consumer
    enc.shutdown()  # fails b, c fast; "a" is in flight and finishes
    for q in queued:
        with pytest.raises(RuntimeError, match="shut down"):
            await q

    release.set()
    assert len(await in_flight) == 1
    enc.shutdown()
    assert not enc._thread.is_alive()


async def test_pad_to_max_batch_gives_static_batch_size():
    class _Padded(_RecordingEncoder):
        max_batch_size = 4
        pad_to_max_batch = True

    enc = _Padded()
    enc.load("model", "cpu")
    try:
        out = await enc.encode(["a", "b"])  # 2 real images
        assert len(out) == 2  # padding dropped
        assert enc.batch_sizes == [4]  # forward saw a full, static batch
    finally:
        enc.shutdown()


def test_load_validates_placeholder_id_and_stops_thread():
    """load() resolves the placeholder id after build(), re-raises its error, and
    shuts the batcher thread down so a failed init does not leak it."""

    class _BadId(_RecordingEncoder):
        def get_image_placeholder_token_id(self) -> int:
            raise ValueError("no placeholder id")

    enc = _BadId()
    with pytest.raises(ValueError, match="no placeholder id"):
        enc.load("model", "cpu")
    assert not enc._thread.is_alive()  # load() cleaned up on validate failure


def test_load_reraises_build_error():
    class _BadBuild(_RecordingEncoder):
        def build(self, model_id: str, device: str) -> None:
            raise RuntimeError("build failed")

    enc = _BadBuild()
    with pytest.raises(RuntimeError, match="build failed"):
        enc.load("model", "cpu")
    enc.shutdown()


class _FakeTokenizer:
    def __init__(self, mapping: dict, unk_token_id=None):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str):
        return self._mapping.get(token, self.unk_token_id)


class _QwenEncoder(QwenPlaceholderMixin, _RecordingEncoder):
    pass


@pytest.mark.parametrize("token_id", [151655, 248056])
def test_qwen_mixin_resolves_per_version_id(token_id):
    enc = _QwenEncoder()
    enc.tokenizer = _FakeTokenizer({QWEN_IMAGE_PLACEHOLDER_TOKEN: token_id})
    assert enc.get_image_placeholder_token_id() == token_id


def test_qwen_mixin_unset_tokenizer_raises():
    enc = _QwenEncoder()
    enc.tokenizer = None
    with pytest.raises(ValueError, match="tokenizer is not set"):
        enc.get_image_placeholder_token_id()


def test_qwen_mixin_token_not_defined_raises():
    enc = _QwenEncoder()
    enc.tokenizer = _FakeTokenizer({"other": 1}, unk_token_id=0)
    with pytest.raises(ValueError, match="does not define placeholder token"):
        enc.get_image_placeholder_token_id()


def test_shutdown_stops_thread():
    enc = _RecordingEncoder()
    enc.load("model", "cpu")
    assert enc._thread.is_alive()
    enc.shutdown()
    assert not enc._thread.is_alive()


def test_batched_custom_encoder_is_abstract():
    """build / forward_batch / get_image_placeholder_token_id stay abstract."""
    with pytest.raises(TypeError):
        BatchedCustomEncoder()
