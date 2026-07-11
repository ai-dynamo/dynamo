# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.async_vision_encoder.

Pin the glue contract: build / forward / close run on one actor thread; encode
returns one tensor per raw; the preprocess barrier fails a request atomically
(no GPU work) if any image's preprocess fails; load fails fast on a build error or
a missing/invalid hardcoded image_token_id and reaps its thread.
"""

import asyncio
import threading
from contextlib import suppress

import pytest
import torch

from dynamo.vllm.multimodal_utils.async_vision_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    Preprocessed,
    VisionEncoderBackend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeBackend(VisionEncoderBackend):
    """A CPU-only fake backend; records its threads."""

    max_batch_cost = 8
    image_token_id = 151655
    # This fake overrides preprocess, so opt into the off-loop pool (the default
    # is 0 ⇒ no pool); the passthrough path is covered by its own test below.
    preprocess_concurrency = 4

    def __init__(self, *, fail_on=None):
        self.fail_on = set(fail_on or ())
        self.build_thread = None
        self.close_thread = None
        self.closed = False
        self.close_calls = 0
        self.model_id = None
        self.forward_threads: list[int] = []

    def build(self, model_id):
        self.build_thread = threading.get_ident()
        self.model_id = model_id

    def preprocess(self, raw):
        if raw in self.fail_on:
            raise ValueError(f"bad input {raw}")
        return Preprocessed(item=raw, cost=1)

    def forward_batch(self, items, target_bucket=None):
        self.forward_threads.append(threading.get_ident())
        return [torch.full((2, 4), float(len(str(it)))) for it in items]

    def close(self):
        self.close_thread = threading.get_ident()
        self.closed = True
        self.close_calls += 1


async def test_encode_returns_one_tensor_per_raw():
    enc = AsyncVisionEncoder(_FakeBackend())
    enc.load("m")
    try:
        out = await enc.encode(["a", "bb", "ccc"])
        assert len(out) == 3
        assert all(t.shape == (2, 4) for t in out)
    finally:
        enc.shutdown()


async def test_preprocess_barrier_fails_atomically_with_no_gpu_work():
    be = _FakeBackend(fail_on={"bad"})
    enc = AsyncVisionEncoder(be)
    enc.load("m")
    try:
        with pytest.raises(ValueError, match="bad input"):
            await enc.encode(["good", "bad"])
        assert be.forward_threads == []  # nothing was submitted
    finally:
        enc.shutdown()


async def test_build_and_forward_share_one_non_main_thread():
    be = _FakeBackend()
    enc = AsyncVisionEncoder(be)
    enc.load("m")
    try:
        await enc.encode(["x"])
        assert be.build_thread is not None
        assert set(be.forward_threads) == {be.build_thread}
        assert be.build_thread != threading.get_ident()
    finally:
        enc.shutdown()


async def test_load_resolves_placeholder_and_passes_model_id():
    be = _FakeBackend()
    enc = AsyncVisionEncoder(be)
    enc.load("my-model")
    try:
        assert enc.get_image_placeholder_token_id() == 151655
        assert be.model_id == "my-model"
    finally:
        enc.shutdown()


async def test_encode_empty_returns_empty():
    enc = AsyncVisionEncoder(_FakeBackend())
    enc.load("m")
    try:
        assert await enc.encode([]) == []
    finally:
        enc.shutdown()


async def test_encode_before_load_raises():
    enc = AsyncVisionEncoder(_FakeBackend())
    with pytest.raises(RuntimeError, match="before load"):
        await enc.encode(["a"])


def test_load_twice_raises():
    enc = AsyncVisionEncoder(_FakeBackend())
    enc.load("m")
    try:
        with pytest.raises(RuntimeError, match="called twice"):
            enc.load("m")
    finally:
        enc.shutdown()


def test_shutdown_runs_backend_close_on_actor_thread():
    be = _FakeBackend()
    enc = AsyncVisionEncoder(be)
    enc.load("m")
    enc.shutdown()
    assert be.closed is True
    assert be.close_thread == be.build_thread  # close on the actor thread


def test_shutdown_is_idempotent():
    be = _FakeBackend()
    enc = AsyncVisionEncoder(be)
    enc.load("m")
    enc.shutdown()
    enc.shutdown()
    assert be.close_calls == 1
    assert enc._batcher is None
    assert enc._pool is None


async def test_shutdown_waits_for_preprocess_before_backend_close():
    class _BlockingPreprocess(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.preprocess_entered = threading.Event()
            self.preprocess_release = threading.Event()
            self.preprocess_active = False
            self.close_overlapped_preprocess = False

        def preprocess(self, raw):
            self.preprocess_active = True
            self.preprocess_entered.set()
            self.preprocess_release.wait(timeout=5)
            self.preprocess_active = False
            return Preprocessed(item=raw, cost=1)

        def close(self):
            self.close_overlapped_preprocess = self.preprocess_active
            super().close()

    be = _BlockingPreprocess()
    enc = AsyncVisionEncoder(be)
    enc.load("m")
    encode_task = asyncio.create_task(enc.encode(["image"]))
    try:
        for _ in range(200):
            if be.preprocess_entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert be.preprocess_entered.is_set()

        shutdown_task = asyncio.create_task(asyncio.to_thread(enc.shutdown))
        await asyncio.sleep(0.05)
        assert be.closed is False

        be.preprocess_release.set()
        await shutdown_task
        with suppress(RuntimeError):
            await encode_task
        assert be.closed is True
        assert be.close_overlapped_preprocess is False
    finally:
        be.preprocess_release.set()
        if not encode_task.done():
            encode_task.cancel()
        enc.shutdown()


def test_load_fails_fast_on_build_error_and_reaps_threads():
    class _BadBuild(_FakeBackend):
        def build(self, model_id):
            raise RuntimeError("build failed")

    enc = AsyncVisionEncoder(_BadBuild())
    with pytest.raises(RuntimeError, match="build failed"):
        enc.load("m")
    assert enc._batcher is None
    assert enc._pool is None


def test_load_fails_fast_on_missing_image_token_id():
    class _NoTokenId(_FakeBackend):
        image_token_id = None  # type: ignore[assignment]  # forgot to hardcode it

    enc = AsyncVisionEncoder(_NoTokenId())
    with pytest.raises(ValueError, match="image_token_id"):
        enc.load("m")
    assert enc._batcher is None
    assert enc._pool is None


def test_shutdown_before_load_is_safe():
    AsyncVisionEncoder(_FakeBackend()).shutdown()  # no-op, no raise


class _PassthroughBackend(VisionEncoderBackend):
    """Keeps the base identity preprocess (no override) — the legal passthrough
    shape at preprocess_concurrency=0."""

    image_token_id = 151655

    def __init__(self):
        self.forward_calls: list[list] = []

    def build(self, model_id):
        pass

    def forward_batch(self, items, target_bucket=None):
        self.forward_calls.append(list(items))
        return [torch.zeros(2, 4) for _ in items]


async def test_passthrough_skips_preprocess_when_no_pool():
    """With no pool (preprocess_concurrency=0) preprocess is skipped and raws go
    straight to the batcher → forward_batch, no barrier, no pool thread."""
    be = _PassthroughBackend()
    # Instance-level boom (invisible to the class-override check) proves the
    # phase is never entered, not merely run through an identity hook.
    be.preprocess = lambda raw: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("preprocess must not run in passthrough mode")
    )
    enc = AsyncVisionEncoder(be)
    enc.load("m")
    try:
        assert enc._pool is None  # no pool created
        out = await enc.encode(["a", "bb"])
        assert len(out) == 2
        assert all(t.shape == (2, 4) for t in out)
        # Passthrough must hand the raws themselves to forward_batch, unchanged
        # and in order. A single request may span multiple micro-batches, so
        # flatten the recorded calls rather than requiring one forward call.
        assert [item for batch in be.forward_calls for item in batch] == ["a", "bb"]
    finally:
        enc.shutdown()


def test_preprocess_concurrency_zero_disables_pool():
    enc = AsyncVisionEncoder(_PassthroughBackend(), preprocess_concurrency=0)
    enc.load("m")
    try:
        assert enc._pool is None
    finally:
        enc.shutdown()


def test_overridden_preprocess_with_zero_concurrency_raises():
    """A backend that overrides preprocess but declares no pool would silently
    never run it — rejected at construction."""

    class _MismatchedBackend(_FakeBackend):
        preprocess_concurrency = 0

    with pytest.raises(ValueError, match="never run"):
        AsyncVisionEncoder(_MismatchedBackend())


def test_driver_override_to_zero_with_overriding_backend_raises():
    """The constructor override participates in the same check: forcing the
    effective concurrency to 0 must not silently skip an overridden preprocess."""
    with pytest.raises(ValueError, match="never run"):
        AsyncVisionEncoder(_FakeBackend(), preprocess_concurrency=0)


def test_preprocess_concurrency_rejects_negative():
    with pytest.raises(ValueError, match="preprocess_concurrency"):
        AsyncVisionEncoder(_FakeBackend(), preprocess_concurrency=-1)


def test_load_bad_batch_cost_fails_without_spawning_pool():
    """A backend whose max_batch_cost the batcher rejects fails load() cleanly: the
    batcher ctor raises before the preprocess pool is ever spawned (nothing to reap)."""

    class _BadBudget(_FakeBackend):
        max_batch_cost = 0  # ThreadedMicroBatcher requires >= 1 → ctor raises

    enc = AsyncVisionEncoder(_BadBudget())
    with pytest.raises(ValueError, match="max_batch_cost"):
        enc.load("m")
    assert enc._pool is None
    assert enc._batcher is None
