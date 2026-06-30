# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.async_vision_encoder.

Pin the glue contract: build / forward / close run on one actor thread; encode
returns one tensor per raw; the A5 preprocess barrier fails a request atomically
(no GPU work) if any image's preprocess fails; load fails fast on a build error or
a missing/invalid hardcoded image_token_id and reaps its thread.
"""

import threading

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
    buckets = None
    image_token_id = 151655

    def __init__(self, *, fail_on=None):
        self.fail_on = set(fail_on or ())
        self.build_thread = None
        self.close_thread = None
        self.closed = False
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


async def test_encode_returns_one_tensor_per_raw():
    enc = AsyncVisionEncoder(_FakeBackend())
    enc.load("m")
    try:
        out = await enc.encode(["a", "bb", "ccc"])
        assert len(out) == 3
        assert all(t.shape == (2, 4) for t in out)
    finally:
        enc.shutdown()


async def test_a5_barrier_fails_atomically_with_no_gpu_work():
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


def test_load_fails_fast_on_build_error_and_reaps_thread():
    class _BadBuild(_FakeBackend):
        def build(self, model_id):
            raise RuntimeError("build failed")

    enc = AsyncVisionEncoder(_BadBuild())
    with pytest.raises(RuntimeError, match="build failed"):
        enc.load("m")
    assert enc._batcher is not None and not enc._batcher._thread.is_alive()


def test_load_fails_fast_on_missing_image_token_id():
    class _NoTokenId(_FakeBackend):
        image_token_id = None  # author forgot to hardcode it

    enc = AsyncVisionEncoder(_NoTokenId())
    with pytest.raises(ValueError, match="image_token_id"):
        enc.load("m")
    assert enc._batcher is not None and not enc._batcher._thread.is_alive()


def test_shutdown_before_load_is_safe():
    AsyncVisionEncoder(_FakeBackend()).shutdown()  # no-op, no raise


def test_preprocess_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="preprocess_concurrency"):
        AsyncVisionEncoder(_FakeBackend(), preprocess_concurrency=0)


def test_load_reaps_pool_if_batcher_ctor_fails():
    """A backend exposing a ladder the batcher rejects must not leak the pool."""

    class _BadBuckets(_FakeBackend):
        max_batch_cost = 8
        buckets = [2, 4]  # max(buckets) < max_batch_cost → batcher ctor raises

    enc = AsyncVisionEncoder(_BadBuckets())
    with pytest.raises(ValueError, match="ladder must cover"):
        enc.load("m")
    assert enc._pool is not None and enc._pool._shutdown is True
    assert enc._batcher is None
