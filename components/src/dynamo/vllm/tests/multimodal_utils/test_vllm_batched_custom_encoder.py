# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.batched_custom_encoder.

Pin the contract + glue to the ThreadedMicroBatcher: load() runs build() on the
batcher thread then validates, encode() preprocesses off-thread and returns one
tensor per URL, the cost/bucket policy flows through, and the Qwen mixin resolves
the placeholder id. The heavy concurrency behavior is covered in
test_vllm_threaded_micro_batcher.py.
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


class _RecEnc(BatchedCustomEncoder):
    max_wait_ms = 200.0

    def __init__(self):
        self.build_thread: int | None = None
        self.fwd_threads: list[int] = []
        self.batches: list[list] = []

    def build(self, model_id: str, device: str) -> None:
        self.build_thread = threading.get_ident()

    def get_image_placeholder_token_id(self) -> int:
        return 7

    def forward_batch(self, items: list) -> list[torch.Tensor]:
        self.fwd_threads.append(threading.get_ident())
        self.batches.append(list(items))
        return [torch.full((1, _HIDDEN), float(i)) for i in range(len(items))]


async def test_encode_returns_one_tensor_per_url():
    enc = _RecEnc()
    enc.load("model", "cpu")
    try:
        out = await enc.encode(["a", "b", "c"])
        assert len(out) == 3 and all(t.shape == (1, _HIDDEN) for t in out)
    finally:
        enc.shutdown()


async def test_build_runs_on_batcher_thread():
    enc = _RecEnc()
    enc.load("model", "cpu")
    try:
        await asyncio.gather(enc.encode(["x"]), enc.encode(["y"]))
        assert (
            enc.build_thread is not None and enc.build_thread != threading.get_ident()
        )
        assert set(enc.fwd_threads) == {enc.build_thread}
    finally:
        enc.shutdown()


async def test_preprocess_feeds_forward_batch():
    class _PreEnc(_RecEnc):
        def preprocess(self, image_url: str) -> object:
            return f"item:{image_url}"

    enc = _PreEnc()
    enc.load("model", "cpu")
    try:
        await enc.encode(["a", "b"])
        assert enc.batches[0] == ["item:a", "item:b"]  # forward saw preprocessed items
    finally:
        enc.shutdown()


async def test_bucket_policy_flows_through():
    """An encoder's bucket_key splits items into separate forward batches."""

    class _BucketEnc(_RecEnc):
        def bucket_key(self, item) -> object:
            return item  # every distinct url is its own bucket

    enc = _BucketEnc()
    enc.load("model", "cpu")
    try:
        await asyncio.gather(enc.encode(["a"]), enc.encode(["b"]))
        # a and b never share a batch
        assert all(len(set(batch)) == 1 for batch in enc.batches)
    finally:
        enc.shutdown()


async def test_build_error_makes_encode_raise():
    class _BadBuild(_RecEnc):
        def build(self, model_id: str, device: str) -> None:
            raise RuntimeError("build failed")

    enc = _BadBuild()
    with pytest.raises(RuntimeError, match="build failed"):
        enc.load("model", "cpu")
    with pytest.raises(RuntimeError, match="after shutdown"):
        await enc.encode(["a"])


def test_validate_failure_reaps_thread():
    class _BadId(_RecEnc):
        def get_image_placeholder_token_id(self) -> int:
            raise ValueError("no placeholder id")

    enc = _BadId()
    with pytest.raises(ValueError, match="no placeholder id"):
        enc.load("model", "cpu")
    assert not enc._batcher._thread.is_alive()  # load() shut the batcher down


def test_double_load_raises():
    """A second load() is rejected rather than orphaning the first batcher's
    (non-daemon) worker thread + model."""
    enc = _RecEnc()
    enc.load("model", "cpu")
    try:
        with pytest.raises(RuntimeError, match="twice"):
            enc.load("model", "cpu")
    finally:
        enc.shutdown()


def test_encode_before_load_raises():
    enc = _RecEnc()
    with pytest.raises(RuntimeError, match="before load"):
        asyncio.run(enc.encode(["a"]))


class _FakeTokenizer:
    def __init__(self, mapping: dict, unk_token_id=None):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str):
        return self._mapping.get(token, self.unk_token_id)


class _QwenEnc(QwenPlaceholderMixin, _RecEnc):
    pass


@pytest.mark.parametrize("token_id", [151655, 248056])
def test_qwen_mixin_resolves_per_version_id(token_id):
    enc = _QwenEnc()
    enc.tokenizer = _FakeTokenizer({QWEN_IMAGE_PLACEHOLDER_TOKEN: token_id})
    assert enc.get_image_placeholder_token_id() == token_id


def test_qwen_mixin_unset_tokenizer_raises():
    enc = _QwenEnc()
    enc.tokenizer = None
    with pytest.raises(ValueError, match="tokenizer is not set"):
        enc.get_image_placeholder_token_id()


def test_batched_custom_encoder_is_abstract():
    with pytest.raises(TypeError):
        BatchedCustomEncoder()
