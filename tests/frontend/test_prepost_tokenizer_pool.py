#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for the async tokenizer executor pool in dynamo.frontend.prepost."""

# mypy seems to be running both sides of the HAS_VLLM if statement
# mypy: ignore-errors

import asyncio
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from .common import check_module_available

HAS_VLLM = check_module_available("vllm.entrypoints.openai.chat_completion.protocol")
if HAS_VLLM:
    from dynamo.frontend import prepost


pytestmark = [
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,  # "Hardware"
    pytest.mark.pre_merge,  # "Lifecyle"
    pytest.mark.unit,  # "Test Type"
    pytest.mark.skipif(not HAS_VLLM, reason="requires vllm"),
]


class _DummyTokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return SimpleNamespace(input_ids=[1, 2, 3])


class _SlotsTokenizer:
    # No __weakref__: weakref.ref() raises TypeError.
    __slots__ = ()

    def __call__(self, prompt, **kwargs):
        return SimpleNamespace(input_ids=[4, 5, 6])


class _RecordingExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdown_calls = 0

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_calls += 1
        super().shutdown(wait, cancel_futures=cancel_futures)


@pytest.fixture(autouse=True)
def _clean_pools(monkeypatch):
    monkeypatch.setattr(prepost, "ThreadPoolExecutor", _RecordingExecutor)
    prepost._ASYNC_TOKENIZER_EXECUTORS.clear()
    prepost._STRONG_ASYNC_TOKENIZER_EXECUTORS.clear()
    yield
    for executor in list(prepost._ASYNC_TOKENIZER_EXECUTORS.values()):
        executor.shutdown()
    for _, executor in list(prepost._STRONG_ASYNC_TOKENIZER_EXECUTORS.values()):
        executor.shutdown()
    prepost._ASYNC_TOKENIZER_EXECUTORS.clear()
    prepost._STRONG_ASYNC_TOKENIZER_EXECUTORS.clear()


async def _call_tokenizer(tokenizer, prompt):
    # make_async returns a Future, so the call must be awaited inside a
    # running loop rather than passed to asyncio.run directly.
    return await prepost._get_async_tokenizer(tokenizer)(prompt)


def test_same_tokenizer_reuses_one_executor():
    tokenizer = _DummyTokenizer()
    prepost._get_async_tokenizer(tokenizer)
    prepost._get_async_tokenizer(tokenizer)

    assert len(prepost._ASYNC_TOKENIZER_EXECUTORS) == 1
    executor = prepost._ASYNC_TOKENIZER_EXECUTORS[tokenizer]
    assert isinstance(executor, _RecordingExecutor)

    result = asyncio.run(_call_tokenizer(tokenizer, "hello"))
    assert result.input_ids == [1, 2, 3]
    assert tokenizer.prompts == ["hello"]


def test_executor_evicted_and_shut_down_when_tokenizer_collected():
    tokenizer = _DummyTokenizer()
    prepost._get_async_tokenizer(tokenizer)
    executor = prepost._ASYNC_TOKENIZER_EXECUTORS[tokenizer]
    tokenizer_ref = weakref.ref(tokenizer)

    # No cached wrapper may survive this del: it would pin the tokenizer.
    del tokenizer
    gc.collect()

    assert tokenizer_ref() is None
    assert len(prepost._ASYNC_TOKENIZER_EXECUTORS) == 0
    assert executor.shutdown_calls == 1


def test_new_tokenizer_after_eviction_gets_fresh_executor():
    first = _DummyTokenizer()
    prepost._get_async_tokenizer(first)
    first_executor = prepost._ASYNC_TOKENIZER_EXECUTORS[first]
    del first
    gc.collect()
    assert len(prepost._ASYNC_TOKENIZER_EXECUTORS) == 0

    second = _DummyTokenizer()
    prepost._get_async_tokenizer(second)
    assert len(prepost._ASYNC_TOKENIZER_EXECUTORS) == 1
    assert prepost._ASYNC_TOKENIZER_EXECUTORS[second] is not first_executor


def test_non_weakrefable_tokenizer_falls_back_to_strong_pool():
    tokenizer = _SlotsTokenizer()
    result = asyncio.run(_call_tokenizer(tokenizer, "hi"))
    assert result.input_ids == [4, 5, 6]

    assert len(prepost._ASYNC_TOKENIZER_EXECUTORS) == 0
    assert len(prepost._STRONG_ASYNC_TOKENIZER_EXECUTORS) == 1

    # The fallback entry must retain the tokenizer itself so its id() cannot
    # be recycled by a different tokenizer while the entry lives.
    retained_tokenizer, _ = prepost._STRONG_ASYNC_TOKENIZER_EXECUTORS[id(tokenizer)]
    assert retained_tokenizer is tokenizer

    prepost._get_async_tokenizer(tokenizer)
    assert len(prepost._STRONG_ASYNC_TOKENIZER_EXECUTORS) == 1
