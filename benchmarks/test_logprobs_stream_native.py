# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT-LLM aggregate-stream handler benchmark.

The fixture first obtains real cumulative CompletionOutput objects from a small
TensorRT-LLM PyTorch engine. The timed operation replays those objects through
AggregatedHandler so the benchmark isolates handler CPU work while preserving
the backend's output types, cumulative token shape, logprobs, and finish data.
The engine setup and generation are outside the timed boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from dynamo.trtllm.request_handlers.aggregated_handler import AggregatedHandler

# The native model setup averages over a minute on one GPU, so keep it out of
# pre-merge while retaining a scheduled post-merge regression lane.
pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.integration,
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.benchmark,
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        os.environ.get("DYN_RUN_LOGPROBS_NATIVE_BENCHMARK") != "1",
        reason="run explicitly with native TensorRT-LLM dependencies",
    ),
]


_MODEL_VOCAB_SIZE = 512
_MODEL_MAX_SEQ_LEN = 512
_STREAM_LENGTHS = (32, 256)
_NUM_CHOICES = 4


class _ReplayResult:
    def __init__(self, responses: list[Any]):
        self._responses = responses
        self._position = 0

    def __aiter__(self) -> _ReplayResult:
        return self

    async def __anext__(self) -> Any:
        if self._position >= len(self._responses):
            raise StopAsyncIteration
        response = self._responses[self._position]
        self._position += 1
        return response

    def abort(self) -> None:
        self._position = len(self._responses)


class _ReplayLLM:
    args = None

    def __init__(self, responses: list[Any]):
        self._responses = responses

    def generate_async(self, **_: Any) -> _ReplayResult:
        return _ReplayResult(self._responses)


class _SnapshotResult:
    def __init__(self, result: Any):
        self._result = result

    def __aiter__(self) -> _SnapshotResult:
        return self

    async def __anext__(self) -> Any:
        return _snapshot_native_response(await self._result.__anext__())

    def abort(self) -> None:
        self._result.abort()


class _SnapshotLLM:
    args = None

    def __init__(self, llm: Any):
        self._llm = llm

    def generate_async(self, **kwargs: Any) -> _SnapshotResult:
        return _SnapshotResult(self._llm.generate_async(**kwargs))


class _ReplayEngine:
    def __init__(self, llm: _ReplayLLM):
        self.llm = llm


class _Context:
    def id(self) -> str:
        return "native-logprobs-benchmark"

    def trace_headers(self) -> dict[str, str]:
        return {}

    def notify_first_token(self) -> None:
        return None

    def async_killed_or_stopped(self) -> asyncio.Task[None]:
        return asyncio.create_task(asyncio.Event().wait())


def _snapshot_logprobs(values: Any) -> Any:
    """Copy mutable native logprob containers before TRT-LLM reuses them."""
    if not isinstance(values, list):
        return values
    return [
        None
        if position is None
        else dict(position)
        if isinstance(position, dict)
        else position
        for position in values
    ]


def _snapshot_native_response(response: Any) -> Any:
    """Keep each native response's cumulative state at its yield boundary.

    TensorRT-LLM reuses and mutates ``RequestOutput`` instances while an async
    stream is consumed. A shallow response copy retains its real backend type;
    copying the output lists prevents later generation steps from turning every
    captured sample into the final response.
    """
    snapshot = copy.copy(response)
    snapshot._outputs = []
    for output in response.outputs:
        output_snapshot = copy.copy(output)
        output_snapshot.token_ids = list(output.token_ids)
        output_snapshot.logprobs = _snapshot_logprobs(output.logprobs)
        output_snapshot.prompt_logprobs = _snapshot_logprobs(output.prompt_logprobs)
        snapshot._outputs.append(output_snapshot)
    return snapshot


def _assert_cumulative_stream(responses: list[Any], length: int) -> None:
    """Prove the native fixture has the cumulative, one-token stream shape."""
    assert responses
    previous_tokens: dict[int, list[int]] = {}
    observed_indexes: set[int] = set()
    finish_reasons: dict[int, str] = {}

    for response in responses:
        assert response.outputs
        for output in response.outputs:
            output_idx = getattr(output, "index", 0) or 0
            assert 0 <= output_idx < _NUM_CHOICES
            observed_indexes.add(output_idx)
            tokens = list(output.token_ids)
            previous = previous_tokens.get(output_idx, [])
            assert tokens[: len(previous)] == previous
            delta_size = len(tokens) - len(previous)
            assert 0 <= delta_size <= 1

            logprobs = list(output.logprobs or [])
            assert len(logprobs) == len(tokens)
            for token_id, position in zip(tokens, logprobs):
                if isinstance(position, dict):
                    assert token_id in position

            previous_tokens[output_idx] = tokens
            if output.finish_reason is not None:
                finish_reasons[output_idx] = output.finish_reason

    assert observed_indexes == set(range(_NUM_CHOICES))
    assert {index: len(tokens) for index, tokens in previous_tokens.items()} == {
        index: length for index in range(_NUM_CHOICES)
    }
    assert finish_reasons == {index: "length" for index in range(_NUM_CHOICES)}


def _native_wire_chunks(responses: list[Any], prompt_tokens: int) -> list[dict]:
    """Build an output-independent oracle from captured native cumulative data."""
    cursor: dict[int, int] = {}
    expected: list[dict] = []
    for response in responses:
        response_outputs = response.outputs
        total_tokens = sum(len(output.token_ids) for output in response_outputs)
        for output in response_outputs:
            output_idx = getattr(output, "index", 0) or 0
            tokens_so_far = cursor.get(output_idx, 0)
            token_ids = list(output.token_ids)
            delta = token_ids[tokens_so_far:]
            positions = list(output.logprobs or [])[tokens_so_far:]
            positions = positions[: len(delta)]
            chunk = {"token_ids": delta, "index": output_idx}

            if positions:
                if isinstance(positions[0], float):
                    chunk["log_probs"] = [float(value) for value in positions]
                else:
                    log_probs = []
                    top_logprobs = []
                    for token_id, position in zip(delta, positions):
                        assert isinstance(position, dict) and position
                        selected = position.get(token_id) or next(
                            iter(position.values())
                        )
                        log_probs.append(float(selected.logprob))
                        top_logprobs.append(
                            [
                                {
                                    "rank": info.rank if hasattr(info, "rank") else 0,
                                    "token_id": candidate_id,
                                    "token": getattr(info, "decoded_token", None),
                                    "logprob": float(info.logprob),
                                }
                                for candidate_id, info in position.items()
                            ]
                        )
                    chunk["log_probs"] = log_probs
                    chunk["top_logprobs"] = top_logprobs

            if output.finish_reason:
                chunk["finish_reason"] = output.finish_reason
            elif response.finished:
                chunk["finish_reason"] = "unknown"
            if output.stop_reason:
                chunk["stop_reason"] = output.stop_reason
            if output.finish_reason or response.finished:
                cached_tokens = min(
                    prompt_tokens, int(getattr(response, "cached_tokens", 0) or 0)
                )
                chunk["completion_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": total_tokens,
                    "total_tokens": prompt_tokens + total_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                }

            expected.append(chunk)
            cursor[output_idx] = len(token_ids)
    return expected


@dataclass
class _NativeReplay:
    handlers: dict[int, AggregatedHandler]
    actual_handlers: dict[int, AggregatedHandler]
    responses: dict[int, list[Any]]

    def _run(self, handlers: dict[int, AggregatedHandler], length: int) -> list[dict]:
        request = {
            "id": "native-logprobs-benchmark",
            "token_ids": [1, 3, 4],
            "stop_conditions": {"max_tokens": length},
            "sampling_options": {"temperature": 0.0, "n": _NUM_CHOICES},
            "output_options": {"logprobs": 2},
        }
        output: list[dict] = []

        async def consume() -> None:
            async for chunk in handlers[length].generate(request, _Context()):
                output.append(chunk)

        asyncio.run(consume())
        return output

    def run_32(self) -> None:
        self._run(self.handlers, 32)

    def run_256(self) -> None:
        self._run(self.handlers, 256)

    def run_actual_32(self) -> list[dict]:
        return self._run(self.actual_handlers, 32)


def _assert_native_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("native TensorRT-LLM benchmark requires CUDA")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    if "L4" not in properties.name or (properties.major, properties.minor) != (8, 9):
        pytest.skip(f"benchmark is calibrated for L4 (sm89), found {properties.name}")
    torch.cuda.synchronize(device)


def _capture_native_stream(llm: Any, length: int) -> list[Any]:
    from tensorrt_llm.llmapi import SamplingParams

    async def collect() -> list[Any]:
        result = llm.generate_async(
            inputs=[1, 3, 4],
            sampling_params=SamplingParams(
                max_tokens=length,
                logprobs=2,
                temperature=0.0,
                end_id=_MODEL_VOCAB_SIZE - 1,
                n=_NUM_CHOICES,
                best_of=_NUM_CHOICES,
            ),
            streaming=True,
        )
        return [_snapshot_native_response(response) async for response in result]

    responses = asyncio.run(collect())
    import torch

    torch.cuda.synchronize()
    return responses


@pytest.fixture(scope="module")
def native_replay() -> _NativeReplay:
    from dynamo.trtllm.constants import DisaggregationMode
    from dynamo.trtllm.request_handlers.aggregated_handler import AggregatedHandler
    from dynamo.trtllm.request_handlers.handler_base import RequestHandlerConfig

    os.environ.setdefault("TLLM_ALLOW_N_GREEDY_DECODING", "1")
    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    _assert_native_gpu()
    model_dir = os.environ.get("DYN_NATIVE_MODEL")
    if not model_dir:
        pytest.fail("DYN_NATIVE_MODEL must name the generated native model")

    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        backend="pytorch",
        dtype="float16",
        max_batch_size=_NUM_CHOICES,
        max_num_tokens=_MODEL_MAX_SEQ_LEN,
        max_seq_len=_MODEL_MAX_SEQ_LEN,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.1,
            tokens_per_block=16,
        ),
        # Keep native proof setup in one process so the check does not depend on
        # a separate Ray GCS service. This still executes TensorRT-LLM on CUDA.
        orchestrator_type=None,
    )

    responses = {
        length: _capture_native_stream(llm, length) for length in _STREAM_LENGTHS
    }
    for length, stream in responses.items():
        _assert_cumulative_stream(stream, length)

    handlers = {
        length: AggregatedHandler(
            RequestHandlerConfig(
                engine=_ReplayEngine(_ReplayLLM(stream)),
                default_sampling_params=SamplingParams(
                    max_tokens=length,
                    logprobs=2,
                    temperature=0.0,
                    end_id=_MODEL_VOCAB_SIZE - 1,
                    n=_NUM_CHOICES,
                    best_of=_NUM_CHOICES,
                ),
                publisher=None,
                disaggregation_mode=DisaggregationMode.AGGREGATED,
                max_seq_len=_MODEL_MAX_SEQ_LEN,
            )
        )
        for length, stream in responses.items()
    }
    actual_handlers = {
        length: AggregatedHandler(
            RequestHandlerConfig(
                engine=_ReplayEngine(_SnapshotLLM(llm)),
                default_sampling_params=SamplingParams(
                    max_tokens=length,
                    logprobs=2,
                    temperature=0.0,
                    end_id=_MODEL_VOCAB_SIZE - 1,
                    n=_NUM_CHOICES,
                    best_of=_NUM_CHOICES,
                ),
                publisher=None,
                disaggregation_mode=DisaggregationMode.AGGREGATED,
                max_seq_len=_MODEL_MAX_SEQ_LEN,
            )
        )
        for length in _STREAM_LENGTHS
    }
    replay = _NativeReplay(
        handlers=handlers,
        actual_handlers=actual_handlers,
        responses=responses,
    )
    try:
        yield replay
    finally:
        llm.shutdown()


def test_native_logprobs_stream_handler_32(benchmark, native_replay: _NativeReplay):
    benchmark(native_replay.run_32)


def test_native_logprobs_stream_handler_256(benchmark, native_replay: _NativeReplay):
    benchmark(native_replay.run_256)


def test_native_logprobs_stream_handler_actual_backend(native_replay: _NativeReplay):
    chunks = native_replay.run_actual_32()
    import torch

    torch.cuda.synchronize()
    assert chunks
    assert {chunk["index"] for chunk in chunks} == set(range(_NUM_CHOICES))
    for chunk in chunks:
        token_ids = chunk["token_ids"]
        assert len(token_ids) == len(chunk.get("log_probs", []))
        assert len(token_ids) == len(chunk.get("top_logprobs", []))
    finish_chunks = [chunk for chunk in chunks if "finish_reason" in chunk]
    assert finish_chunks
    assert all(chunk["finish_reason"] == "length" for chunk in finish_chunks)
    for chunk in finish_chunks:
        usage = chunk["completion_usage"]
        assert usage["prompt_tokens"] == 3
        assert usage["completion_tokens"] >= len(chunk["token_ids"])
        assert usage["total_tokens"] == 3 + usage["completion_tokens"]
        assert 0 <= usage["prompt_tokens_details"]["cached_tokens"] <= 3


def test_native_logprobs_stream_handler_preserves_wire_values(
    native_replay: _NativeReplay,
):
    chunks = native_replay._run(native_replay.handlers, 32)
    expected = _native_wire_chunks(native_replay.responses[32], prompt_tokens=3)
    assert chunks == expected


def _write_native_model(model_dir: Path) -> None:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1729)
    config = LlamaConfig(
        vocab_size=_MODEL_VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=_MODEL_MAX_SEQ_LEN,
        bos_token_id=1,
        eos_token_id=_MODEL_VOCAB_SIZE - 1,
        pad_token_id=0,
        torch_dtype=torch.float16,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    LlamaForCausalLM(config).save_pretrained(model_dir, safe_serialization=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-model", type=Path)
    args = parser.parse_args()
    if args.generate_model is None:
        parser.error("--generate-model is required")
    _write_native_model(args.generate_model)


if __name__ == "__main__":
    main()
