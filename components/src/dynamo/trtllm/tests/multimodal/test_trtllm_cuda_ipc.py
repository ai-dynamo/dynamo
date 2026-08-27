# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA IPC embedding extraction utilities."""

import asyncio
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventType
from queue import Empty
from time import monotonic
from typing import Any, Callable

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
from tensorrt_llm._torch.shared_tensor.shared_tensor import (  # noqa: E402
    SharedTensorContainer,
    _SharedTensorRebuildMethodRegistry,
)

from dynamo.trtllm.multimodal.cuda_ipc import extract_embeddings_from_handles

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(2.0),
    pytest.mark.requested_trtllm_vram_gib(2.0),
    pytest.mark.timeout(240),
]

_RESULT_TIMEOUT_SECONDS = 180
_CHILD_JOIN_TIMEOUT_SECONDS = 10


def _create_tensor_on_gpu() -> torch.Tensor:
    """Create test tensor on GPU."""
    return torch.arange(100 * 2048, dtype=torch.float16, device="cuda").reshape(
        100, 2048
    )


def producer_process(
    create_tensor: Callable[[], torch.Tensor],
    handle_queue: mp.Queue,
    done_event: EventType,
):
    """Producer: creates GPU tensor and shares via CUDA IPC."""
    tensor = create_tensor()

    # Share via CUDA IPC
    container = SharedTensorContainer.from_tensor(tensor)
    handle = container.dump_to_dict()

    handle_queue.put(handle)
    # Keep process alive until consumer is done
    done_event.wait()


def consumer_process(
    handle_queue: mp.Queue, result_queue: mp.Queue, done_event: EventType
):
    """Consumer: receives handle and extracts embedding via CUDA IPC."""
    try:
        # Initialize shared tensor rebuild method registry
        _SharedTensorRebuildMethodRegistry.initialize()

        # Receive handle
        handle = handle_queue.get(timeout=_RESULT_TIMEOUT_SECONDS)

        # Extract embedding via CUDA IPC - pass list of handles directly (async)
        result = asyncio.run(extract_embeddings_from_handles([handle]))

        # Send a regular NumPy payload. Sending a torch.Tensor through a
        # multiprocessing queue adds a second shared-memory lifetime to this
        # CUDA-IPC test and can race the consumer process exiting.
        result_queue.put(result[0].cpu().numpy())
    finally:
        # Always signal producer to exit
        done_event.set()


def _process_state(process: mp.Process) -> str:
    return (
        f"pid={process.pid}, alive={process.is_alive()}, "
        f"exitcode={process.exitcode}"
    )


def _wait_for_consumer_result(
    result_queue: mp.Queue,
    producer: mp.Process,
    consumer: mp.Process,
    timeout: float = _RESULT_TIMEOUT_SECONDS,
) -> Any:
    """Wait for a result while surfacing child exit state promptly."""
    deadline = monotonic() + timeout
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            pytest.fail(
                "Timed out waiting for CUDA IPC consumer result after "
                f"{timeout:.0f}s; producer({_process_state(producer)}), "
                f"consumer({_process_state(consumer)}). Child tracebacks are "
                "printed in the captured subprocess output."
            )

        try:
            return result_queue.get(timeout=min(1.0, remaining))
        except Empty:
            if consumer.exitcode not in (None, 0):
                pytest.fail(
                    "CUDA IPC consumer exited before returning a result: "
                    f"{_process_state(consumer)}"
                )
            if producer.exitcode not in (None, 0):
                pytest.fail(
                    "CUDA IPC producer exited before returning a handle: "
                    f"{_process_state(producer)}"
                )


class TestExtractEmbeddingsFromHandles:
    """Tests for extract_embeddings_from_handles function."""

    def test_extracts_all_embeddings(self):
        """Test that embeddings are extracted successfully from GPU via CUDA IPC."""
        ctx = mp.get_context("spawn")
        handle_queue: mp.Queue[Any] = ctx.Queue()
        result_queue: mp.Queue[Any] = ctx.Queue()
        done_event = ctx.Event()

        # Start processes
        producer = ctx.Process(
            target=producer_process,
            args=(_create_tensor_on_gpu, handle_queue, done_event),
        )
        consumer = ctx.Process(
            target=consumer_process, args=(handle_queue, result_queue, done_event)
        )

        started_processes: list[mp.Process] = []
        try:
            producer.start()
            started_processes.append(producer)
            consumer.start()
            started_processes.append(consumer)

            result_array = _wait_for_consumer_result(result_queue, producer, consumer)
        finally:
            done_event.set()
            for process in reversed(started_processes):
                process.join(timeout=_CHILD_JOIN_TIMEOUT_SECONDS)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=_CHILD_JOIN_TIMEOUT_SECONDS)

        assert consumer.exitcode == 0, _process_state(consumer)
        assert producer.exitcode == 0, _process_state(producer)

        # Verify against expected tensor
        result = torch.from_numpy(result_array)
        expected = _create_tensor_on_gpu().cpu()
        assert result.shape == expected.shape
        assert result.device.type == "cpu"
        assert torch.equal(result, expected)
