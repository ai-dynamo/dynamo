# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA IPC embedding extraction utilities."""

import asyncio
import multiprocessing as mp
import traceback
from multiprocessing.synchronize import Event as EventType
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
]


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
    try:
        tensor = create_tensor()

        # Share via CUDA IPC
        container = SharedTensorContainer.from_tensor(tensor)
        handle = container.dump_to_dict()

        handle_queue.put(handle)
        # Keep process alive until consumer is done
        done_event.wait()
    except Exception as e:
        print(f"Producer error: {e}")
        raise


def consumer_process(
    handle_queue: mp.Queue, result_queue: mp.Queue, done_event: EventType
):
    """Consumer: receives handle and extracts embedding via CUDA IPC."""
    try:
        # Initialize shared tensor rebuild method registry
        _SharedTensorRebuildMethodRegistry.initialize()

        # Receive handle
        handle = handle_queue.get(timeout=10)

        # Extract embedding via CUDA IPC - pass list of handles directly (async)
        result = asyncio.run(extract_embeddings_from_handles([handle]))

        # Avoid sending a torch.Tensor through the queue. PyTorch's multiprocessing
        # reducer would introduce a second shared-memory transfer that is unrelated
        # to the CUDA IPC behavior under test.
        tensor = result[0]
        result_queue.put(("ok", tensor.device.type, tensor.numpy()))
    except Exception as e:
        print(f"Consumer error: {e}")
        result_queue.put(("error", traceback.format_exc()))
        raise
    finally:
        # Always signal producer to exit
        done_event.set()


class TestExtractEmbeddingsFromHandles:
    """Tests for extract_embeddings_from_handles function."""

    @pytest.mark.timeout(60)
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

            # Get the CPU result without invoking PyTorch's queue tensor reducer.
            status, *payload = result_queue.get(timeout=30)
            if status == "error":
                pytest.fail(f"CUDA IPC consumer failed:\n{payload[0]}")
            device_type, result_array = payload
        finally:
            done_event.set()
            for process in reversed(started_processes):
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            handle_queue.close()
            result_queue.close()

        # Verify against expected tensor. Build it on the GPU like the producer:
        # a float16 arange past 2048 rounds differently on CUDA and on the CPU.
        result = torch.from_numpy(result_array)
        expected = _create_tensor_on_gpu().cpu()
        assert result.shape == expected.shape
        assert device_type == "cpu"
        assert torch.equal(result, expected)
