# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA IPC embedding extraction utilities."""

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventType
from typing import Any, Dict, List, Optional

import pytest
import torch
from tensorrt_llm._torch.shared_tensor.shared_tensor import (
    SharedTensorContainer,
    _SharedTensorRebuildMethodRegistry,
)

from dynamo.trtllm.multimodal.cuda_ipc import (
    extract_embeddings_from_disaggregated_params,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
]


@dataclass
class MockDisaggregatedParams:
    """Mock DisaggregatedParams for testing."""

    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = None


def producer_process(handle_queue: mp.Queue, done_event: EventType):
    """Producer: creates GPU tensor and shares via CUDA IPC."""
    try:
        _SharedTensorRebuildMethodRegistry.initialize()

        # Create deterministic tensor on GPU
        tensor = torch.arange(100 * 2048, dtype=torch.float32, device="cuda").reshape(100, 2048)

        # Share via CUDA IPC
        container = SharedTensorContainer.from_tensor(tensor)
        handle = container.dump_to_dict()

        handle_queue.put(handle)
        # Keep process alive until consumer is done
        done_event.wait()
    except Exception as e:
        print(f"Producer error: {e}")
        raise


def consumer_process(handle_queue: mp.Queue, result_queue: mp.Queue, done_event: EventType):
    """Consumer: receives handle and extracts embedding via CUDA IPC."""
    try:
        _SharedTensorRebuildMethodRegistry.initialize()

        # Receive handle
        handle = handle_queue.get(timeout=10)

        # Extract embedding via CUDA IPC
        params = MockDisaggregatedParams(multimodal_embedding_handles=[handle])
        result = extract_embeddings_from_disaggregated_params(params)

        # Send result
        result_queue.put(result[0])
    except Exception as e:
        print(f"Consumer error: {e}")
        raise
    finally:
        # Always signal producer to exit
        done_event.set()


class TestExtractEmbeddingsFromDisaggregatedParams:
    """Tests for extract_embeddings_from_disaggregated_params function."""

    def test_extracts_all_embeddings(self):
        """Test that embeddings are extracted successfully from GPU via CUDA IPC."""
        ctx = mp.get_context('spawn')
        handle_queue: mp.Queue[Any] = ctx.Queue()
        result_queue: mp.Queue[Any] = ctx.Queue()
        done_event = ctx.Event()

        # Start processes
        producer = ctx.Process(target=producer_process, args=(handle_queue, done_event))
        consumer = ctx.Process(target=consumer_process, args=(handle_queue, result_queue, done_event))

        producer.start()
        consumer.start()

        # Get result tensor
        result = result_queue.get(timeout=30)

        consumer.join(timeout=10)
        producer.join(timeout=10)

        # Create expected tensor
        expected = torch.arange(100 * 2048, dtype=torch.float32).reshape(100, 2048)

        # Verify
        assert result.shape == torch.Size([100, 2048])
        assert result.device.type == "cpu"
        assert torch.equal(result, expected)
