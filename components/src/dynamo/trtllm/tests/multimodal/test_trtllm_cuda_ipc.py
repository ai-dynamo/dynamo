# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA IPC embedding extraction utilities.

DIAGNOSTIC COPY: identical test semantics to the original, plus instrumentation
that records where each spawned child spends its time and, when the parent's
result wait times out, snapshots processes, /dev/shm, /tmp and the children's
Python stacks. Not for merge.
"""

import faulthandler
import os
import signal
import subprocess
import sys
import time

_T0 = time.monotonic()


def _diag(msg: str) -> None:
    print(
        f"[ipcdiag pid={os.getpid()} ppid={os.getppid()} t={time.monotonic() - _T0:.1f}s] {msg}",
        file=sys.stderr,
        flush=True,
    )


import multiprocessing as _mp_probe  # noqa: E402

_IS_CHILD = _mp_probe.parent_process() is not None
if _IS_CHILD:
    # Children: dump all-thread stacks on SIGUSR1 and every 20s while alive.
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)
_diag(f"module import start child={_IS_CHILD}")

import asyncio  # noqa: E402
import multiprocessing as mp  # noqa: E402
import queue as _queue  # noqa: E402
from multiprocessing.synchronize import Event as EventType  # noqa: E402
from typing import Any, Callable  # noqa: E402

import pytest  # noqa: E402

_diag("importing torch")
import torch  # noqa: E402

_diag("torch imported")

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
_diag("importing tensorrt_llm shared_tensor")
from tensorrt_llm._torch.shared_tensor.shared_tensor import (  # noqa: E402
    SharedTensorContainer,
    _SharedTensorRebuildMethodRegistry,
)

_diag("tensorrt_llm shared_tensor imported")

from dynamo.trtllm.multimodal.cuda_ipc import (  # noqa: E402
    extract_embeddings_from_handles,
)

_diag("module import done")

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
        _diag("producer: creating tensor")
        tensor = create_tensor()
        _diag("producer: tensor created, sharing")

        # Share via CUDA IPC
        container = SharedTensorContainer.from_tensor(tensor)
        handle = container.dump_to_dict()

        handle_queue.put(handle)
        _diag("producer: handle queued, waiting for done_event")
        # Keep process alive until consumer is done
        done_event.wait()
        _diag("producer: done_event seen, exiting")
    except Exception as e:
        print(f"Producer error: {e}")
        _diag(f"producer: EXCEPTION {type(e).__name__}: {e}")
        raise


def consumer_process(
    handle_queue: mp.Queue, result_queue: mp.Queue, done_event: EventType
):
    """Consumer: receives handle and extracts embedding via CUDA IPC."""
    try:
        _diag("consumer: registry initialize")
        # Initialize shared tensor rebuild method registry
        _SharedTensorRebuildMethodRegistry.initialize()

        _diag("consumer: waiting for handle")
        # Receive handle
        handle = handle_queue.get(timeout=10)

        _diag("consumer: handle received, extracting")
        # Extract embedding via CUDA IPC - pass list of handles directly (async)
        result = asyncio.run(extract_embeddings_from_handles([handle]))

        _diag("consumer: extracted, putting result")
        # Send result
        result_queue.put(result[0])
        _diag("consumer: result put")
    except Exception as e:
        print(f"Consumer error: {e}")
        _diag(f"consumer: EXCEPTION {type(e).__name__}: {e}")
        raise
    finally:
        # Always signal producer to exit
        done_event.set()
        _diag("consumer: done_event set")


def _snapshot(procs) -> str:
    cmds = [
        "ps -eo pid,ppid,etimes,stat,wchan:32,rss,cmd --forest",
        "df -h /dev/shm; du -sh /dev/shm 2>/dev/null; ls -la /dev/shm | wc -l; ls -la /dev/shm | tail -20",
        "ls -la /tmp | grep -iE 'ompi|orte|prte|pmix|ucx|openmpi' | head -40",
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv",
        "env | grep -E '^(OPAL|OMPI|PRTE|PMIX|UCX|CUDA_VISIBLE|LD_LIBRARY_PATH|TLLM)' | cut -c1-240",
    ]
    out = []
    for c in cmds:
        try:
            r = subprocess.run(
                ["bash", "-c", c], capture_output=True, text=True, timeout=20
            )
            out.append(f"$ {c}\n{r.stdout}{r.stderr}")
        except Exception as e:  # noqa: BLE001
            out.append(f"$ {c}\n<failed: {e}>")
    for name, p in procs:
        out.append(f"{name}: pid={p.pid} alive={p.is_alive()} exitcode={p.exitcode}")
    return "\n".join(out)


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

        _diag("parent: starting children")
        producer.start()
        consumer.start()
        _diag(f"parent: started producer={producer.pid} consumer={consumer.pid}")

        # Get result tensor
        try:
            result = result_queue.get(timeout=30)
        except _queue.Empty:
            _diag("parent: TIMEOUT waiting for result; signalling children for stacks")
            for p in (producer, consumer):
                if p.is_alive():
                    try:
                        os.kill(p.pid, signal.SIGUSR1)
                    except OSError:
                        pass
            time.sleep(3)
            snap = _snapshot([("producer", producer), ("consumer", consumer)])
            print(
                f"=== ipcdiag snapshot ===\n{snap}\n=== end snapshot ===",
                file=sys.stderr,
                flush=True,
            )
            raise
        _diag("parent: result received")

        consumer.join(timeout=10)
        producer.join(timeout=10)

        # Verify against expected tensor
        expected = _create_tensor_on_gpu().cpu()
        assert result.shape == expected.shape
        assert result.device.type == "cpu"
        assert torch.equal(result, expected)
