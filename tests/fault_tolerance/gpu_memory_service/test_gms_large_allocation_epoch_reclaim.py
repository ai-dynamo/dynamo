# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import pynvml

from gpu_memory_service.common import cuda_vmm_utils
from gpu_memory_service.common.types import GrantedLockType
from gpu_memory_service.server.allocations import GMSAllocationManager
from gpu_memory_service.server.epochs import GMSEpochManager


def _gpu_memory_free_bytes(device: int = 0) -> int:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
    finally:
        pynvml.nvmlShutdown()


@pytest.mark.asyncio
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.timeout(180)
async def test_new_epoch_large_allocation_waits_for_dead_writer_process(
    tmp_path, monkeypatch
):
    free_before = _gpu_memory_free_bytes()
    size = int(free_before * 0.9)
    assert size > 0

    oom_failures = 0

    def count_oom(size: int, device: int) -> tuple[bool, int]:
        nonlocal oom_failures
        allocated, handle = cuda_vmm_utils.cumem_create_tolerate_oom(size, device)
        if not allocated:
            oom_failures += 1
        return allocated, handle

    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_create_tolerate_oom",
        count_oom,
    )

    epochs = GMSEpochManager(
        GMSAllocationManager(
            device=0,
            allocation_retry_interval=0.1,
            allocation_retry_timeout=120.0,
        )
    )
    holder = None
    allocation_task = None

    try:
        epochs.on_rw_connect()
        first_epoch = epochs.active_rw_epoch_id
        first = await epochs.allocate(size, "weights", lambda: True)
        assert first.epoch_id == first_epoch

        free_after_first = _gpu_memory_free_bytes()
        assert free_after_first < free_before - (size // 2)

        _, exported_fd = epochs.export_allocation(
            GrantedLockType.RW,
            first.allocation_id,
        )
        holder_ready = tmp_path / "holder.ready"
        holder_log = tmp_path / "holder.log"
        holder_script = tmp_path / "hold_import.py"
        holder_script.write_text(
            textwrap.dedent("""
                import sys
                import time
                from pathlib import Path

                from gpu_memory_service.common.cuda_vmm_utils import (
                    cuda_ensure_initialized,
                    cuda_set_current_device,
                    cumem_get_allocation_granularity,
                    cumem_import_from_shareable_handle_close_fd,
                    cumem_map,
                    cumem_address_reserve,
                    cumem_set_access,
                )
                from gpu_memory_service.common.types import GrantedLockType

                fd = int(sys.argv[1])
                size = int(sys.argv[2])
                ready_path = Path(sys.argv[3])

                cuda_ensure_initialized()
                cuda_set_current_device(0)
                granularity = cumem_get_allocation_granularity(0)
                handle = cumem_import_from_shareable_handle_close_fd(fd)
                va = cumem_address_reserve(size, granularity)
                cumem_map(va, size, handle)
                cumem_set_access(va, size, 0, GrantedLockType.RW)
                ready_path.write_text(str(va))

                while True:
                    time.sleep(1.0)
                """),
            encoding="utf-8",
        )

        with holder_log.open("w", encoding="utf-8") as log_file:
            holder = subprocess.Popen(
                [
                    sys.executable,
                    str(holder_script),
                    str(exported_fd),
                    str(first.aligned_size),
                    str(holder_ready),
                ],
                pass_fds=[exported_fd],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        os.close(exported_fd)

        deadline = time.monotonic() + 30.0
        while not holder_ready.exists():
            assert holder.poll() is None, holder_log.read_text(encoding="utf-8")
            assert time.monotonic() < deadline, holder_log.read_text(encoding="utf-8")
            await asyncio.sleep(0.1)

        epochs.on_rw_abort()
        assert epochs.active_rw_epoch_id is None
        assert epochs.allocation_count == 0

        free_after_abort = _gpu_memory_free_bytes()
        assert free_after_abort < free_before - (size // 2)

        epochs.on_rw_connect()
        second_epoch = epochs.active_rw_epoch_id
        assert second_epoch != first_epoch

        allocation_task = asyncio.create_task(
            epochs.allocate(size, "weights", lambda: True)
        )

        deadline = time.monotonic() + 30.0
        while oom_failures == 0:
            assert holder.poll() is None, holder_log.read_text(encoding="utf-8")
            assert not allocation_task.done()
            assert time.monotonic() < deadline
            await asyncio.sleep(0.1)

        assert oom_failures > 0
        assert not allocation_task.done()
        assert epochs.active_rw_epoch_id == second_epoch

        os.killpg(os.getpgid(holder.pid), signal.SIGKILL)
        holder.wait(timeout=30.0)

        second = await asyncio.wait_for(allocation_task, timeout=120.0)
        assert second.epoch_id == second_epoch
        assert epochs.allocation_count == 1
    finally:
        if allocation_task is not None and not allocation_task.done():
            allocation_task.cancel()
            try:
                await allocation_task
            except asyncio.CancelledError:
                pass
        if holder is not None and holder.poll() is None:
            os.killpg(os.getpgid(holder.pid), signal.SIGKILL)
            holder.wait(timeout=30.0)
