# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector adapters; commands and destinations come only from trusted config."""

import asyncio
import os
import signal
import time
from pathlib import Path

import aiohttp
import msgspec

from .models import Config, Plan, RuntimeSnapshot, Worker


class CollectionError(RuntimeError):
    """An acknowledged collector failure or an uncertain remote operation."""


def process_epoch(pid: int) -> str:
    """Identify a Linux process across PID reuse, including host boot identity."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    # comm may contain spaces and parentheses. Fields after its final ')' start
    # at field 3; starttime is field 22.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    # Start ticks alone can collide when independent replicas start together.
    return f"{boot}:{pid}:{fields[19]}"


def selected_workers(
    runtime: RuntimeSnapshot, plan: Plan, backend: str, **options
) -> list[Worker | None]:
    """Plan targets' workers (None for the frontend); `backend` gets `options`."""
    workers = {
        worker.target_id: msgspec.structs.replace(
            worker, start_options={**worker.start_options, **options}
        )
        if worker.backend == backend
        else worker
        for worker in runtime.workers
    }
    return [
        None if target.target_id == "frontend" else workers[target.target_id]
        for target in plan.targets
    ]


class PerfCollector:
    """Attach perf to the existing frontend, without starting it under perf."""

    def __init__(self, pid: int, epoch: str, directory: Path):
        self.pid = pid
        self.epoch = epoch
        self.directory = directory
        self.process: asyncio.subprocess.Process | None = None
        self.control = -1
        self.ack = -1
        self.evidence = {
            "target_id": "frontend",
            "process_epoch": epoch,
            "collector": "linux_perf",
            "artifact": str(directory / "frontend.perf.data"),
            "coverage": "unknown",
        }

    async def start(self) -> None:
        if process_epoch(self.pid) != self.epoch:
            raise CollectionError("Frontend process epoch changed")
        control_read, self.control = os.pipe()
        self.ack, ack_write = os.pipe()
        os.set_blocking(self.ack, False)
        command = (
            f"perf record --pid {self.pid} --event cpu-clock:u --freq 99"
            f" --call-graph dwarf --delay=-1 --control=fd:{control_read},{ack_write}"
        )
        try:
            with (self.directory / "frontend.perf.log").open("wb") as log:
                self.process = await asyncio.create_subprocess_exec(
                    *command.split(),
                    "--output",
                    self.evidence["artifact"],
                    pass_fds=(control_read, ack_write),
                    stdout=log,
                    stderr=log,
                )
        finally:
            os.close(control_read)
            os.close(ack_write)
        # ACK is the boundary at which perf confirms counters are enabled.
        await self._command("enable")
        self.evidence.update(started_at_ns=time.time_ns(), coverage="recording")

    async def _command(self, command: str) -> None:
        os.write(self.control, (command + "\n").encode())
        loop = asyncio.get_running_loop()
        ready = loop.create_future()

        def read_ack() -> None:
            data = os.read(self.ack, 128)
            if not ready.done():
                if data == b"ack\n":
                    ready.set_result(None)
                else:
                    ready.set_exception(CollectionError("perf control did not ACK"))

        loop.add_reader(self.ack, read_ack)
        try:
            await asyncio.wait_for(ready, 5.0)
        finally:
            loop.remove_reader(self.ack)

    async def stop(self) -> None:
        try:
            if self.process is not None:
                self.evidence["stop_requested_at_ns"] = time.time_ns()
                if self.process.returncode is None:
                    # Signal only the recorder we spawned, never the frontend.
                    self.process.send_signal(signal.SIGINT)
                try:
                    code = await asyncio.wait_for(self.process.wait(), 30.0)
                except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                    # A coordinator deadline may be shorter than our flush
                    # timeout. Reap only our recorder before returning failure.
                    if self.process.returncode is None:
                        self.process.kill()
                    await self.process.wait()
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    raise CollectionError(
                        "perf flush timed out; artifact is partial"
                    ) from exc
                self.evidence["stopped_at_ns"] = time.time_ns()
                artifact = Path(self.evidence["artifact"])
                if code not in (0, -signal.SIGINT) or not artifact.is_file():
                    raise CollectionError("perf failed; inspect frontend.perf.log")
                self.evidence["coverage"] = "captured_unverified"
                self.evidence["size_bytes"] = artifact.stat().st_size
        finally:
            for descriptor in (self.control, self.ack):
                if descriptor >= 0:
                    os.close(descriptor)
            self.control = self.ack = -1


class TorchCollector:
    """Use the backend's existing engine/control profiling routes."""

    def __init__(self, worker: Worker, config: Config):
        self.worker = worker
        self.config = config
        self.evidence = {
            "target_id": worker.target_id,
            "process_epoch": worker.process_epoch,
            "collector": "torch_profiler",
            "artifact_location": worker.artifact_location,
            "coverage": "unknown",
            # The coordinator can verify the separate control-process agent;
            # native start/stop hooks do not attest each engine-rank epoch.
            "identity_verification": "operator_config_only",
            "artifact_prefix": worker.start_options.get("profile_prefix")
            if worker.backend == "vllm"
            else None,
            "expected_rank_ids": worker.expected_rank_ids,
            "timestamp_scope": "control_responses_not_native_recording_bounds",
        }

    async def _request(self, action: str, body: dict, timeout: float) -> None:
        url = f"{self.worker.control_url.rstrip('/')}/engine/control/{action}_profile"
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as client,
            client.post(url, json=body) as response,
        ):
            response.raise_for_status()
            # Engine routes send JSON as text/plain and may return 200 with an error.
            result = await response.json(content_type=None)
            if not isinstance(result, dict) or result.get("status") != "ok":
                raise CollectionError(f"{self.worker.target_id}: {result}")

    async def start(self) -> None:
        self.evidence["start_requested_at_ns"] = time.time_ns()
        await self._request(
            "start", self.worker.start_options, self.config.start_timeout_s
        )
        self.evidence.update(started_at_ns=time.time_ns(), coverage="recording")

    async def stop(self) -> None:
        self.evidence["stop_requested_at_ns"] = time.time_ns()
        await self._request("stop", {}, self.config.finalize_timeout_s)
        self.evidence.update(stopped_at_ns=time.time_ns(), coverage="export_unverified")
