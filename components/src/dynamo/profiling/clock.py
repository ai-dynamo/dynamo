# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate alignment and uncertainty; never change a host's system clock."""

import time

import aiohttp

from .collectors import CollectionError
from .models import Config, Worker


def estimate_offset(
    t1: int, t2: int, t3: int, t4: int, local_elapsed_ns: int, remote_elapsed_ns: int
) -> dict:
    local_wall = t4 - t1
    remote_wall = t3 - t2
    # Clock steps invalidate the four-timestamp estimate. Allow timer read
    # overhead, but do not silently accept a negative network delay.
    if (
        local_wall < 0
        or remote_wall < 0
        or abs(local_wall - local_elapsed_ns) > 1_000_000
        or abs(remote_wall - remote_elapsed_ns) > 1_000_000
    ):
        raise CollectionError("Clock changed during sampling")
    delay = local_wall - remote_wall
    if delay < 0:
        raise CollectionError("Invalid clock sample: negative network delay")
    return {
        "offset_ns": ((t2 - t1) + (t3 - t4)) // 2,
        "uncertainty_ns": (delay + 1) // 2,
        "round_trip_ns": delay,
        "sampled_at_ns": t4,
        "scope": "control_process_host",
        "alignment": "estimated",
        "offset_convention": "worker_minus_coordinator",
    }


class WorkerClockSampler:
    async def measure(self, worker: Worker, config: Config) -> dict:
        if not worker.verify_runtime:
            return {
                "alignment": "unknown",
                "identity_verification": "operator_config_only",
                "reason": "worker_agent_not_enabled",
            }
        samples = []
        url = f"{worker.control_url.rstrip('/')}/engine/control/profiling_info"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.clock_probe_timeout_s)
        ) as client:
            for _ in range(config.clock_probe_samples):
                started = time.monotonic_ns()
                t1 = time.time_ns()
                async with client.post(url, json={}) as response:
                    response.raise_for_status()
                    result = await response.json(content_type=None)
                t4 = time.time_ns()
                elapsed = time.monotonic_ns() - started
                if (
                    not isinstance(result, dict)
                    or result.get("status") != "ok"
                    or result.get("schema_version") != 1
                ):
                    raise CollectionError("Invalid worker profiling info")
                if (
                    result.get("process_epoch") != worker.process_epoch
                    or result.get("backend") != worker.backend
                ):
                    raise CollectionError(
                        f"{worker.target_id}: worker runtime identity changed"
                    )
                fields = ("received_at_ns", "sent_at_ns", "elapsed_monotonic_ns")
                if any(type(result.get(key)) is not int for key in fields):
                    raise CollectionError("Invalid worker clock timestamps")
                t2, t3, remote_elapsed = (result[key] for key in fields)
                try:
                    sample = estimate_offset(t1, t2, t3, t4, elapsed, remote_elapsed)
                except CollectionError as exc:
                    # One preempted read must not fail the capture; keep trying.
                    rejected = exc
                    continue
                sample.update(
                    hostname=result.get("hostname"),
                    pid=result.get("pid"),
                    worker_instance_id=result.get("worker_instance_id"),
                    process_epoch=result["process_epoch"],
                    identity_verification="control_process_epoch",
                )
                samples.append(sample)
        if not samples:
            raise rejected
        return min(samples, key=lambda sample: sample["round_trip_ns"])
