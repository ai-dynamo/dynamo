# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-gated proof that a predecessor can no longer access shared HBM."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUQuiescenceProof:
    quiesced: bool
    provider: str
    detail: str = ""

    elapsed_ms: float = 0.0


def _backend_env(backend_name: str, suffix: str) -> str:
    backend = backend_name.upper().replace("-", "_")
    return f"DYN_{backend}_GMS_{suffix}"


def _configured_command(backend_name: str) -> str | None:
    return os.environ.get(
        _backend_env(backend_name, "GPU_QUIESCENCE_COMMAND")
    ) or os.environ.get("DYN_GMS_GPU_QUIESCENCE_COMMAND")


def gpu_quiescence_provider_configured(backend_name: str) -> bool:
    return bool((_configured_command(backend_name) or "").strip())


def _timeout_s(backend_name: str) -> float:
    raw = os.environ.get(
        _backend_env(backend_name, "GPU_QUIESCENCE_TIMEOUT_SECS"),
        os.environ.get("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "1.0"),
    )
    try:
        return max(0.05, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid GPU quiescence timeout %r", raw)
        return 1.0


async def prove_predecessor_gpu_quiescence(
    *, backend_name: str, predecessor_cohort: str | None
) -> GPUQuiescenceProof:
    """Run the configured platform proof without a shell.

    No command means quarantine-only recovery. A provider must exit zero only
    after the predecessor CUDA context is unable to issue or complete accesses
    to the shared allocation. Process death, heartbeats, traffic cessation and
    elapsed time are explicitly insufficient evidence.

    Commands are split with :func:`shlex.split`; ``{backend}`` and
    ``{cohort}`` placeholders are substituted per argument. This interface can
    host a qualified CUDA MPS ``terminate_client`` adapter or a deployment's
    equivalent context-lifecycle authority without baking either into Dynamo.
    """
    command = _configured_command(backend_name)
    if not command:
        return GPUQuiescenceProof(False, "quarantine-only", "no provider configured")
    try:
        argv = [
            arg.format(
                backend=backend_name,
                cohort=predecessor_cohort or "",
            )
            for arg in shlex.split(command)
        ]
    except (ValueError, KeyError) as exc:
        raise RuntimeError("invalid GPU quiescence command") from exc
    if not argv:
        raise RuntimeError("GPU quiescence command is empty")

    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_timeout_s(backend_name)
        )
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RuntimeError("GPU quiescence provider timed out") from exc
    detail = (stdout or stderr).decode("utf-8", errors="replace").strip()[:512]
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if process.returncode != 0:
        logger.warning(
            "GPU quiescence provider rejected reclaim backend=%s rc=%d detail=%s",
            backend_name,
            process.returncode,
            detail,
        )
        return GPUQuiescenceProof(False, "external-command", detail, elapsed_ms)
    return GPUQuiescenceProof(True, "external-command", detail, elapsed_ms)
