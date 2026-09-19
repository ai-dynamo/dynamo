# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from gpu_memory_service.integrations.common.gpu_quiescence import (
    gpu_quiescence_provider_configured,
    prove_predecessor_gpu_quiescence,
)


def test_quiescence_provider_configuration_is_backend_aware(monkeypatch):
    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", raising=False)
    monkeypatch.setenv("DYN_VLLM_GMS_GPU_QUIESCENCE_COMMAND", "/bin/true")

    assert gpu_quiescence_provider_configured("vllm")
    assert not gpu_quiescence_provider_configured("sglang")


@pytest.mark.asyncio
async def test_quiescence_defaults_to_quarantine_only(monkeypatch):
    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", raising=False)
    monkeypatch.delenv("DYN_VLLM_GMS_GPU_QUIESCENCE_COMMAND", raising=False)

    proof = await prove_predecessor_gpu_quiescence(
        backend_name="vllm", predecessor_cohort="cohort-1"
    )

    assert proof.quiesced is False
    assert proof.provider == "quarantine-only"


@pytest.mark.asyncio
async def test_quiescence_external_provider_is_argument_safe(monkeypatch, tmp_path):
    output = tmp_path / "args"
    monkeypatch.setenv(
        "DYN_VLLM_GMS_GPU_QUIESCENCE_COMMAND",
        f"/bin/sh -c 'printf \"$1:$2\" > {output}' proof {{backend}} {{cohort}}",
    )

    proof = await prove_predecessor_gpu_quiescence(
        backend_name="vllm", predecessor_cohort="cohort-2"
    )

    assert proof.quiesced is True
    assert proof.elapsed_ms >= 0
    assert output.read_text() == "vllm:cohort-2"


@pytest.mark.asyncio
async def test_quiescence_provider_rejection_preserves_quarantine(monkeypatch):
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", "/bin/false")

    proof = await prove_predecessor_gpu_quiescence(
        backend_name="sglang", predecessor_cohort="cohort-3"
    )

    assert proof.quiesced is False
    assert proof.provider == "external-command"


@pytest.mark.asyncio
async def test_quiescence_provider_timeout_fails_closed(monkeypatch):
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", "/bin/sh -c 'sleep 1'")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "0.05")

    with pytest.raises(RuntimeError, match="timed out"):
        await prove_predecessor_gpu_quiescence(
            backend_name="sglang", predecessor_cohort="cohort-4"
        )


@pytest.mark.asyncio
async def test_quiescence_provider_is_terminated_on_cancellation(monkeypatch):
    started = asyncio.Event()

    class Process:
        returncode = 0
        killed = False

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True

        async def wait(self):
            return 0

    process = Process()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", "/bin/true")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    task = asyncio.create_task(
        prove_predecessor_gpu_quiescence(
            backend_name="vllm", predecessor_cohort="cohort-5"
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed
