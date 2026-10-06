# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.gpu_quiescence import (
    gpu_quiescence_provider_configured,
    prove_predecessor_gpu_quiescence,
    terminate_current_gpu_cohort_sync,
    wait_for_predecessor_gpu_quiescence_sync,
)


def test_quiescence_provider_configuration_is_backend_aware(monkeypatch):
    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", raising=False)
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "disabled")
    monkeypatch.setenv("DYN_VLLM_GMS_GPU_QUIESCENCE_COMMAND", "/bin/true")

    assert gpu_quiescence_provider_configured("vllm")
    assert not gpu_quiescence_provider_configured("sglang")


def test_process_lifetime_proof_requires_retired_cohort_and_no_mps(
    monkeypatch, tmp_path
):
    from gpu_memory_service.integrations.common import gpu_quiescence
    from gpu_memory_service.integrations.common.process_lifecycle import (
        acquire_writer_guard,
        retire_writer_cohort,
    )

    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", raising=False)
    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", raising=False)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.setattr(gpu_quiescence, "_mps_client_possible", lambda: False)
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "process-lifetime")
    assert not gpu_quiescence.gpu_quiescence_provider_configured("vllm")
    monkeypatch.setenv("DYN_GMS_EXPERIMENTAL_PROCESS_LIFETIME_RECLAIM", "1")
    assert gpu_quiescence.gpu_quiescence_provider_configured("vllm")
    cohort = tmp_path / "old-cohort"
    cohort.touch()
    guard = acquire_writer_guard(cohort)
    try:
        assert not gpu_quiescence.prove_predecessor_gpu_quiescence_sync(
            backend_name="vllm", predecessor_cohort=str(cohort)
        ).quiesced
    finally:
        import os

        os.close(guard)
    asyncio.run(retire_writer_cohort(cohort))
    proof = gpu_quiescence.prove_predecessor_gpu_quiescence_sync(
        backend_name="vllm", predecessor_cohort=str(cohort)
    )
    assert proof.quiesced and proof.provider == "process-lifetime"
    monkeypatch.setattr(gpu_quiescence, "_mps_client_possible", lambda: True)
    assert not gpu_quiescence.prove_predecessor_gpu_quiescence_sync(
        backend_name="vllm", predecessor_cohort=str(cohort)
    ).quiesced


def test_terminate_current_cohort_targets_live_identity(monkeypatch):
    calls = []

    class Session:
        def __init__(self, *args):
            calls.append(("connect", args))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def quiesce_gpu_cohort(self, **kwargs):
            calls.append(("quiesce", kwargs))
            return SimpleNamespace(
                quiesced=True, provider="gms-mps", detail="proven", elapsed_ms=1.0
            )

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("GMS_VLLM_WRITER_COHORT_PATH", "/shared/cohort-a")
    monkeypatch.setattr("gpu_memory_service.client.session._GMSClientSession", Session)

    proof = terminate_current_gpu_cohort_sync(backend_name="vllm")

    assert proof.quiesced
    assert calls[1] == (
        "quiesce",
        {
            "backend": "vllm",
            "predecessor_cohort": "/shared/cohort-a",
            "successor_cohort": "/shared/cohort-a.rank-loss-successor",
            "terminate_host": True,
        },
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_frozen_current_cohort_bounds_proof_rpc(monkeypatch, backend):
    calls = []

    class Session:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def quiesce_gpu_cohort(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                quiesced=False,
                provider="quarantine-only",
                detail="timeout",
                elapsed_ms=0,
            )

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("DYN_GMS_FAILOVER_FROZEN_PREDECESSOR", "1")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_TIMEOUT_SECS", "1.5")
    monkeypatch.setenv(f"GMS_{backend.upper()}_WRITER_COHORT_PATH", "/shared/old")
    monkeypatch.setattr("gpu_memory_service.client.session._GMSClientSession", Session)

    proof = terminate_current_gpu_cohort_sync(backend_name=backend)

    assert not proof.quiesced
    assert calls[0]["response_timeout_ms"] == 2_000


def test_takeover_proof_reuses_precrash_gpu_registration_session(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    calls = []

    class Session:
        def __init__(self, *_args):
            calls.append("connect")

        def register_gpu_client(self, **_kwargs):
            calls.append("register")
            return True

        def quiesce_gpu_cohort(self, **_kwargs):
            calls.append("prove")
            return SimpleNamespace(
                quiesced=True, provider="gms-mps", detail="proven", elapsed_ms=1.0
            )

        def close(self):
            calls.append("close")

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("GMS_SGLANG_WRITER_COHORT_PATH", "/shared/shadow")
    monkeypatch.setenv("GMS_SGLANG_VMM_IPC_SOCKET", "/tmp/test-gms-proof.sock")
    monkeypatch.setattr(gpu_quiescence, "_proof_sessions", {})
    monkeypatch.setattr("gpu_memory_service.client.session._GMSClientSession", Session)

    assert (
        gpu_quiescence.register_gpu_client(
            backend_name="sglang", device=0, cohort="/shared/shadow"
        )
        is None
    )
    assert gpu_quiescence.prove_predecessor_gpu_quiescence_sync(
        backend_name="sglang", predecessor_cohort="/shared/primary"
    ).quiesced
    assert calls == ["connect", "register", "prove"]


def test_failed_cached_proof_discards_session_and_fails_closed(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    calls = []

    class Session:
        def __init__(self, *_args):
            calls.append("connect")

        def quiesce_gpu_cohort(self, **_kwargs):
            calls.append("prove")
            if calls.count("prove") == 1:
                raise ConnectionError("GMS connection failed")
            return SimpleNamespace(
                quiesced=True, provider="gms-mps", detail="proven", elapsed_ms=1.0
            )

        def close(self):
            calls.append("close")

    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("GMS_SGLANG_WRITER_COHORT_PATH", "/shared/shadow")
    monkeypatch.setenv("GMS_SGLANG_VMM_IPC_SOCKET", "/tmp/test-gms-proof.sock")
    monkeypatch.setattr(gpu_quiescence, "_proof_sessions", {})
    monkeypatch.setattr("gpu_memory_service.client.session._GMSClientSession", Session)

    with pytest.raises(ConnectionError, match="GMS connection failed"):
        gpu_quiescence.prove_predecessor_gpu_quiescence_sync(
            backend_name="sglang", predecessor_cohort="/shared/primary"
        )
    assert gpu_quiescence.prove_predecessor_gpu_quiescence_sync(
        backend_name="sglang", predecessor_cohort="/shared/primary"
    ).quiesced
    assert calls == ["connect", "prove", "close", "connect", "prove"]


def test_wait_for_quiescence_retries_authoritative_proof(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    attempts = [
        SimpleNamespace(quiesced=False, detail="context retiring"),
        SimpleNamespace(quiesced=True, detail="proven"),
    ]
    monkeypatch.setattr(
        gpu_quiescence,
        "prove_predecessor_gpu_quiescence_sync",
        lambda **_kwargs: attempts.pop(0),
    )
    monkeypatch.setattr(gpu_quiescence.time, "sleep", lambda _seconds: None)

    proof = wait_for_predecessor_gpu_quiescence_sync(
        backend_name="vllm",
        predecessor_cohort=None,
        timeout_s=1.0,
    )

    assert proof.quiesced is True
    assert attempts == []


def test_wait_for_quiescence_deadline_fails_closed(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    rejected = SimpleNamespace(quiesced=False, detail="not proven")
    monkeypatch.setattr(
        gpu_quiescence,
        "prove_predecessor_gpu_quiescence_sync",
        lambda **_kwargs: rejected,
    )

    proof = wait_for_predecessor_gpu_quiescence_sync(
        backend_name="vllm",
        predecessor_cohort=None,
        timeout_s=0.0,
    )

    assert proof is rejected


@pytest.mark.asyncio
async def test_quiescence_defaults_to_quarantine_only(
    monkeypatch,
):
    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_COMMAND", raising=False)
    monkeypatch.delenv("DYN_VLLM_GMS_GPU_QUIESCENCE_COMMAND", raising=False)
    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", raising=False)

    proof = await prove_predecessor_gpu_quiescence(
        backend_name="vllm", predecessor_cohort=None
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


def _fake_mps_control(tmp_path, servers: dict[str, list[str]], detach_after: int = 0):
    """A control binary whose server/client tables live in a JSON file."""
    import json
    import sys

    state = tmp_path / "mps-state.json"
    state.write_text(
        json.dumps({"servers": servers, "calls": 0, "detach_after": detach_after})
    )
    script = tmp_path / "fake-mps-control"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"path = {str(state)!r}\n"
        "state = json.load(open(path))\n"
        "state['calls'] += 1\n"
        "if state['detach_after'] and state['calls'] > state['detach_after']:\n"
        "    for server in state['servers']:\n"
        "        state['servers'][server] = []\n"
        "words = sys.stdin.read().split()\n"
        "if words[0] == 'get_server_list':\n"
        "    print('\\n'.join(state['servers']))\n"
        "elif words[0] == 'get_client_list':\n"
        "    print('\\n'.join(state['servers'].get(words[1], [])))\n"
        "elif words[0] == 'shutdown_server':\n"
        "    state['servers'].pop(words[1], None)\n"
        "json.dump(state, open(path, 'w'))\n"
    )
    script.chmod(0o755)
    return script, state


def _mps_recycle_env(monkeypatch, tmp_path, script):
    from gpu_memory_service.integrations.common import gpu_quiescence as gq

    monkeypatch.setattr(gq, "_mps_domain_claim_fd", None)
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DYN_GMS_MPS_CONTROL_BINARY", str(script))
    monkeypatch.setenv("DYN_GMS_MPS_RECYCLE_WAIT_SECS", "2")
    monkeypatch.setenv("DYN_GMS_MPS_RECYCLE_SETTLE_SECS", "0.3")
    return gq


def _release_mps_claim(gq):
    if gq._mps_domain_claim_fd is not None:
        os.close(gq._mps_domain_claim_fd)
        gq._mps_domain_claim_fd = None


@pytest.mark.parametrize(
    ("servers", "detach_after", "remaining"),
    [
        ({"421": []}, 0, {}),  # faulted, client-less server is recycled
        ({"421": ["3505"]}, 4, {}),  # predecessor client detaches, then recycled
        ({"421": ["3505"]}, 0, {"421": ["3505"]}),  # live client: left alone
    ],
)
def test_recycle_idle_mps_servers_after_dead_engine(
    monkeypatch, tmp_path, servers, detach_after, remaining
):
    import json

    script, state = _fake_mps_control(tmp_path, servers, detach_after)
    gq = _mps_recycle_env(monkeypatch, tmp_path, script)
    # A previous engine claimed the domain and exited.
    (tmp_path / ".dynamo-sglang-engine.claim").write_bytes(b"C")
    try:
        recycled = gq.recycle_idle_mps_servers("sglang")
    finally:
        _release_mps_claim(gq)
    assert json.loads(state.read_text())["servers"] == remaining
    assert recycled == (0 if remaining else 1)


def test_recycle_idle_mps_servers_skips_first_start(monkeypatch, tmp_path):
    """GMS servers inside cuInit are attached but not yet listed as clients."""
    import json

    script, state = _fake_mps_control(tmp_path, {"500": []})
    gq = _mps_recycle_env(monkeypatch, tmp_path, script)
    try:
        assert gq.recycle_idle_mps_servers("sglang") == 0
    finally:
        _release_mps_claim(gq)
    assert json.loads(state.read_text())["servers"] == {"500": []}
    assert (tmp_path / ".dynamo-sglang-engine.claim").read_bytes() == b"C"
    # The next start in this domain is a restart after this engine exited.
    gq = _mps_recycle_env(monkeypatch, tmp_path, script)
    try:
        assert gq.recycle_idle_mps_servers("sglang") == 1
    finally:
        _release_mps_claim(gq)


def test_recycle_idle_mps_servers_skips_domain_with_live_engine(monkeypatch, tmp_path):
    import fcntl
    import json

    script, state = _fake_mps_control(tmp_path, {"421": []})
    claim = tmp_path / ".dynamo-sglang-engine.claim"
    claim.write_bytes(b"C")
    held = os.open(claim, os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_SH)
    gq = _mps_recycle_env(monkeypatch, tmp_path, script)
    try:
        assert gq.recycle_idle_mps_servers("sglang") == 0
    finally:
        _release_mps_claim(gq)
        os.close(held)
    assert json.loads(state.read_text())["servers"] == {"421": []}


def test_recycle_idle_mps_servers_requires_mps_provider(monkeypatch, tmp_path):
    from gpu_memory_service.integrations.common import gpu_quiescence as gq

    monkeypatch.delenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", raising=False)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DYN_GMS_MPS_CONTROL_BINARY", "/nonexistent")
    assert gq.recycle_idle_mps_servers("sglang") == 0


def _fake_torch(fault_after: int):
    calls = {"query": 0, "device": None}

    class _Stream:
        def __init__(self, device):
            calls["device"] = device

        def query(self):
            calls["query"] += 1
            if calls["query"] > fault_after:
                raise RuntimeError(
                    "CUDA error: an illegal memory access was encountered"
                )
            return True

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        current_device=lambda: 3,
        set_device=lambda _device: None,
        Stream=_Stream,
    )
    return SimpleNamespace(cuda=cuda), calls


def test_gpu_fault_watchdog_fails_stop_on_sticky_cuda_error(monkeypatch):
    import sys
    import threading

    from gpu_memory_service.integrations.common import gpu_quiescence as gq

    torch, calls = _fake_torch(fault_after=3)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DYN_GMS_GPU_FAULT_WATCHDOG_MS", "5")
    faulted = threading.Event()
    seen = []
    assert gq.start_gpu_fault_watchdog(
        "vllm", on_fault=lambda exc: (seen.append(str(exc)), faulted.set())
    )
    assert faulted.wait(5.0)
    assert "illegal memory access" in seen[0]
    # The error must repeat once before the watchdog fails stop.
    assert calls["device"] == 3 and calls["query"] == 5


def test_gpu_fault_watchdog_ignores_a_transient_error(monkeypatch):
    import sys
    import time as _time

    from gpu_memory_service.integrations.common import gpu_quiescence as gq

    torch, calls = _fake_torch(fault_after=10**9)
    stream_cls = torch.cuda.Stream

    class _Flaky(stream_cls):
        def query(self):
            calls["query"] += 1
            if calls["query"] == 2:
                raise RuntimeError("operation not permitted when stream is capturing")
            return True

    torch.cuda.Stream = _Flaky
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DYN_GMS_GPU_FAULT_WATCHDOG_MS", "5")
    faulted = []
    assert gq.start_gpu_fault_watchdog("vllm", on_fault=faulted.append)
    _time.sleep(0.3)
    assert faulted == [] and calls["query"] > 4


def test_gpu_fault_watchdog_can_be_disabled(monkeypatch):
    import sys

    from gpu_memory_service.integrations.common import gpu_quiescence as gq

    torch, _calls = _fake_torch(fault_after=0)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DYN_GMS_GPU_FAULT_WATCHDOG_MS", "0")
    assert gq.start_gpu_fault_watchdog("vllm", on_fault=lambda _exc: None) is False


@pytest.mark.parametrize(
    ("isolation", "started"), [("process", ["sglang"]), ("mps", []), ("", [])]
)
def test_fault_watchdog_without_interlock_only_under_process_isolation(
    monkeypatch, isolation, started
):
    """Without MPS a fault never reaches the TP peers, so only the watchdog ends it."""
    from gpu_memory_service.integrations.common import gpu_quiescence as gq

    monkeypatch.setenv("DYN_GMS_GPU_ISOLATION", isolation)
    monkeypatch.delenv("DYN_GMS_EXPERIMENTAL_PROCESS_LIFETIME_RECLAIM", raising=False)
    calls = []
    monkeypatch.setattr(gq, "start_gpu_fault_watchdog", calls.append)

    gq.arm_gpu_crash_interlock(None, backend_name="sglang")

    assert calls == started
