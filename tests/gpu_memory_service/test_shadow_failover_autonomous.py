# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autonomous (flock-driven) shadow-engine failover — vLLM, TP>=1.

Unlike the orchestrated variant (``test_shadow_failover_orchestrated.py``),
nothing in this test resumes the standby. Two ``--gms-shadow-mode`` engines come
up sharing one flock; one wins it and serves (ACTIVE), the other sleeps
(STANDBY). When the ACTIVE is SIGKILLed the kernel releases its flock, and the
STANDBY wakes *itself*, re-acquires the KV-cache RW layout, and resumes serving —
with no HTTP ``wake_up`` call. Only vLLM supports this path today.

Five stages, asserted on structured surfaces (GMS runtime state + frontend
inference), not log scraping:
  1. two engines initialize concurrently → steady state (1 ACTIVE, 1 STANDBY), no OOM
  2. inference works before failover
  3. SIGKILL the ACTIVE → the STANDBY auto-wakes via flock release (no HTTP resume)
  4. inference works after failover
  5. a fresh shadow can be added back (RO weight import → STANDBY), restoring redundancy
"""

from __future__ import annotations

import logging
import os
import signal
import time

import pytest
from gpu_memory_service.server.fsm import ServerState

from tests.gpu_memory_service.common.runtime import (
    GMSProcessManager,
    VLLMWithGMSProcess,
)
from tests.gpu_memory_service.flow_assertions import assert_completion_ok
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess

pytestmark = [pytest.mark.nightly, pytest.mark.fault_tolerance]

logger = logging.getLogger(__name__)


def _sigkill_process_group(process: ManagedProcess) -> None:
    """SIGKILL the engine's whole process group — a hard crash, the failover
    trigger we actually want to exercise (vs the orchestrated test's graceful path)."""
    pid = process.get_pid()
    if pid is None:
        raise AssertionError("cannot SIGKILL engine: no PID available")
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        logger.warning("process group for pid %d already gone", pid)
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _flock_owner(lock_path: str) -> str | None:
    """Read the failover flock's current owner ("engine-N"), or None.

    The holder stamps its id into the lock file (see FlockFailoverLock); the
    kernel hands the lock to the standby on the holder's death, which rewrites
    this — so it's the source of truth for who is ACTIVE.
    """
    try:
        with open(lock_path, "r") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None


def _split_active_standby(
    engines: dict[str, ManagedProcess], lock_path: str
) -> tuple[str, ManagedProcess, ManagedProcess]:
    owner = _flock_owner(lock_path)
    assert (
        owner in engines
    ), f"flock owner {owner!r} is not a known engine {list(engines)}"
    standby_ids = [eid for eid in engines if eid != owner]
    assert len(standby_ids) == 1, f"expected exactly one standby, got {standby_ids}"
    return owner, engines[owner], engines[standby_ids[0]]


def _wait_for_active_kv(kv_cache_gms, *, timeout: float = 120.0):
    """Poll until an ACTIVE engine holds the KV-cache RW layout.

    KV-cache RW (with allocations) is the stable signal that an engine is live
    and serving. We deliberately do NOT gate on a weights RO session: engines
    import the committed weights and then drop the RO session (the mapping is
    VA-stable without a live session), so the weights server settles into
    COMMITTED with zero readers at steady state.
    """
    deadline = time.monotonic() + timeout
    while True:
        st = kv_cache_gms.get_runtime_state()
        if st.state == ServerState.RW and st.allocation_count > 0:
            return st
        if time.monotonic() > deadline:
            raise TimeoutError(
                "kv-cache GMS never reached RW with allocations "
                f"(state={st.state}, allocs={st.allocation_count})"
            )
        time.sleep(0.5)


@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME)
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "model, tp",
    [
        pytest.param(
            FAULT_TOLERANCE_MODEL_NAME, 2, marks=pytest.mark.gpu_2, id="qwen-tp2"
        ),
    ],
)
def test_shadow_failover_autonomous_vllm(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    tmp_path,
    model,
    tp,
):
    lock_path = str(tmp_path / "failover.lock")

    with GMSProcessManager(
        request,
        VLLMWithGMSProcess,
        tp=tp,
        model=model,
        shadow=True,
        lock_path=lock_path,
    ) as manager:
        frontend_port = manager.frontend_port
        kv_cache_gms = manager.kv_cache_gms

        # ---- STAGE 1: concurrent init → steady state (1 ACTIVE, 1 STANDBY) ----
        logger.info("STAGE 1: starting two shadow engines concurrently")
        engines = manager.start_engines_concurrently(["engine-0", "engine-1"])
        # An ACTIVE engine exists: the flock winner holds the KV-cache RW layout.
        # (No OOM: start_engines_concurrently raises if either engine fails to
        # reach a healthy probe during concurrent init.)
        _wait_for_active_kv(kv_cache_gms)
        for engine_id, engine in engines.items():
            assert engine.get_pid() is not None, f"{engine_id} died during startup"
        active_id, active, standby = _split_active_standby(engines, lock_path)
        logger.info("STAGE 1 ok: ACTIVE=%s, STANDBY=%s", active_id, standby.engine_id)

        # ---- STAGE 2: inference before failover ----
        logger.info("STAGE 2: inference on the ACTIVE engine")
        assert_completion_ok(
            frontend_port,
            "The capital of France is",
            failure_message="pre-failover inference failed",
            success_message="pre-failover inference OK",
            retry_timeout=30.0,
        )

        # ---- STAGE 3: SIGKILL ACTIVE → autonomous flock-driven wake ----
        # NOTE: we never call standby.resume(). Recovery must be driven purely by
        # the kernel releasing the dead engine's flock and the standby waking
        # itself. We prove that structurally: the flock owner flips to the former
        # standby and the KV-cache RW layout is re-established.
        logger.info("STAGE 3: SIGKILL ACTIVE (%s); standby must self-wake", active_id)
        _sigkill_process_group(active)
        # Note: we don't poll active.get_pid() here — reaping the group in the
        # helper makes Popen.poll() unreliable. The flock handoff below is the
        # real proof the ACTIVE died: the kernel only releases the flock when the
        # holder's process exits, so owner == standby implies the ACTIVE is gone.

        _wait_for_active_kv(kv_cache_gms, timeout=180.0)
        new_owner = _flock_owner(lock_path)
        assert new_owner == standby.engine_id, (
            f"flock did not hand off to the standby: owner={new_owner!r}, "
            f"expected {standby.engine_id!r}"
        )
        assert standby.get_pid() is not None, "standby died during failover"
        logger.info("STAGE 3 ok: standby %s is the new ACTIVE", new_owner)

        # ---- STAGE 4: inference after failover ----
        logger.info("STAGE 4: inference on the promoted standby")
        assert_completion_ok(
            frontend_port,
            "The capital of France is",
            failure_message="post-failover inference failed",
            success_message="post-failover inference OK",
            retry_timeout=60.0,
        )

        # ---- STAGE 5: replenish a fresh standby (restore redundancy) ----
        # A new shadow imports the committed weights and parks in STANDBY.
        # start_engine only returns once its health probe passes, which requires
        # a successful weight import + pause — so reaching past it is structural
        # proof the replacement came up as a healthy standby.
        logger.info("STAGE 5: adding a replacement shadow engine")
        replacement = manager.start_engine("engine-2")
        assert (
            replacement.get_pid() is not None
        ), "replacement shadow died during startup"
        # Redundancy restored: still exactly one ACTIVE (the flock owner is
        # unchanged — the replacement parked as STANDBY, not a second active) and
        # the cluster still serves.
        owner_after_replenish = _flock_owner(lock_path)
        assert owner_after_replenish == standby.engine_id, (
            f"replacement perturbed the active engine: owner={owner_after_replenish!r}, "
            f"expected {standby.engine_id!r}"
        )
        _wait_for_active_kv(kv_cache_gms, timeout=60.0)
        assert_completion_ok(
            frontend_port,
            "The capital of France is",
            failure_message="post-replenish inference failed",
            success_message="post-replenish inference OK",
            retry_timeout=30.0,
        )
        logger.info("STAGE 5 ok: redundancy restored (fresh STANDBY added)")
