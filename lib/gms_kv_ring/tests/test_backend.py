# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StorageBackend abstraction layer:
  - StorageTier still satisfies the StorageBackend ABC
  - BackendSupervisor: catches Python exceptions, restarts on
    persistent failure, tracks metrics, marks dead on max_restarts

Crash-isolation caveat (tested): we test PYTHON exceptions only.
A C-level SIGSEGV would kill the whole process and no in-process
supervisor can survive that — out of scope for this layer."""

from __future__ import annotations

import pytest


def test_storage_tier_satisfies_backend_abc():
    """StorageTier inherits from StorageBackend and implements every
    abstract method. Constructing one without arg-errors proves the
    ABC isn't missing implementations."""
    import tempfile

    from gms_kv_ring.daemon.backend import StorageBackend
    from gms_kv_ring.daemon.storage_tier import StorageTier

    with tempfile.TemporaryDirectory() as td:
        st = StorageTier(td)
        assert isinstance(st, StorageBackend)
        assert st.name == "local"


def test_storage_release_does_not_delete_replacement(tmp_path):
    import ctypes
    import zlib

    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    first_bytes = b"first generation"
    first_buf = ctypes.create_string_buffer(first_bytes)
    first = st.demote(
        "eng",
        0,
        0,
        ctypes.addressof(first_buf),
        len(first_bytes),
        zlib.crc32(first_bytes),
    )
    second_bytes = b"second generation"
    second_buf = ctypes.create_string_buffer(second_bytes)
    second = st.demote(
        "eng",
        0,
        0,
        ctypes.addressof(second_buf),
        len(second_bytes),
        zlib.crc32(second_bytes),
    )

    assert st.release_if_current("eng", 0, 0, first) is False
    assert st.get("eng", 0, 0) is second
    dst = ctypes.create_string_buffer(len(second_bytes))
    assert st.promote(
        "eng", 0, 0, ctypes.addressof(dst), len(second_bytes)
    ) == zlib.crc32(second_bytes)
    assert dst.raw == second_bytes


# BackendSupervisor
# ---------------------------------------------------------------------------


class _FlakyBackend:
    """Test fixture — a fake StorageBackend that lets us drive
    failure injection deterministically. Implements the supervised
    method names but raises if `fail_next` is set."""

    name = "flaky"

    def __init__(self):
        self.fail_next = 0  # number of upcoming calls to fail
        self.calls = 0
        self.closed = False

    def close(self):
        self.closed = True

    def demote(self, *a, **kw):
        return self._maybe_fail()

    def promote(self, *a, **kw):
        return self._maybe_fail()

    def get(self, *a, **kw):
        return self._maybe_fail()

    def release_slot(self, *a, **kw):
        return self._maybe_fail()

    def release_engine(self, *a, **kw):
        return self._maybe_fail()

    def n_slots(self, *a, **kw):
        return self._maybe_fail() or 0

    def total_bytes(self, *a, **kw):
        return self._maybe_fail() or 0

    def bytes_by_engine(self, *a, **kw):
        return self._maybe_fail() or {}

    def stats(self, *a, **kw):
        return self._maybe_fail() or {}

    def prune_older_than(self, *a, **kw):
        return self._maybe_fail() or 0

    def enforce_byte_quota(self, *a, **kw):
        return self._maybe_fail() or 0

    def enforce_per_engine_byte_quota(self, *a, **kw):
        return self._maybe_fail() or 0

    def _sweep_once(self, *a, **kw):
        v = self._maybe_fail()
        return v if v is not None else (0, 0, 0)

    def _maybe_fail(self):
        self.calls += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("synthetic flaky-backend failure")
        return None


def test_supervisor_passes_through_when_backend_healthy():
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    sup = BackendSupervisor(_FlakyBackend(), factory=_FlakyBackend)
    assert sup.n_slots() == 0
    assert sup.consecutive_failures == 0
    assert sup.total_failures == 0


def test_supervisor_propagates_single_failure_without_restart():
    """One failure < restart_after_failures; supervisor catches +
    counts but re-raises and does NOT restart."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    backend = _FlakyBackend()
    backend.fail_next = 1
    sup = BackendSupervisor(
        backend,
        factory=_FlakyBackend,
        restart_after_failures=3,
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        sup.demote(0, 0, 0, 0, 0, 0)
    assert sup.consecutive_failures == 1
    assert sup.total_failures == 1
    assert sup.restarts == 0
    assert sup.backend is backend, "no restart yet"


def test_supervisor_restarts_after_n_consecutive_failures():
    """After `restart_after_failures` failures in a row, supervisor
    calls the factory and retries on the fresh instance. The retry's
    success resets the counter."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    instances: list[_FlakyBackend] = []

    def factory():
        b = _FlakyBackend()
        instances.append(b)
        return b

    backend = factory()
    backend.fail_next = 3
    sup = BackendSupervisor(
        backend,
        factory=factory,
        restart_after_failures=3,
    )

    # First two failures count up but don't trigger restart.
    for _ in range(2):
        with pytest.raises(RuntimeError):
            sup.demote(0, 0, 0, 0, 0, 0)
    assert sup.restarts == 0

    # Third failure triggers restart + retry. The retry uses a NEW
    # backend (factory-created, no pending failures) so it succeeds.
    sup.demote(0, 0, 0, 0, 0, 0)
    assert sup.restarts == 1
    assert sup.backend is not backend
    assert sup.consecutive_failures == 0
    assert backend.closed is True


def test_supervisor_resets_counter_on_recovery():
    """A single failure followed by a success leaves the supervisor
    in clean state — no lingering restart-pending counter."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    backend = _FlakyBackend()
    backend.fail_next = 1
    sup = BackendSupervisor(
        backend,
        factory=_FlakyBackend,
        restart_after_failures=5,
    )
    with pytest.raises(RuntimeError):
        sup.demote(0, 0, 0, 0, 0, 0)
    assert sup.consecutive_failures == 1
    sup.demote(0, 0, 0, 0, 0, 0)
    assert sup.consecutive_failures == 0


def test_supervisor_marks_dead_after_max_restarts():
    """If the factory keeps producing failing backends, supervisor
    eventually marks itself dead and refuses further calls."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    def perma_failing_factory():
        b = _FlakyBackend()
        b.fail_next = 100  # always fails
        return b

    sup = BackendSupervisor(
        perma_failing_factory(),
        factory=perma_failing_factory,
        restart_after_failures=1,
        max_restarts=2,
    )
    # Burn through restarts:
    for _ in range(10):
        try:
            sup.demote(0, 0, 0, 0, 0, 0)
        except (RuntimeError,):
            pass
        if sup.dead:
            break
    assert sup.dead, (
        f"supervisor should be dead after max_restarts; "
        f"restarts={sup.restarts} dead={sup.dead}"
    )
    with pytest.raises(RuntimeError, match="dead"):
        sup.demote(0, 0, 0, 0, 0, 0)


def test_supervisor_metrics_increment():
    """Failures + restarts bump the labeled metrics so dashboards
    can see backend hot-spots in production."""
    from gms_kv_ring.common import metrics
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    backend = _FlakyBackend()
    backend.fail_next = 4
    before_fail = metrics.backend_call_failures.get(backend="flaky")
    before_restart = metrics.backend_restarts.get(backend="flaky")
    sup = BackendSupervisor(
        backend,
        factory=_FlakyBackend,
        restart_after_failures=3,
    )
    for _ in range(2):
        try:
            sup.demote(0, 0, 0, 0, 0, 0)
        except RuntimeError:
            pass
    sup.demote(0, 0, 0, 0, 0, 0)  # third failure triggers restart + retry
    after_fail = metrics.backend_call_failures.get(backend="flaky")
    after_restart = metrics.backend_restarts.get(backend="flaky")
    assert after_fail >= before_fail + 3
    assert after_restart == before_restart + 1


def test_supervisor_invalid_restart_after_rejected():
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    with pytest.raises(ValueError):
        BackendSupervisor(
            _FlakyBackend(),
            factory=_FlakyBackend,
            restart_after_failures=0,
        )


def test_supervisor_proxies_non_method_attributes():
    """Non-method backend attrs (e.g., `name`) pass through unwrapped
    — no exception handling needed for plain reads."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    sup = BackendSupervisor(_FlakyBackend(), factory=_FlakyBackend)
    assert sup.name == "flaky"


# ---------------------------------------------------------------------------
# Daemon wires supervisor by default; can be disabled
# ---------------------------------------------------------------------------


def test_daemon_wraps_backend_with_supervisor_by_default(tmp_path):
    """Daemon's storage_tier attribute is a BackendSupervisor (not a
    raw StorageTier) by default — exception-isolated path."""
    from gms_kv_ring.daemon.server import Daemon
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    d = Daemon(str(tmp_path / "s.sock"), storage_dir=str(tmp_path / "store"))
    assert isinstance(d.storage_tier, BackendSupervisor)


def test_daemon_supervise_backend_false_uses_raw_backend(tmp_path):
    """For tests that need to monkey-patch backend internals, the
    `supervise_backend=False` flag bypasses the supervisor."""
    from gms_kv_ring.daemon.server import Daemon
    from gms_kv_ring.daemon.storage_tier import StorageTier

    d = Daemon(
        str(tmp_path / "s.sock"),
        storage_dir=str(tmp_path / "store"),
        supervise_backend=False,
    )
    assert isinstance(d.storage_tier, StorageTier)


def test_daemon_accepts_custom_backend(tmp_path):
    """A caller-supplied backend overrides the default StorageTier."""
    from gms_kv_ring.daemon.server import Daemon
    from gms_kv_ring.daemon.storage_tier import StorageTier

    custom = StorageTier(str(tmp_path / "custom"))
    d = Daemon(
        str(tmp_path / "s.sock"), storage_backend=custom, supervise_backend=False
    )
    assert d.storage_tier is custom
