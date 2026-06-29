# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process supervision for `StorageBackend` instances.

The supervisor wraps every backend method call. On a Python exception
it:
  1. Logs the exception with the backend name + method.
  2. Increments the `backend_call_failures_total{backend, method}`
     metric so dashboards see backend hot-spots.
  3. Counts consecutive failures. After `restart_after_failures`
     consecutive failures, calls a user-provided factory to
     re-instantiate the backend (effective "restart"). The supervisor
     then RETRIES the method once on the fresh instance. Subsequent
     successful calls reset the failure counter.

Crash-isolation caveat: a C-level SIGSEGV from a backend dependency
(say, `nixl_sys` crashing inside an RDMA op) kills the whole daemon
process. No in-process supervisor can catch that — Python never
regains control. To survive C crashes we'd need to run the backend
inside a subprocess worker; the supervisor's structure here is
intentionally compatible with that evolution (just swap the factory
for a subprocess-bootstrap).

This module is intentionally tiny (~150 LOC). It deliberately does
NOT try to be a generic process manager — that's systemd's job."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from gms_kv_ring.daemon.backend import StorageBackend

logger = logging.getLogger(__name__)


# Wrapped backend method names. The supervisor proxies these and
# applies the failure / restart policy. Other attrs pass through
# unchanged (`name`, `_slots`, etc. — read-only data is fine direct).
_SUPERVISED_METHODS = frozenset(
    {
        "demote",
        "promote",
        "get",
        "release_slot",
        "release_if_current",
        "release_engine",
        "n_slots",
        "total_bytes",
        "bytes_by_engine",
        "stats",
        "prune_older_than",
        "enforce_byte_quota",
        "enforce_per_engine_byte_quota",
        "_sweep_once",
    }
)


class BackendSupervisor:
    """Wraps a `StorageBackend`, catches Python exceptions, restarts
    on persistent failure.

    Usage:
        backend = LocalStorageBackend("/var/lib/gms/storage")
        sup = BackendSupervisor(
            backend, factory=lambda: LocalStorageBackend("/var/lib/..."),
            restart_after_failures=3,
        )
        # Use sup just like a backend:
        sup.demote(...)
        sup.promote(...)

    Read sup.failures / sup.restarts for introspection."""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        factory: Optional[Callable[[], StorageBackend]] = None,
        restart_after_failures: int = 3,
        max_restarts: int = 10,
    ) -> None:
        if restart_after_failures < 1:
            raise ValueError("restart_after_failures must be >= 1")
        self._backend = backend
        self._factory = factory
        self._restart_after = int(restart_after_failures)
        self._max_restarts = int(max_restarts)
        self._lock = threading.Lock()
        self.consecutive_failures = 0
        self.total_failures = 0
        self.restarts = 0
        self._dead = False  # set when max_restarts exceeded

    @property
    def backend(self) -> StorageBackend:
        """Current backend instance. Changes across restarts."""
        return self._backend

    @property
    def name(self) -> str:
        return self._backend.name

    @property
    def dead(self) -> bool:
        """True iff max_restarts was exceeded — backend is permanently
        unusable, every call raises immediately."""
        return self._dead

    def __getattr__(self, item: str) -> Any:
        # Called only when item isn't found in the supervisor itself.
        # If the underlying method is one we supervise, return a
        # wrapped callable; otherwise pass through.
        attr = getattr(self._backend, item)
        if item not in _SUPERVISED_METHODS or not callable(attr):
            return attr

        def _supervised(*args, **kwargs):
            return self._call(item, *args, **kwargs)

        return _supervised

    def _call(self, method: str, *args, **kwargs) -> Any:
        from gms_kv_ring.common import metrics

        if self._dead:
            raise RuntimeError(
                f"backend {self._backend.name!r} is dead "
                f"(exceeded {self._max_restarts} restarts)"
            )
        try:
            result = getattr(self._backend, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.consecutive_failures += 1
                self.total_failures += 1
                metrics.backend_call_failures.inc(
                    backend=self._backend.name,
                )
                logger.warning(
                    "backend %r method %r failed (consecutive=%d): %s",
                    self._backend.name,
                    method,
                    self.consecutive_failures,
                    exc,
                    exc_info=True,
                )
                if (
                    self.consecutive_failures >= self._restart_after
                    and self._factory is not None
                ):
                    self._restart_locked()
                    # Retry once on the fresh instance. If THIS raises,
                    # propagate — we just rebuilt the backend, so a
                    # post-restart failure is a real problem.
                    return getattr(self._backend, method)(*args, **kwargs)
            raise
        else:
            with self._lock:
                if self.consecutive_failures:
                    logger.info(
                        "backend %r recovered after %d consecutive failures",
                        self._backend.name,
                        self.consecutive_failures,
                    )
                self.consecutive_failures = 0
            return result

    def _restart_locked(self) -> None:
        """Re-instantiate the backend via the factory. Caller MUST
        hold self._lock."""
        from gms_kv_ring.common import metrics

        if self.restarts >= self._max_restarts:
            self._dead = True
            logger.error(
                "backend %r: %d restarts exceeded, marking dead",
                self._backend.name,
                self._max_restarts,
            )
            return
        old_name = self._backend.name
        try:
            new_backend = self._factory()
        except Exception:  # noqa: BLE001
            logger.error(
                "backend %r: factory raised during restart",
                old_name,
                exc_info=True,
            )
            self._dead = True
            return
        old_backend = self._backend
        self._backend = new_backend
        close = getattr(old_backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "backend %r: close failed during restart",
                    old_name,
                    exc_info=True,
                )
        self.restarts += 1
        self.consecutive_failures = 0
        metrics.backend_restarts.inc(backend=new_backend.name)
        logger.warning(
            "backend %r restarted (restart #%d after %d total failures)",
            new_backend.name,
            self.restarts,
            self.total_failures,
        )
