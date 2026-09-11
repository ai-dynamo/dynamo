# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast failover fencing for SGLang subprocess crashes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import signal
import threading
import time
from collections.abc import Iterable
from typing import Any

from dynamo.common.gms_failover import release_attached_gms_failover_lock
from dynamo.common.utils.env import env_bool as _truthy_env

logger = logging.getLogger(__name__)


def _watchdog_enabled() -> bool:
    return _truthy_env("DYN_SGLANG_GMS_FAILOVER_CHILD_WATCHDOG", default=True)


def _poll_interval_s() -> float:
    raw = os.environ.get("DYN_SGLANG_GMS_FAILOVER_CHILD_WATCHDOG_POLL_MS", "5")
    try:
        return max(0.01, float(raw) / 1000.0)
    except ValueError:
        logger.warning(
            "Ignoring invalid DYN_SGLANG_GMS_FAILOVER_CHILD_WATCHDOG_POLL_MS=%r",
            raw,
        )
        return 0.05


_UNREGISTER_TIMEOUT_S = 0.1
_FENCE_RETRY_INTERVAL_S = 0.05


def _process_alive(proc: Any) -> bool:
    is_alive = getattr(proc, "is_alive", None)
    if callable(is_alive):
        try:
            return bool(is_alive())
        except Exception:
            logger.debug("SGLang process liveness check failed", exc_info=True)
            return True

    pid = getattr(proc, "pid", None)
    if not pid:
        return False
    return _pid_running(int(pid))


def _process_failed(proc: Any) -> bool:
    if _process_alive(proc):
        return False
    exitcode = getattr(proc, "exitcode", None)
    return exitcode != 0


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as stat_file:
            parts = stat_file.read().split()
        return len(parts) < 3 or parts[2] != "Z"
    except OSError:
        return True


def _terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        logger.warning("[GMS failover] cannot SIGKILL SGLang child pid=%s", pid)


def _request_owner_shutdown() -> None:
    """Close the failed request plane after its ownership handoff is complete.

    The TP children are already fenced, so graceful draining cannot complete;
    SIGTERM would add the normal five-second grace period and then wait on
    handlers whose engine no longer exists.  This kills only the Dynamo/SGLang
    engine process.  The GMS sidecars continue owning weights and KV memory.
    """
    os.kill(os.getpid(), signal.SIGKILL)


def _child_pids(engine: Any) -> list[int]:
    get_all_child_pids = getattr(engine, "get_all_child_pids", None)
    if callable(get_all_child_pids):
        try:
            return [int(pid) for pid in get_all_child_pids() if pid]
        except Exception:
            logger.debug("SGLang get_all_child_pids failed", exc_info=True)

    result = getattr(engine, "_scheduler_init_result", None)
    return [int(pid) for pid in getattr(result, "all_child_pids", []) if pid]


def _watchdog_processes(engine: Any) -> tuple[list[Any], list[str]]:
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    watchdog = getattr(tokenizer_manager, "_subprocess_watchdog", None)
    processes = list(getattr(watchdog, "_processes", []) or [])
    names = list(getattr(watchdog, "_names", []) or [])
    if processes and len(names) != len(processes):
        names = [f"process_{idx}" for idx in range(len(processes))]
    return processes, names


def _failed_watchdog_child(engine: Any) -> tuple[str, Any] | None:
    processes, names = _watchdog_processes(engine)
    for proc, name in zip(processes, names):
        if _process_failed(proc):
            return str(name), proc
    return None


def _scheduler_dead(engine: Any) -> bool:
    if _failed_watchdog_child(engine) is not None:
        return True

    pids = _child_pids(engine)
    return bool(pids) and any(not _pid_running(pid) for pid in pids)


def _fence_children(engine: Any, *, wait_s: float = 0.25) -> bool:
    pids = set(_child_pids(engine))
    processes, _names = _watchdog_processes(engine)
    pids.update(int(proc.pid) for proc in processes if getattr(proc, "pid", None))
    for pid in pids:
        if _pid_running(pid):
            _terminate_pid(pid)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if all(not _pid_running(pid) for pid in pids):
            return True
        time.sleep(0.01)

    alive = [pid for pid in pids if _pid_running(pid)]
    if alive:
        logger.warning(
            "[GMS failover] SGLang child fence timed out; still-running pids=%s",
            alive,
        )
        return False
    return True


class SGLangGmsFailoverChildWatchdog:
    """Release active ownership promptly after fenced SGLang child failure."""

    def __init__(
        self, target: Any, engine: Any, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._target = target
        self._engine = engine
        self._loop = loop
        self._stop = threading.Event()
        self._released = threading.Event()
        self._shutdown_requested = threading.Event()
        self._trigger_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._fence_thread: threading.Thread | None = None
        self._previous_sigquit_handler: Any = None
        self._previous_sigquit_callback: tuple[Any, tuple[Any, ...]] | None = None
        self._sigquit_uses_asyncio = False
        self._sigquit_handler: Any = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._install_sigquit_hook()
        self._install_subprocess_watchdog_hook()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-gms-failover-child-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        current = threading.current_thread()
        if self._thread is not None and self._thread is not current:
            self._thread.join(timeout=0.2)
            self._thread = None
        if self._fence_thread is not None and self._fence_thread is not current:
            self._fence_thread.join(timeout=0.3)
            self._fence_thread = None
        self._restore_sigquit_hook()

    def _run(self) -> None:
        interval = _poll_interval_s()
        while not self._stop.wait(interval):
            if (
                self._released.is_set()
                or getattr(self._target, "_gms_failover_lock", None) is None
            ):
                return
            failed = _failed_watchdog_child(self._engine)
            if failed is None and not _scheduler_dead(self._engine):
                continue

            name = failed[0] if failed is not None else "pid"
            self._trigger_failure(f"detected SGLang child failure name={name}")
            return

    def _install_sigquit_hook(self) -> None:
        """Route SGLang child-failure SIGQUIT through the fenced handoff.

        A remote TP-rank loss can make a local scheduler send SIGQUIT before
        either watchdog thread observes the dead child. SGLang's default
        handler sleeps for crash diagnostics, delaying lock release and router
        replay. In GMS failover mode the diagnostics belong to the failed
        process, while this owner must fence and hand off immediately.
        """

        # SGLang installs its running-phase signal handlers lazily with the
        # tokenizer receive loop. Initialize that idempotent loop first;
        # otherwise the first request overwrites this hook with its blocking
        # crash-diagnostics handler. This starts no inference or KV writes.
        manager = getattr(self._engine, "tokenizer_manager", None)
        initialize_loop = getattr(manager, "auto_create_handle_loop", None)
        if callable(initialize_loop):
            initialize_loop()

        def sigquit_callback() -> None:
            self._trigger_failure("SGLang SIGQUIT reported child failure")

        # SGLang installs the running-phase handler with add_signal_handler.
        # Replacing only signal.signal would leave that event-loop callback
        # active, so both handlers would run and diagnostics would still sleep.
        previous = getattr(self._loop, "_signal_handlers", {}).get(signal.SIGQUIT)
        if previous is not None:
            self._previous_sigquit_callback = (
                previous._callback,
                previous._args,
            )
        try:
            self._loop.add_signal_handler(signal.SIGQUIT, sigquit_callback)
            self._sigquit_handler = sigquit_callback
            self._sigquit_uses_asyncio = True
            logger.info("[GMS failover] hooked SGLang SIGQUIT child-failure path")
            return
        except (NotImplementedError, RuntimeError, ValueError):
            logger.debug(
                "[GMS failover] could not hook SGLang asyncio SIGQUIT; "
                "falling back to signal.signal"
            )

        def raw_sigquit_handler(_signum, _frame) -> None:
            sigquit_callback()

        try:
            self._previous_sigquit_handler = signal.getsignal(signal.SIGQUIT)
            signal.signal(signal.SIGQUIT, raw_sigquit_handler)
            self._sigquit_handler = raw_sigquit_handler
            logger.info("[GMS failover] hooked SGLang SIGQUIT child-failure path")
        except ValueError:
            logger.debug(
                "[GMS failover] could not hook SGLang SIGQUIT outside main thread"
            )

    def _restore_sigquit_hook(self) -> None:
        if self._sigquit_handler is None:
            return
        if self._sigquit_uses_asyncio:
            current = getattr(self._loop, "_signal_handlers", {}).get(signal.SIGQUIT)
            try:
                if current is not None and current._callback is self._sigquit_handler:
                    if self._previous_sigquit_callback is None:
                        self._loop.remove_signal_handler(signal.SIGQUIT)
                    else:
                        callback, args = self._previous_sigquit_callback
                        self._loop.add_signal_handler(signal.SIGQUIT, callback, *args)
            except (NotImplementedError, RuntimeError, ValueError):
                logger.debug("[GMS failover] could not restore SGLang asyncio SIGQUIT")
        else:
            try:
                if signal.getsignal(signal.SIGQUIT) is self._sigquit_handler:
                    signal.signal(signal.SIGQUIT, self._previous_sigquit_handler)
            except ValueError:
                logger.debug(
                    "[GMS failover] could not restore SGLang SIGQUIT outside main thread"
                )
        self._sigquit_handler = None
        self._previous_sigquit_handler = None
        self._previous_sigquit_callback = None
        self._sigquit_uses_asyncio = False

    def _install_subprocess_watchdog_hook(self) -> None:
        tokenizer_manager = getattr(self._engine, "tokenizer_manager", None)
        watchdog = getattr(tokenizer_manager, "_subprocess_watchdog", None)
        check_processes = getattr(watchdog, "_check_processes", None)
        if not callable(check_processes) or getattr(
            watchdog, "_dynamo_gms_failover_hooked", False
        ):
            return

        def patched_check_processes() -> bool:
            failed = _failed_watchdog_child(self._engine)
            if failed is not None:
                self._trigger_failure(
                    f"SGLang subprocess watchdog detected child failure name={failed[0]}"
                )
                # The GMS watchdog owns the fail-closed fence and handoff. Do not
                # call SGLang's original failure path: it sends SIGQUIT, whose
                # crash-diagnostics handler intentionally sleeps for five seconds
                # before closing the request-plane stream.
                return True
            return check_processes()

        watchdog._check_processes = patched_check_processes
        watchdog._dynamo_gms_failover_hooked = True
        logger.info("[GMS failover] hooked SGLang subprocess watchdog")

    def _trigger_failure(self, reason: str) -> None:
        with self._trigger_lock:
            if (
                self._released.is_set()
                or getattr(self._target, "_gms_failover_lock", None) is None
            ):
                return
            # Claim the failure before fencing. SIGQUIT, the subprocess hook,
            # and the polling thread can report the same crash concurrently.
            self._released.set()

        logger.warning(
            "[GMS failover] %s; fencing children before releasing active lock",
            reason,
        )
        self._fence_thread = threading.Thread(
            target=self._fence_and_handoff,
            name="sglang-gms-failover-fence",
            daemon=True,
        )
        self._fence_thread.start()

    def _fence_and_handoff(self) -> None:
        while not self._stop.is_set():
            try:
                fenced = _fence_children(self._engine)
            except Exception:
                fenced = False
                logger.warning(
                    "[GMS failover] SGLang child fence failed; retaining active lock",
                    exc_info=True,
                )
            if fenced:
                self._handoff_after_fence()
                return
            logger.error(
                "[GMS failover] SGLang children are not fenced; retaining active "
                "lock and retrying"
            )
            self._stop.wait(_FENCE_RETRY_INTERVAL_S)

    def _handoff_after_fence(self) -> None:
        handoff = self._release_after_fence()
        try:
            future = asyncio.run_coroutine_threadsafe(handoff, self._loop)
        except RuntimeError:
            handoff.close()
            logger.debug(
                "[GMS failover] SGLang child watchdog release could not be scheduled",
                exc_info=True,
            )
            self._request_owner_shutdown_once()
            return
        try:
            future.result(timeout=0.25)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "[GMS failover] SGLang child watchdog release did not finish before shutdown"
            )
            self._request_owner_shutdown_once()
        except Exception:
            logger.debug(
                "[GMS failover] SGLang child watchdog release failed", exc_info=True
            )
            self._request_owner_shutdown_once()

    def _request_owner_shutdown_once(self) -> None:
        with self._trigger_lock:
            if self._shutdown_requested.is_set():
                return
            self._shutdown_requested.set()
        _request_owner_shutdown()

    async def _unregister_endpoint(self) -> bool:
        endpoint = getattr(self._target, "generate_endpoint", None)
        unregister = getattr(endpoint, "unregister_endpoint_instance", None)
        if not callable(unregister):
            return True

        unregister_task = asyncio.ensure_future(unregister())
        done, _pending = await asyncio.wait(
            (unregister_task,), timeout=_UNREGISTER_TIMEOUT_S
        )
        if unregister_task not in done:
            unregister_task.cancel()
            logger.warning(
                "[GMS failover] SGLang child watchdog endpoint unregister timed out"
            )
            return False
        try:
            unregister_task.result()
        except Exception:
            logger.debug(
                "[GMS failover] SGLang child watchdog endpoint unregister failed",
                exc_info=True,
            )
            return False
        return True

    async def _release_after_fence(self) -> None:
        unregistered = await self._unregister_endpoint()
        released = False
        if unregistered:
            try:
                released = await release_attached_gms_failover_lock(
                    self._target, backend_name="sglang"
                )
            except Exception:
                logger.debug(
                    "[GMS failover] SGLang child watchdog lock release failed",
                    exc_info=True,
                )
        if not unregistered or not released:
            logger.warning(
                "[GMS failover] retaining active lock until failed owner exits"
            )

        shutdown_event = getattr(self._target, "shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()
        logger.info(
            "[GMS failover] sglang fenced handoff complete; closing request plane"
        )
        self._request_owner_shutdown_once()


def maybe_start_gms_failover_child_watchdog(
    target: Any,
    engine: Any,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> SGLangGmsFailoverChildWatchdog | None:
    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE") or not _watchdog_enabled():
        return None
    if getattr(target, "_gms_failover_lock", None) is None:
        return None

    loop = loop or asyncio.get_running_loop()
    watchdog = SGLangGmsFailoverChildWatchdog(target, engine, loop)
    target._gms_failover_child_watchdog = watchdog
    watchdog.start()
    logger.info("[GMS failover] started SGLang child watchdog")
    return watchdog


def maybe_start_rank_liveness(
    target: Any,
    engine: Any,
    *,
    node_rank: int,
    leader_host: str | None,
    loop: asyncio.AbstractEventLoop | None = None,
    expected_ranks: Iterable[int] | None = None,
    cohort_identity: str | None = None,
):
    """Start the cross-node ZMQ rank-liveness channel for SGLang.

    Worker nodes (node_rank>=1) heartbeat the leader. The leader (node_rank==0)
    monitors those heartbeats and, on a worker going silent (process death),
    reuses the child watchdog's fence+release path — so a remote rank crash is
    detected in ~one heartbeat-timeout instead of via the NCCL collective timeout.
    Local detection (child watchdog) and pure hangs (engine/NCCL watchdog) are
    unaffected; this only adds the fast cross-node *crash* path.
    """
    # Rank liveness is added later in the failover stack. Keep this earlier
    # orchestration change usable on its own, and do not import the optional
    # module unless shadow failover is actually enabled.
    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return None
    try:
        from dynamo.common import rank_liveness as rl
    except ImportError:
        logger.debug("[GMS liveness] rank-liveness support is not installed")
        return None

    if not rl.liveness_enabled():
        return None

    loop = loop or asyncio.get_running_loop()

    def trigger_handoff(reason: str) -> None:
        watchdog = getattr(target, "_gms_failover_child_watchdog", None)
        if watchdog is None:
            # Even when periodic child polling is disabled, rank liveness must
            # use the same unregister -> unlock -> owner-exit ordering.
            watchdog = SGLangGmsFailoverChildWatchdog(target, engine, loop)
            target._gms_failover_child_watchdog = watchdog
        watchdog._trigger_failure(reason)

    if node_rank >= 1:
        if not leader_host:
            logger.warning(
                "[GMS liveness] no leader host for worker rank %d; skipping", node_rank
            )
            return None

        def on_leader_lost(rank: int, reason: str) -> None:
            trigger_handoff(f"cross-node leader rank {rank} liveness lost ({reason})")

        client = rl.RankLivenessClient(
            leader_host,
            node_rank,
            connect_addr=rl.leader_connect_addr(leader_host, cohort_identity),
            on_leader_lost=on_leader_lost,
        )
        if target is not None:
            target._gms_rank_liveness_client = client
        client.start()
        return client

    def on_rank_lost(rank: int, reason: str) -> None:
        trigger_handoff(f"cross-node rank {rank} liveness lost ({reason})")

    monitor = rl.RankLivenessMonitor(
        on_rank_lost,
        bind_addr=rl.leader_bind_addr(cohort_identity),
        expected_ranks=expected_ranks,
    )
    target._gms_rank_liveness_monitor = monitor
    monitor.start()
    return monitor
    return monitor
