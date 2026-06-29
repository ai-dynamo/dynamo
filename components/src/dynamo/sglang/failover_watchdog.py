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
from typing import Any, Iterable

from dynamo.common.gms_failover import release_attached_gms_failover_lock

logger = logging.getLogger(__name__)


def _truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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
    # Historical test helper name. Any failed SGLang child is fatal for serving
    # and must fence all children before the old endpoint can linger in discovery.
    if _failed_watchdog_child(engine) is not None:
        return True

    pids = _child_pids(engine)
    return bool(pids) and any(not _pid_running(pid) for pid in pids)


def _fence_children(engine: Any, *, wait_s: float = 0.25) -> None:
    pids = _child_pids(engine)
    for pid in pids:
        if _pid_running(pid):
            _terminate_pid(pid)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if all(not _pid_running(pid) for pid in pids):
            return
        time.sleep(0.01)

    alive = [pid for pid in pids if _pid_running(pid)]
    if alive:
        logger.warning(
            "[GMS failover] SGLang child fence timed out; still-running pids=%s",
            alive,
        )


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
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._install_subprocess_watchdog_hook()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-gms-failover-child-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None

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
            return check_processes()

        setattr(watchdog, "_check_processes", patched_check_processes)
        setattr(watchdog, "_dynamo_gms_failover_hooked", True)
        logger.info("[GMS failover] hooked SGLang subprocess watchdog")

    def _trigger_failure(self, reason: str) -> None:
        if (
            self._released.is_set()
            or getattr(self._target, "_gms_failover_lock", None) is None
        ):
            return

        logger.warning(
            "[GMS failover] %s; fencing children before releasing active lock",
            reason,
        )
        _fence_children(self._engine)
        self._released.set()
        future = asyncio.run_coroutine_threadsafe(
            self._release_after_fence(), self._loop
        )
        try:
            future.result(timeout=0.25)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "[GMS failover] SGLang child watchdog release did not finish before shutdown"
            )
        except RuntimeError:
            logger.debug(
                "[GMS failover] SGLang child watchdog release could not be scheduled",
                exc_info=True,
            )
        except Exception:
            logger.debug(
                "[GMS failover] SGLang child watchdog release failed", exc_info=True
            )

    async def _release_after_fence(self) -> None:
        endpoint = getattr(self._target, "generate_endpoint", None)
        unregister = getattr(endpoint, "unregister_endpoint_instance", None)
        if callable(unregister):
            try:
                await unregister()
            except Exception:
                logger.debug(
                    "[GMS failover] SGLang child watchdog endpoint unregister failed",
                    exc_info=True,
                )

        await release_attached_gms_failover_lock(self._target, backend_name="sglang")
        shutdown_event = getattr(self._target, "shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()


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
    setattr(target, "_gms_failover_child_watchdog", watchdog)
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
):
    """Start the cross-node ZMQ rank-liveness channel for SGLang.

    Worker nodes (node_rank>=1) heartbeat the leader. The leader (node_rank==0)
    monitors those heartbeats and, on a worker going silent (process death),
    reuses the child watchdog's fence+release path — so a remote rank crash is
    detected in ~one heartbeat-timeout instead of via the NCCL collective timeout.
    Local detection (child watchdog) and pure hangs (engine/NCCL watchdog) are
    unaffected; this only adds the fast cross-node *crash* path.
    """
    from dynamo.common import rank_liveness as rl

    if not rl.liveness_enabled() or not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return None

    if node_rank >= 1:
        if not leader_host:
            logger.warning("[GMS liveness] no leader host for worker rank %d; skipping", node_rank)
            return None
        client = rl.RankLivenessClient(leader_host, node_rank)
        if target is not None:
            setattr(target, "_gms_rank_liveness_client", client)
        client.start()
        return client

    # Leader side: trigger failover when a worker rank dies.
    loop = loop or asyncio.get_running_loop()

    def on_rank_lost(rank: int, reason: str) -> None:
        watchdog = getattr(target, "_gms_failover_child_watchdog", None)
        if watchdog is not None:
            watchdog._trigger_failure(f"cross-node rank {rank} liveness lost ({reason})")
            return
        logger.warning(
            "[GMS liveness] rank %d lost (%s); fencing + releasing lock directly", rank, reason
        )
        try:
            _fence_children(engine)
        except Exception:
            logger.debug("[GMS liveness] fence on rank loss failed", exc_info=True)
        asyncio.run_coroutine_threadsafe(
            release_attached_gms_failover_lock(target, backend_name="sglang"), loop
        )

    monitor = rl.RankLivenessMonitor(
        on_rank_lost, expected_ranks=expected_ranks
    )
    setattr(target, "_gms_rank_liveness_monitor", monitor)
    monitor.start()
    return monitor
