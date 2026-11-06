# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spawn-owned native KvRouter processes for the FE/EPP parity stack.

The parent never imports Dynamo. Each child applies its env overlay before
loading the native runtime, so pytest collection stays lazy and a blocked
constructor cannot stall the loopback Kubernetes API.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
import time
import traceback
import uuid
from contextlib import ExitStack
from typing import Any, Mapping, NoReturn

logger = logging.getLogger(__name__)

_JOIN_S = 5.0
_SPAWN = multiprocessing.get_context("spawn")
_PLAIN = (type(None), bool, int, float, str)


def _reap_started_process(process: Any) -> None:
    """Bounded terminate/kill for a child that never became an owner."""
    errors: list[BaseException] = []
    try:
        if process.is_alive():
            process.terminate()
            process.join(_JOIN_S)
        if process.is_alive():
            process.kill()
            process.join(_JOIN_S)
    except BaseException as exc:  # noqa: BLE001 - report after the other step
        errors.append(exc)
    if process.is_alive():
        errors.append(
            RuntimeError(
                f"native router child pid={process.pid} survived startup failure"
            )
        )
    if errors:
        _raise_cleanup(errors)


def _raise_cleanup(errors: list[BaseException]) -> None:
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    detail = "; ".join(f"{type(err).__name__}: {err}" for err in errors)
    raise RuntimeError(detail) from errors[0]


class NativeRouterFailure(RuntimeError):
    """Ordinary owner failure. Not a typed routing known-gap."""

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        super().__init__(message)
        self.timed_out = timed_out


def _require_plain(value: Any) -> Any:
    if type(value) in _PLAIN:
        return value
    if type(value) is tuple:
        return tuple(_require_plain(item) for item in value)
    if type(value) is list:
        return [_require_plain(item) for item in value]
    if type(value) is dict:
        plain: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise NativeRouterFailure(
                    f"unsupported serialization key {type(key).__name__}"
                )
            plain[key] = _require_plain(item)
        return plain
    raise NativeRouterFailure(f"unsupported serialization {type(value).__name__}")


def _apply_env(overlay: Mapping[str, str | None]) -> None:
    for name, value in overlay.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _constraint_payload(constraints: Any) -> dict[str, Any]:
    required = constraints.required_taints
    preferred = constraints.preferred_taints
    required_out = None if required is None else [str(item) for item in required]
    preferred_out = (
        None
        if preferred is None
        else {str(key): float(value) for key, value in dict(preferred).items()}
    )
    return _require_plain(
        {"required_taints": required_out, "preferred_taints": preferred_out}
    )


def _build_router(endpoint: Any, spec: Mapping[str, Any], config: Any) -> Any:
    # Imported only after the child env overlay is applied.
    from dynamo.llm import KvRouter

    holder: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _create() -> None:
        try:
            holder["router"] = KvRouter(
                endpoint=endpoint,
                block_size=spec["block_size"],
                kv_router_config=config,
            )
        except BaseException as exc:  # noqa: BLE001 - returned across the pipe
            error["exc"] = exc

    thread = threading.Thread(target=_create, name="native-kv-router-ctor", daemon=True)
    thread.start()
    started = time.monotonic()
    timeout = float(spec["construct_timeout_s"])
    while thread.is_alive():
        thread.join(2.0)
        if time.monotonic() - started > timeout:
            raise TimeoutError(
                f"KvRouter initialization timed out after {timeout:.0f}s "
                f"waiting for {spec['num_workers']} workers to register"
            )
        for index, pid in enumerate(spec["worker_pids"]):
            if not _pid_alive(int(pid)):
                raise RuntimeError(
                    f"Worker {index} exited while waiting for KvRouter to find "
                    f"{spec['num_workers']} workers"
                )
    if "exc" in error:
        raise error["exc"]
    return holder["router"]


def _identity(instance: Any) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "namespace": instance.namespace,
        "component": instance.component,
        "endpoint": instance.endpoint,
    }


async def _catalog(state: dict[str, Any]) -> dict[str, Any]:
    spec = state["spec"]
    runtime = state["runtime"]

    async def _client(suffix: str) -> Any:
        endpoint = runtime.endpoint(f"{spec['namespace']}.{spec['component']}.{suffix}")
        return await endpoint.client()

    generate = await _client("generate")
    query = await _client(spec["query_endpoint"])
    record = await _client(spec["record_endpoint"])
    return {
        "generate_ids": list(generate.instance_ids()),
        "generate": [_identity(item) for item in generate.instances()],
        "query": [_identity(item) for item in query.instances()],
        "record_ids": list(record.instance_ids()),
        "record": [_identity(item) for item in record.instances()],
    }


async def _dispatch(state: dict[str, Any], message: Mapping[str, Any]) -> Any:
    op = message["op"]
    payload = message["payload"]
    remaining = float(message["deadline"]) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{op} timed out before the native operation started")
    if op == "shutdown":
        state["router"] = None
        runtime = state.pop("runtime", None)
        if runtime is not None:
            runtime.shutdown()
        state["stop"] = True
        return {"shutdown": True}
    if op == "initialize":
        if state["router"] is not None:
            raise RuntimeError("native router initialize already completed")
        spec = state["spec"]
        from dynamo.llm import KvRouterConfig
        from dynamo.runtime import DistributedRuntime

        runtime = DistributedRuntime(
            asyncio.get_running_loop(),
            spec["store_backend"],
            spec["request_plane"],
            event_plane=spec["event_plane"],
        )
        state["runtime"] = runtime
        endpoint = runtime.endpoint(
            f"{spec['namespace']}.{spec['component']}.generate"
        )
        config = KvRouterConfig(**spec["config_kwargs"])
        os.environ["DYN_ROUTER_MIN_INITIAL_WORKERS"] = str(spec["num_workers"])
        state["router"] = await asyncio.wait_for(
            asyncio.to_thread(_build_router, endpoint, spec, config),
            timeout=remaining,
        )
        return {
            "config_json": config.to_json(),
            "policy_parameters": config.policy_parameters("aggregated"),
        }
    router = state["router"]
    if router is None:
        raise RuntimeError(f"{op} requires an initialized native router")
    if op == "catalog":
        return await asyncio.wait_for(_catalog(state), timeout=remaining)
    if op == "dump_events":
        return await asyncio.wait_for(router.dump_events(), timeout=remaining)
    if op == "get_overlap_scores":
        return await asyncio.wait_for(
            router.get_overlap_scores(
                payload["token_ids"],
                include_shared=payload["include_shared"],
                cache_namespace=payload["cache_namespace"],
            ),
            timeout=remaining,
        )
    if op == "get_potential_loads":
        return await asyncio.wait_for(
            router.get_potential_loads(
                payload["token_ids"],
                cache_namespace=payload["cache_namespace"],
            ),
            timeout=remaining,
        )
    if op == "best_worker":
        kwargs = dict(payload["kwargs"])
        mode = payload["constraints_mode"]
        if mode == "value":
            from dynamo.llm import RoutingConstraints

            fields = payload["constraints"]
            built: dict[str, Any] = {}
            if fields["required_taints"] is not None:
                built["required_taints"] = set(fields["required_taints"])
            if fields["preferred_taints"] is not None:
                built["preferred_taints"] = dict(fields["preferred_taints"])
            kwargs["routing_constraints"] = RoutingConstraints(**built)
        elif mode == "none":
            kwargs["routing_constraints"] = None
        return await asyncio.wait_for(
            router.best_worker(payload["token_ids"], **kwargs),
            timeout=remaining,
        )
    if op == "free":
        await asyncio.wait_for(router.free(payload["request_id"]), timeout=remaining)
        return None
    if op == "generate_complete":
        # Child-only: preserves the helper's open retry and stream oracle.
        from tests.router.helper import send_request_via_python_kv_router

        return await asyncio.wait_for(
            send_request_via_python_kv_router(
                kv_python_router=router,
                model_name=payload["model_name"],
                token_ids=payload["token_ids"],
                stop_conditions=payload["stop_conditions"],
                worker_id=payload["worker_id"],
                dp_rank=payload["dp_rank"],
            ),
            timeout=remaining,
        )
    raise RuntimeError(f"unsupported native router operation {op}")


def _child_main(conn: Any, spec: dict[str, Any]) -> None:
    """Run one native router. Env is applied before Dynamo is imported."""
    _apply_env(spec["env"])
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    state: dict[str, Any] = {
        "spec": spec,
        "router": None,
        "runtime": None,
        "stop": False,
    }

    def _reader() -> None:
        while not state["stop"]:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                state["stop"] = True
                loop.call_soon_threadsafe(loop.stop)
                return
            if message.get("owner") != spec["owner_id"]:
                response = {
                    "id": message.get("id"),
                    "owner": spec["owner_id"],
                    "ok": False,
                    "error": {
                        "type": "NativeRouterFailure",
                        "message": "owner identity mismatch",
                        "trace": "",
                        "op": message.get("op"),
                        "pid": os.getpid(),
                    },
                }
            else:
                future = asyncio.run_coroutine_threadsafe(
                    _dispatch(state, message), loop
                )
                try:
                    result = future.result()
                    response = {
                        "id": message["id"],
                        "owner": spec["owner_id"],
                        "ok": True,
                        "result": _require_plain(result),
                    }
                except BaseException as exc:  # noqa: BLE001 - pipe boundary
                    response = {
                        "id": message.get("id"),
                        "owner": spec["owner_id"],
                        "ok": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc)[:2000],
                            "trace": traceback.format_exc()[-4000:],
                            "op": message.get("op"),
                            "pid": os.getpid(),
                        },
                    }
            try:
                conn.send(response)
            except (EOFError, OSError):
                state["stop"] = True
                loop.call_soon_threadsafe(loop.stop)
                return
            if state["stop"]:
                loop.call_soon_threadsafe(loop.stop)
                return

    threading.Thread(target=_reader, name="native-router-reader", daemon=True).start()
    try:
        loop.run_forever()
    finally:
        loop.close()
        conn.close()


class NativeRouterOwner:
    """One retained spawn child. Close is idempotent and reaps only this child."""

    def __init__(self, process: Any, conn: Any, owner_id: str, role: str) -> None:
        self._process = process
        self._conn = conn
        self._owner_id = owner_id
        self._role = role
        self._seq = 1
        self._busy = False
        self._poisoned = False
        self._closed = False
        self._conn_closed = False
        self._io_thread: threading.Thread | None = None
        self._escalation: str | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    @property
    def alive(self) -> bool:
        return bool(self._process.is_alive())

    @classmethod
    def spawn(
        cls,
        *,
        role: str,
        exit_stack: ExitStack,
        env: Mapping[str, str | None],
        config_kwargs: Mapping[str, Any],
        namespace: str,
        component: str,
        block_size: int,
        num_workers: int,
        construct_timeout_s: float,
        worker_pids: list[int],
        query_endpoint: str,
        record_endpoint: str,
    ) -> NativeRouterOwner:
        parent_conn, child_conn = _SPAWN.Pipe(duplex=True)
        spec = {
            "owner_id": uuid.uuid4().hex,
            "env": dict(env),
            "config_kwargs": dict(config_kwargs),
            "namespace": namespace,
            "component": component,
            "block_size": block_size,
            "num_workers": num_workers,
            "construct_timeout_s": construct_timeout_s,
            "worker_pids": list(worker_pids),
            "query_endpoint": query_endpoint,
            "record_endpoint": record_endpoint,
            "store_backend": "kubernetes",
            "request_plane": "tcp",
            "event_plane": "zmq",
        }
        process = _SPAWN.Process(
            target=_child_main,
            args=(child_conn, spec),
            name=f"native-router-{role}",
            daemon=False,
        )
        try:
            process.start()
        except BaseException as exc:
            for end in (child_conn, parent_conn):
                try:
                    end.close()
                except OSError:
                    pass
            retained: BaseException | None = None
            try:
                _reap_started_process(process)
            except BaseException as reap_exc:
                retained = reap_exc
            if process.is_alive():
                try:
                    exit_stack.callback(_reap_started_process, process)
                except BaseException as register_exc:
                    register_exc.native_process = process
                    raise register_exc from (retained or exc)
                survived = RuntimeError(
                    f"native router child pid={process.pid} survived startup failure"
                )
                survived.native_process = process
                raise survived from (retained or exc)
            if retained is not None:
                raise retained from exc
            raise
        child_conn.close()
        owner = cls(process, parent_conn, spec["owner_id"], role)
        try:
            exit_stack.callback(owner.close)
        except BaseException as exc:
            try:
                owner.close()
            except BaseException as close_exc:
                raise close_exc from exc
            raise
        logger.info(
            "native router owner role=%s parent_pid=%s child_pid=%s",
            role,
            os.getpid(),
            process.pid,
        )
        return owner

    def call(self, op: str, payload: Any, timeout_s: float) -> Any:
        box: dict[str, Any] = {}
        thread = self._start_io(op, payload, timeout_s, box)
        thread.join(max(0.0, timeout_s))
        return self._finish_io(op, thread, box, timed_out=thread.is_alive())

    async def call_async(self, op: str, payload: Any, timeout_s: float) -> Any:
        box: dict[str, Any] = {}
        thread = self._start_io(op, payload, timeout_s, box)
        try:
            await asyncio.wait_for(asyncio.to_thread(thread.join), timeout_s)
        except (TimeoutError, asyncio.CancelledError) as exc:
            try:
                self._poison(f"{op} async budget ended")
            except BaseException as cleanup:
                raise cleanup from exc
            raise
        return self._finish_io(op, thread, box, timed_out=thread.is_alive())

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        acknowledged = False
        if (
            not self._poisoned
            and self._escalation is None
            and self.alive
            and not self._busy
        ):
            try:
                self.call("shutdown", {}, _JOIN_S)
                acknowledged = True
            except BaseException as exc:  # noqa: BLE001 - keep cleaning up
                errors.append(exc)
        try:
            if self.alive:
                self._process.join(_JOIN_S)
        except BaseException as exc:  # noqa: BLE001 - keep cleaning up
            errors.append(exc)
        if self.alive:
            try:
                self._reap(self._escalation or "shutdown-escalation")
            except BaseException as exc:  # noqa: BLE001 - keep cleaning up
                errors.append(exc)
        else:
            try:
                self._join_io()
            except BaseException as exc:  # noqa: BLE001 - keep cleaning up
                errors.append(exc)
        try:
            self._close_conn()
        except BaseException as exc:  # noqa: BLE001 - keep cleaning up
            errors.append(exc)
        self._log_exit()
        if self.alive:
            errors.append(
                RuntimeError(
                    f"native router child pid={self.pid} role={self._role} "
                    "did not exit"
                )
            )
        elif not self._poisoned and (not acknowledged or self.exitcode != 0):
            errors.append(
                RuntimeError(
                    f"native router child pid={self.pid} role={self._role} "
                    f"exitcode={self.exitcode} alive={self.alive} "
                    f"shutdown_ack={acknowledged}"
                )
            )
        if errors:
            _raise_cleanup(errors)
        self._closed = True

    def _start_io(
        self, op: str, payload: Any, timeout_s: float, box: dict[str, Any]
    ) -> threading.Thread:
        if timeout_s <= 0:
            raise NativeRouterFailure(
                f"{op} requires a positive timeout", timed_out=True
            )
        if self._poisoned:
            raise NativeRouterFailure(f"{op} owner role={self._role} is poisoned")
        if not self.alive:
            raise NativeRouterFailure(
                f"{op} child pid={self.pid} exitcode={self.exitcode} is not alive"
            )
        if self._busy:
            raise NativeRouterFailure(f"{op} already has an in-flight native operation")
        message = _require_plain(
            {
                "id": self._seq,
                "owner": self._owner_id,
                "op": op,
                "payload": payload,
                "deadline": time.monotonic() + timeout_s,
            }
        )
        self._seq += 1
        self._busy = True

        def _io() -> None:
            try:
                self._conn.send(message)
                box["response"] = self._conn.recv()
            except BaseException as exc:  # noqa: BLE001 - parent IO boundary
                box["error"] = exc

        try:
            thread = threading.Thread(
                target=_io, name=f"native-router-io-{self.pid}", daemon=True
            )
            thread.start()
        except BaseException as exc:
            self._io_thread = None
            try:
                self._poison(f"{op} IO thread failed to start")
            except BaseException as cleanup:
                raise cleanup from exc
            raise
        self._io_thread = thread
        return thread

    def _finish_io(
        self,
        op: str,
        thread: threading.Thread,
        box: dict[str, Any],
        *,
        timed_out: bool,
    ) -> Any:
        request_id = self._seq - 1
        if timed_out or thread.is_alive() or "error" in box:
            reason = (
                "timed out" if timed_out or thread.is_alive() else "child IO failed"
            )
            detail = box.get("error")
            failure = NativeRouterFailure(
                f"{op} pid={self.pid} role={self._role} {reason}: {detail}",
                timed_out=timed_out or thread.is_alive(),
            )
            self._fail_after_reap(failure, f"{op} {reason}")
        response = box.get("response")
        if (
            type(response) is not dict
            or response.get("id") != request_id
            or response.get("owner") != self._owner_id
        ):
            self._fail_after_reap(
                NativeRouterFailure(
                    f"{op} pid={self.pid} protocol mismatch response={response!r}"
                ),
                f"{op} protocol mismatch",
            )
        if response.get("ok") is True:
            if not self.alive and not (op == "shutdown" and self.exitcode == 0):
                self._fail_after_reap(
                    NativeRouterFailure(
                        f"{op} pid={self.pid} role={self._role} "
                        f"exitcode={self.exitcode} is not alive"
                    ),
                    f"{op} child dead exitcode={self.exitcode}",
                )
            self._busy = False
            if self._io_thread is thread and not thread.is_alive():
                self._io_thread = None
            return response.get("result")
        error = response.get("error") or {}
        message = (
            f"{op} pid={error.get('pid')} {error.get('type')}: "
            f"{error.get('message')}\n{error.get('trace')}"
        )
        if error.get("type") == "TimeoutError":
            self._fail_after_reap(
                NativeRouterFailure(message, timed_out=True),
                f"{op} native timeout",
            )
        self._busy = False
        if self._io_thread is thread and not thread.is_alive():
            self._io_thread = None
        raise NativeRouterFailure(message, timed_out=False)

    def _fail_after_reap(self, failure: NativeRouterFailure, reason: str) -> NoReturn:
        try:
            self._poison(reason)
        except BaseException as cleanup:
            raise failure from cleanup
        raise failure

    def _poison(self, reason: str) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        self._escalation = reason
        self._reap(reason)

    def _reap(self, reason: str) -> None:
        self._escalation = reason
        errors: list[BaseException] = []
        try:
            if self.alive:
                self._process.terminate()
                self._process.join(_JOIN_S)
            if self.alive:
                self._process.kill()
                self._process.join(_JOIN_S)
        except BaseException as exc:  # noqa: BLE001 - report after the other steps
            errors.append(exc)
        try:
            self._join_io()
        except BaseException as exc:  # noqa: BLE001 - report after the other steps
            errors.append(exc)
        try:
            self._close_conn()
        except BaseException as exc:  # noqa: BLE001 - report after the other steps
            errors.append(exc)
        self._busy = False
        self._log_exit()
        if self.alive:
            errors.append(
                RuntimeError(
                    f"native router child pid={self.pid} role={self._role} "
                    f"still alive after escalation ({reason})"
                )
            )
        if errors:
            _raise_cleanup(errors)

    def _join_io(self) -> None:
        thread = self._io_thread
        if thread is None or thread is threading.current_thread():
            return
        if thread.ident is None:
            raise RuntimeError(
                f"native router IO thread role={self._role} pid={self.pid} "
                "was not started"
            )
        thread.join(_JOIN_S)
        if thread.is_alive():
            raise RuntimeError(
                f"native router IO thread role={self._role} pid={self.pid} "
                f"still alive after {_JOIN_S}s"
            )
        if self._io_thread is thread:
            self._io_thread = None

    def _close_conn(self) -> None:
        if self._conn_closed:
            return
        self._conn.close()
        self._conn_closed = True

    def _log_exit(self) -> None:
        logger.info(
            "native router owner role=%s child_pid=%s exitcode=%s "
            "alive=%s escalation=%s",
            self._role,
            self.pid,
            self.exitcode,
            self.alive,
            self._escalation,
        )
