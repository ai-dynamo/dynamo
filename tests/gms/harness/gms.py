# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.common.protocol.messages import (
    GetEventHistoryRequest,
    GetEventHistoryResponse,
    GetRuntimeStateRequest,
    GetRuntimeStateResponse,
)
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.server.rpc import GMSRPCServer

from tests.utils.managed_process import ManagedProcess

from .runtime import DYNAMO_BIN


def _socket_has_live_gms(socket_path: str) -> bool:
    if not os.path.exists(socket_path):
        return False
    try:
        with _GMSRPCTransport(socket_path) as transport:
            transport.connect()
            transport.request(
                GetRuntimeStateRequest(),
                GetRuntimeStateResponse,
            )
    except Exception:
        return False
    return True


def _prepare_socket_path_for_launch(socket_path: str) -> None:
    if not os.path.exists(socket_path):
        return
    if _socket_has_live_gms(socket_path):
        raise RuntimeError(f"GMS already active at {socket_path}")
    os.unlink(socket_path)


class GMSServerProcess(ManagedProcess):
    def __init__(self, request, device: int, tag: str = "weights"):
        self.device = device
        self.tag = tag
        self.socket_path = get_socket_path(device, tag)

        log_dir = f"{request.node.name}_gms_{tag}_{device}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python",
                "-m",
                "gpu_memory_service",
                "--device",
                str(device),
                "--tag",
                tag,
            ],
            env={
                **os.environ,
                "PATH": f"{DYNAMO_BIN}:{os.environ.get('PATH', '')}",
                "DYN_LOG": "debug",
            },
            timeout=60,
            display_output=True,
            terminate_all_matching_process_names=False,
            log_dir=log_dir,
            display_name=f"gms_{tag}",
            health_check_funcs=[self._runtime_state_ready],
        )

    def __enter__(self):
        _prepare_socket_path_for_launch(self.socket_path)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            if os.path.exists(self.socket_path) and not _socket_has_live_gms(
                self.socket_path
            ):
                os.unlink(self.socket_path)

    def _socket_has_live_gms(self) -> bool:
        return _socket_has_live_gms(self.socket_path)

    def _prepare_socket_path_for_launch(self) -> None:
        _prepare_socket_path_for_launch(self.socket_path)

    def _runtime_state_ready(self, timeout: float = 30) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not os.path.exists(self.socket_path):
                time.sleep(0.1)
                continue
            try:
                self.get_runtime_state()
                return True
            except Exception:
                time.sleep(0.1)
                continue
            time.sleep(0.1)
        return False

    def get_runtime_state(self) -> GetRuntimeStateResponse:
        with _GMSRPCTransport(self.socket_path) as transport:
            transport.connect()
            return transport.request(
                GetRuntimeStateRequest(),
                GetRuntimeStateResponse,
            )

    def get_event_history(self) -> GetEventHistoryResponse:
        with _GMSRPCTransport(self.socket_path) as transport:
            transport.connect()
            return transport.request(
                GetEventHistoryRequest(),
                GetEventHistoryResponse,
            )


class ServerThread:
    def __init__(self, server, socket_path: str):
        self.server = server
        self.socket_path = socket_path
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._exception: BaseException | None = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._task = loop.create_task(self.server.serve())
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._exception = exc
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while True:
            if self._exception is not None:
                raise self._exception
            if self.server._server is not None and os.path.exists(self.socket_path):
                try:
                    with _GMSRPCTransport(self.socket_path) as transport:
                        transport.connect()
                        transport.request(
                            GetRuntimeStateRequest(),
                            GetRuntimeStateResponse,
                        )
                    return
                except Exception:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"GMS socket did not appear at {self.socket_path}")
            time.sleep(0.01)

    def stop(self) -> None:
        if self._loop is not None:

            def cancel() -> None:
                if self.server._server is not None:
                    self.server._server.close()
                if self._task is not None:
                    self._task.cancel()

            self._loop.call_soon_threadsafe(cancel)
        self._thread.join(timeout=5)
        if self._exception is not None:
            raise self._exception
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def disconnect_rw_session(self, timeout: float = 5.0) -> None:
        if self._loop is None:
            raise RuntimeError("GMS server thread is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._disconnect_rw_session(), self._loop
        )
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise TimeoutError("Timed out disconnecting RW session") from exc

    async def _disconnect_rw_session(self) -> None:
        conn = self.server._gms._sessions._locking.rw_conn
        if conn is None:
            raise RuntimeError("No active RW session to disconnect")
        await self.server._gms.cleanup_connection(conn)


class ThreadedGMSServer:
    def __init__(self, device: int, tag: str = "weights"):
        self.device = device
        self.tag = tag
        self.socket_path = get_socket_path(device, tag)
        self.server = GMSRPCServer(self.socket_path, device)
        self._thread = ServerThread(self.server, self.socket_path)

    def __enter__(self) -> "ThreadedGMSServer":
        _prepare_socket_path_for_launch(self.socket_path)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._thread.stop()

    def get_runtime_state(self) -> GetRuntimeStateResponse:
        with _GMSRPCTransport(self.socket_path) as transport:
            transport.connect()
            return transport.request(
                GetRuntimeStateRequest(),
                GetRuntimeStateResponse,
            )

    def get_event_history(self) -> GetEventHistoryResponse:
        with _GMSRPCTransport(self.socket_path) as transport:
            transport.connect()
            return transport.request(
                GetEventHistoryRequest(),
                GetEventHistoryResponse,
            )

    def disconnect_rw_session(self) -> None:
        self._thread.disconnect_rw_session()
