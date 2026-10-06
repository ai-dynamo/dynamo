# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import Any, Dict, Literal, Optional

import aiohttp

from tests.router.helper import (
    generate_random_suffix,
    get_kv_indexer_command,
    get_kv_indexer_test_env,
    get_runtime,
    get_select_service_command,
    poll_for_worker_instances,
    wait_for_indexer_workers_active,
    wait_for_selection_service_ready,
)
from tests.router.mocker_config import MockerConfig
from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import (
    allocate_contiguous_ports,
    allocate_ports,
    deallocate_ports,
)

logger = logging.getLogger(__name__)

MODEL_NAME = ROUTER_MODEL_NAME
BLOCK_SIZE = 16
BASE_PORT = 9100
BASE_PORT_BOOTSTRAP = 10100
BASE_PORT_ZMQ = 11100


def _build_mocker_command(
    endpoint: str,
    store_backend: str,
    num_workers: int,
    mocker_args: MockerConfig | Mapping[str, Any],
    worker_type: Literal["prefill", "decode"] | None = None,
) -> list[str]:
    """Build the mocker CLI command with all arguments."""
    command = [
        sys.executable,
        "-m",
        "dynamo.mocker",
        "--model-path",
        MODEL_NAME,
        "--endpoint",
        endpoint,
        "--discovery-backend",
        store_backend,
        "--num-workers",
        str(num_workers),
    ]

    if worker_type is not None:
        command.extend(["--disaggregation-mode", worker_type])

    command.extend(MockerConfig.from_value(mocker_args).to_cli_args())

    return command


class MockerProcess:
    """Manage mocker engine instances with a shared Tokio runtime."""

    def __init__(
        self,
        request,
        mocker_args: MockerConfig | Mapping[str, Any] | None = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        raw_kv_events: bool = False,
        standalone_indexer: bool = False,
        standalone_selector: bool = False,
        model_name: str = "mocker",
        zmq_replay: bool = False,
    ):
        if standalone_selector and not standalone_indexer:
            raise ValueError("standalone_selector requires standalone_indexer=True")

        namespace_suffix = generate_random_suffix()
        self.namespace = f"test-namespace-{namespace_suffix}"
        self.component_name = "mocker"
        self.model_name = model_name
        self.endpoint = f"dyn://{self.namespace}.{self.component_name}.generate"
        self.num_workers = num_mockers
        self._zmq_kv_events_ports: list[int] = []
        self._zmq_replay_ports: list[int] = []
        self._sidecar_ports: list[int] = []
        self._standalone_indexer = standalone_indexer
        self._standalone_selector = standalone_selector
        self._standalone_indexer_port: Optional[int] = None
        self._standalone_indexer_b_port: Optional[int] = None
        self._standalone_selector_port: Optional[int] = None
        self._indexer_process: Optional[ManagedProcess] = None
        self._indexer_b_process: Optional[ManagedProcess] = None
        self._selector_process: Optional[ManagedProcess] = None
        self._mocker_processes: list[ManagedProcess] = []
        self._exit_stack: ExitStack | None = None
        self._request = request
        self._store_backend = store_backend
        self._request_plane = request_plane
        self._mocker_config = MockerConfig.from_value(mocker_args)
        self.worker_id_to_zmq_ports: dict[int, dict[int, str]] = {}
        request.addfinalizer(self._release_ports)

        process_config = self._mocker_config
        self.dp_size = process_config.dp_size
        self.data_parallel_size = self.dp_size

        if raw_kv_events:
            dp_size = process_config.dp_size or 1
            self._zmq_kv_events_ports = allocate_contiguous_ports(
                num_mockers, dp_size, BASE_PORT_ZMQ
            )
            bases = [self._zmq_kv_events_ports[i * dp_size] for i in range(num_mockers)]
            if not standalone_indexer:
                process_config = replace(
                    process_config,
                    zmq_kv_events_ports=",".join(str(port) for port in bases),
                )
            logger.info(
                "Allocated ZMQ KV event ports %s (bases: %s) for %s workers",
                self._zmq_kv_events_ports,
                bases,
                num_mockers,
            )

        if zmq_replay and raw_kv_events:
            dp_size = process_config.dp_size or 1
            self._zmq_replay_ports = allocate_contiguous_ports(
                num_mockers, dp_size, BASE_PORT_ZMQ + 1000
            )
            replay_bases = [
                self._zmq_replay_ports[i * dp_size] for i in range(num_mockers)
            ]
            if not standalone_indexer:
                process_config = replace(
                    process_config,
                    zmq_replay_ports=",".join(str(port) for port in replay_bases),
                )
            logger.info(
                "Allocated ZMQ replay ports %s (bases: %s) for %s workers",
                self._zmq_replay_ports,
                replay_bases,
                num_mockers,
            )

        if standalone_indexer:
            self._sidecar_ports = allocate_ports(
                3 if standalone_selector else 2, BASE_PORT
            )
            self._standalone_indexer_port = self._sidecar_ports[0]
            self._standalone_indexer_b_port = self._sidecar_ports[1]
            if standalone_selector:
                self._standalone_selector_port = self._sidecar_ports[2]
            self._process = None
        else:
            command = _build_mocker_command(
                endpoint=self.endpoint,
                store_backend=store_backend,
                num_workers=num_mockers,
                mocker_args=process_config,
            )
            env = os.environ.copy()
            env["DYN_REQUEST_PLANE"] = request_plane
            self._process = ManagedProcess(
                command=command,
                env=env,
                timeout=60,
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=request.node.name,
                terminate_all_matching_process_names=False,
                display_name="dynamo-mocker",
            )

        logger.info(
            "Created mocker process with %s worker(s), endpoint: %s%s%s",
            num_mockers,
            self.endpoint,
            ", standalone_indexer=True" if standalone_indexer else "",
            ", standalone_selector=True" if standalone_selector else "",
        )

    @property
    def standalone_indexer_url(self) -> Optional[str]:
        if self._standalone_indexer_port is not None:
            return f"http://localhost:{self._standalone_indexer_port}"
        return None

    @property
    def standalone_indexer_b_url(self) -> Optional[str]:
        if self._standalone_indexer_b_port is not None:
            return f"http://localhost:{self._standalone_indexer_b_port}"
        return None

    @property
    def standalone_selector_url(self) -> Optional[str]:
        if self._standalone_selector_port is not None:
            return f"http://localhost:{self._standalone_selector_port}"
        return None

    def __enter__(self):
        stack = ExitStack()
        try:
            if self._standalone_indexer:
                block_size = self._mocker_config.block_size or BLOCK_SIZE
                indexer_cmd = [
                    *get_kv_indexer_command(),
                    "--block-size",
                    str(block_size),
                    "--port",
                    str(self._standalone_indexer_port),
                ]
                self._indexer_process = ManagedProcess(
                    command=indexer_cmd,
                    timeout=120,
                    display_output=True,
                    health_check_ports=[self._standalone_indexer_port],
                    health_check_urls=[],
                    log_dir=self._request.node.name,
                    terminate_all_matching_process_names=False,
                    display_name="dynamo-kv-indexer",
                    env=get_kv_indexer_test_env(),
                )
                logger.info(
                    "Starting standalone indexer on port %s",
                    self._standalone_indexer_port,
                )
                stack.enter_context(self._indexer_process)

                if self._standalone_selector:
                    selector_cmd = [
                        *get_select_service_command(),
                        "--port",
                        str(self._standalone_selector_port),
                    ]
                    self._selector_process = ManagedProcess(
                        command=selector_cmd,
                        timeout=120,
                        display_output=True,
                        health_check_ports=[self._standalone_selector_port],
                        health_check_urls=[],
                        log_dir=self._request.node.name,
                        terminate_all_matching_process_names=False,
                        display_name="dynamo-select-service",
                        env=os.environ.copy(),
                    )
                    logger.info(
                        "Starting standalone selection service on port %s",
                        self._standalone_selector_port,
                    )
                    stack.enter_context(self._selector_process)
            else:
                logger.info(
                    "Starting mocker process with %s worker(s)", self.num_workers
                )
                if self._process is None:
                    raise RuntimeError("Mocker process was not configured")
                stack.enter_context(self._process)
        except Exception:
            stack.close()
            self._release_ports()
            raise

        self._exit_stack = stack.pop_all()
        return self

    async def launch_workers_with_indexer(self, endpoint):
        """Launch workers one-by-one and register them with the standalone indexer."""
        client = await endpoint.client()
        known_ids: set[int] = set()
        if self._exit_stack is None:
            raise RuntimeError("MockerProcess must be entered before launching workers")

        dp_size = self._mocker_config.dp_size or 1

        for i in range(self.num_workers):
            base_port = self._zmq_kv_events_ports[i * dp_size]
            mocker_config = replace(
                self._mocker_config, zmq_kv_events_ports=str(base_port)
            )
            if self._zmq_replay_ports:
                replay_base = self._zmq_replay_ports[i * dp_size]
                mocker_config = replace(
                    mocker_config, zmq_replay_ports=str(replay_base)
                )

            command = _build_mocker_command(
                endpoint=self.endpoint,
                store_backend=self._store_backend,
                num_workers=1,
                mocker_args=mocker_config,
            )
            env = os.environ.copy()
            env["DYN_REQUEST_PLANE"] = self._request_plane
            proc = ManagedProcess(
                command=command,
                env=env,
                timeout=60,
                display_output=True,
                health_check_ports=[],
                health_check_urls=[],
                log_dir=self._request.node.name,
                terminate_all_matching_process_names=False,
                display_name=f"mocker-{i}",
            )
            self._exit_stack.enter_context(proc)
            self._mocker_processes.append(proc)

            new_worker_id = None
            for _ in range(120):
                ids = set(client.instance_ids())
                new = ids - known_ids
                if new:
                    new_worker_id = new.pop()
                    known_ids.add(new_worker_id)
                    break
                await asyncio.sleep(0.5)

            if new_worker_id is None:
                raise RuntimeError(
                    f"Timed out waiting for mocker {i} to register "
                    f"(known_ids={known_ids})"
                )

            zmq_addresses = {}
            register_url = f"{self.standalone_indexer_url}/register"
            replay_base = (
                self._zmq_replay_ports[i * dp_size] if self._zmq_replay_ports else None
            )
            async with aiohttp.ClientSession() as session:
                for dp_rank in range(dp_size):
                    port = base_port + dp_rank
                    zmq_endpoint = f"tcp://127.0.0.1:{port}"
                    zmq_addresses[dp_rank] = zmq_endpoint

                    payload = {
                        "instance_id": new_worker_id,
                        "endpoint": zmq_endpoint,
                        "dp_rank": dp_rank,
                        "model_name": self.model_name,
                        "block_size": self._mocker_config.block_size or BLOCK_SIZE,
                    }
                    if replay_base is not None:
                        payload[
                            "replay_endpoint"
                        ] = f"tcp://127.0.0.1:{replay_base + dp_rank}"
                    async with session.post(register_url, json=payload) as response:
                        if response.status != 201:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to register instance {new_worker_id} "
                                f"dp_rank {dp_rank}: {response.status} {body}"
                            )

                if self.standalone_selector_url:
                    select_payload = {
                        "worker_id": new_worker_id,
                        "model_name": self.model_name,
                        "endpoint": self.endpoint,
                        "kv_events_endpoints": zmq_addresses,
                        "block_size": self._mocker_config.block_size or BLOCK_SIZE,
                        "data_parallel_start_rank": 0,
                        "data_parallel_size": dp_size,
                        "max_num_batched_tokens": (
                            self._mocker_config.max_num_batched_tokens or 8192
                        ),
                    }
                    async with session.post(
                        f"{self.standalone_selector_url}/workers",
                        json=select_payload,
                    ) as response:
                        if response.status != 201:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to register selection service worker "
                                f"{new_worker_id}: {response.status} {body}"
                            )

            self.worker_id_to_zmq_ports[new_worker_id] = zmq_addresses
            logger.info(
                "Mocker %s: worker_id=%s, zmq_addresses=%s",
                i,
                new_worker_id,
                zmq_addresses,
            )

        await wait_for_indexer_workers_active(
            self.standalone_indexer_url, self.worker_id_to_zmq_ports
        )
        if self.standalone_selector_url:
            await wait_for_selection_service_ready(
                self.standalone_selector_url,
                set(self.worker_id_to_zmq_ports),
            )
        logger.info(
            "All %s mockers launched and registered with indexer",
            self.num_workers,
        )

    def launch_indexer(self):
        """Launch indexer B with indexer A as its recovery peer."""
        if not self._standalone_indexer or self._standalone_indexer_b_port is None:
            raise RuntimeError("launch_indexer requires standalone_indexer=True")
        if not self.worker_id_to_zmq_ports:
            raise RuntimeError("launch_indexer requires workers to be registered first")

        if self._exit_stack is None:
            raise RuntimeError("MockerProcess must be entered before launching indexer")

        block_size = self._mocker_config.block_size or BLOCK_SIZE
        worker_entries = []
        for worker_id, zmq_addresses in self.worker_id_to_zmq_ports.items():
            for dp_rank, zmq_endpoint in zmq_addresses.items():
                worker_entries.append(f"{worker_id}:{dp_rank}={zmq_endpoint}")
        workers_arg = ",".join(worker_entries)

        indexer_b_cmd = [
            *get_kv_indexer_command(),
            "--block-size",
            str(block_size),
            "--port",
            str(self._standalone_indexer_b_port),
            "--peers",
            f"http://localhost:{self._standalone_indexer_port}",
            "--workers",
            workers_arg,
            "--model-name",
            self.model_name,
        ]
        self._indexer_b_process = ManagedProcess(
            command=indexer_b_cmd,
            timeout=120,
            display_output=True,
            health_check_ports=[self._standalone_indexer_b_port],
            health_check_urls=[],
            log_dir=self._request.node.name,
            terminate_all_matching_process_names=False,
            display_name="dynamo-kv-indexer-b",
            env=get_kv_indexer_test_env(),
        )
        logger.info(
            "Starting standalone indexer B on port %s with peer http://localhost:%s",
            self._standalone_indexer_b_port,
            self._standalone_indexer_port,
        )
        self._exit_stack.enter_context(self._indexer_b_process)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping mocker process(es)")
        try:
            if self._exit_stack is not None:
                return self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
            return None
        finally:
            self._exit_stack = None
            self._mocker_processes.clear()
            self._indexer_b_process = None
            self._selector_process = None
            self._indexer_process = None
            self._release_ports()

    def _release_ports(self) -> None:
        for label, ports in (
            ("ZMQ KV event", self._zmq_kv_events_ports),
            ("ZMQ replay", self._zmq_replay_ports),
            ("sidecar", self._sidecar_ports),
        ):
            if not ports:
                continue
            deallocate_ports(ports)
            logger.info("Deallocated %s ports %s", label, ports)
            ports.clear()


class DisaggMockerProcess:
    """Manage prefill or decode mocker instances for disaggregated serving."""

    def __init__(
        self,
        request,
        namespace: str,
        worker_type: Literal["prefill", "decode"],
        mocker_args: MockerConfig | Mapping[str, Any] | None = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        enable_bootstrap: bool = False,
        event_plane: Optional[str] = None,
        raw_kv_events: bool = False,
        env_overrides: Optional[Dict[str, str]] = None,
    ):
        if worker_type not in ("prefill", "decode"):
            raise ValueError(
                f"worker_type must be 'prefill' or 'decode', got {worker_type}"
            )

        self.namespace = namespace
        self.worker_type = worker_type
        self.num_workers = num_mockers
        self._bootstrap_ports: list[int] = []
        self._zmq_kv_events_ports: list[int] = []
        request.addfinalizer(self._release_ports)

        if worker_type == "prefill":
            self.component_name = "prefill"
            self.endpoint = f"dyn://{self.namespace}.prefill.generate"
        else:
            self.component_name = "backend"
            self.endpoint = f"dyn://{self.namespace}.backend.generate"

        mocker_config = MockerConfig.from_value(mocker_args)
        if enable_bootstrap and worker_type == "prefill":
            self._bootstrap_ports = allocate_ports(num_mockers, BASE_PORT_BOOTSTRAP)
            mocker_config = replace(
                mocker_config,
                bootstrap_ports=",".join(str(port) for port in self._bootstrap_ports),
            )
            logger.info(
                "Allocated bootstrap ports %s for %s prefill workers",
                self._bootstrap_ports,
                num_mockers,
            )

        if raw_kv_events:
            dp_size = mocker_config.dp_size or 1
            self._zmq_kv_events_ports = allocate_contiguous_ports(
                num_mockers, dp_size, BASE_PORT_ZMQ
            )
            bases = [self._zmq_kv_events_ports[i * dp_size] for i in range(num_mockers)]
            mocker_config = replace(
                mocker_config,
                zmq_kv_events_ports=",".join(str(port) for port in bases),
            )
            logger.info(
                "Allocated ZMQ KV event ports %s (bases: %s) for %s %s workers",
                self._zmq_kv_events_ports,
                bases,
                num_mockers,
                worker_type,
            )

        command = _build_mocker_command(
            endpoint=self.endpoint,
            store_backend=store_backend,
            num_workers=num_mockers,
            mocker_args=mocker_config,
            worker_type=worker_type,
        )
        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request_plane
        if event_plane is not None:
            env["DYN_EVENT_PLANE"] = event_plane
        if event_plane == "zmq" and request_plane != "nats":
            env.pop("NATS_SERVER", None)
        env.update(env_overrides or {})

        self._process = ManagedProcess(
            command=command,
            env=env,
            timeout=60,
            display_output=True,
            health_check_ports=[],
            health_check_urls=[],
            log_dir=request.node.name,
            terminate_all_matching_process_names=False,
            display_name=f"dynamo-mocker-{worker_type}",
        )
        logger.info(
            "Created %s mocker process with %s worker(s), endpoint: %s",
            worker_type,
            num_mockers,
            self.endpoint,
        )

    @property
    def bootstrap_ports(self) -> list[int]:
        return self._bootstrap_ports

    def __enter__(self):
        logger.info(
            "Starting %s mocker process with %s worker(s)",
            self.worker_type,
            self.num_workers,
        )
        try:
            self._process.__enter__()
        except Exception:
            self._release_ports()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping %s mocker process", self.worker_type)
        try:
            return self._process.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._release_ports()

    def _release_ports(self) -> None:
        for label, ports in (
            ("bootstrap", self._bootstrap_ports),
            ("ZMQ KV event", self._zmq_kv_events_ports),
        ):
            if not ports:
                continue
            deallocate_ports(ports)
            logger.info("Deallocated %s ports %s", label, ports)
            ports.clear()


def wait_for_disagg_workers(
    workers: DisaggMockerProcess,
    store_backend: str,
    request_plane: str,
    event_plane: Optional[str],
) -> list[int]:
    async def wait_for_workers() -> list[int]:
        runtime = get_runtime(
            store_backend=store_backend,
            request_plane=request_plane,
            event_plane=event_plane,
        )
        endpoint = runtime.endpoint(
            f"{workers.namespace}.{workers.component_name}.generate"
        )
        return await poll_for_worker_instances(endpoint, workers.num_workers)

    return asyncio.run(wait_for_workers())


@contextmanager
def launch_disagg_workers(
    request,
    namespace: str,
    registration_order: str,
    *,
    prefill_mocker_args: MockerConfig | Mapping[str, Any],
    decode_mocker_args: MockerConfig | Mapping[str, Any],
    num_prefill_mockers: int,
    num_decode_mockers: int,
    enable_disagg_bootstrap: bool,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    event_plane: Optional[str] = None,
    raw_kv_events: bool = False,
) -> Iterator[tuple[DisaggMockerProcess, DisaggMockerProcess]]:
    if registration_order not in ("prefill_first", "decode_first"):
        raise ValueError(f"Unexpected registration order: {registration_order}")

    if registration_order == "prefill_first":
        logger.info("Starting %s prefill mocker instances (first)", num_prefill_mockers)
        with DisaggMockerProcess(
            request,
            namespace=namespace,
            worker_type="prefill",
            mocker_args=prefill_mocker_args,
            num_mockers=num_prefill_mockers,
            store_backend=store_backend,
            request_plane=request_plane,
            enable_bootstrap=enable_disagg_bootstrap,
            event_plane=event_plane,
            raw_kv_events=raw_kv_events,
        ) as prefill_workers:
            logger.info("Prefill workers using endpoint: %s", prefill_workers.endpoint)
            wait_for_disagg_workers(
                prefill_workers, store_backend, request_plane, event_plane
            )
            logger.info(
                "Starting %s decode mocker instances (second)", num_decode_mockers
            )
            with DisaggMockerProcess(
                request,
                namespace=namespace,
                worker_type="decode",
                mocker_args=decode_mocker_args,
                num_mockers=num_decode_mockers,
                store_backend=store_backend,
                request_plane=request_plane,
                event_plane=event_plane,
                raw_kv_events=raw_kv_events,
            ) as decode_workers:
                logger.info(
                    "Decode workers using endpoint: %s", decode_workers.endpoint
                )
                wait_for_disagg_workers(
                    decode_workers, store_backend, request_plane, event_plane
                )
                yield prefill_workers, decode_workers
        return

    logger.info("Starting %s decode mocker instances (first)", num_decode_mockers)
    with DisaggMockerProcess(
        request,
        namespace=namespace,
        worker_type="decode",
        mocker_args=decode_mocker_args,
        num_mockers=num_decode_mockers,
        store_backend=store_backend,
        request_plane=request_plane,
        event_plane=event_plane,
        raw_kv_events=raw_kv_events,
    ) as decode_workers:
        logger.info("Decode workers using endpoint: %s", decode_workers.endpoint)
        wait_for_disagg_workers(
            decode_workers, store_backend, request_plane, event_plane
        )
        logger.info(
            "Starting %s prefill mocker instances (second)", num_prefill_mockers
        )
        with DisaggMockerProcess(
            request,
            namespace=namespace,
            worker_type="prefill",
            mocker_args=prefill_mocker_args,
            num_mockers=num_prefill_mockers,
            store_backend=store_backend,
            request_plane=request_plane,
            enable_bootstrap=enable_disagg_bootstrap,
            event_plane=event_plane,
            raw_kv_events=raw_kv_events,
        ) as prefill_workers:
            logger.info("Prefill workers using endpoint: %s", prefill_workers.endpoint)
            wait_for_disagg_workers(
                prefill_workers, store_backend, request_plane, event_plane
            )
            yield prefill_workers, decode_workers
