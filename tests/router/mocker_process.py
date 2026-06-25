# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

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
    mocker_args: Dict[str, Any],
    worker_type: Optional[str] = None,
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

    if worker_type == "prefill":
        command.extend(["--disaggregation-mode", "prefill"])
    elif worker_type == "decode":
        command.extend(["--disaggregation-mode", "decode"])

    if "speedup_ratio" in mocker_args:
        command.extend(["--speedup-ratio", str(mocker_args["speedup_ratio"])])
    if "block_size" in mocker_args:
        command.extend(["--block-size", str(mocker_args["block_size"])])
    if "num_gpu_blocks" in mocker_args:
        command.extend(
            ["--num-gpu-blocks-override", str(mocker_args["num_gpu_blocks"])]
        )
    if "max_num_seqs" in mocker_args:
        command.extend(["--max-num-seqs", str(mocker_args["max_num_seqs"])])
    if "max_num_batched_tokens" in mocker_args:
        command.extend(
            ["--max-num-batched-tokens", str(mocker_args["max_num_batched_tokens"])]
        )
    if "enable_prefix_caching" in mocker_args:
        if mocker_args["enable_prefix_caching"]:
            command.append("--enable-prefix-caching")
        else:
            command.append("--no-enable-prefix-caching")
    if "enable_chunked_prefill" in mocker_args:
        if mocker_args["enable_chunked_prefill"]:
            command.append("--enable-chunked-prefill")
        else:
            command.append("--no-enable-chunked-prefill")
    if "preemption_mode" in mocker_args:
        command.extend(["--preemption-mode", str(mocker_args["preemption_mode"])])
    if "dp_size" in mocker_args:
        command.extend(["--data-parallel-size", str(mocker_args["dp_size"])])
    if "planner_profile_data" in mocker_args:
        command.extend(
            ["--planner-profile-data", str(mocker_args["planner_profile_data"])]
        )
    if mocker_args.get("aic_perf_model") is True:
        command.append("--aic-perf-model")
    if "aic_system" in mocker_args:
        command.extend(["--aic-system", str(mocker_args["aic_system"])])
    if "aic_backend_version" in mocker_args:
        command.extend(
            ["--aic-backend-version", str(mocker_args["aic_backend_version"])]
        )
    if "aic_tp_size" in mocker_args:
        command.extend(["--aic-tp-size", str(mocker_args["aic_tp_size"])])
    if mocker_args.get("durable_kv_events") is True:
        command.append("--durable-kv-events")
    if "bootstrap_ports" in mocker_args:
        command.extend(["--bootstrap-ports", mocker_args["bootstrap_ports"]])
    if "zmq_kv_events_ports" in mocker_args:
        command.extend(["--zmq-kv-events-ports", mocker_args["zmq_kv_events_ports"]])
    if "zmq_replay_ports" in mocker_args:
        command.extend(["--zmq-replay-ports", mocker_args["zmq_replay_ports"]])

    return command


class MockerProcess:
    """Manage mocker engine instances with a shared Tokio runtime."""

    def __init__(
        self,
        request,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        zmq_kv_events: bool = False,
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
        self._standalone_indexer = standalone_indexer
        self._standalone_selector = standalone_selector
        self._standalone_indexer_port: Optional[int] = None
        self._standalone_indexer_b_port: Optional[int] = None
        self._standalone_selector_port: Optional[int] = None
        self._indexer_process: Optional[ManagedProcess] = None
        self._indexer_b_process: Optional[ManagedProcess] = None
        self._selector_process: Optional[ManagedProcess] = None
        self._mocker_processes: list[ManagedProcess] = []
        self._request = request
        self._store_backend = store_backend
        self._request_plane = request_plane
        self._mocker_args_orig: Dict[str, Any] = (mocker_args or {}).copy()
        self.worker_id_to_zmq_ports: dict[int, dict[int, str]] = {}

        mocker_args = self._mocker_args_orig.copy()
        self.dp_size = mocker_args.get("dp_size")
        self.data_parallel_size = self.dp_size

        if zmq_kv_events:
            dp_size = mocker_args.get("dp_size", 1)
            self._zmq_kv_events_ports = allocate_contiguous_ports(
                num_mockers, dp_size, BASE_PORT_ZMQ
            )
            bases = [self._zmq_kv_events_ports[i * dp_size] for i in range(num_mockers)]
            if not standalone_indexer:
                mocker_args["zmq_kv_events_ports"] = ",".join(str(p) for p in bases)
            logger.info(
                "Allocated ZMQ KV event ports %s (bases: %s) for %s workers",
                self._zmq_kv_events_ports,
                bases,
                num_mockers,
            )

        if zmq_replay and zmq_kv_events:
            dp_size = mocker_args.get("dp_size", 1)
            self._zmq_replay_ports = allocate_contiguous_ports(
                num_mockers, dp_size, BASE_PORT_ZMQ + 1000
            )
            replay_bases = [
                self._zmq_replay_ports[i * dp_size] for i in range(num_mockers)
            ]
            if not standalone_indexer:
                mocker_args["zmq_replay_ports"] = ",".join(str(p) for p in replay_bases)
            logger.info(
                "Allocated ZMQ replay ports %s (bases: %s) for %s workers",
                self._zmq_replay_ports,
                replay_bases,
                num_mockers,
            )

        if standalone_indexer:
            sidecar_ports = allocate_ports(3 if standalone_selector else 2, BASE_PORT)
            self._standalone_indexer_port = sidecar_ports[0]
            self._standalone_indexer_b_port = sidecar_ports[1]
            if standalone_selector:
                self._standalone_selector_port = sidecar_ports[2]
            request.addfinalizer(lambda: deallocate_ports(sidecar_ports))
            self._process = None
        else:
            command = _build_mocker_command(
                endpoint=self.endpoint,
                store_backend=store_backend,
                num_workers=num_mockers,
                mocker_args=mocker_args,
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
        if self._standalone_indexer:
            block_size = self._mocker_args_orig.get("block_size", BLOCK_SIZE)
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
            self._indexer_process.__enter__()

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
                self._selector_process.__enter__()
        else:
            logger.info("Starting mocker process with %s worker(s)", self.num_workers)
            self._process.__enter__()
        return self

    async def launch_workers_with_indexer(self, endpoint):
        """Launch workers one-by-one and register them with the standalone indexer."""
        client = await endpoint.client()
        known_ids: set[int] = set()
        dp_size = self._mocker_args_orig.get("dp_size", 1)

        for i in range(self.num_workers):
            mocker_args = self._mocker_args_orig.copy()
            base_port = self._zmq_kv_events_ports[i * dp_size]
            mocker_args["zmq_kv_events_ports"] = str(base_port)
            if self._zmq_replay_ports:
                replay_base = self._zmq_replay_ports[i * dp_size]
                mocker_args["zmq_replay_ports"] = str(replay_base)

            command = _build_mocker_command(
                endpoint=self.endpoint,
                store_backend=self._store_backend,
                num_workers=1,
                mocker_args=mocker_args,
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
            proc.__enter__()
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
                        "block_size": self._mocker_args_orig.get(
                            "block_size", BLOCK_SIZE
                        ),
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
                        "block_size": self._mocker_args_orig.get(
                            "block_size", BLOCK_SIZE
                        ),
                        "data_parallel_start_rank": 0,
                        "data_parallel_size": dp_size,
                        "max_num_batched_tokens": self._mocker_args_orig.get(
                            "max_num_batched_tokens",
                            8192,
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

        block_size = self._mocker_args_orig.get("block_size", BLOCK_SIZE)
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
        self._indexer_b_process.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping mocker process(es)")
        for process in self._mocker_processes:
            try:
                process.__exit__(exc_type, exc_val, exc_tb)
            except Exception as error:
                logger.warning("Error stopping mocker process: %s", error)
        self._mocker_processes.clear()

        if self._indexer_b_process is not None:
            try:
                self._indexer_b_process.__exit__(exc_type, exc_val, exc_tb)
            except Exception as error:
                logger.warning("Error stopping indexer B process: %s", error)
            self._indexer_b_process = None

        if self._selector_process is not None:
            try:
                self._selector_process.__exit__(exc_type, exc_val, exc_tb)
            except Exception as error:
                logger.warning("Error stopping selection service process: %s", error)
            self._selector_process = None

        if self._indexer_process is not None:
            try:
                self._indexer_process.__exit__(exc_type, exc_val, exc_tb)
            except Exception as error:
                logger.warning("Error stopping indexer process: %s", error)
            self._indexer_process = None

        if self._process is not None:
            self._process.__exit__(exc_type, exc_val, exc_tb)
        if self._zmq_kv_events_ports:
            deallocate_ports(self._zmq_kv_events_ports)
            logger.info("Deallocated ZMQ KV event ports %s", self._zmq_kv_events_ports)
            self._zmq_kv_events_ports = []
        if self._zmq_replay_ports:
            deallocate_ports(self._zmq_replay_ports)
            logger.info("Deallocated ZMQ replay ports %s", self._zmq_replay_ports)
            self._zmq_replay_ports = []


class DisaggMockerProcess:
    """Manage prefill or decode mocker instances for disaggregated serving."""

    def __init__(
        self,
        request,
        namespace: str,
        worker_type: str,
        mocker_args: Optional[Dict[str, Any]] = None,
        num_mockers: int = 1,
        store_backend: str = "etcd",
        request_plane: str = "nats",
        enable_bootstrap: bool = False,
        event_plane: Optional[str] = None,
        zmq_kv_events: bool = False,
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

        if worker_type == "prefill":
            self.component_name = "prefill"
            self.endpoint = f"dyn://{self.namespace}.prefill.generate"
        else:
            self.component_name = "backend"
            self.endpoint = f"dyn://{self.namespace}.backend.generate"

        mocker_args = (mocker_args or {}).copy()
        if enable_bootstrap and worker_type == "prefill":
            self._bootstrap_ports = allocate_ports(num_mockers, BASE_PORT_BOOTSTRAP)
            mocker_args["bootstrap_ports"] = ",".join(
                str(port) for port in self._bootstrap_ports
            )
            logger.info(
                "Allocated bootstrap ports %s for %s prefill workers",
                self._bootstrap_ports,
                num_mockers,
            )

        if zmq_kv_events:
            dp_size = mocker_args.get("dp_size", 1)
            self._zmq_kv_events_ports = allocate_contiguous_ports(
                num_mockers, dp_size, BASE_PORT_ZMQ
            )
            bases = [self._zmq_kv_events_ports[i * dp_size] for i in range(num_mockers)]
            mocker_args["zmq_kv_events_ports"] = ",".join(str(port) for port in bases)
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
            mocker_args=mocker_args,
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
        self._process.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping %s mocker process", self.worker_type)
        self._process.__exit__(exc_type, exc_val, exc_tb)
        if self._bootstrap_ports:
            deallocate_ports(self._bootstrap_ports)
            logger.info("Deallocated bootstrap ports %s", self._bootstrap_ports)
            self._bootstrap_ports = []
        if self._zmq_kv_events_ports:
            deallocate_ports(self._zmq_kv_events_ports)
            logger.info("Deallocated ZMQ KV event ports %s", self._zmq_kv_events_ports)
            self._zmq_kv_events_ports = []


def _wait_for_disagg_workers(
    workers: DisaggMockerProcess,
    store_backend: str,
    request_plane: str,
    event_plane: Optional[str],
) -> None:
    async def wait_for_workers() -> None:
        runtime = get_runtime(
            store_backend=store_backend,
            request_plane=request_plane,
            event_plane=event_plane,
        )
        endpoint = runtime.endpoint(
            f"{workers.namespace}.{workers.component_name}.generate"
        )
        await poll_for_worker_instances(endpoint, workers.num_workers)

    asyncio.run(wait_for_workers())


@contextmanager
def launch_disagg_workers(
    request,
    namespace: str,
    registration_order: str,
    *,
    prefill_mocker_args: Dict[str, Any],
    decode_mocker_args: Dict[str, Any],
    num_prefill_mockers: int,
    num_decode_mockers: int,
    enable_disagg_bootstrap: bool,
    store_backend: str = "etcd",
    request_plane: str = "nats",
    event_plane: Optional[str] = None,
    zmq_kv_events: bool = False,
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
            zmq_kv_events=zmq_kv_events,
        ) as prefill_workers:
            logger.info("Prefill workers using endpoint: %s", prefill_workers.endpoint)
            _wait_for_disagg_workers(
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
                zmq_kv_events=zmq_kv_events,
            ) as decode_workers:
                logger.info(
                    "Decode workers using endpoint: %s", decode_workers.endpoint
                )
                _wait_for_disagg_workers(
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
        zmq_kv_events=zmq_kv_events,
    ) as decode_workers:
        logger.info("Decode workers using endpoint: %s", decode_workers.endpoint)
        _wait_for_disagg_workers(
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
            zmq_kv_events=zmq_kv_events,
        ) as prefill_workers:
            logger.info("Prefill workers using endpoint: %s", prefill_workers.endpoint)
            _wait_for_disagg_workers(
                prefill_workers, store_backend, request_plane, event_plane
            )
            yield prefill_workers, decode_workers
