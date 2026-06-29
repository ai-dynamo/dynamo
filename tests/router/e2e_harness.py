# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, ContextManager

from tests.router.common import (
    _test_router_basic,
    _test_router_decisions,
    _test_router_decisions_disagg,
    _test_router_indexers_sync,
)
from tests.router.helper import generate_random_suffix, get_runtime
from tests.utils.constants import DefaultPort
from tests.utils.port_utils import allocate_ports, deallocate_ports
from tests.utils.test_output import resolve_test_output_path

logger = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None else int(raw)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.lower() not in {"0", "false", "no", "off"}


def resolve_router_gpu_start_index(gpu_start_index: int) -> int:
    override = os.environ.get("DYNAMO_ROUTER_E2E_GPU_START_INDEX")
    if override is None:
        return gpu_start_index
    try:
        base_index = int(override)
    except ValueError as exc:
        raise ValueError(
            "DYNAMO_ROUTER_E2E_GPU_START_INDEX must be an integer"
        ) from exc
    return base_index + gpu_start_index


def _resolve_router_e2e_num_requests(default: int = 10) -> int:
    override = os.environ.get("DYNAMO_ROUTER_E2E_NUM_REQUESTS")
    if override is None:
        return default
    try:
        value = int(override)
    except ValueError as exc:
        raise ValueError("DYNAMO_ROUTER_E2E_NUM_REQUESTS must be an integer") from exc
    if value <= 0:
        raise ValueError("DYNAMO_ROUTER_E2E_NUM_REQUESTS must be positive")
    return value


@contextmanager
def maybe_router_gms_servers():
    if os.environ.get("DYNAMO_ROUTER_E2E_ENABLE_GMS") != "1":
        yield
        return

    from tests.gpu_memory_service.common.gms import GMSServer

    devices = os.environ.get("DYNAMO_ROUTER_E2E_GMS_DEVICES", "0,1")
    tags = os.environ.get("DYNAMO_ROUTER_E2E_GMS_TAGS", "weights,kv_cache")
    device_ids = [int(part.strip()) for part in devices.split(",") if part.strip()]
    tag_names = [part.strip() for part in tags.split(",") if part.strip()]

    with ExitStack() as stack:
        for device in device_ids:
            for tag in tag_names:
                stack.enter_context(GMSServer(device=device, tag=tag))
        yield


TEST_PROMPT = (
    "In a quiet meadow tucked between rolling hills, a plump gray rabbit nibbled on "
    "clover beneath the shade of a gnarled oak tree. Its ears twitched at the faint "
    "rustle of leaves, but it remained calm, confident in the safety of its burrow "
    "just a few hops away. The late afternoon sun warmed its fur, and tiny dust "
    "motes danced in the golden light as bees hummed lazily nearby. Though the "
    "rabbit lived a simple life, every day was an adventure of scents, shadows, and "
    "snacks-an endless search for the tastiest patch of greens and the softest spot "
    "to nap."
)


def allocate_frontend_ports(request, count: int) -> list[int]:
    ports = allocate_ports(count, DefaultPort.FRONTEND.value)
    request.addfinalizer(lambda: deallocate_ports(ports))
    return ports


def build_test_payload(model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": True,
        "max_tokens": 10,
    }


class ManagedEngineProcessMixin:
    process_name = "worker"
    cleanup_name = "worker resources"
    init_delay_seconds = 5
    init_delay_reason = "initialize before starting next worker"
    cleanup_delay_seconds = 2

    def __enter__(self):
        logger.info(
            "[%s] Starting %d worker processes sequentially...",
            self.__class__.__name__,
            len(self.worker_processes),
        )

        for i, process in enumerate(self.worker_processes):
            logger.info(
                "[%s] Starting %s %d...", self.__class__.__name__, self.process_name, i
            )
            try:
                process._logger = logging.getLogger(process.__class__.__name__)
                process._command_name = process.command[0]
                process.log_dir = resolve_test_output_path(process.log_dir)
                os.makedirs(process.log_dir, exist_ok=True)
                log_name = f"{process._command_name}.log.txt"
                process._log_path = os.path.join(process.log_dir, log_name)

                if process.data_dir:
                    process._remove_directory(process.data_dir)

                process._terminate_all_matching_process_names()
                logger.info(
                    "[%s] Launching process %d (pid will be assigned)...",
                    self.__class__.__name__,
                    i,
                )
                process._start_process()
                logger.info(
                    "[%s] Worker %d launched with PID: %s",
                    self.__class__.__name__,
                    i,
                    process.proc.pid if process.proc else "unknown",
                )
                time.sleep(process.delayed_start)

                if i < len(self.worker_processes) - 1:
                    logger.info(
                        "[%s] Waiting %ss for worker %d to %s...",
                        self.__class__.__name__,
                        self.init_delay_seconds,
                        i,
                        self.init_delay_reason,
                    )
                    time.sleep(self.init_delay_seconds)

            except Exception:
                logger.exception(
                    "[%s] Failed to start worker %d", self.__class__.__name__, i
                )
                try:
                    process.__exit__(None, None, None)
                except Exception as cleanup_err:
                    logger.warning(
                        "[%s] Error during cleanup: %s",
                        self.__class__.__name__,
                        cleanup_err,
                    )
                raise

        logger.info(
            "[%s] All %d workers launched with sequential initialization.",
            self.__class__.__name__,
            len(self.worker_processes),
        )
        logger.info(
            "[%s] Waiting for health checks to complete...", self.__class__.__name__
        )

        for i, process in enumerate(self.worker_processes):
            logger.info(
                "[%s] Checking health for worker %d...", self.__class__.__name__, i
            )
            try:
                elapsed = process._check_ports(process.timeout)
                process._check_urls(process.timeout - elapsed)
                process._check_funcs(process.timeout - elapsed)
                logger.info(
                    "[%s] Worker %d health checks passed", self.__class__.__name__, i
                )
            except Exception:
                logger.error(
                    "[%s] Worker %d health check failed", self.__class__.__name__, i
                )
                self.__exit__(None, None, None)
                raise

        logger.info(
            "[%s] All workers started successfully and passed health checks!",
            self.__class__.__name__,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, process in enumerate(self.worker_processes):
            logger.info("Stopping %s %d", self.process_name, i)
            process.__exit__(exc_type, exc_val, exc_tb)

        logger.info("Waiting for %s to fully clean up...", self.cleanup_name)
        time.sleep(self.cleanup_delay_seconds)


def get_engine_endpoint(engine_workers, request_plane: str, component_name: str):
    runtime = get_runtime(request_plane=request_plane)
    return runtime.endpoint(f"{engine_workers.namespace}.{component_name}.generate")


def _create_engine_process(
    *,
    engine_process_cls,
    engine_args_name: str,
    engine_args: dict[str, Any],
    request,
    request_plane: str,
    default_process_kwargs: dict[str, Any],
    engine_process_kwargs: dict[str, Any] | None,
):
    process_kwargs = (
        default_process_kwargs
        if engine_process_kwargs is None
        else engine_process_kwargs
    )
    return engine_process_cls(
        request,
        request_plane=request_plane,
        **{engine_args_name: engine_args},
        **process_kwargs,
    )


def run_basic_router_test(
    *,
    engine_process_cls,
    engine_args_name: str,
    engine_args: dict[str, Any],
    num_workers: int,
    single_gpu: bool,
    request,
    request_plane: str,
    block_size: int,
    model_name: str,
    frontend_timeout: int = 180,
    engine_process_kwargs: dict[str, Any] | None = None,
    test_payload: dict[str, Any] | None = None,
    num_requests: int = 10,
    router_mode: str = "kv",
    min_initial_workers: int | None = None,
):
    process = _create_engine_process(
        engine_process_cls=engine_process_cls,
        engine_args_name=engine_args_name,
        engine_args=engine_args,
        request=request,
        request_plane=request_plane,
        default_process_kwargs={
            "num_workers": num_workers,
            "single_gpu": single_gpu,
        },
        engine_process_kwargs=engine_process_kwargs,
    )
    with ExitStack() as stack:
        stack.enter_context(maybe_router_gms_servers())
        engine_workers = stack.enter_context(process)
        frontend_port = allocate_frontend_ports(request, 1)[0]
        _test_router_basic(
            engine_workers=engine_workers,
            block_size=block_size,
            request=request,
            frontend_port=frontend_port,
            test_payload=test_payload or build_test_payload(model_name),
            num_requests=_resolve_router_e2e_num_requests(num_requests),
            frontend_timeout=frontend_timeout,
            store_backend="etcd",
            request_plane=request_plane,
            router_mode=router_mode,
            min_initial_workers=min_initial_workers,
        )


def run_router_decisions_test(
    *,
    engine_process_cls,
    engine_args_name: str,
    engine_args: dict[str, Any],
    request,
    request_plane: str,
    model_name: str,
    block_size: int,
    component_name: str,
    num_workers: int,
    single_gpu: bool,
    test_dp_rank: bool,
    extra_process_kwargs: dict[str, Any] | None = None,
    initial_wait: float = 0.25,
    engine_process_kwargs: dict[str, Any] | None = None,
    test_kwargs: dict[str, Any] | None = None,
):
    default_process_kwargs = {
        "num_workers": num_workers,
        "single_gpu": single_gpu,
        **(extra_process_kwargs or {}),
    }
    process = _create_engine_process(
        engine_process_cls=engine_process_cls,
        engine_args_name=engine_args_name,
        engine_args=engine_args,
        request=request,
        request_plane=request_plane,
        default_process_kwargs=default_process_kwargs,
        engine_process_kwargs=engine_process_kwargs,
    )
    with ExitStack() as stack:
        stack.enter_context(maybe_router_gms_servers())
        engine_workers = stack.enter_context(process)
        endpoint = get_engine_endpoint(engine_workers, request_plane, component_name)
        scenario_kwargs = dict(test_kwargs or {})
        for argument, attribute in (
            ("standalone_indexer_url", "standalone_indexer_url"),
            ("standalone_selector_url", "standalone_selector_url"),
        ):
            value = getattr(engine_workers, attribute, None)
            if value is not None:
                scenario_kwargs.setdefault(argument, value)
        _test_router_decisions(
            engine_workers,
            endpoint,
            model_name,
            request,
            test_dp_rank=test_dp_rank,
            block_size=block_size,
            initial_wait=initial_wait,
            **scenario_kwargs,
        )


def run_disagg_router_decisions_test(
    *,
    engine_process_cls,
    engine_args_name: str,
    engine_args: dict[str, Any],
    request,
    request_plane: str,
    model_name: str,
    block_size: int,
    num_prefill_workers: int,
    num_decode_workers: int,
    prefill_process_kwargs: dict[str, Any] | None = None,
    decode_process_kwargs: dict[str, Any] | None = None,
    worker_context_factory: Callable[[str], ContextManager[tuple[Any, Any]]]
    | None = None,
    test_payload: dict[str, Any] | None = None,
    test_kwargs: dict[str, Any] | None = None,
    strict_timing: bool = True,
    progressive_request_count: int = 4,
):
    shared_namespace = f"test-namespace-{generate_random_suffix()}"
    frontend_port = allocate_frontend_ports(request, 1)[0]

    prefill_kwargs = {
        "namespace": shared_namespace,
        **(prefill_process_kwargs or {}),
    }
    decode_kwargs = {
        "namespace": shared_namespace,
        **(decode_process_kwargs or {}),
    }

    def run_test(prefill_workers, decode_workers):
        scenario_kwargs = {
            "strict_timing": strict_timing,
            "progressive_request_count": progressive_request_count,
            **(test_kwargs or {}),
        }
        _test_router_decisions_disagg(
            prefill_workers=prefill_workers,
            decode_workers=decode_workers,
            block_size=block_size,
            request=request,
            frontend_port=frontend_port,
            test_payload=test_payload or build_test_payload(model_name),
            request_plane=request_plane,
            **scenario_kwargs,
        )

    with ExitStack() as stack:
        stack.enter_context(maybe_router_gms_servers())
        if worker_context_factory is not None:
            workers = stack.enter_context(worker_context_factory(shared_namespace))
            run_test(*workers)
            return

        prefill_workers = stack.enter_context(
            engine_process_cls(
                request,
                num_workers=num_prefill_workers,
                request_plane=request_plane,
                **{engine_args_name: engine_args},
                **prefill_kwargs,
            )
        )
        decode_workers = stack.enter_context(
            engine_process_cls(
                request,
                num_workers=num_decode_workers,
                request_plane=request_plane,
                **{engine_args_name: engine_args},
                **decode_kwargs,
            )
        )
        run_test(prefill_workers, decode_workers)


def run_indexers_sync_test(
    *,
    engine_process_cls,
    engine_args_name: str,
    engine_args: dict[str, Any],
    request,
    runtime_services_dynamic_ports,
    store_backend: str,
    durable_kv_events: bool,
    request_plane: str,
    block_size: int,
    model_name: str,
    num_workers: int,
    extra_process_kwargs: dict[str, Any] | None = None,
    engine_process_kwargs: dict[str, Any] | None = None,
):
    nats_process, _etcd_process = runtime_services_dynamic_ports
    process_kwargs = extra_process_kwargs or {}

    process = _create_engine_process(
        engine_process_cls=engine_process_cls,
        engine_args_name=engine_args_name,
        engine_args=engine_args,
        request=request,
        request_plane=request_plane,
        default_process_kwargs={
            "num_workers": num_workers,
            "single_gpu": True,
            "store_backend": store_backend,
            "durable_kv_events": durable_kv_events,
            **process_kwargs,
        },
        engine_process_kwargs=engine_process_kwargs,
    )
    with ExitStack() as stack:
        stack.enter_context(maybe_router_gms_servers())
        engine_workers = stack.enter_context(process)
        _test_router_indexers_sync(
            engine_workers=engine_workers,
            block_size=block_size,
            model_name=model_name,
            num_workers=num_workers,
            store_backend=store_backend,
            request_plane=request_plane,
            test_nats_interruption=not durable_kv_events,
            nats_server=nats_process if not durable_kv_events else None,
            durable_kv_events=durable_kv_events,
            standalone_indexer_url=getattr(
                engine_workers, "standalone_indexer_url", None
            ),
            standalone_indexer_b_url=getattr(
                engine_workers, "standalone_indexer_b_url", None
            ),
            test_zmq_replay=bool(
                getattr(engine_workers, "standalone_indexer_url", None)
            ),
        )
