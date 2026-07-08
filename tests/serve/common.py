# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common base classes and utilities for engine tests (vLLM, TRT-LLM, etc.)"""

import collections
import concurrent.futures
import dataclasses
import logging
import os
import signal
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Optional

import pytest

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.conftest import ServicePorts
from tests.utils.client import send_request
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import (
    EngineConfig,
    EngineLogError,
    EngineProcess,
    ResponseValidationError,
)
from tests.utils.payload_builder import (
    make_chat_health_check,
    make_completions_health_check,
    make_images_health_check,
)
from tests.utils.payloads import ChatPayload, CompletionPayload, ImagesPayload
from tests.utils.port_utils import allocate_port, deallocate_port

DEFAULT_TIMEOUT = 10

SERVE_TEST_DIR = os.path.join(WORKSPACE_DIR, "tests/serve")
logger = logging.getLogger(__name__)


def _tail_logs(content: str, *, lines: int = 80) -> str:
    if not content:
        return "<no server logs captured>"
    return "".join(content.splitlines(keepends=True)[-lines:]).rstrip()


# Payload-class → factory for an endpoint-readiness check. isinstance matches
# subclasses (e.g. ChatPayloadWithLogprobs) and is robust to ad-hoc `endpoint`
# string overrides on derived payloads.
_ENDPOINT_HEALTH_CHECK_FACTORIES = (
    (CompletionPayload, make_completions_health_check),
    (ChatPayload, make_chat_health_check),
    (ImagesPayload, make_images_health_check),
)


def _is_multimodal_chat(payload: ChatPayload) -> bool:
    # make_chat_health_check sends a text-only probe; multimodal endpoints
    # (e.g. Qwen3-VL disagg/EPD) won't accept it, so the probe spins until
    # pytest-timeout. Detect via OpenAI-style list content with any non-text
    # part (image_url, video_url, input_audio, ...).
    messages = (getattr(payload, "body", None) or {}).get("messages") or []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    return True
    return False


def _payload_eligible_for_check(payload: Any, payload_cls: type) -> bool:
    if not isinstance(payload, payload_cls):
        return False
    if payload_cls is ChatPayload and _is_multimodal_chat(payload):
        return False
    return True


def _with_endpoint_readiness_checks(
    config: EngineConfig, frontend_port: int
) -> EngineConfig:
    new_checks = [
        factory(frontend_port, config.model)
        for payload_cls, factory in _ENDPOINT_HEALTH_CHECK_FACTORIES
        if any(
            _payload_eligible_for_check(p, payload_cls) for p in config.request_payloads
        )
    ]
    if not new_checks:
        return config
    return dataclasses.replace(
        config,
        health_check_funcs=[*config.health_check_funcs, *new_checks],
    )


# Stable lifecycle tokens emitted by the Rust backend-common Worker
# (dynamo_backend_common::worker). We key off these rather than full
# human-readable sentences so a log reword doesn't break the test.
_RUST_WORKER_CLEANUP_TOKEN = "Engine cleanup complete"
_RUST_WORKER_DRAIN_TOKEN = "drain: waiting for prefill to quiesce"


def assert_rust_worker_drained(
    server_process: EngineProcess,
    logger: logging.Logger,
    *,
    deadline_s: float = 90.0,
) -> bool:
    """Assert the Rust Worker reached ``Engine cleanup complete`` within
    ``deadline_s`` of the prefill SIGTERM (proving drain -> cleanup ran), and
    return whether the drain loop observably engaged.

    Keyed off worker-log tokens, not metrics: the drain runs at shutdown and the
    process exits right after, so ``/metrics`` is already gone. The log-format
    coupling is isolated here per ``tests/CLAUDE.md``. The returned
    ``drain: waiting for prefill to quiesce`` flag is a diagnostic only (always
    present for engines that return None); the loop logic is unit-tested in Rust.
    """
    end = time.time() + deadline_s
    while True:
        log = server_process.read_logs() or ""
        if _RUST_WORKER_CLEANUP_TOKEN in log:
            drain_engaged = _RUST_WORKER_DRAIN_TOKEN in log
            logger.info(
                "Rust Worker shutdown observed (drain loop engaged=%s)", drain_engaged
            )
            return drain_engaged
        if time.time() >= end:
            raise EngineLogError(
                f"Rust Worker '{_RUST_WORKER_CLEANUP_TOKEN}' not observed within "
                f"{deadline_s:.0f}s of prefill SIGTERM.\n\nLog tail:\n"
                f"{_tail_logs(log, lines=60)}"
            )
        time.sleep(0.5)


def _format_request_failure(
    *,
    config: EngineConfig,
    payload: Any,
    server_process: EngineProcess,
    error: Exception,
) -> str:
    server_state = "running" if server_process.is_running() else "not running"
    try:
        url = payload.url()
    except Exception:
        url = "<payload.url() raised>"
    return (
        f"{type(payload).__name__} request failed for config '{config.name}' "
        f"(method={payload.method}, url={url}, timeout={payload.timeout}s, "
        f"server_pid={server_process.get_pid()}, server_state={server_state}, "
        f"log_path={server_process.log_path})\n"
        f"Original error: {type(error).__name__}: {error}\n\n"
        "Last 80 server log lines:\n"
        f"{_tail_logs(server_process.read_logs(), lines=80)}"
    )


@dataclasses.dataclass
class _PreparedDeployment:
    config: EngineConfig
    merged_env: dict
    frontend_port: int
    system_ports: list
    disagg_bootstrap_port: Optional[int]
    extra_allocated_ports: list[int]


def _prepare_deployment(
    config: EngineConfig,
    request: Any,
    *,
    ports: ServicePorts | None,
    extra_env: Optional[Dict[str, str]],
) -> _PreparedDeployment:
    """Build the launch env (profile/KV overrides, dynamic ports, bootstrap
    port) and the port-adjusted config shared by all deployment runners."""
    merged_env: dict[str, str] = {}
    if extra_env:
        merged_env.update(extra_env)

    # In serial mode (no parallel scheduler), pass the marker's KV cache budget
    # so the launch script's small default doesn't starve larger models.
    # The parallel scheduler already sets this env var per-test.
    if "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES" not in os.environ:
        kv_mark = request.node.get_closest_marker("requested_vllm_kv_cache_bytes")
        if kv_mark:
            merged_env.setdefault(
                "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", str(int(kv_mark.args[0]))
            )

    if "_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS" not in os.environ:
        sglang_kv_mark = request.node.get_closest_marker("requested_sglang_kv_tokens")
        if sglang_kv_mark:
            merged_env.setdefault(
                "_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS",
                str(int(sglang_kv_mark.args[0])),
            )

    if "_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS" not in os.environ:
        trtllm_kv_mark = request.node.get_closest_marker("requested_trtllm_kv_tokens")
        if trtllm_kv_mark:
            merged_env.setdefault(
                "_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS",
                str(int(trtllm_kv_mark.args[0])),
            )

    if "_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES" not in os.environ:
        trtllm_vram_mark = request.node.get_closest_marker("requested_trtllm_vram_gib")
        if trtllm_vram_mark:
            gib_to_bytes = int(trtllm_vram_mark.args[0] * 1024**3)
            merged_env.setdefault(
                "_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES",
                str(gib_to_bytes),
            )

    # Stagger engine startup under xdist to avoid vLLM profiling race
    # (vLLM bug #10643: concurrent profilers miscount each other's memory).
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id.startswith("gw"):
        worker_num = int(worker_id.removeprefix("gw"))
        if worker_num > 0:
            stagger_s = worker_num * 15
            logger.info("Staggering startup by %ds (xdist %s)", stagger_s, worker_id)
            time.sleep(stagger_s)

    # Track additional ports allocated for multi-GPU tests (for cleanup in finally)
    extra_allocated_ports: list[int] = []

    if ports is not None:
        dynamic_frontend_port = int(ports.frontend_port)
        dynamic_system_ports = [int(p) for p in ports.system_ports]

        # The environments are used by the bash scripts to set the ports.
        merged_env["DYN_HTTP_PORT"] = str(dynamic_frontend_port)

        # If no system ports are provided, explicitly ensure we don't pass any
        # stale DYN_SYSTEM_PORT* values via extra_env.
        if not dynamic_system_ports:
            for k in list(merged_env.keys()):
                if k == "DYN_SYSTEM_PORT":
                    merged_env.pop(k, None)
                    continue
                if k.startswith("DYN_SYSTEM_PORT") and k != "DYN_SYSTEM_PORT":
                    suffix = k.removeprefix("DYN_SYSTEM_PORT")
                    if suffix.isdigit():
                        merged_env.pop(k, None)
                        continue
                if k.startswith("DYN_SYSTEM_PORT_WORKER"):
                    suffix = k.removeprefix("DYN_SYSTEM_PORT_WORKER")
                    if suffix.isdigit():
                        merged_env.pop(k, None)
        else:
            # Alias for PORT1 (many scripts only read this).
            merged_env["DYN_SYSTEM_PORT"] = str(dynamic_system_ports[0])
            merged_env["DYN_SYSTEM_PORT1"] = str(dynamic_system_ports[0])
            for idx, port in enumerate(dynamic_system_ports, start=1):
                merged_env[f"DYN_SYSTEM_PORT{idx}"] = str(port)
                merged_env[f"DYN_SYSTEM_PORT_WORKER{idx}"] = str(port)

        # Unique ZMQ port for vLLM KV event publishing (avoids xdist collisions).
        if ports.kv_event_port:
            merged_env["DYN_VLLM_KV_EVENT_PORT"] = str(ports.kv_event_port)
            # For multi-worker scripts (xpu_2 router tests), allocate separate
            # KV event ports for each worker to avoid ZMQ bind collisions.
            if len(dynamic_system_ports) >= 2:
                kv_port1 = ports.kv_event_port
                kv_port2 = allocate_port(ports.kv_event_port + 1)
                extra_allocated_ports.append(kv_port2)
                merged_env["DYN_VLLM_KV_EVENT_PORT1"] = str(kv_port1)
                merged_env["DYN_VLLM_KV_EVENT_PORT2"] = str(kv_port2)

        if ports.fpm_port:
            merged_env["DYN_FORWARDPASS_METRIC_PORT"] = str(ports.fpm_port)

        # Per-worker NIXL side-channel ports (avoids xdist collisions on 20097).
        for idx, port in enumerate(ports.nixl_side_channel_ports, start=1):
            merged_env[f"DYN_VLLM_NIXL_SIDE_CHANNEL_PORT{idx}"] = str(port)

        # Ensure EngineProcess health checks hit the correct frontend port.
        config = dataclasses.replace(config, frontend_port=dynamic_frontend_port)

    else:
        # Backward compat: infer from config/extra_env if no explicit ports are passed.
        dynamic_frontend_port = int(config.frontend_port)
        # Preserve the historical two-port behavior in this branch. Tests that
        # need tighter control should pass `ports=...` to avoid default port
        # collisions under xdist.
        dynamic_system_ports = [
            int(
                merged_env.get("DYN_SYSTEM_PORT1")
                or merged_env.get("DYN_SYSTEM_PORT")
                or DefaultPort.SYSTEM1.value
            ),
            int(merged_env.get("DYN_SYSTEM_PORT2") or DefaultPort.SYSTEM2.value),
        ]

    config = _with_endpoint_readiness_checks(config, dynamic_frontend_port)

    # Disagg scripts need a unique bootstrap port so parallel runs don't collide.
    disagg_bootstrap_port: int | None = None
    if config.script_name and "disagg" in config.script_name:
        disagg_bootstrap_port = allocate_port(12000)
        merged_env["DYN_DISAGG_BOOTSTRAP_PORT"] = str(disagg_bootstrap_port)

    return _PreparedDeployment(
        config=config,
        merged_env=merged_env,
        frontend_port=dynamic_frontend_port,
        system_ports=dynamic_system_ports,
        disagg_bootstrap_port=disagg_bootstrap_port,
        extra_allocated_ports=extra_allocated_ports,
    )


def _sigterm_prefill_worker(
    server_process: EngineProcess, logger: logging.Logger
) -> bool:
    """SIGTERM the prefill worker's launcher process, found under the script's
    process tree by its cmdline. Returns False if not found."""
    import psutil

    try:
        root = psutil.Process(server_process.get_pid())
    except psutil.NoSuchProcess:
        return False
    for proc in [root, *root.children(recursive=True)]:
        try:
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        joined = " ".join(cmdline)
        # Require the `unified_main` token, not just the disagg arg, so we don't
        # signal an engine-spawned child/MPI rank that inherited the same args.
        if "unified_main" in joined and "--disaggregation-mode prefill" in joined:
            logger.info("Sending SIGTERM to prefill worker pid=%d", proc.pid)
            proc.send_signal(signal.SIGTERM)
            return True
    return False


def run_prefill_drain_deployment(
    config: EngineConfig,
    request: Any,
    *,
    ports: ServicePorts | None,
    burst_size: int = 96,
    burst_max_tokens: int = 256,
    burst_prompt: str = "The quick brown fox jumps over the lazy dog. " * 160,
) -> None:
    """Prefill-drain e2e: launch a disaggregated deployment, fire a concurrent
    burst, SIGTERM only the prefill worker while requests are in flight, and
    assert the Rust Worker drove graceful shutdown via
    :func:`assert_rust_worker_drained`.

    Backend-agnostic — the same burst exercises all three engines. Signaling the
    worker directly (not via harness teardown) avoids the harness's ~8s SIGKILL
    so the drain can run to completion.
    """
    logger = logging.getLogger(request.node.name)
    logger.info("Starting %s prefill-drain test", config.name)

    prep = _prepare_deployment(config, request, ports=ports, extra_env=None)
    chat_url = f"http://localhost:{prep.frontend_port}/v1/chat/completions"
    chat_body = {
        "model": config.model,
        "messages": [{"role": "user", "content": burst_prompt}],
        "max_tokens": burst_max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        with EngineProcess.from_script(
            prep.config, request, extra_env=prep.merged_env
        ) as server:
            # Warm-up: confirm the pipeline serves before stressing it. Retry
            # because disagg readiness can briefly 5xx just after registration.
            warm = None
            for attempt in range(5):
                warm = send_request(
                    url=chat_url, payload=chat_body, timeout=120, method="POST"
                )
                if warm.status_code == 200:
                    break
                logger.warning(
                    "warm-up attempt %d got %d; retrying", attempt + 1, warm.status_code
                )
                time.sleep(2.0)
            assert (
                warm is not None and warm.status_code == 200
            ), f"warm-up request failed after retries: {warm.status_code} {warm.text[:300]}"

            # Fire a concurrent burst and do NOT wait for it.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=burst_size)
            futures = [
                pool.submit(
                    send_request,
                    url=chat_url,
                    payload=chat_body,
                    timeout=180,
                    method="POST",
                )
                for _ in range(burst_size)
            ]
            logger.info("Fired burst of %d concurrent requests", burst_size)

            # Gate on the burst being observably in flight (not a fixed sleep)
            # so the SIGTERM lands while transfers are pending, regardless of
            # GPU speed.
            want_in_flight = max(burst_size // 2, 8)
            gate_deadline = time.time() + 15.0
            in_flight = 0
            while time.time() < gate_deadline:
                in_flight = sum(1 for f in futures if not f.done())
                if in_flight >= want_in_flight:
                    break
                time.sleep(0.1)
            logger.info("burst in-flight=%d; signaling prefill worker", in_flight)

            assert _sigterm_prefill_worker(
                server, logger
            ), "could not locate the prefill worker process to signal"

            drain_engaged = assert_rust_worker_drained(server, logger)

            # Functional floor: some burst requests served. Requests routed
            # after the prefill unregister legitimately fail, so this is a
            # floor, not "all succeed". Tally outcomes by status code (or
            # exception type) so a 0-OK failure shows how the burst died.
            outcomes: collections.Counter = collections.Counter()
            try:
                for fut in concurrent.futures.as_completed(futures, timeout=200):
                    try:
                        outcomes[fut.result().status_code] += 1
                    except Exception as e:
                        outcomes[type(e).__name__] += 1
            except concurrent.futures.TimeoutError:
                logger.warning("burst tally timed out waiting for stragglers")
            ok = outcomes[200]
            logger.info(
                "burst: %d/%d returned 200 across prefill drain "
                "(drain loop engaged=%s, outcomes=%s)",
                ok,
                burst_size,
                drain_engaged,
                dict(outcomes),
            )
            assert ok >= 1, (
                "no burst request completed across the prefill drain "
                f"(outcomes: {dict(outcomes)})"
            )
            pool.shutdown(wait=False)
    finally:
        if prep.disagg_bootstrap_port is not None:
            deallocate_port(prep.disagg_bootstrap_port)
        for p in prep.extra_allocated_ports:
            deallocate_port(p)


def run_serve_deployment(
    config: EngineConfig,
    request: Any,
    *,
    ports: ServicePorts | None = None,  # pass `dynamo_dynamic_ports` here
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    """Run a standard serve deployment test for any EngineConfig.

    - Launches the engine via EngineProcess.from_script
    - Builds a payload (with optional override/mutator)
    - Iterates configured endpoints and validates responses and logs
    """

    logger = logging.getLogger(request.node.name)
    logger.info("Starting %s test_deployment", config.name)

    assert (
        config.request_payloads is not None and len(config.request_payloads) > 0
    ), "request_payloads must be provided on EngineConfig"

    logger.info("Using model: %s", config.model)
    logger.info("Script: %s", config.script_name)

    prep = _prepare_deployment(config, request, ports=ports, extra_env=extra_env)
    config = prep.config
    merged_env = prep.merged_env
    dynamic_frontend_port = prep.frontend_port
    dynamic_system_ports = prep.system_ports
    disagg_bootstrap_port = prep.disagg_bootstrap_port
    extra_allocated_ports = prep.extra_allocated_ports

    try:
        with EngineProcess.from_script(
            config, request, extra_env=merged_env
        ) as server_process:
            for _payload in config.request_payloads:
                logger.info("TESTING: Payload: %s", _payload.__class__.__name__)

                # Make a per-iteration copy so tests can safely override ports/fields
                # without mutating shared config instances across parametrized cases.
                payload = deepcopy(_payload)
                # inject model
                if hasattr(payload, "with_model"):
                    payload = payload.with_model(config.model)

                # Default behavior: requests go to the frontend port, except metrics which target
                # worker system ports (mapped from DefaultPort -> per-test ports).
                if getattr(payload, "endpoint", "") == "/metrics":
                    if payload.port == DefaultPort.SYSTEM1.value:
                        if len(dynamic_system_ports) < 1:
                            raise RuntimeError(
                                "Payload targets SYSTEM_PORT1 but no system ports were provided "
                                f"(payload={payload.__class__.__name__})"
                            )
                        payload.port = dynamic_system_ports[0]
                    elif payload.port == DefaultPort.SYSTEM2.value:
                        if len(dynamic_system_ports) < 2:
                            raise RuntimeError(
                                "Payload targets SYSTEM_PORT2 but only 1 system port was provided "
                                f"(payload={payload.__class__.__name__})"
                            )
                        payload.port = dynamic_system_ports[1]
                else:
                    payload.port = dynamic_frontend_port

                # Optional extra system ports for specialized payloads (e.g. LoRA control-plane APIs).
                # BasePayload always defines `system_ports` (usually empty); map defaults
                # (SYSTEM_PORT1/2) to per-test system ports when present.
                if payload.system_ports:
                    mapped_system_ports: list[int] = []
                    for p in payload.system_ports:
                        if p == DefaultPort.SYSTEM1.value:
                            if len(dynamic_system_ports) < 1:
                                raise RuntimeError(
                                    "Payload.system_ports includes SYSTEM_PORT1 but no system ports were provided "
                                    f"(payload={payload.__class__.__name__})"
                                )
                            mapped_system_ports.append(dynamic_system_ports[0])
                        elif p == DefaultPort.SYSTEM2.value:
                            if len(dynamic_system_ports) < 2:
                                raise RuntimeError(
                                    "Payload.system_ports includes SYSTEM_PORT2 but only 1 system port was provided "
                                    f"(payload={payload.__class__.__name__})"
                                )
                            mapped_system_ports.append(dynamic_system_ports[1])
                        else:
                            mapped_system_ports.append(p)
                    payload.system_ports = mapped_system_ports

                for _ in range(payload.repeat_count):
                    # Re-issue the request (server stays up) on validation
                    # failure when payload.max_attempts > 1. See tests/README.md
                    # "Flaky Tests" for when this is appropriate. Backoff
                    # factor 1.5 keeps the worst-case sleep budget bounded
                    # for max_attempts up to ~6.
                    last_err: Optional[ResponseValidationError] = None
                    try:
                        for attempt in range(payload.max_attempts):
                            try:
                                response = send_request(
                                    url=payload.url(),
                                    payload=payload.body,
                                    timeout=payload.timeout,
                                    method=payload.method,
                                    stream=payload.http_stream,
                                )
                                server_process.check_response(payload, response)
                                last_err = None
                                break
                            except ResponseValidationError as e:
                                last_err = e
                                if attempt < payload.max_attempts - 1:
                                    wait = 1.0 * (1.5**attempt)
                                    logger.warning(
                                        "%s request failed (attempt %d/%d): %s — retrying in %.1fs",
                                        type(payload).__name__,
                                        attempt + 1,
                                        payload.max_attempts,
                                        e,
                                        wait,
                                    )
                                    time.sleep(wait)
                    except Exception as e:
                        # Transport / connection failures (and payload.url()
                        # failures) aren't retried by design; the inner loop
                        # only retries ResponseValidationError. Re-raise with
                        # the server's last 80 log lines so a CI failure is
                        # diagnosable in one pass rather than yielding a bare
                        # ReadTimeout.
                        raise RuntimeError(
                            _format_request_failure(
                                config=config,
                                payload=payload,
                                server_process=server_process,
                                error=e,
                            )
                        ) from e
                    if last_err is not None:
                        raise last_err

                # Call final_validation if the payload has one (e.g., CachedTokensChatPayload)
                if hasattr(payload, "final_validation"):
                    payload.final_validation()
    finally:
        if disagg_bootstrap_port is not None:
            deallocate_port(disagg_bootstrap_port)
        for p in extra_allocated_ports:
            deallocate_port(p)


def params_with_model_mark(configs: Mapping[str, EngineConfig]):
    """Return pytest params for a config dict, adding a model marker per param.

    This enables simple model collection after pytest filtering.
    """
    params = []
    for config_name, cfg in configs.items():
        marks = list(getattr(cfg, "marks", []))
        marks.append(pytest.mark.model(cfg.model))
        params.append(pytest.param(config_name, marks=marks))
    return params
