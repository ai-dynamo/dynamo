# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import shlex
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

pytestmark = [pytest.mark.nightly, pytest.mark.fault_tolerance]

logger = logging.getLogger(__name__)

_DEFAULT_KV_BLOCK_SIZE = 32
_MIN_FREE_GPU_MEMORY_FRACTION = 0.8
_DEFAULT_MAX_FAILOVER_READY_SECONDS = 5.0
_DEFAULT_POST_RESTORE_RESPONSE_TIMEOUT_SECONDS = 30.0
_DEFAULT_RETRY_INTERVAL_SECONDS = 0.05
_DEFAULT_RESPONSE_MAX_TOKENS = 128
_DEFAULT_COHERENCE_MAX_TOKENS = 8
_DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
_DEFAULT_COHERENCE_PROMPT = (
    "What is seven plus five? /no_think\n"
    "Reply with exactly the number 12 and no other words."
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_json_object(name: str, default: dict[str, Any]) -> dict[str, Any]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return dict(default)

    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def _redact(text: str) -> str:
    text = re.sub(r"(HF_TOKEN=)[^\s]+", r"\1<redacted>", text)
    text = re.sub(r'("HF_TOKEN"\s*:\s*")[^"]+', r"\1<redacted>", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", text)
    return text


def _tail_log(process: ManagedProcess, lines: int = 160) -> str:
    logs = process.read_logs().splitlines()
    return _redact("\n".join(logs[-lines:]))


def _fail_with_logs(message: str, processes: dict[str, ManagedProcess]) -> None:
    sections = [message]
    for name, process in processes.items():
        sections.append(f"\n--- tail {name} ---\n{_tail_log(process)}")
    pytest.fail("\n".join(sections))


def _require_running(name: str, process: ManagedProcess) -> None:
    if not process.is_running():
        raise RuntimeError(
            f"{name} exited early with code "
            f"{process.proc.returncode if process.proc else '<not-started>'}"
        )


def _wait_until(
    label: str,
    predicate,
    *,
    timeout_s: float,
    interval_s: float = 1.0,
    processes: dict[str, ManagedProcess] | None = None,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return_value = predicate()
            if return_value:
                logger.info("wait succeeded: %s", label)
                return return_value
        except Exception as exc:
            last_error = exc
        time.sleep(interval_s)

    message = f"Timed out waiting for {label}"
    if last_error is not None:
        message += f": {last_error}"
    if processes:
        _fail_with_logs(message, processes)
    raise TimeoutError(message)


def _wait_for_log(
    name: str,
    process: ManagedProcess,
    needle: str,
    *,
    timeout_s: float,
    processes: dict[str, ManagedProcess],
    interval_s: float = 1.0,
) -> None:
    def contains_needle() -> bool:
        _require_running(name, process)
        return needle in process.read_logs()

    _wait_until(
        f"{name} log contains {needle!r}",
        contains_needle,
        timeout_s=timeout_s,
        interval_s=interval_s,
        processes=processes,
    )


def _wait_for_any_log(
    name: str,
    process: ManagedProcess,
    needles: tuple[str, ...],
    *,
    timeout_s: float,
    processes: dict[str, ManagedProcess],
    interval_s: float = 1.0,
) -> str:
    def contains_any_needle() -> str | None:
        _require_running(name, process)
        logs = process.read_logs()
        return next((needle for needle in needles if needle in logs), None)

    return _wait_until(
        f"{name} log contains any of {needles!r}",
        contains_any_needle,
        timeout_s=timeout_s,
        interval_s=interval_s,
        processes=processes,
    )


def _wait_for_any_process_log(
    process_by_name: dict[str, ManagedProcess],
    needle: str,
    *,
    timeout_s: float,
    processes: dict[str, ManagedProcess],
    interval_s: float = 1.0,
) -> tuple[str, ManagedProcess]:
    def contains_needle() -> tuple[str, ManagedProcess] | None:
        for name, process in process_by_name.items():
            _require_running(name, process)
            if needle in process.read_logs():
                return name, process
        return None

    return _wait_until(
        f"one process log contains {needle!r}",
        contains_needle,
        timeout_s=timeout_s,
        interval_s=interval_s,
        processes=processes,
    )


def _http_json(url: str, *, timeout_s: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_http_json(
    url: str,
    label: str,
    *,
    timeout_s: float,
    processes: dict[str, ManagedProcess],
    process_to_check: tuple[str, ManagedProcess] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] | None = None

    def request_succeeds() -> bool:
        nonlocal result
        if process_to_check is not None:
            _require_running(process_to_check[0], process_to_check[1])
        result = _http_json(url)
        return True

    _wait_until(label, request_succeeds, timeout_s=timeout_s, processes=processes)
    assert result is not None
    return result


def _wait_for_model(
    frontend_port: int,
    model: str,
    *,
    timeout_s: float,
    processes: dict[str, ManagedProcess],
) -> None:
    def model_is_registered() -> bool:
        response = _http_json(f"http://127.0.0.1:{frontend_port}/v1/models")
        models = response.get("data", [])
        return any(entry.get("id") == model for entry in models)

    _wait_until(
        f"frontend model registration for {model}",
        model_is_registered,
        timeout_s=timeout_s,
        processes=processes,
    )


def _post_chat(
    frontend_port: int,
    model: str,
    prompt: str,
    *,
    timeout_s: float,
    retry_status_codes: tuple[int, ...] = (404, 503),
    retry_interval_s: float | None = None,
    max_tokens: int | None = None,
):
    if max_tokens is None:
        max_tokens = _env_int(
            "DYN_TRTLLM_SHADOW_FAILOVER_RESPONSE_MAX_TOKENS",
            _DEFAULT_RESPONSE_MAX_TOKENS,
        )
    chat_template_kwargs = _env_json_object(
        "DYN_TRTLLM_SHADOW_FAILOVER_CHAT_TEMPLATE_KWARGS",
        _DEFAULT_CHAT_TEMPLATE_KWARGS,
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    request = urllib.request.Request(
        f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    request_timeout_s = max(1.0, min(120.0, timeout_s))
    if retry_interval_s is None:
        retry_interval_s = _env_float(
            "DYN_TRTLLM_SHADOW_FAILOVER_RETRY_INTERVAL_SECONDS",
            _DEFAULT_RETRY_INTERVAL_SECONDS,
        )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=request_timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in retry_status_codes:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(min(retry_interval_s, max(0.0, deadline - time.monotonic())))

    raise TimeoutError(f"chat request did not succeed: {last_error}")


def _extract_message_text(response: dict[str, Any]) -> tuple[str, str, str | None]:
    choices = response.get("choices") or []
    assert choices, f"response did not include choices: {response}"
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    finish_reason = choices[0].get("finish_reason")
    return content, reasoning, finish_reason


def _visible_answer_text(content: str) -> str:
    match = re.search(r"</think\s*>", content, flags=re.IGNORECASE)
    if match:
        return content[match.end() :].strip()
    if re.search(r"<think\b", content, flags=re.IGNORECASE):
        return ""
    return content.strip()


def _assert_coherent_post_restore_response(response: dict[str, Any]) -> None:
    content, reasoning, finish_reason = _extract_message_text(response)
    visible_answer = _visible_answer_text(content)
    normalized = " ".join(visible_answer.replace("\n", " ").split()).lower()
    assert content.strip(), f"post-restore response content is empty: {response}"
    assert finish_reason in {"stop", "length", None}, (
        f"unexpected post-restore finish_reason={finish_reason!r}: {response}"
    )
    assert visible_answer, (
        "post-restore response did not include a visible answer outside of an "
        f"unfinished think block. content={content!r} reasoning_len={len(reasoning)}"
    )
    assert "12" in normalized or "twelve" in normalized, (
        "post-restore response did not contain the expected arithmetic answer "
        f"12/twelve. visible_answer={visible_answer!r} "
        f"content={content!r} reasoning_len={len(reasoning)}"
    )


def _cuda_visible_devices(tp_size: int) -> str:
    override = os.environ.get("DYN_TRTLLM_SHADOW_FAILOVER_CUDA_VISIBLE_DEVICES")
    if override:
        return override
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current:
        return current
    return ",".join(str(index) for index in range(tp_size))


def _visible_device_count(visible_devices: str) -> int:
    return len([device for device in visible_devices.split(",") if device.strip()])


def _assert_free_gpu_memory_fraction_is_not_cheating(
    free_gpu_memory_fraction: float, engine_yaml: Path
) -> None:
    if free_gpu_memory_fraction < _MIN_FREE_GPU_MEMORY_FRACTION:
        raise ValueError(
            "DYN_TRTLLM_SHADOW_FAILOVER_FREE_GPU_MEMORY_FRACTION must be >= "
            f"{_MIN_FREE_GPU_MEMORY_FRACTION}, got {free_gpu_memory_fraction}"
        )

    config = yaml.safe_load(engine_yaml.read_text(encoding="utf-8")) or {}
    yaml_fraction = (config.get("kv_cache_config") or {}).get(
        "free_gpu_memory_fraction"
    )
    if (
        yaml_fraction is not None
        and float(yaml_fraction) < _MIN_FREE_GPU_MEMORY_FRACTION
    ):
        raise ValueError(
            f"{engine_yaml} sets kv_cache_config.free_gpu_memory_fraction="
            f"{yaml_fraction}, below {_MIN_FREE_GPU_MEMORY_FRACTION}"
        )


def _write_default_engine_yaml(
    tmp_path: Path, *, tp_size: int, free_gpu_memory_fraction: float
) -> Path:
    engine_yaml = tmp_path / "trtllm_shadow_failover_engine.yaml"
    config = {
        "backend": "pytorch",
        "trust_remote_code": True,
        "tensor_parallel_size": tp_size,
        "max_num_tokens": _env_int(
            "DYN_TRTLLM_SHADOW_FAILOVER_MAX_NUM_TOKENS", 2048
        ),
        "max_batch_size": _env_int("DYN_TRTLLM_SHADOW_FAILOVER_MAX_BATCH_SIZE", 1),
        "enable_autotuner": False,
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        "kv_cache_config": {
            "enable_block_reuse": True,
            "free_gpu_memory_fraction": free_gpu_memory_fraction,
        },
        "print_iter_log": True,
        "stream_interval": 10,
    }
    engine_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return engine_yaml


def _engine_yaml(
    tmp_path: Path, *, tp_size: int, free_gpu_memory_fraction: float
) -> Path:
    override = os.environ.get("DYN_TRTLLM_SHADOW_FAILOVER_ENGINE_YAML")
    if override:
        engine_yaml = Path(override).expanduser()
        if not engine_yaml.exists():
            raise FileNotFoundError(
                f"DYN_TRTLLM_SHADOW_FAILOVER_ENGINE_YAML does not exist: {engine_yaml}"
            )
    else:
        engine_yaml = _write_default_engine_yaml(
            tmp_path,
            tp_size=tp_size,
            free_gpu_memory_fraction=free_gpu_memory_fraction,
        )

    _assert_free_gpu_memory_fraction_is_not_cheating(
        free_gpu_memory_fraction, engine_yaml
    )
    return engine_yaml


def _base_env(
    *,
    namespace: str,
    etcd_endpoint: str,
    gms_socket_dir: Path,
    failover_lock_path: Path,
    cuda_visible_devices: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_EVENT_PLANE": "zmq",
            "DYN_LOG": os.environ.get("DYN_LOG", "debug"),
            "DYN_NAMESPACE": namespace,
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_ETCD_LEASE_TTL_SECS": os.environ.get(
                "DYN_TRTLLM_SHADOW_FAILOVER_ETCD_LEASE_TTL_SECS",
                "600",
            ),
            "ETCD_ENDPOINTS": etcd_endpoint,
            "FAILOVER_LOCK_PATH": str(failover_lock_path),
            "GMS_SOCKET_DIR": str(gms_socket_dir),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.setdefault("MPI4PY_MPIABI", "openmpi")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("OMPI_MCA_btl", "self,tcp")
    env.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    env.setdefault("OMPI_MCA_coll_ucc_enable", "0")
    env.setdefault("OMPI_MCA_pml", "ob1")
    env.setdefault("UCX_TLS", "tcp,self")
    env.setdefault("HF_MODULES_CACHE", str(gms_socket_dir / "hf-modules-cache"))
    Path(env["HF_MODULES_CACHE"]).mkdir(parents=True, exist_ok=True)
    return env


def _enable_shadow_prewarm(
    env: dict[str, str], *, gms_socket_dir: Path, shadow_count: int
) -> None:
    env.setdefault("DYN_TRTLLM_SHADOW_PREWARM_EXPECTED_STANDBYS", str(shadow_count))
    env.setdefault(
        "DYN_TRTLLM_SHADOW_PREWARM_DIR",
        str(gms_socket_dir / "shadow-prewarm"),
    )
    env.setdefault(
        "DYN_TRTLLM_SHADOW_PREWARM_LOCK_PATH",
        str(gms_socket_dir / "shadow-prewarm.lock"),
    )
    env.setdefault(
        "DYN_TRTLLM_SHADOW_SLEEP_TAGS",
        "shadow_failover",
    )


def _engine_command(
    *,
    namespace: str,
    model: str,
    served_model_name: str,
    engine_yaml: Path,
    free_gpu_memory_fraction: float,
    kv_block_size: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dynamo.trtllm",
        "--namespace",
        namespace,
        "--endpoint",
        f"dyn://{namespace}.tensorrt_llm.generate",
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--modality",
        "text",
        "--extra-engine-args",
        str(engine_yaml),
        "--load-format",
        "gms",
        "--gms-shadow-mode",
        "--free-gpu-memory-fraction",
        str(free_gpu_memory_fraction),
        "--publish-events-and-metrics",
        "--kv-block-size",
        str(kv_block_size),
        "--discovery-backend",
        "etcd",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]
    command.extend(
        shlex.split(os.environ.get("DYN_TRTLLM_SHADOW_FAILOVER_EXTRA_ARGS", ""))
    )
    return command


def _start_engine(
    stack: ExitStack,
    request,
    *,
    log_prefix: str,
    name: str,
    engine_id: str,
    system_port: int,
    command: list[str],
    env: dict[str, str],
) -> ManagedProcess:
    process = ManagedProcess(
        command=command,
        env={
            **env,
            "CONTAINER_NAME": name,
            "DYN_SYSTEM_PORT": str(system_port),
            "ENGINE_ID": engine_id,
        },
        timeout=1,
        display_output=True,
        log_dir=f"{log_prefix}_{name}",
        display_name=name,
        terminate_all_matching_process_names=False,
    )
    return stack.enter_context(process)


def _kill_process_group(process: ManagedProcess, *, wait: bool = True) -> float:
    pid = process.get_pid()
    if pid is None:
        return time.monotonic()
    try:
        parent_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return time.monotonic()
    child_pgids: list[int] = []
    for child in process.subprocesses():
        try:
            child_pgid = os.getpgid(child.pid)
        except ProcessLookupError:
            continue
        if child_pgid != parent_pgid and child_pgid not in child_pgids:
            child_pgids.append(child_pgid)

    signal_started_at = time.monotonic()
    logger.info(
        "Sending SIGKILL to process groups: child_pgids=%s parent_pgid=%s",
        child_pgids,
        parent_pgid,
    )
    for pgid in [*child_pgids, parent_pgid]:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if wait and process.proc is not None:
        process.proc.wait(timeout=60)
    elif process.proc is not None:
        process.proc.poll()
    logger.info(
        "SIGKILL process group %s in %.2fs",
        "wait completed" if wait else "signals sent",
        time.monotonic() - signal_started_at,
    )
    return signal_started_at


def _gms_ready_file_exists(gms_socket_dir: Path) -> bool:
    return (gms_socket_dir / "gms-ready").exists()


@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(QWEN)
@pytest.mark.timeout(3600)
@pytest.mark.trtllm
def test_gms_shadow_engine_failover_trtllm(
    request, tmp_path, predownload_models
):
    model = os.environ.get("DYN_TRTLLM_SHADOW_FAILOVER_MODEL", QWEN)
    served_model_name = os.environ.get(
        "DYN_TRTLLM_SHADOW_FAILOVER_SERVED_MODEL_NAME", model
    )
    tp_size = _env_int("DYN_TRTLLM_SHADOW_FAILOVER_TP", 1)
    free_gpu_memory_fraction = _env_float(
        "DYN_TRTLLM_SHADOW_FAILOVER_FREE_GPU_MEMORY_FRACTION", 0.85
    )
    cuda_visible_devices = _cuda_visible_devices(tp_size)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRTLLM shadow failover")
    if _visible_device_count(cuda_visible_devices) < tp_size:
        pytest.skip(
            f"TRTLLM shadow failover needs at least {tp_size} visible GPU(s), "
            f"got CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}"
        )
    shadow_count = _env_int("DYN_TRTLLM_SHADOW_FAILOVER_SHADOW_COUNT", 2)
    if shadow_count < 1:
        raise ValueError("DYN_TRTLLM_SHADOW_FAILOVER_SHADOW_COUNT must be >= 1")
    max_failover_ready_s = _env_float(
        "DYN_TRTLLM_SHADOW_FAILOVER_MAX_READY_SECONDS",
        _env_float(
            "DYN_TRTLLM_SHADOW_FAILOVER_MAX_POST_KILL_RESPONSE_SECONDS",
            _DEFAULT_MAX_FAILOVER_READY_SECONDS,
        ),
    )
    post_restore_response_timeout_s = _env_float(
        "DYN_TRTLLM_SHADOW_FAILOVER_POST_RESTORE_RESPONSE_TIMEOUT_SECONDS",
        _DEFAULT_POST_RESTORE_RESPONSE_TIMEOUT_SECONDS,
    )

    engine_yaml = _engine_yaml(
        tmp_path,
        tp_size=tp_size,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )
    namespace = (
        f"{request.node.name}-{os.getpid()}".replace("[", "-").replace("]", "")
    )
    log_prefix = namespace
    socket_root = Path(
        os.environ.get(
            "DYN_TRTLLM_SHADOW_FAILOVER_SOCKET_ROOT", tempfile.gettempdir()
        )
    )
    socket_root.mkdir(parents=True, exist_ok=True)
    gms_socket_dir = Path(tempfile.mkdtemp(prefix="gms-", dir=socket_root))
    failover_lock_path = gms_socket_dir / "failover.lock"
    allocated_ports = allocate_ports(count=shadow_count + 4, start_port=10000)
    frontend_port = allocated_ports[0]
    primary_system_port = allocated_ports[1]
    standby_system_ports = allocated_ports[2 : 2 + shadow_count]
    etcd_client_port = allocated_ports[-2]
    etcd_peer_port = allocated_ports[-1]
    etcd_endpoint = f"http://127.0.0.1:{etcd_client_port}"
    base_env = _base_env(
        namespace=namespace,
        etcd_endpoint=etcd_endpoint,
        gms_socket_dir=gms_socket_dir,
        failover_lock_path=failover_lock_path,
        cuda_visible_devices=cuda_visible_devices,
    )
    _enable_shadow_prewarm(
        base_env,
        gms_socket_dir=gms_socket_dir,
        shadow_count=shadow_count,
    )
    engine_command = _engine_command(
        namespace=namespace,
        model=model,
        served_model_name=served_model_name,
        engine_yaml=engine_yaml,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        kv_block_size=_DEFAULT_KV_BLOCK_SIZE,
    )

    processes: dict[str, ManagedProcess] = {}
    try:
        with ExitStack() as stack:
            etcd = stack.enter_context(
                ManagedProcess(
                    command=[
                        "etcd",
                        "--data-dir",
                        str(tmp_path / "etcd-data"),
                        "--listen-client-urls",
                        etcd_endpoint,
                        "--advertise-client-urls",
                        etcd_endpoint,
                        "--listen-peer-urls",
                        f"http://127.0.0.1:{etcd_peer_port}",
                        "--initial-advertise-peer-urls",
                        f"http://127.0.0.1:{etcd_peer_port}",
                        "--initial-cluster",
                        f"default=http://127.0.0.1:{etcd_peer_port}",
                        "--log-level",
                        "info",
                    ],
                    env=base_env,
                    health_check_urls=[f"{etcd_endpoint}/health"],
                    timeout=60,
                    display_output=True,
                    log_dir=f"{log_prefix}_etcd",
                    display_name="etcd",
                    terminate_all_matching_process_names=False,
                )
            )
            processes["etcd"] = etcd

            gms = stack.enter_context(
                ManagedProcess(
                    command=[sys.executable, "-m", "gpu_memory_service.cli.server"],
                    env=base_env,
                    health_check_funcs=[
                        lambda: _gms_ready_file_exists(gms_socket_dir)
                    ],
                    timeout=300,
                    display_output=True,
                    log_dir=f"{log_prefix}_gms",
                    display_name="gms",
                    terminate_all_matching_process_names=False,
                )
            )
            processes["gms"] = gms

            primary = _start_engine(
                stack,
                request,
                log_prefix=log_prefix,
                name="primary",
                engine_id="0",
                system_port=primary_system_port,
                command=engine_command,
                env=base_env,
            )
            processes["primary"] = primary
            _wait_for_log(
                "primary",
                primary,
                "[Shadow] Primary acquired startup lock",
                timeout_s=300,
                processes=processes,
            )

            standbys: dict[str, ManagedProcess] = {}
            standby_ports_by_name: dict[str, int] = {}
            for engine_index, standby_system_port in enumerate(
                standby_system_ports, start=1
            ):
                standby_name = (
                    "standby" if shadow_count == 1 else f"standby-{engine_index}"
                )
                standby = _start_engine(
                    stack,
                    request,
                    log_prefix=log_prefix,
                    name=standby_name,
                    engine_id=str(engine_index),
                    system_port=standby_system_port,
                    command=engine_command,
                    env=base_env,
                )
                processes[standby_name] = standby
                standbys[standby_name] = standby
                standby_ports_by_name[standby_name] = standby_system_port

            for standby_name, standby in standbys.items():
                _wait_for_log(
                    standby_name,
                    standby,
                    "[Shadow] Standby startup probe now passing; "
                    "starting RO engine init before failover",
                    timeout_s=600,
                    processes=processes,
                )
                ro_marker = _wait_for_any_log(
                    standby_name,
                    standby,
                    (
                        "Connected with ro lock (granted=ro), committed=True",
                        "[GMS] Created ro allocator for tag=weights",
                    ),
                    timeout_s=900,
                    processes=processes,
                )
                logger.info(
                    "%s GMS RO pre-failover marker: %s",
                    standby_name,
                    ro_marker,
                )
                _wait_for_log(
                    standby_name,
                    standby,
                    "[Shadow] Standby parked for fast failover",
                    timeout_s=900,
                    processes=processes,
                )
                _wait_for_log(
                    standby_name,
                    standby,
                    "[Shadow] Standby engine initialized in RO mode; "
                    "waiting for failover lock",
                    timeout_s=900,
                    processes=processes,
                )

            _wait_for_log(
                "primary",
                primary,
                "[Shadow] Primary prewarm complete; starting serving path",
                timeout_s=1800,
                processes=processes,
            )
            _wait_for_http_json(
                f"http://127.0.0.1:{primary_system_port}/health",
                "primary health",
                timeout_s=1800,
                processes=processes,
                process_to_check=("primary", primary),
            )
            frontend = stack.enter_context(
                DynamoFrontendProcess(
                    request,
                    frontend_port=frontend_port,
                    router_mode="kv",
                    extra_args=[
                        "--namespace",
                        namespace,
                        "--model-name",
                        served_model_name,
                        "--kv-cache-block-size",
                        str(_DEFAULT_KV_BLOCK_SIZE),
                        "--discovery-backend",
                        "etcd",
                        "--request-plane",
                        "tcp",
                        "--event-plane",
                        "zmq",
                        "--router-min-initial-workers",
                        "1",
                    ],
                    extra_env=base_env,
                    log_dir=f"{log_prefix}_frontend",
                    display_name="frontend",
                )
            )
            processes["frontend"] = frontend
            _wait_for_model(
                frontend_port,
                served_model_name,
                timeout_s=300,
                processes=processes,
            )

            primary_response = _post_chat(
                frontend_port,
                served_model_name,
                "What is two plus two? Answer in one short sentence.",
                timeout_s=600,
            )
            primary_content, _, _ = _extract_message_text(primary_response)
            assert primary_content.strip(), (
                f"primary response was empty: {primary_response}"
            )

            _require_running("primary", primary)
            for standby_name, standby in standbys.items():
                assert "[Shadow] Lock acquired, starting standby endpoint" not in (
                    standby.read_logs()
                ), f"{standby_name} became active before primary failover"
                assert "[Shadow] MDC published; standby is now discoverable" not in (
                    standby.read_logs()
                ), f"{standby_name} was made routable before primary failover"

            pre_failover_response = _post_chat(
                frontend_port,
                served_model_name,
                "What is two plus two? Answer in one short sentence.",
                timeout_s=120,
            )
            pre_failover_content, _, _ = _extract_message_text(pre_failover_response)
            assert pre_failover_content.strip(), (
                f"pre-failover response was empty: {pre_failover_response}"
            )

            failover_started_at = _kill_process_group(primary, wait=False)
            processes.pop("primary", None)

            winner_name, winner = _wait_for_any_process_log(
                standbys,
                "[Shadow] Lock acquired, starting standby endpoint",
                timeout_s=max_failover_ready_s,
                interval_s=0.05,
                processes=processes,
            )
            _wait_for_http_json(
                f"http://127.0.0.1:{standby_ports_by_name[winner_name]}/health",
                f"{winner_name} health after failover",
                timeout_s=30,
                processes=processes,
                process_to_check=(winner_name, winner),
            )
            failover_ready_s = time.monotonic() - failover_started_at
            assert failover_ready_s <= max_failover_ready_s, (
                f"standby failover readiness took {failover_ready_s:.2f}s, "
                f"exceeding {max_failover_ready_s:.2f}s"
            )

            post_restore_started_at = time.monotonic()
            post_restore_response = _post_chat(
                frontend_port,
                served_model_name,
                os.environ.get(
                    "DYN_TRTLLM_SHADOW_FAILOVER_COHERENCE_PROMPT",
                    _DEFAULT_COHERENCE_PROMPT,
                ),
                timeout_s=post_restore_response_timeout_s,
                retry_status_codes=(404, 500, 502, 503, 504),
                max_tokens=_env_int(
                    "DYN_TRTLLM_SHADOW_FAILOVER_COHERENCE_MAX_TOKENS",
                    _DEFAULT_COHERENCE_MAX_TOKENS,
                ),
            )
            post_restore_response_s = time.monotonic() - post_restore_started_at
            kill_to_coherent_s = time.monotonic() - failover_started_at
            logger.info(
                "%s failover ready in %.2fs; coherent post-restore response "
                "in %.2fs (%.2fs after SIGKILL)",
                winner_name,
                failover_ready_s,
                post_restore_response_s,
                kill_to_coherent_s,
            )
            _assert_coherent_post_restore_response(post_restore_response)
    finally:
        deallocate_ports(allocated_ports)
        shutil.rmtree(gms_socket_dir, ignore_errors=True)
