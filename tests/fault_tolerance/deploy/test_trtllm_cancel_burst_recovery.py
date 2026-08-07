# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM AIPerf cancellation burst recovery scenario.

This test covers the production symptom where a burst of long prompts followed by
client cancellations can leave a deployment wedged even though Kubernetes pods
remain healthy. The oracle is serving behavior: visible drain plus a fresh request
that must complete after the cancellation storm.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.utils.client import wait_for_model_availability
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment
from tests.utils.test_output import resolve_test_output_path

logger = logging.getLogger(__name__)

_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class _Topology:
    name: str
    template: str
    worker_services: tuple[str, ...]


_TOPOLOGIES = {
    "agg": _Topology(
        name="agg",
        template=(
            "tests/fault_tolerance/deploy/templates/trtllm/"
            "agg_cancel_burst_recovery.yaml"
        ),
        worker_services=("TRTLLMWorker",),
    ),
    "disagg": _Topology(
        name="disagg",
        template=(
            "tests/fault_tolerance/deploy/templates/trtllm/"
            "disagg_cancel_burst_recovery.yaml"
        ),
        worker_services=("prefill", "decode"),
    ),
}


@dataclass(frozen=True)
class _Profile:
    cycles: int
    pressure_request_count: int
    storm_request_count: int
    recovery_request_count: int
    concurrency: int


_PROFILES = {
    "default": _Profile(
        cycles=1,
        pressure_request_count=16,
        storm_request_count=32,
        recovery_request_count=3,
        concurrency=8,
    ),
    "soak": _Profile(
        cycles=5,
        pressure_request_count=16,
        storm_request_count=32,
        recovery_request_count=3,
        concurrency=8,
    ),
}

_DEBUG_FRESH_PROBE_FAILURES = {"off", "timeout_records"}


def _scenario_set_values(request) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in request.config.getoption("scenario_set_values", default=[]):
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _template_path(topology: _Topology) -> str:
    return os.path.join(os.getcwd(), topology.template)


def _pod_url(
    deployment: ManagedDeployment, service_name: str, remote_port: int
) -> tuple[str, Any]:
    pods = deployment.get_pods([service_name])
    ready = [pod for pod in pods[service_name] if pod.ready()]
    assert ready, f"No ready {service_name} pod found"

    port_forward = deployment.port_forward(ready[0], remote_port)
    assert port_forward is not None, f"Failed to port-forward {service_name}"
    return f"http://localhost:{port_forward.local_port}", port_forward


def _frontend_url(deployment: ManagedDeployment) -> tuple[str, Any]:
    return _pod_url(
        deployment, deployment.frontend_service_name, deployment.deployment_spec.port
    )


def _worker_metrics_urls(
    deployment: ManagedDeployment, service_names: tuple[str, ...]
) -> tuple[dict[str, str], dict[str, Any]]:
    urls: dict[str, str] = {}
    port_forwards: dict[str, Any] = {}
    for service_name in service_names:
        url, port_forward = _pod_url(
            deployment, service_name, deployment.deployment_spec.system_port
        )
        urls[service_name] = url
        port_forwards[service_name] = port_forward
    return urls, port_forwards


def _run_aiperf_phase(
    *,
    name: str,
    url: str,
    artifact_dir: Path,
    request_count: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    request_cancellation_rate: int = 0,
    request_cancellation_delay: float = 0.0,
    allow_nonzero: bool = False,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aiperf",
        "profile",
        "--artifact-dir",
        str(artifact_dir),
        "--model",
        _MODEL,
        "--url",
        url,
        "--endpoint",
        "/v1/chat/completions",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--request-timeout-seconds",
        "90",
        "--synthetic-input-tokens-mean",
        str(input_tokens),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(output_tokens),
        "--output-tokens-stddev",
        "0",
        "--random-seed",
        "100",
        "--ui",
        "simple",
        "--verbose",
    ]
    if request_cancellation_rate:
        cmd.extend(["--request-cancellation-rate", str(request_cancellation_rate)])
        if request_cancellation_delay:
            cmd.extend(
                ["--request-cancellation-delay", str(request_cancellation_delay)]
            )

    logger.info("Running %s: %s", name, " ".join(cmd))
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    (artifact_dir / "aiperf.stdout.log").write_text(result.stdout)
    (artifact_dir / "aiperf.stderr.log").write_text(result.stderr)

    if result.returncode != 0 and not allow_nonzero:
        raise AssertionError(
            f"AIPerf phase {name!r} failed with return code {result.returncode}. "
            f"See {artifact_dir}"
        )
    if result.returncode != 0:
        logger.info(
            "AIPerf phase %s exited with return code %s; validating raw records",
            name,
            result.returncode,
        )


def _iter_raw_records(artifact_dir: Path) -> list[dict[str, Any]]:
    raw_path = artifact_dir / "profile_export.jsonl"
    if not raw_path.exists():
        raise AssertionError(
            f"Mandatory raw AIPerf records missing: {raw_path}. "
            "CI requires profile_export.jsonl so all-timeout fresh probes fail "
            "even when profile_export_aiperf.json is absent."
        )

    records: list[dict[str, Any]] = []
    with raw_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Malformed raw AIPerf JSONL row: {exc}") from exc
    if not records:
        raise AssertionError(f"Mandatory raw AIPerf record file is empty: {raw_path}")
    return records


def _record_text(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, default=str).lower()


def _has_nonempty_error_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_l = str(key).lower()
            if key_l in {"error", "exception"} and child not in (None, "", [], {}):
                return True
            if _has_nonempty_error_field(child):
                return True
    elif isinstance(value, list):
        return any(_has_nonempty_error_field(child) for child in value)
    return False


def _is_error_record(record: dict[str, Any]) -> bool:
    text = _record_text(record)
    status = record.get("status_code") or record.get("http_status")
    if isinstance(status, int) and status >= 400:
        return True
    if _has_nonempty_error_field(record):
        return True
    return any(
        marker in text
        for marker in (
            "timeout",
            "abort",
            "connection reset",
            "connection closed",
        )
    )


def _is_cancel_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("was_cancelled") is True:
        return True
    error = record.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        error_type = str(error.get("type") or "").lower()
        message = str(error.get("message") or "").lower()
        if code == 499 or "cancel" in error_type or "cancel" in message:
            return True
    text = _record_text(record)
    return any(
        marker in text
        for marker in (
            "abort",
            "connection reset",
            "connection closed",
            "disconnect",
        )
    )


def _extract_ms(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    stack: list[Any] = [record]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                if key in keys and isinstance(value, (int, float)):
                    return float(value)
                stack.append(value)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _assert_cancel_storm_records(artifact_dir: Path, min_cancelled: int) -> None:
    records = _iter_raw_records(artifact_dir)
    cancelled = sum(1 for record in records if _is_cancel_record(record))
    logger.info(
        "Cancel storm raw records: total=%d cancelled=%d min_cancelled=%d",
        len(records),
        cancelled,
        min_cancelled,
    )
    assert cancelled >= min_cancelled, (
        f"Expected at least {min_cancelled} cancellation records in {artifact_dir}, "
        f"got {cancelled}/{len(records)}"
    )


def _assert_fresh_probe_records(artifact_dir: Path, expected_successes: int) -> None:
    records = _iter_raw_records(artifact_dir)
    errors = [record for record in records if _is_error_record(record)]
    successes = len(records) - len(errors)
    ttfts = [
        value
        for value in (
            _extract_ms(
                record,
                (
                    "time_to_first_token",
                    "time_to_first_token_ms",
                    "ttft",
                    "ttft_ms",
                ),
            )
            for record in records
        )
        if value is not None
    ]
    logger.info(
        "Fresh probe raw records: total=%d successes=%d errors=%d ttft_ms=%s",
        len(records),
        successes,
        len(errors),
        ttfts,
    )

    assert successes >= expected_successes, (
        f"Fresh probe did not make forward progress: successes={successes}, "
        f"expected={expected_successes}, raw_records={len(records)}"
    )
    assert not errors, (
        f"Fresh probe had timeout/error records after cancellation storm: "
        f"errors={len(errors)}, raw_records={len(records)}, artifact_dir={artifact_dir}"
    )


def _inject_debug_fresh_probe_timeout_records(
    artifact_dir: Path, record_count: int
) -> None:
    raw_path = artifact_dir / "profile_export.jsonl"
    if raw_path.exists():
        backup_path = artifact_dir / "profile_export.before_debug_injection.jsonl"
        backup_path.write_text(raw_path.read_text())

    records = [
        {
            "status_code": 599,
            "error": {
                "type": "DebugInjectedTimeout",
                "message": (
                    "debug_fresh_probe_failure=timeout_records forced this "
                    "fresh probe record to validate the serving oracle"
                ),
            },
            "metadata": {
                "debug_fresh_probe_failure": "timeout_records",
                "phase": "fresh_probe",
            },
        }
        for _ in range(record_count)
    ]
    raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _assert_frontend_health(url: str) -> None:
    response = requests.get(f"{url}/health", timeout=30)
    response.raise_for_status()


def _scrape_metrics(base_url: str) -> str:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    return response.text


def _write_metrics_snapshot(
    artifact_dir: Path,
    *,
    frontend_url: str,
    worker_metrics_urls: dict[str, str],
    label: str,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frontend_metrics = _scrape_metrics(frontend_url)
    (artifact_dir / f"frontend_metrics_{label}.prom").write_text(frontend_metrics)

    worker_metrics: dict[str, str] = {}
    for service_name, metrics_url in worker_metrics_urls.items():
        metrics = _scrape_metrics(metrics_url)
        worker_metrics[service_name] = metrics
        (artifact_dir / f"{service_name}_metrics_{label}.prom").write_text(metrics)
    return worker_metrics


def _metric_samples(metrics: str, metric_name: str) -> list[float]:
    samples: list[float] = []
    for raw_line in metrics.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        has_name = line.startswith(f"{metric_name} ") or line.startswith(
            f"{metric_name}{{"
        )
        if not has_name:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            samples.append(float(parts[1]))
        except ValueError:
            continue
    return samples


def _assert_trtllm_metrics_available(
    metrics_by_service: dict[str, str], artifact_dir: Path
) -> None:
    missing = [
        service_name
        for service_name, metrics in metrics_by_service.items()
        if "trtllm_" not in metrics
    ]
    assert not missing, (
        "TRT-LLM worker(s) did not expose trtllm_* metrics: "
        f"{missing}. Saved metrics under {artifact_dir}"
    )


def _wait_for_visible_inflight_drain(
    *,
    worker_metrics_url: str,
    artifact_dir: Path,
    timeout_s: float = 60.0,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    last_metrics = ""
    last_dynamo_samples: list[float] = []
    last_trtllm_samples: list[float] = []
    while time.monotonic() < deadline:
        last_metrics = _scrape_metrics(worker_metrics_url)
        last_dynamo_samples = _metric_samples(
            last_metrics, "dynamo_component_inflight_requests"
        )
        last_trtllm_samples = _metric_samples(
            last_metrics, "trtllm_num_requests_running"
        )
        if last_dynamo_samples and max(last_dynamo_samples) == 0:
            (artifact_dir / "trtllm_worker_metrics_drained.prom").write_text(
                last_metrics
            )
            logger.info(
                "Worker inflight drained: dynamo_component_inflight_requests=%s "
                "trtllm_num_requests_running=%s",
                last_dynamo_samples,
                last_trtllm_samples,
            )
            return
        time.sleep(2)

    if last_metrics:
        (artifact_dir / "trtllm_worker_metrics_not_drained.prom").write_text(
            last_metrics
        )
    assert last_dynamo_samples, (
        "TRT-LLM worker metrics were available, but "
        "dynamo_component_inflight_requests was missing. "
        f"Saved metrics under {artifact_dir}"
    )
    raise AssertionError(
        "Dynamo worker inflight did not visibly drain after cancellation storm: "
        f"dynamo_component_inflight_requests samples={last_dynamo_samples}, "
        f"trtllm_num_requests_running samples={last_trtllm_samples}, "
        f"artifact_dir={artifact_dir}"
    )


def _wait_for_all_visible_inflight_drain(
    *,
    worker_metrics_urls: dict[str, str],
    artifact_dir: Path,
) -> None:
    for service_name, metrics_url in worker_metrics_urls.items():
        _wait_for_visible_inflight_drain(
            worker_metrics_url=metrics_url,
            artifact_dir=artifact_dir / service_name,
        )


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.nightly
@pytest.mark.timeout(7200)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "topology_name",
    (
        pytest.param("agg", marks=pytest.mark.gpu_1),
        pytest.param("disagg", marks=pytest.mark.gpu_2),
    ),
)
async def test_trtllm_cancel_burst_recovery(
    request,
    topology_name: str,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    """AIPerf long-prompt cancellation storm must not wedge TRT-LLM serving.

    The serving oracle is a separate fresh request phase after each storm. The
    fresh phase is validated from mandatory raw AIPerf records, so an all-timeout
    probe fails deterministically even if AIPerf does not write an aggregate
    ``profile_export_aiperf.json``.
    """

    set_values = _scenario_set_values(request)
    profile_name = set_values.get("profile", "default")
    assert (
        profile_name in _PROFILES
    ), f"Unknown profile {profile_name!r}; expected one of {sorted(_PROFILES)}"
    profile = _PROFILES[profile_name]
    debug_fresh_probe_failure = set_values.get("debug_fresh_probe_failure", "off")
    assert debug_fresh_probe_failure in _DEBUG_FRESH_PROBE_FAILURES, (
        f"Unknown debug_fresh_probe_failure {debug_fresh_probe_failure!r}; "
        f"expected one of {sorted(_DEBUG_FRESH_PROBE_FAILURES)}"
    )
    topology = _TOPOLOGIES[topology_name]

    spec = DeploymentSpec(_template_path(topology))
    spec.name = f"trtllm-{topology.name}-cancel-burst-recovery"
    if image:
        spec.set_image(image)
    spec.set_model(_MODEL)
    spec.set_logging(True, "debug")

    log_dir = Path(resolve_test_output_path(request.node.name))
    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=spec,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        url, port_forward = _frontend_url(deployment)
        worker_urls, worker_port_forwards = _worker_metrics_urls(
            deployment, topology.worker_services
        )
        try:
            assert wait_for_model_availability(
                url, "/v1/chat/completions", _MODEL, logger
            ), "Model never became available through Frontend"

            startup_metrics = _write_metrics_snapshot(
                log_dir / "startup",
                frontend_url=url,
                worker_metrics_urls=worker_urls,
                label="startup",
            )
            _assert_trtllm_metrics_available(startup_metrics, log_dir / "startup")

            for cycle in range(profile.cycles):
                cycle_dir = log_dir / f"cycle_{cycle}"
                logger.info("Starting TRT-LLM cancel burst recovery cycle %d", cycle)

                _run_aiperf_phase(
                    name=f"cycle-{cycle}-pressure",
                    url=url,
                    artifact_dir=cycle_dir / "pressure",
                    request_count=profile.pressure_request_count,
                    concurrency=profile.concurrency,
                    input_tokens=4096,
                    output_tokens=512,
                )
                _write_metrics_snapshot(
                    cycle_dir / "pressure",
                    frontend_url=url,
                    worker_metrics_urls=worker_urls,
                    label="after_pressure",
                )

                _run_aiperf_phase(
                    name=f"cycle-{cycle}-cancel-storm",
                    url=url,
                    artifact_dir=cycle_dir / "cancel_storm",
                    request_count=profile.storm_request_count,
                    concurrency=profile.concurrency,
                    input_tokens=4096,
                    output_tokens=512,
                    request_cancellation_rate=100,
                    request_cancellation_delay=0.25,
                    allow_nonzero=True,
                )
                _assert_cancel_storm_records(
                    cycle_dir / "cancel_storm",
                    min_cancelled=max(1, profile.storm_request_count // 2),
                )
                cancel_metrics = _write_metrics_snapshot(
                    cycle_dir / "cancel_storm",
                    frontend_url=url,
                    worker_metrics_urls=worker_urls,
                    label="after_cancel_storm",
                )
                _assert_trtllm_metrics_available(
                    cancel_metrics, cycle_dir / "cancel_storm"
                )
                _wait_for_all_visible_inflight_drain(
                    worker_metrics_urls=worker_urls,
                    artifact_dir=cycle_dir / "cancel_storm",
                )

                _assert_frontend_health(url)
                _run_aiperf_phase(
                    name=f"cycle-{cycle}-fresh-probe",
                    url=url,
                    artifact_dir=cycle_dir / "fresh_probe",
                    request_count=profile.recovery_request_count,
                    concurrency=1,
                    input_tokens=128,
                    output_tokens=32,
                )
                if debug_fresh_probe_failure == "timeout_records":
                    logger.warning(
                        "Injecting debug timeout records into fresh probe raw "
                        "records; this run is expected to fail the serving oracle"
                    )
                    _inject_debug_fresh_probe_timeout_records(
                        cycle_dir / "fresh_probe",
                        record_count=profile.recovery_request_count,
                    )
                _assert_fresh_probe_records(
                    cycle_dir / "fresh_probe",
                    expected_successes=profile.recovery_request_count,
                )
                fresh_metrics = _write_metrics_snapshot(
                    cycle_dir / "fresh_probe",
                    frontend_url=url,
                    worker_metrics_urls=worker_urls,
                    label="after_fresh_probe",
                )
                _assert_trtllm_metrics_available(
                    fresh_metrics, cycle_dir / "fresh_probe"
                )
        finally:
            port_forwards = {"Frontend": port_forward, **worker_port_forwards}
            for label, pf in port_forwards.items():
                try:
                    pf.stop()
                except OSError as exc:
                    logger.debug("%s port-forward cleanup failed: %s", label, exc)
