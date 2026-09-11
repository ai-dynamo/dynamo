# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster KVCR Guard resiliency verification.

This test is intentionally outside the generic deployment matrix. It needs a
KVCR-capable image, two GPU nodes, and a real RDMA resource. Run it explicitly:

    DYNAMO_UCX_NET_DEVICES=mlx5_0:1 \
    DYNAMO_KVCR_COMPATIBILITY_DIGEST=qwen3-0.6b-example-v1 \
    python3 -m pytest tests/deploy/test_kvcr_guard.py \
      -m framework_with_kvcr --image=<image> --namespace=<empty-namespace> \
      --skip-service-restart -v -s

The test renders the supported memory-service example, preserves phase logs,
kills only one vLLM EngineCore, and proves that the surviving engine retrieves
the prefix from the failed engine's KVCR Guard over UCX RDMA.
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from tests.deploy.dgd_utils import (
    DeploymentSpec,
    ManagedDeployment,
    validate_chat_response,
)
from tests.utils.client import send_request, wait_for_model_availability
from tests.utils.prometheus import sum_metric_samples

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3-0.6B"
WORKER = "worker"
MAIN = "main"
KVCR_SERVICES = "kvcr-services"
READINESS_TIMEOUT = 900
FAILURE_TIMEOUT = 240
TRANSFER_BLOCKS = "vllm:kvcr_transfer_blocks_total"
TIER_READ_BYTES = "vllm:kv_offload_tiering_read_bytes_total"
TIER_WRITE_BYTES = "vllm:kv_offload_tiering_write_bytes_total"
PROMPT_TOKENS = "vllm:prompt_tokens_by_source_total"
GUARD_PROMOTED = "KVCR_EVENT guard_promoted"
RDMA_PROTOCOL_RE = re.compile(r"\brc_mlx5\b")

# Large enough to make the transfer visible, while remaining comfortably below
# the small model's context limit and the example's 2 GiB host-memory pool.
LONG_PREFIX = (
    "KVCR resiliency validation preserves this numbered cache prefix. " * 600
    + "Reply with one short sentence confirming that the prefix was received."
)


def _kubectl(
    namespace: str,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _exec(
    namespace: str,
    pod_name: str,
    container: str,
    *command: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _kubectl(
        namespace,
        "exec",
        pod_name,
        "-c",
        container,
        "--",
        *command,
        check=check,
    )


def _pod_json(namespace: str, pod_name: str) -> dict:
    result = _kubectl(namespace, "get", "pod", pod_name, "-o", "json")
    return json.loads(result.stdout)


def _container_status(namespace: str, pod_name: str, container: str) -> dict:
    pod = _pod_json(namespace, pod_name)
    statuses = {status["name"]: status for status in pod["status"]["containerStatuses"]}
    return statuses[container]


def _metrics(namespace: str, pod_name: str, system_port: int) -> str:
    snippet = (
        "import urllib.request;"
        "print(urllib.request.urlopen("
        f"'http://127.0.0.1:{system_port}/metrics',timeout=10).read().decode())"
    )
    return _exec(namespace, pod_name, MAIN, "python3", "-c", snippet).stdout


def _metric_delta(
    before: str,
    after: str,
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    return sum_metric_samples(after, name, labels) - sum_metric_samples(
        before, name, labels
    )


def _logs(namespace: str, pod_name: str, container: str) -> str:
    return _kubectl(
        namespace,
        "logs",
        pod_name,
        "-c",
        container,
        "--tail=20000",
    ).stdout


def _rdma_counter(
    namespace: str,
    pod_name: str,
    container: str,
    device_port: str,
    counter: str,
) -> int:
    device, port = device_port.split(":", 1)
    port_path = f"/host/sys/class/infiniband/{device}/ports/{port}"
    state_path = f"{port_path}/state"
    counter_path = f"{port_path}/counters/{counter}"
    state = _exec(namespace, pod_name, container, "cat", state_path).stdout.strip()
    assert state == "4: ACTIVE", f"{device_port} is not active: {state}"
    result = _exec(namespace, pod_name, container, "cat", counter_path)
    return int(result.stdout.strip()) * 4


def _wait_until(predicate, description: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error: subprocess.CalledProcessError | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except subprocess.CalledProcessError as error:
            last_error = error
        time.sleep(2)
    detail = ""
    if last_error is not None:
        detail = f": {last_error.stderr.strip() or last_error.stdout.strip()}"
    raise AssertionError(
        f"Timed out waiting for {description} after {timeout}s{detail}"
    )


def _capture_workers(deployment: ManagedDeployment, pods: list, suffix: str) -> None:
    for pod in pods:
        deployment.get_pod_manifest_logs_metrics(WORKER, pod, suffix)


def _render_manifest(tmp_path: Path, image: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    deploy_script = repo_root / "examples/backends/vllm/deploy/kvcr/deploy.sh"
    env = os.environ.copy()
    env["DYNAMO_VLLM_IMAGE"] = image
    env["KVCR_MEMORY_SERVICE_ENABLED"] = "true"
    result = subprocess.run(
        [str(deploy_script), "--render-only"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    manifest = yaml.safe_load(result.stdout)
    worker = next(
        component
        for component in manifest["spec"]["components"]
        if component["name"] == WORKER
    )
    pod_spec = worker["podTemplate"]["spec"]
    pod_spec["volumes"].append(
        {
            "name": "rdma-counters",
            "hostPath": {"path": "/sys", "type": "Directory"},
        }
    )
    for container in pod_spec["containers"]:
        container["volumeMounts"].append(
            {
                "name": "rdma-counters",
                "mountPath": "/host/sys",
                "readOnly": True,
            }
        )
    manifest_path = tmp_path / "kvcr-memory-service.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


def _request(url: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": LONG_PREFIX}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": False,
    }
    response = send_request(url, payload, timeout=120, log_level=logging.DEBUG)
    data = validate_chat_response(response, MODEL, min_content_length=1)
    return data["choices"][0]["message"]["content"]


@pytest.mark.pre_merge
@pytest.mark.unit
@pytest.mark.gpu_0
def test_rdma_counter_uses_configured_active_port(monkeypatch) -> None:
    commands = []
    port_state = {"value": "4: ACTIVE"}

    def fake_exec(*args, **_kwargs):
        commands.append(args)
        stdout = port_state["value"] if args[-1].endswith("/state") else "1024"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(f"{__name__}._exec", fake_exec)

    value = _rdma_counter("test", "worker-0", "main", "mlx5_8:2", "port_rcv_data")

    assert value == 4096
    assert commands[0][-1] == "/host/sys/class/infiniband/mlx5_8/ports/2/state"
    assert commands[1][-1] == (
        "/host/sys/class/infiniband/mlx5_8/ports/2/counters/port_rcv_data"
    )

    port_state["value"] = "5: ACTIVE_DEFER"
    with pytest.raises(AssertionError, match="mlx5_8:2 is not active"):
        _rdma_counter("test", "worker-0", "main", "mlx5_8:2", "port_rcv_data")


@pytest.mark.framework_with_kvcr
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.fault_tolerance
# Nightly is the lifecycle budget. The test remains manual until a dedicated
# framework_with_kvcr two-host RDMA lane is provisioned.
@pytest.mark.nightly
@pytest.mark.e2e
# No framework marker: this is a manual, opt-in two-host RDMA test selected by
# framework_with_kvcr, not by the generic vLLM GPU lanes.
@pytest.mark.gpu_2
@pytest.mark.slow
@pytest.mark.model(MODEL)
@pytest.mark.timeout(3600)
async def test_kvcr_memory_service_guard_serves_after_engine_restart(
    image: str,
    namespace: str,
    skip_service_restart: bool,
    request,
    tmp_path: Path,
) -> None:
    """Prove a Guard serves remote KV while its local vLLM engine is down."""
    assert image, "--image is required for the KVCR Guard test"
    assert namespace, "--namespace is required for the KVCR Guard test"
    assert skip_service_restart, (
        "--skip-service-restart is required; this test must not restart shared "
        "Dynamo platform services"
    )
    assert os.environ.get(
        "DYNAMO_KVCR_COMPATIBILITY_DIGEST"
    ), "DYNAMO_KVCR_COMPATIBILITY_DIGEST is required"
    ucx_device = os.environ.get("DYNAMO_UCX_NET_DEVICES", "")
    assert re.fullmatch(
        r"mlx5_\d+:\d+", ucx_device
    ), "DYNAMO_UCX_NET_DEVICES must name one device and port, e.g. mlx5_0:1"

    deployment_spec = DeploymentSpec(_render_manifest(tmp_path, image))
    deployment_spec.namespace = namespace
    logger.info(
        "KVCR_TEST start image=%s namespace=%s ucx_device=%s",
        image,
        namespace,
        ucx_device,
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        readiness_timeout=READINESS_TIMEOUT,
    ) as deployment:
        pods = deployment.get_pods([WORKER])[WORKER]
        assert (
            len(pods) == 2
        ), f"Expected two worker Pods, found {[p.name for p in pods]}"
        pods.sort(
            key=lambda pod: int(
                pod.raw["metadata"]["labels"]["grove.io/podclique-pod-index"]
            )
        )
        indices = {
            pod.raw["metadata"]["labels"]["grove.io/podclique-pod-index"]
            for pod in pods
        }
        nodes = {pod.raw["spec"]["nodeName"] for pod in pods}
        assert indices == {"0", "1"}, f"Unexpected Grove indices: {indices}"
        assert len(nodes) == 2, f"Workers must run on separate hosts, found {nodes}"

        gpu_inventory = {}
        for pod in pods:
            gpu_inventory[pod.name] = _exec(
                namespace, pod.name, MAIN, "nvidia-smi", "-L"
            ).stdout.strip()
            for container in (MAIN, KVCR_SERVICES):
                _exec(
                    namespace,
                    pod.name,
                    container,
                    "bash",
                    "-c",
                    "compgen -G '/dev/infiniband/uverbs*' >/dev/null",
                )
        logger.info(
            "KVCR_TEST topology workers=%s gpus=%s",
            {pod.name: pod.raw["spec"]["nodeName"] for pod in pods},
            gpu_inventory,
        )

        frontend = deployment.get_pods([deployment.frontend_service_name])[
            deployment.frontend_service_name
        ][0]
        port_forward = deployment.port_forward(frontend, deployment_spec.port)
        assert port_forward is not None, "Could not port-forward the Dynamo frontend"
        base_url = f"http://127.0.0.1:{port_forward.local_port}"
        assert wait_for_model_availability(
            url=base_url,
            endpoint=deployment_spec.endpoint,
            model=MODEL,
            logger=logger,
            max_attempts=30,
        )
        request_url = f"{base_url}{deployment_spec.endpoint}"

        before_seed = {
            pod.name: _metrics(namespace, pod.name, deployment_spec.system_port)
            for pod in pods
        }
        baseline_content = _request(request_url)
        seed_deltas: dict[str, dict[str, float]] = {}

        def source_cache_persisted() -> bool:
            for pod in pods:
                after = _metrics(namespace, pod.name, deployment_spec.system_port)
                seed_deltas[pod.name] = {
                    "blocks": _metric_delta(
                        before_seed[pod.name],
                        after,
                        TRANSFER_BLOCKS,
                        {"operation": "local_fill"},
                    ),
                    "bytes": _metric_delta(
                        before_seed[pod.name],
                        after,
                        TIER_WRITE_BYTES,
                        {"tier": "1:kvcr"},
                    ),
                }
            persisted = [
                name
                for name, deltas in seed_deltas.items()
                if all(value > 0 for value in deltas.values())
            ]
            return len(persisted) == 1

        _wait_until(source_cache_persisted, "source KVCR cache persistence", 60)
        source_name = next(
            name
            for name, deltas in seed_deltas.items()
            if all(value > 0 for value in deltas.values())
        )
        source = next(pod for pod in pods if pod.name == source_name)
        target = next(pod for pod in pods if pod.name != source.name)
        assert seed_deltas[target.name] == {"blocks": 0.0, "bytes": 0.0}, (
            "The target was unexpectedly seeded by the first request: " f"{seed_deltas}"
        )
        logger.info(
            "KVCR_TEST seed source=%s target=%s blocks=%d bytes=%d",
            source.name,
            target.name,
            int(seed_deltas[source.name]["blocks"]),
            int(seed_deltas[source.name]["bytes"]),
        )
        _capture_workers(deployment, pods, ".before-failure")

        main_before = _container_status(namespace, source.name, MAIN)
        sidecar_before = _container_status(namespace, source.name, KVCR_SERVICES)
        guard_promotions_before = _logs(namespace, source.name, KVCR_SERVICES).count(
            GUARD_PROMOTED
        )
        _exec(
            namespace,
            source.name,
            KVCR_SERVICES,
            "touch",
            "/run/kvcr/hold-engine-start",
        )
        kill_result = _exec(
            namespace,
            source.name,
            MAIN,
            "pkill",
            "-9",
            "-f",
            "[V]LLM::EngineCore",
            check=False,
        )
        assert kill_result.returncode == 0, (
            "Could not terminate the source EngineCore: "
            f"{kill_result.stderr.strip()}"
        )

        def engine_restarted() -> bool:
            return (
                _container_status(namespace, source.name, MAIN)["restartCount"]
                >= main_before["restartCount"] + 1
            )

        _wait_until(
            engine_restarted, "the source vLLM container restart", FAILURE_TIMEOUT
        )
        _wait_until(
            lambda: _logs(namespace, source.name, KVCR_SERVICES).count(GUARD_PROMOTED)
            > guard_promotions_before,
            "KVCR Guard promotion",
            FAILURE_TIMEOUT,
        )
        sidecar_failed = _container_status(namespace, source.name, KVCR_SERVICES)
        assert sidecar_failed["restartCount"] == sidecar_before["restartCount"]
        assert sidecar_failed["ready"], "KVCR service sidecar became unready"
        assert (
            _exec(
                namespace,
                source.name,
                MAIN,
                "pgrep",
                "-f",
                "[V]LLM::EngineCore",
                check=False,
            ).returncode
            != 0
        ), "Source EngineCore restarted despite the hold marker"
        _capture_workers(deployment, pods, ".source-failed")

        target_before = _metrics(namespace, target.name, deployment_spec.system_port)
        source_protocols_before = len(
            RDMA_PROTOCOL_RE.findall(_logs(namespace, source.name, KVCR_SERVICES))
        )
        target_protocols_before = len(
            RDMA_PROTOCOL_RE.findall(_logs(namespace, target.name, MAIN))
        )
        xmit_before = _rdma_counter(
            namespace,
            source.name,
            KVCR_SERVICES,
            ucx_device,
            "port_xmit_data",
        )
        recv_before = _rdma_counter(
            namespace,
            target.name,
            MAIN,
            ucx_device,
            "port_rcv_data",
        )
        recovered_content = _request(request_url)
        assert recovered_content == baseline_content
        transfer_deltas: dict[str, float] = {}

        def target_metrics_published() -> bool:
            target_after = _metrics(namespace, target.name, deployment_spec.system_port)
            transfer_deltas.update(
                blocks=_metric_delta(
                    target_before,
                    target_after,
                    TRANSFER_BLOCKS,
                    {"operation": "remote_deliver"},
                ),
                tier_bytes=_metric_delta(
                    target_before,
                    target_after,
                    TIER_READ_BYTES,
                    {"tier": "1:kvcr"},
                ),
                external_tokens=_metric_delta(
                    target_before,
                    target_after,
                    PROMPT_TOKENS,
                    {"source": "external_kv_transfer"},
                ),
            )
            return all(value > 0 for value in transfer_deltas.values())

        _wait_until(target_metrics_published, "target KVCR transfer metrics", 60)
        blocks = transfer_deltas["blocks"]
        tier_bytes = transfer_deltas["tier_bytes"]
        external_tokens = transfer_deltas["external_tokens"]

        source_logs = _logs(namespace, source.name, KVCR_SERVICES)
        target_logs = _logs(namespace, target.name, MAIN)
        assert (
            len(RDMA_PROTOCOL_RE.findall(source_logs)) > source_protocols_before
        ), "Source Guard did not select UCX rc_mlx5 for the remote request"
        assert (
            len(RDMA_PROTOCOL_RE.findall(target_logs)) > target_protocols_before
        ), "Target did not select UCX rc_mlx5 for the remote request"
        xmit_bytes = (
            _rdma_counter(
                namespace,
                source.name,
                KVCR_SERVICES,
                ucx_device,
                "port_xmit_data",
            )
            - xmit_before
        )
        recv_bytes = (
            _rdma_counter(
                namespace,
                target.name,
                MAIN,
                ucx_device,
                "port_rcv_data",
            )
            - recv_before
        )
        assert xmit_bytes >= tier_bytes, (
            "Source HCA transmit bytes did not cover the KVCR payload: "
            f"hca={xmit_bytes}, kvcr={int(tier_bytes)}"
        )
        assert recv_bytes >= tier_bytes, (
            "Target HCA receive bytes did not cover the KVCR payload: "
            f"hca={recv_bytes}, kvcr={int(tier_bytes)}"
        )
        logger.info(
            "KVCR_TEST transfer source=%s target=%s blocks=%d bytes=%d "
            "external_tokens=%d hca_xmit_bytes=%d hca_recv_bytes=%d "
            "protocol=rc_mlx5",
            source.name,
            target.name,
            int(blocks),
            int(tier_bytes),
            int(external_tokens),
            xmit_bytes,
            recv_bytes,
        )
        _capture_workers(deployment, pods, ".remote-delivery")

        _exec(
            namespace,
            source.name,
            KVCR_SERVICES,
            "rm",
            "-f",
            "/run/kvcr/hold-engine-start",
        )
        _wait_until(
            lambda: bool(_metrics(namespace, source.name, deployment_spec.system_port)),
            "the source engine metrics endpoint recovery",
            READINESS_TIMEOUT,
        )
        logger.info(
            "KVCR_TEST result=PASS source_main_restarts=%d "
            "source_sidecar_restarts=%d guard_blocks=%d guard_bytes=%d",
            _container_status(namespace, source.name, MAIN)["restartCount"]
            - main_before["restartCount"],
            _container_status(namespace, source.name, KVCR_SERVICES)["restartCount"]
            - sidecar_before["restartCount"],
            int(blocks),
            int(tier_bytes),
        )
        _capture_workers(deployment, pods, ".source-recovered")
