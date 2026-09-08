# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real image upgrades with a Service client and a two-GPU quota.

Operator stays at the candidate version. This suite does not upgrade the
operator binary, exercise Grove, or change worker discovery namespaces by hand.
"""

import asyncio
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from scripts.compatibility.rolling_contract import (
    check_budget,
    check_requests,
    converged,
    pod_record,
    request_count,
)
from scripts.compatibility.runner import check, command, matrix
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.utils.test_output import resolve_test_output_path

ROOT = Path(__file__).resolve().parents[2]


async def kubectl(namespace, *args, timeout=120):
    return await asyncio.to_thread(
        command, "kubectl", "--request-timeout=30s", "-n", namespace, *args, timeout=120
    )


def manifest(name, old, scenario, model, client_image):
    # One immutable model snapshot per Pod; no code from the checkout is
    # installed in either inference component. Init containers use no GPU.
    download = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({model['id']!r}, revision={model['revision']!r}, local_dir='/model')"
    )
    components = []
    for role, kind, image in [
        ("Frontend", "frontend", old["frontend"]),
        ("decode", "worker", old["worker"]),
    ]:
        args = (
            ["python3", "-m", "dynamo.frontend", "--http-port", "8000"]
            if kind == "frontend"
            else [
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                "/model",
                "--served-model-name",
                model["id"],
                "--tp",
                "1",
                "--mem-fraction-static",
                "0.65",
                "--enable-metrics",
            ]
        )
        if kind == "worker" and scenario == "embedding":
            args += [
                "--embedding-worker",
                "--use-sglang-tokenizer",
                "--page-size",
                "16",
            ]
        container = {
            "name": "main",
            "image": image,
            "command": args,
            "workingDir": "/tmp",
            "env": [
                {"name": "HF_HUB_OFFLINE", "value": "1"},
                {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                {"name": "DYN_REQUEST_PLANE", "value": "tcp"},
                {"name": "DYN_SYSTEM_PORT", "value": "8081"},
                # Active canaries would contaminate request-counter evidence.
                {"name": "DYN_HEALTH_CHECK_ENABLED", "value": "false"},
            ],
            "volumeMounts": [{"name": "model", "mountPath": "/model"}],
        }
        if kind == "worker":
            container["resources"] = {
                "requests": {"nvidia.com/gpu": "1"},
                "limits": {"nvidia.com/gpu": "1"},
            }
        components.append(
            {
                "name": role,
                "type": kind,
                "replicas": 2,
                "podTemplate": {
                    "metadata": {
                        "annotations": {
                            "nvidia.com/deployment-rolling-update-max-surge": "0",
                            "nvidia.com/deployment-rolling-update-max-unavailable": "1",
                        }
                    },
                    "spec": {
                        "containers": [container],
                        "initContainers": [
                            {
                                "name": "model",
                                "image": client_image,
                                "command": ["python3", "-c", download],
                                "envFrom": [{"secretRef": {"name": "hf-token-secret"}}],
                                "volumeMounts": [
                                    {"name": "model", "mountPath": "/model"}
                                ],
                            }
                        ],
                        "volumes": [{"name": "model", "emptyDir": {}}],
                    },
                },
            }
        )
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": name,
            "annotations": {
                "nvidia.com/enable-grove": "false",
                "nvidia.com/dynamo-discovery-backend": "etcd",
            },
        },
        "spec": {"components": components},
    }


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.sglang
@pytest.mark.core
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.timeout(5400)  # Includes initial model pulls plus two bounded rollouts.
@pytest.mark.parametrize("age", [1, 2], ids=["n-1", "n-2"])
@pytest.mark.parametrize("scenario", ["embedding", "chat"])
@pytest.mark.parametrize(
    "first", ["decode", "Frontend"], ids=["worker-first", "frontend-first"]
)
async def test_n2_rolling_upgrade(request, tmp_path, age, scenario, first):
    frontend = request.config.getoption("--frontend-image")
    worker = request.config.getoption("--image")
    check(
        bool(frontend and worker),
        "Pass --frontend-image and --image with candidate images",
    )
    namespace = request.config.getoption("--namespace") or "default"
    config = json.loads((ROOT / "scripts/compatibility/releases.json").read_text())
    version = re.search(
        r'^version = "(\d+\.\d+\.\d+)', (ROOT / "Cargo.toml").read_text(), re.M
    )[1]
    pairs = matrix(config["releases"], version.rsplit(".", 1)[0], frontend, worker)
    old = {"frontend": pairs[2 * age - 1][1], "worker": pairs[2 * age][2]}
    name = "n2-" + uuid.uuid4().hex[:10]
    output = Path(resolve_test_output_path(name))
    output.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "dgd.json"
    source.write_text(
        json.dumps(manifest(name, old, scenario, config["models"][scenario], worker))
    )
    (output / "initial-dgd.json").write_text(source.read_text())
    spec = DeploymentSpec(str(source), system_port=8081)
    candidates = {"Frontend": frontend, "decode": worker}
    report = {
        "old": old,
        "candidate": candidates,
        "model": config["models"][scenario],
        "order": first,
        "status": "failed",
        "stages": [],
    }
    client_name = name + "-client"
    quota_name = name + "-budget"
    config_name = name + "-client-code"
    resources = tmp_path / "resources.json"
    model = config["models"][scenario]
    resources.write_text(
        json.dumps(
            {
                "apiVersion": "v1",
                "kind": "List",
                "items": [
                    {
                        "apiVersion": "v1",
                        "kind": "ResourceQuota",
                        "metadata": {"name": quota_name},
                        "spec": {"hard": {"requests.nvidia.com/gpu": "2"}},
                    },
                    {
                        "apiVersion": "v1",
                        "kind": "ConfigMap",
                        "metadata": {"name": config_name},
                        "data": {
                            p: (ROOT / "scripts/compatibility" / p).read_text()
                            for p in ("runner.py", "rolling_client.py")
                        },
                    },
                ],
            }
        )
    )
    # This namespace must be dedicated to the suite; the workflow owns its vCluster.
    existing = json.loads(
        await kubectl(namespace, "get", "resourcequota", "-o", "json")
    )
    check(
        not existing["items"], "Use a dedicated test namespace without existing quotas"
    )

    async def client_exec(code):
        return await kubectl(
            namespace, "exec", client_name, "--", "python3", "-c", code
        )

    async def control(phase):
        await client_exec(
            "from pathlib import Path; p=Path('/state/control.tmp'); "
            f"p.write_text({json.dumps({'phase': phase})!r}); p.replace('/state/control.json')"
        )

    async def summary():
        return json.loads(
            await client_exec(
                "from pathlib import Path; p=Path('/state/summary.json'); "
                "print(p.read_text() if p.exists() else '{}')"
            )
        )

    async def observe():
        raw = json.loads(
            await kubectl(
                namespace,
                "get",
                "pods",
                "-l",
                f"nvidia.com/dynamo-graph-deployment-name={name}",
                "-o",
                "json",
            )
        )
        pods = [pod_record(p) for p in raw["items"]]
        with (output / "pods.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "pods": pods}) + "\n")
        check_budget(pods)

        async def capture_log(pod):
            try:
                logs = await kubectl(
                    namespace, "logs", pod["name"], "-c", "main", "--tail=2000"
                )
                (output / (pod["name"] + ".log")).write_text(logs)
            except subprocess.SubprocessError as error:
                # A Pod may terminate between list and logs. Preserve earlier
                # snapshots; log collection must not hide request failures.
                with (output / "log-collection-errors.txt").open("a") as f:
                    f.write(f"{pod['name']}: {error}\n")

        await asyncio.gather(*(capture_log(pod) for pod in pods if pod["ready"]))
        return pods

    async def counters(pods):
        endpoints = {
            p["uid"]: f"http://{p['ip']}:8081/metrics"
            for p in pods
            if p["component"] == "decode" and p["ready"] and p["ip"]
        }
        code = """import json, requests
result = {}
for uid, url in ENDPOINTS.items():
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        result[uid] = response.text
    except requests.RequestException:
        pass
print(json.dumps(result))
""".replace(
            "ENDPOINTS", repr(endpoints)
        )
        raw = json.loads(await client_exec(code))
        with (output / "worker-metrics.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "metrics": raw}) + "\n")
        return {uid: request_count(value) for uid, value in raw.items()}

    async def hold(phase, seconds=30):
        await control(phase)
        deadline = time.monotonic() + 180
        started = time.monotonic()
        while time.monotonic() < deadline:
            await observe()
            current = await summary()
            if current:
                check(
                    current["failures"] == 0,
                    "Client recorded a failed request; see client.jsonl",
                )
                counts = current["phases"].get(phase, {})
                if (
                    time.monotonic() - started >= seconds
                    and len(counts) == 3
                    and min(counts.values()) >= 2
                ):
                    return current
            await asyncio.sleep(2)
        raise TimeoutError(f"Insufficient successful traffic during {phase}")

    client_created = False
    client_stopped = False

    async def stop_client():
        await control("stop")
        await kubectl(
            namespace,
            "wait",
            "--for=jsonpath={.status.phase}=Succeeded",
            f"pod/{client_name}",
            "--timeout=120s",
        )
        logs = await kubectl(namespace, "logs", client_name)
        (output / "client.jsonl").write_text(logs)
        return json.loads(logs.splitlines()[-1])["summary"]

    try:
        await kubectl(namespace, "apply", "-f", str(resources))
        async with ManagedDeployment(
            log_dir=name,
            deployment_spec=spec,
            namespace=namespace,
            skip_service_restart=True,
            readiness_timeout=1800,
        ) as deployment:
            pods = await observe()
            for role in candidates:
                check(
                    converged(
                        pods,
                        role,
                        old["frontend" if role == "Frontend" else "worker"],
                        set(),
                    ),
                    f"Initial {role} replicas not ready",
                )
            client = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": client_name},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "client",
                            "image": worker,
                            "workingDir": "/tmp",
                            "command": [
                                "python3",
                                "/client/rolling_client.py",
                                "--base",
                                f"http://{name}-frontend.{namespace}.svc.cluster.local:8000",
                                "--scenario",
                                scenario,
                                "--model",
                                model["id"],
                                "--dimensions",
                                str(model.get("dimensions", 1024)),
                                "--state",
                                "/state",
                            ],
                            "volumeMounts": [
                                {"name": "code", "mountPath": "/client"},
                                {"name": "state", "mountPath": "/state"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "code", "configMap": {"name": config_name}},
                        {"name": "state", "emptyDir": {}},
                    ],
                },
            }
            client_file = tmp_path / "client.json"
            client_file.write_text(json.dumps(client))
            await kubectl(namespace, "apply", "-f", str(client_file))
            client_created = True
            try:
                await kubectl(
                    namespace,
                    "wait",
                    "--for=condition=Ready",
                    f"pod/{client_name}",
                    "--timeout=300s",
                    timeout=330,
                )
                await hold("before")
                phases = ["before"]
                for role in [first, "Frontend" if first == "decode" else "decode"]:
                    before = await observe()
                    old_uids = {p["uid"] for p in before if p["component"] == role}
                    initial_counts = await counters(before) if role == "decode" else {}
                    observed_counts = dict(initial_counts)
                    phase = "upgrade-" + role.lower()
                    phases.append(phase)
                    await control(phase)
                    await deployment.update_component_images(
                        {role: candidates[role]}, version
                    )
                    deadline = time.monotonic() + 1200
                    overlap = False
                    while time.monotonic() < deadline:
                        pods = await observe()
                        ready = [
                            p for p in pods if p["component"] == role and p["ready"]
                        ]
                        overlap |= any(p["uid"] in old_uids for p in ready) and any(
                            p["uid"] not in old_uids for p in ready
                        )
                        if role == "decode":
                            observed_counts.update(await counters(pods))
                        current = await summary()
                        check(
                            current.get("failures", 0) == 0,
                            "Request failed during image rollout",
                        )
                        if converged(pods, role, candidates[role], old_uids):
                            break
                        await asyncio.sleep(2)
                    else:
                        raise TimeoutError(
                            f"{role} did not converge to candidate in 1200s"
                        )
                    report["stages"].append(
                        {
                            "component": role,
                            "old_uids": sorted(old_uids),
                            "ready_overlap_observed": overlap,
                            "final_pods": pods,
                        }
                    )
                    if role == "decode":
                        old_delta = sum(
                            observed_counts.get(uid, value) - value
                            for uid, value in initial_counts.items()
                        )
                        new_delta = sum(
                            value
                            for uid, value in observed_counts.items()
                            if uid not in old_uids
                        )
                        report["stages"][-1]["request_counter_deltas"] = {
                            "old": old_delta,
                            "new": new_delta,
                        }
                        check(
                            bool(initial_counts) and old_delta >= 2 and new_delta >= 2,
                            "Missing request-counter evidence for both worker revisions during rollout",
                        )
                    check(
                        overlap,
                        f"No old/new Ready overlap observed for {role}; cannot claim rolling coverage",
                    )
                    # Keep phase traffic going after convergence to cover requests
                    # that were already in flight when the final old Pod exited.
                    await hold(phase)
                    stable = "after-" + role.lower()
                    phases.append(stable)
                    await hold(stable)
                await deployment._wait_for_ready(timeout=120)
                final = await stop_client()
                client_stopped = True
                check_requests(final, phases)
                report["requests"] = final
                report["status"] = "passed"
            finally:
                if not client_stopped:
                    await stop_client()
                    client_stopped = True

    finally:
        try:
            if client_created:
                logs = await kubectl(namespace, "logs", client_name)
                (output / "client.jsonl").write_text(logs)
        finally:
            (output / "report.json").write_text(json.dumps(report, indent=2))
            await kubectl(
                namespace,
                "delete",
                "pod",
                client_name,
                "--ignore-not-found",
                "--wait=true",
                "--timeout=120s",
            )
            await kubectl(
                namespace,
                "delete",
                "dynamographdeployment",
                name,
                "--ignore-not-found",
                "--cascade=foreground",
                "--wait=true",
                "--timeout=300s",
                timeout=330,
            )
            # Wait for GPU Pods to be gone before the next parameterized path.
            await kubectl(
                namespace,
                "delete",
                "pods",
                "-l",
                f"nvidia.com/dynamo-graph-deployment-name={name}",
                "--ignore-not-found",
                "--wait=true",
                "--timeout=180s",
                timeout=210,
            )
            await kubectl(
                namespace, "delete", "-f", str(resources), "--ignore-not-found"
            )
