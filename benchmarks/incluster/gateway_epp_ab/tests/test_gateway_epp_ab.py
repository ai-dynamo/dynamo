# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two arms of the gateway A/B benchmark differ only in their routing path."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

BENCH_DIR = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3-0.6B"
WORKERS = 2048


def load_render():
    spec = importlib.util.spec_from_file_location(
        "gateway_epp_ab_render", BENCH_DIR / "render.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render = load_render()


def load_docs(name: str) -> list[dict[str, Any]]:
    with open(BENCH_DIR / name) as f:
        return [doc for doc in yaml.safe_load_all(f) if doc]


def by_kind(docs: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    (doc,) = [doc for doc in docs if doc["kind"] == kind]
    return doc


def component(dgd: dict[str, Any], component_type: str) -> dict[str, Any]:
    (found,) = [c for c in dgd["spec"]["components"] if c["type"] == component_type]
    return found


def containers(comp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in comp["podTemplate"]["spec"]["containers"]}


def arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


@pytest.fixture(scope="module")
def arm_a() -> list[dict[str, Any]]:
    return load_docs("a-frontend.yaml")


@pytest.fixture(scope="module")
def arm_b() -> list[dict[str, Any]]:
    return load_docs("b-epp.yaml")


def test_workers_differ_only_by_the_frontend_sidecar(arm_a, arm_b):
    worker_a = component(by_kind(arm_a, "DynamoGraphDeployment"), "decode")
    worker_b = copy.deepcopy(
        component(by_kind(arm_b, "DynamoGraphDeployment"), "decode")
    )
    sidecar = worker_b.pop("frontendSidecar")
    spec = worker_b["podTemplate"]["spec"]
    spec["containers"] = [c for c in spec["containers"] if c["name"] != sidecar]
    assert worker_a == worker_b


def test_routers_get_the_same_image_resources_and_settings(arm_a, arm_b):
    frontend = component(by_kind(arm_a, "DynamoGraphDeployment"), "frontend")
    epp = component(by_kind(arm_b, "DynamoGraphDeployment"), "epp")
    assert frontend["replicas"] == epp["replicas"] == 1
    assert (
        frontend["podTemplate"]["spec"]["volumes"]
        == epp["podTemplate"]["spec"]["volumes"]
    )
    a, b = containers(frontend)["main"], containers(epp)["main"]
    for field in ("image", "resources", "volumeMounts"):
        assert a[field] == b[field], field
    # The EPP always runs the KV router; the Frontend is told to. Any other
    # setting, such as router tuning, must be set on both.
    assert {"name": "DYN_ROUTER_MODE", "value": "kv"} in a["env"]
    assert [e for e in a["env"] if e["name"] != "DYN_ROUTER_MODE"] == b["env"]


def test_sidecar_runs_the_frontend_image_in_direct_mode(arm_a, arm_b):
    frontend = component(by_kind(arm_a, "DynamoGraphDeployment"), "frontend")
    worker = component(by_kind(arm_b, "DynamoGraphDeployment"), "decode")
    sidecar = containers(worker)[worker["frontendSidecar"]]
    assert sidecar["image"] == containers(frontend)["main"]["image"]
    assert sidecar["args"] == ["-m", "dynamo.frontend", "--router-mode", "direct"]


@pytest.mark.parametrize("arm", ["a-frontend.yaml", "b-epp.yaml"])
def test_each_arm_runs_2048_qwen_mockers(arm):
    worker = component(by_kind(load_docs(arm), "DynamoGraphDeployment"), "decode")
    assert worker["replicas"] == WORKERS
    main = containers(worker)["main"]
    assert main["command"] == ["python3", "-m", "dynamo.mocker"]
    assert arg_value(main["args"], "--model-path") == MODEL
    assert arg_value(main["args"], "--model-name") == MODEL


def test_routes_differ_only_in_backend_and_arm_header(arm_a, arm_b):
    route_a = copy.deepcopy(by_kind(arm_a, "HTTPRoute"))
    route_b = copy.deepcopy(by_kind(arm_b, "HTTPRoute"))
    (rule_a,) = route_a["spec"]["rules"]
    (rule_b,) = route_b["spec"]["rules"]
    assert rule_a.pop("backendRefs") == [
        {"group": "", "kind": "Service", "name": "qwen-a-frontend", "port": 8000}
    ]
    assert rule_b.pop("backendRefs") == [
        {
            "group": "inference.networking.k8s.io",
            "kind": "InferencePool",
            "name": "qwen-b-pool",
            "port": 8000,
        }
    ]
    headers = {}
    for name, route in (("qwen-a", route_a), ("qwen-b", route_b)):
        assert route["metadata"]["name"] == name
        (match,) = route["spec"]["rules"][0]["matches"]
        headers[name] = {h["name"]: h["value"] for h in match["headers"]}
        assert headers[name] == {
            "X-Gateway-Model-Name": MODEL,
            render.ARM_HEADER: name,
        }
        route["metadata"]["name"] = "route"
        match["headers"] = []
    assert route_a == route_b


def test_read_objects_takes_kubectl_output_for_several_manifests(arm_b):
    # kubectl create --dry-run=client -o json prints one object per manifest,
    # back to back, rather than a List.
    text = "\n".join(json.dumps(obj, indent=2) for obj in arm_b) + "\n"
    assert render.read_objects(text) == arm_b
    wrapped = json.dumps({"apiVersion": "v1", "kind": "List", "items": arm_b})
    assert render.read_objects(wrapped) == arm_b


def test_render_arm_applies_the_run_settings(arm_b):
    args = argparse.Namespace(
        replicas=8,
        frontend_image="registry.test/frontend:x",
        worker_image="registry.test/planner:x",
        model_cache_pvc="shared-cache",
        image_pull_secret="pull-secret",
        gateway_name="gw",
        gateway_namespace="gw-ns",
        arm_header="bench-ns/qwen-b",
    )
    objects = render.render_arm(copy.deepcopy(arm_b), args)
    dgd = by_kind(objects, "DynamoGraphDeployment")
    worker = component(dgd, "decode")
    assert worker["replicas"] == 8
    images = {name: c["image"] for name, c in containers(worker).items()}
    assert images == {
        "main": "registry.test/planner:x",
        "sidecar-frontend": "registry.test/frontend:x",
    }
    epp = component(dgd, "epp")
    assert containers(epp)["main"]["image"] == "registry.test/frontend:x"
    assert epp["replicas"] == 1
    for comp in (worker, epp):
        spec = comp["podTemplate"]["spec"]
        assert spec["imagePullSecrets"] == [{"name": "pull-secret"}]
        assert (
            spec["volumes"][0]["persistentVolumeClaim"]["claimName"] == "shared-cache"
        )
    route = by_kind(objects, "HTTPRoute")
    (parent,) = route["spec"]["parentRefs"]
    assert (parent["name"], parent["namespace"]) == ("gw", "gw-ns")
    headers = route["spec"]["rules"][0]["matches"][0]["headers"]
    assert {
        "name": render.ARM_HEADER,
        "type": "Exact",
        "value": "bench-ns/qwen-b",
    } in headers


def test_render_job_names_the_job_and_its_inputs():
    (job,) = load_docs("aiperf-job.yaml")
    spec = job["spec"]["template"]["spec"]
    # Without a token, aiperf cannot discover and scrape every pod by itself.
    assert spec["automountServiceAccountToken"] is False
    args = argparse.Namespace(
        name="qwen-b-aiperf",
        configmap="qwen-b-bench",
        image="registry.test/frontend:x",
        model_cache_pvc="shared-cache",
        image_pull_secret="",
    )
    (rendered,) = render.render_job([copy.deepcopy(job)], args)
    spec = rendered["spec"]["template"]["spec"]
    assert rendered["metadata"]["name"] == "qwen-b-aiperf"
    assert spec["containers"][0]["image"] == "registry.test/frontend:x"
    volumes = {v["name"]: v for v in spec["volumes"]}
    assert volumes["bench"]["configMap"]["name"] == "qwen-b-bench"
    assert (
        volumes["model-cache"]["persistentVolumeClaim"]["claimName"] == "shared-cache"
    )
    assert "imagePullSecrets" not in spec


def pod(name, role, ip, ports, phase="Running", annotations=None):
    return {
        "metadata": {
            "name": name,
            "labels": {render.COMPONENT_TYPE_LABEL: role} if role else {},
            "annotations": annotations or {},
        },
        "spec": {
            "containers": [
                {
                    "name": container,
                    "ports": [{"name": n, "containerPort": p} for n, p in cp],
                }
                for container, cp in ports.items()
            ]
        },
        "status": {"phase": phase, "podIP": ip},
    }


def test_metrics_targets_cover_every_component_and_aiperf_samples_workers():
    pods = [
        pod(
            "qwen-b-epp-0",
            "epp",
            "10.0.0.1",
            {"main": [("grpc", 9002), ("grpc-health", 9003), ("metrics", 9090)]},
        ),
        *(
            pod(
                f"qwen-b-worker-{i}",
                "decode",
                f"10.0.1.{i}",
                {
                    "main": [("system", 9090), ("nixl", 19090)],
                    "sidecar-frontend": [("http", 8000)],
                },
            )
            for i in range(3)
        ),
        pod(
            "qwen-b-worker-pending",
            "decode",
            "",
            {"main": [("system", 9090)]},
            phase="Pending",
        ),
    ]
    gateway = [
        pod(
            "inference-gateway-0",
            None,
            "fd00::7",
            {"agentgateway": [("http", 80)]},
            annotations={"prometheus.io/port": "15020"},
        )
    ]
    targets = render.metrics_targets(pods, gateway)
    assert ("epp", "qwen-b-epp-0", "main", "http://10.0.0.1:9090/metrics") in targets
    assert (
        "decode",
        "qwen-b-worker-2",
        "sidecar-frontend",
        "http://10.0.1.2:8000/metrics",
    ) in targets
    assert (
        "decode",
        "qwen-b-worker-2",
        "main",
        "http://10.0.1.2:9090/metrics",
    ) in targets
    assert (
        "gateway",
        "inference-gateway-0",
        "agentgateway",
        "http://[fd00::7]:15020/metrics",
    ) in targets
    # Three running workers with two endpoints each, the EPP and the Gateway.
    assert len(targets) == 8

    # aiperf scrapes the router and the first worker pod; the Gateway only
    # reaches the snapshots.
    urls = render.server_metrics_urls(targets, sample_workers=1)
    assert urls == [
        "http://10.0.0.1:9090/metrics",
        "http://10.0.1.0:9090/metrics",
        "http://10.0.1.0:8000/metrics",
    ]


@pytest.mark.parametrize("script", ["run.sh", "bench.sh"])
def test_scripts_parse(script):
    subprocess.run(["bash", "-n", str(BENCH_DIR / script)], check=True)


def run_script(tmp_path, args, settings=None, extra_env=None):
    """run.sh with a kubectl that records its calls and fails, and no other cluster."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "kubectl-calls"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f"#!/bin/sh\necho \"$*\" >> '{calls}'\nexit 1\n")
    kubectl.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "KUBE_CONTEXT": "test-context",
        "NAMESPACE": "test-namespace",
        "RESULTS": str(tmp_path / "results"),
        **({"TMPDIR": os.environ["TMPDIR"]} if "TMPDIR" in os.environ else {}),
        **(settings or {}),
        **(extra_env or {}),
    }
    result = subprocess.run(
        ["bash", str(BENCH_DIR / "run.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, calls


@pytest.mark.parametrize(
    "settings, message",
    [
        (
            {"ISL": "512", "PREFIX_LENGTH": "512"},
            "PREFIX_LENGTH (512) must be shorter than ISL (512)",
        ),
        ({"REQUEST_RATES": "0"}, "REQUEST_RATES must be positive numbers, not '0'"),
        (
            {"REQUEST_RATES": "4,fast"},
            "REQUEST_RATES must be positive numbers, not 'fast'",
        ),
        (
            {"REQUEST_RATES": "", "CONCURRENCIES": ""},
            "set REQUEST_RATES or CONCURRENCIES",
        ),
        ({"CONCURRENCIES": "8,0"}, "CONCURRENCIES must be positive integers, not '0'"),
        ({"WORKERS": "0"}, "WORKERS must be a positive integer, not '0'"),
        ({"WARMUP_S": "-1"}, "WARMUP_S must be a whole number, not '-1'"),
        ({"KEEP": "yes"}, "KEEP must be 0 or 1, not 'yes'"),
    ],
)
def test_run_rejects_settings_that_cannot_work_before_touching_the_cluster(
    tmp_path, settings, message
):
    result, calls = run_script(tmp_path, ["all"], settings)
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert not calls.exists()


def test_run_accepts_the_default_settings_and_reaches_preflight(tmp_path):
    result, calls = run_script(tmp_path, ["deploy", "a"], {"REQUEST_RATES": "0.5,2"})
    assert result.returncode == 1, result.stderr
    assert "cannot list DGDs" in result.stderr
    assert (
        "--context test-context -n test-namespace get dynamographdeployments"
        in calls.read_text()
    )


def test_plot_finds_aiperf_behind_pips_sh_launcher(tmp_path):
    # pip writes a /bin/sh launcher instead of a #! line when the
    # interpreter's path is too long. A stub aiperf package stands in for aiperf.
    plot_package = tmp_path / "site" / "aiperf" / "plot"
    plot_package.mkdir(parents=True)
    (plot_package.parent / "__init__.py").write_text("")
    (plot_package / "__init__.py").write_text("")
    (plot_package / "default_plot_config.yaml").write_text("multi_run_plots: {}\n")
    record = tmp_path / "aiperf-call.json"
    launcher = tmp_path / "bin" / "aiperf"
    launcher.parent.mkdir()
    launcher.write_text(
        "#!/bin/sh\n"
        f"'''exec' {sys.executable} \"$0\" \"$@\"\n"
        "' '''\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "config = open(args[args.index('--config') + 1]).read()\n"
        f"json.dump({{'args': args, 'config': config}}, open('{record}', 'w'))\n"
    )
    launcher.chmod(0o755)
    results = tmp_path / "results"
    results.mkdir()
    result, calls = run_script(
        tmp_path,
        ["plot", str(results)],
        extra_env={"PYTHONPATH": str(tmp_path / "site")},
    )
    assert result.returncode == 0, result.stderr
    call = json.loads(record.read_text())
    assert call["args"][:2] == ["plot", str(results)]
    assert call["args"][-2:] == ["--output", str(results / "plots")]
    config = yaml.safe_load(call["config"])
    assert config["multi_run_plots"] == {}
    assert config["experiment_classification"] == {
        "baselines": ["a-frontend", "*/a-frontend/*"],
        "treatments": ["b-epp", "*/b-epp/*"],
        "default": "treatment",
    }
    assert not calls.exists()
