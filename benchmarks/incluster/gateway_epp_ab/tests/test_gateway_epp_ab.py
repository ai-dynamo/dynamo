# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two arms of the gateway A/B benchmark differ only in their routing path."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shlex
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


def run_script(tmp_path, args, settings=None, extra_env=None, kubectl_body="exit 1"):
    """run.sh with a kubectl that records its calls, and no other cluster.

    The kubectl runs kubectl_body, a shell snippet, which fails by default.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "kubectl-calls"
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(f"#!/bin/sh\necho \"$*\" >> '{calls}'\n{kubectl_body}\n")
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
        ({"WORKLOAD": "agentic"}, "WORKLOAD is 'synthetic' or 'agentx', not 'agentic'"),
        (
            {"WORKLOAD": "agentx", "AGENTX_CONCURRENCIES": "64,0"},
            "AGENTX_CONCURRENCIES must be positive integers, not '0'",
        ),
        (
            {"WORKLOAD": "agentx", "AGENTX_DURATION_S": "15m"},
            "AGENTX_DURATION_S must be a positive integer, not '15m'",
        ),
        (
            {"WORKLOAD": "agentx", "AGENTX_TRACES": "all"},
            "AGENTX_TRACES must be empty or a positive integer, not 'all'",
        ),
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


def test_run_agentx_needs_no_synthetic_load_points(tmp_path):
    settings = {"WORKLOAD": "agentx", "REQUEST_RATES": "", "CONCURRENCIES": ""}
    result, _ = run_script(tmp_path, ["deploy", "a"], settings)
    assert result.returncode == 1, result.stderr
    assert "cannot list DGDs" in result.stderr


@pytest.mark.parametrize(
    "image, warned",
    [
        (None, True),  # the 1.5.1 default
        ("registry.test/dynamo-frontend:1.5.0", True),
        ("registry.test/dynamo:main-dynamo-frontend-0123abcd", False),
    ],
)
def test_deploy_warns_about_images_without_the_latency_fixes(tmp_path, image, warned):
    settings = {"FRONTEND_IMAGE": image} if image else {}
    result, _ = run_script(tmp_path, ["deploy", "a"], settings)
    assert ("lacks the Frontend and EPP TCP_NODELAY fixes" in result.stderr) == warned


# A kubectl that passes preflight up to the model cache, and reports
# $GATEWAY_IMAGE as the image of the Gateway's pod.
PREFLIGHT_KUBECTL = """case "$*" in
  *"get dynamographdeployments,"*) exit 0 ;;
  *" get gateway "*) echo True ;;
  *"gateway.networking.k8s.io/gateway-name="*) echo "$GATEWAY_IMAGE" ;;
  *) exit 1 ;;
esac"""


@pytest.mark.parametrize(
    "image, warned",
    [
        ("cr.agentgateway.dev/agentgateway:v1.0.0", True),  # the old installer pin
        ("cr.agentgateway.dev/agentgateway:v1.1.0", True),
        ("cr.agentgateway.dev/agentgateway:v1.6.0", False),
        ("cr.agentgateway.dev/agentgateway:v1.10.0", False),
        ("registry.test/other-gateway:v1.0.0", False),
    ],
)
def test_preflight_warns_about_agentgateway_before_1_2(tmp_path, image, warned):
    result, _ = run_script(
        tmp_path,
        ["deploy", "b"],
        extra_env={"GATEWAY_IMAGE": image},
        kubectl_body=PREFLIGHT_KUBECTL,
    )
    # preflight stops at the missing model cache, after the Gateway checks.
    assert "PVC model-cache is missing" in result.stderr, result.stderr
    assert ("opens a new TLS connection to the EPP" in result.stderr) == warned


# The params.env that run.sh writes for an arm, with the default workload.
BENCH_PARAMS = {
    "ARM": "b-epp",
    "ARM_HEADER": "test-namespace/qwen-b",
    "MODEL": MODEL,
    "GATEWAY_URL": "http://gateway.test:80",
    "WORKLOAD": "synthetic",
    "REQUEST_RATES": "100,200,400",
    "CONCURRENCIES": "",
    "ROUNDS": "1",
    "ISL": "1024",
    "OSL": "128",
    "PREFIX_LENGTH": "512",
    "NUM_PREFIXES": "256",
    "DURATION_S": "60",
    "WARMUP_S": "10",
    "COOLDOWN_S": "0",
    "SEED": "42",
    "AGENTX_CONCURRENCIES": "512,2048,4096",
    "AGENTX_DURATION_S": "900",
    "AGENTX_DATASET": "semianalysis_cc_traces_weka_062126",
    "AGENTX_MAX_CONTEXT": "128000",
    "AGENTX_TRACES": "",
    "SMOKE_TIMEOUT_S": "1",
    "SERVER_METRICS_FORMATS": "json csv",
}


def run_bench(tmp_path, params):
    """bench.sh with an aiperf that records each call, and no cluster.

    The python3 on PATH runs one-liners and swallows the scripts bench.sh
    pipes to it: the smoke request and the metrics snapshots. Returns the
    result and the arm's results directory.
    """
    inputs, results, bin_dir = (
        tmp_path / "bench",
        tmp_path / "results",
        tmp_path / "bin",
    )
    for directory in (inputs, results, bin_dir):
        directory.mkdir()
    settings = {**BENCH_PARAMS, **params}
    (inputs / "params.env").write_text(
        "".join(f"{name}={shlex.quote(value)}\n" for name, value in settings.items())
    )
    (inputs / "targets.tsv").write_text("")
    (inputs / "server-metrics.txt").write_text("http://10.0.0.1:9090/metrics\n")
    # run.sh has already collected, so bench.sh exits as soon as it is done.
    (results / "COLLECTED").touch()
    python = shlex.quote(sys.executable)
    (bin_dir / "python3").write_text(
        f'#!/bin/sh\nif [ "$1" = -c ]; then exec {python} "$@"; fi\ncat > /dev/null\n'
    )
    (bin_dir / "aiperf").write_text(
        f"#!/bin/sh\nexec {python} - \"$@\" <<'EOF'\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "env = {k: v for k, v in os.environ.items() if k.startswith(('AIPERF_', 'HF_'))}\n"
        "out = args[args.index('--artifact-dir') + 1]\n"
        "json.dump({'args': args, 'env': env}, open(os.path.join(out, 'call.json'), 'w'))\n"
        "EOF\n"
    )
    for script in ("python3", "aiperf"):
        (bin_dir / script).chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "BENCH_INPUTS": str(inputs),
        "BENCH_RESULTS": str(results),
        # As in aiperf-job.yaml.
        "HF_HUB_OFFLINE": "1",
    }
    result = subprocess.run(
        ["bash", str(BENCH_DIR / "bench.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, results / settings["ARM"]


def aiperf_calls(out):
    """The aiperf call of each run, in the order bench.sh ran them."""
    runs = [line.split("\t")[0] for line in (out / "runs.tsv").read_text().splitlines()]
    return {run: json.loads((out / run / "call.json").read_text()) for run in runs}


def test_bench_sweeps_synthetic_prompts_by_rate_and_concurrency(tmp_path):
    result, out = run_bench(tmp_path, {"REQUEST_RATES": "4", "CONCURRENCIES": "8"})
    assert result.returncode == 0, result.stdout + result.stderr
    calls = aiperf_calls(out)
    assert list(calls) == ["rps4_r0", "c8_r0"]
    rate, closed = calls["rps4_r0"]["args"], calls["c8_r0"]["args"]
    assert arg_value(rate, "--request-rate") == "4"
    assert arg_value(rate, "--arrival-pattern") == "poisson"
    # 60 s and 10 s of arrivals at 4 requests per second.
    assert arg_value(rate, "--request-count") == "240"
    assert arg_value(rate, "--warmup-request-count") == "40"
    assert arg_value(rate, "--num-dataset-entries") == "280"
    # The 1024-token prompt is a 512-token shared prefix and 512 tokens of its own.
    assert arg_value(rate, "--synthetic-input-tokens-mean") == "512"
    assert arg_value(rate, "--prefix-prompt-length") == "512"
    assert arg_value(closed, "--concurrency") == "8"
    assert arg_value(closed, "--request-count") == "500"
    assert arg_value(closed, "--warmup-request-count") == "16"
    for call in calls.values():
        assert "--scenario" not in call["args"]
        # The tokenizer comes from the model cache, offline.
        assert call["env"] == {"HF_HUB_OFFLINE": "1"}


def test_bench_replays_agentx_sessions_through_the_scenario(tmp_path):
    params = {
        "WORKLOAD": "agentx",
        "AGENTX_CONCURRENCIES": "4,3000",
        "AGENTX_DURATION_S": "60",
        "AGENTX_TRACES": "2",
        "ROUNDS": "2",
    }
    result, out = run_bench(tmp_path, params)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = aiperf_calls(out)
    assert list(calls) == ["agentx4_r0", "agentx3000_r0", "agentx4_r1", "agentx3000_r1"]
    for (run, call), seed in zip(calls.items(), ("42", "43", "44", "45")):
        args = call["args"]
        assert arg_value(args, "--scenario") == "inferencex-agentx-mvp"
        assert (
            arg_value(args, "--public-dataset") == "semianalysis_cc_traces_weka_062126"
        )
        assert arg_value(args, "--max-context-length") == "128000"
        assert arg_value(args, "--num-dataset-entries") == "2"
        assert arg_value(args, "--concurrency") == run.split("_")[0][len("agentx") :]
        assert arg_value(args, "--benchmark-duration") == "60"
        assert arg_value(args, "--random-seed") == seed
        # 60 s is below the scenario's minimum of 900 s.
        assert "--unsafe-override" in args
        # The scenario sets the load and the lengths itself.
        for flag in (
            "--request-rate",
            "--request-count",
            "--warmup-request-count",
            "--synthetic-input-tokens-mean",
        ):
            assert flag not in args
        # aiperf takes any value of HF_HUB_OFFLINE as offline, so it is unset.
        assert call["env"] == {
            "HF_HOME": "/hf-cache",
            # Twice the largest point, for the requests of subagents.
            "AIPERF_HTTP_CONNECTION_LIMIT": "6000",
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
        }
    command = (out / "agentx4_r0" / "command.txt").read_text()
    assert command.startswith("env -u HF_HUB_OFFLINE HF_HOME=/hf-cache ")


def test_agentx_hugging_face_home_is_the_jobs_scratch_volume():
    # aiperf runs as a user that cannot write to the model cache, so the
    # corpus goes to an emptyDir that the Job mounts where bench.sh points.
    (job,) = load_docs("aiperf-job.yaml")
    spec = job["spec"]["template"]["spec"]
    volumes = {v["name"]: v for v in spec["volumes"]}
    assert volumes["hf-cache"] == {"name": "hf-cache", "emptyDir": {}}
    (container,) = spec["containers"]
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert f"HF_HOME={mounts['hf-cache']}\n" in (BENCH_DIR / "bench.sh").read_text()


def test_bench_runs_full_length_agentx_without_overrides(tmp_path):
    params = {"WORKLOAD": "agentx", "AGENTX_CONCURRENCIES": "8"}
    result, out = run_bench(tmp_path, params)
    assert result.returncode == 0, result.stdout + result.stderr
    call = aiperf_calls(out)["agentx8_r0"]
    assert arg_value(call["args"], "--benchmark-duration") == "900"
    assert "--unsafe-override" not in call["args"]
    assert "--num-dataset-entries" not in call["args"]
    assert call["env"]["AIPERF_HTTP_CONNECTION_LIMIT"] == "2500"


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
