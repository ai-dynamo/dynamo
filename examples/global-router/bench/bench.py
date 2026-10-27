# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Global Router pool-selection policies on a multi-region Mocker fleet.

An operator-local JSON file (``--config``, kept outside Git) names each region's
cluster, kubectl provider, images, and the pool shapes. Every write is a
CloudFormation stack: a CPU node group per region, then backend, control, and
load-generator stacks that wrap Kubernetes objects through the cluster's
existing CDK kubectl provider (``rr-backend/cluster.py``).
"""

import argparse
import base64
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cluster = _load("rr_cluster", "rr-backend/cluster.py")
backend = _load("rr_backend", "rr-backend/manifests.py")
control = _load("rr_control", "rr-control-plane/manifests.py")
mcs = _load("rr_mcs", "rr-backend/mcs.py")

PURPOSE = "cpu-bench"
NODE_SELECTOR = {"dynamo.nvidia.com/purpose": PURPOSE}
TAINT_KEY = "dynamo.nvidia.com/bench"
TOLERATION = {
    "key": TAINT_KEY,
    "operator": "Equal",
    "value": "true",
    "effect": "NoSchedule",
}
NODEGROUP = "global-router-bench-cpu"
AIPERF_VERSION = "0.13.0"
LOADGEN_IMAGE = (
    "python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
)
SLOW_START_SYSCTL = "net.ipv4.tcp_slow_start_after_idle"
NODELAY_DIR = "/opt/nodelay"


def stack(member, suffix):
    return f"global-router-bench-{member['short']}-{suffix}"


# Node group -------------------------------------------------------------------------


def nodegroup_template(member, settings, gpu_nodegroup, launch_template_data):
    """A CPU managed node group beside ``gpu_nodegroup``, sharing its role, subnet and
    security groups so pods on it reach the same private peers.

    A launch template that names no security groups gets the cluster security
    group from EKS, so the copy names none either."""
    groups = launch_template_data.get("SecurityGroupIds") or [
        group
        for interface in launch_template_data.get("NetworkInterfaces", [])
        for group in interface.get("Groups", [])
    ]
    count = settings["count"]
    launch_template_data = {
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": settings["diskGiB"],
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
    }
    if groups:
        launch_template_data["SecurityGroupIds"] = groups
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            "LaunchTemplate": {
                "Type": "AWS::EC2::LaunchTemplate",
                "Properties": {"LaunchTemplateData": launch_template_data},
            },
            "Nodegroup": {
                "Type": "AWS::EKS::Nodegroup",
                "Properties": {
                    "ClusterName": member["clusterName"],
                    "NodegroupName": NODEGROUP,
                    "NodeRole": gpu_nodegroup["nodeRole"],
                    "Subnets": gpu_nodegroup["subnets"],
                    "InstanceTypes": [settings["instanceType"]],
                    "AmiType": "AL2023_x86_64_STANDARD",
                    "CapacityType": "ON_DEMAND",
                    "ScalingConfig": {
                        "MinSize": count,
                        "DesiredSize": count,
                        "MaxSize": count,
                    },
                    "Labels": NODE_SELECTOR,
                    "Taints": [
                        {"Key": TAINT_KEY, "Value": "true", "Effect": "NO_SCHEDULE"}
                    ],
                    "LaunchTemplate": {
                        "Id": {"Ref": "LaunchTemplate"},
                        "Version": {
                            "Fn::GetAtt": ["LaunchTemplate", "LatestVersionNumber"]
                        },
                    },
                },
            },
        },
    }


def render_nodegroup(cfg, member):
    region = member["region"]
    gpu = cluster.aws(
        region,
        "eks",
        "describe-nodegroup",
        "--cluster-name",
        member["clusterName"],
        "--nodegroup-name",
        member["gpuNodegroup"],
    )["nodegroup"]
    template = gpu["launchTemplate"]
    data = cluster.aws(
        region,
        "ec2",
        "describe-launch-template-versions",
        "--launch-template-id",
        template["id"],
        "--versions",
        str(template["version"]),
    )["LaunchTemplateVersions"][0]["LaunchTemplateData"]
    return nodegroup_template(member, cfg["nodegroup"], gpu, data)


# TCP settings -----------------------------------------------------------------------
# Every run measures with the TCP settings a tuned deployment has: Nagle off on each
# HTTP server in the request path, and no slow-start restart on any sender, so an
# idle connection keeps its send window. Without them a cross-region hop costs 2-4
# round trips instead of 1.


def nodelay_configmap(cfg, name):
    """The LD_PRELOAD accept shim (``nodelay.c``, built to ``cfg["nodelayShim"]``)."""
    shim = Path(cfg["nodelayShim"]).read_bytes()
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": cfg["namespace"]},
        "binaryData": {"libnodelay.so": base64.b64encode(shim).decode()},
    }


def nagle_off(pod, configmap):
    """Preloads the accept shim into every container of ``pod``: the frontend
    images here predate the frontend's own TCP_NODELAY (#16728)."""
    pod.setdefault("volumes", []).append(
        {"name": "nodelay", "configMap": {"name": configmap, "defaultMode": 0o755}}
    )
    for container in pod["containers"]:
        container.setdefault("env", []).append(
            {"name": "LD_PRELOAD", "value": f"{NODELAY_DIR}/libnodelay.so"}
        )
        container.setdefault("volumeMounts", []).append(
            {"name": "nodelay", "mountPath": NODELAY_DIR, "readOnly": True}
        )


def no_slow_start_after_idle(pod):
    """Sets the sysctl in the pod's own network namespace, which does not
    inherit the node's value. Kubelet allows no unsafe sysctls here, so a
    privileged init container writes it."""
    path = "/proc/sys/" + SLOW_START_SYSCTL.replace(".", "/")
    pod.setdefault("initContainers", []).append(
        {
            "name": "tcp-sysctl",
            "image": LOADGEN_IMAGE,
            "command": ["sh", "-c", f"echo 0 > {path}"],
            "securityContext": {
                "privileged": True,
                "runAsUser": 0,
                "runAsNonRoot": False,
            },
        }
    )


def tune_http_servers(manifest, cfg, configmap, names):
    """Both TCP settings for the Deployments in ``names``, with the shim's
    ConfigMap in the same manifest."""
    for item in manifest["items"]:
        if item["kind"] == "Deployment" and item["metadata"]["name"] in names:
            pod = item["spec"]["template"]["spec"]
            nagle_off(pod, configmap)
            no_slow_start_after_idle(pod)
    manifest["items"].append(nodelay_configmap(cfg, configmap))
    return manifest


# Kubernetes workloads ---------------------------------------------------------------


def backend_manifest(cfg, member):
    """Both pools of a region in one list, with their shared Namespace once."""
    items = {}
    for pool in cfg["pools"]:
        rendered = backend.render(
            cfg["namespace"],
            pool["stem"],
            member["site"],
            member["plannerImage"],
            member["relayImage"],
            cfg["model"],
            NODE_SELECTOR,
            worker_replicas=cfg["workerReplicas"],
            mocker_args=pool["mockerArgs"],
            tolerations=[TOLERATION],
            frontend_cpu=cfg["frontendCpu"],
        )
        for item in rendered["items"]:
            items[(item["kind"], item["metadata"]["name"])] = item
    return tune_http_servers(
        {"apiVersion": "v1", "kind": "List", "items": list(items.values())},
        cfg,
        "nodelay-backend",
        {f"{pool['stem']}-frontend" for pool in cfg["pools"]},
    )


def routing_config(cfg):
    """Every region's pools, for each region's Global Discovery Service."""
    return {
        "pools": [
            {
                "site_id": member["site"],
                "namespace": cfg["namespace"],
                "dgd_name": pool["stem"],
                "model": cfg["model"],
                "location": {"region": member["region"], "cluster": member["site"]},
            }
            for member in cfg["regions"]
            for pool in cfg["pools"]
        ]
    }


def control_manifest(cfg, member, policy):
    rendered = control.render(
        cfg["namespace"],
        member["controlImage"],
        routing_config(cfg),
        replicas=cfg["routerReplicas"],
        router_policy=policy,
        router_cpu=cfg["routerCpu"],
        node_selector=NODE_SELECTOR,
        tolerations=[TOLERATION],
    )
    return tune_http_servers(rendered, cfg, "nodelay-control", {"global-router"})


def aiperf_config(cfg, concurrency):
    """The AgentX harness shape: replay the public Claude Code traces at a fixed
    concurrency, with a warmup, then one profiling phase long enough to cover
    the shared measurement window in every region."""
    load = cfg["load"]
    return {
        "schemaVersion": "2.0",
        "randomSeed": 12345,
        "benchmark": {
            "model": cfg["model"],
            "endpoint": {
                "urls": ["http://global-router:8000"],
                "urlStrategy": "round_robin",
                "type": "chat",
                "streaming": True,
                "timeout": 21600.0,
                "useServerTokenCount": True,
                "extra": {"ignore_eos": True},
            },
            "dataset": {
                "type": "public",
                "dataset": load["dataset"],
                "entries": load["entries"],
                "sampling": "sequential",
                "ignoreTraceDelays": True,
            },
            "tokenizer": {"name": cfg["model"]},
            "warmup": {
                "type": "concurrency",
                "requests": load["warmupRequests"],
                "concurrency": concurrency,
            },
            "profiling": {
                "type": "concurrency",
                "duration": load["profilingS"],
                "concurrency": concurrency,
                "gracePeriod": load["graceS"],
            },
            "artifacts": {
                "dir": "/work/artifacts",
                "summary": ["json"],
                "records": ["jsonl"],
            },
            "gpuTelemetry": {"enabled": False},
            "serverMetrics": {"enabled": False},
            "runtime": {
                "workers": load["aiperfWorkers"],
                "recordProcessors": load["recordProcessors"],
                "ui": "none",
            },
        },
    }


def loadgen_manifest(cfg, label, concurrency, start_at):
    load = cfg["load"]
    name = f"loadgen-{label}"
    env = {
        "AIPERF_VERSION": AIPERF_VERSION,
        "DATASET_REPO": load["datasetRepo"],
        "TOKENIZER": cfg["model"],
        "START_AT": str(start_at),
        "WINDOW_OFFSET_S": str(load["windowOffsetS"]),
        "MEASURE_S": str(load["measureS"]),
        "METRICS_URLS": ",".join(
            f"{pool['stem']}=http://{pool['stem']}-frontend:8000/metrics"
            for pool in cfg["pools"]
        ),
        "HF_HOME": "/work/hf",
        "PYTHONUNBUFFERED": "1",
    }
    meta = {"name": name, "namespace": cfg["namespace"]}
    manifest = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": meta,
                "data": {
                    "aiperf.json": json.dumps(aiperf_config(cfg, concurrency)),
                    "run.py": (HERE / "loadgen_run.py").read_text(),
                },
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": meta,
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 3600,
                    "template": {
                        "metadata": {"labels": {"app": name}},
                        "spec": {
                            "restartPolicy": "Never",
                            "automountServiceAccountToken": False,
                            "nodeSelector": NODE_SELECTOR,
                            "tolerations": [TOLERATION],
                            "containers": [
                                {
                                    "name": "loadgen",
                                    "image": LOADGEN_IMAGE,
                                    "command": ["python3", "/bench/run.py"],
                                    "env": [
                                        {"name": key, "value": value}
                                        for key, value in env.items()
                                    ],
                                    "resources": {
                                        "requests": {
                                            "cpu": load["cpu"],
                                            "memory": load["memory"],
                                        },
                                        "limits": {
                                            "cpu": load["cpu"],
                                            "memory": load["memory"],
                                        },
                                    },
                                    "volumeMounts": [
                                        {"name": "bench", "mountPath": "/bench"},
                                        {"name": "work", "mountPath": "/work"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "bench", "configMap": {"name": name}},
                                {"name": "work", "emptyDir": {}},
                            ],
                        },
                    },
                },
            },
        ],
    }
    # aiperf uploads the prompts, so it is a sender too.
    no_slow_start_after_idle(manifest["items"][1]["spec"]["template"]["spec"])
    return manifest


# Node access over SSM ---------------------------------------------------------------


def bench_nodes(member):
    reservations = cluster.aws(
        member["region"],
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:eks:cluster-name,Values={member['clusterName']}",
        f"Name=tag:eks:nodegroup-name,Values={NODEGROUP}",
        "Name=instance-state-name,Values=running",
    )["Reservations"]
    return [
        instance["InstanceId"]
        for reservation in reservations
        for instance in reservation["Instances"]
    ]


def ssm_send(member, instances, script, timeout_s):
    return cluster.aws(
        member["region"],
        "ssm",
        "send-command",
        "--document-name",
        "AWS-RunShellScript",
        "--instance-ids",
        *instances,
        "--parameters",
        json.dumps({"commands": [script], "executionTimeout": [str(timeout_s)]}),
    )["Command"]["CommandId"]


def ssm_output(member, command_id, instance):
    while True:
        try:
            value = cluster.aws(
                member["region"],
                "ssm",
                "get-command-invocation",
                "--command-id",
                command_id,
                "--instance-id",
                instance,
            )
        except RuntimeError as error:
            if "InvocationDoesNotExist" not in str(error):
                raise
            value = {"Status": "Pending"}
        if value["Status"] not in ("Pending", "InProgress", "Delayed"):
            return value
        time.sleep(5)


def cpu_sampler_script(namespace, samples, interval_s):
    """Samples per-pod CPU on a node from cgroup v2 ``cpu.stat`` and prints,
    for each pod in ``namespace``, its busiest interval in cores. Pod names
    come from the kubelet's ``/var/log/pods/<namespace>_<pod>_<uid>`` dirs."""
    return f"""python3 - <<'PY'
import glob, json, os, time
def snap():
    out = {{}}
    for directory in glob.glob("/var/log/pods/{namespace}_*"):
        _, pod, uid = os.path.basename(directory).rsplit("_", 2)
        slices = glob.glob("/sys/fs/cgroup/kubepods.slice/**/kubepods*-pod%s.slice/cpu.stat"
                           % uid.replace("-", "_"), recursive=True)
        if not slices:
            continue
        with open(slices[0]) as stat:
            usage = int(dict(line.split() for line in stat)["usage_usec"])
        out[pod] = (time.monotonic(), usage)
    return out
peak = {{}}
previous = snap()
for _ in range({samples}):
    time.sleep({interval_s})
    current = snap()
    for pod, (stamp, used) in current.items():
        if pod in previous and stamp > previous[pod][0]:
            cores = (used - previous[pod][1]) / 1e6 / (stamp - previous[pod][0])
            peak[pod] = max(peak.get(pod, 0.0), round(cores, 2))
    previous = current
print("BENCH_CPU " + json.dumps(peak, sort_keys=True))
PY"""


def summary_script(namespace, label):
    return (
        f"grep -h 'BENCH_SUMMARY ' /var/log/pods/{namespace}_loadgen-{label}-*/loadgen/*.log"
        " | tail -n 1 | sed 's/^.*BENCH_SUMMARY //'"
    )


def http_probe_script(model, targets, sizes, requests, warmup, idle_s):
    """Times the first streamed token of chat requests from a node to each pool
    frontend in ``targets`` (name -> pod IP), for each prompt size in ``sizes``
    (name -> repetitions of a 27-character phrase), on three kinds of connection:
    ``hot`` (back to back on one connection), ``idle`` (the same connection after
    ``idle_s`` quiet, which shrinks Linux's congestion window) and ``cold`` (a
    new connection per request). The prompt repeats, so after ``warmup``
    requests every worker serves it from its prefix cache, and the difference
    between a remote and the local target is the cost of the path. The probe
    uploads the prompts from the node's own network namespace, so it turns off
    slow-start restart there for its run, as for every other sender."""
    return f"""previous=$(sysctl -n {SLOW_START_SYSCTL})
sysctl -qw {SLOW_START_SYSCTL}=0
trap 'sysctl -qw {SLOW_START_SYSCTL}='"$previous" EXIT
python3 - <<'PY'
import http.client, json, statistics, time
targets = {json.dumps(targets)}
sizes = {json.dumps(sizes)}
def body(repeats):
    text = "lorem ipsum dolor sit amet " * repeats
    return json.dumps({{"model": {json.dumps(model)}, "stream": True, "max_tokens": 4,
                       "ignore_eos": True, "stream_options": {{"include_usage": True}},
                       "messages": [{{"role": "user", "content": text}}]}}).encode()
def send(conn, payload):
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=payload,
                 headers={{"Content-Type": "application/json"}})
    response = conn.getresponse()
    first, tokens = None, None
    while True:
        line = response.readline()
        if not line:
            break
        if line.startswith(b"data:"):
            if first is None:
                first = time.perf_counter() - start
            if b"prompt_tokens" in line:
                tokens = json.loads(line[5:])["usage"]["prompt_tokens"]
    return first * 1000, tokens
out = {{}}
for name, ip in targets.items():
    for size, repeats in sizes.items():
        payload = body(repeats)
        conn = http.client.HTTPConnection(ip, 8000, timeout=300)
        tokens = None
        for _ in range({warmup}):
            _, tokens = send(conn, payload)
        result = {{"bytes": len(payload), "prompt_tokens": tokens}}
        hot = [send(conn, payload)[0] for _ in range({requests})]
        idle = []
        for _ in range({requests}):
            time.sleep({idle_s})
            idle.append(send(conn, payload)[0])
        conn.close()
        cold = []
        for _ in range({requests}):
            fresh = http.client.HTTPConnection(ip, 8000, timeout=300)
            cold.append(send(fresh, payload)[0])
            fresh.close()
        for mode, values in (("hot", hot), ("idle", idle), ("cold", cold)):
            result[mode + "_ttft_ms_p50"] = round(statistics.median(values), 1)
        out.setdefault(name, {{}})[size] = result
print("BENCH_HTTP " + json.dumps(out, sort_keys=True))
PY"""


def ping_script(targets, count):
    """Round-trip times from a node to each pod IP in ``targets`` (name -> IP):
    ICMP ping, and the TCP handshake to the frontend's port 8000, which is the
    path the Global Ingress uses and works where ICMP is filtered."""
    return f"""python3 - <<'PY'
import json, re, socket, statistics, subprocess, time
out = {{}}
for name, ip in {json.dumps(targets)}.items():
    ping = subprocess.run(["ping", "-c", "{count}", "-i", "0.2", "-q", ip],
                          capture_output=True, text=True)
    match = re.search(r"= ([0-9.]+)/([0-9.]+)/([0-9.]+)", ping.stdout)
    connect = []
    for _ in range({count}):
        start = time.perf_counter()
        with socket.create_connection((ip, 8000), timeout=5):
            connect.append((time.perf_counter() - start) * 1000)
    out[name] = {{
        "icmp_ms_min_avg_max": [float(v) for v in match.groups()] if match else None,
        "tcp_connect_ms_p50": round(statistics.median(connect), 2),
        "tcp_connect_ms_min": round(min(connect), 2),
    }}
print("BENCH_PING " + json.dumps(out, sort_keys=True))
PY"""


def tcp_check_script(namespace):
    """For each HTTP server on a node in ``namespace`` (pool frontends and the
    Global Ingress): whether the accept shim is loaded, whether a connection it
    accepts has TCP_NODELAY set, and its pod's slow-start sysctl. The check
    connects from inside the pod's network namespace and reads the accepted
    socket's option through ``pidfd_getfd``."""
    return f"""python3 - <<'PY'
import ctypes, glob, json, os, socket, subprocess, time
libc = ctypes.CDLL(None, use_errno=True)
CLONE_NEWNET, SYS_PIDFD_GETFD = 0x40000000, 438
SERVERS = (b"dynamo.frontend", b"global-ingress")
def cmdline(pid):
    try:
        return open("/proc/%d/cmdline" % pid, "rb").read()
    except OSError:
        return b""
def accepted_nodelay(pid):
    host = os.open("/proc/self/ns/net", os.O_RDONLY)
    libc.setns(os.open("/proc/%d/ns/net" % pid, os.O_RDONLY), CLONE_NEWNET)
    client = socket.create_connection(("127.0.0.1", 8000), timeout=5)
    libc.setns(host, CLONE_NEWNET)
    time.sleep(0.5)
    mine, value = client.getsockname(), None
    pidfd = os.pidfd_open(pid)
    for fd in os.listdir("/proc/%d/fd" % pid):
        copy = libc.syscall(SYS_PIDFD_GETFD, pidfd, int(fd), 0)
        if copy < 0:
            continue
        try:
            sock = socket.socket(fileno=copy)
        except OSError:
            os.close(copy)
            continue
        try:
            if sock.family == socket.AF_INET and sock.getpeername() == mine:
                value = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        except OSError:
            pass
        finally:
            sock.close()
    client.close()
    return value
out = []
for logdir in glob.glob("/var/log/pods/{namespace}_*"):
    pod, uid = os.path.basename(logdir).split("_")[1:3]
    procs = glob.glob("/sys/fs/cgroup/kubepods.slice/**/kubepods*-pod%s.slice/**/cgroup.procs"
                      % uid.replace("-", "_"), recursive=True)
    pids = [int(p) for f in procs for p in open(f).read().split()]
    pid = next((p for p in pids if any(s in cmdline(p) for s in SERVERS)), None)
    if pid is None:
        continue
    sysctl = subprocess.run(["nsenter", "-t", str(pid), "-n", "sysctl", "-n", "{SLOW_START_SYSCTL}"],
                            capture_output=True, text=True).stdout.strip()
    out.append(dict(pod=pod, shim_loaded="libnodelay.so" in open("/proc/%d/maps" % pid).read(),
                    accepted_nodelay=accepted_nodelay(pid), slow_start_after_idle=sysctl))
print("BENCH_TCP " + json.dumps(out, sort_keys=True))
PY"""


# Results ----------------------------------------------------------------------------


def combine(results):
    """One row per region plus a total, from each region's collected result."""
    rows = []
    totals = {"rps": 0.0, "tps": 0.0, "completed": 0, "errors": 0, "pools": {}}
    weighted = {"ttft_p50": 0.0, "ttft_p99": 0.0, "itl_p50": 0.0}
    for region, result in sorted(results.items()):
        client = result["summary"]["client"]
        rows.append(
            {
                "region": region,
                "rps": client["requests_per_second"],
                "tps": client["output_tokens_per_second"],
                "ttft_p50": client["ttft_ms"]["p50"],
                "ttft_p99": client["ttft_ms"]["p99"],
                "itl_p50": client["itl_ms"]["p50"],
                "completed": client["completed_requests"],
                "errors": client["profiling_errors"],
            }
        )
        totals["rps"] += client["requests_per_second"] or 0.0
        totals["tps"] += client["output_tokens_per_second"] or 0.0
        totals["completed"] += client["completed_requests"]
        totals["errors"] += client["profiling_errors"]
        for key in weighted:
            value = rows[-1][key]
            if value is not None:
                weighted[key] += value * client["completed_requests"]
        for stem, count in result["summary"]["pool_requests"].items():
            totals["pools"][f"{region}/{stem}"] = count
    completed = max(totals["completed"], 1)
    rows.append(
        {
            "region": "total",
            "rps": totals["rps"],
            "tps": totals["tps"],
            **{key: value / completed for key, value in weighted.items()},
            "completed": totals["completed"],
            "errors": totals["errors"],
        }
    )
    return rows, totals["pools"]


def frontend_ips(cfg, evidence):
    """Each region's first-pool frontend pod IP, by region short name."""
    pool = cfg["pools"][0]["stem"]
    targets = {}
    for member in cfg["regions"]:
        pods = cluster.read(member, "pods", cfg["namespace"], "", ".items", evidence)
        targets[member["short"]] = next(
            pod["status"]["podIP"]
            for pod in pods
            if pod["metadata"]["name"].startswith(f"{pool}-frontend-")
            and pod["status"].get("podIP")
        )
    return targets


def from_every_region(cfg, script, marker, timeout_s):
    """Runs ``script`` on one bench node per region at once and returns each
    region's JSON from the output line that starts with ``marker``."""
    sent = {
        member["short"]: (member, node, ssm_send(member, [node], script, timeout_s))
        for member in cfg["regions"]
        for node in bench_nodes(member)[:1]
    }
    matrix = {}
    for source, (member, node, command_id) in sent.items():
        result = ssm_output(member, command_id, node)
        line = next(
            (
                line
                for line in result["StandardOutputContent"].splitlines()
                if line.startswith(marker)
            ),
            None,
        )
        if line is None:
            sys.exit(f"{source}: {result['StandardErrorContent'][-800:]}")
        matrix[source] = json.loads(line.removeprefix(marker))
    return matrix


def from_every_node(cfg, script, marker, timeout_s):
    """Runs ``script`` on every bench node and returns, per region, the
    concatenated JSON lists from the output lines that start with ``marker``."""
    sent = [
        (member, nodes, ssm_send(member, nodes, script, timeout_s))
        for member in cfg["regions"]
        for nodes in [bench_nodes(member)]
    ]
    out = {}
    for member, nodes, command_id in sent:
        for node in nodes:
            result = ssm_output(member, command_id, node)
            lines = [
                line.removeprefix(marker)
                for line in result["StandardOutputContent"].splitlines()
                if line.startswith(marker)
            ]
            if not lines:
                sys.exit(
                    f"{member['short']}/{node}: {result['StandardErrorContent'][-800:]}"
                )
            out.setdefault(member["short"], []).extend(json.loads(lines[-1]))
    return out


def write_matrix(path, targets, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"targets": targets, "from": matrix}, indent=2))
    print(json.dumps(matrix, indent=2))


# Commands ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("nodegroup", "backend", "control", "pods", "loadgen", "collect"):
        sub = commands.add_parser(name)
        sub.add_argument("--region", required=True)
        if name == "nodegroup":
            sub.add_argument("action", choices=["render", "up", "down"])
        if name == "control":
            sub.add_argument(
                "--policy",
                choices=["none", "round-robin", "least-in-flight"],
                required=True,
                help="none passes no --router-policy, for images that predate it",
            )
        if name in ("loadgen", "collect"):
            sub.add_argument("--label", required=True)
        if name == "loadgen":
            sub.add_argument("action", choices=["up", "down"])
            sub.add_argument("--concurrency", type=int)
            sub.add_argument("--start-at", type=float, help="Shared UTC epoch seconds")
    commands.add_parser("hub")
    table = commands.add_parser("table")
    table.add_argument("--label", required=True)
    ping = commands.add_parser(
        "ping", help="round trip from each region to every region's pool frontend"
    )
    ping.add_argument("--count", type=int, default=20)
    probe = commands.add_parser(
        "probe",
        help="first-token time from each region to every region's pool frontend",
    )
    probe.add_argument("--requests", type=int, default=10)
    probe.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Per target and size; at least the pool's workers",
    )
    probe.add_argument("--idle-s", type=float, default=2.0)
    probe.add_argument(
        "--out",
        default="probe.json",
        help="File name under the evidence results directory",
    )
    commands.add_parser(
        "tcp-check", help="whether every HTTP server pod has the bench's TCP settings"
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.command == "tcp-check":
        servers = from_every_node(
            cfg, tcp_check_script(cfg["namespace"]), "BENCH_TCP ", 120
        )
        print(json.dumps(servers, indent=2))
        expected = len(cfg["pools"]) + cfg["routerReplicas"]
        for member in cfg["regions"]:
            pods = servers.get(member["short"], [])
            if len(pods) != expected or not all(
                pod["accepted_nodelay"] == 1 and pod["slow_start_after_idle"] == "0"
                for pod in pods
            ):
                sys.exit(f"{member['short']}: HTTP servers without the TCP settings")
        return
    if args.command == "ping":
        targets = frontend_ips(cfg, args.evidence)
        matrix = from_every_region(
            cfg, ping_script(targets, args.count), "BENCH_PING ", 600
        )
        write_matrix(args.evidence / "results" / "ping.json", targets, matrix)
        return
    if args.command == "probe":
        targets = frontend_ips(cfg, args.evidence)
        # The AgentX mean prompt in our runs was about 39k tokens; 128k is the tail.
        sizes = {"tiny": 1, "agentx_mean": 6000, "agentx_128k": 20000}
        script = http_probe_script(
            cfg["model"], targets, sizes, args.requests, args.warmup, args.idle_s
        )
        matrix = from_every_region(cfg, script, "BENCH_HTTP ", 3600)
        write_matrix(args.evidence / "results" / args.out, targets, matrix)
        return
    if args.command == "hub":
        members = [member["site"] for member in cfg["regions"]]
        stems = [pool["stem"] for pool in cfg["pools"]]
        print(json.dumps(mcs.render(cfg["namespace"], stems, members), indent=2))
        return
    if args.command == "table":
        directory = args.evidence / "results" / args.label
        results = {
            member["region"]: json.loads(
                (directory / f"{member['region']}.json").read_text()
            )
            for member in cfg["regions"]
            if (directory / f"{member['region']}.json").exists()
        }
        rows, pools = combine(results)
        print(json.dumps({"rows": rows, "pool_requests": pools}, indent=2))
        return

    member = next(m for m in cfg["regions"] if m["region"] == args.region)
    if args.command == "nodegroup":
        name = stack(member, "cpu")
        if args.action == "render":
            print(json.dumps(render_nodegroup(cfg, member), indent=2))
        elif args.action == "up":
            path = args.evidence / f"{member['region']}-{name}-template.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(render_nodegroup(cfg, member), indent=2))
            existing = cluster.aws(
                member["region"],
                "cloudformation",
                "list-stacks",
                "--query",
                "StackSummaries[?StackStatus!='DELETE_COMPLETE'].StackName",
            )
            cluster.aws(
                member["region"],
                "cloudformation",
                "update-stack" if name in existing else "create-stack",
                "--stack-name",
                name,
                "--template-body",
                "file://" + str(path.resolve()),
            )
            print(f"submitted {name}")
        else:
            cluster.aws(
                member["region"], "cloudformation", "delete-stack", "--stack-name", name
            )
            print(f"deleting {name}")
    elif args.command == "backend":
        value = cluster.deploy(
            member,
            stack(member, "backend"),
            backend_manifest(cfg, member),
            args.evidence,
        )
        print(json.dumps(value))
    elif args.command == "control":
        value = cluster.deploy(
            member,
            stack(member, "control"),
            control_manifest(
                cfg, member, None if args.policy == "none" else args.policy
            ),
            args.evidence,
        )
        print(json.dumps(value))
    elif args.command == "pods":
        pods = cluster.read(
            member, "pods", cfg["namespace"], "", ".items", args.evidence
        )
        for pod in pods or []:
            statuses = pod["status"].get("containerStatuses", [])
            ready = sum(status.get("ready", False) for status in statuses)
            restarts = sum(status.get("restartCount", 0) for status in statuses)
            print(
                f"{pod['metadata']['name']:48} {pod['status'].get('phase', '?'):10}"
                f" ready={ready}/{len(statuses)} restarts={restarts}"
                f" node={pod['spec'].get('nodeName', '-')}"
            )
    elif args.command == "loadgen":
        name = stack(member, f"loadgen-{args.label}")
        if args.action == "down":
            cluster.aws(
                member["region"], "cloudformation", "delete-stack", "--stack-name", name
            )
            print(f"deleting {name}")
            return
        if args.concurrency is None or args.start_at is None:
            sys.exit("loadgen up needs --concurrency and --start-at")
        manifest = loadgen_manifest(cfg, args.label, args.concurrency, args.start_at)
        print(json.dumps(cluster.deploy(member, name, manifest, args.evidence)))
        # Sample pod CPU on the bench nodes over the measurement window only:
        # aiperf's trace rebuild before it uses every core it has.
        load = cfg["load"]
        delay = max(0, int(args.start_at + load["windowOffsetS"] - time.time()))
        samples = load["measureS"] // 15
        script = f"sleep {delay}\n" + cpu_sampler_script(cfg["namespace"], samples, 15)
        command_id = ssm_send(
            member, bench_nodes(member), script, delay + samples * 15 + 600
        )
        record = (
            args.evidence / "results" / args.label / f"{member['region']}.cpu-command"
        )
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(command_id)
        print(f"cpu sampler {command_id}")
    elif args.command == "collect":
        directory = args.evidence / "results" / args.label
        instances = bench_nodes(member)
        command_id = ssm_send(
            member, instances, summary_script(cfg["namespace"], args.label), 120
        )
        summary = None
        for instance in instances:
            text = ssm_output(member, command_id, instance)[
                "StandardOutputContent"
            ].strip()
            if text:
                summary = json.loads(text)
        if summary is None:
            sys.exit(f"no BENCH_SUMMARY for {args.label} in {member['region']}")
        cpu = {}
        cpu_command = (
            (directory / f"{member['region']}.cpu-command").read_text().strip()
        )
        for instance in instances:
            text = ssm_output(member, cpu_command, instance)["StandardOutputContent"]
            for line in text.splitlines():
                if line.startswith("BENCH_CPU "):
                    cpu.update(json.loads(line.removeprefix("BENCH_CPU ")))
        out = directory / f"{member['region']}.json"
        out.write_text(
            json.dumps({"summary": summary, "peak_cpu_cores": cpu}, indent=2)
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
