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
TOLERATION = {"key": TAINT_KEY, "operator": "Equal", "value": "true", "effect": "NoSchedule"}
NODEGROUP = "global-router-bench-cpu"
AIPERF_VERSION = "0.13.0"
LOADGEN_IMAGE = (
    "python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
)


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
    return {"apiVersion": "v1", "kind": "List", "items": list(items.values())}


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
    return control.render(
        cfg["namespace"],
        member["controlImage"],
        routing_config(cfg),
        replicas=cfg["routerReplicas"],
        router_policy=policy,
        router_cpu=cfg["routerCpu"],
        node_selector=NODE_SELECTOR,
        tolerations=[TOLERATION],
    )


def aiperf_config(cfg, concurrency):
    """The AgentX harness shape: replay the public Claude Code traces at a fixed
    concurrency, with a warmup, then one window of settle + measure."""
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
                "duration": load["settleS"] + load["measureS"],
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
        "SETTLE_S": str(load["settleS"]),
        "MEASURE_S": str(load["measureS"]),
        "METRICS_URLS": ",".join(
            f"{pool['stem']}=http://{pool['stem']}-frontend:8000/metrics"
            for pool in cfg["pools"]
        ),
        "HF_HOME": "/work/hf",
        "PYTHONUNBUFFERED": "1",
    }
    meta = {"name": name, "namespace": cfg["namespace"]}
    return {
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
    """Samples per-container CPU on a node with crictl and prints, for each
    container in ``namespace``, its busiest interval in cores."""
    return f"""python3 - <<'PY'
import json, subprocess, time
def snap():
    stats = json.loads(subprocess.run(["crictl", "stats", "-o", "json"],
                                      capture_output=True, text=True, check=True).stdout)
    out = {{}}
    for entry in stats.get("stats", []):
        labels = entry["attributes"].get("labels", {{}})
        if labels.get("io.kubernetes.pod.namespace") != "{namespace}":
            continue
        key = labels.get("io.kubernetes.pod.name", "?") + "/" + labels.get("io.kubernetes.container.name", "?")
        cpu = entry.get("cpu") or {{}}
        out[key] = (int(cpu.get("timestamp", 0)), int((cpu.get("usageCoreNanoSeconds") or {{}}).get("value", 0)))
    return out
peak = {{}}
previous = snap()
for _ in range({samples}):
    time.sleep({interval_s})
    current = snap()
    for key, (stamp, used) in current.items():
        if key in previous and stamp > previous[key][0]:
            cores = (used - previous[key][1]) / (stamp - previous[key][0])
            peak[key] = max(peak.get(key, 0.0), round(cores, 2))
    previous = current
print("BENCH_CPU " + json.dumps(peak, sort_keys=True))
PY"""


def summary_script(namespace, label):
    return (
        f"grep -h 'BENCH_SUMMARY ' /var/log/pods/{namespace}_loadgen-{label}-*/loadgen/*.log"
        " | tail -n 1 | sed 's/^.*BENCH_SUMMARY //'"
    )


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
                "--policy", choices=["round-robin", "least-in-flight"], required=True
            )
        if name in ("loadgen", "collect"):
            sub.add_argument("--label", required=True)
        if name == "loadgen":
            sub.add_argument("action", choices=["up", "down"])
            sub.add_argument("--concurrency", type=int)
            sub.add_argument("--start-at", type=float, help="Shared UTC epoch seconds")
            sub.add_argument("--cpu-samples", type=int, default=40)
    commands.add_parser("hub")
    table = commands.add_parser("table")
    table.add_argument("--label", required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.command == "hub":
        members = [member["site"] for member in cfg["regions"]]
        stems = [pool["stem"] for pool in cfg["pools"]]
        print(json.dumps(mcs.render(cfg["namespace"], stems, members), indent=2))
        return
    if args.command == "table":
        directory = args.evidence / "results" / args.label
        results = {
            path.stem: json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))
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
            cluster.aws(
                member["region"],
                "cloudformation",
                "create-stack",
                "--stack-name",
                name,
                "--template-body",
                "file://" + str(path.resolve()),
            )
            print(f"submitted {name}")
        else:
            cluster.aws(member["region"], "cloudformation", "delete-stack", "--stack-name", name)
            print(f"deleting {name}")
    elif args.command == "backend":
        value = cluster.deploy(
            member, stack(member, "backend"), backend_manifest(cfg, member), args.evidence
        )
        print(json.dumps(value))
    elif args.command == "control":
        value = cluster.deploy(
            member,
            stack(member, "control"),
            control_manifest(cfg, member, args.policy),
            args.evidence,
        )
        print(json.dumps(value))
    elif args.command == "pods":
        pods = cluster.read(member, "pods", cfg["namespace"], "", ".items", args.evidence)
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
            cluster.aws(member["region"], "cloudformation", "delete-stack", "--stack-name", name)
            print(f"deleting {name}")
            return
        if args.concurrency is None or args.start_at is None:
            sys.exit("loadgen up needs --concurrency and --start-at")
        manifest = loadgen_manifest(cfg, args.label, args.concurrency, args.start_at)
        print(json.dumps(cluster.deploy(member, name, manifest, args.evidence)))
        # Sample container CPU on the bench nodes for the whole run.
        delay = max(0, int(args.start_at - time.time()))
        script = f"sleep {delay}\n" + cpu_sampler_script(
            cfg["namespace"], args.cpu_samples, 15
        )
        command_id = ssm_send(
            member, bench_nodes(member), script, delay + args.cpu_samples * 15 + 600
        )
        record = args.evidence / "results" / args.label / f"{member['region']}.cpu-command"
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
            text = ssm_output(member, command_id, instance)["StandardOutputContent"].strip()
            if text:
                summary = json.loads(text)
        if summary is None:
            sys.exit(f"no BENCH_SUMMARY for {args.label} in {member['region']}")
        cpu = {}
        cpu_command = (directory / f"{member['region']}.cpu-command").read_text().strip()
        for instance in instances:
            text = ssm_output(member, cpu_command, instance)["StandardOutputContent"]
            for line in text.splitlines():
                if line.startswith("BENCH_CPU "):
                    cpu.update(json.loads(line.removeprefix("BENCH_CPU ")))
        out = directory / f"{member['region']}.json"
        out.write_text(json.dumps({"summary": summary, "peak_cpu_cores": cpu}, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
