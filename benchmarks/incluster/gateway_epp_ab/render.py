#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fill in one run's settings for the gateway A/B benchmark (see README.md).

run.sh pipes ``kubectl create --dry-run=client -o json`` output through the
``arm`` and ``job`` commands, so the YAML files stay plain manifests that
kubectl can also apply as they are. ``targets`` lists the /metrics endpoints
of a deployed arm for bench.sh. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

ROUTER_TYPES = ("frontend", "epp")
ARM_HEADER = "X-Dynamo-AB-Arm"
COMPONENT_TYPE_LABEL = "nvidia.com/dynamo-component-type"
# Container ports that serve Prometheus text on /metrics: the Frontend's HTTP
# port (the sidecar's too), the EPP's metrics port and a worker's system port.
METRICS_PORTS = ("http", "metrics", "system")
# agentgateway serves its metrics here; its proxy pods also announce the port
# in the prometheus.io/port annotation, which takes precedence.
GATEWAY_METRICS_PORT = "15020"
GATEWAY_ROLE = "gateway"

Target = tuple[str, str, str, str]  # role, pod, container, URL


def read_objects(text: str) -> list[dict[str, Any]]:
    """The objects in kubectl JSON output: one object per manifest, back to back."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        obj, position = decoder.raw_decode(text, position)
        objects.extend(obj["items"] if obj.get("kind") == "List" else [obj])
    return objects


def set_model_cache(spec: dict[str, Any], claim: str) -> None:
    for volume in spec.get("volumes", []):
        if "persistentVolumeClaim" in volume:
            volume["persistentVolumeClaim"]["claimName"] = claim


def set_pull_secret(spec: dict[str, Any], secret: str) -> None:
    if secret:
        spec["imagePullSecrets"] = [{"name": secret}]


def render_arm(
    objects: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Set images, worker count, model cache, Gateway and arm header of one arm."""
    for obj in objects:
        if obj["kind"] == "DynamoGraphDeployment":
            for component in obj["spec"]["components"]:
                spec = component["podTemplate"]["spec"]
                if component["type"] in ROUTER_TYPES:
                    images = {"main": args.frontend_image}
                else:
                    component["replicas"] = args.replicas
                    images = {"main": args.worker_image}
                    if "frontendSidecar" in component:
                        images[component["frontendSidecar"]] = args.frontend_image
                for container in spec["containers"]:
                    container["image"] = images[container["name"]]
                set_model_cache(spec, args.model_cache_pvc)
                set_pull_secret(spec, args.image_pull_secret)
        elif obj["kind"] == "HTTPRoute":
            for parent in obj["spec"]["parentRefs"]:
                parent["name"] = args.gateway_name
                parent["namespace"] = args.gateway_namespace
            for rule in obj["spec"]["rules"]:
                for match in rule["matches"]:
                    for header in match.get("headers", []):
                        if header["name"] == ARM_HEADER:
                            header["value"] = args.arm_header
        else:
            raise SystemExit(f"unexpected {obj['kind']} in an arm manifest")
    return objects


def render_job(
    objects: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Name the aiperf Job and set its image, ConfigMap and model cache."""
    (job,) = objects
    job["metadata"]["name"] = args.name
    spec = job["spec"]["template"]["spec"]
    for container in spec["containers"]:
        container["image"] = args.image
    for volume in spec["volumes"]:
        if "configMap" in volume:
            volume["configMap"]["name"] = args.configmap
    set_model_cache(spec, args.model_cache_pvc)
    set_pull_secret(spec, args.image_pull_secret)
    return objects


def pod_host(pod: dict[str, Any]) -> str | None:
    """The address of a running pod, bracketed if IPv6; None for any other pod."""
    status = pod.get("status", {})
    ip = status.get("podIP")
    if status.get("phase") != "Running" or not ip:
        return None
    if pod["metadata"].get("deletionTimestamp"):
        return None
    return f"[{ip}]" if ":" in ip else ip


def metrics_targets(
    pods: list[dict[str, Any]], gateway_pods: list[dict[str, Any]]
) -> list[Target]:
    """Every /metrics endpoint of the arm's pods and the Gateway's proxy pods."""
    targets: list[Target] = []
    for pod in pods:
        host = pod_host(pod)
        if host is None:
            continue
        name = pod["metadata"]["name"]
        role = pod["metadata"].get("labels", {}).get(COMPONENT_TYPE_LABEL, "unknown")
        for container in pod["spec"]["containers"]:
            for port in container.get("ports", []):
                if port.get("name") in METRICS_PORTS:
                    url = f"http://{host}:{port['containerPort']}/metrics"
                    targets.append((role, name, container["name"], url))
    for pod in gateway_pods:
        host = pod_host(pod)
        if host is None:
            continue
        annotations = pod["metadata"].get("annotations", {})
        port = annotations.get("prometheus.io/port", GATEWAY_METRICS_PORT)
        path = annotations.get("prometheus.io/path", "/metrics")
        container = pod["spec"]["containers"][0]["name"]
        targets.append(
            (
                GATEWAY_ROLE,
                pod["metadata"]["name"],
                container,
                f"http://{host}:{port}{path}",
            )
        )
    return targets


def server_metrics_urls(targets: list[Target], sample_workers: int) -> list[str]:
    """What aiperf scrapes during each run: the router and a few workers.

    Scraping every worker from aiperf would load the client in proportion to
    the worker count; bench.sh snapshots all of them between runs instead.
    The Gateway is left to the snapshots too: aiperf (0.12 and 0.13) rejects
    agentgateway's whole endpoint because of its OpenMetrics info metric.
    """
    workers = sorted(
        {
            pod
            for role, pod, _, _ in targets
            if role not in (GATEWAY_ROLE, *ROUTER_TYPES)
        }
    )
    sampled = set(workers[:sample_workers])
    return [
        url for role, pod, _, url in targets if role in ROUTER_TYPES or pod in sampled
    ]


def write_targets(args: argparse.Namespace) -> None:
    with open(args.pods) as f:
        pods = json.load(f)["items"]
    with open(args.gateway_pods) as f:
        gateway_pods = json.load(f)["items"]
    targets = metrics_targets(pods, gateway_pods)
    roles = {role for role, _, _, _ in targets}
    for required in (GATEWAY_ROLE, args.router):
        if required not in roles:
            raise SystemExit(f"no running {required} pod with a metrics port found")
    with open(args.targets_out, "w") as f:
        f.writelines("\t".join(target) + "\n" for target in targets)
    urls = server_metrics_urls(targets, args.sample_workers)
    with open(args.server_metrics_out, "w") as f:
        f.writelines(url + "\n" for url in urls)
    print(
        f"{len(targets)} metrics endpoints; aiperf scrapes {len(urls)} of them",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    arm = commands.add_parser("arm", help="an arm's DGD and HTTPRoute, JSON on stdin")
    arm.add_argument("--replicas", type=int, required=True, help="mocker workers")
    arm.add_argument("--frontend-image", required=True)
    arm.add_argument("--worker-image", required=True)
    arm.add_argument("--model-cache-pvc", required=True)
    arm.add_argument("--image-pull-secret", default="")
    arm.add_argument("--gateway-name", required=True)
    arm.add_argument("--gateway-namespace", required=True)
    arm.add_argument("--arm-header", required=True, help=f"{ARM_HEADER} value")

    job = commands.add_parser("job", help="the aiperf Job, JSON on stdin")
    job.add_argument("--name", required=True)
    job.add_argument("--configmap", required=True)
    job.add_argument("--image", required=True)
    job.add_argument("--model-cache-pvc", required=True)
    job.add_argument("--image-pull-secret", default="")

    targets = commands.add_parser("targets", help="the /metrics endpoints of an arm")
    targets.add_argument("--pods", required=True, help="the DGD's pods, as JSON")
    targets.add_argument("--gateway-pods", required=True, help="the proxy's pods")
    targets.add_argument("--router", required=True, choices=ROUTER_TYPES)
    targets.add_argument("--sample-workers", type=int, default=4)
    targets.add_argument("--targets-out", required=True)
    targets.add_argument("--server-metrics-out", required=True)

    args = parser.parse_args(argv)
    if args.command == "targets":
        write_targets(args)
        return
    objects = read_objects(sys.stdin.read())
    render = render_arm if args.command == "arm" else render_job
    # kubectl applies or creates every item of a List.
    rendered = {"apiVersion": "v1", "kind": "List", "items": render(objects, args)}
    json.dump(rendered, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
