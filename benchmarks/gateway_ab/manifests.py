# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes manifests for the Frontend vs Gateway+EPP A/B benchmark.

Every deployment is built by the same function from the same ``BenchConfig``,
so the mocker workers are identical by construction and the only difference
is the routing path. Both sides go through the same Gateway:

* ``frontend`` (A): client -> Gateway -> Dynamo Frontend (KV router) -> mocker
  workers, through a plain HTTPRoute to the Frontend's Service.
* ``epp`` (B): client -> Gateway -> EPP (KV router, Dynamo runtime mode)
  -> Frontend sidecar in direct mode inside the chosen worker pod -> worker.
* ``direct`` (optional): a copy of A's deployment that the client reaches
  without the Gateway, to measure what the Gateway itself costs.

Manifests are plain dicts so they can be applied as JSON with no YAML
dependency.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

TOPOLOGIES = ("agg", "disagg")
# A and B, both behind the Gateway.
SIDES = ("frontend", "epp")
# With the direct copy of A. A sits in the middle, so the measuring order,
# which reverses every other point, keeps A next to both arms it is
# compared with.
ARMS = ("direct", "frontend", "epp")
# Suffix of each arm's deployment name.
TAGS = {"frontend": "fe", "epp": "epp", "direct": "dir"}

# Route header that selects one deployment's route on the shared Gateway.
# Every route serves the same model there, so the model-name header alone
# cannot tell them apart.
TARGET_HEADER = "X-Dynamo-AB-Target"
MODEL_HEADER = "X-Gateway-Model-Name"
# The one endpoint the benchmark calls. Routes match it exactly, and both
# headers above, so a broader route on a shared Gateway cannot win the match
# and take the traffic.
CHAT_PATH = "/v1/chat/completions"

PART_OF = "app.kubernetes.io/part-of"
# Every object also carries its run's --name-prefix, so teardown of one run
# leaves another run's objects in the same namespace alone.
INSTANCE = "app.kubernetes.io/instance"

SIDECAR_NAME = "sidecar-frontend"
LOADGEN_LABEL = "gateway-ab-loadgen"
# Annotation holding a hash of the loadgen pod spec, so a changed image or
# secret recreates the pod instead of silently reusing the old one.
LOADGEN_SPEC_HASH = "gateway-ab/spec-hash"

# PID 1 of the loadgen pod. Every aiperf run is detached and ends up
# reparented to PID 1, so it must reap; `sleep infinity` does not, and each
# run would leave zombies behind.
REAPER = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGTERM, lambda *_: os._exit(0))\n"
    "while True:\n"
    "    try:\n"
    "        os.waitpid(-1, 0)\n"
    "    except ChildProcessError:\n"
    "        time.sleep(1)\n"
)


@dataclasses.dataclass
class BenchConfig:
    """Everything that shapes the deployments. Workload knobs live in loadgen.py."""

    model: str = "Qwen/Qwen3-0.6B"
    dynamo_version: str = "1.5.1"
    frontend_image: str = ""
    worker_image: str = ""
    name_prefix: str = "ab"
    # Qualifies the route header, since runs in two namespaces can share a
    # Gateway and a name prefix.
    namespace: str = "gateway-ab"

    agg_workers: int = 2
    prefill_workers: int = 2
    decode_workers: int = 2

    # The mocker's KV block size. The Frontend and the Dynamo-mode EPP both read
    # it from the worker's model card, so the two routers always agree.
    block_size: int = 16
    speedup_ratio: float = 1.0
    mocker_extra_args: tuple[str, ...] = ()

    # CPU requests only. CPU limits are left unset on purpose: CFS throttling
    # adds latency spikes that would dominate a latency comparison.
    router_cpu: str = "2"
    worker_cpu: str = "1"
    sidecar_cpu: str = "1"
    loadgen_cpu: str = "4"

    hf_secret: str | None = None
    image_pull_secret: str | None = None

    gateway_name: str = "inference-gateway"
    gateway_namespace: str = "agentgateway-system"

    # Also deploy the direct arm. It is a copy of A rather than A itself:
    # both replay the same prompts, and whichever ran second would find them
    # in A's prefix caches.
    with_direct: bool = False

    def __post_init__(self) -> None:
        if not self.frontend_image:
            self.frontend_image = (
                f"nvcr.io/nvidia/ai-dynamo/dynamo-frontend:{self.dynamo_version}"
            )
        if not self.worker_image:
            # No dedicated mocker image exists; the planner image carries both
            # dynamo.mocker and dynamo.frontend.
            self.worker_image = (
                f"nvcr.io/nvidia/ai-dynamo/dynamo-planner:{self.dynamo_version}"
            )
        for field in ("agg_workers", "prefill_workers", "decode_workers"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be at least 1")


def arms(cfg: BenchConfig) -> tuple[str, ...]:
    """The deployments of each topology, in measuring order."""
    return ARMS if cfg.with_direct else SIDES


def dgd_name(cfg: BenchConfig, topology: str, side: str) -> str:
    _check(topology, side)
    return f"{cfg.name_prefix}-{topology}-{TAGS[side]}"


def frontend_service(cfg: BenchConfig, topology: str, side: str = "frontend") -> str:
    """Service the operator creates for the Frontend of A or the direct arm."""
    return f"{dgd_name(cfg, topology, side)}-frontend"


def route_name(cfg: BenchConfig, topology: str, side: str) -> str:
    return f"{dgd_name(cfg, topology, side)}-route"


def loadgen_pod_name(cfg: BenchConfig) -> str:
    return f"{cfg.name_prefix}-loadgen"


def labels(cfg: BenchConfig) -> dict[str, str]:
    """Labels on every object of one run; teardown selects by them."""
    return {PART_OF: "gateway-ab", INSTANCE: cfg.name_prefix}


def request_headers(cfg: BenchConfig, topology: str, side: str) -> dict[str, str]:
    """Headers sent on every request to one arm.

    The target header selects the arm's route on the Gateway, so it is the
    only header that differs between A and B. The direct arm reaches its
    Frontend without the Gateway and ignores both.
    """
    return {
        MODEL_HEADER: cfg.model,
        TARGET_HEADER: f"{cfg.namespace}/{dgd_name(cfg, topology, side)}",
    }


def build_dgd(cfg: BenchConfig, topology: str, side: str) -> dict[str, Any]:
    _check(topology, side)
    components: list[dict[str, Any]] = [_router_component(cfg, side)]
    for ctype, replicas, disagg_mode in worker_roles(cfg, topology):
        components.append(_worker_component(cfg, side, ctype, replicas, disagg_mode))
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": dgd_name(cfg, topology, side),
            "labels": {
                **labels(cfg),
                "gateway-ab/topology": topology,
                "gateway-ab/side": side,
            },
        },
        "spec": {"components": components},
    }


def build_httproute(cfg: BenchConfig, topology: str, side: str) -> dict[str, Any]:
    """Route from the shared Gateway to B's InferencePool or to A's Frontend.

    The two routes differ only in their backend and the target header value.
    They match every header the arm's requests carry, so they outrank a route
    that matches fewer.
    """
    if side not in SIDES:
        raise ValueError(f"only {SIDES} go through the Gateway, not {side!r}")
    name = dgd_name(cfg, topology, side)
    if side == "epp":
        backend = {
            "group": "inference.networking.k8s.io",
            "kind": "InferencePool",
            # The operator names the pool after the DGD.
            "name": f"{name}-pool",
            "port": 8000,
        }
    else:
        backend = {
            "group": "",
            "kind": "Service",
            "name": frontend_service(cfg, topology),
            "port": 8000,
        }
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": route_name(cfg, topology, side),
            "labels": labels(cfg),
        },
        "spec": {
            "parentRefs": [
                {
                    "group": "gateway.networking.k8s.io",
                    "kind": "Gateway",
                    "name": cfg.gateway_name,
                    "namespace": cfg.gateway_namespace,
                }
            ],
            "rules": [
                {
                    "matches": [
                        {
                            "headers": [
                                {"name": header, "type": "Exact", "value": value}
                                for header, value in request_headers(
                                    cfg, topology, side
                                ).items()
                            ],
                            "path": {"type": "Exact", "value": CHAT_PATH},
                        }
                    ],
                    "backendRefs": [backend],
                    "timeouts": {"request": "300s"},
                }
            ],
        },
    }


def build_loadgen_pod(cfg: BenchConfig) -> dict[str, Any]:
    """Long-lived pod that runs aiperf against every arm.

    One pod for the whole experiment: every arm is measured from the same
    client, on the same node, with the same aiperf and tokenizer cache.
    """
    container: dict[str, Any] = {
        "name": "loadgen",
        "image": cfg.frontend_image,
        "command": ["python3", "-c", REAPER],
        "env": [
            {"name": "HF_HOME", "value": "/results/.hf"},
            # aiperf prints a rich UI otherwise; keep the logs plain.
            {"name": "TERM", "value": "dumb"},
        ],
        "resources": {
            "requests": {"cpu": cfg.loadgen_cpu, "memory": "4Gi"},
        },
        "volumeMounts": [{"name": "results", "mountPath": "/results"}],
    }
    _add_hf_secret(cfg, container)
    spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [{"name": "results", "emptyDir": {}}],
        "terminationGracePeriodSeconds": 5,
    }
    _add_pull_secret(cfg, spec)
    pod_labels = {**labels(cfg), "app": LOADGEN_LABEL}
    # The labels are hashed too: a pod without them would escape teardown.
    spec_hash = hashlib.sha256(
        json.dumps({"spec": spec, "labels": pod_labels}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": loadgen_pod_name(cfg),
            "labels": pod_labels,
            "annotations": {LOADGEN_SPEC_HASH: spec_hash},
        },
        "spec": spec,
    }


def build_all(cfg: BenchConfig, topologies: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every manifest for the given topologies, loadgen pod first."""
    out: list[dict[str, Any]] = [build_loadgen_pod(cfg)]
    for topology in topologies:
        out += [build_dgd(cfg, topology, side) for side in arms(cfg)]
        out += [build_httproute(cfg, topology, side) for side in SIDES]
    return out


def _router_component(cfg: BenchConfig, side: str) -> dict[str, Any]:
    if side != "epp":
        container: dict[str, Any] = {
            "name": "main",
            "image": cfg.frontend_image,
            "env": [{"name": "DYN_ROUTER_MODE", "value": "kv"}],
            "resources": {"requests": {"cpu": cfg.router_cpu, "memory": "2Gi"}},
        }
        name, ctype = "Frontend", "frontend"
    else:
        # In Dynamo runtime mode the EPP learns the model and the block size
        # from the workers' model cards, as the Frontend does.
        container = {
            "name": "main",
            "image": cfg.frontend_image,
            "resources": {"requests": {"cpu": cfg.router_cpu, "memory": "2Gi"}},
        }
        name, ctype = "Epp", "epp"
    _add_hf_secret(cfg, container)
    return {
        "name": name,
        "type": ctype,
        "replicas": 1,
        "podTemplate": {"spec": _pod_spec(cfg, [container])},
    }


def mocker_args(cfg: BenchConfig, disagg_mode: str | None) -> list[str]:
    """Arguments for one mocker worker; shared by every backend and side."""
    args = [
        "--model-path",
        cfg.model,
        "--model-name",
        cfg.model,
        "--block-size",
        str(cfg.block_size),
        "--speedup-ratio",
        str(cfg.speedup_ratio),
        "--enable-prefix-caching",
    ]
    if disagg_mode:
        args += ["--disaggregation-mode", disagg_mode]
    return args + list(cfg.mocker_extra_args)


def worker_roles(cfg: BenchConfig, topology: str) -> list[tuple[str, int, str | None]]:
    """(component type, replicas, disaggregation mode) for each worker group."""
    _check(topology, "frontend")
    if topology == "agg":
        return [("decode", cfg.agg_workers, None)]
    return [
        ("prefill", cfg.prefill_workers, "prefill"),
        ("decode", cfg.decode_workers, "decode"),
    ]


def _worker_component(
    cfg: BenchConfig,
    side: str,
    ctype: str,
    replicas: int,
    disagg_mode: str | None,
) -> dict[str, Any]:
    worker: dict[str, Any] = {
        "name": "main",
        "image": cfg.worker_image,
        "command": ["python3", "-m", "dynamo.mocker"],
        "args": mocker_args(cfg, disagg_mode),
        "resources": {"requests": {"cpu": cfg.worker_cpu, "memory": "2Gi"}},
    }
    _add_hf_secret(cfg, worker)
    containers = [worker]
    component: dict[str, Any] = {
        "name": ctype,
        "type": ctype,
        "replicas": replicas,
    }
    if side == "epp":
        # The EPP picks the worker before the request reaches the pod, so the
        # sidecar forwards without a second selection. It runs the same
        # Frontend image as the A side so the HTTP and tokenizer code match.
        sidecar: dict[str, Any] = {
            "name": SIDECAR_NAME,
            "image": cfg.frontend_image,
            "command": ["python3"],
            "args": ["-m", "dynamo.frontend", "--router-mode", "direct"],
            "resources": {"requests": {"cpu": cfg.sidecar_cpu, "memory": "2Gi"}},
        }
        _add_hf_secret(cfg, sidecar)
        containers.append(sidecar)
        component["frontendSidecar"] = SIDECAR_NAME
    component["podTemplate"] = {"spec": _pod_spec(cfg, containers)}
    return component


def _pod_spec(cfg: BenchConfig, containers: list[dict[str, Any]]) -> dict[str, Any]:
    spec: dict[str, Any] = {"containers": containers}
    _add_pull_secret(cfg, spec)
    return spec


def _add_hf_secret(cfg: BenchConfig, container: dict[str, Any]) -> None:
    if cfg.hf_secret:
        container["envFrom"] = [{"secretRef": {"name": cfg.hf_secret}}]


def _add_pull_secret(cfg: BenchConfig, spec: dict[str, Any]) -> None:
    if cfg.image_pull_secret:
        spec["imagePullSecrets"] = [{"name": cfg.image_pull_secret}]


def _check(topology: str, side: str) -> None:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; expected one of {TOPOLOGIES}")
    if side not in ARMS:
        raise ValueError(f"unknown side {side!r}; expected one of {ARMS}")
