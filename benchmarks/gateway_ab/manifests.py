# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes manifests for the Frontend vs Gateway+EPP A/B benchmark.

Both sides of a comparison are built by the same function from the same
``BenchConfig``, so the mocker workers are identical by construction and the
only difference is the routing path:

* ``frontend`` (A): client -> Dynamo Frontend (KV router) -> mocker workers.
* ``gateway`` (B): client -> Gateway -> EPP (KV router, Dynamo runtime mode)
  -> Frontend sidecar in direct mode inside the chosen worker pod -> worker.

Manifests are plain dicts so they can be applied as JSON with no YAML
dependency.
"""

from __future__ import annotations

import dataclasses
from typing import Any

TOPOLOGIES = ("agg", "disagg")
SIDES = ("frontend", "gateway")

# Route header that pins load to one B-side pool. Every pool shares the
# Gateway, so the model-name header alone cannot tell agg from disagg.
TARGET_HEADER = "X-Dynamo-AB-Target"
MODEL_HEADER = "X-Gateway-Model-Name"

SIDECAR_NAME = "sidecar-frontend"
LOADGEN_LABEL = "gateway-ab-loadgen"


@dataclasses.dataclass
class BenchConfig:
    """Everything that shapes the deployments. Workload knobs live in run.py."""

    model: str = "Qwen/Qwen3-0.6B"
    dynamo_version: str = "1.5.1"
    frontend_image: str = ""
    worker_image: str = ""
    name_prefix: str = "ab"

    agg_workers: int = 2
    prefill_workers: int = 2
    decode_workers: int = 2

    # Must match on both sides: the EPP reads DYN_KV_CACHE_BLOCK_SIZE while the
    # Frontend learns the block size from the worker's model card. A mismatch
    # changes prefix block hashes and silently degrades routing on one side.
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


def dgd_name(cfg: BenchConfig, topology: str, side: str) -> str:
    _check(topology, side)
    return f"{cfg.name_prefix}-{topology}-{'fe' if side == 'frontend' else 'gw'}"


def frontend_service(cfg: BenchConfig, topology: str) -> str:
    """Service the operator creates for the A-side Frontend component."""
    return f"{dgd_name(cfg, topology, 'frontend')}-frontend"


def route_name(cfg: BenchConfig, topology: str) -> str:
    return f"{dgd_name(cfg, topology, 'gateway')}-route"


def loadgen_pod_name(cfg: BenchConfig) -> str:
    return f"{cfg.name_prefix}-loadgen"


def request_headers(cfg: BenchConfig, topology: str) -> dict[str, str]:
    """Headers sent on every request, identical for A and B.

    The A side ignores them; sending them anyway keeps request bytes equal.
    """
    return {
        MODEL_HEADER: cfg.model,
        TARGET_HEADER: dgd_name(cfg, topology, "gateway"),
    }


def build_dgd(cfg: BenchConfig, topology: str, side: str) -> dict[str, Any]:
    _check(topology, side)
    components: list[dict[str, Any]] = [_router_component(cfg, side)]
    if topology == "agg":
        components.append(_worker_component(cfg, side, "decode", cfg.agg_workers, None))
    else:
        components.append(
            _worker_component(cfg, side, "prefill", cfg.prefill_workers, "prefill")
        )
        components.append(
            _worker_component(cfg, side, "decode", cfg.decode_workers, "decode")
        )
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": dgd_name(cfg, topology, side),
            "labels": {
                "app.kubernetes.io/part-of": "gateway-ab",
                "gateway-ab/topology": topology,
                "gateway-ab/side": side,
            },
        },
        "spec": {"components": components},
    }


def build_httproute(cfg: BenchConfig, topology: str) -> dict[str, Any]:
    """Route from the shared Gateway to the B-side InferencePool."""
    name = dgd_name(cfg, topology, "gateway")
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": route_name(cfg, topology),
            "labels": {"app.kubernetes.io/part-of": "gateway-ab"},
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
                                {"name": TARGET_HEADER, "type": "Exact", "value": name}
                            ],
                            "path": {"type": "PathPrefix", "value": "/"},
                        }
                    ],
                    "backendRefs": [
                        {
                            "group": "inference.networking.k8s.io",
                            "kind": "InferencePool",
                            # The operator names the pool after the DGD.
                            "name": f"{name}-pool",
                            "port": 8000,
                        }
                    ],
                    "timeouts": {"request": "300s"},
                }
            ],
        },
    }


def build_loadgen_pod(cfg: BenchConfig) -> dict[str, Any]:
    """Long-lived pod that runs aiperf against both sides.

    One pod for the whole experiment: A and B are measured from the same
    client, on the same node, with the same aiperf and tokenizer cache.
    """
    container: dict[str, Any] = {
        "name": "loadgen",
        "image": cfg.frontend_image,
        "command": ["sleep", "infinity"],
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
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": loadgen_pod_name(cfg),
            "labels": {
                "app.kubernetes.io/part-of": "gateway-ab",
                "app": LOADGEN_LABEL,
            },
        },
        "spec": spec,
    }


def build_all(cfg: BenchConfig, topologies: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every manifest for the given topologies, loadgen pod first."""
    out: list[dict[str, Any]] = [build_loadgen_pod(cfg)]
    for topology in topologies:
        out.append(build_dgd(cfg, topology, "frontend"))
        out.append(build_dgd(cfg, topology, "gateway"))
        out.append(build_httproute(cfg, topology))
    return out


def _router_component(cfg: BenchConfig, side: str) -> dict[str, Any]:
    if side == "frontend":
        container: dict[str, Any] = {
            "name": "main",
            "image": cfg.frontend_image,
            "env": [{"name": "DYN_ROUTER_MODE", "value": "kv"}],
            "resources": {"requests": {"cpu": cfg.router_cpu, "memory": "2Gi"}},
        }
        name, ctype = "Frontend", "frontend"
    else:
        container = {
            "name": "main",
            "image": cfg.frontend_image,
            "env": [
                {"name": "DYN_MODEL_NAME", "value": cfg.model},
                {"name": "DYN_KV_CACHE_BLOCK_SIZE", "value": str(cfg.block_size)},
            ],
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


def _worker_component(
    cfg: BenchConfig,
    side: str,
    ctype: str,
    replicas: int,
    disagg_mode: str | None,
) -> dict[str, Any]:
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
    args += list(cfg.mocker_extra_args)
    worker: dict[str, Any] = {
        "name": "main",
        "image": cfg.worker_image,
        "command": ["python3", "-m", "dynamo.mocker"],
        "args": args,
        "resources": {"requests": {"cpu": cfg.worker_cpu, "memory": "2Gi"}},
    }
    _add_hf_secret(cfg, worker)
    containers = [worker]
    component: dict[str, Any] = {
        "name": ctype,
        "type": ctype,
        "replicas": replicas,
    }
    if side == "gateway":
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
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {SIDES}")
