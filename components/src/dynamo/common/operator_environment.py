# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only projection of operator-owned main-container environment bindings.

Values of None are generated, opaque and inert, not empty literals. The shared
operator_environment.json corpus checks this projection against rendered Go pods.
"""

import posixpath
import re
from collections.abc import Mapping
from typing import Any

MULTINODE_BINDINGS = {"DYNAMO_LEADER_ADDRESS": None, "DYNAMO_RANK": None}
_INTRAPOD_GMS = {"GMS_SOCKET_DIR": "/gms-intrapod-control"}
_INTERPOD_GMS = {
    "ENGINE_ID": None,
    "GMS_SOCKET_DIR": "/run/gms/shared",
    "FAILOVER_LOCK_PATH": "/run/gms/shared/failover.lock",
    "DYN_SYSTEM_STARTING_HEALTH_STATUS": "notready",
}
_INTRAPOD_FAILOVER = {
    "ENGINE_ID": None,
    "CONTAINER_NAME": None,
    "FAILOVER_LOCK_PATH": "/gms-intrapod-control/failover.lock",
    "DYN_SYSTEM_STARTING_HEALTH_STATUS": "notready",
    "DYN_SYSTEM_PORT": None,
    "DYN_SYSTEM_ENABLED": "true",
    "DYN_FORWARDPASS_METRIC_PORT": None,
}
_COLD_FAILOVER = {
    "DYN_VLLM_GMS_SHADOW_MODE": "true",
    "VLLM_NIXL_SIDE_CHANNEL_PORT": None,
    "DYN_VLLM_KV_EVENT_PORT": None,
}
_ELASTIC_EP_RAY = {"POD_IP": None, "VLLM_DP_MASTER_IP": None}
_POLICY_BINDINGS = {
    "DYN_TOPOLOGY_ENABLED": "true",
    "DYN_TOPOLOGY_MOUNT_PATH": "/etc/dynamo/topology",
}
_WORKER_TYPES = frozenset({"worker", "prefill", "decode"})
_BACKEND_PATTERNS = {
    name: re.compile(
        r"python[0-9.]*\s+[^|&;]*-m\s+[^|&;]*dynamo\." + name + r"[^|&;]*", re.ASCII
    )
    for name in ("vllm", "sglang", "trtllm")
}


def _pod_spec(component: Mapping[str, Any]) -> Mapping[str, Any]:
    return (component.get("podTemplate") or {}).get("spec") or {}


def _main(component: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        (
            container
            for container in _pod_spec(component).get("containers") or []
            if container.get("name") == "main"
        ),
        {},
    )


def _native(component: Mapping[str, Any]) -> bool:
    templates = [
        component,
        *(
            role
            for role in component.get("roles") or []
            if role.get("name") == "leader"
        ),
    ]
    return any(
        container.get("name") == "runtime"
        for template in templates
        for container in _pod_spec(template).get("initContainers") or []
    )


def component_backend(dgd: Mapping[str, Any], component: Mapping[str, Any]) -> str:
    """Match determineBackendFrameworkForComponent, before role projection."""

    if component.get("type") not in _WORKER_TYPES:
        return "noop"
    explicit = (dgd.get("spec") or {}).get("backendFramework") or ""
    templates = [
        component,
        *(
            role
            for role in component.get("roles") or []
            if role.get("podTemplate") is not None
        ),
    ]
    selected = ""
    for template in templates:
        main = _main(template)
        command, args = main.get("command") or [], main.get("args") or []
        launch = " ".join([*command, *args])
        if (
            command
            and "$(" not in command[0]
            and posixpath.basename(posixpath.normpath(command[0])) == "vllm-rs"
        ):
            detected = ["vllm"]
        else:
            detected = [
                name
                for name, pattern in _BACKEND_PATTERNS.items()
                if pattern.search(launch)
            ]
        if len(detected) > 1:
            if not explicit:
                raise ValueError("multiple backend frameworks detected from command")
            detected = []
        framework = detected[0] if detected else explicit
        if detected and explicit and framework != explicit:
            raise ValueError(
                f"backend framework mismatch: detected {framework!r}, configured {explicit!r}"
            )
        if framework and framework != "noop":
            if selected and selected != framework:
                raise ValueError(
                    "multiple backend frameworks detected across pod templates"
                )
            selected = framework
    return selected or explicit or "noop"


def _option(tokens: list[str], *names: str) -> str | None:
    """Match Go's last-option parsing, including shell control terminators."""

    value = None
    for index, token in enumerate(tokens):
        normalized = token.replace("_", "-")
        for name in names:
            if normalized == name and index + 1 < len(tokens):
                value = tokens[index + 1]
                if name == "--distributed-executor-backend" and not value.startswith(
                    "-"
                ):
                    value = re.split(r"[;&|<>()]", value, maxsplit=1)[0]
            if normalized.startswith(name + "="):
                candidate = re.split(r"[;&|<>()]", token.split("=", 1)[1], maxsplit=1)[
                    0
                ]
                if not candidate.startswith("--"):
                    value = candidate
    return value


def owned_environment(
    dgd: Mapping[str, Any], component: Mapping[str, Any]
) -> tuple[dict[str, str | None], set[str]]:
    """Project final owned bindings and omissions; do not render a PodSpec."""

    bindings: dict[str, str | None] = {}
    omitted: set[str] = set()
    multinode = component.get("multinode")
    nodes = multinode.get("nodeCount", 2) if isinstance(multinode, Mapping) else 1
    nodes = nodes if isinstance(nodes, int) else 1
    if isinstance(multinode, Mapping) and nodes > 1:
        bindings.update(MULTINODE_BINDINGS)
    experimental = component.get("experimental") or {}
    gms = experimental.get("gpuMemoryService")
    failover = experimental.get("failover")
    snapshot = (experimental.get("checkpoint") or {}).get("enabled")
    if isinstance(gms, Mapping) and gms.get("mode") in (None, "", "IntraPod"):
        bindings.update(_INTRAPOD_GMS)
        if snapshot:
            bindings["DYN_GMS_USE_V1"] = "true"
    if isinstance(gms, Mapping) and gms.get("mode") == "InterPod":
        bindings.update(_INTERPOD_GMS)
        omitted.add("DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS")
    if isinstance(failover, Mapping) and failover.get("mode") in (None, "", "IntraPod"):
        bindings.update(_INTRAPOD_FAILOVER)
        omitted.update(
            {"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", "DYN_HEALTH_CHECK_ENABLED"}
        )
        if snapshot:
            omitted.add("DYN_VLLM_GMS_SHADOW_MODE")
        else:
            bindings.update(_COLD_FAILOVER)
            if nodes > 1:
                bindings["NNODES"] = str(nodes)

    native = _native(component)
    backend = component_backend(dgd, component)
    if not native and backend == "vllm":
        main = _main(component)
        tokens = [
            token
            for part in [*(main.get("command") or []), *(main.get("args") or [])]
            for token in part.split()
        ]
        if (
            isinstance(gms, Mapping)
            and gms.get("mode") == "InterPod"
            and isinstance(failover, Mapping)
            and failover.get("mode") == "InterPod"
        ):
            bindings["DYN_VLLM_GMS_SHADOW_MODE"] = "true"
        if nodes > 1:
            annotations = {
                **((dgd.get("metadata") or {}).get("annotations") or {}),
                **((dgd.get("spec") or {}).get("annotations") or {}),
                **(
                    ((component.get("podTemplate") or {}).get("metadata") or {}).get(
                        "annotations"
                    )
                    or {}
                ),
            }
            mode = str(
                annotations.get("nvidia.com/vllm-distributed-executor-backend") or "mp"
            ).lower()
            if mode not in {"mp", "ray"}:
                mode = "mp"
            role_owned = any(
                isinstance(role.get("podTemplate"), Mapping)
                for role in component.get("roles") or []
            )
            if (
                role_owned and _option(tokens, "--distributed-executor-backend") == "mp"
            ) or (not role_owned and mode == "mp"):
                bindings["VLLM_NIXL_SIDE_CHANNEL_HOST"] = None
        elif (
            any(token.replace("_", "-") == "--enable-elastic-ep" for token in tokens)
            and _option(tokens, "--data-parallel-backend", "-dpb") == "ray"
        ):
            bindings.update(_ELASTIC_EP_RAY)

    policy = ((dgd.get("spec") or {}).get("experimental") or {}).get("kvTransferPolicy")
    if (
        not native
        and component.get("type") in _WORKER_TYPES
        and isinstance(policy, Mapping)
        and (policy.get("labelKey") or policy.get("clusterTopologyName"))
    ):
        bindings.update(_POLICY_BINDINGS)
        bindings["DYN_KV_TRANSFER_DOMAIN"] = str(policy.get("domain") or "")
        bindings["DYN_KV_TRANSFER_ENFORCEMENT"] = str(
            policy.get("enforcement") or "required"
        )
        if policy.get("preferredWeight") is None:
            omitted.add("DYN_KV_TRANSFER_PREFERRED_WEIGHT")
        else:
            bindings["DYN_KV_TRANSFER_PREFERRED_WEIGHT"] = None
    return bindings, omitted
