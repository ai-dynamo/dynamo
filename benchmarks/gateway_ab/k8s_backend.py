# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes backend: DynamoGraphDeployments, HTTPRoutes, and a loadgen pod.

Side B runs exactly as in production: the Gateway, the EPP in Dynamo runtime
mode with its Kubernetes pod reflector, and a direct-mode Frontend sidecar in
every worker pod. Side A's Frontend sits behind the same Gateway through a
plain HTTPRoute to its Service.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from typing import Any

from . import manifests
from .kube import FatalError, ForbiddenError, Kube, KubeError, retry
from .loadgen import LoadGen

log = logging.getLogger("gateway_ab")

# Stamped on every DGD at deploy time; `run` refuses a mismatch.
FINGERPRINT = "gateway-ab/fingerprint"
REQUIRED_CRDS = (
    "dynamographdeployments.nvidia.com",
    "inferencepools.inference.networking.k8s.io",
    "httproutes.gateway.networking.k8s.io",
)
# Container states that waiting will not fix: TERMINAL_POD_WAITING_REASONS in
# deploy/utils/dynamo_deployment.py, which needs kubernetes_asyncio and so is
# not imported here, plus the image-pull failures a mistyped image causes.
TERMINAL_WAITING_REASONS = frozenset(
    {"CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff", "InvalidImageName"}
)


class K8sBackend:
    name = "kubernetes"
    setup_hint = "Install prerequisites with benchmarks/gateway_ab/setup_cluster.sh"

    def __init__(self, args: argparse.Namespace, cfg: manifests.BenchConfig) -> None:
        self.args = args
        self.cfg = cfg
        self.kube = Kube(args.context, args.namespace)
        self.loadgen = LoadGen(self.kube, manifests.loadgen_pod_name(cfg))

    def describe(self) -> str:
        return f"context {self.kube.context}, namespace {self.kube.namespace}"

    def environment(self) -> dict[str, Any]:
        try:
            nodes = self.kube.get_json(["nodes"], namespaced=False) or {"items": []}
        except KubeError:
            # Listing nodes needs cluster-scoped RBAC that a namespace user
            # may lack; the node list is informational only.
            nodes = {"items": []}
        return {
            "backend": self.name,
            "context": self.kube.context,
            "namespace": self.kube.namespace,
            "nodes": [
                {
                    "name": n["metadata"]["name"],
                    "cpu": n["status"]["allocatable"].get("cpu"),
                    "kubelet": n["status"]["nodeInfo"].get("kubeletVersion"),
                }
                for n in nodes["items"]
            ],
        }

    def deployment_identity(self) -> dict[str, Any]:
        """What deployments are built from beyond BenchConfig: nothing here."""
        return {}

    def target_identity(self) -> dict[str, Any]:
        """Where a run measures; resuming elsewhere would mix two clusters."""
        return {
            "context": self.kube.context,
            "namespace": self.kube.namespace,
            "gateway_url": self.args.gateway_url,
        }

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> list[str]:
        problems = []
        try:
            self.kube.kubectl(
                ["get", "namespace", self.kube.namespace], namespaced=False
            )
        except (KubeError, subprocess.TimeoutExpired) as e:
            return [f"cannot reach namespace {self.kube.namespace!r}: {e}"]
        try:
            for crd in REQUIRED_CRDS:
                try:
                    if not self.kube.exists(["crd", crd], namespaced=False):
                        problems.append(f"CRD {crd} is not installed")
                except ForbiddenError:
                    # Credentials scoped to the namespace cannot read CRDs;
                    # deploying fails clearly if one is missing.
                    log.warning("Not allowed to read CRD %s; skipping that check", crd)
            gateway = f"{self.cfg.gateway_namespace}/{self.cfg.gateway_name}"
            try:
                gw = self.kube.get_json(
                    [
                        "-n",
                        self.cfg.gateway_namespace,
                        "gateway",
                        self.cfg.gateway_name,
                    ],
                    namespaced=False,
                )
                if gw is None:
                    problems.append(f"Gateway {gateway} not found")
                elif not _condition_true(gw, "Programmed"):
                    problems.append(f"Gateway {gateway} is not Programmed")
            except ForbiddenError:
                # Finding the Gateway's URL reads it too.
                if not self.args.gateway_url:
                    problems.append(
                        f"not allowed to read Gateway {gateway}; pass --gateway-url"
                    )
            # Name the flag, not its value: the value is only a Secret's name,
            # but logging it trips code scanning's clear-text-secret check.
            for flag, secret in (
                ("--hf-secret", self.cfg.hf_secret),
                ("--image-pull-secret", self.cfg.image_pull_secret),
            ):
                if secret and not self.kube.exists(["secret", secret]):
                    problems.append(
                        f"the Secret named by {flag} was not found in "
                        f"{self.kube.namespace}"
                    )
        except (KubeError, subprocess.TimeoutExpired) as e:
            problems.append(f"cannot check the cluster: {e}")
        return problems

    # -- deploy ------------------------------------------------------------

    def deploy_loadgen(self) -> None:
        pod = manifests.loadgen_pod_name(self.cfg)
        want = manifests.build_loadgen_pod(self.cfg)
        current = self.kube.get_json(["pod", pod])
        if current is not None:
            # restartPolicy is Never, so a terminated loadgen never comes back;
            # and a pod built from another image or secret must not be reused.
            terminated = current.get("status", {}).get("phase") in (
                "Failed",
                "Succeeded",
            )
            stale = (
                current["metadata"]
                .get("annotations", {})
                .get(manifests.LOADGEN_SPEC_HASH)
                != want["metadata"]["annotations"][manifests.LOADGEN_SPEC_HASH]
            )
            if terminated or stale:
                log.info("Recreating loadgen pod %s", pod)
                self.kube.kubectl(["delete", "pod", pod, "--wait=true"])
                current = None
        if current is None:
            self.kube.apply([want])
        self.kube.wait(f"pod/{pod}", "Ready", 600)

    def deploy_topology(self, topology: str, fingerprint: str) -> None:
        # Every arm, deployed now or not: a direct arm left by an earlier run
        # must not keep its pods alongside this one.
        names = [manifests.dgd_name(self.cfg, topology, s) for s in manifests.ARMS]
        if any(self.kube.exists(["dynamographdeployment", n]) for n in names):
            # Always start from fresh deployments. An in-place update that
            # rolls the EPP and the workers together can bind the new EPP to
            # the outgoing worker generation's namespace, where it stays and
            # answers 503 once those workers drain (#16581). Recreating also
            # gives every run cold caches and router state.
            log.info("Recreating %s for a clean start", topology)
            self.teardown_topology(topology)
            self._wait_pods_gone(names)
        arms = manifests.arms(self.cfg)
        log.info("Deploying %s (%s)", topology, ", ".join(arms))
        dgds = [manifests.build_dgd(self.cfg, topology, s) for s in arms]
        for dgd in dgds:
            dgd["metadata"]["annotations"] = {FINGERPRINT: fingerprint}
        routes = [
            manifests.build_httproute(self.cfg, topology, s) for s in manifests.SIDES
        ]
        self.kube.apply([*dgds, *routes])
        for side in arms:
            name = manifests.dgd_name(self.cfg, topology, side)
            log.info("Waiting for DynamoGraphDeployment %s to settle", name)
            retry(
                lambda: self._check_settled(name),
                attempts=max(1, self.args.ready_timeout // 5),
                delay_s=5,
                what=f"DynamoGraphDeployment {name} to settle",
            )

    def deployed_fingerprint(self, topology: str) -> str | None:
        """The fingerprint all of a topology's DGDs carry, or None."""
        prints = set()
        for side in manifests.arms(self.cfg):
            dgd = self.kube.get_json(
                ["dynamographdeployment", manifests.dgd_name(self.cfg, topology, side)]
            )
            if dgd is None:
                return None
            prints.add(dgd["metadata"].get("annotations", {}).get(FINGERPRINT))
        return prints.pop() if len(prints) == 1 else None

    def _wait_pods_gone(self, names: list[str]) -> None:
        def check() -> None:
            for name in names:
                pods = self.kube.get_json(
                    ["pods", "-l", f"nvidia.com/dynamo-graph-deployment-name={name}"]
                ) or {"items": []}
                if pods["items"]:
                    raise KubeError(f"{len(pods['items'])} pods of {name} remain")

        retry(check, attempts=60, delay_s=5, what="old pods to terminate")

    def _check_settled(self, name: str) -> None:
        """Raise unless the DGD's current spec is fully rolled out and Ready.

        ``kubectl wait --for=condition=Ready`` right after an apply can match
        the previous generation's condition before the operator has seen the
        change, and measurement would then start mid-rollout with old and new
        pods both serving. A pod in a state waiting will not fix fails at once.
        """
        pods = self.kube.get_json(
            ["pods", "-l", f"nvidia.com/dynamo-graph-deployment-name={name}"]
        ) or {"items": []}
        for pod in pods["items"]:
            status = pod.get("status", {})
            for c in [
                *status.get("initContainerStatuses", []),
                *status.get("containerStatuses", []),
            ]:
                waiting = c.get("state", {}).get("waiting") or {}
                if waiting.get("reason") in TERMINAL_WAITING_REASONS:
                    raise FatalError(
                        f"{name}: pod {pod['metadata']['name']} container "
                        f"{c.get('name')}: {waiting['reason']} "
                        f"{waiting.get('message', '')}".rstrip()
                    )
        dgd = self.kube.get_json(["dynamographdeployment", name])
        if dgd is None:
            raise KubeError(f"{name} not found")
        generation = dgd["metadata"].get("generation")
        status = dgd.get("status", {})
        if status.get("observedGeneration") != generation:
            raise KubeError(f"{name}: generation {generation} not observed yet")
        ready = next(
            (c for c in status.get("conditions", []) if c.get("type") == "Ready"),
            None,
        )
        if not ready or ready.get("status") != "True":
            raise KubeError(f"{name}: not Ready")
        if ready.get("observedGeneration", generation) != generation:
            raise KubeError(f"{name}: Ready condition is from an older generation")
        phase = (status.get("rollingUpdate") or {}).get("phase", "")
        if phase not in ("", "Completed"):
            raise KubeError(f"{name}: rolling update {phase}")
        for comp, s in status.get("components", {}).items():
            want = s.get("replicas")
            if want is not None and not (
                s.get("readyReplicas") == s.get("updatedReplicas", want) == want
            ):
                raise KubeError(f"{name}: component {comp} not fully updated and ready")
        if any(p["metadata"].get("deletionTimestamp") for p in pods["items"]):
            raise KubeError(f"{name}: old pods still terminating")

    def diagnose(self, topology: str, side: str) -> str:
        """Best-effort hint for the failure modes seen in practice."""
        dgd = manifests.dgd_name(self.cfg, topology, side)
        if side != "epp":
            route = "the HTTPRoute status and " if side == "frontend" else ""
            return (
                f"Check {route}the Frontend and worker logs: kubectl logs -l "
                f"nvidia.com/dynamo-graph-deployment-name={dgd}"
            )
        logs = self.kube.kubectl(
            [
                "logs",
                "-l",
                f"nvidia.com/dynamo-graph-deployment-name={dgd},"
                "nvidia.com/dynamo-component-type=epp",
                "--tail=-1",
            ],
            check=False,
            timeout=60,
        ).stdout
        if topology == "disagg" and "Prefill worker discovered" not in logs:
            return (
                "The EPP never activated prefill routing, so decode sidecars reject "
                "requests without a prefill worker id. If this EPP image is 1.5.1 or "
                "older, it detects prefill workers only by the legacy "
                "ModelType::Prefill bit, which the mocker does not emit; use one with "
                "the worker_type-based detection (see README, 'Disaggregated serving "
                "needs'). Otherwise check that the prefill workers registered: "
                f"kubectl logs -l nvidia.com/dynamo-graph-deployment-name={dgd}"
            )
        return (
            "Check the HTTPRoute status and the EPP logs: "
            f"kubectl logs -l nvidia.com/dynamo-graph-deployment-name={dgd},"
            "nvidia.com/dynamo-component-type=epp"
        )

    def teardown_topology(self, topology: str) -> None:
        log.info("Tearing down %s", topology)
        names = [manifests.dgd_name(self.cfg, topology, s) for s in manifests.ARMS]
        self.kube.kubectl(
            [
                "delete",
                "dynamographdeployment",
                *names,
                "--ignore-not-found",
                "--wait=true",
            ],
            timeout=600,
        )
        self.kube.kubectl(
            [
                "delete",
                "httproute",
                *(manifests.route_name(self.cfg, topology, s) for s in manifests.SIDES),
                "--ignore-not-found",
            ]
        )

    def teardown_all(self) -> None:
        self.kube.delete_by_label(
            "dynamographdeployments.nvidia.com,httproutes.gateway.networking.k8s.io,pods",
            ",".join(f"{k}={v}" for k, v in manifests.labels(self.cfg).items()),
        )

    def url(self, topology: str, side: str) -> str:
        """A and B share the Gateway's URL; the target header picks the route."""
        if side == "direct":
            return f"http://{manifests.frontend_service(self.cfg, topology, side)}:8000"
        if self.args.gateway_url:
            return self.args.gateway_url.rstrip("/")
        ns, name = self.cfg.gateway_namespace, self.cfg.gateway_name
        gw = self.kube.get_json(["-n", ns, "gateway", name], namespaced=False) or {}
        http_ports = {
            listener.get("port")
            for listener in gw.get("spec", {}).get("listeners", [])
            if listener.get("protocol") == "HTTP"
        }
        svcs = self.kube.get_json(
            ["-n", ns, "svc", "-l", f"gateway.networking.k8s.io/gateway-name={name}"],
            namespaced=False,
        )
        # The Service also exposes ports that are not listeners (Istio lists
        # its status port first), so take one that a plaintext listener serves.
        for svc in (svcs or {}).get("items", []):
            for port in svc["spec"].get("ports", []):
                if port.get("port") in http_ports:
                    return (
                        f"http://{svc['metadata']['name']}.{ns}"
                        f".svc.cluster.local:{port['port']}"
                    )
        raise KubeError(
            f"no Service port of Gateway {ns}/{name} serves an HTTP listener; "
            "pass --gateway-url"
        )


def _condition_true(obj: dict[str, Any], kind: str) -> bool:
    for c in obj.get("status", {}).get("conditions", []):
        if c.get("type") == kind:
            return c.get("status") == "True"
    return False
