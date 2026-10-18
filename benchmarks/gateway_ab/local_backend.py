# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local backend: the whole comparison in Docker on one host, no Kubernetes.

Each topology gets its own Docker network and discovery volume. Workers are
the same mocker processes as on Kubernetes, discovering each other through
the file discovery backend.

- Side A: mocker workers and a Dynamo Frontend with the KV router.
- Side B: mocker workers, Envoy configured the way Gateway API Inference
  Extension gateways are (ext_proc to the EPP, original-destination
  forwarding), the EPP in Dynamo runtime mode, and one direct-mode Frontend.

Kubernetes is replaced in exactly one place. A Dynamo-mode EPP maps a picked
worker id to its pod's address through a Kubernetes pod reflector; here it
runs with ``DYN_EPP_STATIC_ENDPOINT`` pointing at the direct-mode Frontend,
which forwards each request to the worker (and prefill worker) the EPP named
in the routing headers. Selection, KV events, load bookkeeping, and the
ext_proc protocol are the production code paths.

What differs from the Kubernetes topology: one direct-mode Frontend serves
every B worker instead of a sidecar per worker pod, and all containers share
the host's CPUs with the load generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import subprocess
from typing import Any, Sequence

from . import manifests
from .kube import FatalError, KubeError, retry
from .loadgen import RESULTS_ROOT, LoadGen

log = logging.getLogger("gateway_ab")

LABEL = "gateway-ab-local"
LOADGEN = "gwab-loadgen"
HF_VOLUME = "gwab-hf"
# Docker's default soft limit of 1024 open files is far below what Kubernetes
# gives a container.
NOFILE = ["--ulimit", "nofile=1048576:1048576"]
# File discovery watches the store with inotify, and the kernel caps inotify
# instances per user id across the whole host (fs.inotify.max_user_instances,
# often 128). The containers therefore keep the images' unprivileged default
# user: as root they would share that budget with every root daemon on the
# host, and the EPP fails with "Too many open files" in disaggregated mode.
MIN_INOTIFY_INSTANCES = 512
FRONTEND_PORT = 8000
GATEWAY_PORT = 8080
EPP_PORT = 9002


class Docker:
    """docker CLI wrapper. ``exec`` matches ``Kube.exec`` so LoadGen can use it."""

    def run(
        self,
        args: Sequence[str],
        *,
        input: str | None = None,
        check: bool = True,
        timeout: float | None = 600,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["docker", *args]
        log.debug("$ %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, input=input, capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError as e:
            raise KubeError("docker not found on PATH") from e
        if check and proc.returncode != 0:
            raise KubeError(
                f"{' '.join(cmd[:4])}... failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return proc

    def exec(
        self,
        container: str,
        command: Sequence[str],
        *,
        input: str | None = None,
        check: bool = True,
        timeout: float | None = 120,
    ) -> subprocess.CompletedProcess[str]:
        args = ["exec"]
        if input is not None:
            args.append("-i")
        return self.run(
            [*args, container, *command], input=input, check=check, timeout=timeout
        )

    def exists(self, kind: str, name: str) -> bool:
        return self.run([kind, "inspect", name], check=False).returncode == 0


def namespace(cfg: manifests.BenchConfig, topology: str, side: str) -> str:
    """Dynamo namespace of one side's workers."""
    return f"{cfg.name_prefix}-{topology}-{'fe' if side == 'frontend' else 'gw'}"


def container(topology: str, role: str) -> str:
    return f"gwab-{topology}-{role}"


def envoy_config(epp_host: str) -> str:
    """Envoy bootstrap in JSON (valid YAML), passed inline via --config-yaml.

    Mirrors what Gateway API Inference Extension gateways program: ext_proc to
    the EPP with full-duplex streamed bodies in both directions, and an
    original-destination cluster that forwards to the address the EPP returns
    in ``x-gateway-destination-endpoint``. The EPP refuses any other body mode.
    """
    hcm = {
        "@type": "type.googleapis.com/envoy.extensions.filters.network."
        "http_connection_manager.v3.HttpConnectionManager",
        "stat_prefix": "gateway",
        "stream_idle_timeout": "300s",
        "route_config": {
            "virtual_hosts": [
                {
                    "name": "all",
                    "domains": ["*"],
                    "routes": [
                        {
                            "match": {"prefix": "/"},
                            "route": {"cluster": "picked_endpoint", "timeout": "300s"},
                        }
                    ],
                }
            ]
        },
        "http_filters": [
            {
                "name": "envoy.filters.http.ext_proc",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.http."
                    "ext_proc.v3.ExternalProcessor",
                    "grpc_service": {
                        "envoy_grpc": {"cluster_name": "epp"},
                        "timeout": "30s",
                    },
                    "failure_mode_allow": False,
                    "message_timeout": "30s",
                    "allow_mode_override": True,
                    "send_body_without_waiting_for_header_response": True,
                    "processing_mode": {
                        "request_header_mode": "SEND",
                        "response_header_mode": "SEND",
                        "request_body_mode": "FULL_DUPLEX_STREAMED",
                        "response_body_mode": "FULL_DUPLEX_STREAMED",
                        "request_trailer_mode": "SEND",
                        "response_trailer_mode": "SEND",
                    },
                    "mutation_rules": {"allow_all_routing": True},
                },
            },
            {
                "name": "envoy.filters.http.router",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.http."
                    "router.v3.Router"
                },
            },
        ],
    }
    config = {
        "static_resources": {
            "listeners": [
                {
                    "name": "gateway",
                    "address": {
                        "socket_address": {
                            "address": "0.0.0.0",
                            "port_value": GATEWAY_PORT,
                        }
                    },
                    "filter_chains": [
                        {
                            "filters": [
                                {
                                    "name": "envoy.filters.network.http_connection_manager",
                                    "typed_config": hcm,
                                }
                            ]
                        }
                    ],
                }
            ],
            "clusters": [
                {
                    "name": "picked_endpoint",
                    "type": "ORIGINAL_DST",
                    "lb_policy": "CLUSTER_PROVIDED",
                    "connect_timeout": "5s",
                    "original_dst_lb_config": {
                        "use_http_header": True,
                        "http_header_name": "x-gateway-destination-endpoint",
                    },
                },
                {
                    "name": "epp",
                    "type": "STRICT_DNS",
                    "connect_timeout": "5s",
                    "typed_extension_protocol_options": {
                        "envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                            "@type": "type.googleapis.com/envoy.extensions.upstreams."
                            "http.v3.HttpProtocolOptions",
                            "explicit_http_config": {"http2_protocol_options": {}},
                        }
                    },
                    "load_assignment": {
                        "cluster_name": "epp",
                        "endpoints": [
                            {
                                "lb_endpoints": [
                                    {
                                        "endpoint": {
                                            "address": {
                                                "socket_address": {
                                                    "address": epp_host,
                                                    "port_value": EPP_PORT,
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                },
            ],
        }
    }
    return json.dumps(config)


# Counts the model cards each Dynamo namespace has registered in the file
# discovery store. Runs in a throwaway container with the store mounted.
_COUNT_CARDS = r"""
import collections, json, os, sys, urllib.parse
root = "/kv/v1/mdc"
counts = collections.Counter()
for name in os.listdir(root) if os.path.isdir(root) else []:
    counts[urllib.parse.unquote(name).split("/")[0]] += 1
print(json.dumps(counts))
"""


class LocalBackend:
    name = "local"

    def __init__(self, args: argparse.Namespace, cfg: manifests.BenchConfig) -> None:
        self.args = args
        self.cfg = cfg
        self.docker = Docker()
        self.loadgen = LoadGen(self.docker, LOADGEN)
        self.envoy_image = args.envoy_image
        self.epp_binary = (
            str(pathlib.Path(args.epp_binary).resolve()) if args.epp_binary else None
        )

    def describe(self) -> str:
        return f"docker, EPP {self.epp_binary or 'from the frontend image'}"

    def environment(self) -> dict[str, Any]:
        version = self.docker.run(
            ["version", "--format", "{{.Server.Version}}"], check=False
        ).stdout.strip()
        return {
            "backend": self.name,
            "docker": version,
            "nodes": [{"name": "localhost", "cpu": os.cpu_count()}],
            "epp_binary": self.epp_binary or "image /epp",
        }

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> list[str]:
        if self.docker.run(["version"], check=False).returncode != 0:
            return ["cannot reach the Docker daemon"]
        problems = []
        try:
            instances = int(
                pathlib.Path("/proc/sys/fs/inotify/max_user_instances").read_text()
            )
        except (OSError, ValueError):
            instances = None
        if instances is not None and instances < MIN_INOTIFY_INSTANCES:
            log.warning(
                "fs.inotify.max_user_instances is %d; file discovery in the "
                "disaggregated topology may exhaust it ('Too many open files'). "
                "Raise it with: sudo sysctl fs.inotify.max_user_instances=%d",
                instances,
                MIN_INOTIFY_INSTANCES * 2,
            )
        if self.epp_binary and not pathlib.Path(self.epp_binary).is_file():
            problems.append(f"--epp-binary {self.epp_binary} does not exist")
        for image in (self.cfg.frontend_image, self.cfg.worker_image, self.envoy_image):
            if not self.docker.exists("image", image):
                log.info("Pulling %s", image)
                if self.docker.run(
                    ["pull", image], check=False, timeout=1800
                ).returncode:
                    problems.append(f"cannot pull image {image}")
        return problems

    # -- deploy ------------------------------------------------------------

    def deploy_loadgen(self) -> None:
        spec = [
            *NOFILE,
            "-e",
            "HF_HOME=/hf",
            "-e",
            "TERM=dumb",
            "-v",
            f"{HF_VOLUME}:/hf",
            # aiperf's artifacts; the image's default user cannot create /results.
            "--tmpfs",
            f"{RESULTS_ROOT}:rw,mode=1777",
            "--entrypoint",
            "python3",
            self.cfg.frontend_image,
            "-c",
            manifests.REAPER,
        ]
        spec_hash = hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:16]
        current = self.docker.run(
            [
                "inspect",
                "--format",
                f'{{{{.State.Running}}}} {{{{index .Config.Labels "{LABEL}/spec"}}}}',
                LOADGEN,
            ],
            check=False,
        )
        if current.returncode == 0:
            if current.stdout.split() == ["true", spec_hash]:
                return
            # Stopped, or started from another image or configuration.
            self.docker.run(["rm", "-f", LOADGEN])
        self._make_writable(HF_VOLUME)
        self.docker.run(
            [
                "run",
                "-d",
                "--name",
                LOADGEN,
                "--label",
                f"{LABEL}=1",
                "--label",
                f"{LABEL}/spec={spec_hash}",
                *spec,
            ]
        )

    def deploy_topology(self, topology: str) -> None:
        # Always from scratch: a fresh discovery store holds no stale cards
        # from killed workers, and every run starts with cold caches.
        self.teardown_topology(topology)
        net = f"gwab-{topology}"
        kv = f"gwab-{topology}-kv"
        labels = ["--label", f"{LABEL}=1", "--label", f"{LABEL}/topology={topology}"]
        self.docker.run(["network", "create", *labels, net])
        self.docker.run(["volume", "create", *labels, kv])
        self._make_writable(kv, HF_VOLUME)
        self.docker.run(["network", "connect", net, LOADGEN])

        def start(
            role: str,
            image: str,
            env: dict[str, str],
            docker_args: Sequence[str] = (),
            command: Sequence[str] = (),
        ) -> None:
            cmd = ["run", "-d", "--name", container(topology, role), "--network", net]
            cmd += [*labels, *NOFILE, "-v", f"{kv}:/kv", "-v", f"{HF_VOLUME}:/hf"]
            env = {
                "DYN_DISCOVERY_BACKEND": "file",
                "DYN_FILE_KV": "/kv",
                "HF_HOME": "/hf",
                **env,
            }
            for key, value in env.items():
                cmd += ["-e", f"{key}={value}"]
            self.docker.run([*cmd, *docker_args, image, *command])

        log.info("Starting %s workers (both sides)", topology)
        for side in manifests.SIDES:
            ns = namespace(self.cfg, topology, side)
            tag = "fe" if side == "frontend" else "gw"
            for ctype, replicas, mode in manifests.worker_roles(self.cfg, topology):
                for i in range(replicas):
                    start(
                        f"{tag}-{ctype}{i}",
                        self.cfg.worker_image,
                        {"DYN_NAMESPACE": ns},
                        command=[
                            "python3",
                            "-m",
                            "dynamo.mocker",
                            *manifests.mocker_args(self.cfg, mode),
                        ],
                    )
        # The frontend image's entrypoint is the EPP binary.
        start(
            "frontend",
            self.cfg.frontend_image,
            {
                "DYN_NAMESPACE": namespace(self.cfg, topology, "frontend"),
                "DYN_ROUTER_MODE": "kv",
            },
            docker_args=["--entrypoint", "python3"],
            command=["-m", "dynamo.frontend", "--http-port", str(FRONTEND_PORT)],
        )
        start(
            "direct",
            self.cfg.frontend_image,
            {"DYN_NAMESPACE": namespace(self.cfg, topology, "gateway")},
            docker_args=["--entrypoint", "python3"],
            command=[
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(FRONTEND_PORT),
                "--router-mode",
                "direct",
            ],
        )
        direct_ip = self.docker.run(
            [
                "inspect",
                "--format",
                f'{{{{(index .NetworkSettings.Networks "{net}").IPAddress}}}}',
                container(topology, "direct"),
            ]
        ).stdout.strip()
        epp_args = ["--entrypoint", "/epp"]
        if self.epp_binary:
            epp_args = [
                "-v",
                f"{self.epp_binary}:/epp-dev:ro",
                "--entrypoint",
                "/epp-dev",
            ]
        start(
            "epp",
            self.cfg.frontend_image,
            {
                "DYN_NAMESPACE": namespace(self.cfg, topology, "gateway"),
                "DYN_MODEL_NAME": self.cfg.model,
                "DYN_KV_CACHE_BLOCK_SIZE": str(self.cfg.block_size),
                # Envoy reaches the EPP over plaintext h2 on the private network.
                "DYN_SECURE_SERVING": "false",
                "DYN_EPP_STATIC_ENDPOINT": f"{direct_ip}:{FRONTEND_PORT}",
            },
            docker_args=epp_args,
        )
        self.docker.run(
            [
                "run",
                "-d",
                "--name",
                container(topology, "gateway"),
                "--network",
                net,
                *labels,
                *NOFILE,
                self.envoy_image,
                "--config-yaml",
                envoy_config(container(topology, "epp")),
            ]
        )
        self._wait_for_workers(topology, kv)

    def _make_writable(self, *volumes: str) -> None:
        """Open named volumes to the images' default user.

        Fresh volumes are root-owned, and a cache written by an earlier run under
        another user would refuse writes, so this is recursive.
        """
        mounts = [arg for i, v in enumerate(volumes) for arg in ("-v", f"{v}:/v{i}")]
        self.docker.run(
            [
                "run",
                "--rm",
                "--user",
                "0",
                *mounts,
                "--entrypoint",
                "chmod",
                self.cfg.frontend_image,
                "-R",
                "a+rwX",
                *(f"/v{i}" for i in range(len(volumes))),
            ]
        )

    def _wait_for_workers(self, topology: str, kv: str) -> None:
        """Wait until every worker of both sides has registered its model card."""
        want = {
            namespace(self.cfg, topology, side): sum(
                replicas
                for _, replicas, _ in manifests.worker_roles(self.cfg, topology)
            )
            for side in manifests.SIDES
        }

        def check() -> None:
            self._check_running(topology)
            out = self.docker.run(
                [
                    "run",
                    "--rm",
                    "--user",
                    "0",
                    "-v",
                    f"{kv}:/kv:ro",
                    "--entrypoint",
                    "python3",
                    self.cfg.frontend_image,
                    "-c",
                    _COUNT_CARDS,
                ],
                timeout=60,
            ).stdout
            have = json.loads(out)
            missing = {ns: n for ns, n in want.items() if have.get(ns, 0) < n}
            if missing:
                raise KubeError(f"registered {have}, want {want}")

        log.info("Waiting for %s workers to register", topology)
        retry(
            check,
            attempts=max(1, self.args.ready_timeout // 3),
            delay_s=3,
            what=f"{topology} workers to register",
        )

    def _check_running(self, topology: str) -> None:
        """Fail fast when a container exited, e.g. an EPP without the static knob."""
        out = self.docker.run(
            [
                "ps",
                "-a",
                "--filter",
                f"label={LABEL}/topology={topology}",
                "--format",
                "{{.Names}} {{.State}}",
            ]
        ).stdout
        dead = [line.split()[0] for line in out.splitlines() if "running" not in line]
        if dead:
            raise FatalError(
                f"containers exited: {', '.join(dead)} " f"(see: docker logs {dead[0]})"
            )

    def diagnose(self, topology: str, side: str) -> str:
        if side != "gateway":
            return f"Check the logs: docker logs {container(topology, 'frontend')}"
        epp = container(topology, "epp")
        state = self.docker.run(
            ["inspect", "--format", "{{.State.Status}}", epp], check=False
        ).stdout.strip()
        logs = self.docker.run(["logs", epp], check=False)
        text = logs.stdout + logs.stderr
        if state != "running":
            hint = ""
            if "Too many open files" in text:
                hint = (
                    " It ran out of inotify instances, which file discovery uses; "
                    "raise them with: sudo sysctl fs.inotify.max_user_instances=1024."
                )
            elif "POD_NAMESPACE" in text or "kube" in text.lower():
                hint = (
                    " This EPP tried to reach Kubernetes, so it lacks "
                    "DYN_EPP_STATIC_ENDPOINT support; pass --epp-binary with an EPP "
                    "built from a branch that has it."
                )
            return f"The EPP exited ({state}).{hint} See: docker logs {epp}"
        if topology == "disagg" and "Prefill worker discovered" not in text:
            return (
                "The EPP never activated prefill routing; it predates worker_type-based "
                f"prefill detection. Pass --epp-binary with a newer build. See: docker logs {epp}"
            )
        return f"Check: docker logs {epp}; docker logs {container(topology, 'gateway')}"

    def teardown_topology(self, topology: str) -> None:
        log.info("Tearing down %s", topology)
        names = self.docker.run(
            ["ps", "-aq", "--filter", f"label={LABEL}/topology={topology}"]
        ).stdout.split()
        if names:
            self.docker.run(["rm", "-f", *names])
        net = f"gwab-{topology}"
        self.docker.run(["network", "disconnect", "-f", net, LOADGEN], check=False)
        self.docker.run(["network", "rm", net], check=False)
        self.docker.run(["volume", "rm", "-f", f"gwab-{topology}-kv"], check=False)

    def teardown_all(self) -> None:
        for topology in manifests.TOPOLOGIES:
            self.teardown_topology(topology)
        self.docker.run(["rm", "-f", LOADGEN], check=False)

    def url(self, topology: str, side: str) -> str:
        if side == "frontend":
            return f"http://{container(topology, 'frontend')}:{FRONTEND_PORT}"
        return f"http://{container(topology, 'gateway')}:{GATEWAY_PORT}"
