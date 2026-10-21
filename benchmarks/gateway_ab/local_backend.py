# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local backend: the whole comparison in Docker on one host, no Kubernetes.

Each topology gets its own Docker network and discovery volume. Workers are
the same mocker processes as on Kubernetes, discovering each other through
the file discovery backend. One Envoy stands in for the shared Gateway, with
a route per side selected by the target header, as on Kubernetes.

- Side A: mocker workers and a Dynamo Frontend with the KV router, behind
  Envoy on a route with ext_proc turned off.
- Side B: mocker workers, the EPP in Dynamo runtime mode, and one direct-mode
  Frontend, behind Envoy on a route configured the way Gateway API Inference
  Extension gateways are (ext_proc to the EPP, original-destination
  forwarding).
- Direct arm (optional): a copy of side A that the client reaches without
  Envoy.

Kubernetes is replaced in exactly one place. A Dynamo-mode EPP maps a picked
worker id to its pod's address through a Kubernetes pod reflector; here it
runs with ``DYN_EPP_STATIC_ENDPOINT`` pointing at the direct-mode Frontend,
which forwards each request to the worker (and prefill worker) the EPP named
in the routing headers. Selection, KV events, load bookkeeping, and the
ext_proc protocol are the production code paths.

What differs from the Kubernetes topology: one direct-mode Frontend serves
every B worker instead of a sidecar per worker pod, Envoy stands in for
agentgateway, and all containers share the host's CPUs with the load
generator.

Every container, network, and volume is named after ``--name-prefix`` and
labeled with it, so two runs with different prefixes can share a Docker host.
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

# Label keys. ``LABEL`` holds the run's prefix, ``TOPOLOGY_LABEL``
# ``<prefix>/<topology>``, and ``FINGERPRINT_LABEL`` the deployment fingerprint.
LABEL = "gateway-ab-local"
TOPOLOGY_LABEL = f"{LABEL}/topology"
FINGERPRINT_LABEL = f"{LABEL}/fingerprint"
# Shared by every run: it only caches tokenizers.
HF_VOLUME = "gwab-hf"
DEFAULT_ENVOY_IMAGE = "envoyproxy/envoy:v1.35.3"
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
# The container of each arm's Frontend. B's runs in direct mode behind the EPP.
FRONTEND_ROLES = {
    "frontend": "frontend",
    "direct": "dir-frontend",
    "epp": "epp-frontend",
}
EXT_PROC = "envoy.filters.http.ext_proc"


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
    """Dynamo namespace of one arm's workers: its Kubernetes deployment's name."""
    return manifests.dgd_name(cfg, topology, side)


def network(cfg: manifests.BenchConfig, topology: str) -> str:
    return f"gwab-{cfg.name_prefix}-{topology}"


def volume(cfg: manifests.BenchConfig, topology: str) -> str:
    """The topology's file-discovery store."""
    return f"{network(cfg, topology)}-kv"


def container(cfg: manifests.BenchConfig, topology: str, role: str) -> str:
    return f"{network(cfg, topology)}-{role}"


def loadgen_name(cfg: manifests.BenchConfig) -> str:
    return f"gwab-{cfg.name_prefix}-loadgen"


def envoy_config(cfg: manifests.BenchConfig, topology: str) -> str:
    """Envoy bootstrap in JSON (valid YAML), passed inline via --config-yaml.

    One listener with a route per side, selected by the target header:

    - B's route is what Gateway API Inference Extension gateways program:
      ext_proc to the EPP with full-duplex streamed bodies in both
      directions, and an original-destination cluster that forwards to the
      address the EPP returns in ``x-gateway-destination-endpoint``. The EPP
      refuses any other body mode.
    - A's route turns ext_proc off and forwards to A's Frontend, as a plain
      HTTPRoute to its Service does on Kubernetes.
    """

    def route(side: str, cluster: str) -> dict[str, Any]:
        # The same match as the arm's HTTPRoute on Kubernetes.
        return {
            "match": {
                "path": manifests.CHAT_PATH,
                "headers": [
                    {"name": name, "string_match": {"exact": value}}
                    for name, value in manifests.request_headers(
                        cfg, topology, side
                    ).items()
                ],
            },
            "route": {"cluster": cluster, "timeout": "300s"},
        }

    frontend_route = route("frontend", "frontend")
    frontend_route["typed_per_filter_config"] = {
        EXT_PROC: {
            "@type": "type.googleapis.com/envoy.extensions.filters.http."
            "ext_proc.v3.ExtProcPerRoute",
            "disabled": True,
        }
    }
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
                    "routes": [frontend_route, route("epp", "picked_endpoint")],
                }
            ]
        },
        "http_filters": [
            {
                "name": EXT_PROC,
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
                _dns_cluster(
                    "frontend",
                    container(cfg, topology, FRONTEND_ROLES["frontend"]),
                    FRONTEND_PORT,
                ),
                {
                    **_dns_cluster("epp", container(cfg, topology, "epp"), EPP_PORT),
                    "typed_extension_protocol_options": {
                        "envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                            "@type": "type.googleapis.com/envoy.extensions.upstreams."
                            "http.v3.HttpProtocolOptions",
                            "explicit_http_config": {"http2_protocol_options": {}},
                        }
                    },
                },
            ],
        }
    }
    return json.dumps(config)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as e:
        raise KubeError(f"cannot read --epp-binary: {e}") from e
    return digest.hexdigest()


def _dns_cluster(name: str, host: str, port: int) -> dict[str, Any]:
    """A cluster of one container, found by its name on the Docker network."""
    return {
        "name": name,
        "type": "STRICT_DNS",
        "connect_timeout": "5s",
        "load_assignment": {
            "cluster_name": name,
            "endpoints": [
                {
                    "lb_endpoints": [
                        {
                            "endpoint": {
                                "address": {
                                    "socket_address": {
                                        "address": host,
                                        "port_value": port,
                                    }
                                }
                            }
                        }
                    ]
                }
            ],
        },
    }


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
    setup_hint = "Check that Docker is installed and its daemon is running"

    def __init__(self, args: argparse.Namespace, cfg: manifests.BenchConfig) -> None:
        self.args = args
        self.cfg = cfg
        self.docker = Docker()
        self.loadgen = LoadGen(self.docker, loadgen_name(cfg))
        self.envoy_image = args.envoy_image or DEFAULT_ENVOY_IMAGE
        self.epp_binary = (
            str(pathlib.Path(args.epp_binary).resolve()) if args.epp_binary else None
        )
        self._epp_digest: str | None = None

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

    def deployment_identity(self) -> dict[str, Any]:
        """What containers are built from beyond BenchConfig.

        The EPP binary by content, since a rebuild usually keeps its path, and
        every image by id, since a tag can be re-pointed.
        """
        return {
            "epp_binary_sha256": self._epp_sha256(),
            "image_ids": {
                image: self._image_id(image)
                for image in (
                    self.cfg.frontend_image,
                    self.cfg.worker_image,
                    self.envoy_image,
                )
            },
        }

    def target_identity(self) -> dict[str, Any]:
        """The Docker daemon a run measures on; resuming on another would mix hosts."""
        return {
            "docker_host": os.environ.get("DOCKER_HOST")
            or self.docker.run(["context", "show"], check=False).stdout.strip()
        }

    def _epp_sha256(self) -> str | None:
        """The digest of --epp-binary when this run first read it."""
        if self.epp_binary is None:
            return None
        if self._epp_digest is None:
            self._epp_digest = _sha256(self.epp_binary)
        return self._epp_digest

    def _image_id(self, image: str) -> str | None:
        proc = self.docker.run(
            ["image", "inspect", "--format", "{{.Id}}", image], check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    def _labels(self, topology: str | None = None) -> list[str]:
        out = ["--label", f"{LABEL}={self.cfg.name_prefix}"]
        if topology is not None:
            out += ["--label", f"{TOPOLOGY_LABEL}={self.cfg.name_prefix}/{topology}"]
        return out

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> list[str]:
        try:
            if self.docker.run(["version"], check=False).returncode != 0:
                return ["cannot reach the Docker daemon"]
        except KubeError as e:
            return [str(e)]
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
        elif self.epp_binary:
            log.warning(
                "Side B runs the EPP from %s while side A runs the KV router in %s. "
                "Unless that image was built from the same commit, the deltas "
                "include every router change between the two builds.",
                self.epp_binary,
                self.cfg.frontend_image,
            )
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
        name = loadgen_name(self.cfg)
        current = self.docker.run(
            [
                "inspect",
                "--format",
                f'{{{{.State.Running}}}} {{{{index .Config.Labels "{LABEL}/spec"}}}}',
                name,
            ],
            check=False,
        )
        if current.returncode == 0:
            if current.stdout.split() == ["true", spec_hash]:
                return
            # Stopped, or started from another image or configuration.
            self.docker.run(["rm", "-f", name])
        self._make_writable(HF_VOLUME)
        self.docker.run(
            [
                "run",
                "-d",
                "--name",
                name,
                *self._labels(),
                "--label",
                f"{LABEL}/spec={spec_hash}",
                *spec,
            ]
        )

    def deploy_topology(self, topology: str, fingerprint: str) -> None:
        # Always from scratch: a fresh discovery store holds no stale cards
        # from killed workers, and every run starts with cold caches.
        self.teardown_topology(topology)
        net = network(self.cfg, topology)
        kv = volume(self.cfg, topology)
        labels = self._labels(topology)
        self.docker.run(["network", "create", *labels, net])
        self.docker.run(["volume", "create", *labels, kv])
        self._make_writable(kv, HF_VOLUME)
        self.docker.run(["network", "connect", net, loadgen_name(self.cfg)])
        labels += ["--label", f"{FINGERPRINT_LABEL}={fingerprint}"]

        def start(
            role: str,
            image: str,
            env: dict[str, str],
            docker_args: Sequence[str] = (),
            command: Sequence[str] = (),
        ) -> None:
            cmd = [
                "run",
                "-d",
                "--name",
                container(self.cfg, topology, role),
                "--network",
                net,
            ]
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

        arms = manifests.arms(self.cfg)
        log.info("Starting %s workers (%s)", topology, ", ".join(arms))
        for side in arms:
            ns = namespace(self.cfg, topology, side)
            for ctype, replicas, mode in manifests.worker_roles(self.cfg, topology):
                for i in range(replicas):
                    # As in Kubernetes, the command replaces the image's
                    # entrypoint, so the frontend image can run workers too.
                    start(
                        f"{manifests.TAGS[side]}-{ctype}{i}",
                        self.cfg.worker_image,
                        {"DYN_NAMESPACE": ns},
                        docker_args=["--entrypoint", "python3"],
                        command=[
                            "-m",
                            "dynamo.mocker",
                            *manifests.mocker_args(self.cfg, mode),
                        ],
                    )
        for side in arms:
            env = {"DYN_NAMESPACE": namespace(self.cfg, topology, side)}
            command = ["-m", "dynamo.frontend", "--http-port", str(FRONTEND_PORT)]
            if side == "epp":
                command += ["--router-mode", "direct"]
            else:
                env["DYN_ROUTER_MODE"] = "kv"
            # The frontend image's entrypoint is the EPP binary.
            start(
                FRONTEND_ROLES[side],
                self.cfg.frontend_image,
                env,
                docker_args=["--entrypoint", "python3"],
                command=command,
            )
        epp_frontend_ip = self.docker.run(
            [
                "inspect",
                "--format",
                f'{{{{(index .NetworkSettings.Networks "{net}").IPAddress}}}}',
                container(self.cfg, topology, FRONTEND_ROLES["epp"]),
            ]
        ).stdout.strip()
        epp_args = ["--entrypoint", "/epp"]
        if self.epp_binary:
            # The container runs whatever is at the path now, so a rebuild
            # since the run recorded the binary would go unrecorded.
            if _sha256(self.epp_binary) != self._epp_sha256():
                raise KubeError(
                    f"--epp-binary {self.epp_binary} changed during the run; "
                    "start again"
                )
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
                "DYN_NAMESPACE": namespace(self.cfg, topology, "epp"),
                # Envoy reaches the EPP over plaintext h2 on the private network.
                "DYN_SECURE_SERVING": "false",
                "DYN_EPP_STATIC_ENDPOINT": f"{epp_frontend_ip}:{FRONTEND_PORT}",
            },
            docker_args=epp_args,
        )
        self.docker.run(
            [
                "run",
                "-d",
                "--name",
                container(self.cfg, topology, "gateway"),
                "--network",
                net,
                *labels,
                *NOFILE,
                self.envoy_image,
                "--config-yaml",
                envoy_config(self.cfg, topology),
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
        """Wait until every worker of every arm has registered its model card."""
        want = {
            namespace(self.cfg, topology, side): sum(
                replicas
                for _, replicas, _ in manifests.worker_roles(self.cfg, topology)
            )
            for side in manifests.arms(self.cfg)
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

    def _topology_filter(self, topology: str) -> list[str]:
        return ["--filter", f"label={TOPOLOGY_LABEL}={self.cfg.name_prefix}/{topology}"]

    def deployed_fingerprint(self, topology: str) -> str | None:
        """The fingerprint of a topology whose every container runs, or None."""
        out = self.docker.run(
            [
                "ps",
                "-a",
                *self._topology_filter(topology),
                "--format",
                f'{{{{.State}}}} {{{{.Label "{FINGERPRINT_LABEL}"}}}}',
            ]
        ).stdout
        rows = [line.split(maxsplit=1) for line in out.splitlines() if line.strip()]
        workers = sum(n for _, n, _ in manifests.worker_roles(self.cfg, topology))
        # Workers and a Frontend per arm, plus the EPP and the gateway. A dead
        # or missing container leaves a deployment unfit to measure.
        expected = len(manifests.arms(self.cfg)) * (workers + 1) + 2
        if len(rows) != expected or any(row[0] != "running" for row in rows):
            return None
        prints = {row[-1] for row in rows}
        return prints.pop() if len(prints) == 1 else None

    def _check_running(self, topology: str) -> None:
        """Fail fast when a container exited, e.g. an EPP without the static knob."""
        out = self.docker.run(
            [
                "ps",
                "-a",
                *self._topology_filter(topology),
                "--format",
                "{{.Names}} {{.State}}",
            ]
        ).stdout
        dead = [
            name
            for name, state in (line.split() for line in out.splitlines())
            if state != "running"
        ]
        if dead:
            raise FatalError(
                f"containers exited: {', '.join(dead)} " f"(see: docker logs {dead[0]})"
            )

    def diagnose(self, topology: str, side: str) -> str:
        if side != "epp":
            logs = f"docker logs {container(self.cfg, topology, FRONTEND_ROLES[side])}"
            if side == "frontend":
                logs += f"; docker logs {container(self.cfg, topology, 'gateway')}"
            return f"Check the logs: {logs}"
        epp = container(self.cfg, topology, "epp")
        state = self.docker.run(
            ["inspect", "--format", "{{.State.Status}}", epp], check=False
        ).stdout.strip()
        logs = self.docker.run(["logs", epp], check=False)
        text = logs.stdout + logs.stderr
        # An EPP with the static endpoint says so at startup; one without it
        # reaches for Kubernetes once its workers register, and kube-rs fails
        # with "Failed to infer configuration".
        static = "DYN_EPP_STATIC_ENDPOINT is set" in text
        if state != "running":
            hint = ""
            if "Too many open files" in text:
                hint = (
                    " It ran out of inotify instances, which file discovery uses; "
                    "raise them with: sudo sysctl fs.inotify.max_user_instances=1024."
                )
            elif not static and "infer config" in text.lower():
                hint = (
                    " This EPP tried to reach Kubernetes, so it lacks "
                    "DYN_EPP_STATIC_ENDPOINT support; pass --epp-binary with an EPP "
                    "built from a branch that has it."
                )
            return f"The EPP exited ({state}).{hint} See: docker logs {epp}"
        if topology == "disagg" and "Prefill worker discovered" not in text:
            return (
                "The EPP has not discovered a prefill worker, so the direct-mode "
                "Frontend rejects requests without a prefill worker id. Check that "
                "the prefill workers registered: docker logs "
                f"{container(self.cfg, topology, 'epp-prefill0')}; docker logs {epp}"
            )
        return (
            f"Check: docker logs {epp}; docker logs "
            f"{container(self.cfg, topology, 'gateway')}"
        )

    def teardown_topology(self, topology: str) -> None:
        log.info("Tearing down %s", topology)
        names = self.docker.run(
            ["ps", "-aq", *self._topology_filter(topology)]
        ).stdout.split()
        if names:
            self.docker.run(["rm", "-f", *names])
        net = network(self.cfg, topology)
        self.docker.run(
            ["network", "disconnect", "-f", net, loadgen_name(self.cfg)], check=False
        )
        self.docker.run(["network", "rm", net], check=False)
        kv = volume(self.cfg, topology)
        rm = self.docker.run(["volume", "rm", "-f", kv], check=False)
        if rm.returncode != 0 and self.docker.exists("volume", kv):
            # `volume create` would reuse it, and its model cards from the
            # last run would pass the readiness check before any worker is up.
            raise KubeError(
                f"cannot remove volume {kv} (is a container outside this run "
                f"using it?): {rm.stderr.strip()}"
            )

    def teardown_all(self) -> None:
        for topology in manifests.TOPOLOGIES:
            self.teardown_topology(topology)
        self.docker.run(["rm", "-f", loadgen_name(self.cfg)], check=False)

    def url(self, topology: str, side: str) -> str:
        """A and B share Envoy's URL; the target header picks the route."""
        if side == "direct":
            role = FRONTEND_ROLES[side]
            return f"http://{container(self.cfg, topology, role)}:{FRONTEND_PORT}"
        return f"http://{container(self.cfg, topology, 'gateway')}:{GATEWAY_PORT}"
