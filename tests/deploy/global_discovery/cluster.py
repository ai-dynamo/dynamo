# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kubectl for one cluster of a multi-cluster test, and HTTP through a port-forward.

The deploy test helpers in dgd_utils.py talk to the one cluster that
``KUBECONFIG`` selects. A multi-cluster test needs one client per cluster, so
each :class:`Cluster` passes its own ``--kubeconfig`` to kubectl.
"""

import json
import logging
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import requests

from tests.deploy.vcluster_utils import retry_vcluster_api

logger = logging.getLogger(__name__)

# Errors from a vCluster API tunnel that recovers on its own; the same pattern
# as .github/scripts/retry_kubectl.sh.
_TUNNEL_ERRORS = re.compile(
    r"Unable to connect to the server:.*(EOF|connection refused|connection reset"
    r"|i/o timeout|TLS handshake timeout)"
    r"|The connection to the server .* was refused"
    r"|read: connection reset by peer",
    re.IGNORECASE,
)

_Value = TypeVar("_Value")


class KubectlError(RuntimeError):
    pass


class _TunnelError(KubectlError):
    """kubectl failed because the vCluster API tunnel dropped."""

    def __init__(self, result: subprocess.CompletedProcess):
        super().__init__(result.stderr.strip())
        self.result = result


@dataclass(frozen=True)
class Cluster:
    """kubectl bound to one cluster and one namespace."""

    role: str
    kubeconfig: Optional[str]
    namespace: str

    def kubectl(
        self,
        *args: str,
        stdin: Optional[str] = None,
        check: bool = True,
        timeout_s: float = 120,
    ) -> subprocess.CompletedProcess:
        command = ["kubectl"]
        if self.kubeconfig:
            command.append(f"--kubeconfig={self.kubeconfig}")
        command += [f"--namespace={self.namespace}", "--request-timeout=30s", *args]

        def run() -> subprocess.CompletedProcess:
            result = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if result.returncode != 0 and _TUNNEL_ERRORS.search(result.stderr):
                raise _TunnelError(result)
            return result

        try:
            result = retry_vcluster_api(
                f"running kubectl {args[0]} on the {self.role} cluster",
                run,
                (_TunnelError,),
                logger,
            )
        except _TunnelError as error:
            result = error.result
        if check and result.returncode != 0:
            raise KubectlError(
                f"{self.role}: kubectl {' '.join(args)} exited "
                f"{result.returncode}: {result.stderr.strip()}"
            )
        return result

    def apply(self, *objects: dict[str, Any]) -> None:
        manifest = json.dumps({"apiVersion": "v1", "kind": "List", "items": objects})
        self.kubectl(
            "apply",
            "--server-side",
            "--force-conflicts",
            "--field-manager=global-discovery-e2e",
            "-f",
            "-",
            stdin=manifest,
        )

    def get(self, kind: str, name: str) -> Optional[dict[str, Any]]:
        """The object as JSON, or None when it does not exist."""
        result = self.kubectl("get", kind, name, "--ignore-not-found", "-o", "json")
        return json.loads(result.stdout) if result.stdout.strip() else None

    def names(self, kind: str, selector: str) -> list[str]:
        result = self.kubectl("get", kind, "-l", selector, "-o", "name")
        return result.stdout.split()

    def delete(self, kind: str, name: str) -> None:
        """Start deleting the object; does not wait for it to go."""
        self.kubectl("delete", kind, name, "--ignore-not-found", "--wait=false")

    def service_address(self, name: str, port_name: str) -> str:
        """``<ClusterIP>:<port>`` of a Service, routable from the other vCluster."""
        service = self.get("service", name)
        if service is None:
            raise KubectlError(f"{self.role}: Service {name} does not exist")
        cluster_ip = service["spec"].get("clusterIP")
        if not cluster_ip or cluster_ip == "None":
            raise KubectlError(f"{self.role}: Service {name} has no ClusterIP")
        ports = service["spec"].get("ports", [])
        port = next((p["port"] for p in ports if p.get("name") == port_name), None)
        if port is None:
            raise KubectlError(f"{self.role}: Service {name} has no port {port_name}")
        if ":" in cluster_ip:
            cluster_ip = f"[{cluster_ip}]"
        return f"{cluster_ip}:{port}"

    def save(self, directory: Path, file_name: str, *args: str) -> str:
        """Run kubectl for diagnostics and write its output, errors included, to a file."""
        try:
            result = self.kubectl(*args, check=False, timeout_s=60)
            text = result.stdout + (
                f"\n[stderr]\n{result.stderr}" if result.stderr else ""
            )
        except Exception as error:  # Diagnostics must never mask the test failure.
            text = f"[kubectl {' '.join(args)} failed: {error!r}]"
        (directory / f"{self.role}-{file_name}").write_text(text)
        return text


def wait_for(
    description: str,
    probe: Callable[[], tuple[Optional[_Value], str]],
    timeout_s: float,
    interval_s: float = 2,
) -> _Value:
    """Call ``probe`` until it returns a value; it also returns what it saw.

    On timeout the error names the last observation, which is the useful
    part of a failure in a test that waits on several clusters.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        value, observed = probe()
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out after {timeout_s:.0f}s waiting for {description}; "
                f"last observed: {observed}"
            )
        time.sleep(interval_s)


class PortForwardHttp:
    """GET JSON from a Service through ``kubectl port-forward``.

    The forward ends when its pod restarts or the vCluster tunnel drops, so a
    failed request reopens it once and tries again.
    """

    def __init__(self, cluster: Cluster, service: str, port: int, log_path: Path):
        self._cluster = cluster
        self._service = service
        self._port = port
        self._log_path = log_path
        self._process: Optional[subprocess.Popen] = None
        self._base = ""

    def __enter__(self) -> "PortForwardHttp":
        self._open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._close()

    def get_json(self, path: str) -> Any:
        for attempt in (1, 2):
            try:
                response = requests.get(f"{self._base}{path}", timeout=10)
                response.raise_for_status()
                return response.json()
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt == 2:
                    raise
                logger.warning(
                    "GET %s through the port-forward failed, reopening it: %s",
                    path,
                    error,
                )
                self._close()
                self._open()

    def _open(self) -> None:
        # A forward started while the vCluster tunnel is down exits at once.
        retry_vcluster_api(
            f"opening a port-forward to service/{self._service} on the "
            f"{self._cluster.role} cluster",
            self._try_open,
            (KubectlError,),
            logger,
        )

    def _try_open(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            local_port = probe.getsockname()[1]
        command = ["kubectl"]
        if self._cluster.kubeconfig:
            command.append(f"--kubeconfig={self._cluster.kubeconfig}")
        command += [
            f"--namespace={self._cluster.namespace}",
            "port-forward",
            "--address=127.0.0.1",
            f"service/{self._service}",
            f"{local_port}:{self._port}",
        ]
        # Output goes to a file: kubectl logs every connection, and a pipe
        # nobody reads would fill and stall the forward.
        with self._log_path.open("a") as log:
            self._process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                    self._base = f"http://127.0.0.1:{local_port}"
                    return
            except OSError:
                time.sleep(0.5)
        self._close()
        raise KubectlError(
            f"{self._cluster.role}: port-forward to service/{self._service} did "
            f"not open; see {self._log_path}"
        )

    def _close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
