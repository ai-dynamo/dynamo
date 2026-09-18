# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a deployment is controlled.

Separate from :mod:`transport` on purpose. Transport says how to *reach*
Dynamo; a Deployment says whether we may *manipulate* it. A test that only
queries is handed a Dynamo with ``deployment=None`` and therefore cannot
restart anything or read container logs -- the capability is absent from the
object, not merely discouraged.
"""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .capabilities import Capability, Report, all_unknown, from_worker_flags

# Per-backend differences, kept in one place. vLLM takes --model; the others
# take --model-path.
BACKENDS: Dict[str, Dict[str, str]] = {
    "vllm": {"module": "dynamo.vllm", "model_flag": "--model"},
    "sglang": {"module": "dynamo.sglang", "model_flag": "--model-path"},
    "trtllm": {"module": "dynamo.trtllm", "model_flag": "--model-path"},
}

DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0"


class NotControllable(RuntimeError):
    """Raised when a test asks a Deployment-less Dynamo to change state."""


class Deployment:
    """Base: start() returns the base URL the frontend is reachable at."""

    def start(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:
        raise NotControllable(
            "this Dynamo was attached, not deployed by the test; "
            "construct Dynamo.deploy(...) to control lifecycle"
        )

    def kill(self) -> None:
        self.stop()

    def logs(self, tail: int = 200) -> str:
        raise NotControllable("no deployment handle; logs are unavailable")

    def restart_component(self, role: str, **flags: str) -> str:
        raise NotControllable(
            f"cannot restart {role}: this Dynamo was attached, not deployed by "
            "the test; construct Dynamo.deploy(...) to control lifecycle"
        )

    def capabilities(self) -> Dict[Capability, Report]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Attached(Deployment):
    """An already-running Dynamo we did not create. Query only."""

    base_url: str

    def start(self) -> str:
        return self.base_url

    def capabilities(self) -> Dict[Capability, Report]:
        """We did not launch it, so its flags are not ours to read."""
        return all_unknown(f"attached:{self.base_url}")


@dataclass
class Docker(Deployment):
    """A single container running frontend + worker.

    Uses the ``file`` discovery backend so no etcd or NATS is required -- both
    processes share the container filesystem. The default request plane is TCP,
    so nothing else needs to be stood up either.
    """

    image: str = DEFAULT_IMAGE
    model: str = "Qwen/Qwen3-0.6B"
    backend: str = "vllm"
    port: int = 8000
    gpus: Optional[str] = "all"
    name: Optional[str] = None
    hf_cache: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    worker_args: List[str] = field(default_factory=list)
    frontend_args: List[str] = field(default_factory=list)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(
                f"unknown backend {self.backend!r}; known: {sorted(BACKENDS)}"
            )
        self.name = self.name or f"dynamo-v2-{uuid.uuid4().hex[:8]}"

    # -- the command the container runs -----------------------------------
    def container_command(self) -> str:
        spec = BACKENDS[self.backend]
        worker = [
            "python3",
            "-m",
            spec["module"],
            spec["model_flag"],
            self.model,
            "--served-model-name",
            self.model,
            "--discovery-backend",
            "file",
            *self.worker_args,
        ]
        frontend = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--http-port",
            "8000",
            "--discovery-backend",
            "file",
            *self.frontend_args,
        ]
        return f"set -e\n{shlex.join(frontend)} &\nexec {shlex.join(worker)}\n"

    def capabilities(self) -> Dict[Capability, Report]:
        """Derived from the flags this deployment was launched with."""
        flags: Dict[str, str] = {}
        args = list(self.worker_args)
        for i, token in enumerate(args):
            if token.startswith("--"):
                value = (
                    args[i + 1]
                    if i + 1 < len(args) and not args[i + 1].startswith("--")
                    else ""
                )
                flags[token[2:]] = value
        return from_worker_flags(flags, f"docker:{self.image}", backend=self.backend)

    def _docker(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=600
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"docker {' '.join(args[:2])} failed ({proc.returncode}): "
                f"{proc.stderr.strip()[:400]}"
            )
        return proc.stdout

    def start(self) -> str:
        cmd = [
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
            "-p",
            f"{self.port}:8000",
            "--ipc",
            "host",
            "--shm-size",
            "16g",
        ]
        if self.gpus:
            cmd += ["--gpus", self.gpus]
        if self.hf_cache:
            cmd += [
                "-e",
                f"HF_HOME={self.hf_cache}",
                "-v",
                f"{self.hf_cache}:{self.hf_cache}",
            ]
        for key, value in self.env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += ["--entrypoint", "bash", self.image, "-lc", self.container_command()]
        self._docker(*cmd)
        self._started = True
        return f"http://localhost:{self.port}"

    def stop(self) -> None:
        if self._started:
            self._docker("rm", "-f", self.name, check=False)
            self._started = False

    def kill(self) -> None:
        if self._started:
            self._docker("kill", self.name, check=False)

    def args_for(self, role: str) -> List[str]:
        try:
            return {"frontend": self.frontend_args, "worker": self.worker_args}[role]
        except KeyError:
            raise ValueError(
                f"unknown component role {role!r}; known: frontend, worker"
            ) from None

    def restart_component(self, role: str, **flags: str) -> str:
        """Bring one component back with additional flags.

        Both processes share a container in this topology, so recreating it
        bounces the other component too. That is a property of the deployment
        shape, not of the request: a container-per-component or Kubernetes
        provider would restart only the named one. The flags, however, are
        always routed to the named component alone.
        """
        target = self.args_for(role)
        for key, value in flags.items():
            target += [f"--{key.replace('_', '-')}", str(value)]
        self.stop()
        time.sleep(1)
        return self.start()

    def logs(self, tail: int = 200) -> str:
        return self._docker("logs", "--tail", str(tail), self.name, check=False)
