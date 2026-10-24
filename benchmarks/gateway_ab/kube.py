# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin kubectl wrapper. Every call pins ``--context`` and ``-n``.

A long benchmark must not follow the ambient current-context: another shell
switching contexts mid-run would otherwise redirect the remaining runs to a
different cluster.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Sequence

log = logging.getLogger(__name__)


class KubeError(RuntimeError):
    pass


class ForbiddenError(KubeError):
    """RBAC denied the read, so the object's presence is unknown."""


def _absent(stderr: str) -> bool:
    """Whether a failed ``kubectl get`` means the object or its kind is absent.

    Anything else (API timeouts, throttling, RBAC) is an error, not absence:
    reading it as "not found" would skip a forced recreate or end a wait for
    pods to terminate while they are still there.
    """
    return "(NotFound)" in stderr or "doesn't have a resource type" in stderr


class Kube:
    def __init__(self, context: str | None, namespace: str) -> None:
        self._context = context
        self.namespace = namespace

    @property
    def context(self) -> str:
        """The pinned context, resolved from the current context on first use.

        Resolved lazily so commands that never reach the cluster (``render``,
        ``report``) work on a machine without kubectl or a kubeconfig.
        """
        if self._context is None:
            self._context = _run(["kubectl", "config", "current-context"]).strip()
        return self._context

    def kubectl(
        self,
        args: Sequence[str],
        *,
        input: str | None = None,
        namespaced: bool = True,
        check: bool = True,
        timeout: float | None = 300,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["kubectl", "--context", self.context]
        if namespaced:
            cmd += ["-n", self.namespace]
        cmd += list(args)
        log.debug("$ %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, input=input, capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError as e:
            raise KubeError("kubectl not found on PATH") from e
        if check and proc.returncode != 0:
            raise KubeError(
                f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return proc

    def apply(self, objects: Sequence[dict[str, Any]]) -> None:
        doc = json.dumps({"apiVersion": "v1", "kind": "List", "items": list(objects)})
        out = self.kubectl(["apply", "-f", "-"], input=doc).stdout
        for line in out.splitlines():
            log.info("  %s", line)

    def get_json(
        self, args: Sequence[str], *, namespaced: bool = True
    ) -> dict[str, Any] | None:
        """The object as JSON, or None when it does not exist."""
        proc = self.kubectl(
            ["get", *args, "-o", "json"], namespaced=namespaced, check=False
        )
        if proc.returncode != 0:
            if _absent(proc.stderr):
                return None
            error = ForbiddenError if "(Forbidden)" in proc.stderr else KubeError
            raise error(f"kubectl get {' '.join(args)}: {proc.stderr.strip()}")
        return json.loads(proc.stdout)

    def exists(self, args: Sequence[str], *, namespaced: bool = True) -> bool:
        return self.get_json(args, namespaced=namespaced) is not None

    def wait(self, resource: str, condition: str, timeout_s: int) -> None:
        self.kubectl(
            [
                "wait",
                resource,
                f"--for=condition={condition}",
                f"--timeout={timeout_s}s",
            ],
            timeout=timeout_s + 30,
        )

    def exec(
        self,
        pod: str,
        command: Sequence[str],
        *,
        input: str | None = None,
        check: bool = True,
        timeout: float | None = 120,
    ) -> subprocess.CompletedProcess[str]:
        args = ["exec"]
        if input is not None:
            args.append("-i")
        args += [pod, "--", *command]
        return self.kubectl(args, input=input, check=check, timeout=timeout)

    def delete_by_label(self, kinds: str, selector: str) -> None:
        self.kubectl(
            ["delete", kinds, "-l", selector, "--ignore-not-found", "--wait=true"],
            timeout=600,
        )


class FatalError(KubeError):
    """A failure that waiting will not fix; ``retry`` re-raises it at once."""


def retry(fn, *, attempts: int, delay_s: float, what: str):
    """Call ``fn`` until it returns without raising, up to ``attempts`` times."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except FatalError:
            raise
        except Exception as e:  # noqa: BLE001 - surfaced after the last attempt
            last = e
            log.debug("%s: attempt %d/%d failed: %s", what, i + 1, attempts, e)
            time.sleep(delay_s)
    raise KubeError(f"{what} did not succeed after {attempts} attempts: {last}")


def _run(cmd: Sequence[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise KubeError(f"{cmd[0]} not found on PATH") from e
    if proc.returncode != 0:
        raise KubeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return proc.stdout
