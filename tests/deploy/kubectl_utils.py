# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kubectl helpers for deploy tests that create Kubernetes objects from manifests.

Kubectl runs commands in one namespace, each with a timeout. use_namespace and
created delete what a test creates when it exits, including after a create
that fails partway. Owner labels each object with the test run that creates
it, so deletion skips objects that the run did not create, and delete_leftovers
removes what an interrupted earlier run left behind. wait_for_job waits for a
batch Job.

Tests create DGDs through ManagedDeployment (dgd_utils.py) instead. This module
imports only the standard library and PyYAML, so test_kubectl_utils.py checks
it without a cluster.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import yaml

logger = logging.getLogger(__name__)

GPU_RESOURCE = "nvidia.com/gpu"
KUBECTL_TIMEOUT = 120
LOGS_TIMEOUT = 300
DELETE_TIMEOUT = 600
POLL_INTERVAL = 10


class KubectlError(RuntimeError):
    """A kubectl command failed."""


# What cleanup records instead of raising: kubectl failures and timeouts, a
# missing kubectl binary, and kubectl output that does not parse.
CLEANUP_ERRORS = (KubectlError, subprocess.SubprocessError, OSError, ValueError)


@dataclass(frozen=True)
class Owner:
    """The label that marks the objects one test run creates.

    The key names the test and the value names the run. A run deletes only
    objects with its own value; delete_leftovers deletes those with any value.
    """

    label: str
    run: str

    @classmethod
    def new(cls, label: str) -> Owner:
        """An owner for a new run, named after the run's start time in UTC."""
        started = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return cls(label, f"{started}-{secrets.token_hex(3)}")

    def mark(self, document: dict[str, Any]) -> None:
        document["metadata"].setdefault("labels", {})[self.label] = self.run

    def owns(self, obj: dict[str, Any]) -> bool:
        """Whether this run created the object."""
        return _labels(obj).get(self.label) == self.run

    def labeled(self, obj: dict[str, Any]) -> bool:
        """Whether any run of the test created the object."""
        return self.label in _labels(obj)


def _labels(obj: dict[str, Any]) -> dict[str, str]:
    return obj["metadata"].get("labels") or {}


def _describe(documents: Iterable[dict[str, Any]]) -> str:
    return ", ".join(
        f"{document['kind']}/{document['metadata']['name']}" for document in documents
    )


class Kubectl:
    """Run kubectl against one namespace in the current context."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.cleanup_errors: list[str] = []

    def run(
        self,
        *args: str,
        input_: str | None = None,
        check: bool = True,
        timeout: float = KUBECTL_TIMEOUT,
    ) -> subprocess.CompletedProcess[str]:
        command = ["kubectl", f"--namespace={self.namespace}", *args]
        logger.debug("Running %s", shlex.join(command))
        result = subprocess.run(
            command,
            input=input_,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise KubectlError(
                f"{shlex.join(command)} exited with {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result

    def get(self, kind: str, name: str) -> dict[str, Any]:
        return json.loads(self.run("get", kind, name, "--output=json").stdout)

    def read(self, document: dict[str, Any]) -> dict[str, Any] | None:
        """The live object that a manifest document names, or None."""
        result = self.run(
            "get",
            "--filename=-",
            "--output=json",
            input_=yaml.safe_dump(document, sort_keys=False),
            check=False,
        )
        if result.returncode != 0:
            if "NotFound" in result.stderr:
                return None
            raise KubectlError(
                f"could not read {_describe([document])}: {result.stderr.strip()}"
            )
        live = json.loads(result.stdout)
        # kubectl returns objects that it reads from stdin as a List.
        if live["kind"] == "List":
            (live,) = live["items"]
        return live

    def logs(self, target: str, *args: str) -> str:
        return self.run("logs", target, *args, timeout=LOGS_TIMEOUT).stdout

    def pod(self, selector: str) -> str:
        """Return the one Pod with the label."""
        names = self.run(
            "get", "pods", f"--selector={selector}", "--output=name"
        ).stdout.split()
        if len(names) != 1:
            raise AssertionError(
                f"expected one Pod with label {selector}, found {names or 'none'}"
            )
        return names[0]

    def get_in_cluster(
        self, pod: str, url: str, container: str = "main"
    ) -> tuple[str, str]:
        """GET a URL with curl from inside a Pod; return the status and body.

        The Dynamo operator runs each DGD component in a container named main.
        """
        result = self.run(
            "exec",
            pod,
            f"--container={container}",
            "--",
            "curl",
            "--silent",
            "--max-time",
            "60",
            "--write-out",
            "\n%{http_code}",
            url,
            check=False,
        )
        body, _, status = result.stdout.rpartition("\n")
        return status.strip() or f"no response ({result.stderr.strip()})", body

    def delete(self, manifest: str) -> None:
        """Delete a manifest's objects and wait until they and their
        dependents, such as a Job's Pods, are gone."""
        self.run(
            "delete",
            "--ignore-not-found",
            "--wait=true",
            "--cascade=foreground",
            f"--timeout={DELETE_TIMEOUT}s",
            "--filename=-",
            input_=manifest,
            timeout=DELETE_TIMEOUT + 60,
        )

    def cleanup(self, description: str, action: Callable[[], None]) -> None:
        """Run one cleanup step and record a failure instead of raising it,
        so later steps still run and an earlier test failure stays the one
        that pytest reports."""
        try:
            action()
        except CLEANUP_ERRORS as error:
            message = f"{description}: {error}"
            logger.error("Cleanup failed: %s", message)
            self.cleanup_errors.append(message)

    def snapshot(self) -> str:
        """Pods and recent warning events in the namespace, for failure messages."""
        pods = self.run("get", "pods", "--output=wide", check=False)
        events = self.run(
            "get",
            "events",
            "--field-selector=type=Warning",
            "--sort-by=.lastTimestamp",
            check=False,
        )
        recent = "\n".join(events.stdout.splitlines()[-20:])
        return (
            f"Pods:\n{pods.stdout or pods.stderr}\n"
            f"Recent warning events:\n{recent or events.stderr}"
        )


def missing_crds(kubectl: Kubectl, crds: Iterable[str]) -> list[str]:
    """The CRDs, by name, that the cluster does not have."""
    missing = []
    for crd in crds:
        result = kubectl.run("get", "crd", crd, "--output=name", check=False)
        if result.returncode == 0:
            continue
        if "NotFound" not in result.stderr:
            raise KubectlError(f"could not read CRD {crd}: {result.stderr.strip()}")
        missing.append(crd)
    return missing


@contextmanager
def use_namespace(kubectl: Kubectl, owner: Owner) -> Iterator[bool]:
    """Use the namespace, creating it if it does not exist.

    Yields whether this run created it. A namespace that the run creates is
    deleted on exit, with everything in it; an existing one is kept.
    """
    namespace = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": kubectl.namespace},
    }
    if kubectl.read(namespace) is not None:
        yield False
        return
    with created(kubectl, [namespace], owner):
        yield True


@contextmanager
def created(
    kubectl: Kubectl,
    documents: list[dict[str, Any]],
    owner: Owner,
    *,
    before_delete: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Create the documents, and on exit delete the objects that this run created.

    Labels each document with owner first. The exit path also runs when the
    create fails: kubectl creates the documents one at a time, so some can
    exist when another fails, and the API server can accept an object after
    kubectl times out. It keeps objects that the run did not create, such as
    one that already had a document's name.

    before_delete runs first on exit, to save logs that deletion removes. Its
    failures are only logged: missing diagnostics must not fail a run.
    """
    for document in documents:
        owner.mark(document)
    names = _describe(documents)
    try:
        kubectl.run(
            "create",
            "--filename=-",
            input_=yaml.safe_dump_all(documents, sort_keys=False),
        )
        logger.info("Created %s", names)
        yield
    finally:
        if before_delete is not None:
            try:
                before_delete()
            except (KubectlError, subprocess.SubprocessError, OSError) as error:
                logger.warning("Could not save diagnostics for %s: %s", names, error)
        kubectl.cleanup(
            f"deleting {names}", partial(_delete_owned, kubectl, documents, owner)
        )


def _delete_owned(
    kubectl: Kubectl, documents: list[dict[str, Any]], owner: Owner
) -> None:
    """Delete the documents' objects that this run created, and wait for them
    and their dependents to go."""
    owned = []
    for document in documents:
        live = kubectl.read(document)
        if live is None:
            continue
        if owner.owns(live):
            owned.append(document)
        else:
            logger.info("Keeping %s: this run did not create it", _describe([live]))
    if owned:
        kubectl.delete(yaml.safe_dump_all(owned, sort_keys=False))


def delete_leftovers(
    kubectl: Kubectl, groups: list[list[dict[str, Any]]], owner: Owner
) -> None:
    """Delete the objects that earlier, interrupted runs of the test left behind.

    groups holds the test's documents in the order that it creates them.
    Fails before deleting anything if an object with one of their names lacks
    the owner label, such as one that a person created. Deletes in reverse
    order, so that workloads release a volume before its claim goes, for
    example.
    """
    leftovers: list[list[dict[str, Any]]] = []
    unowned: list[dict[str, Any]] = []
    for documents in reversed(groups):
        group = []
        for document in documents:
            live = kubectl.read(document)
            if live is None:
                continue
            if owner.labeled(live):
                group.append(document)
            else:
                unowned.append(document)
        if group:
            leftovers.append(group)
    if unowned:
        raise AssertionError(
            f"Namespace {kubectl.namespace} already has {_describe(unowned)} "
            f"without the {owner.label} label, so this test did not create "
            "them and will not delete them. Use another namespace, or delete "
            "them yourself."
        )
    for group in leftovers:
        logger.info("Deleting %s, left by an earlier run", _describe(group))
        kubectl.delete(yaml.safe_dump_all(group, sort_keys=False))


def log_tail(kubectl: Kubectl, target: str, lines: int = 40) -> str:
    result = kubectl.run("logs", target, f"--tail={lines}", check=False)
    return result.stdout or result.stderr


class StatusReader:
    """Read an object's status while waiting on it.

    A few consecutive failed reads only log a warning, so a brief API server
    or credential hiccup does not end a half-hour wait.
    """

    max_failures = 6

    def __init__(self, kubectl: Kubectl, kind: str, name: str) -> None:
        self._kubectl = kubectl
        self._kind = kind
        self._name = name
        self._failures = 0

    def read(self) -> dict[str, Any] | None:
        try:
            status = self._kubectl.get(self._kind, self._name).get("status", {})
        except (KubectlError, subprocess.TimeoutExpired) as error:
            self._failures += 1
            if self._failures > self.max_failures:
                raise
            logger.warning(
                "Could not read %s/%s, will retry: %s", self._kind, self._name, error
            )
            return None
        self._failures = 0
        return status


async def wait_for_job(kubectl: Kubectl, name: str, timeout: float) -> None:
    """Wait for a Job's Complete condition, as ``kubectl wait`` does, but
    stop as soon as the Job fails."""
    reader = StatusReader(kubectl, "job", name)
    deadline = time.monotonic() + timeout
    conditions: dict[str, dict[str, Any]] = {}
    while True:
        status = reader.read()
        if status is not None:
            conditions = {
                condition["type"]: condition
                for condition in status.get("conditions", [])
                if condition.get("status") == "True"
            }
            if "Complete" in conditions:
                return
            if "Failed" in conditions:
                raise AssertionError(
                    f"Job {name} failed: "
                    f"{conditions['Failed'].get('message', '')}\n"
                    f"Log tail:\n{log_tail(kubectl, f'job/{name}')}\n"
                    f"{kubectl.snapshot()}"
                )
        if time.monotonic() >= deadline:
            hint = ""
            if "SuccessCriteriaMet" in conditions:
                hint = (
                    " Its Pod succeeded, but the Job controller could not "
                    "finish the Job. Check the Pod's tolerations: on AKS, a "
                    f"{GPU_RESOURCE} toleration with operator Equal causes this."
                )
            raise AssertionError(
                f"Job {name} did not complete within {timeout:.0f}s.{hint}\n"
                f"{kubectl.snapshot()}"
            )
        await asyncio.sleep(POLL_INTERVAL)
