# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the Slime tutorial end-to-end test (test_slime_kuberay.py).

tests/deploy/slime/ holds the tutorial's manifests, one file per step. These
helpers apply a run's cluster-specific options to them, read the run's expected
counts from the RayJob, create and delete the objects with kubectl, and parse
the logs that the tutorial's checks read. They import only the standard library
and PyYAML, so test_slime_utils.py checks them without a cluster.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shlex
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

logger = logging.getLogger(__name__)

MANIFEST_DIR = Path(__file__).resolve().parent / "slime"

# Names that the tutorial's manifests and commands use.
TUTORIAL_CLAIM = "slime-data"
SETUP_JOB = "slime-setup"
RAYJOB = "slime-grpo"
ROLE_LABEL = "dynamo.nvidia.com/slime-role"
FRONTEND_ROLE = "frontend"
REQUIRED_CRDS = ("dynamographdeployments.nvidia.com", "rayjobs.ray.io")

# Marks every object that this test creates. In an existing namespace, the test
# deletes only leftovers that carry it, never a reader's own tutorial objects.
OWNER_LABEL = "dynamo.nvidia.com/slime-e2e-test"
OWNER_LABEL_VALUE = "true"

GPU_RESOURCE = "nvidia.com/gpu"
KUBECTL_TIMEOUT = 120
LOGS_TIMEOUT = 300
DELETE_TIMEOUT = 600
POLL_INTERVAL = 10

RAYJOB_FAILED = frozenset({"FAILED", "STOPPED"})
RAYJOB_DEPLOYMENT_FAILED = frozenset({"Failed", "ValidationFailed"})

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_SELECTED_WORKER = re.compile(
    r'Selected worker router_mode="(?P<mode>[^"]*)" worker_id=(?P<worker>\d+)'
)
_WEIGHT_RELOAD = 'POST /update_weights_from_disk HTTP/1.1" 200'
_DETECTED_ENGINES = re.compile(r"Detected external SGLang engines: (?P<engines>\[.*)")
_ENGINE_URL = re.compile(r"'url': '([^']+)'")
_ROLLOUT_METRICS = re.compile(r"\brollout (?P<index>\d+): \{'rollout/")
_STEP_METRICS = re.compile(r"\bstep (?P<index>\d+): \{'train/")
_LOGPROB_ABS_DIFF = re.compile(
    r"'train/train_rollout_logprob_abs_diff': (?P<value>[^,}]+)"
)


@dataclass(frozen=True)
class SlimeTutorialOptions:
    """Cluster-specific choices for one run of the tutorial.

    storage_class: the ReadWriteMany StorageClass for the tutorial's claim.
    existing_claim: an existing ReadWriteMany claim to mount instead. The run
        neither creates nor deletes it.
    slime_image: replaces the tutorial's Slime image in the setup Job and the
        Ray head.
    gpu_toleration: a taint key that every Pod requesting a GPU tolerates, as
        the tutorial asks for clusters whose GPU nodes are tainted.
    """

    storage_class: str = ""
    existing_claim: str = ""
    slime_image: str = ""
    gpu_toleration: str = ""


@dataclass(frozen=True)
class SlimeTutorialManifests:
    """The tutorial's manifests, with one run's options applied."""

    pvc: dict[str, Any] | None
    setup_job: dict[str, Any]
    engine_services: list[dict[str, Any]]
    dgd: dict[str, Any]
    functions: dict[str, Any]
    rayjob: dict[str, Any]

    @classmethod
    def load(
        cls, options: SlimeTutorialOptions, directory: Path = MANIFEST_DIR
    ) -> SlimeTutorialManifests:
        def read(name: str) -> list[dict[str, Any]]:
            text = (directory / name).read_text(encoding="utf-8")
            return [document for document in yaml.safe_load_all(text) if document]

        (pvc,) = read("pvc.yaml")
        (setup_job,) = read("setup-job.yaml")
        (dgd,) = read("dgd.yaml")
        (functions,) = read("slime-functions.yaml")
        (rayjob,) = read("rayjob.yaml")
        engine_services = read("engine-services.yaml")

        tutorial_image = setup_job["spec"]["template"]["spec"]["containers"][0]["image"]
        for manifest in (setup_job, dgd, rayjob):
            _customize(manifest, options, tutorial_image)
        if options.existing_claim:
            pvc = None
        else:
            pvc["spec"]["storageClassName"] = options.storage_class
        manifests = cls(
            pvc=pvc,
            setup_job=setup_job,
            engine_services=engine_services,
            dgd=dgd,
            functions=functions,
            rayjob=rayjob,
        )
        for documents in manifests.creation_order():
            for document in documents:
                labels = document["metadata"].setdefault("labels", {})
                labels[OWNER_LABEL] = OWNER_LABEL_VALUE
        return manifests

    def creation_order(self) -> list[list[dict[str, Any]]]:
        """The tutorial's objects in the order that its pages create them."""
        groups = [] if self.pvc is None else [[self.pvc]]
        return groups + [
            [self.setup_job],
            self.engine_services,
            [self.dgd],
            [self.functions],
            [self.rayjob],
        ]

    def engine_roles(self) -> list[str]:
        """The worker role label that each engine Service selects."""
        return [
            service["spec"]["selector"][ROLE_LABEL] for service in self.engine_services
        ]

    def served_model_name(self) -> str:
        """The model name that the engines register with the frontend."""
        names = {
            value
            for component in self.dgd["spec"]["components"]
            for container in component["podTemplate"]["spec"]["containers"]
            for value in command_flags(container.get("args", [])).get(
                "--served-model-name", []
            )
        }
        if len(names) != 1:
            raise ValueError(f"expected one --served-model-name, found {names}")
        return names.pop()


def _customize(
    manifest: dict[str, Any], options: SlimeTutorialOptions, tutorial_image: str
) -> None:
    """Apply a run's options to every Pod spec in one manifest, in place."""
    for node in list(_mappings(manifest)):
        claim = node.get("persistentVolumeClaim")
        if (
            options.existing_claim
            and isinstance(claim, dict)
            and claim.get("claimName") == TUTORIAL_CLAIM
        ):
            claim["claimName"] = options.existing_claim

        containers = node.get("containers")
        if not isinstance(containers, list):
            continue
        if options.slime_image:
            for container in containers:
                if container.get("image") == tutorial_image:
                    container["image"] = options.slime_image
        if options.gpu_toleration and any(_requests_gpu(c) for c in containers):
            node.setdefault("tolerations", []).append(
                {
                    "key": options.gpu_toleration,
                    # Exists, not Equal: on clusters that run the
                    # ExtendedResourceToleration and PodTolerationRestriction
                    # admission plugins, such as AKS, an Equal toleration next
                    # to the injected Exists one makes the Pod un-updatable,
                    # and the setup Job never completes.
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            )


def _mappings(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every mapping in a parsed manifest, outermost first."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mappings(child)


def _requests_gpu(container: dict[str, Any]) -> bool:
    resources = container.get("resources") or {}
    return any(
        GPU_RESOURCE in (resources.get(kind) or {}) for kind in ("requests", "limits")
    )


def command_flags(tokens: list[str]) -> dict[str, list[str]]:
    """Map each --flag in a command line to the values that follow it."""
    flags: dict[str, list[str]] = {}
    values: list[str] | None = None
    for token in tokens:
        if token.startswith("--"):
            values = flags.setdefault(token, [])
        elif values is not None:
            values.append(token)
    return flags


def _single(flags: dict[str, list[str]], name: str) -> str:
    values = flags.get(name, [])
    if len(values) != 1:
        raise ValueError(f"expected one value for {name}, found {values}")
    return values[0]


@dataclass(frozen=True)
class SlimeTrainingPlan:
    """The run that the RayJob's entrypoint and runtime environment describe."""

    engine_addresses: tuple[str, ...]
    rollout_url: str
    hf_checkpoint: str
    ref_load: str
    prompt_data: str
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    global_batch_size: int

    @classmethod
    def from_rayjob(cls, rayjob: dict[str, Any]) -> SlimeTrainingPlan:
        flags = command_flags(shlex.split(rayjob["spec"]["entrypoint"]))
        env = yaml.safe_load(rayjob["spec"]["runtimeEnvYAML"])["env_vars"]
        return cls(
            engine_addresses=tuple(flags["--rollout-external-engine-addrs"]),
            rollout_url=env["DYNAMO_ROLLOUT_URL"],
            hf_checkpoint=_single(flags, "--hf-checkpoint"),
            ref_load=_single(flags, "--ref-load"),
            prompt_data=_single(flags, "--prompt-data"),
            num_rollout=int(_single(flags, "--num-rollout")),
            rollout_batch_size=int(_single(flags, "--rollout-batch-size")),
            n_samples_per_prompt=int(_single(flags, "--n-samples-per-prompt")),
            global_batch_size=int(_single(flags, "--global-batch-size")),
        )

    @property
    def rollout_requests(self) -> int:
        """Requests through the frontend: one per sample, 640 in the tutorial."""
        return self.num_rollout * self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def train_steps(self) -> int:
        """Optimizer steps over all rollouts, 10 in the tutorial."""
        return self.rollout_requests // self.global_batch_size

    @property
    def weight_syncs(self) -> int:
        """Weight loads per engine: one before the first rollout and one after
        each rollout's training, 11 in the tutorial."""
        return self.num_rollout + 1

    @property
    def staged_paths(self) -> tuple[PurePosixPath, ...]:
        """Paths that the trainer reads and the setup Job must create."""
        return (
            PurePosixPath(self.hf_checkpoint),
            PurePosixPath(self.ref_load),
            PurePosixPath(self.prompt_data).parent,
        )


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


@dataclass(frozen=True)
class RoutedRequests:
    """The frontend's routing decisions: one "Selected worker" line each."""

    total: int
    by_worker: Counter[str]
    router_modes: frozenset[str]


def routed_requests(frontend_log: str) -> RoutedRequests:
    """Count the frontend's routing decisions, as the tutorial's grep does."""
    total = 0
    by_worker: Counter[str] = Counter()
    modes: set[str] = set()
    for line in strip_ansi(frontend_log).splitlines():
        if "Selected worker" not in line:
            continue
        total += 1
        match = _SELECTED_WORKER.search(line)
        if match:
            by_worker[match["worker"]] += 1
            modes.add(match["mode"])
    return RoutedRequests(total, by_worker, frozenset(modes))


def count_weight_reloads(engine_log: str) -> int:
    """Count an engine's successful /update_weights_from_disk calls."""
    return sum(_WEIGHT_RELOAD in line for line in engine_log.splitlines())


@dataclass(frozen=True)
class DriverLogSummary:
    """What the Slime driver log says about the run.

    Rollouts and steps are sets of indices, so a log that repeats lines after
    the submitter reconnects to the job still counts each one once.
    """

    engine_urls: tuple[str, ...]
    rollouts: frozenset[int]
    logprob_abs_diff_by_step: dict[int, float]


def summarize_driver_log(driver_log: str) -> DriverLogSummary:
    engine_urls: tuple[str, ...] = ()
    rollouts: set[int] = set()
    diffs: dict[int, float] = {}
    for line in strip_ansi(driver_log).splitlines():
        if engines := _DETECTED_ENGINES.search(line):
            engine_urls = tuple(_ENGINE_URL.findall(engines["engines"]))
        if rollout := _ROLLOUT_METRICS.search(line):
            rollouts.add(int(rollout["index"]))
        if step := _STEP_METRICS.search(line):
            diff = _LOGPROB_ABS_DIFF.search(line)
            diffs[int(step["index"])] = float(diff["value"]) if diff else math.nan
    return DriverLogSummary(engine_urls, frozenset(rollouts), diffs)


def parse_ls_listing(log: str) -> dict[str, set[str]]:
    """Parse the output of ``ls DIR...``: each ``DIR:`` header, then its entries."""
    listing: dict[str, set[str]] = {}
    entries: set[str] | None = None
    for raw_line in log.splitlines():
        line = raw_line.rstrip()
        if line.startswith("/") and line.endswith(":"):
            entries = listing[line[:-1]] = set()
        elif not line:
            entries = None
        elif entries is not None:
            entries.add(line)
    return listing


def missing_paths(
    listing: dict[str, set[str]], paths: tuple[PurePosixPath, ...]
) -> list[str]:
    return [
        str(path)
        for path in paths
        if path.name not in listing.get(str(path.parent), set())
    ]


class KubectlError(RuntimeError):
    """A kubectl command failed."""


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

    def logs(self, target: str, *args: str) -> str:
        return self.run("logs", target, *args, timeout=LOGS_TIMEOUT).stdout

    def pod(self, selector: str) -> str:
        """Return the one Pod with the label, as each engine Service needs."""
        names = self.run(
            "get", "pods", f"--selector={selector}", "--output=name"
        ).stdout.split()
        if len(names) != 1:
            raise AssertionError(
                f"expected one Pod with label {selector}, found {names or 'none'}"
            )
        return names[0]

    def get_in_cluster(self, pod: str, url: str) -> tuple[str, str]:
        """GET a URL with curl from inside a Pod; return the status and body."""
        result = self.run(
            "exec",
            pod,
            "--container=main",
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

    def delete(self, manifest: str) -> subprocess.CompletedProcess[str]:
        """Delete a manifest's objects and wait until they and their
        dependents, such as a Job's Pods, are gone."""
        return self.run(
            "delete",
            "--ignore-not-found",
            "--wait=true",
            "--cascade=foreground",
            f"--timeout={DELETE_TIMEOUT}s",
            "--filename=-",
            input_=manifest,
            check=False,
            timeout=DELETE_TIMEOUT + 60,
        )

    def cleanup(
        self, description: str, action: Callable[[], subprocess.CompletedProcess[str]]
    ) -> None:
        """Run one cleanup command and record a failure instead of raising it,
        so later steps still run and an earlier test failure stays the one
        that pytest reports."""
        try:
            result = action()
        except (KubectlError, subprocess.SubprocessError, OSError) as error:
            message = f"{description}: {error}"
        else:
            if result.returncode == 0:
                return
            message = f"{description}: {result.stderr.strip()}"
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


def missing_crds(kubectl: Kubectl) -> list[str]:
    missing = []
    for crd in REQUIRED_CRDS:
        result = kubectl.run("get", "crd", crd, "--output=name", check=False)
        if result.returncode == 0:
            continue
        if "NotFound" not in result.stderr:
            raise KubectlError(f"could not read CRD {crd}: {result.stderr.strip()}")
        missing.append(crd)
    return missing


@contextmanager
def tutorial_namespace(kubectl: Kubectl) -> Iterator[bool]:
    """Use the namespace, creating it if it does not exist.

    Yields whether this run created it. A namespace created here is deleted on
    exit, as the tutorial's clean-up step does; an existing one is kept.
    """
    result = kubectl.run(
        "get", "namespace", kubectl.namespace, "--output=name", check=False
    )
    is_new = result.returncode != 0
    if is_new:
        if "NotFound" not in result.stderr:
            raise KubectlError(
                f"could not read namespace {kubectl.namespace}: "
                f"{result.stderr.strip()}"
            )
        kubectl.run("create", "namespace", kubectl.namespace)
        logger.info("Created namespace %s", kubectl.namespace)
    try:
        yield is_new
    finally:
        if is_new:
            kubectl.cleanup(
                f"deleting namespace {kubectl.namespace}",
                lambda: kubectl.run(
                    "delete",
                    "namespace",
                    kubectl.namespace,
                    "--ignore-not-found",
                    "--wait=true",
                    f"--timeout={DELETE_TIMEOUT}s",
                    check=False,
                    timeout=DELETE_TIMEOUT + 60,
                ),
            )


def _describe(documents: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{document['kind']}/{document['metadata']['name']}" for document in documents
    )


def _owned_or_absent(kubectl: Kubectl, document: dict[str, Any]) -> bool:
    """Whether the namespace lacks the document's object, or has it with
    OWNER_LABEL."""
    kind, name = document["kind"], document["metadata"]["name"]
    result = kubectl.run("get", kind, name, "--output=json", check=False)
    if result.returncode != 0:
        if "NotFound" not in result.stderr:
            raise KubectlError(f"could not read {kind}/{name}: {result.stderr.strip()}")
        return True
    labels = json.loads(result.stdout)["metadata"].get("labels") or {}
    return labels.get(OWNER_LABEL) == OWNER_LABEL_VALUE


def delete_leftovers(kubectl: Kubectl, manifests: SlimeTutorialManifests) -> None:
    """Delete the tutorial's objects that an interrupted run left behind.

    Fails before deleting anything if an object with one of the tutorial's
    names lacks OWNER_LABEL, such as a reader's own tutorial deployment or its
    claim. Goes in reverse creation order, so the RayJob and the DGD release
    the volume before its claim is deleted. Without this, a finished RayJob or
    Job from an earlier run would satisfy this run's waits.
    """
    groups = list(reversed(manifests.creation_order()))
    unowned = [
        document
        for documents in groups
        for document in documents
        if not _owned_or_absent(kubectl, document)
    ]
    if unowned:
        raise AssertionError(
            f"Namespace {kubectl.namespace} already has {_describe(unowned)} "
            f"without the {OWNER_LABEL} label, so this test did not create "
            "them and will not delete them. Use another namespace, or delete "
            "them yourself."
        )
    for documents in groups:
        result = kubectl.delete(yaml.safe_dump_all(documents, sort_keys=False))
        if result.returncode != 0:
            raise KubectlError(
                f"could not delete leftover {_describe(documents)}: "
                f"{result.stderr.strip()}"
            )


@contextmanager
def created(
    kubectl: Kubectl,
    documents: list[dict[str, Any]],
    *,
    before_delete: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Create the documents, and on exit delete them and wait for them to go.

    before_delete runs first on exit, to save logs that deletion removes. Its
    failures are only logged: missing diagnostics must not fail a run.
    """
    manifest = yaml.safe_dump_all(documents, sort_keys=False)
    names = _describe(documents)
    kubectl.run("create", "--filename=-", input_=manifest)
    logger.info("Created %s", names)
    try:
        yield
    finally:
        if before_delete is not None:
            try:
                before_delete()
            except (KubectlError, subprocess.SubprocessError, OSError) as error:
                logger.warning("Could not save diagnostics for %s: %s", names, error)
        kubectl.cleanup(f"deleting {names}", lambda: kubectl.delete(manifest))


def _log_tail(kubectl: Kubectl, target: str, lines: int = 40) -> str:
    result = kubectl.run("logs", target, f"--tail={lines}", check=False)
    return result.stdout or result.stderr


class _StatusReader:
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
    reader = _StatusReader(kubectl, "job", name)
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
                    f"Log tail:\n{_log_tail(kubectl, f'job/{name}')}\n"
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


async def wait_for_rayjob(
    kubectl: Kubectl, name: str, start_timeout: float, run_timeout: float
) -> dict[str, Any]:
    """Wait for Ray to start the RayJob's job, then for the job to succeed.

    The first timeout covers what the tutorial's ``kubectl wait`` for RUNNING
    covers: the Ray head's start and the job's submission. Raises as soon as
    KubeRay reports a failure. Returns the RayJob's final status.
    """
    reader = _StatusReader(kubectl, "rayjob", name)
    deadline = time.monotonic() + start_timeout
    started = False
    job_status = deployment_status = ""
    last: tuple[str, str] | None = None
    while True:
        status = reader.read()
        if status is not None:
            job_status = status.get("jobStatus", "")
            deployment_status = status.get("jobDeploymentStatus", "")
            if (job_status, deployment_status) != last:
                last = (job_status, deployment_status)
                logger.info(
                    "RayJob %s: jobStatus=%s jobDeploymentStatus=%s",
                    name,
                    job_status or "-",
                    deployment_status or "-",
                )
            if (
                job_status in RAYJOB_FAILED
                or deployment_status in RAYJOB_DEPLOYMENT_FAILED
            ):
                raise AssertionError(
                    f"RayJob {name} failed: jobStatus={job_status} "
                    f"jobDeploymentStatus={deployment_status} "
                    f"reason={status.get('reason', '')} "
                    f"message={status.get('message', '')}\n"
                    f"Driver log tail:\n{_log_tail(kubectl, f'job/{name}')}\n"
                    f"{kubectl.snapshot()}"
                )
            if job_status == "SUCCEEDED" and deployment_status == "Complete":
                return status
            if not started and job_status not in ("", "PENDING"):
                started = True
                deadline = time.monotonic() + run_timeout
        if time.monotonic() >= deadline:
            phase = "finish" if started else "start"
            raise AssertionError(
                f"RayJob {name} did not {phase} in time: jobStatus={job_status} "
                f"jobDeploymentStatus={deployment_status}\n{kubectl.snapshot()}"
            )
        await asyncio.sleep(POLL_INTERVAL)
