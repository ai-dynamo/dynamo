# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the Slime tutorial end-to-end test (test_slime_kuberay.py).

tests/deploy/slime/ holds the tutorial's manifests, one file per step. These
helpers apply a run's cluster-specific options to them, read the run's expected
counts from the RayJob, wait for the RayJob, and parse the logs that the
tutorial's checks read; kubectl_utils.py creates and deletes the objects. They
import only the standard library and PyYAML, so test_slime_utils.py checks them
without a cluster.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import shlex
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from tests.deploy.kubectl_utils import (
    GPU_RESOURCE,
    POLL_INTERVAL,
    Kubectl,
    StatusReader,
    log_tail,
)

logger = logging.getLogger(__name__)

# TODO(#18569): once the Slime tutorial is on main, add a GPU-free test that
# parses the manifests on its pages, fills in their placeholders, and checks
# that they match these copies, so that a tutorial edit fails CI until this
# directory changes too.
MANIFEST_DIR = Path(__file__).resolve().parent / "slime"

# Names that the tutorial's manifests and commands use.
TUTORIAL_CLAIM = "slime-data"
SETUP_JOB = "slime-setup"
RAYJOB = "slime-grpo"
ROLE_LABEL = "dynamo.nvidia.com/slime-role"
FRONTEND_ROLE = "frontend"
REQUIRED_CRDS = ("dynamographdeployments.nvidia.com", "rayjobs.ray.io")

# Marks each object that this test creates, with the run's ID as its value
# (kubectl_utils.Owner). A run deletes only objects with this label: its own
# when it exits, and earlier runs' leftovers when it starts. A reader's own
# tutorial objects lack the label, so the test never deletes them.
OWNER_LABEL = "dynamo.nvidia.com/slime-e2e-test"

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
    gpu_tolerations: taint keys that every Pod requesting a GPU tolerates, as
        the tutorial asks for clusters whose GPU nodes are tainted.
    gpu_node_selector: node labels that every Pod requesting a GPU must match,
        such as a label of nodes whose GPUs are not split into MIG slices,
        which the trainer outgrows.
    """

    storage_class: str = ""
    existing_claim: str = ""
    slime_image: str = ""
    gpu_tolerations: tuple[str, ...] = ()
    gpu_node_selector: dict[str, str] = field(default_factory=dict)


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
        return cls(
            pvc=pvc,
            setup_job=setup_job,
            engine_services=engine_services,
            dgd=dgd,
            functions=functions,
            rayjob=rayjob,
        )

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
        if not any(_requests_gpu(c) for c in containers):
            continue
        for key in options.gpu_tolerations:
            node.setdefault("tolerations", []).append(
                {
                    "key": key,
                    # Exists, not Equal: on clusters that run the
                    # ExtendedResourceToleration and PodTolerationRestriction
                    # admission plugins, such as AKS, an Equal toleration next
                    # to the injected Exists one makes the Pod un-updatable,
                    # and the setup Job never completes.
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            )
        if options.gpu_node_selector:
            node.setdefault("nodeSelector", {}).update(options.gpu_node_selector)


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


async def wait_for_rayjob(
    kubectl: Kubectl, name: str, start_timeout: float, run_timeout: float
) -> dict[str, Any]:
    """Wait for Ray to start the RayJob's job, then for the job to succeed.

    The first timeout covers what the tutorial's ``kubectl wait`` for RUNNING
    covers: the Ray head's start and the job's submission. Raises as soon as
    KubeRay reports a failure. Returns the RayJob's final status.
    """
    reader = StatusReader(kubectl, "rayjob", name)
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
                    f"Driver log tail:\n{log_tail(kubectl, f'job/{name}')}\n"
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
