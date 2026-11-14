# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test of the Slime reinforcement-learning tutorial on Kubernetes.

Runs the tutorial's three pages in order, with its manifests from
tests/deploy/slime/, against the current kubectl context:

1. Installation and Setup: the shared ReadWriteMany volume, and the setup Job
   that stages Qwen3-0.6B and GSM8K on it.
2. Serve Inference with Dynamo: the engine Services, and the DGD with a Dynamo
   frontend and two SGLang engines. Then the three addresses that the trainer
   uses must answer from inside the namespace.
3. Train with Slime on KubeRay: the ConfigMap with Slime's custom functions,
   the RayJob that runs 10 GRPO steps, and the tutorial's "Verify the Run"
   checks.

The RayJob's status decides whether training worked. The other checks are the
ones the tutorial tells readers to run, which read logs: the frontend routed
every rollout request round-robin, both engines loaded every weight update,
and the engines' token log-probabilities match the trainer's.

No CI job selects this test. Run it by path against a cluster with three free
GPUs, the Dynamo platform, and KubeRay; tests/deploy/README.md has the command.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path

import pytest
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.deploy.dgdr_utils import parse_served_model_ids
from tests.deploy.slime_utils import (
    FRONTEND_ROLE,
    LOGS_TIMEOUT,
    RAYJOB,
    ROLE_LABEL,
    SETUP_JOB,
    Kubectl,
    SlimeTrainingPlan,
    SlimeTutorialManifests,
    SlimeTutorialOptions,
    count_weight_reloads,
    created,
    delete_leftovers,
    missing_crds,
    missing_paths,
    parse_ls_listing,
    routed_requests,
    summarize_driver_log,
    tutorial_namespace,
    wait_for_job,
    wait_for_rayjob,
)
from tests.utils.output_paths import resolve_test_output_path

logger = logging.getLogger(__name__)

# Each wait has the timeout of the tutorial's matching kubectl wait. A run with
# the images cached takes about 10 minutes; a cold run spends most of its time
# pulling the 18 GB Slime image and the 14 GB SGLang runtime image.
SETUP_TIMEOUT = 30 * 60
DGD_READY_TIMEOUT = 15 * 60
RAYJOB_START_TIMEOUT = 30 * 60
# The 10 steps took about 3 minutes on H100.
RAYJOB_RUN_TIMEOUT = 30 * 60
SUBMITTER_TIMEOUT = 5 * 60
MODEL_LIST_TIMEOUT = 2 * 60
CLEANUP_ALLOWANCE = 15 * 60
TEST_TIMEOUT = (
    SETUP_TIMEOUT
    + DGD_READY_TIMEOUT
    + RAYJOB_START_TIMEOUT
    + RAYJOB_RUN_TIMEOUT
    + SUBMITTER_TIMEOUT
    + MODEL_LIST_TIMEOUT
    + CLEANUP_ALLOWANCE
)

# The tutorial reads train/train_rollout_logprob_abs_diff near 0.01 as proof
# that the engines generated with the weights that Slime synced; H100 runs
# measured 0.0096 to 0.0128. The bound leaves room for other GPUs and still
# fails a run whose engines serve weights other than the trainer's.
MAX_LOGPROB_ABS_DIFF = 0.05

pytestmark = [
    pytest.mark.k8s,
    pytest.mark.deploy,
    pytest.mark.e2e,
    # No CI job selects this test: every CI selector names a framework marker,
    # and this test carries none. A framework marker would also make
    # tests/conftest.py skip the test wherever SGLang is not installed, such as
    # on the workstation that drives the cluster.
    pytest.mark.nightly,
    # Three GPUs at once: two engines and the trainer. There is no gpu_3.
    pytest.mark.gpu_4,
    pytest.mark.slow,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
]


@pytest.fixture
def slime_options(request: pytest.FixtureRequest) -> SlimeTutorialOptions:
    options = SlimeTutorialOptions(
        storage_class=request.config.getoption("--slime-storage-class"),
        existing_claim=request.config.getoption("--slime-pvc"),
        slime_image=request.config.getoption("--slime-image"),
        gpu_toleration=request.config.getoption("--slime-gpu-toleration"),
    )
    if options.storage_class and options.existing_claim:
        pytest.fail("Pass --slime-storage-class or --slime-pvc, not both")
    if not (options.storage_class or options.existing_claim):
        pytest.skip(
            "The Slime tutorial test needs --slime-storage-class or --slime-pvc"
        )
    return options


@pytest.mark.timeout(TEST_TIMEOUT)
async def test_slime_tutorial(
    slime_options: SlimeTutorialOptions,
    namespace: str | None,
    image: str | None,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """Run the Slime tutorial's steps and checks, then delete what it created."""
    if not namespace:
        pytest.skip("The Slime tutorial test needs --namespace")

    kubectl = Kubectl(namespace)
    output_dir = Path(resolve_test_output_path(request.node.name))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = SlimeTutorialManifests.load(slime_options)
    plan = SlimeTrainingPlan.from_rayjob(manifests.rayjob)
    deployment_spec = _deployment_spec(
        manifests, tmp_path, image, request.config.getoption("--frontend-image")
    )

    missing = missing_crds(kubectl)
    assert not missing, (
        "Install the Dynamo platform and the KubeRay operator, as the tutorial's "
        f"first page does. Missing CRDs: {missing}"
    )

    async with AsyncExitStack() as stack:
        logger.info("Installation and Setup: staging the model and dataset")
        if not stack.enter_context(tutorial_namespace(kubectl)):
            delete_leftovers(kubectl, manifests)
        if manifests.pvc is not None:
            stack.enter_context(created(kubectl, [manifests.pvc]))
        stack.enter_context(
            created(
                kubectl,
                [manifests.setup_job],
                before_delete=partial(
                    _save_output,
                    kubectl,
                    output_dir / "setup-job.log",
                    "logs",
                    f"job/{SETUP_JOB}",
                ),
            )
        )
        await wait_for_job(kubectl, SETUP_JOB, SETUP_TIMEOUT)
        _check_staged_data(kubectl, plan, output_dir)

        logger.info("Serve Inference with Dynamo: starting the engines")
        stack.enter_context(created(kubectl, manifests.engine_services))
        await stack.enter_async_context(
            ManagedDeployment(
                log_dir=request.node.name,
                deployment_spec=deployment_spec,
                namespace=namespace,
                frontend_service_name=FRONTEND_ROLE,
                # NATS and etcd belong to the platform, not to this namespace.
                skip_service_restart=True,
                readiness_timeout=DGD_READY_TIMEOUT,
                fail_fast_startup=True,
            )
        )
        await _check_trainer_addresses(
            kubectl,
            plan,
            engine_pod_selector=f"{ROLE_LABEL}={manifests.engine_roles()[0]}",
            model=manifests.served_model_name(),
        )

        logger.info("Train with Slime on KubeRay: running %d steps", plan.train_steps)
        stack.enter_context(created(kubectl, [manifests.functions]))
        stack.enter_context(
            created(
                kubectl,
                [manifests.rayjob],
                before_delete=partial(_save_rayjob, kubectl, output_dir),
            )
        )
        await wait_for_rayjob(kubectl, RAYJOB, RAYJOB_START_TIMEOUT, RAYJOB_RUN_TIMEOUT)
        # The submitter Job, named after the RayJob, holds the driver log.
        await wait_for_job(kubectl, RAYJOB, SUBMITTER_TIMEOUT)

        logger.info("Verify the Run")
        _check_routing(kubectl, plan)
        _check_weight_reloads(kubectl, plan, manifests.engine_roles())
        _check_driver_log(kubectl, plan, output_dir)

    assert not kubectl.cleanup_errors, "Cleanup failed:\n" + "\n".join(
        kubectl.cleanup_errors
    )


def _deployment_spec(
    manifests: SlimeTutorialManifests,
    directory: Path,
    image: str | None,
    frontend_image: str | None,
) -> DeploymentSpec:
    """Wrap the tutorial's DGD for ManagedDeployment, with the image options
    that the other deploy tests take: --image sets every component, and
    --frontend-image then sets the frontend."""
    path = directory / "dgd.yaml"
    path.write_text(yaml.safe_dump(manifests.dgd, sort_keys=False), encoding="utf-8")
    spec = DeploymentSpec(str(path))
    if image:
        spec.set_image(image)
    if frontend_image:
        spec.set_image(frontend_image, service_name=FRONTEND_ROLE)
    return spec


def _save_output(kubectl: Kubectl, path: Path, *args: str) -> None:
    result = kubectl.run(*args, check=False, timeout=LOGS_TIMEOUT)
    path.write_text(result.stdout or result.stderr, encoding="utf-8")


def _save_rayjob(kubectl: Kubectl, output_dir: Path) -> None:
    """Keep the RayJob's status and the driver log, which deleting it removes."""
    _save_output(
        kubectl, output_dir / "rayjob.yaml", "get", "rayjob", RAYJOB, "--output=yaml"
    )
    _save_output(kubectl, output_dir / "driver.log", "logs", f"job/{RAYJOB}")


def _check_staged_data(
    kubectl: Kubectl, plan: SlimeTrainingPlan, output_dir: Path
) -> None:
    """The setup Job's log ends with a listing of what it staged; the trainer's
    checkpoints and dataset must be in it."""
    log = kubectl.logs(f"job/{SETUP_JOB}")
    (output_dir / "setup-job.log").write_text(log, encoding="utf-8")
    listing = parse_ls_listing(log)
    missing = missing_paths(listing, plan.staged_paths)
    assert not missing, f"The setup Job did not stage {missing}. It listed: {listing}"


def _served_models(body: str) -> set[str]:
    try:
        return parse_served_model_ids(body)
    except AssertionError:
        return set()


async def _check_trainer_addresses(
    kubectl: Kubectl,
    plan: SlimeTrainingPlan,
    engine_pod_selector: str,
    model: str,
) -> None:
    """Reach the trainer's addresses from inside the namespace, as the
    tutorial does from an engine Pod."""
    pod = kubectl.pod(engine_pod_selector)
    urls = [f"http://{address}/health_generate" for address in plan.engine_addresses]
    urls.append(f"{plan.rollout_url}/health")
    statuses = {url: kubectl.get_in_cluster(pod, url)[0] for url in urls}
    assert all(
        status == "200" for status in statuses.values()
    ), f"Not every address that the trainer uses answered 200: {statuses}"

    # Beyond the tutorial: wait until the frontend lists the model, so that a
    # slow engine registration cannot fail the first rollout.
    deadline = time.monotonic() + MODEL_LIST_TIMEOUT
    while True:
        status, body = kubectl.get_in_cluster(pod, f"{plan.rollout_url}/v1/models")
        if status == "200" and model in _served_models(body):
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"The frontend did not list {model} within {MODEL_LIST_TIMEOUT}s. "
                f"Last response, {status}: {body[:500]}"
            )
        await asyncio.sleep(5)


def _check_routing(kubectl: Kubectl, plan: SlimeTrainingPlan) -> None:
    """The tutorial's routing check: the frontend logs a "Selected worker"
    line for each request that it routes, round-robin across the engines."""
    frontend = kubectl.pod(f"{ROLE_LABEL}={FRONTEND_ROLE}")
    routed = routed_requests(kubectl.logs(frontend, "--container=main"))
    assert routed.total == plan.rollout_requests, (
        f"The frontend routed {routed.total} requests; the RayJob's flags make "
        f"{plan.rollout_requests}"
    )
    assert routed.router_modes == {"round-robin"}, routed.router_modes
    engines = len(plan.engine_addresses)
    assert (
        sorted(routed.by_worker.values())
        == [plan.rollout_requests // engines] * engines
    ), (
        f"Round-robin should split the {plan.rollout_requests} requests evenly "
        f"across {engines} engines: {dict(routed.by_worker)}"
    )


def _check_weight_reloads(
    kubectl: Kubectl, plan: SlimeTrainingPlan, roles: list[str]
) -> None:
    """The tutorial's weight-update check: each engine reloads once before the
    first rollout and once after each rollout's training."""
    reloads = {
        role: count_weight_reloads(
            kubectl.logs(kubectl.pod(f"{ROLE_LABEL}={role}"), "--container=main")
        )
        for role in roles
    }
    assert reloads == dict.fromkeys(
        roles, plan.weight_syncs
    ), f"Each engine should load {plan.weight_syncs} weight updates: {reloads}"


def _check_driver_log(
    kubectl: Kubectl, plan: SlimeTrainingPlan, output_dir: Path
) -> None:
    """The milestones that the tutorial's training-log step points out."""
    log = kubectl.logs(f"job/{RAYJOB}")
    (output_dir / "driver.log").write_text(log, encoding="utf-8")
    summary = summarize_driver_log(log)

    expected_urls = sorted(f"http://{address}" for address in plan.engine_addresses)
    assert (
        sorted(summary.engine_urls) == expected_urls
    ), f"Slime detected engines {summary.engine_urls}, expected {expected_urls}"
    assert summary.rollouts == set(range(plan.num_rollout)), (
        f"The driver log has rollout metrics for {sorted(summary.rollouts)}, "
        f"expected rollouts 0 to {plan.num_rollout - 1}"
    )
    diffs = summary.logprob_abs_diff_by_step
    assert sorted(diffs) == list(range(plan.train_steps)), (
        f"The driver log has step metrics for {sorted(diffs)}, "
        f"expected steps 0 to {plan.train_steps - 1}"
    )
    # Written as "not <=" so that a missing value, read as NaN, also fails.
    far = {
        step: diff for step, diff in diffs.items() if not diff <= MAX_LOGPROB_ABS_DIFF
    }
    assert not far, (
        "The engines' token log-probabilities differ from the trainer's by more "
        f"than {MAX_LOGPROB_ABS_DIFF} at steps {far}"
    )
