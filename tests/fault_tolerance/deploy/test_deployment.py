# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
import time
from contextlib import contextmanager

import pytest

from tests.fault_tolerance.deploy.client import client
from tests.fault_tolerance.deploy.parse_results import main as parse_results
from tests.fault_tolerance.deploy.scenarios import scenarios
from tests.utils.managed_deployment import ManagedDeployment


@pytest.fixture(params=scenarios.keys())
def scenario(request):
    return scenarios[request.param]


@contextmanager
def _clients(
    logger,
    num_clients,
    request,
    deployment_spec,
    namespace,
    model,
    requests_per_client,
    input_token_length,
    output_token_length,
    max_retries,
    max_request_rate,
):
    procs = []
    ctx = multiprocessing.get_context("spawn")
    for i in range(num_clients):
        procs.append(
            ctx.Process(
                target=client,
                args=(
                    deployment_spec,
                    namespace,
                    model,
                    request.node.name,
                    i,
                    requests_per_client,
                    input_token_length,
                    output_token_length,
                    max_retries,
                    max_request_rate,
                ),
            )
        )
        procs[-1].start()
    yield procs

    for proc in procs:
        logger.debug(f"{proc} waiting for join")
        proc.join()
        logger.debug(f"{proc} joined")


def _inject_failures(failures, logger, deployment: ManagedDeployment):  # noqa: F811
    for failure in failures:
        time.sleep(failure.time)

        pods = deployment.get_pods(failure.pod_name)[failure.pod_name]

        num_pods = len(pods)

        if not pods:
            continue

        replicas = failure.replicas

        if not replicas:
            replicas = num_pods

        logger.info(f"Injecting failure for: {failure}")

        for x in range(replicas):
            pod = pods[x % num_pods]

            if failure.command == "delete_pod":
                deployment.get_pod_logs(failure.pod_name, pod, ".before_delete")
                pod.delete(force=True)
            else:
                processes = deployment.get_processes(pod)
                for process in processes:
                    if failure.command in process.command:
                        logger.info(
                            f"Terminating {failure.pod_name} Pid {process.pid} Command {process.command}"
                        )
                        process.kill(failure.signal)


global_result_list = []


@pytest.fixture(autouse=True)
def results_table(request, scenario):  # noqa: F811
    yield
    parse_results(
        logs_dir=None,
        log_paths=[request.node.name],
        tablefmt="fancy_grid",
        sla=scenario.load.sla,
    )
    global_result_list.append(request.node.name)


@pytest.fixture(autouse=True, scope="session")
def results_summary():
    yield
    parse_results(
        logs_dir=None,
        log_paths=global_result_list,
        tablefmt="fancy_grid",
    )


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_fault_scenario(
    scenario,  # noqa: F811
    request,
    image,
    namespace,
):
    """
    Test dynamo serve deployments with injected failures
    """

    logger = logging.getLogger(request.node.name)

    scenario.deployment.disable_grove()

    scenario.deployment.name = "fault-tolerance-test"

    if image:
        scenario.deployment.set_image(image)

    if scenario.model:
        scenario.deployment.set_model(scenario.model)
        model = scenario.model
    else:
        model = scenario.deployment["VllmDecodeWorker"].model

    scenario.deployment.set_logging(True, "info")

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=scenario.deployment,
    ) as deployment:
        with _clients(
            logger,
            scenario.load.clients,
            request,
            scenario.deployment,
            namespace,
            model,
            scenario.load.requests_per_client,
            scenario.load.input_token_length,
            scenario.load.output_token_length,
            scenario.load.max_retries,
            scenario.load.max_request_rate,
        ):
            _inject_failures(scenario.failures, logger, deployment)
