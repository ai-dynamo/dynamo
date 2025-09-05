# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
import time
from contextlib import contextmanager
from multiprocessing import Process

import pytest

from tests.fault_tolerance.deploy.client import client
from tests.fault_tolerance.deploy.parse_results import main as parse_results
from tests.fault_tolerance.deploy.scenarios import (
    Load,
    deployment_specs,
    failure_scenarios,
)
from tests.utils.managed_deployment import ManagedDeployment

multiprocessing.set_start_method("spawn")


@pytest.fixture(params=failure_scenarios.keys())
def failures(request):
    return failure_scenarios[request.param]


@pytest.fixture(params=list(deployment_specs.keys()))
def deployment_spec(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_specs[request.param]


@pytest.fixture
def load(
    max_request_rate,
    max_retries,
    num_clients,
    input_token_length,
    output_token_length,
    requests_per_client,
):
    return Load(
        num_clients,
        requests_per_client,
        input_token_length,
        output_token_length,
        max_retries,
        max_request_rate,
    )


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
    for i in range(num_clients):
        procs.append(
            Process(
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
                        process.wait()


global_result_list = []


@pytest.fixture(autouse=True)
def results_table(request, sla):
    yield
    parse_results(
        logs_dir=None, log_paths=[request.node.name], tablefmt="fancy_grid", sla=sla
    )
    global_result_list.append(request.node.name)


@pytest.fixture(autouse=True, scope="session")
def results_summary(sla):
    yield
    parse_results(
        logs_dir=None, log_paths=global_result_list, tablefmt="fancy_grid", sla=sla
    )


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_fault_scenario(
    deployment_spec,  # noqa: F811
    request,
    image,
    namespace,
    model,
    failures,  # noqa: F811
    load,
):
    """
    Test dynamo serve deployments with injected failures
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)

    deployment_spec.disable_grove()

    deployment_spec.name = "fault-tolerance-test"

    if image:
        deployment_spec.set_image(image)

    if model:
        deployment_spec.set_model(model)
    else:
        model = deployment_spec["VllmDecodeWorker"].model

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
    ) as deployment:
        with _clients(
            logger,
            load.num_clients,
            request,
            deployment_spec,
            namespace,
            model,
            load.requests_per_client,
            load.input_token_length,
            load.output_token_length,
            load.max_retries,
            load.max_request_rate,
        ):
            _inject_failures(failures, logger, deployment)
