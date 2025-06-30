# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from contextlib import contextmanager
from multiprocessing import Process

import psutil
import pytest

from tests.fault_tolerance.client import client
from tests.fault_tolerance.parse_results import main as parse_results
from tests.fault_tolerance.utils.circus_controller import CircusController
from tests.serve.test_dynamo_serve import DynamoServeProcess
from tests.utils.managed_process import terminate_process_tree


def _set_deployment_args(request, max_num_seqs):
    decode_worker_name = "VllmWorker"
    args = {}

    if max_num_seqs is not None:
        args[f"--{decode_worker_name}.max_num_seqs"] = max_num_seqs

    return args


def _list_vllm_worker_processes():
    processes = []
    for ps_process in psutil.process_iter(["name", "cmdline"]):
        try:
            if "from multiprocessing.spawn import spawn_main;" in " ".join(
                ps_process.cmdline()
            ):
                processes.append(ps_process.pid)
        except Exception:
            pass
    return processes


@contextmanager
def _clients(
    logger,
    num_clients,
    request,
    deployment_graph,
    server_process,
    payload,
    requests_per_client,
    input_token_length,
    output_token_length,
    max_retries,
):
    procs = []
    for i in range(num_clients):
        procs.append(
            Process(
                target=client,
                args=(
                    deployment_graph,
                    server_process,
                    payload,
                    request.node.name,
                    i,
                    requests_per_client,
                    input_token_length,
                    output_token_length,
                    max_retries,
                ),
            )
        )
        procs[-1].start()
    yield procs

    for proc in procs:
        logger.debug(f"{proc} waiting for join")
        proc.join()
        logger.debug(f"{proc} joined")


def _inject_failures(failures, logger):
    circus_controller = CircusController.from_state_file("dynamo")

    for failure_time, component in failures:
        time.sleep(failure_time)
        for component_name, number in component:
            logger.info(f"Injecting failure for: {component_name}")

            if "dynamo" in component_name:
                result = circus_controller.client.call(
                    {"command": "list", "properties": {"name": f"{component_name}"}}
                )
                if result["status"] == "error":
                    logger.warning(f"component {component_name} not found {result}")
                    continue

                num_processes = len(result["pids"])
                if number is None:
                    number = num_processes
                for x in range(number):
                    pid = result["pids"][x % num_processes]
                    logger.info(f"Terminating {component_name} Pid {pid}")
                    terminate_process_tree(pid, logger, immediate_kill=True)
            elif "vllm" in component_name:
                vllm_processes = _list_vllm_worker_processes()
                num_processes = len(vllm_processes)
                if number is None:
                    number = len(vllm_processes)
                for x in range(number):
                    pid = vllm_processes[x % num_processes]
                    terminate_process_tree(pid, logger, immediate_kill=True)

    circus_controller.close()


global_result_list = []


@pytest.fixture(autouse=True)
def results_table(request):
    yield
    parse_results(logs_dir=None, log_paths=[request.node.name], tablefmt="fancy")
    global_result_list.append(request.node.name)


@pytest.fixture(autouse=True, scope="session")
def results_summary():
    yield
    parse_results(logs_dir=None, log_paths=global_result_list, tablefmt="fancy")


@pytest.mark.e2e
@pytest.mark.slow
async def test_worker_failure(
    deployment_graph_test,
    request,
    runtime_services,
    num_clients,
    requests_per_client,
    worker_metrics,
    respawn,
    failures,
    input_token_length,
    output_token_length,
    max_num_seqs,
    max_retries,
    display_dynamo_output,
    nvidia_smi,
    separate_process_logs,
    hf_hub_offline,
):
    """
    Test dynamo serve deployments with injected failures
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")
    deployment_graph, payload = deployment_graph_test

    if hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        if "HF_HUB_OFFLINE" in os.environ:
            del os.environ["HF_HUB_OFFLINE"]
    if respawn:
        os.environ["DYN_CIRCUS_RESPAWN"] = "1"
    else:
        if "DYN_CIRCUS_RESPAWN" in os.environ:
            del os.environ["DYN_CIRCUS_RESPAWN"]

    if separate_process_logs:
        os.environ["DYN_CIRCUS_LOG_DIR"] = os.path.abspath(request.node.name)

    deployment_args = _set_deployment_args(request, max_num_seqs)

    with DynamoServeProcess(
        deployment_graph,
        request,
        display_output=display_dynamo_output,
        args=deployment_args,
    ) as server_process:
        server_process.wait_for_ready(payload)

        with _clients(
            logger,
            num_clients,
            request,
            deployment_graph,
            server_process,
            payload,
            requests_per_client,
            input_token_length,
            output_token_length,
            max_retries,
        ):
            _inject_failures(failures, logger)
