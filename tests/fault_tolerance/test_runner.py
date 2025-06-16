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


import asyncio
import logging
import random
import time
from multiprocessing import Process, Queue

import psutil
import pytest
import requests
import sys
import os
import json
from datetime import datetime

from dynamo.llm import KvMetricsAggregator
from dynamo.runtime import dynamo_worker
from tests.fault_tolerance.utils.circus_controller import CircusController
from tests.utils.managed_process import terminate_process, terminate_process_tree
from tests.fault_tolerance.utils.metrics import worker_metrics, nvidia_smi
from tests.serve.test_dynamo_serve import DynamoServeProcess
from tests.utils.deployment_graph import (
    DeploymentGraph,
    Payload,
    completions_response_handler,
    chat_completions_response_handler
)

text_prompt = "Tell me a short joke about AI."

text_payload = Payload(
    payload_chat={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [
            {
                "role": "user",
                "content": text_prompt,  # Shorter prompt
            }
        ],
        "max_tokens": 150,  
        "temperature": 0.1,
#        "seed": 10,
        "ignore_eos": True,
        "min_tokens": 150,
        "stream": False,
    },
    expected_log=[],
    expected_response=["AI"],
)

deployment_graphs = {
    "agg-tp-1-dp-1": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/agg_tp_1_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg-tp-1-dp-8": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/agg_tp_1_dp_8.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg-tp-2-dp-4": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/agg_tp_2_dp_4.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-1-dp-1-d-tp-1-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_1_dp_1_d_tp_1_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-1-dp-1-d-tp-1-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_1_dp_1_d_tp_1_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-1-dp-4-d-tp-4-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_1_dp_4_d_tp_4_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-1-dp-4-d-tp-4-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_1_dp_4_d_tp_4_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-2-dp-2-d-tp-4-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_2_dp_2_d_tp_4_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
}

failure_scenarios = {
    "decode_worker": [[10,[("dynamo_vllmworker",1)]]],
    "prefill_worker": [[10,[("dynamo_prefillworker",1)]]],
    "frontend": [[10,[("dynamo_frontend",1)]]],
    "processor": [[10,[("dynamo_processor",1)]]],
    "vllm_worker": [[10,[("vllm_worker",1)]]],
    "none":[],
}
@pytest.fixture(
    params=["none", "decode_worker","prefill_worker","frontend","processor","vllm_worker"]
)
def failures(request):
    return failure_scenarios[request.param]

@pytest.fixture(
    params=list(deployment_graphs.keys())
)
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_graphs[request.param]

def _get_random_prompt(length):
    word_list = [f"{i}" for i in range(10)]
    return " ".join(random.choices(word_list, k=length))


def _single_request(
    url,
    payload,
    logger,
    retry_attempts=1,
    input_token_length=100,
    output_token_length=100,
    timeout=30,
    retry_delay=1
):
    prompt = _get_random_prompt(input_token_length)
    payload["messages"][0]["content"] = prompt
    payload["max_tokens"] = output_token_length
    response = None
    end_time = None
    start_time = time.time()
    results = []

    while retry_attempts:
        start_request_time = time.time()
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )
            end_time = time.time()

            content = None

            try:
                content = response.json()
            except:
                pass

            results.append({"status":response.status_code, "result": content, "request_elapsed_time":end_time - start_request_time})

            if response.status_code != 200:
                time.sleep(retry_delay)
                retry_attempts -= 1
                continue
            else:
                break

        except (requests.RequestException, requests.Timeout) as e:
            results.append({"status":str(e), "result":None, "request_elapsed_time":time.time()- start_request_time})
            logger.warning("Retrying due to Request failed: %s", e)
            time.sleep(retry_delay)
            retry_attempts -= 1
            continue

    return {"time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),"results":results, "total_time": time.time() - start_time}


def client(
        deployment_graph,
        server_process,
        payload,
        log_dir,
        index,
        requests_per_client,
        input_token_length,
        output_token_length,
        max_retries,
        retry_delay=1):
    try:
        log_path = os.path.join(log_dir,f"client_{index}.log.txt")
        with open(log_path,"w") as log:
            logger = logging.getLogger(f"CLIENT: {index}")
            url = f"http://localhost:{server_process.port}/{deployment_graph.endpoints[0]}"
         
            for i in range(requests_per_client):
                result=_single_request(url, payload.payload_chat, logger,
                                       max_retries,
                                       input_token_length=input_token_length,
                                       output_token_length=output_token_length,
                                       retry_delay=retry_delay)
                logger.info(f"Request: {i} Status: {result['results'][-1]['status']} Latency: {result['results'][-1]['request_elapsed_time']}")
          
                log.write(json.dumps(result)+"\n")
                log.flush()
    except Exception as e:
        print(e)
    logger.info("Exiting")
    
def _set_deployment_args(request,
                         max_num_seqs):

    decode_worker_name = "VllmWorker"
    if "mock" in request.node.name:
        decode_worker_name = "MockVllmWorker"
    args = {}

    if max_num_seqs is not None:
        args[f"--{decode_worker_name}.max_num_seqs"] = max_num_seqs
    
    return args

def _list_vllm_worker_processes():
    processes = []
    for ps_process in psutil.process_iter(["name", "cmdline"]):
        if "from multiprocessing.spawn import spawn_main;" in " ".join(ps_process.cmdline()):
            logging.info(f"vllm worker process {ps_process} {ps_process.cmdline()}")
            processes.append(ps_process.pid)
    return processes

@pytest.mark.e2e
@pytest.mark.slow
async def test_worker_failure(
        deployment_graph_test, request, runtime_services, num_clients, requests_per_client, worker_metrics,respawn, failures,
        input_token_length,output_token_length, max_num_seqs, max_retries,display_dynamo_output, nvidia_smi
):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")
    deployment_graph, payload = deployment_graph_test
    
    if respawn:
        os.environ["DYN_CIRCUS_RESPAWN"]="1"
    else:
        if "DYN_CIRCUS_RESPAWN" in os.environ:
            del os.environ["DYN_CIRCUS_RESPAWN"]

    deployment_args = _set_deployment_args(request,
                                           max_num_seqs)
    
    with DynamoServeProcess(
            deployment_graph, request, display_output=display_dynamo_output,args=deployment_args
    ) as server_process:

        server_process.wait_for_ready(payload)
        
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
                        max_retries
                    ),
                )
            )
            procs[-1].start()

        circus_controller = CircusController.from_state_file("dynamo")

        for (failure_time,component) in failures:
            time.sleep(failure_time)
            watcher_list = await circus_controller._list_watchers()
            for component_name, number in component:
                logger.info(f"Injecting failure for: {component_name}")

                if "dynamo" in component_name:               
                    result = circus_controller.client.call(
                        {"command": "list", "properties": {"name": f"{component_name}"}}
                    )
                    if result["status"] == "error":
                        logger.warn(f"component {component_name} not found {result}")
                        continue

                    num_processes = len(result["pids"])
                    if number is None:
                        number = num_processes
                    for x in range(number):
                        pid = result["pids"][x%num_processes]
                        logger.info(f"Terminating {component_name} Pid {pid}")
                        terminate_process_tree(pid,logger)
                elif "vllm" in component_name:
                    vllm_processes = _list_vllm_worker_processes()
                    num_processes = len(vllm_processes)
                    if number is None:
                        number = len(vllm_processes)
                    for x in range(number):
                        pid = vllm_processes[x%num_processes]
                        terminate_process_tree(pid, logger)
                    
        for proc in procs:
            logger.info(f"{proc} waiting for join")
            proc.join()
            logger.info(f"{proc} joined")


        circus_controller.close()
