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
    "agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="configs/disagg.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "mock_agg": (
        DeploymentGraph(
            module="graphs.mock_agg:Frontend",
            config="configs/mock_agg.yaml",
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
    "mock_decode_worker": [[10,[("dynamo_mockvllmworker",1)]]],
    "none":[],
    "mock_none":[]
}
@pytest.fixture(
    params=["none", "decode_worker","prefill_worker","frontend","processor"]
)
def failures(request):
    scenario = failure_scenarios[request.param]
    if "mock" in request.node.name:
        param = "mock_" + request.param
    else:
        param = request.param
    return failure_scenarios[param]

@pytest.fixture(
    params=[
        pytest.param("agg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
        pytest.param("disagg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
        pytest.param("mock_agg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),

    ]
)
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_graphs[request.param]


def _terminate_process(process):
    try:
        logging.info("Terminating %s", process)
        process.terminate()
    except psutil.AccessDenied:
        logging.warning("Access denied for PID %s", process.pid)
    except psutil.NoSuchProcess:
        logging.warning("PID %s no longer exists", process.pid)
    except psutil.TimeoutExpired:
        logging.warning("PID %s did not terminate before timeout, killing", process.pid)
        process.kill()


def _terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            _terminate_process(child)
        _terminate_process(parent)
    except psutil.NoSuchProcess:
        # Process already terminated
        pass


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
            start_time = time.time()
            elapsed = 0.0

            for i in range(requests_per_client):
                result=_single_request(url, payload.payload_chat, logger,
                                       max_retries,
                                       input_token_length=input_token_length,
                                       output_token_length=output_token_length,
                                       retry_delay=retry_delay)
                logger.info(f"Request: {i} Status: {result['results'][-1]['status']}")
          
                log.write(json.dumps(result)+"\n")
                log.flush()
    except Exception as e:
        print(e)
    logger.info("Exiting")
    

def run_metrics_process(log_dir):
    asyncio.run(get_metrics(log_dir))

@dynamo_worker()
async def get_metrics(runtime,log_dir):
    # Log # processes
    # Log # metrics per vllm worker
    circus_controller = None
    pipeline = None
    log_path = os.path.join(log_dir,"watcher.log.txt")
    with open(log_path,"w") as log:
        while True:
            try:
                await asyncio.sleep(0.5)

                if not circus_controller:
                    circus_controller = CircusController.from_state_file("dynamo")
                if not pipeline:
                        pipeline = (
                            await runtime.namespace("dynamo")
                            .component("VllmWorker")
                            .endpoint("load_metrics")
                            .client()
                        )

                watchers = []
                for x in await circus_controller._list_watchers():
                    result = circus_controller.client.call(
                        {"command": "list", "properties": {"name": f"{x}"}}
                    )
                    watchers.append((x, result))
           
                metrics = []
                for x in pipeline.instance_ids():
                    async for worker_metric in await pipeline.direct(None,x):
                        metrics.append((x,worker_metric.data()))
                record = {"time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                          "watchers":watchers,
                          "metrics":metrics}
                log.write(json.dumps(record)+"\n")
                log.flush()
            except Exception as e:
                pass


@pytest.fixture
def worker_metrics(request):
    print(request.node.name)
    process = Process(target=run_metrics_process,
                      args=(request.node.name,))
    process.start()
    yield
    process.kill()
    

def _set_deployment_args(request,
                         max_num_seqs):

    decode_worker_name = "VllmWorker"
    if "mock" in request.node.name:
        decode_worker_name = "MockVllmWorker"
    args = {}

    if max_num_seqs is not None:
        args[f"--{decode_worker_name}.max_num_seqs"] = max_num_seqs
    
    return args

    
@pytest.mark.e2e
@pytest.mark.slow
async def test_worker_failure(
        deployment_graph_test, request, runtime_services, num_clients, requests_per_client, worker_metrics,respawn, failures,
        input_token_length,output_token_length, max_num_seqs, max_retries,display_dynamo_output
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
                    _terminate_process_tree(pid)
                    
        for proc in procs:
            logger.info(f"{proc} waiting for join")
            proc.join()
            logger.info(f"{proc} joined")


        circus_controller.close()
