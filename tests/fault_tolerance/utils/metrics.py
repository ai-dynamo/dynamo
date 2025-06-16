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

import os
import asyncio
import pytest
from dynamo.runtime import dynamo_worker
from tests.fault_tolerance.utils.circus_controller import CircusController
import json
from datetime import datetime
from tests.utils.managed_process import ManagedProcess
from multiprocessing import Process
import psutil
import logging

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
                    
                vllm_processes = []
                
                for ps_process in psutil.process_iter(["name", "cmdline"]):
                    try:
                        if "spawn" in " ".join(ps_process.cmdline()):
                            vllm_processes.append(ps_process.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process may have terminated or become inaccessible during iteration
                        pass
                        
                record = {"time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                          "watchers":watchers,
                          "metrics":metrics,
                          "vllm_processes":vllm_processes}
                log.write(json.dumps(record)+"\n")
                log.flush()
            except Exception as e:
                pass


@pytest.fixture
def worker_metrics(request):
    process = Process(target=run_metrics_process,
                      args=(request.node.name,))
    process.start()
    yield
    process.kill()


class NvidiaSMI(ManagedProcess):
    def __init__(self, request):
        super().__init__(
            command=[
                "nvidia-smi",
                "dmon",
                "--select=puc",
            ],
            health_check_ports=[],
            terminate_existing=True,
            display_output=False,
            data_dir=None,
            log_dir=request.node.name,
        )

@pytest.fixture
def nvidia_smi(request):
    with NvidiaSMI(request) as nvidia_smi_process:
        yield nvidia_smi_process
