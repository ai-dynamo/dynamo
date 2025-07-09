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
import signal
import subprocess
import time

import pynvml
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_dynamo_env():
    """Get environment variables with Dynamo runtime configuration"""
    env = os.environ.copy()

    # Ensure ETCD and NATS are configured for Dynamo runtime
    if "ETCD_ENDPOINTS" not in env:
        logger.warning("ETCD_ENDPOINTS not set, using default localhost:2379")
        env["ETCD_ENDPOINTS"] = "localhost:2379"

    if "NATS_SERVER" not in env:
        logger.warning("NATS_SERVER not set, using default nats://localhost:4222")
        env["NATS_SERVER"] = "nats://localhost:4222"

    logger.debug(f"Using ETCD_ENDPOINTS: {env['ETCD_ENDPOINTS']}")
    logger.debug(f"Using NATS_SERVER: {env['NATS_SERVER']}")

    return env


def get_dynamo_serve_cmd(config_file_path, disaggregated=False):
    config_file_path = os.path.abspath(config_file_path)
    graph_target = "graphs.disagg:Frontend" if disaggregated else "graphs.agg:Frontend"
    return [
        "dynamo",
        "serve",
        graph_target,
        "-f",
        config_file_path,
    ]


def get_available_gpu_count():
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()

        if gpu_count > 0:
            logger.info(f"Detected {gpu_count} GPUs in the system:")
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = memory.total / (1024 * 1024)
                free_memory_mb = memory.free / (1024 * 1024)
                logger.info(
                    f"  GPU {i}: {name}, Total Memory: {total_memory_mb:.2f} MB, Free Memory: {free_memory_mb:.2f} MB"
                )
        else:
            logger.warning("No GPUs detected with pynvml.")

        pynvml.nvmlShutdown()
        return gpu_count
    except ImportError:
        logger.error(
            "pynvml module not found. Please install it with 'pip install pynvml'"
        )
        return 0
    except pynvml.NVMLError as e:
        logger.error(f"NVML Error: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return 0


def shutdown_deployment(dynamo_process):
    os.killpg(os.getpgid(dynamo_process.pid), signal.SIGINT)
    dynamo_process.communicate()

    try:
        current_pid = os.getpid()
        ps_cmd = ["ps", "-ef"]
        ps_output = subprocess.check_output(ps_cmd, text=True)
        for line in ps_output.splitlines():
            if "python" in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        if pid != current_pid:  # Exclude current process
                            os.kill(pid, signal.SIGKILL)
                    except ValueError:
                        continue
    except Exception as e:
        logger.error(f"Error killing Python processes: {e}")
    time.sleep(5)


def wait_for_server_ready(
    model_name: str,
    port: int,
    timeout: int = 300,
):
    logger.info("Waiting for the server to be ready...")
    endpoint_url = f"http://localhost:{port}/v1/chat/completions"
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            # Send a simple request to check if the server is up
            response = requests.post(
                endpoint_url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            logger.debug(f"Response: {response.text}")
            if response.status_code != 200:
                logger.info(
                    f"Server returned status code {response.status_code}, waiting..."
                )
                time.sleep(5)
                continue
            logger.info(f"Server is ready after {time.time() - start_time:.2f} seconds")
            server_ready = True
            break

        except (requests.RequestException, ConnectionError) as e:
            logger.info(f"Server not ready yet: {e}")
        time.sleep(5)

    return server_ready
