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
import subprocess
import tempfile

import yaml

from dynamo.planner.config import ServiceConfig
from dynamo.planner.defaults import SLAPlannerDefaults
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Initialize and run Prometheus server with Dynamo config."""
    config = ServiceConfig.get_parsed_config("Prometheus")

    logger.info(f"Prometheus config: {config}")

    await start_prometheus_server(config)


async def start_prometheus_server(config):
    logger.info("Starting prometheus server...")

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    yaml.dump(config, temp_file)
    temp_file.close()
    config_path = temp_file.name

    prometheus_port = SLAPlannerDefaults.port
    cmd = [
        "prometheus",
        f"--config.file={config_path}",
        f"--web.listen-address=0.0.0.0:{prometheus_port}",
    ]

    logger.info(f"Prometheus cmd: {cmd}")

    process = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
    )

    # Keep the worker running
    try:
        while True:
            await asyncio.sleep(1)
            if process.poll() is not None:
                logger.error("Prometheus process died")
                break
    except asyncio.CancelledError:
        logger.info("Shutting down Prometheus...")
        process.terminate()
        process.wait()
        raise


if __name__ == "__main__":
    # The dynamo_worker decorator handles runtime setup
    import asyncio

    asyncio.run(worker())
