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
import subprocess
import tempfile

import yaml

from dynamo.sdk import service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    workers=1,
    image=DYNAMO_IMAGE,
)
class Prometheus:
    def __init__(self):
        """Initialize Frontend service with HTTP server and model configuration."""
        self.config = ServiceConfig.get_parsed_config("Prometheus")
        self.process = None

        logger.info(f"Prometheus config: {self.config}")

        self.start_prometheus_server()

    def start_prometheus_server(self):
        logger.info("Starting prometheus server...")

        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        )
        yaml.dump(self.config, self.temp_file)
        self.temp_file.close()
        config_path = self.temp_file.name

        cmd = [
            "prometheus",
            f"--config.file={config_path}",
        ]

        logger.info(f"Prometheus cmd: {cmd}")

        self.process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
        )
