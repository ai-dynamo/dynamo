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
from pathlib import Path

from components.simple_load_balancer import SimpleLoadBalancer
from fastapi import FastAPI
from pydantic import BaseModel

import dynamo.sdk as sdk
from dynamo.sdk import depends, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


def get_dynamo_run_binary():
    """Find the dynamo-run binary path in SDK or fallback to 'dynamo-run' command."""
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/dynamo-run"
    if not binary_path.exists():
        return "dynamo-run"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    """Configuration for the Frontend service including model and HTTP server settings."""

    served_model_name: str
    endpoint: str
    port: int = 8080


# TODO: move these to common for all LLMs once we adopt dynamo-run
@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    workers=1,
    image=DYNAMO_IMAGE,
    app=FastAPI(title="LLM Example"),
)
class Frontend:
    worker = depends(SimpleLoadBalancer)

    def __init__(self):
        """Initialize Frontend service with HTTP server and model configuration."""
        config = ServiceConfig.get_instance()
        frontend_config = FrontendConfig(**config.get("Frontend", {}))
        self.frontend_config = frontend_config
        self.process = None

        self.start_ingress_and_processor()

    def start_ingress_and_processor(self):
        """Starting dynamo-run based ingress and processor"""
        logger.info(
            f"Starting HTTP server and processor on port {self.frontend_config.port}"
        )
        dynamo_run_binary = get_dynamo_run_binary()
        endpoint = f"dyn://{self.frontend_config.endpoint}"

        logger.info(
            f"Starting HTTP server and processor on port {self.frontend_config.port}"
        )
        logger.info(f"Endpoint: {endpoint}")

        self.process = subprocess.Popen(
            [
                dynamo_run_binary,
                "in=http",
                f"out={endpoint}",
                "--http-port",
                str(self.frontend_config.port),
            ],
            stdout=None,
            stderr=None,
        )
