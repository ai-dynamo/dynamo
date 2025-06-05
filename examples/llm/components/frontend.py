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
import subprocess
from pathlib import Path

from components.planner_service import Planner
from components.processor import Processor
from components.worker import VllmWorker
from pydantic import BaseModel

from dynamo import sdk
from dynamo.sdk import api, depends, on_shutdown, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)

# TODO: temp workaround to avoid port conflict with subprocess HTTP server; remove this once ingress is fixed
os.environ["DYNAMO_PORT"] = "3999"


def get_http_binary_path():
    """Find the HTTP binary path in SDK or fallback to 'http' command."""
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/http"
    if not binary_path.exists():
        return "http"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    """Configuration for the Frontend service including model and HTTP server settings."""

    served_model_name: str
    endpoint: str
    port: int = 8080


# todo this should be called ApiServer
@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Frontend:
    planner = depends(Planner)
    worker = depends(VllmWorker)
    processor = depends(Processor)

    def __init__(self):
        """Initialize Frontend service with HTTP server and model configuration."""
        frontend_config = FrontendConfig(**ServiceConfig.get_parsed_config("Frontend"))
        self.frontend_config = frontend_config
        self.process = None
        self.setup_model()
        self.start_http_server()

    def setup_model(self):
        """Configure the model for HTTP service using llmctl."""
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                self.frontend_config.served_model_name,
            ],
            check=False,
        )
        subprocess.run(
            [
                "llmctl",
                "http",
                "add",
                "chat-models",
                self.frontend_config.served_model_name,
                self.frontend_config.endpoint,
            ],
            check=False,
        )

    def start_http_server(self):
        """Start the HTTP server on the configured port."""
        logger.info("Starting HTTP server")
        http_binary = get_http_binary_path()

        self.process = subprocess.Popen(
            [http_binary, "-p", str(self.frontend_config.port)],
            stdout=None,
            stderr=None,
        )

    @api()
    def dummy_api(self) -> None:
        """
        Dummy API to enable the HTTP server for the Dynamo operator.
        This API is not used by the model.

        NOTE: this is a temporary solution to expose ingress
        for the LLM examples. Will be fixed and removed in the future.
        The resulting api_endpoints in dynamo.yaml will be incorrect.
        """

    @on_shutdown
    def cleanup(self):
        """Clean up resources before shutdown."""

        # circusd manages shutdown of http server process, we just need to remove the model using the on_shutdown hook
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                self.frontend_config.served_model_name,
            ],
            check=False,
        )
