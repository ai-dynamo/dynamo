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
import signal
import subprocess
import sys
from pathlib import Path

from components.processor import Processor
from components.worker import VllmWorker
from pydantic import BaseModel

from dynamo import sdk
from dynamo.sdk import depends, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


def get_http_binary_path():
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/http"
    if not binary_path.exists():
        return "http"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    served_model_name: str
    endpoint: str
    port: int = 8080


# todo this should be called ApiServer
@service(
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Frontend:
    worker = depends(VllmWorker)
    processor = depends(Processor)

    def __init__(self):
        config = ServiceConfig.get_instance()
        frontend_config = FrontendConfig(**config.get("Frontend", {}))
        self.frontend_config = frontend_config
        self.process = None

        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)

        # Initial setup
        self.setup_model()
        self.start_http_server()

        try:
            if self.process:
                self.process.wait()
        except KeyboardInterrupt:
            self.cleanup()

    def setup_model(self):
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                self.frontend_config.served_model_name,
            ]
        )
        # Add the model
        subprocess.run(
            [
                "llmctl",
                "http",
                "add",
                "chat-models",
                self.frontend_config.served_model_name,
                self.frontend_config.endpoint,
            ]
        )

    def start_http_server(self):
        logger.info("Starting HTTP server")
        http_binary = get_http_binary_path()
        self.process = subprocess.Popen(
            [http_binary, "-p", str(self.frontend_config.port)],
            stdout=None,
            stderr=None,
        )

    def cleanup(self):
        logger.info("Cleaning up before shutdown...")
        subprocess.run(
            [
                "llmctl",
                "http",
                "remove",
                "chat-models",
                self.frontend_config.served_model_name,
            ]
        )
        if self.process:
            logger.info("Terminating HTTP process")
            self.process.terminate()
            self.process.wait(timeout=10)

    def handle_exit(self, signum, frame):
        logger.debug(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
