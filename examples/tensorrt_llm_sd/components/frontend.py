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

from components.worker import TensorRTLLMWorker
from fastapi import FastAPI
from pydantic import BaseModel

from dynamo import sdk
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
    port: int = 8000
    router: str = "round-robin"
    block_size: int = 32


# todo this should be called ApiServer
@service(
    dynamo={
        "namespace": "dynamo",
    },
    workers=1,
    image=DYNAMO_IMAGE,
    app=FastAPI(title="TensorRT-LLM Example"),
)
class Frontend:
    worker = depends(TensorRTLLMWorker)

    def __init__(self):
        """Initialize Frontend service with HTTP server and model configuration."""
        self.frontend_config = FrontendConfig(
            **ServiceConfig.get_parsed_config("Frontend")
        )
        self.process = None

        logger.warning(f"Frontend config: {self.frontend_config}")

        self.start_ingress_and_processor()

    def start_ingress_and_processor(self):
        """Starting dynamo-run based ingress and processor"""
        logger.info(
            f"Starting HTTP server and processor on port {self.frontend_config.port}"
        )
        dynamo_run_binary = get_dynamo_run_binary()

        cmd = [
            dynamo_run_binary,
            "in=http",
            "out=dyn",
            "--http-port",
            str(self.frontend_config.port),
            "--router-mode",
            self.frontend_config.router,
        ]

        logger.info(f"Frontend cmd: {cmd}")

        self.process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
        )

    def close(self):
        """Clean up resources by terminating the subprocess."""
        if self.process is not None:
            try:
                logger.info("Terminating subprocess...")
                self.process.terminate()
                # Wait for process to terminate with a timeout
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess did not terminate gracefully, forcing kill")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error while terminating subprocess: {e}")
            finally:
                self.process = None

    def __del__(self):
        """Destructor to ensure subprocess is cleaned up."""
        self.close()
