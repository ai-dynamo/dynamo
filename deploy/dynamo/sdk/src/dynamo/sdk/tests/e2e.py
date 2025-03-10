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

# This is a simple example of a pipeline that uses Dynamo to deploy a backend, middle, and frontend service. Use this to test
# changes made to CLI, SDK, etc

import os
from pydantic import BaseModel

from dynamo.sdk import api, depends, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
"""


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={
        "enabled": True,
        "namespace": "inference",
    },
    workers=3,
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Generate tokens."""
        req_text = req.text
        print(f"Backend received: {req_text}")
        text = f"{req_text}-back"
        for token in text.split():
            yield f"Backend: {token}"


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    dynamo={"enabled": True, "namespace": "inference"},
)
class Middle:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting middle")
        config = ServiceConfig.get_instance()
        # loading a small model via VLLM that is not used to test the arg parsing
        import sys
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.utils import FlexibleArgumentParser
        from vllm.engine.async_llm_engine import AsyncLLMEngine


        try:
            os.environ["VLLM_LOG_LEVEL"] = "DEBUG"
            # Get VLLM args using new pattern
            vllm_args = config.as_args("Middle", prefix="vllm_")
            print(f"VLLM args to parse: {vllm_args}")
            
            # Create and use parser - MUCH SIMPLER!
            parser = FlexibleArgumentParser()
            parser = AsyncEngineArgs.add_cli_args(parser)
            args = parser.parse_args(vllm_args)  # Pass args directly here!
            self.engine_args = AsyncEngineArgs.from_cli_args(args)
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        
        except ImportError:
            print("VLLM imports not available, skipping engine arg parsing")
        except Exception as e:
            print(f"Error parsing VLLM args: {e}")

    @dynamo_endpoint()
    async def generate(self, req: RequestType):
        """Forward requests to backend."""
        from vllm.sampling_params import SamplingParams

        req_text = req.text
        print(f"Middle received: {req_text}")
        text = f"{req_text}-mid"
        next_request = RequestType(text=text).model_dump_json()
        async for response in self.backend.generate(next_request):
            print(f"Middle received response: {response}")
            yield f"Middle: {response}"


@service(resources={"cpu": "1"}, traffic={"timeout": 60})  # Regular HTTP API
class Frontend:
    middle = depends(Middle)

    def __init__(self) -> None:
        print("Starting frontend")
        self.config = ServiceConfig.get_instance()

        # Get required configuration (new dict pattern)
        self.model = self.config.require("Frontend", "model")
        
        # Get optional configurations with defaults (new dict pattern)
        self.temperature = self.config.get("Frontend", {}).get("temperature", 0.7)
        self.max_tokens = self.config.get("Frontend", {}).get("max_tokens", 1024)
        self.stream = self.config.get("Frontend", {}).get("stream", True)
        
        print(f"Frontend initialized with model={self.model}, "
              f"temp={self.temperature}, max_tokens={self.max_tokens}")
        
        # Get all configs for a service (new dict pattern)
        all_frontend_configs = self.config.get("Frontend", {})
        print(f"All Frontend configs: {all_frontend_configs}")
        
        # Check other service configs (new dict pattern)
        if self.config.get("Middle", {}).get("special_mode") == "fast":
            print("Using Middle service in fast mode")

    @api
    async def generate(self, text):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        print(f"Frontend received type: {type(text)}")
        txt = RequestType(text=text)
        print(f"Frontend sending: {type(txt)}")
        async for response in self.middle.generate(txt.model_dump_json()):
            yield f"Frontend: {response}"
