# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import base64
import logging
import os
import time
import uuid
from pathlib import Path

import uvloop
import yaml
from dynamo.runtime import DistributedRuntime, dynamo_endpoint
from dynamo.llm import ModelInput, ModelType, register_model

from fastvideo import VideoGenerator

from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
    VideoNvExt,
)
from dynamo.common.utils.graceful_shutdown import install_signal_handlers

logger = logging.getLogger(__name__)


class FastVideoBackend:
    def __init__(self, config: dict) -> None:
        logger.info("Starting FastVideo backend")

        self.generator: VideoGenerator | None = None
        self.model_name = config.get("Backend", {}).get(
            "model_name", "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
        )
        self.num_gpus = config.get("Backend", {}).get("num_gpus", 1)
        self.video_storage_path = Path(
            config.get("Backend", {}).get("video_storage_path", "/videos")
        )

        # Set FastVideo environment variable for attention backend
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = config.get("Backend", {}).get(
            "attention_backend", "TORCH_SDPA"
        )

        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

        self.video_storage_path.mkdir(parents=True, exist_ok=True)

    async def initialize_model(self):
        logger.info("Loading FastVideo model: %s", self.model_name)

        try:
            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self.generator = await loop.run_in_executor(
                None,
                lambda: VideoGenerator.from_pretrained(
                    self.model_name, num_gpus=self.num_gpus
                ),
            )

            logger.info("FastVideo backend initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize FastVideo model: %s", e, exc_info=True)
            raise

    async def generate_video(self, request: NvCreateVideoRequest) -> VideoData:
        if self.generator is None:
            raise ValueError("Generator is not initialized")

        logger.info("Generating video for prompt: %s", request.prompt[:50])
        start_time = time.monotonic()

        try:
            video_path = self.video_storage_path / f"{uuid.uuid4()}.mp4"
            
            # Use default size if not provided
            size = request.size or "832x480"
            width, height = map(int, size.split("x"))

            # Use default nvext if not provided
            nvext = request.nvext
            if nvext is None:
                nvext = VideoNvExt(
                    num_frames=49,
                    fps=24,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    seed=int(time.time()),
                )

            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                lambda: self.generator.generate_video(
                    prompt=request.prompt,
                    output_path=str(video_path),
                    save_video=True,
                    height=height,
                    width=width,
                    num_frames=(
                        nvext.num_frames
                        if request.seconds is None or request.seconds == 0
                        else request.seconds * (nvext.fps or 24)
                    ),
                    fps=nvext.fps or 24,
                    num_inference_steps=nvext.num_inference_steps or 50,
                    guidance_scale=nvext.guidance_scale or 5.0,
                    seed=nvext.seed or int(time.time()),
                ),
            )

            generation_time = time.monotonic() - start_time
            logger.info("Video generation completed in %.2fs", generation_time)

            # Return response based on format
            if request.response_format == "url":
                return VideoData(
                    url=str(video_path),
                    b64_json=None,
                )
            else:  # b64_json
                video_b64 = base64.b64encode(video_path.read_bytes()).decode()
                return VideoData(
                    url=None,
                    b64_json=video_b64,
                )

        except Exception as e:
            logger.error("Error generating video: %s", e, exc_info=True)
            raise

    @dynamo_endpoint(NvCreateVideoRequest, NvVideosResponse)
    async def create_video(self, request: NvCreateVideoRequest):
        logger.info("Received video generation request")

        try:
            video_data = await self.generate_video(request)

            # Generate a unique ID for this video generation (OpenAI Sora format)
            video_id = f"video_{uuid.uuid4().hex}"

            response = NvVideosResponse(
                id=video_id,
                object="video",
                created=int(time.time()),
                model=request.model,  # Echo back the requested model
                status="completed",
                data=[video_data],
            )

            logger.info("Request completed successfully with ID: %s", video_id)
            yield response.model_dump()

        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            raise


async def backend_worker(
    runtime: DistributedRuntime,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: list,
):
    namespace_name = "dynamo"
    component_name = "backend"
    endpoint_name = "generate"

    component = runtime.namespace(namespace_name).component(component_name)
    logger.info("Initialized component %s/%s", namespace_name, component_name)

    endpoint = component.endpoint(endpoint_name)
    logger.info("Serving endpoint %s", endpoint_name)

    # Register endpoint for graceful shutdown
    shutdown_endpoints.append(endpoint)

    backend = FastVideoBackend(_get_config())
    await backend.initialize_model()

    # Register the model with Dynamo's discovery system
    ready_event = asyncio.Event()

    try:
        await asyncio.gather(
            endpoint.serve_endpoint(backend.create_video, graceful_shutdown=True),
            _register_model(endpoint, backend.model_name, ready_event),
        )
    except Exception as e:
        logger.error("Failed to serve endpoint: %s", e, exc_info=True)
        raise


async def _register_model(
    endpoint, model_name: str, readiness_gate: asyncio.Event = None
):
    try:
        await register_model(
            ModelInput.Text,  # Video models accept text prompts
            ModelType.Videos,  # Output type is videos
            endpoint,
            model_name,  # model_path
            model_name,  # served_model_name
        )
        logger.info("Successfully registered model: %s", model_name)
    except Exception as e:
        logger.error("Failed to register model: %s", e, exc_info=True)
        raise RuntimeError("Model registration failed")

    if readiness_gate:
        readiness_gate.set()

    logger.info("Model ready: %s", model_name)


def _get_config() -> dict:
    config_path = Path(os.environ.get("FASTVIDEO_CONFIG_PATH", "config.yaml"))
    with config_path.open() as f:
        return yaml.safe_load(f)


async def main():
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "tcp", False)

    shutdown_event = asyncio.Event()
    shutdown_endpoints = []

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    await backend_worker(runtime, shutdown_event, shutdown_endpoints)


if __name__ == "__main__":
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("FASTVIDEO_LOG_LEVEL") == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    uvloop.install()
    asyncio.run(main())
