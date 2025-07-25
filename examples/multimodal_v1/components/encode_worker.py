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

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import AsyncIterator, Tuple

import torch
import uvloop
from transformers import AutoImageProcessor, LlavaForConditionalGeneration
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import connect
from utils.args import Config, base_parse_args, parse_endpoint
from utils.image_loader import ImageLoader
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest

configure_dynamo_logging()
logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CACHE_SIZE_MAXIMUM = 8


class VllmEncodeWorker:
    def __init__(self, args: argparse.Namespace, engine_args: AsyncEngineArgs) -> None:
        self.downstream_endpoint = args.downstream_endpoint
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        # self.vision_model = load_vision_model(self.model)
        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            self.model, device_map="auto", torch_dtype=torch.float16
        ).eval()

        self.min_workers = 1

    def cleanup(self):
        pass

    async def generate(
        self, request: vLLMMultimodalRequest
    ) -> AsyncIterator[MyRequestOutput]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id

        # The following steps encode the requested image and provided useful embeddings.
        # 1. Open the image from the provided URL.
        # 2. Process the image using the image processor.
        # 3. Run the image through the vision model's vision tower.
        # 4. Run the results of the vision tower through the multi-modal projector.
        # 5. Create a descriptor for the embeddings.
        # 6. Create a write operation using the serialized request and the descriptor.
        # 7. Await for the write operation to complete.
        # 8. Yield the encode response.

        try:
            image = await self.image_loader.load_image(request.image_url)

            logger.debug(f"Processing image for request: {{ id: {request_id} }}")
            image_embeds = self.image_processor(images=image, return_tensors="pt")
            # [gluo NOTE] The commented section is for VLM generalization support,
            # will use more generic approach once utils/model.py is fixed,
            # see utils/models.py for details.
            # # Add a batch dimension to everything
            # for item in image_embeds:
            #     image_embeds[item] = image_embeds[item].unsqueeze(0).to(DEVICE)
            # logger.debug(f"Image embeds: {image_embeds}")

            # image_grid_thw = (
            #     image_embeds["image_grid_thw"].tolist()
            #     if "image_grid_thw" in image_embeds
            #     else None
            # )
            # image_sizes = (
            #     image_embeds["image_sizes"].tolist()
            #     if "image_sizes" in image_embeds
            #     else [image.size]
            # )
            # logger.debug(
            #     f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
            # )

            # with torch.no_grad():
            #     embeddings = self.vision_model.get_multimodal_embeddings(**image_embeds)
            #     if isinstance(embeddings, tuple) or isinstance(embeddings, list):
            #         # The result multimodal_embeddings may be a list or tuple of tensors, with each
            #         # tensor corresponding to a multimodal data item (image or video).
            #         # TODO: for multi-image support, this result will contain multiple tensors.
            #         embeddings = embeddings[0].unsqueeze(0)
            #     logger.debug(
            #         f"Embeddings: {{ shape: {embeddings.shape}, dtype: {embeddings.dtype}, device: {embeddings.device}, ptr: {embeddings.data_ptr()}, elements: {{ count: {embeddings.numel()}, size: {embeddings.element_size()} }} }}."
            #     )

            with torch.no_grad():
                logger.debug(f"Vision model device: {self.vision_model.device}")
                vision_outputs = self.vision_model.vision_tower(
                    image_embeds["pixel_values"].to(self.vision_model.device)
                )
                logger.debug("Vision model completed.")

                embeddings = vision_outputs.last_hidden_state
                embeddings = self.vision_model.multi_modal_projector(embeddings)

            descriptor = connect.Descriptor(embeddings)

            with self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.to_serialized()
                # Clear the image URL as hint that the image is passed as embeddings.
                request.image_url = None

                logger.debug(f"Request: {request.model_dump_json()}")

                # Get the response generator
                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json()
                )
                await readable.wait_for_completion()

                async for response in response_generator:
                    output = MyRequestOutput.model_validate_json(response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                    ).model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Startup started.")
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            self.downstream_endpoint
        )
        self.pd_worker_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )

        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector(runtime=runtime, namespace=parsed_namespace)
        await self._connector.initialize()

        logger.info("Startup completed.")

    @classmethod
    def parse_args(cls) -> Tuple[argparse.Namespace, Config]:
        DEFAULT_ENDPOINT = "dyn://dynamo.encoder.generate"
        DEFAULT_DOWNSTREAM_ENDPOINT = "dyn://dynamo.llm.generate"

        parser = FlexibleArgumentParser(
            description="vLLM based encoder for Dynamo LLM."
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            default=DEFAULT_ENDPOINT,
            help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_ENDPOINT}'",
        )
        parser.add_argument(
            "--downstream-endpoint",
            type=str,
            default=DEFAULT_DOWNSTREAM_ENDPOINT,
            help=f"The endpoint string of the downstream LLM in 'dyn://namespace.component.endpoint' format. Default: '{DEFAULT_DOWNSTREAM_ENDPOINT}'",
        )

        args, config = base_parse_args(parser)

        return args, config


async def graceful_shutdown(runtime):
    """
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Runtime setup
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # worker setup
    args, config = VllmEncodeWorker.parse_args()
    await init(runtime, args, config)


async def init(runtime: DistributedRuntime, args: argparse.Namespace, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    handler = VllmEncodeWorker(args, config.engine_args)
    await handler.async_init(runtime)

    logger.info(f"Starting to serve the {args.endpoint} endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
