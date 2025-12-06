# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import AsyncIterator, Tuple

import uvloop
from transformers import AutoImageProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.args import Config, base_parse_args, parse_endpoint
from utils.encode_utils import encode_image_embeddings, get_encoder_components
from utils.image_loader import ImageLoader
from utils.model import load_vision_model
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
    def __init__(
        self,
        args: argparse.Namespace,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = load_vision_model(self.model)
        self.min_workers = 1

        # Get encoder components for the model
        self.vision_encoder, self.projector = get_encoder_components(
            self.model, self.vision_model
        )

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
            if not request.multimodal_input.image_url:
                raise ValueError("image_url is required for the encode worker.")

            image = await self.image_loader.load_image(
                request.multimodal_input.image_url
            )

            logger.debug(f"Processing image for request: {{ id: {request_id} }}")
            image_embeds = self.image_processor(images=image, return_tensors="pt")

            # Encode the image embeddings using model-specific encoder
            embeddings = encode_image_embeddings(
                model_name=self.model,
                image_embeds=image_embeds,
                vision_encoder=self.vision_encoder,
                projector=self.projector,
            )

            image_grid_thw = (
                image_embeds["image_grid_thw"].tolist()
                if "image_grid_thw" in image_embeds
                else None
            )
            logger.debug(
                f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
            )

            request.image_grid_thw = image_grid_thw
            request.embeddings_shape = tuple(embeddings.shape)
            descriptor = connect.Descriptor(embeddings)

            with await self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.metadata()
                # Clear the image URL as hint that the image is passed as embeddings.
                request.multimodal_input.image_url = None

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
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()

        logger.info("Startup completed.")

    @classmethod
    def parse_args(cls) -> Tuple[argparse.Namespace, Config]:
        DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
        DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.encoder.generate"
        DEFAULT_DOWNSTREAM_ENDPOINT = f"dyn://{DYN_NAMESPACE}.llm.generate"

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


@dynamo_worker()
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

    generate_endpoint = component.endpoint(config.endpoint)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        args.downstream_endpoint
    )
    pd_worker_client = (
        await runtime.namespace(parsed_namespace)
        .component(parsed_component_name)
        .endpoint(parsed_endpoint_name)
        .client()
    )

    handler = VllmEncodeWorker(args, config.engine_args, pd_worker_client)
    await handler.async_init(runtime)

    logger.info("Waiting for PD Worker Instances ...")
    await pd_worker_client.wait_for_instances()

    logger.info(f"Starting to serve the {args.endpoint} endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
