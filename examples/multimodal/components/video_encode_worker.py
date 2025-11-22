# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os
import signal
import sys
from io import BytesIO
from queue import Queue
from typing import AsyncIterator, Optional, Tuple

import av
import numpy as np
import torch
import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.args import Config, base_parse_args, parse_endpoint
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest
from utils.video_utils import (
    calculate_frame_sampling_indices,
    get_video_metadata,
    load_video_content,
    open_video_container,
    prepare_tensor_for_rdma,
    read_video_pyav,
    resize_video_frames,
)

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
        self.min_workers = 1

        # Video processing parameters
        self.num_frames_to_sample = args.num_frames_to_sample
        self.frame_height = 336
        self.frame_width = 336
        self.frame_channels = 3
        self._video_content_cache: dict[str, BytesIO] = {}
        self._cache_queue: Queue[str] = Queue(maxsize=CACHE_SIZE_MAXIMUM)

        self._http_timeout = 60.0

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
        video_url = request.multimodal_input.video_url

        if video_url is None:
            raise ValueError("Video URL is required.")

        container: Optional[av.container.InputContainer] = None

        try:
            video_content_stream = await load_video_content(
                video_url,
                self._video_content_cache,
                self._cache_queue,
                self._http_timeout,
            )

            # Open video container using utility function
            container = await open_video_container(video_content_stream, video_url)

            if not container or not container.streams.video:
                logger.error(f"No video stream found in {video_url}.")
                raise ValueError(f"No video stream in {video_url}.")

            # Get video metadata using utility function
            total_frames, duration_sec = get_video_metadata(container)

            # Calculate frame sampling indices using utility function
            indices = calculate_frame_sampling_indices(
                total_frames, self.num_frames_to_sample, duration_sec, video_url
            )

            if not container:
                raise ValueError(f"Container is None for {video_url}")

            # Decode video frames
            clip_np: np.ndarray = await read_video_pyav(container, indices)

            if clip_np.size == 0:
                raise ValueError(
                    f"Failed to extract any video frames from {video_url} for indices {indices.tolist()}. Clip is empty."
                )

            logger.debug(
                f"Successfully extracted {len(clip_np) if clip_np.ndim > 1 and clip_np.shape[0] > 0 else 0} frames for {video_url} with original shape {clip_np.shape}."
            )

            # Convert the NumPy array from the video decoder into a PyTorch tensor.
            # This is a required step to use PyTorch functions for GPU-accelerated image processing.
            frames_tensor_orig_res = torch.from_numpy(clip_np)  # Shape: (T, H, W, C)

            # Resize frames using utility function
            resized_frames_tensor_hwc = resize_video_frames(
                frames_tensor_orig_res, self.frame_height, self.frame_width
            )

            # Prepare tensor for RDMA using utility function
            tensor_for_descriptor = prepare_tensor_for_rdma(
                resized_frames_tensor_hwc, request_id
            )

            request.embeddings_shape = tuple(tensor_for_descriptor.shape)
            descriptor = connect.Descriptor(tensor_for_descriptor)

            with await self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.metadata()
                # Clear the image URL as hint that the image is passed as embeddings.
                request.multimodal_input.video_url = None

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
        except (
            FileNotFoundError,
            av.FFmpegError,
            ValueError,
        ) as e:
            logger.error(
                f"Error processing request {request_id} ({video_url[:100]}...): {type(e).__name__} - {e}"
            )
            raise  # Re-raise to be handled by the service framework
        except Exception as e:
            logger.exception(
                f"Unexpected error processing request {request_id} ({video_url[:100]}...): {e}"
            )
            raise
        finally:
            if container:
                await asyncio.to_thread(container.close)

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
        parser.add_argument(
            "--num-frames-to-sample",
            type=int,
            default=8,
            help="Number of frames to sample from the video. Default: 8",
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
