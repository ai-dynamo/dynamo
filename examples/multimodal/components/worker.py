# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import copy
import logging
import os
import signal
import sys
from typing import Tuple

import torch
import uvloop
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.inputs.data import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM

import dynamo.nixl_connect as connect
from dynamo.llm import ZmqKvEventPublisher, ZmqKvEventPublisherConfig
from dynamo.runtime import Component, DistributedRuntime, Endpoint, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from publisher import StatLoggerFactory
from utils.args import (
    Config,
    base_parse_args,
    configure_ports_with_etcd,
    overwrite_args,
    parse_endpoint,
)
from utils.image_loader import ImageLoader
from utils.model import construct_mm_data
from utils.protocol import MyRequestOutput, vLLMMultimodalRequest

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class VllmBaseWorker:
    @classmethod
    def parse_args(cls) -> Tuple[argparse.Namespace, Config]:
        parser = FlexibleArgumentParser(
            description="vLLM based encoder for Dynamo LLM."
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            help="Dynamo endpoint string in 'dyn://namespace.component.endpoint' format.  Default value will vary based on the worker type, see --worker-type for details.",
        )
        parser.add_argument(
            "--downstream-endpoint",
            type=str,
            help="The endpoint string of the downstream LLM in 'dyn://namespace.component.endpoint' format. Default value will vary based on the worker type, see --worker-type for details.",
        )
        parser.add_argument(
            "--worker-type",
            type=str,
            choices=["prefill", "decode", "encode_prefill"],
            required=True,
            help="Specify the type of worker. Must be one of: 'prefill', 'decode', 'encode_prefill'",
        )
        parser.add_argument(
            "--enable-disagg",
            action="store_true",
            help="Enable disaggregated mode, where prefill and decode are handled by separate workers."
            " If not set, the '*prefill' worker type will handle both prefill and decode.",
        )

        # use endpoint_overwrite to set the default endpoint based on worker type
        def endpoint_overwrite(args):
            DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
            # default endpoint for this worker
            if args.worker_type == "prefill":
                args.endpoint = args.endpoint or f"dyn://{DYN_NAMESPACE}.llm.generate"
            elif args.worker_type == "decode":
                args.endpoint = (
                    args.endpoint or f"dyn://{DYN_NAMESPACE}.decoder.generate"
                )
            elif args.worker_type == "encode_prefill":
                args.endpoint = (
                    args.endpoint or f"dyn://{DYN_NAMESPACE}.encoder.generate"
                )
            # set downstream endpoint for disaggregated workers
            if args.enable_disagg:
                args.downstream_endpoint = (
                    args.downstream_endpoint
                    or f"dyn://{DYN_NAMESPACE}.decoder.generate"
                )

            return args

        args, config = base_parse_args(parser, endpoint_overwrite)

        return args, config

    def __init__(
        self,
        args: argparse.Namespace,
        component: Component,
        endpoint: Endpoint,
        config: Config,
    ):
        self.enable_disagg = args.enable_disagg
        self.endpoint = args.endpoint
        self.downstream_endpoint = args.downstream_endpoint
        self.engine_args = config.engine_args
        self.config = config
        self.setup_vllm_engine(component, endpoint)

    async def async_init(self, runtime: DistributedRuntime):
        pass

    def setup_vllm_engine(self, component: Component, endpoint: Endpoint):
        """Initialize the vLLM engine.
        This method sets up the vLLM engine client, and configures the dynamo-aware KV
        event publisher and metrics stats logger based on component and endpoint.
        """

        os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Load default sampling params from `generation_config.json`
        self.default_sampling_params = (
            self.engine_args.create_model_config().get_diff_sampling_param()
        )

        # Taken from build_async_engine_client_from_engine_args()
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = self.engine_args.create_engine_config(usage_context=usage_context)

        # Create vLLM engine with metrics logger and KV event publisher attached
        self.stats_logger = StatLoggerFactory(
            component,
            self.engine_args.data_parallel_rank or 0,
            metrics_labels=[("model", self.config.model)],
        )
        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=[self.stats_logger],
            disable_log_requests=self.engine_args.disable_log_requests,
            disable_log_stats=self.engine_args.disable_log_stats,
        )

        # TODO Hack to get data, move this to registering in ETCD
        self.stats_logger.set_num_gpu_blocks_all(
            vllm_config.cache_config.num_gpu_blocks
        )
        self.stats_logger.set_request_total_slots_all(
            vllm_config.scheduler_config.max_num_seqs
        )
        self.stats_logger.init_publish()

        # TODO: We start off with a valid endpoint, then we increment it by dp_rank
        # May no longer be valid. Lets remove the increment behavior from vLLM and here
        zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
            self.engine_args.kv_events_config.endpoint,
            data_parallel_rank=self.engine_args.data_parallel_rank or 0,
        ).replace("*", "127.0.0.1")

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=endpoint.lease_id(),
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
        )
        self.kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)

        logger.info(f"Reading Events from {zmq_endpoint}")

        logger.info(f"VllmWorker for {self.engine_args.model} has been initialized")

    async def generate(self, request: vLLMMultimodalRequest):
        raise NotImplementedError(
            "This method should be implemented in subclasses to handle the generation logic."
        )

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Override in subclasses if cleanup is needed."""
        pass


class VllmDecodeWorker(VllmBaseWorker):
    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received decode request: {{ id: {request.request_id} }}.")

        # Decode worker doesn't process embeddings, so we pass None or empty tensor
        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

        async for response in gen:
            logger.debug(f"Response kv_transfer_params: {response.kv_transfer_params}")
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
                metrics=response.metrics,
                kv_transfer_params=response.kv_transfer_params,
            ).model_dump_json()


class VllmPDWorker(VllmBaseWorker):
    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Startup started.")

        if self.enable_disagg:
            (
                parsed_namespace,
                parsed_component_name,
                parsed_endpoint_name,
            ) = parse_endpoint(self.downstream_endpoint)
            self.decode_worker_client = (
                await runtime.namespace(parsed_namespace)
                .component(parsed_component_name)
                .endpoint(parsed_endpoint_name)
                .client()
            )

        if "video" in self.engine_args.model.lower():
            self.EMBEDDINGS_DTYPE = torch.uint8
        else:
            self.EMBEDDINGS_DTYPE = torch.float16

        self.EMBEDDINGS_DEVICE = "cpu"

        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        parsed_namespace, _, _ = parse_endpoint(self.endpoint)
        self._connector = connect.Connector()
        await self._connector.initialize()

        self.image_loader = ImageLoader()

        logger.info("VllmPDWorker has been initialized")

    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(f"Got raw request: {request}")
        if type(request) is not vLLMMultimodalRequest:
            if type(request) is str:
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received PD request: {{ id: {request.request_id} }}.")

        embeddings, descriptor = None, None

        # Process embeddings using the connector
        # Create a descriptor based on the embedding shape.
        embeddings = torch.empty(
            request.embeddings_shape,
            dtype=self.EMBEDDINGS_DTYPE,
            device=self.EMBEDDINGS_DEVICE,
        )
        descriptor = connect.Descriptor(embeddings)

        if (
            request.multimodal_input.image_url is None
            and request.multimodal_input.video_url is None
        ):
            if descriptor is None:
                raise RuntimeError(
                    "Descriptor is None in PD worker - cannot process embeddings"
                )

            read_op = await self._connector.begin_read(
                request.serialized_request, descriptor
            )
            await read_op.wait_for_completion()
            if "video" in self.engine_args.model.lower():
                video_numpy = embeddings.numpy()
                multi_modal_data = construct_mm_data(
                    self.engine_args.model,
                    self.EMBEDDINGS_DTYPE,
                    video_numpy=video_numpy,
                )
            else:
                multi_modal_data = construct_mm_data(
                    self.engine_args.model,
                    self.EMBEDDINGS_DTYPE,
                    image_embeds=embeddings,
                    image_grid_thw=request.image_grid_thw,
                )
        else:
            # Use PIL image instead of image embeddings
            multi_modal_data = {
                "image": await self.image_loader.load_image(
                    request.multimodal_input.image_url
                )
            }

        # Remove the image features from the request as they are not required
        request.multimodal_input.image_url = None
        request.multimodal_input.video_url = None
        request.serialized_request = None

        pd_request = copy.deepcopy(request)
        # Do prefill and remote decode if enable_disagg is true
        if self.enable_disagg:
            extra_args = pd_request.sampling_params.extra_args or {}
            extra_args["kv_transfer_params"] = {
                "do_remote_decode": True,
            }
            pd_request.sampling_params.extra_args = extra_args
            pd_request.sampling_params.max_tokens = 1
            pd_request.sampling_params.min_tokens = 1

            logger.debug("Prefill request: %s", pd_request)

        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=pd_request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=pd_request.sampling_params,
            request_id=pd_request.request_id,
        )

        if self.enable_disagg:
            decode_request = copy.deepcopy(request)
            async for prefill_response in gen:
                # Update the prompt token id in the decode request to the one
                # in response, which has image templated filled in. So that
                # the decode worker will fetch correct amount of KV blocks.
                decode_request.engine_prompt[
                    "prompt_token_ids"
                ] = prefill_response.prompt_token_ids
                logger.debug(
                    f"Prefill response kv_transfer_params: {prefill_response.kv_transfer_params}"
                )
                extra_args = decode_request.sampling_params.extra_args or {}
                extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
                extra_args.pop("serialized_request", None)
                decode_request.sampling_params.extra_args = extra_args
                logger.debug("Decode request: %s", decode_request)
                async for (
                    decode_response
                ) in await self.decode_worker_client.round_robin(
                    decode_request.model_dump_json()
                ):
                    output = MyRequestOutput.model_validate_json(decode_response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                        metrics=output.metrics,
                        kv_transfer_params=output.kv_transfer_params,
                    ).model_dump_json()

        else:
            async for response in gen:
                logger.debug(
                    f"Response kv_transfer_params: {response.kv_transfer_params}"
                )
                yield MyRequestOutput(
                    request_id=response.request_id,
                    prompt=response.prompt,
                    prompt_token_ids=response.prompt_token_ids,
                    prompt_logprobs=response.prompt_logprobs,
                    outputs=response.outputs,
                    finished=response.finished,
                    metrics=response.metrics,
                    kv_transfer_params=response.kv_transfer_params,
                ).model_dump_json()


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
    args, config = VllmBaseWorker.parse_args()

    # vLLM config overwrites
    etcd_client = runtime.do_not_use_etcd_client()
    await configure_ports_with_etcd(config, etcd_client)
    overwrite_args(config)
    await init(runtime, args, config)


async def init(runtime: DistributedRuntime, args: argparse.Namespace, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    if args.worker_type in ["prefill", "encode_prefill"]:
        handler: VllmBaseWorker = VllmPDWorker(
            args, component, generate_endpoint, config
        )
    elif args.worker_type == "decode":
        handler = VllmDecodeWorker(args, component, generate_endpoint, config)
    await handler.async_init(runtime)

    logger.info(f"Starting to serve the {args.endpoint} endpoint...")

    metrics_labels = [("model", config.model)]

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=metrics_labels
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks, metrics_labels=metrics_labels
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
