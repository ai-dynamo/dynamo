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

import asyncio
import copy
import logging
import os
import signal
import sys

import torch
import uvloop
from args import Config, configure_ports_with_etcd, overwrite_args, parse_args
from handlers import PrefillWorkerHandler
from transformers import AutoImageProcessor
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.inputs.data import TokensPrompt
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo.llm import (
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import connect
from utils.image_loader import ImageLoader
from utils.logging import check_required_workers
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
            help="Specify the type of worker. Must be one of: 'encode', 'prefill', 'decode', 'encode_prefill'",
        )
        parser.add_argument(
            "--enable-disagg",
            action="store_true",
            help="Enable disaggregated mode, where prefill and decode are handled by separate workers."
            " If not set, the '*prefill' worker type will handle both prefill and decode.",
        )

        # use endpoint_overwrite to set the default endpoint based on worker type
        def endpoint_overwrite(args):
            # default endpoint for this worker
            if args.worker_type == "prefill":
                args.endpoint = args.endpoint or "dyn://dynamo.llm.generate"
            elif args.worker_type == "decode":
                args.endpoint = args.endpoint or "dyn://dynamo.decode.generate"
            elif args.worker_type == "encode_prefill":
                args.endpoint = args.endpoint or "dyn://dynamo.encode.generate"
            # set downstream endpoint for disaggregated workers
            if args.enable_disagg:
                args.downstream_endpoint = (
                    args.downstream_endpoint or "dyn://dynamo.decode.generate"
                )

            return args

        args, config = base_parse_args(parser, endpoint_overwrite)

        return args, config

    def __init__(self, args: argparse.Namespace, engine_args: AsyncEngineArgs):
        self.enable_disagg = args.enable_disagg
        self.downstream_endpoint = args.downstream_endpoint
        self.engine_args = engine_args
        self.setup_vllm_engine()

    def setup_vllm_engine(self):
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
        factory = [
            StatLoggerFactory(component, self.engine_args.data_parallel_rank or 0)
        ]

        self.engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=factory,
            disable_log_requests=engine_args.disable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

        # TODO Hack to get data, move this to registering in ETCD
        factory[0].set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
        factory[0].set_request_total_slots_all(
            vllm_config.scheduler_config.max_num_seqs
        )
        factory[0].init_publish()

        # TODO: We start off with a valid endpoint, then we increment it by dp_rank
        # May no longer be valid. Lets remove the increment behavior from vLLM and here
        zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
            self.engine_args.kv_events_config.endpoint,
            data_parallel_rank=self.engine_args.data_parallel_rank or 0,
        ).replace("*", "127.0.0.1")

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=generate_endpoint.lease_id(),
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
        )
        self.kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)

        logger.info(f"Reading Events from {zmq_endpoint}")

        logger.info(f"VllmWorker for {self.engine_args.model} has been initialized")
        return vllm_config

    async def async_init(self):
        pass


# [gluo WIP]
class VllmDecodeWorker(VllmBaseWorker):
    async def async_init(self):
        await super().async_init()
        logger.info("VllmDecodeWorker has been initialized")

    @endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(
            f"Received generate request in DecodeWorker: {{ id: {request.request_id} }}."
        )

        # Decode worker doesn't process embeddings, so we pass None or empty tensor
        gen = self.engine_client.generate(
            # prompt=request.engine_prompt,
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                # multi_modal_data={"image": None}
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


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmPDWorker(VllmBaseWorker):
    decode_worker = depends(VllmDecodeWorker)

    @async_on_start
    async def async_init(self):
        await super().async_init()

        if self.enable_disagg:
            runtime = dynamo_context["runtime"]
            comp_ns, comp_name = VllmDecodeWorker.dynamo_address()  # type: ignore
            self.decode_worker_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("generate")
                .client()
            )
            await check_required_workers(self.decode_worker_client, self.min_workers)

        EMBEDDINGS_DTYPE = torch.float16
        EMBEDDINGS_DEVICE = "cpu"
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()

        # embeddings_shape, self.embeddings_dtype = get_vision_embeddings_info(
        #     self.engine_args.model, self.engine_args.num_patches
        # )
        embeddings_shape = (1, 577, 4096)
        logger.debug(f"Embeddings shape: {embeddings_shape}")
        self.embedding_size = embeddings_shape[1]

        embeddings = torch.empty(
            embeddings_shape, dtype=EMBEDDINGS_DTYPE, device=EMBEDDINGS_DEVICE
        )

        descriptor = connect.Descriptor(embeddings)

        # Register the descriptor w/ NIXL (this is optional, if not done here the connect subsytem will take care of this automatically).
        # descriptor.register_memory(self._connector)
        self._embeddings_descriptor = (embeddings, descriptor)

        self.image_loader = ImageLoader()
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.engine_args.model, trust_remote_code=True
        )

        logger.info("VllmPDWorker has been initialized")

    @endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(
            f"Received generate request in PDWorker: {{ id: {request.request_id} }}."
        )

        if request.image_url is None:
            # Process embeddings using the connector
            embeddings, descriptor = self._embeddings_descriptor

            if descriptor is None:
                logger.error("in PD worker, descriptor is None")

            read_op = await self._connector.begin_read(
                request.serialized_request, descriptor
            )
            await read_op.wait_for_completion()
            logger.debug(f"in PD worker, image features: {embeddings}")
            multi_modal_data = embeddings
        else:
            # Use PIL image instead of image embeddings
            multi_modal_data = await self.image_loader.load_image(request.image_url)
            # multi_modal_data = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(dtype=torch.float16)
            # image input is expected to be (image_num, channel, height, width)
            # logger.info(f"Image features shape: {multi_modal_data.shape}")
            # multi_modal_data = multi_modal_data.unsqueeze(0)

        # Remove the image features from the request as they are not required
        request.image_url = None
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
                multi_modal_data={"image": multi_modal_data},
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
                # logger.debug(f"Prefill response: {prefill_response}")
                # request_output = MyRequestOutput.model_validate_json(prefill_response.model_dump_json())
                logger.debug(
                    f"Prefill response kv_transfer_params: {prefill_response.kv_transfer_params}"
                )
                extra_args = decode_request.sampling_params.extra_args or {}
                extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
                extra_args.pop("serialized_request", None)
                decode_request.sampling_params.extra_args = extra_args
                logger.debug("Decode request: %s", decode_request)
                async for decode_response in await self.decode_worker_client.round_robin(
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
    config = parse_args()

    etcd_client = runtime.etcd_client()
    await configure_ports_with_etcd(config, etcd_client)
    overwrite_args(config)

    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    if config.is_prefill_worker:
        await init_prefill(runtime, config)
    else:
        await init(runtime, config)


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    engine_client, _, default_sampling_params = setup_vllm_engine(config)

    # TODO register_prefill in similar vein to register_llm

    handler = PrefillWorkerHandler(component, engine_client, default_sampling_params)

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    prefill_worker_client = (
        await runtime.namespace(config.namespace)
        .component("prefill")  # TODO don't hardcode
        .endpoint("generate")
        .client()
    )

    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        await register_llm(
            ModelType.Backend,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.engine_args.block_size,
        )

    logger.info(f"VllmWorker for {config.model} has been initialized")

    # [gluo WIP]
    VllmPDWorker()

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        # Cleanup background tasks
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
