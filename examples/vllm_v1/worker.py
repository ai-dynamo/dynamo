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
import logging
import os
import uuid

import uvloop
from args import Config, cmd_line_args, create_vllm_arg_map
from publisher import StatLoggerFactory
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
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

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class RequestHandler:
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, component, engine, default_sampling_params):
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    async def generate(self, request):
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])

        sampling_params = SamplingParams(**self.default_sampling_params)
        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        num_output_tokens_so_far = 0
        gen = self.engine_client.generate(prompt, sampling_params, request_id)
        async for res in gen:
            # res is vllm's RequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
            yield out
            num_output_tokens_so_far = next_total_toks


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    await init(runtime, cmd_line_args())


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    await register_llm(
        ModelType.Backend,
        generate_endpoint,
        config.model_path,
        config.model_name,
        kv_cache_block_size=config.kv_block_size,
    )

    arg_map = create_vllm_arg_map(config)
    logger.info(f"VLLM config: {arg_map}")

    engine_args = AsyncEngineArgs(**arg_map)
    # Load default sampling params from `generation_config.json`
    default_sampling_params = (
        engine_args.create_model_config().get_diff_sampling_param()
    )

    # Taken from build_async_engine_client_from_engine_args()
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    factory = StatLoggerFactory(component)
    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        stat_loggers=[factory],
        disable_log_requests=engine_args.disable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    # TODO Hack to get data, move this to registering in ETCD
    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.set_request_total_slots_all(vllm_config.scheduler_config.max_num_seqs)
    factory.init_publish()

    logger.info(f"VllmWorker for {config.model_path} has been initialized")

    base_zmq_endpoint = vllm_config.kv_events_config.endpoint
    dp_local_rank = vllm_config.parallel_config.data_parallel_rank_local

    zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
        base_zmq_endpoint, data_parallel_rank=dp_local_rank
    ).replace("*", "127.0.0.1")

    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=generate_endpoint.lease_id(),
        kv_block_size=engine_args.block_size,
        zmq_endpoint=zmq_endpoint,
    )
    _ = ZmqKvEventPublisher(component=component, config=zmq_config)

    logger.info(f"Reading Events from {zmq_endpoint}")

    handler = RequestHandler(component, engine_client, default_sampling_params)

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
