# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import random
import socket
import os
import sys
from typing import Any, Dict, Optional, Union

import sglang as sgl
import uvloop
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_ip
from utils.protocol import DisaggPreprocessedRequest
from utils.sgl_utils import parse_sglang_args_inc
from utils.sgl_http_server import SglangHttpServer

from dynamo.llm import (
    ModelType,
    WorkerMetricsPublisher,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class RequestHandler:
    def __init__(
        self,
        engine: sgl.Engine,
        server_args: ServerArgs,
        component,
        decode_client: Optional[Any] = None,
    ):
        self.engine = engine
        self.server_args = server_args
        self.component = component
        self.metrics_publisher = WorkerMetricsPublisher()

        if server_args.disaggregation_mode != "null":
            self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()
            if decode_client is None:
                raise ValueError(
                    "decode_client must be provided when disaggregation_mode is not 'null'"
                )
            self.decode_client = decode_client
            logging.info(
                f"Disaggregation enabled - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
            )

        logging.info("Request handler initialized")

    def setup_metrics(self):
        """Set up metrics publisher - call this after handler creation"""
        self.metrics_publisher.publish(
            request_active_slots=0,
            request_total_slots=1024,
            kv_active_blocks=0,
            kv_total_blocks=1024,
            num_requests_waiting=0,
            gpu_cache_usage_perc=0.0,
            gpu_prefix_cache_hit_rate=0.0,
        )
        task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logging.debug("metrics publisher endpoint created")
        )

    async def create_metrics_publisher_endpoint(self):
        logging.debug("Creating metrics publisher endpoint")
        await self.metrics_publisher.create_endpoint(self.component)

    def _update_metrics(self):
        """Update metrics with current engine state"""
        # TODO: remove this once the following upstream changes are merged:
        #   • sgl-project/sglang#6721 – "Expose runtime KV-cache & request metrics"
        logging.warning(
            "Publishing placeholder metrics in SGLangWorker; these are NOT real engine metrics yet and will be replaced once upstream support lands."
        )
        self.metrics_publisher.publish(
            request_active_slots=1,
            request_total_slots=100,
            kv_active_blocks=random.randint(0, 500),
            kv_total_blocks=1000,
            num_requests_waiting=0,
            gpu_cache_usage_perc=random.uniform(0.1, 0.8),
            gpu_prefix_cache_hit_rate=random.uniform(0.0, 0.5),
        )

    def _get_bootstrap_info(self):
        """Bootstrap info from tokenizer manager"""
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_ip()

        return bootstrap_host, bootstrap_port

    def _build_sampling_params(self, request: dict) -> dict:
        sampling_params = {}
        if request["sampling_options"]["temperature"]:
            sampling_params["temperature"] = request["sampling_options"]["temperature"]
        if request["sampling_options"]["top_p"]:
            sampling_params["top_p"] = request["sampling_options"]["top_p"]
        if request["sampling_options"]["top_k"]:
            sampling_params["top_k"] = request["sampling_options"]["top_k"]
        sampling_params["max_new_tokens"] = request["stop_conditions"]["max_tokens"]
        if request["stop_conditions"]["ignore_eos"]:
            sampling_params["ignore_eos"] = request["stop_conditions"]["ignore_eos"]
        return sampling_params

    def _get_request_batch_size(self, request: dict):
        """Get batch size from request, returns None for single requests"""
        if request["batch_token_ids"] is not None:
            return len(request["batch_token_ids"])
        return None

    def _is_batch_request(self, request: dict):
        """Check if request is in batch mode"""
        return request["batch_token_ids"] is not None

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    async def generate(self, request: dict):
        is_batch = self._is_batch_request(request)
        batch_size = self._get_request_batch_size(request)

        # TODO: maintain a mapping from SGLang's Ouput struct to LLMEngineOuput
        sampling_params = self._build_sampling_params(request)

        if self.server_args.disaggregation_mode != "null":
            if is_batch:
                bootstrap_room = [
                    self._generate_bootstrap_room() for _ in range(batch_size)
                ]
                bootstrap_host = [self.bootstrap_host] * batch_size
                bootstrap_port = [self.bootstrap_port] * batch_size
            else:
                bootstrap_host = self.bootstrap_host
                bootstrap_port = self.bootstrap_port
                bootstrap_room = self._generate_bootstrap_room()

            # decode worker request
            disagg_request = DisaggPreprocessedRequest(
                request=request,
                sampling_params=sampling_params,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
            )

            # prefill response is not used
            prefill = await self.engine.async_generate(
                input_ids=request["token_ids"]
                if not is_batch
                else request["batch_token_ids"],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
            )
            prefill_task = asyncio.create_task(self._prefill_generator(prefill))

            decode = await self.decode_client.generate(disagg_request.model_dump_json())

            async for out in self._process_stream(
                decode, unpack=True, is_batch=is_batch
            ):
                yield out

            await prefill_task
        else:
            g = await self.engine.async_generate(
                input_ids=request["token_ids"]
                if not is_batch
                else request["batch_token_ids"],
                sampling_params=sampling_params,
                stream=True,
            )

            async for out in self._process_stream(g, unpack=False, is_batch=is_batch):
                yield out

    async def _process_stream(self, stream_source, unpack: bool, is_batch: bool):
        # Initialize based on batch mode
        num_output_tokens_so_far: Union[Dict[int, int], int]
        if is_batch:
            num_output_tokens_so_far = {}
        else:
            num_output_tokens_so_far = 0

        async for res in stream_source:
            data = res.data() if unpack else res
            finish_reason = data["meta_info"]["finish_reason"]

            if is_batch:
                # Handle batch response
                assert isinstance(num_output_tokens_so_far, dict)
                index = data.get("index", 0)
                if index not in num_output_tokens_so_far:
                    num_output_tokens_so_far[index] = 0

                if finish_reason:
                    out = {
                        "token_ids": [],
                        "finish_reason": finish_reason["type"],
                        "index": index,
                    }
                else:
                    next_total_toks = len(data["output_ids"])
                    new_tokens = data["output_ids"][num_output_tokens_so_far[index] :]
                    out = {
                        "token_ids": new_tokens,
                        "index": index,
                    }
                    num_output_tokens_so_far[index] = next_total_toks
            else:
                # Handle single response
                assert isinstance(num_output_tokens_so_far, int)
                if finish_reason:
                    out = {"token_ids": [], "finish_reason": finish_reason["type"]}
                else:
                    next_total_toks = len(data["output_ids"])
                    out = {"token_ids": data["output_ids"][num_output_tokens_so_far:]}
                    num_output_tokens_so_far = next_total_toks

            yield out

    async def _prefill_generator(self, prefill):
        async for _ in prefill:
            pass

    async def flush_cache(self, request: dict):
        _ = request  # Suppress mypy unused variable warning
        print(f"DEBUG: Worker flush_cache called with request: {request}")
        
        try:
            # Add timeout to prevent hanging
            await asyncio.wait_for(
                self.engine.tokenizer_manager.flush_cache(), 
                timeout=10.0
            )
            logging.info("Prefill worker cache flushed")
            print(f"DEBUG: About to yield response")
            yield {"status": "success", "message": "Cache flushed"}
            print(f"DEBUG: Yielded response")
        except asyncio.TimeoutError:
            print(f"DEBUG: flush_cache timed out after 10 seconds")
            yield {"status": "error", "message": "Cache flush timed out"}
        except Exception as e:
            print(f"DEBUG: flush_cache error: {e}")
            yield {"status": "error", "message": f"Cache flush failed: {str(e)}"}


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    server_args = parse_sglang_args_inc(sys.argv[1:])
    await init(runtime, server_args)


async def setup_flush_endpoint(component: Any):
    flush_endpoint = component.endpoint("flush_cache")
    return flush_endpoint

async def init(runtime: DistributedRuntime, server_args: ServerArgs):
    """Initialize worker (either prefill or aggregated)"""

    engine = sgl.Engine(server_args=server_args)

    component = runtime.namespace("dynamo").component("worker")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await register_llm(
        ModelType.Backend,
        endpoint,
        server_args.model_path,
        server_args.served_model_name,
        kv_cache_block_size=server_args.page_size,
    )

    if server_args.disaggregation_mode != "null":
        decode_client = (
            await runtime.namespace("dynamo")
            .component("decode")
            .endpoint("generate")
            .client()
        )
        handler = RequestHandler(engine, server_args, component, decode_client)
    else:
        handler = RequestHandler(engine, server_args, component)

    # Set up metrics in background
    handler.setup_metrics()

    # Set up ZMQ kv event publisher
    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=endpoint.lease_id(),
        kv_block_size=server_args.page_size,
    )
    _ = ZmqKvEventPublisher(component=component, config=zmq_config)

    tasks = [endpoint.serve_endpoint(handler.generate)]

    if os.environ.get("DYN_SGL_HTTP_SERVER"):
        flush_endpoint = component.endpoint("flush_cache")
        tasks.append(flush_endpoint.serve_endpoint(handler.flush_cache))
        http_server = SglangHttpServer(server_args, runtime)
        asyncio.create_task(http_server.start_server())

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
