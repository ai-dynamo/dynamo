# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import random
import signal
import socket
import sys
from typing import Any, Dict, Optional, Union

import sglang as sgl
import uvloop
import zmq
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_ip, get_zmq_socket

from dynamo.llm import (
    ForwardPassMetrics,
    KvStats,
    ModelType,
    WorkerMetricsPublisher,
    WorkerStats,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.common import (
    BaseWorkerHandler,
    DisaggPreprocessedRequest,
    graceful_shutdown,
    parse_sglang_args_inc,
    setup_native_endpoints,
)

configure_dynamo_logging()


class RequestHandler(BaseWorkerHandler):
    def __init__(
        self,
        engine: sgl.Engine,
        server_args: ServerArgs,
        component,
        decode_client: Optional[Any] = None,
    ):
        super().__init__(engine, server_args, component, decode_client)
        self.metrics_publisher = WorkerMetricsPublisher()

        self.zmq_context = zmq.asyncio.Context()  # type: ignore
        self.receive_metrics_from_scheduler = None

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
        """Set up metrics publisher"""
        self.receive_metrics_from_scheduler = get_zmq_socket(
            self.zmq_context, zmq.PULL, self.engine.port_args.metrics_ipc_name, True
        )

        self.init_publish()
        asyncio.create_task(self._receive_and_publish_metrics_loop())

        task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logging.debug("metrics publisher endpoint created")
        )

    def init_publish(self):
        """Publish initial set of warmup metrics"""
        worker_stats = WorkerStats(
            request_active_slots=0,
            request_total_slots=1024,
            num_requests_waiting=0,
            data_parallel_rank=0,
        )

        kv_stats = KvStats(
            kv_active_blocks=0,
            kv_total_blocks=1024,
            gpu_cache_usage_perc=0,
            gpu_prefix_cache_hit_rate=0,
        )

        metrics = ForwardPassMetrics(
            worker_stats=worker_stats,
            kv_stats=kv_stats,
            spec_decode_stats=None,
        )

        self.metrics_publisher.publish(metrics)

    async def create_metrics_publisher_endpoint(self):
        logging.debug("Creating metrics publisher endpoint")
        await self.metrics_publisher.create_endpoint(self.component)

    async def _receive_and_publish_metrics_loop(self):
        """Receive metrics from SGL scheduler and publish them"""
        while True:
            try:
                kv_metrics = await self.receive_metrics_from_scheduler.recv_pyobj()  # type: ignore
                worker_stats = WorkerStats(
                    request_active_slots=kv_metrics.request_active_slots,
                    request_total_slots=kv_metrics.request_total_slots,
                    num_requests_waiting=kv_metrics.num_requests_waiting,
                    data_parallel_rank=kv_metrics.data_parallel_rank,  # Note: 0 means it's either 0 or None from sglang
                )
                kv_stats = KvStats(
                    kv_active_blocks=kv_metrics.kv_active_blocks,
                    kv_total_blocks=kv_metrics.kv_total_blocks,
                    gpu_cache_usage_perc=kv_metrics.gpu_cache_usage_perc,
                    gpu_prefix_cache_hit_rate=kv_metrics.gpu_prefix_cache_hit_rate,
                )
                spec_dec_stats = None
                metrics = ForwardPassMetrics(
                    worker_stats=worker_stats,
                    kv_stats=kv_stats,
                    spec_decode_stats=spec_dec_stats,
                )

                self.metrics_publisher.publish(metrics)
            except Exception:
                logging.exception("Failed to recieve or publish metrics")

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

        try:
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
        except asyncio.CancelledError:
            raise GeneratorExit(
                "SGLang decode engine was shut down during token generation"
            ) from None

    async def _prefill_generator(self, prefill):
        async for _ in prefill:
            pass

    async def flush_cache(self, request: dict):
        _ = request
        asyncio.create_task(self.engine.tokenizer_manager.flush_cache())
        yield {
            "status": "success",
            "message": "Cache flush initiated. Check backend logs for status",
        }


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # TODO: Better handle non-sglang args
    sys_argv = sys.argv[1:]
    migration_limit = 0
    try:
        idx = sys_argv.index("--migration-limit")
        migration_limit = int(sys_argv[idx + 1])
        del sys_argv[idx : idx + 2]  # Remove the args from sys_argv
    except Exception:
        pass

    server_args = parse_sglang_args_inc(sys_argv)
    await init(runtime, server_args, migration_limit)


async def init(
    runtime: DistributedRuntime, server_args: ServerArgs, migration_limit: int
):
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
        migration_limit=migration_limit,
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

    # Set up the engine metrics reciever
    handler.setup_metrics()

    # Set up ZMQ kv event publisher
    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=endpoint.lease_id(),
        kv_block_size=server_args.page_size,
    )
    _ = ZmqKvEventPublisher(component=component, config=zmq_config)

    tasks = [endpoint.serve_endpoint(handler.generate, graceful_shutdown=False)]

    tasks.extend(setup_native_endpoints(server_args, component, handler))

    await asyncio.gather(*tasks)


def main():
    uvloop.install()
    asyncio.run(worker())


if __name__ == "__main__":
    main()
