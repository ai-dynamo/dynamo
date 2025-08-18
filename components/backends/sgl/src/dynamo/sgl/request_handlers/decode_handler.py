import asyncio
import logging
import random
import socket

import sglang as sgl
from sglang.srt.utils import get_ip

from dynamo._core import Client, Component
from dynamo.llm import WorkerMetricsPublisher, ZmqKvEventPublisher
from dynamo.sgl.args import Config, DisaggregationMode
from dynamo.sgl.protocol import DisaggPreprocessedRequest
from dynamo.sgl.request_handlers.handler_base import BaseWorkerHandler


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        metrics_publisher: WorkerMetricsPublisher,
        kv_publisher: ZmqKvEventPublisher = None,
        prefill_client: Client = None,
    ):
        super().__init__(
            component, engine, config, metrics_publisher, kv_publisher, prefill_client
        )
        if self.serving_mode == DisaggregationMode.DECODE:
            self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()
            if self.prefill_client is None:
                raise ValueError(
                    "prefill_client must be provided when serving_mode is decode"
                )
            self.prefill_client = prefill_client
            logging.info(
                f"Disaggregation enabled - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
            )

        logging.info("Decode worker handler initialized")

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

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    async def generate(self, request: str):
        sampling_params = self._build_sampling_params(request)

        if self.serving_mode == DisaggregationMode.DECODE:
            bootstrap_host = self.bootstrap_host
            bootstrap_port = self.bootstrap_port
            bootstrap_room = self._generate_bootstrap_room()

            # remote prefill request
            disagg_request = DisaggPreprocessedRequest(
                request=request,
                sampling_params=sampling_params,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
            )

            # prefill response is not used
            prefill = await self.prefill_client.generate(
                disagg_request.model_dump_json()
            )
            prefill_task = asyncio.create_task(self._consume_prefill(prefill))

            # decode request
            decode = await self.engine.async_generate(
                input_ids=request["token_ids"],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_host,
                bootstrap_port=bootstrap_port,
                bootstrap_room=bootstrap_room,
            )

            async for out in self._process_stream(decode, unpack=True):
                yield out

            await prefill_task
        else:
            agg = await self.engine.async_generate(
                input_ids=request["token_ids"],
                sampling_params=sampling_params,
                stream=True,
            )
            async for out in self._process_stream(agg, unpack=False):
                yield out

    async def _process_stream(self, stream_source, unpack: bool):
        num_output_tokens_so_far = 0

        async for res in stream_source:
            data = res.data() if unpack else res
            finish_reason = data["meta_info"]["finish_reason"]

            if finish_reason:
                out = {"token_ids": [], "finish_reason": finish_reason["type"]}
            else:
                next_total_toks = len(data["output_ids"])
                out = {"token_ids": data["output_ids"][num_output_tokens_so_far:]}
                num_output_tokens_so_far = next_total_toks

            yield out

    async def _consume_prefill(self, prefill):
        async for _ in prefill:
            pass
