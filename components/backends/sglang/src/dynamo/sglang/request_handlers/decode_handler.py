# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import sglang as sgl

from dynamo._core import Client, Component
from dynamo.llm import WorkerMetricsPublisher, ZmqKvEventPublisher
from dynamo.sglang.args import Config, DisaggregationMode
from dynamo.sglang.protocol import DisaggPreprocessedRequest
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


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
            if self.prefill_client is None:
                raise ValueError(
                    "prefill_client must be provided when serving_mode is decode"
                )
            self.prefill_client = prefill_client
            logging.info("Decode worker handler initialized")

        logging.info("Worker handler initialized")

    def cleanup(self):
        self.engine.shutdown()
        logging.info("Engine shutdown")
        super().cleanup()

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

    async def generate(self, request: dict):
        sampling_params = self._build_sampling_params(request)

        if self.serving_mode == DisaggregationMode.DECODE:
            # request the bootstrap info from the target prefill worker
            prefill_stream = await self.prefill_client.generate(
                DisaggPreprocessedRequest(
                    request=request,
                    sampling_params=sampling_params,
                ).model_dump()
            )

            bootstrap_info = None
            async for info in prefill_stream:
                bootstrap_info = info.data()
                break

            if not bootstrap_info:
                raise RuntimeError("No bootstrap info received from prefill worker")

            decode = await self.engine.async_generate(
                input_ids=request["token_ids"],
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_info["bootstrap_host"],
                bootstrap_port=bootstrap_info["bootstrap_port"],
                bootstrap_room=bootstrap_info["bootstrap_room"],
            )

            async for out in self._process_stream(decode):
                yield out
        else:
            agg = await self.engine.async_generate(
                input_ids=request["token_ids"],
                sampling_params=sampling_params,
                stream=True,
            )
            async for out in self._process_stream(agg):
                yield out

    async def _process_stream(self, stream_source):
        num_output_tokens_so_far = 0

        async for res in stream_source:
            try:
                next_total_toks = len(res["output_ids"])
            except KeyError:
                raise ValueError(
                    f"Missing 'output_ids' in response. This often happens when using skip_tokenizer_init=False. "
                    f"If you're using ModelType.CHAT or custom model configurations, you may need to modify "
                    f"the tokenization/detokenization logic in your handler. Response keys: {list(res.keys())}"
                )
            out = {"token_ids": res["output_ids"][num_output_tokens_so_far:]}
            num_output_tokens_so_far = next_total_toks
            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason:
                out = {"token_ids": [], "finish_reason": finish_reason["type"]}

            yield out
