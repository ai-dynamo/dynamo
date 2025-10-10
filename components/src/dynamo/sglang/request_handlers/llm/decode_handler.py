# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import sglang as sgl

from dynamo._core import Client, Component
from dynamo.sglang.args import Config, DisaggregationMode
from dynamo.sglang.protocol import DisaggPreprocessedRequest
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class DecodeWorkerHandler(BaseWorkerHandler):
    """Handler for decode workers in both aggregated and disaggregated serving modes."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher,
        prefill_client: Optional[Client] = None,
        prefill_router_client: Optional[Client] = None,
    ) -> None:
        """Initialize decode worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Metrics publisher for the worker.
            prefill_client: Optional client for prefill worker in disaggregated mode.
            prefill_router_client: Optional client for prefill router in disaggregated mode.

        Raises:
            ValueError: If prefill_client is not provided in decode serving mode.
        """
        super().__init__(
            component,
            engine,
            config,
            publisher,
            prefill_client,
        )
        if self.serving_mode == DisaggregationMode.DECODE:
            if self.prefill_client is None:
                raise ValueError(
                    "prefill_client must be provided when serving_mode is decode"
                )
            self.prefill_client = prefill_client
            logging.info("Decode worker handler initialized")

        self.prefill_router_client = prefill_router_client
        logging.info("Worker handler initialized")

    def cleanup(self) -> None:
        """Shutdown the engine and cleanup resources."""
        self.engine.shutdown()
        logging.info("Engine shutdown")
        super().cleanup()

    def _build_sampling_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Build sampling params from request format.

        Args:
            request: Request dict in either token-based or OpenAI format.

        Returns:
            Dict of sampling parameters for SGLang engine.
        """
        if self.skip_tokenizer_init:
            # Token-based request format
            sampling_opts = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})

            param_mapping = {
                "temperature": sampling_opts.get("temperature"),
                "top_p": sampling_opts.get("top_p"),
                "top_k": sampling_opts.get("top_k"),
                "max_new_tokens": stop_conditions.get("max_tokens"),
                "ignore_eos": stop_conditions.get("ignore_eos"),
            }
        else:
            # OpenAI request format
            param_mapping = {
                "temperature": request.get("temperature"),
                "top_p": request.get("top_p"),
                "top_k": request.get("top_k"),
                "max_new_tokens": request.get("max_tokens"),
            }

        return {k: v for k, v in param_mapping.items() if v is not None}

    async def generate(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate response in aggregated or disaggregated mode.

        Args:
            request: Request dict with input and sampling parameters.

        Yields:
            Response dicts with token_ids or OpenAI-formatted chunks.

        Raises:
            RuntimeError: If no bootstrap info received from prefill worker.
        """
        sampling_params = self._build_sampling_params(request)
        input_param = self._get_input_param(request)

        if self.serving_mode == DisaggregationMode.DECODE:
            # request the bootstrap info from the target prefill worker
            if (
                self.prefill_router_client is not None
                and self.prefill_router_client.instance_ids()
            ):
                token_ids = request["token_ids"]
                stream = await self.prefill_router_client.generate(token_ids)
                result = await anext(stream)
                (
                    worker_id,
                    overlap,
                ) = result.data()  # Returns tuple (worker_id, overlap_amount)
                logging.info(f"Best prefill worker ID: {worker_id}, overlap: {overlap}")

                prefill_stream = await self.prefill_client.direct(
                    DisaggPreprocessedRequest(
                        request=request,
                        sampling_params=sampling_params,
                    ).model_dump(),
                    worker_id,
                )
            else:
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
                **input_param,
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=bootstrap_info["bootstrap_host"],
                bootstrap_port=bootstrap_info["bootstrap_port"],
                bootstrap_room=bootstrap_info["bootstrap_room"],
            )

            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(decode):
                    yield out
            else:
                async for out in self._process_text_stream(decode):
                    yield out
        else:
            agg = await self.engine.async_generate(
                **input_param,
                sampling_params=sampling_params,
                stream=True,
            )
            if self.skip_tokenizer_init:
                async for out in self._process_token_stream(agg):
                    yield out
            else:
                async for out in self._process_text_stream(agg):
                    yield out

    async def _process_token_stream(
        self, stream_source: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process token-based stream output.

        Args:
            stream_source: Async generator from engine.async_generate.

        Yields:
            Dict with token_ids and optional finish_reason.
        """
        num_output_tokens_so_far = 0

        async for res in stream_source:
            out = {}
            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason:
                out["finish_reason"] = finish_reason["type"]

            output_ids = res.get("output_ids", [])
            # If request is not finished yet, but there are no outputs, return an error.
            if not output_ids and not finish_reason:
                yield {"finish_reason": "error", "token_ids": []}
                break

            next_total_toks = len(output_ids)
            out["token_ids"] = output_ids[num_output_tokens_so_far:]
            num_output_tokens_so_far = next_total_toks
            yield out

    async def _process_text_stream(
        self, stream_source: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process text-based stream output in OpenAI format.

        Args:
            stream_source: Async generator from engine.async_generate.

        Yields:
            OpenAI-formatted chat completion chunk dicts.
        """
        count = 0

        async for res in stream_source:
            index = res.get("index", 0)
            text = res.get("text", "")

            finish_reason = res["meta_info"]["finish_reason"]
            finish_reason_type = finish_reason["type"] if finish_reason else None
            next_count = len(text)
            delta = text[count:]

            choice_data = {
                "index": index,
                "delta": {"role": "assistant", "content": delta},
                "finish_reason": finish_reason_type,
            }

            response = {
                "id": res["meta_info"]["id"],
                "created": int(time.time()),
                "choices": [choice_data],
                "model": self.config.server_args.served_model_name,
                "object": "chat.completion.chunk",
            }
            yield response
            count = next_count
