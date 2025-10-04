# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import sglang as sgl

from dynamo._core import Component
from dynamo.sglang.args import Config
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, component: Component, engine: sgl.Engine, config: Config):
        super().__init__(component, engine, config, None, None, None)
        logging.info(
            f"Prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    def cleanup(self):
        self.engine.shutdown()
        logging.info("Prefill engine shutdown")
        super().cleanup()

    async def generate(self, request):
        # Use base handler method to parse request
        request = self._parse_request(request)

        # Use mixin method to yield bootstrap and process
        async for result in self._yield_bootstrap_and_process(
            self._process_prefill_request, request
        ):
            yield result

    async def _process_prefill_request(self, request: dict, bootstrap_room: int):
        """Process the prefill request with the given bootstrap room"""
        input_param = self._get_input_param(request["request"])

        generation_kwargs = {
            **input_param,
            "sampling_params": request["sampling_params"],
            "stream": True,
        }

        # Add bootstrap info using mixin method
        bootstrap_info = self._build_bootstrap_info(bootstrap_room)
        generation_kwargs = self._add_bootstrap_to_generation(
            generation_kwargs, bootstrap_info
        )

        results = await self.engine.async_generate(**generation_kwargs)

        asyncio.create_task(self._consume_results(results))

    async def _consume_results(self, results):
        async for _ in results:
            pass
