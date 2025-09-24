# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import random
import socket

import sglang as sgl
from sglang.srt.utils import get_ip

from dynamo._core import Component
from dynamo.sglang.args import Config
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, component: Component, engine: sgl.Engine, config: Config):
        self.engine = engine
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info()
        super().__init__(component, engine, config, None, None, None)
        logging.info(
            f"Prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    def cleanup(self):
        self.engine.shutdown()
        logging.info("Prefill engine shutdown")
        super().cleanup()

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

    async def generate(self, request: dict):
        bootstrap_room = self._generate_bootstrap_room()

        bootstrap_info = {
            "bootstrap_host": self.bootstrap_host,
            "bootstrap_port": self.bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }

        yield bootstrap_info

        input_param = self._get_input_param(request["request"])

        results = await self.engine.async_generate(
            **input_param,
            sampling_params=request["sampling_params"],
            stream=True,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=bootstrap_room,
        )

        asyncio.create_task(self._consume_results(results))

    async def _consume_results(self, results):
        async for _ in results:
            pass
