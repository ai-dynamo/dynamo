# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

import sglang as sgl

from dynamo._core import Component
from dynamo.sglang.args import Config
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler


class PrefillWorkerHandler(BaseWorkerHandler):
    """Handler for prefill workers in disaggregated serving mode."""

    def __init__(
        self, component: Component, engine: sgl.Engine, config: Config
    ) -> None:
        """Initialize prefill worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
        """
        self.engine = engine
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info(self.engine)
        super().__init__(component, engine, config)
        logging.info(
            f"Prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    def cleanup(self) -> None:
        """Shutdown the prefill engine and cleanup resources."""
        self.engine.shutdown()
        logging.info("Prefill engine shutdown")
        super().cleanup()

    async def generate(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate prefill output and provide bootstrap info for decode worker.

        Args:
            request: Request dict with 'request' and 'sampling_params' keys.

        Yields:
            Bootstrap info dict with host, port, and room for decode worker connection.
        """
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

    async def _consume_results(self, results: AsyncGenerator[Any, None]) -> None:
        """Consume async generator results without processing.

        Args:
            results: Async generator from engine.async_generate.
        """
        async for _ in results:
            pass
