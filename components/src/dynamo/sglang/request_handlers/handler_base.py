# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import sglang as sgl

from dynamo._core import Client, Component, Context
from dynamo.llm import WorkerMetricsPublisher, ZmqKvEventPublisher
from dynamo.sglang.args import Config, DisaggregationMode


class BaseWorkerHandler(ABC):
    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        metrics_publisher: WorkerMetricsPublisher = None,
        kv_publisher: ZmqKvEventPublisher = None,
        prefill_client: Client = None,
    ):
        self.component = component
        self.engine = engine
        self.config = config
        self.metrics_publisher = metrics_publisher
        self.kv_publisher = kv_publisher
        self.prefill_client = prefill_client
        self.serving_mode = config.serving_mode
        self.skip_tokenizer_init = config.server_args.skip_tokenizer_init

    @abstractmethod
    async def generate(self, request: dict, context: Context):
        pass

    def cleanup(self):
        pass

    def _get_input_param(self, request: dict) -> dict:
        """Get the appropriate input parameter for SGLang"""
        if self.skip_tokenizer_init:
            return {"input_ids": request["token_ids"]}
        else:
            # use sglang's chat templating itself but leave tokenization to the
            # interal engine's TokenizerManager
            prompt = self.engine.tokenizer_manager.tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True
            )
            return {"prompt": prompt}

    async def _handle_cancellation(self, sglang_request_id: str, context: Context):
        """Background task to handle cancellation by monitoring context state."""
        try:
            logging.debug(
                f"Cancellation monitor started for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
            )
            # Wait asynchronously for cancellation signal instead of polling
            await context.async_killed_or_stopped()
            logging.info(
                f"Cancellation signal received for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
            )
            # Call abort_request on the tokenizer_manager through the engine
            if (
                hasattr(self.engine, "tokenizer_manager")
                and self.engine.tokenizer_manager
            ):
                try:
                    # Use SGLang's abort_request API
                    logging.debug(
                        f"Calling SGLang abort_request for Request ID {sglang_request_id}"
                    )
                    self.engine.tokenizer_manager.abort_request(
                        rid=sglang_request_id, abort_all=False
                    )
                    is_prefill = self.serving_mode == DisaggregationMode.PREFILL
                    logging.debug(
                        f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {context.id()}"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to abort SGLang request {sglang_request_id}: {e}"
                    )
            else:
                logging.error(
                    f"SGLang tokenizer_manager not found for abort request: {context.id()}"
                )
        except asyncio.CancelledError:
            # Task was cancelled, which is expected when generation completes
            logging.info(
                f"Cancellation monitor task cancelled for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
            )
            pass

    @asynccontextmanager
    async def _cancellation_monitor(
        self, sglang_request_id: str, context: Context
    ) -> AsyncGenerator[asyncio.Task, None]:
        """
        Context manager for monitoring request cancellation.
        Automatically creates a background task to monitor for cancellation and
        cleans it up when the context exits.
        Args:
            sglang_request_id: The SGLang request ID to abort if cancellation occurs
            context: Context object for cancellation handling
        Yields:
            asyncio.Task: The cancellation monitoring task
        """
        logging.info(
            f"Creating cancellation monitor task for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
        )
        cancellation_task = asyncio.create_task(
            self._handle_cancellation(sglang_request_id, context)
        )

        try:
            yield cancellation_task
        finally:
            # Clean up the background cancellation task
            if not cancellation_task.done():
                logging.info(
                    f"Cancelling cancellation monitor task for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
                )
                cancellation_task.cancel()
                try:
                    await cancellation_task
                except asyncio.CancelledError:
                    pass
            else:
                logging.info(
                    f"Cancellation monitor task already completed for SGLang Request ID {sglang_request_id}, Context: {context.id()}"
                )
