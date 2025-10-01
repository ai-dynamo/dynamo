# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

import sglang as sgl

from dynamo._core import Client, Component
from dynamo.llm import WorkerMetricsPublisher, ZmqKvEventPublisher
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.args import Config
from dynamo.sglang.engine_monitor import SglangEngineMonitor


class BaseWorkerHandler(ABC):
    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        metrics_publisher: WorkerMetricsPublisher = None,
        kv_publisher: ZmqKvEventPublisher = None,
        prefill_client: Client = None,
        runtime: Optional[DistributedRuntime] = None,
        endpoint_name: str = "generate",
    ):
        self.component = component
        self.engine = engine
        self.config = config
        self.metrics_publisher = metrics_publisher
        self.kv_publisher = kv_publisher
        self.prefill_client = prefill_client
        self.serving_mode = config.serving_mode
        self.skip_tokenizer_init = config.server_args.skip_tokenizer_init
        
        # Initialize engine monitor if runtime is provided
        self.engine_monitor: Optional[SglangEngineMonitor] = None
        if runtime:
            self.engine_monitor = SglangEngineMonitor(
                runtime=runtime,
                engine=engine,
                endpoint_name=endpoint_name
            )

    @abstractmethod
    async def generate(self, request: str):
        pass

    async def cleanup(self):
        """Cleanup handler resources including engine monitor."""
        if self.engine_monitor:
            await self.engine_monitor.stop()

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
