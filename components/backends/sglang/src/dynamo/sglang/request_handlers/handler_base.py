# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import sglang as sgl

from dynamo._core import Client, Component
from dynamo.llm import WorkerMetricsPublisher, ZmqKvEventPublisher
from dynamo.sglang.args import Config


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
    async def generate(self, request: str):
        pass

    def cleanup(self):
        pass

    def _get_input_param(self, request: dict) -> dict:
        """Get the appropriate input parameter for SGLang"""
        print(request)
        return (
            {"input_ids": request["token_ids"]} 
            if self.skip_tokenizer_init 
            else {"prompt": request["messages"]}
        )
