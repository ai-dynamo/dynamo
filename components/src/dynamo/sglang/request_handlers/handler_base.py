# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import socket
from abc import ABC, abstractmethod

import sglang as sgl
from sglang.srt.utils import get_ip

from dynamo._core import Client, Component
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher


class BaseWorkerHandler(ABC):
    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: DynamoSglangPublisher = None,
        prefill_client: Client = None,
    ):
        self.component = component
        self.engine = engine
        self.config = config
        if publisher is not None:
            self.metrics_publisher = publisher.metrics_publisher
            self.kv_publisher = publisher.kv_publisher
        else:
            self.metrics_publisher = None
            self.kv_publisher = None
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
        if self.skip_tokenizer_init:
            return {"input_ids": request["token_ids"]}
        else:
            # use sglang's chat templating itself but leave tokenization to the
            # interal engine's TokenizerManager
            prompt = self.engine.tokenizer_manager.tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True
            )
            return {"prompt": prompt}

    @staticmethod
    def _generate_bootstrap_room() -> int:
        """Generate a unique bootstrap room ID"""
        return random.randint(0, 2**63 - 1)

    @staticmethod
    def _get_bootstrap_info(engine: sgl.Engine) -> tuple[str, int]:
        """Extract bootstrap info from SGLang engine"""
        inner_tm = engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_ip()

        return bootstrap_host, bootstrap_port
