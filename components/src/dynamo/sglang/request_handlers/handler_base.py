# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import socket
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import sglang as sgl
from sglang.srt.utils import get_local_ip_auto

from dynamo._core import Client, Component
from dynamo.sglang.args import Config
from dynamo.sglang.publisher import DynamoSglangPublisher


class BaseWorkerHandler(ABC):
    """Abstract base class for SGLang worker handlers."""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        prefill_client: Optional[Client] = None,
    ) -> None:
        """Initialize base worker handler.

        Args:
            component: The Dynamo runtime component.
            engine: The SGLang engine instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher for the worker.
            prefill_client: Optional client for prefill worker in disaggregated mode.
        """
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
    async def generate(self, request: Dict[str, Any]):
        """Generate response from request.

        Args:
            request: Request dict with input and parameters.

        Yields:
            Response data (format varies by handler implementation).
        """
        pass

    def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses as needed."""
        pass

    def _get_input_param(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get the appropriate input parameter for SGLang engine.

        Args:
            request: Request dict with token_ids or messages.

        Returns:
            Dict with either input_ids or prompt for engine.
        """
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
        """Generate a unique bootstrap room ID for disaggregated serving.

        Returns:
            Random 63-bit integer.
        """
        return random.randint(0, 2**63 - 1)

    @staticmethod
    def _get_bootstrap_info(engine: sgl.Engine) -> Tuple[str, int]:
        """Extract bootstrap host and port from SGLang engine.

        Args:
            engine: The SGLang engine instance.

        Returns:
            Tuple of (bootstrap_host, bootstrap_port).
        """
        inner_tm = engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_local_ip_auto()

        return bootstrap_host, bootstrap_port
