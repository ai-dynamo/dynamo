# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample backend that demonstrates the minimal Backend contract."""

from typing import Any, Dict

from dynamo.backend import Backend
from dynamo.llm import ModelRuntimeConfig
from dynamo.example_backend.handlers import ExampleHandler


class ExampleBackend(Backend):
    """Minimal backend — implements only the three required abstract methods."""

    def extract_runtime_config(self, engine: Any) -> ModelRuntimeConfig:
        """Return a minimal runtime config so the frontend discovers this model and enables chat/completions."""
        config = ModelRuntimeConfig()
        config.max_num_seqs = 1
        config.max_num_batched_tokens = 1
        config.enable_local_indexer = getattr(self.config, "enable_local_indexer", True)
        return config

    async def create_engine(self) -> None:
        # Example backend has no real engine; the handler generates a fixed response.
        return None

    def create_handler(
        self, engine: Any, component: Any, endpoint: Any
    ) -> ExampleHandler:
        return ExampleHandler(
            component=component,
            shutdown_event=self.shutdown_event,
            token_delay=self.config.backend.token_delay,
        )

    def get_health_check_payload(self, engine: Any) -> Dict[str, Any]:
        return {
            "token_ids": [1],
            "sampling_options": {"temperature": 0.0},
            "stop_conditions": {
                "max_tokens": 1,
                "stop": None,
                "stop_token_ids": None,
                "include_stop_str_in_output": False,
                "ignore_eos": False,
            },
        }
