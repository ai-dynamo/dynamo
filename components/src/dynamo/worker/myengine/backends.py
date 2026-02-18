# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample backend that demonstrates the minimal BaseBackend contract."""

from typing import Any, Dict

from dynamo.common.backend import BaseBackend

from dynamo.worker.myengine.handlers import MyEngineHandler


class MyEngineBackend(BaseBackend):
    """Minimal backend — implements only the three required abstract methods."""

    async def create_engine(self) -> None:
        # MyEngine has no real engine; the handler generates a fixed response.
        return None

    def create_handler(self, engine: Any, component: Any, endpoint: Any) -> MyEngineHandler:
        return MyEngineHandler(
            component=component,
            shutdown_event=self.shutdown_event,
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
