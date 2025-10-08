# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SGLang Native APIs: https://docs.sglang.ai/basic_usage/native_api.html
# Code: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py

import asyncio
import logging
from typing import List, Optional, Tuple

import sglang as sgl

from dynamo._core import Component


class NativeApiHandler:
    """Handler to add sglang native API endpoints to workers"""

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        metrics_labels: Optional[List[Tuple[str, str]]] = None,
    ):
        self.component = component
        self.engine = engine
        self.metrics_labels = metrics_labels
        self.native_api_tasks = []

    async def init_native_apis(
        self,
    ) -> List[asyncio.Task]:
        """
        Initialize and register native API endpoints.
        Returns list of tasks to be gathered.
        """
        logging.info("Initializing native SGLang API endpoints")

        self.tm = self.engine.tokenizer_manager

        tasks = []

        model_info_ep = self.component.endpoint("get_model_info")
        tasks.extend(
            [
                model_info_ep.serve_endpoint(
                    self.get_model_info,
                    graceful_shutdown=True,
                    metrics_labels=self.metrics_labels,
                    http_endpoint_path="/get_model_info",
                ),
            ]
        )

        self.native_api_tasks = tasks
        logging.info(f"Registered {len(tasks)} native API endpoints")
        return tasks

    async def get_model_info(self, request: dict):
        _ = request
        result = {
            "model_path": self.tm.server_args.model_path,
            "tokenizer_path": self.tm.server_args.tokenizer_path,
            "preferred_sampling_params": self.tm.server_args.preferred_sampling_params,
            "weight_version": self.tm.server_args.weight_version,
        }

        yield {"data": [result]}
