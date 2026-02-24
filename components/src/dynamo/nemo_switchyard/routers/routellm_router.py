#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
RouteLLM router — classifies query complexity and routes to strong/weak pool.

Requires the ``routellm`` and ``transformers`` packages.  All heavy
imports are deferred to :meth:`initialize` so that other routers work
without these dependencies installed.

Usage::

    python -m dynamo.nemo_switchyard \\
        --router-type routellm \\
        --routellm-algorithm mf \\
        --model-path <tokenizer-model> \\
        --threshold 0.5 \\
        --pool strong=strong_pool.router.generate \\
        --pool weak=weak_pool.router.generate
"""

import asyncio
import logging

from ..base import BaseModelRouter, RouterConfig
from ..registry import register_router

logger = logging.getLogger(__name__)


class RouteLLMRouter(BaseModelRouter):
    """Routes requests via RouteLLM classification (strong/weak)."""

    # Expected config.extra keys:
    #   model_path (str, required)  — HF tokenizer model
    #   routellm_algorithm (str)    — RouteLLM algorithm name (default "mf")
    #   threshold (float)           — routing threshold 0.0-1.0 (default 0.5)
    #   checkpoint_path (str|None)  — optional checkpoint for the RouteLLM router

    def __init__(self, config: RouterConfig):
        super().__init__(config)

        if len(config.pool_names) != 2:
            raise ValueError(
                f"RouteLLM requires exactly 2 pools (strong, weak), "
                f"got {len(config.pool_names)}: {config.pool_names}"
            )

        self._strong_pool = config.pool_names[0]
        self._weak_pool = config.pool_names[1]

        self._model_path = config.extra.get("model_path", "")
        if not self._model_path:
            raise ValueError("RouteLLM router requires 'model_path' in config.extra")

        self._algorithm = config.extra.get("routellm_algorithm", "mf")
        self._threshold = float(config.extra.get("threshold", 0.5))
        self._checkpoint_path = config.extra.get("checkpoint_path")

        self._controller = None
        self._tokenizer = None

        self._stats = {
            "total": 0,
            "strong_routes": 0,
            "weak_routes": 0,
            "fallback_routes": 0,
            "errors": 0,
        }

    async def initialize(self) -> None:
        # Deferred imports — only needed when this router is actually used
        from routellm.controller import Controller
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer from %s", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )

        logger.info(
            "Initializing RouteLLM: algorithm=%s, threshold=%.2f",
            self._algorithm,
            self._threshold,
        )
        if self._checkpoint_path:
            router_config = {
                self._algorithm: {"checkpoint_path": self._checkpoint_path}
            }
        else:
            router_config = None

        self._controller = Controller(
            routers=[self._algorithm],
            strong_model="strong",
            weak_model="weak",
            config=router_config,
        )

    async def cleanup(self) -> None:
        self._controller = None
        self._tokenizer = None

    async def route(self, request: dict) -> str:
        token_ids = request.get("token_ids", [])
        self._stats["total"] += 1

        # Detokenize to recover prompt text
        try:
            prompt = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            logger.exception("Failed to detokenize, falling back to %s pool", self._strong_pool)
            self._stats["fallback_routes"] += 1
            return self._strong_pool

        if not prompt or not prompt.strip():
            logger.warning(
                "Empty prompt after detokenization, falling back to %s pool",
                self._strong_pool,
            )
            self._stats["fallback_routes"] += 1
            return self._strong_pool

        # Classify with RouteLLM — Controller.route() is synchronous,
        # so run in executor to avoid blocking the event loop.
        try:
            loop = asyncio.get_running_loop()
            routed = await loop.run_in_executor(
                None, self._controller.route, prompt, self._algorithm, self._threshold
            )
            if routed == "strong":
                self._stats["strong_routes"] += 1
                pool = self._strong_pool
            else:
                self._stats["weak_routes"] += 1
                pool = self._weak_pool
            logger.debug("Routed to %s (prompt: %.80s...)", pool, prompt.strip())
            return pool
        except Exception:
            logger.exception(
                "RouteLLM classification failed, falling back to %s pool",
                self._strong_pool,
            )
            self._stats["fallback_routes"] += 1
            self._stats["errors"] += 1
            return self._strong_pool

    def get_stats(self) -> dict:
        return dict(self._stats)


register_router("routellm", RouteLLMRouter)
