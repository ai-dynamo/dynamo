# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Optional

from dynamo._core import Context
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelRuntimeConfig, register_model
from dynamo.llm.exceptions import (
    CannotConnect,
    DynamoException,
    EngineShutdown,
    Unknown,
)
from dynamo.runtime.logging import configure_dynamo_logging

from .engine import GenerateChunk, GenerateRequest, LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    namespace: str
    component: str = "backend"
    endpoint: str = "generate"
    model_name: str = ""
    served_model_name: Optional[str] = None
    model_input: ModelInput = field(default_factory=lambda: ModelInput.Tokens)
    endpoint_types: str = "chat,completions"
    discovery_backend: str = "etcd"
    request_plane: str = "tcp"
    event_plane: str = "nats"
    use_kv_events: bool = False
    custom_jinja_template: Optional[str] = None
    metrics_labels: list = field(default_factory=list)

    @classmethod
    def from_runtime_config(
        cls,
        runtime_cfg,
        model_name: str,
        served_model_name: Optional[str] = None,
        model_input: Optional[ModelInput] = None,
        **overrides,
    ) -> "WorkerConfig":
        """Build from any object that carries DynamoRuntimeConfig fields.

        Works with vllm.Config, trtllm.Config (inherit DynamoRuntimeConfig
        directly) and sglang DynamoConfig (nested in config.dynamo_args).
        """
        kwargs = {
            "namespace": runtime_cfg.namespace,
            "component": getattr(runtime_cfg, "component", None) or "backend",
            "endpoint": getattr(runtime_cfg, "endpoint", None) or "generate",
            "model_name": model_name,
            "served_model_name": served_model_name,
            "endpoint_types": getattr(
                runtime_cfg, "endpoint_types", "chat,completions"
            ),
            "discovery_backend": runtime_cfg.discovery_backend,
            "request_plane": runtime_cfg.request_plane,
            "event_plane": runtime_cfg.event_plane,
            "use_kv_events": getattr(runtime_cfg, "use_kv_events", False),
            "custom_jinja_template": getattr(
                runtime_cfg, "custom_jinja_template", None
            ),
        }
        if model_input is not None:
            kwargs["model_input"] = model_input
        kwargs.update(overrides)
        return cls(**kwargs)


class Worker:
    def __init__(self, engine: LLMEngine, config: WorkerConfig):
        self.config = config
        self.engine = engine

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        async def _monitor_cancel():
            await context.async_killed_or_stopped()
            try:
                await self.engine.abort(context)
            except Exception:
                logger.debug("Error during request abort", exc_info=True)

        cancel_task = asyncio.create_task(_monitor_cancel())
        try:
            async for chunk in self.engine.generate(request, context):
                if context.is_stopped():
                    break
                yield chunk
        except DynamoException:
            raise
        except Exception as exc:
            raise Unknown(f"Engine generate failed: {exc}") from exc
        finally:
            if not cancel_task.done():
                cancel_task.cancel()
                try:
                    await cancel_task
                except asyncio.CancelledError:
                    pass

    async def run(self) -> None:
        configure_dynamo_logging()
        cfg = self.config
        shutdown_event = asyncio.Event()

        try:
            runtime, loop = create_runtime(
                discovery_backend=cfg.discovery_backend,
                request_plane=cfg.request_plane,
                event_plane=cfg.event_plane,
                use_kv_events=cfg.use_kv_events,
            )
        except DynamoException:
            raise
        except Exception as exc:
            raise CannotConnect(f"Failed to create runtime: {exc}") from exc

        endpoint = runtime.endpoint(f"{cfg.namespace}.{cfg.component}.{cfg.endpoint}")
        shutdown_endpoints = [endpoint]

        install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

        try:
            engine_config = await self.engine.start()
        except DynamoException:
            raise
        except Exception as exc:
            raise EngineShutdown(f"Engine initialization failed: {exc}") from exc

        try:
            runtime_config = ModelRuntimeConfig()
            if engine_config.total_kv_blocks is not None:
                runtime_config.total_kv_blocks = engine_config.total_kv_blocks
            if engine_config.max_num_seqs is not None:
                runtime_config.max_num_seqs = engine_config.max_num_seqs
            if engine_config.max_num_batched_tokens is not None:
                runtime_config.max_num_batched_tokens = (
                    engine_config.max_num_batched_tokens
                )

            model_type = parse_endpoint_types(cfg.endpoint_types)

            served_name = cfg.served_model_name or cfg.model_name

            await register_model(
                cfg.model_input,
                model_type,
                endpoint,
                cfg.model_name,
                served_name,
                context_length=engine_config.context_length,
                kv_cache_block_size=engine_config.kv_cache_block_size,
                runtime_config=runtime_config,
                custom_template_path=cfg.custom_jinja_template,
            )

            logger.info(
                "Serving %s on %s.%s.%s",
                served_name,
                cfg.namespace,
                cfg.component,
                cfg.endpoint,
            )

            await endpoint.serve_endpoint(
                self.generate,
                graceful_shutdown=True,
                metrics_labels=cfg.metrics_labels,
            )
        finally:
            await self.engine.cleanup()
            logger.info("Engine cleanup complete")
