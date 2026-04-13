# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional backend entry point for Dynamo inference engines.

Instead of inheriting from an ABC, backend authors write an async main
function that parses args, starts the engine, and calls ``serve()`` with
callbacks::

    async def my_backend(argv=None):
        engine = await start_my_engine(argv)
        await serve(
            worker_config=...,
            engine_config=...,
            generate=engine.generate,
            cleanup=engine.shutdown,
        )

    if __name__ == "__main__":
        run(my_backend)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional

from dynamo._core import Context
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import ModelInput, ModelRuntimeConfig, register_model
from dynamo.llm.exceptions import CannotConnect, DynamoException, Unknown
from dynamo.runtime.logging import configure_dynamo_logging

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Metadata returned after engine startup, used for model registration."""

    model: str
    served_model_name: Optional[str] = None
    context_length: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None


@dataclass
class WorkerConfig:
    """Runtime connection and model registration parameters."""

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
    ) -> WorkerConfig:
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


# Callback type aliases
GenerateFn = Callable[[dict, Context], AsyncGenerator[dict, None]]
AbortFn = Callable[[Context], Awaitable[None]]
CleanupFn = Callable[[], Awaitable[None]]


async def serve(
    worker_config: WorkerConfig,
    engine_config: EngineConfig,
    generate: GenerateFn,
    abort: AbortFn | None = None,
    cleanup: CleanupFn | None = None,
) -> None:
    """Start the Dynamo runtime and serve inference requests.

    The caller has already parsed args and started their engine.  This
    function handles everything else: runtime creation, signal handlers,
    model registration, request cancellation, and graceful shutdown.

    Args:
        worker_config: Runtime connection and model registration params.
        engine_config: Model metadata for registration.
        generate: Async generator yielding response chunks per request.
                  Each chunk: ``{"token_ids": [...]}``.
                  Final chunk must include ``"finish_reason"`` and
                  ``"completion_usage"``.
        abort: Called when a request is cancelled.  Optional.
        cleanup: Called once on shutdown to release engine resources.  Optional.
    """
    configure_dynamo_logging()
    cfg = worker_config
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
    install_signal_handlers(loop, runtime, [endpoint], shutdown_event)

    async def _handle_request(
        request: dict, context: Context
    ) -> AsyncGenerator[dict, None]:
        async def _monitor_cancel():
            await context.async_killed_or_stopped()
            if abort is not None:
                try:
                    await abort(context)
                except Exception:
                    logger.debug("Error during request abort", exc_info=True)

        cancel_task = asyncio.create_task(_monitor_cancel())
        try:
            async for chunk in generate(request, context):
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

    try:
        runtime_config = ModelRuntimeConfig()
        if engine_config.total_kv_blocks is not None:
            runtime_config.total_kv_blocks = engine_config.total_kv_blocks
        if engine_config.max_num_seqs is not None:
            runtime_config.max_num_seqs = engine_config.max_num_seqs
        if engine_config.max_num_batched_tokens is not None:
            runtime_config.max_num_batched_tokens = engine_config.max_num_batched_tokens

        served_name = cfg.served_model_name or cfg.model_name
        await register_model(
            cfg.model_input,
            parse_endpoint_types(cfg.endpoint_types),
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
            _handle_request,
            graceful_shutdown=True,
            metrics_labels=cfg.metrics_labels,
        )
    finally:
        if cleanup is not None:
            await cleanup()
        logger.info("Engine cleanup complete")


def run(start, argv=None):
    """Sync entry point.  Calls ``start(argv)`` under uvloop."""
    import uvloop

    uvloop.run(start(argv))
