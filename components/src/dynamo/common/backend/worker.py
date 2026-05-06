# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import enum
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

from .engine import EngineConfig, GenerateChunk, GenerateRequest, LLMEngine

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
    event_plane: Optional[str] = None
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


class _LifecycleState(enum.Enum):
    INIT = "init"  # _start_engine has not been called
    STARTING = "starting"  # engine.start() in flight (lock held)
    RUNNING = "running"  # engine.start() returned successfully
    STOPPING = "stopping"  # engine.cleanup() in flight (lock held)
    STOPPED = "stopped"  # cleanup done, never started, or start failed


class Worker:
    def __init__(self, engine: LLMEngine, config: WorkerConfig):
        self.config = config
        self.engine = engine
        # Lifecycle is INIT -> STARTING -> RUNNING -> STOPPING -> STOPPED.
        # A single lock serializes start and cleanup so engine.start() and
        # engine.cleanup() can never run concurrently and cleanup never
        # observes a half-built engine. STARTING/STOPPING are only visible
        # to the lock holder; other coroutines see INIT/RUNNING/STOPPED.
        self._state = _LifecycleState.INIT
        self._lifecycle_lock = asyncio.Lock()

    async def _start_engine(self) -> EngineConfig:
        # Holds _lifecycle_lock through the entire engine.start() call so a
        # cleanup arriving on the signal-handler path waits for start to
        # finish before deciding what to do. On exit the state is RUNNING
        # (success) or STOPPED (failure / shutdown-before-start) — never
        # STARTING.
        async with self._lifecycle_lock:
            if self._state == _LifecycleState.STOPPED:
                # Shutdown signal arrived before start; abort cleanly.
                raise EngineShutdown("Shutdown requested before engine start")
            if self._state != _LifecycleState.INIT:
                raise RuntimeError(
                    f"_start_engine called in unexpected state: {self._state.value}"
                )
            self._state = _LifecycleState.STARTING
            try:
                config = await self.engine.start()
            except BaseException:
                # Mark stopped so a follow-up _cleanup_once is a no-op.
                self._state = _LifecycleState.STOPPED
                raise
            self._state = _LifecycleState.RUNNING
            return config

    async def _cleanup_once(self) -> None:
        # Serialized with _start_engine via _lifecycle_lock. Called from two
        # paths: the graceful-shutdown signal handler (before runtime.shutdown
        # tears down Rust services) and run()'s finally block. The lock makes
        # whichever caller arrives second wait until the first finishes, and
        # the state machine ensures engine.cleanup() runs at most once and
        # only when the engine actually started.
        async with self._lifecycle_lock:
            if self._state in (_LifecycleState.INIT, _LifecycleState.STOPPED):
                # INIT: shutdown arrived before _start_engine; nothing to do.
                # STOPPED: already cleaned up, or start failed and never
                # produced anything to clean up.
                self._state = _LifecycleState.STOPPED
                return
            assert (
                self._state == _LifecycleState.RUNNING
            ), f"_cleanup_once invoked in unexpected state: {self._state.value}"
            self._state = _LifecycleState.STOPPING
            try:
                await self.engine.cleanup()
                logger.info("Engine cleanup complete")
            finally:
                # Mark stopped even on failure so a follow-up call no-ops;
                # engines like vLLM/TRT-LLM tear down NCCL groups in
                # cleanup() and a second attempt can hang or raise.
                self._state = _LifecycleState.STOPPED

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
                if "index" not in chunk:
                    chunk["index"] = 0
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

        install_signal_handlers(
            loop,
            runtime,
            shutdown_endpoints,
            shutdown_event,
            cleanup_callback=self._cleanup_once,
        )

        try:
            engine_config = await self._start_engine()
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
            await self._cleanup_once()
