# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import enum
import logging
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Optional

import tensorrt_llm.llmapi.llm as trtllm_llm_module
from tensorrt_llm import LLM, MultimodalEncoder
from tensorrt_llm.llmapi.llm import BaseLLM
from transformers import PretrainedConfig

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine_monitor import TrtllmEngineMonitor
from dynamo.trtllm.preprocess_pool import (
    PreprocessPool,
    ProcessorCall,
    record_input_processor_calls,
)

logger = logging.getLogger(__name__)

# Model architectures without standalone encoder support in TRT-LLM
# (missing @register_vision_encoder). These handle vision encoding
# inside the main model (prefill/decode) instead.
_UNSUPPORTED_STANDALONE_ENCODER_ARCHS = {"Llama4ForConditionalGeneration"}


def _release_preprocessed_inputs(future: Future[Any]) -> None:
    """Release the shared tensors of preprocessed inputs that will not be submitted."""
    if future.cancelled() or future.exception() is not None:
        return
    multimodal_params = future.result().multimodal_params
    if multimodal_params is not None:
        multimodal_params.release_handles()


def _close_started_pool(future: Future[PreprocessPool]) -> None:
    """Close a preprocessing pool that started after startup was cancelled."""
    if not future.cancelled() and future.exception() is None:
        future.result().close()


class Backend(str, enum.Enum):
    """Supported TensorRT-LLM backend types."""

    PYTORCH = "pytorch"


class TensorRTLLMEngine:
    def __init__(
        self,
        engine_args: dict[str, Any],
        disaggregation_mode: Optional[DisaggregationMode] = None,
    ) -> None:
        self._llm: Optional[LLM] = None
        self._health_monitor: Optional[TrtllmEngineMonitor] = None
        self._preprocess_pool: Optional[PreprocessPool] = None
        # Threads that run multimodal input preprocessing: one per pool process, or
        # one without a pool. The executor is created on first use.
        self._preprocess_threads = 1
        self._preprocess_executor: Optional[ThreadPoolExecutor] = None
        self.disaggregation_mode = (
            disaggregation_mode
            if disaggregation_mode is not None
            else DisaggregationMode.AGGREGATED
        )
        # NOTE: `engine_args` may be reused by callers (e.g., for logging or other workers).
        # Copy it so that our internal `pop()` doesn't leak side effects.
        engine_args = dict(engine_args)
        backend = engine_args.pop("backend", Backend.PYTORCH)
        if backend == Backend.PYTORCH:
            self._llm_cls = LLM
        else:
            raise ValueError(
                f"Unsupported {backend=}. Available backends: {[b.value for b in Backend]}."
            )

        self.engine_args = engine_args

    @property
    def encoder_available(self) -> bool:
        """Whether the multimodal encoder LLM is initialized."""
        return self._llm is not None

    async def initialize(self) -> None:
        if not self._llm:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                # Initialize the multimodal encoder for full EPD
                # Prefill/decode workers initialize the standard TRT-LLM `LLM` from `engine_args`
                # (model, backend settings, kv cache config, etc.). ENCODE workers instead use
                # TRT-LLM's `MultimodalEncoder`, which has a different constructor surface.
                # Keep an explicit allowlist so LLM-only arguments are not forwarded.
                model = self.engine_args.get("model")

                # Skip MultimodalEncoder for architectures that handle vision
                # encoding inside the main model (e.g. Llama4).
                if self._is_unsupported_encoder_arch(model):  # type: ignore[arg-type]
                    return

                max_batch_size = self.engine_args.get("max_batch_size", 1)
                logging.info(
                    f"Initializing multimodal encoder with max_batch_size: {max_batch_size}"
                )
                encoder_kwargs: dict[str, Any] = {
                    "model": model,
                    "max_batch_size": max_batch_size,
                }
                for arg_name in (
                    "trust_remote_code",
                    "tensor_parallel_size",
                    "model_kwargs",
                ):
                    if arg_name in self.engine_args:
                        encoder_kwargs[arg_name] = self.engine_args[arg_name]

                # MultimodalEncoder and LLM both inherit from BaseLLM in TRT-LLM,
                # so storing either in self._llm is valid.
                self._llm = MultimodalEncoder(**encoder_kwargs)
            else:
                # Prefill/decode workers: initialize standard TRT-LLM `LLM` with full engine_args
                # (model path, backend settings, KV cache config, disaggregation settings, etc.)
                self._llm = self._llm_cls(**self.engine_args)

    async def start_preprocess_pool(
        self, num_processes: int, processor_calls: list[ProcessorCall]
    ) -> None:
        """Run the LLM's multimodal input processing in ``num_processes`` processes.

        ``processor_calls`` are the create_input_processor calls recorded while the
        LLM was built.
        """
        starter = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mm-preprocess-start"
        )
        future = starter.submit(
            PreprocessPool.start,
            trtllm_llm_module,
            self.llm,
            processor_calls,
            num_processes,
        )
        starter.shutdown(wait=False)
        try:
            pool = await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # The processes keep starting, so close the pool once they have.
            future.add_done_callback(_close_started_pool)
            raise
        self._preprocess_pool = pool
        self._preprocess_threads = num_processes

    async def preprocess(
        self, inputs: Any, sampling_params: Any, disaggregated_params: Any
    ) -> Any:
        """Run the LLM's input preprocessing for one request off the event loop.

        Requests are preprocessed one at a time, or one per process when a
        preprocessing pool runs.
        """
        if self._preprocess_executor is None:
            self._preprocess_executor = ThreadPoolExecutor(
                max_workers=self._preprocess_threads,
                thread_name_prefix="mm-preprocess",
            )
        # Keep the caller's context variables, as asyncio.to_thread does.
        future = self._preprocess_executor.submit(
            contextvars.copy_context().run,
            self.llm.preprocess,
            inputs,
            sampling_params,
            disaggregated_params,
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # Preprocessing that already started still finishes, and its result
            # holds shared tensors until they are released.
            if not future.cancel():
                future.add_done_callback(_release_preprocessed_inputs)
            raise

    async def cleanup(self) -> None:
        await self.stop_health_monitor()
        self.shutdown()

    def shutdown(self) -> None:
        if self._preprocess_executor is not None:
            self._preprocess_executor.shutdown(wait=False, cancel_futures=True)
            self._preprocess_executor = None
        if self._preprocess_pool is not None:
            self._preprocess_pool.close()
            self._preprocess_pool = None
        if self._llm:
            try:
                self._llm.shutdown()
            except Exception as e:
                logging.error(f"Error during shutdown: {e}")
            finally:
                self._llm = None

    @property
    def llm(self) -> BaseLLM:
        if not self._llm:
            raise RuntimeError("Engine not initialized")
        return self._llm

    def supports_health_check(self) -> bool:
        return self._get_health_check_fn() is not None

    def check_health(self) -> bool:
        if self._llm is None:
            return False
        check_health = self._get_health_check_fn()
        if check_health is None:
            return True
        return bool(check_health())

    def get_health_check_fatal_error(self) -> Optional[BaseException]:
        if self._llm is None:
            return None
        executor = getattr(self._llm, "_executor", None)
        return getattr(executor, "_fatal_error", None)

    def start_health_monitor(
        self, runtime: Optional[Any] = None, shutdown_event: Optional[Any] = None
    ) -> Optional[TrtllmEngineMonitor]:
        if self._health_monitor is None:
            self._health_monitor = TrtllmEngineMonitor(
                self, runtime=runtime, shutdown_event=shutdown_event
            )
        return self._health_monitor

    async def stop_health_monitor(self) -> None:
        if self._health_monitor is None:
            return
        await self._health_monitor.stop()
        self._health_monitor = None

    def _get_health_check_fn(self):
        if self._llm is None:
            return None
        check_health = getattr(self._llm, "_check_health", None)
        if callable(check_health):
            return check_health
        executor = getattr(self._llm, "_executor", None)
        check_health = getattr(executor, "check_health", None)
        if callable(check_health):
            return check_health
        return None

    def get_attention_dp_size(self) -> int:
        """Return attention_dp_size (tensor_parallel_size if attention DP enabled, else 1).
        When attention DP is enabled, each attention DP rank becomes a separate routing target.
        """
        if not self._llm:
            return 1
        enable_attention_dp = getattr(self.llm.args, "enable_attention_dp", False)
        tensor_parallel_size = getattr(self.llm.args, "tensor_parallel_size", 1)
        return tensor_parallel_size if enable_attention_dp else 1

    def get_kv_cache_capacity(self) -> dict[str, int]:
        """Return the initialized engine's primary GPU KV-cache capacity."""
        try:
            get_capacity = self.llm.get_kv_cache_capacity
        except AttributeError:
            return {}
        return get_capacity()

    @staticmethod
    def _is_unsupported_encoder_arch(model_path: str) -> bool:
        """Return True if *model_path*'s architecture is not supported by
        TRT-LLM's standalone MultimodalEncoder."""
        try:
            config, _ = PretrainedConfig.get_config_dict(model_path)
            archs = config.get("architectures") or []
            return any(a in _UNSUPPORTED_STANDALONE_ENCODER_ARCHS for a in archs)
        except Exception:
            return False


@asynccontextmanager
async def get_llm_engine(
    engine_args: dict[str, Any],
    disaggregation_mode: Optional[DisaggregationMode] = None,
    component_gauges: Any = None,
    mm_preprocess_processes: int = 0,
) -> AsyncGenerator[TensorRTLLMEngine, None]:
    """Get TensorRT-LLM engine instance with load time tracking.

    Args:
        engine_args: Engine configuration arguments.
        disaggregation_mode: Optional disaggregation mode configuration.
        component_gauges: Optional LLMBackendGauges instance for recording load time.
        mm_preprocess_processes: Processes for multimodal input processing; 0
            keeps it in the worker process.
    """
    # Time engine initialization
    start_time = time.time()

    engine = TensorRTLLMEngine(engine_args, disaggregation_mode)
    processor_calls: list[ProcessorCall] = []
    try:
        if mm_preprocess_processes:
            # The preprocessing processes build their input processors with the
            # LLM's create_input_processor call, so record it as the LLM is built.
            with record_input_processor_calls(trtllm_llm_module) as processor_calls:
                await engine.initialize()
        else:
            await engine.initialize()
        load_time = time.time() - start_time
        logger.debug(f"TensorRT-LLM engine initialized in {load_time:.2f}s")

        # Record model load time immediately after measurement
        if component_gauges:
            component_gauges.set_model_load_time(load_time)

        if mm_preprocess_processes:
            await engine.start_preprocess_pool(mm_preprocess_processes, processor_calls)

        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()
