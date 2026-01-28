# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from tensorrt_llm import LLM, MultimodalEncoder
from tensorrt_llm.llmapi.llm import BaseLLM

from dynamo.trtllm.constants import DisaggregationMode

logger = logging.getLogger(__name__)


class Backend(str, enum.Enum):
    """Supported TensorRT-LLM backend types."""

    PYTORCH = "pytorch"
    AUTODEPLOY = "_autodeploy"


class TensorRTLLMEngine:
    def __init__(
        self,
        engine_args,
        disaggregation_mode: Optional[DisaggregationMode] = None,
    ):
        self._llm: Optional[LLM] = None
        self.disaggregation_mode = (
            disaggregation_mode
            if disaggregation_mode is not None
            else DisaggregationMode.AGGREGATED
        )
        # NOTE: `engine_args` may be reused by callers (e.g., for logging or other workers).
        # Copy it so that our internal `pop()` / pruning doesn't leak side effects.
        engine_args = dict(engine_args)
        backend = engine_args.pop("backend", Backend.PYTORCH)
        if backend == Backend.PYTORCH:
            self._llm_cls = LLM
        elif backend == Backend.AUTODEPLOY:
            from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM

            self._llm_cls = AutoDeployLLM
            self._prune_engine_args_for_autodeploy(engine_args)
        else:
            raise ValueError(
                f"Unsupported {backend=}. Available backends: {[b.value for b in Backend]}."
            )

        self.engine_args = engine_args

    async def initialize(self):
        if not self._llm:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                # Initialize the multimodal encoder for full EPD
                # Prefill/decode workers initialize the standard TRT-LLM `LLM` from `engine_args`
                # (model, backend settings, kv cache config, etc.). ENCODE workers instead use
                # TRT-LLM's `MultimodalEncoder`, which has a different constructor surface.
                # We intentionally pass only the supported parameters to avoid unexpected kwargs.
                max_batch_size = self.engine_args.get("max_batch_size", 1)
                model = self.engine_args.get("model")
                logging.info(
                    f"Initializing multimodal encoder with max_batch_size: {max_batch_size}"
                )
                # MultimodalEncoder and LLM both inherit from BaseLLM in TRT-LLM,
                # so storing either in self._llm is valid.
                self._llm = MultimodalEncoder(
                    model=model,
                    max_batch_size=max_batch_size,
                )
            else:
                # Prefill/decode workers: initialize standard TRT-LLM `LLM` with full engine_args
                # (model path, backend settings, KV cache config, disaggregation settings, etc.)
                self._llm = self._llm_cls(**self.engine_args)

    async def cleanup(self):
        if self._llm:
            try:
                self._llm.shutdown()
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
            finally:
                self._llm = None

    @property
    def llm(self) -> BaseLLM:
        if not self._llm:
            raise RuntimeError("Engine not initialized")
        return self._llm

    @staticmethod
    def _prune_engine_args_for_autodeploy(engine_args) -> None:
        """Remove entries from `self.engine_args` that the autodeploy backend does not support."""
        # TODO(2ez4bz/lucaslie): consider handling this in AutoDeploy's `LlmArgs` itself.
        unsupported_fields = [
            # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/auto_deploy/
            # llm_args.py#L313
            "build_config",
            # https://github.com/NVIDIA/TensorRT-LLM/blob/b51258acdd968599b2c3756d5a5326e7d750e7bf/
            # tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py#L384
            "scheduler_config",
            # The below all come from:
            # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc5/tensorrt_llm/_torch/auto_deploy/
            # llm_args.py#L316
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
            "moe_cluster_parallel_size",
            "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
            "enable_attention_dp",
            "cp_config",
        ]
        for field_name in unsupported_fields:
            if engine_args.pop(field_name, None) is not None:
                TensorRTLLMEngine._warn_about_unsupported_field(field_name)

    @staticmethod
    def _warn_about_unsupported_field(field_name: str) -> None:
        logger.warning(
            "`%s` cannot be used with the `_autodeploy` backend. Ignoring.",
            field_name,
        )


@asynccontextmanager
async def get_llm_engine(
    engine_args,
    disaggregation_mode: Optional[DisaggregationMode] = None,
) -> AsyncGenerator[TensorRTLLMEngine, None]:
    engine = TensorRTLLMEngine(engine_args, disaggregation_mode)
    try:
        await engine.initialize()
        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()
