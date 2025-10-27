# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.runtime.logging import configure_dynamo_logging

from .engine_monitor import VllmEngineMonitor

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def build_sampling_params(
    request: Dict[str, Any], default_sampling_params: Dict[str, Any]
) -> SamplingParams:
    """
    Build SamplingParams from a PreprocessedRequest.

    Args:
        request: The PreprocessedRequest dict with 'sampling_options' and 'stop_conditions'
        default_sampling_params: Default sampling parameters to initialize with

    Returns:
        SamplingParams configured from the request
    """
    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = False

    # Apply sampling_options
    for key, value in request["sampling_options"].items():
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)

    # Apply stop_conditions
    for key, value in request["stop_conditions"].items():
        if value is not None and hasattr(sampling_params, key):
            # Do not add stop key to sampling params - dynamo handles stop conditions directly
            if key == "stop":
                continue
            setattr(sampling_params, key, value)

    return sampling_params


class BaseWorkerHandler(ABC):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, runtime, component, engine, default_sampling_params):
        self.runtime = runtime
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers = None
        self.engine_monitor = VllmEngineMonitor(runtime, engine)

    @abstractmethod
    async def generate(self, request, context) -> AsyncGenerator[dict, None]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill):
        """Background task that monitors for context cancellation and aborts the request."""
        try:
            await context.async_killed_or_stopped()
            # If we reach here, the context was stopped or killed
            await self.engine_client.abort(request_id)
            logger.debug(
                f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
            )
        except asyncio.CancelledError:
            # Task was cancelled, normal cleanup if not aborted
            pass
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(self, context, request_id, is_prefill=False):
        """Context manager that creates and automatically cleans up an abort monitoring task."""
        task = asyncio.create_task(self._monitor_abort(context, request_id, is_prefill))
        try:
            yield task
        finally:
            # Cancel the abort monitoring task when exiting the context
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def cleanup(self):
        """Override in subclasses if cleanup is needed."""
        pass

    async def generate_tokens(
        self, prompt, sampling_params, request_id, data_parallel_rank=None
    ):
        try:
            gen = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
                data_parallel_rank=data_parallel_rank,
            )

            num_output_tokens_so_far = 0
            try:
                async for res in gen:
                    # res is vllm's RequestOutput

                    if not res.outputs:
                        yield {"finish_reason": "error", "token_ids": []}
                        break

                    output = res.outputs[0]
                    next_total_toks = len(output.token_ids)
                    out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
                    if output.finish_reason:
                        out["finish_reason"] = output.finish_reason
                    if output.stop_reason:
                        out["stop_reason"] = output.stop_reason
                    yield out
                    num_output_tokens_so_far = next_total_toks
            except asyncio.CancelledError:
                # raise EngineShGeneratorExit when engine exits so that frontend can migrate the request
                raise GeneratorExit(
                    "Decode engine was shut down during token generation"
                ) from None

        except EngineDeadError as e:
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        component,
        engine,
        default_sampling_params,
    ):
        super().__init__(runtime, component, engine, default_sampling_params)

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation
        request_id = context.id()
        logger.debug(f"Decode Request ID: {request_id}")

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])

        # Build sampling params from request
        sampling_params = build_sampling_params(request, self.default_sampling_params)

        # Extract disaggregated_params from request (set by prefill router in Rust frontend)
        disaggregated_params = request.get("disaggregated_params")
        if disaggregated_params:
            # Prefill was performed - use the disaggregated params
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = disaggregated_params.get(
                "kv_transfer_params"
            )
            logger.debug(
                f"Using disaggregated params from prefill for request {request_id}"
            )

        dp_rank = request.get("dp_rank", None)

        async with self._abort_monitor(context, request_id):
            try:
                async for tok in self.generate_tokens(
                    prompt, sampling_params, request_id, data_parallel_rank=dp_rank
                ):
                    yield tok
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, runtime, component, engine, default_sampling_params):
        super().__init__(runtime, component, engine, default_sampling_params)

    async def generate(self, request, context):
        # Use context ID for request tracking and correlation with decode phase
        request_id = context.id()
        logger.debug(f"Prefill Request ID: {request_id}")

        token_ids = request["token_ids"]
        prompt = TokensPrompt(prompt_token_ids=token_ids)

        # Build sampling params from request using shared utility
        sampling_params = build_sampling_params(request, self.default_sampling_params)

        # Configure for prefill-only mode with remote decode
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args["kv_transfer_params"] = {
            "do_remote_decode": True,
        }
        # Override for prefill: only generate 1 token
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1

        dp_rank = request.get("dp_rank", None)

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self.engine_client.generate(
                    prompt, sampling_params, request_id, data_parallel_rank=dp_rank
                )
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)

            try:
                async for res in gen:
                    logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                    token_ids = res.outputs[0].token_ids if res.outputs else []

                    output: Dict[str, Any] = {
                        "token_ids": list(token_ids),
                        "disaggregated_params": (
                            {"kv_transfer_params": res.kv_transfer_params}
                            if res.kv_transfer_params
                            else None
                        ),
                    }

                    yield output
            except asyncio.CancelledError:
                # raise the error because we cannot migrate prefill requests
                raise GeneratorExit(
                    "Prefill engine was shut down during token generation"
                ) from None
