# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict

import msgspec
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.runtime.logging import configure_dynamo_logging

from .engine_monitor import VllmEngineMonitor

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class BaseWorkerHandler(ABC):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, runtime, component, engine, default_sampling_params):
        self.runtime = runtime
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publisher = None
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

    async def generate_tokens(self, prompt, sampling_params, request_id):
        try:
            gen = self.engine_client.generate(prompt, sampling_params, request_id)

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
        prefill_worker_client=None,
        prefill_router_client=None,
    ):
        super().__init__(runtime, component, engine, default_sampling_params)
        self.prefill_worker_client = prefill_worker_client
        self.prefill_router_client = prefill_router_client
        self.can_prefill = 0
        self._prefill_check_task = None

        if self.prefill_worker_client or self.prefill_router_client:
            self._prefill_check_task = asyncio.create_task(self._prefill_check_loop())

    async def _prefill_check_loop(self):
        """Background task that checks prefill router/worker availability every 5 seconds."""
        while True:
            try:
                router_count = (
                    len(self.prefill_router_client.instance_ids())
                    if self.prefill_router_client is not None
                    else 0
                )
                worker_count = (
                    len(self.prefill_worker_client.instance_ids())
                    if self.prefill_worker_client is not None
                    else 0
                )
                self.can_prefill = max(router_count, worker_count)
                logger.debug(
                    f"Prefill availability - Routers: {router_count}, Workers: {worker_count}"
                )
            except asyncio.CancelledError:
                logger.warning("Prefill check loop cancelled.")
                raise
            except Exception as e:
                logger.error(f"Error in prefill check loop: {e}")

            await asyncio.sleep(5)

    def cleanup(self):
        """Cancel background tasks."""
        if self._prefill_check_task is not None:
            self._prefill_check_task.cancel()
        super().cleanup()

    async def generate(self, request, context):
        request_id = str(uuid.uuid4().hex)
        logger.debug(f"New Request ID: {request_id}")

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])

        sampling_params = SamplingParams(**self.default_sampling_params)

        sampling_params.detokenize = False
        for key, value in request["sampling_options"].items():
            if value is not None and hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        for key, value in request["stop_conditions"].items():
            if value is not None and hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        # Use prefill router or worker if available
        if self.can_prefill:
            # Create prefill sampling params with modifications
            prefill_sampling_params = deepcopy(sampling_params)
            if prefill_sampling_params.extra_args is None:
                prefill_sampling_params.extra_args = {}
            prefill_sampling_params.extra_args["kv_transfer_params"] = {
                "do_remote_decode": True,
            }
            prefill_sampling_params.max_tokens = 1
            prefill_sampling_params.min_tokens = 1

            try:
                # Send request with sampling_params and request_id in extra_args
                prefill_request = request.copy()
                # TODO (PeaBrane): this smells a bit bad as not we have two nestings
                # of extra_args (an inner one again in sampling_params)
                prefill_request["extra_args"] = {
                    "sampling_params": msgspec.to_builtins(prefill_sampling_params),
                    "request_id": request_id,
                }

                # Try router first if available, fallback to worker
                if (
                    self.prefill_router_client is not None
                    and self.prefill_router_client.instance_ids()
                ):
                    # Call router's generate endpoint which returns LLMEngineOutput
                    prefill_response = await anext(
                        await self.prefill_router_client.generate(
                            prefill_request, context=context
                        )
                    )
                elif self.prefill_worker_client is not None:
                    # Fallback to direct worker with same format
                    prefill_response = await anext(
                        await self.prefill_worker_client.round_robin(
                            prefill_request, context=context
                        )
                    )
                else:
                    raise ValueError("No prefill router or worker available")

                prefill_output = prefill_response.data()

                # Extract kv_transfer_params from response
                kv_transfer_params = prefill_output.get("extra_args", {}).get(
                    "kv_transfer_params"
                )
                if kv_transfer_params:
                    if sampling_params.extra_args is None:
                        sampling_params.extra_args = {}
                    sampling_params.extra_args[
                        "kv_transfer_params"
                    ] = kv_transfer_params

            except Exception as e:
                if context.is_stopped() or context.is_killed():
                    logger.debug(f"Aborted Remote Prefill Request ID: {request_id}")
                    return
                logger.warning(f"Prefill error: {e}, falling back to local prefill")

        async with self._abort_monitor(context, request_id):
            try:
                async for tok in self.generate_tokens(
                    prompt, sampling_params, request_id
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
        # Extract from PreprocessedRequest format - request_id and sampling_params from extra_args
        extra_args = request.get("extra_args", {})
        request_id = extra_args.get("request_id", str(uuid.uuid4().hex))
        logger.debug(f"New Prefill Request ID: {request_id}")

        token_ids = request["token_ids"]
        prompt = TokensPrompt(prompt_token_ids=token_ids)

        # Get sampling_params from extra_args
        sampling_params_dict = extra_args.get("sampling_params", {})
        sampling_params = msgspec.convert(sampling_params_dict, SamplingParams)

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self.engine_client.generate(prompt, sampling_params, request_id)
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
                        "extra_args": (
                            {"kv_transfer_params": res.kv_transfer_params}
                            if res.kv_transfer_params
                            else {}
                        ),
                    }

                    yield output
            except asyncio.CancelledError:
                # raise the error because we cannot migrate prefill requests
                raise GeneratorExit(
                    "Prefill engine was shut down during token generation"
                ) from None
