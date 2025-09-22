# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import AsyncGenerator

import msgspec
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

from dynamo.runtime.logging import configure_dynamo_logging

from .engine_monitor import VllmEngineMonitor
from .protocol import MyRequestOutput

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
        router_client_container=None,
    ):
        super().__init__(runtime, component, engine, default_sampling_params)
        self.prefill_worker_client = prefill_worker_client
        self.can_prefill = 0
        self._prefill_check_task = None

        # NEW: Router client container enables intelligent prefill worker selection
        # This container holds the FrontendRouterClient which communicates with the KV router
        # to select optimal prefill workers based on KV cache overlap and load balancing.
        # The client is created in background to avoid blocking startup if NATS is slow.
        self.router_client_container = router_client_container

        if router_client_container and router_client_container.ready:
            logger.info(
                "Decode worker initialized with frontend router NATS client for intelligent prefill selection"
            )
        else:
            logger.info(
                "Decode worker initialized - frontend router client will be created in background"
            )

        if self.prefill_worker_client is not None:
            self._prefill_check_task = asyncio.create_task(self._prefill_check_loop())

    def _has_intelligent_routing(self):
        """Check if intelligent prefill routing is available."""
        return (
            self.router_client_container
            and self.router_client_container.ready
            and self.router_client_container.client is not None
        )

    async def _prefill_check_loop(self):
        """Background task that checks prefill worker availability every 5 seconds."""
        while True:
            try:
                if self.prefill_worker_client is not None:
                    self.can_prefill = len(self.prefill_worker_client.instance_ids())
                    logger.debug(f"Current Prefill Workers: {self.can_prefill}")
                else:
                    self.can_prefill = 0
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

        # TODO Change to prefill queue
        if self.can_prefill:
            # Create a copy for prefill with specific modifications
            prefill_sampling_params = deepcopy(sampling_params)

            if prefill_sampling_params.extra_args is None:
                prefill_sampling_params.extra_args = {}
            prefill_sampling_params.extra_args["kv_transfer_params"] = {
                "do_remote_decode": True,
            }
            prefill_sampling_params.max_tokens = 1
            prefill_sampling_params.min_tokens = 1

            prefill_request = {
                "token_ids": request["token_ids"],
                "sampling_params": msgspec.to_builtins(prefill_sampling_params),
                "request_id": request_id,
            }

            try:
                # INTELLIGENT PREFILL WORKER SELECTION:
                # This implementation replaces random round-robin selection with KV cache-aware routing.
                # The system now considers:
                # 1. KV cache overlap: Workers with cached blocks from similar requests are preferred
                # 2. Load balancing: Current worker utilization affects selection
                # 3. Graceful fallback: If intelligent routing fails, falls back to random selection
                # 4. Dynamic availability: Router client may become available after startup
                if self._has_intelligent_routing():
                    try:
                        # Use FrontendRouterClient for NATS-based communication with KV router
                        # Input: request_id (for tracking) + token_ids (for cache overlap analysis)
                        # Output: worker_id of the optimal prefill worker
                        prefill_worker_id = await self.router_client_container.client.select_prefill_worker(
                            request_id, request["token_ids"]
                        )

                        logger.debug(
                            f"Frontend KV router selected prefill worker {prefill_worker_id} for request {request_id}"
                        )

                        # SAFETY CHECK: Ensure the selected worker ID belongs to the prefill component
                        try:
                            prefill_ids = (
                                set(self.prefill_worker_client.instance_ids())
                                if self.prefill_worker_client is not None
                                else set()
                            )
                        except Exception as e:
                            logger.warning(f"Failed to fetch prefill instance IDs: {e}")
                            prefill_ids = set()

                        if prefill_ids and prefill_worker_id not in prefill_ids:
                            logger.warning(
                                f"Router returned non-prefill worker ID {prefill_worker_id}; falling back to round-robin for request {request_id}"
                            )
                            prefill_response = await anext(
                                await self.prefill_worker_client.round_robin(
                                    prefill_request, context=context
                                )
                            )
                        else:
                            # Send request directly to the selected prefill worker (bypassing round-robin)
                            # This ensures the intelligently selected worker processes the request
                            prefill_response = await anext(
                                await self.prefill_worker_client.direct(
                                    prefill_request,
                                    instance_id=prefill_worker_id,
                                    context=context,
                                )
                            )

                    except Exception as router_error:
                        # FALLBACK STRATEGY: If intelligent routing fails for any reason
                        # (network issues, router unavailable, etc.), gracefully degrade to
                        # the original random round-robin selection to maintain system availability
                        logger.warning(
                            f"Frontend router selection failed: {router_error}"
                        )
                        logger.debug(
                            f"Using fallback random prefill worker selection for request {request_id}"
                        )
                        prefill_response = await anext(
                            await self.prefill_worker_client.round_robin(
                                prefill_request, context=context
                            )
                        )
                else:
                    # FALLBACK: No intelligent routing available yet - use original random selection
                    # This handles cases where router client is still being created in background
                    logger.debug(
                        f"Frontend router client not ready, using random prefill worker selection for request {request_id}"
                    )
                    prefill_response = await anext(
                        await self.prefill_worker_client.round_robin(
                            prefill_request, context=context
                        )
                    )
            except Exception as e:
                # TODO: Cancellation does not propagate until the first token is received
                if context.is_stopped() or context.is_killed():
                    logger.debug(f"Aborted Remote Prefill Request ID: {request_id}")
                    # TODO: Raise asyncio.CancelledError into bindings
                    return
                raise e

            prefill_response = MyRequestOutput.model_validate_json(
                prefill_response.data()
            )

            # Modify original sampling_params for decode
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args[
                "kv_transfer_params"
            ] = prefill_response.kv_transfer_params

        try:
            async for tok in self.generate_tokens(prompt, sampling_params, request_id):
                if context.is_stopped() or context.is_killed():
                    await self.engine_client.abort(request_id)
                    logger.debug(f"Aborted Request ID: {request_id}")
                    # TODO: Raise asyncio.CancelledError into bindings
                    break

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
        request_id = request["request_id"]
        logger.debug(f"New Prefill Request ID: {request_id}")

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
        sampling_params = msgspec.convert(request["sampling_params"], SamplingParams)

        try:
            gen = self.engine_client.generate(prompt, sampling_params, request_id)
        except EngineDeadError as e:
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)

        # Generate only 1 token in prefill
        try:
            async for res in gen:
                if context.is_stopped() or context.is_killed():
                    await self.engine_client.abort(request_id)
                    logger.debug(f"Aborted Prefill Request ID: {request_id}")
                    # TODO: Raise asyncio.CancelledError into bindings
                    break

                logger.debug(f"kv transfer params: {res.kv_transfer_params}")
                yield MyRequestOutput(
                    request_id=res.request_id,
                    prompt=res.prompt,
                    prompt_token_ids=res.prompt_token_ids,
                    prompt_logprobs=res.prompt_logprobs,
                    outputs=res.outputs,
                    finished=res.finished,
                    metrics=res.metrics,
                    kv_transfer_params=res.kv_transfer_params,
                ).model_dump_json()
        except asyncio.CancelledError:
            # raise the error because we cannot migrate prefill requests
            raise GeneratorExit(
                "Prefill engine was shut down during token generation"
            ) from None
