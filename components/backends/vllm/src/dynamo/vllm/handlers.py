# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import AsyncGenerator

import msgspec
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from dynamo.runtime.logging import configure_dynamo_logging

from .protocol import MyRequestOutput

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class BaseWorkerHandler(ABC):
    """
    Request handler for the generate and clear_kv_blocks endpoints.
    """

    def __init__(self, component, engine, default_sampling_params):
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publisher = None

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
        gen = self.engine_client.generate(prompt, sampling_params, request_id)

        num_output_tokens_so_far = 0
        try:
            async for res in gen:
                # res is vllm's RequestOutput

                # This is the expected way for a request to end.
                # The new token ID will be eos, don't forward it.
                if res.finished:
                    yield {"finish_reason": "stop", "token_ids": []}
                    break

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


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self, component, engine, default_sampling_params, prefill_worker_client=None
    ):
        super().__init__(component, engine, default_sampling_params)
        self.prefill_worker_client = prefill_worker_client
        self.can_prefill = 0
        self._prefill_check_task = None

        if self.prefill_worker_client is not None:
            self._prefill_check_task = asyncio.create_task(self._prefill_check_loop())

    async def _prefill_check_loop(self):
        """Background task that checks prefill worker availability every 5 seconds."""
        while True:
            try:
                if self.prefill_worker_client is not None:
                    self.can_prefill = len(self.prefill_worker_client.instance_ids())
                    logger.debug(f"Current Prefill Workers: {self.can_prefill}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in prefill check loop: {e}")
                await asyncio.sleep(5)  # Still sleep on error to avoid tight loop

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

            # TODO Change to prefill queue
            if self.prefill_worker_client is not None:
                try:
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

        async for tok in self.generate_tokens(prompt, sampling_params, request_id):
            if context.is_stopped() or context.is_killed():
                await self.engine_client.abort(request_id)
                logger.debug(f"Aborted Request ID: {request_id}")
                # TODO: Raise asyncio.CancelledError into bindings
                break

            yield tok


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(self, component, engine, default_sampling_params):
        super().__init__(component, engine, default_sampling_params)

    async def generate(self, request, context):
        request_id = request["request_id"]
        logger.debug(f"New Prefill Request ID: {request_id}")

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])
        sampling_params = msgspec.convert(request["sampling_params"], SamplingParams)

        gen = self.engine_client.generate(prompt, sampling_params, request_id)

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
