# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni handler for text-to-text generation using vLLM-Omni orchestrator."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict

from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm_omni.entrypoints import AsyncOmni

from dynamo.vllm.handlers import BaseWorkerHandler, build_sampling_params

logger = logging.getLogger(__name__)


class OmniHandler(BaseWorkerHandler):
    """Handler for multi-stage pipelines using vLLM-Omni's AsyncOmni orchestrator."""

    def __init__(
        self,
        runtime,
        component,
        config,
        default_sampling_params: Dict[str, Any],
        shutdown_event: asyncio.Event | None = None,
    ):
        """Initialize handler with AsyncOmni orchestrator."""
        logger.info(
            f"Initializing OmniHandler for multi-stage pipelines with model: {config.model}"
        )

        # Initialize AsyncOmni with stage configuration
        # Note: stage_configs_path is validated as required in args.py
        logger.info(f"Using stage config from: {config.stage_configs_path}")

        omni_kwargs = {
            "model": config.model,
            "trust_remote_code": config.engine_args.trust_remote_code,
            "stage_configs_path": config.stage_configs_path,
        }

        self.engine_client = AsyncOmni(**omni_kwargs)

        # Initialize attributes needed from BaseWorkerHandler
        # We don't call super().__init__() because VllmEngineMonitor expects AsyncLLM,
        # but AsyncOmni manages its own engines internally

        # TODO: Kv publishers not supported yet
        # TODO: Adopt to baseworker initialization pattern
        self.default_sampling_params = default_sampling_params
        self.config = config
        self.model_max_len = config.engine_args.max_model_len
        self.shutdown_event = shutdown_event
        logger.info("OmniHandler initialized successfully for text-to-text generation")

    async def generate(
        self, request: Dict[str, Any], context
    ) -> AsyncGenerator[Dict, None]:
        """Generate text using AsyncOmni orchestrator. Currently supports text-to-text only."""
        request_id = context.id()
        logger.debug(f"Omni Request ID: {request_id}")

        # Extract token_ids from internal protocol format
        token_ids = request.get("token_ids")
        if not token_ids:
            logger.error(f"Request {request_id}: No token_ids found in request")
            yield {
                "finish_reason": "error: No token_ids in request",
                "token_ids": [],
            }
            return

        logger.info(
            f"Request {request_id}: Generating text for {len(token_ids)} input tokens"
        )

        # Build sampling parameters from request
        sampling_params = self._build_sampling_params(request)
        sampling_params_list = [sampling_params]

        tokens_prompt: TokensPrompt = {
            "prompt_token_ids": token_ids,
        }

        async with self._abort_monitor(context, request_id):
            try:
                num_output_tokens_so_far = 0

                async for stage_output in self.engine_client.generate(
                    prompt=tokens_prompt,  # Pass TokensPrompt format
                    request_id=request_id,
                    sampling_params_list=sampling_params_list,
                ):
                    # stage_output is OmniRequestOutput
                    # For text generation: stage_output.request_output is a single vLLM RequestOutput
                    if (
                        stage_output.final_output_type == "text"
                        and stage_output.request_output
                    ):
                        vllm_output = stage_output.request_output

                        if not vllm_output.outputs:
                            logger.warning(f"Request {request_id} returned no outputs")
                            yield {
                                "finish_reason": "error: No outputs from vLLM engine",
                                "token_ids": [],
                            }
                            break

                        output = vllm_output.outputs[0]
                        next_total_toks = len(output.token_ids)

                        out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}

                        if output.finish_reason:
                            out["finish_reason"] = self._normalize_finish_reason(
                                output.finish_reason
                            )
                            out["completion_usage"] = self._build_completion_usage(
                                vllm_output
                            )
                            logger.debug(
                                f"Completed generation for request {request_id}: "
                                f"{next_total_toks} output tokens, finish_reason={output.finish_reason}"
                            )

                        if output.stop_reason:
                            out["stop_reason"] = output.stop_reason

                        yield out
                        num_output_tokens_so_far = next_total_toks

            except GeneratorExit:
                # Shutdown was triggered during generation
                logger.info(f"Request {request_id} aborted due to shutdown")
                raise
            except Exception as e:
                logger.error(f"Error during generation for request {request_id}: {e}")
                yield {
                    "finish_reason": f"error: {str(e)}",
                    "token_ids": [],
                }

    def _build_sampling_params(self, request: Dict[str, Any]) -> SamplingParams:
        """Build sampling params using shared handler utility."""
        return build_sampling_params(
            request, self.default_sampling_params, self.model_max_len
        )

    def cleanup(self):
        """Cleanup AsyncOmni orchestrator resources."""
        try:
            if hasattr(self, "engine_client"):
                self.engine_client.close()
                logger.info("AsyncOmni orchestrator closed")
        except Exception as e:
            logger.error(f"Error closing AsyncOmni orchestrator: {e}")
