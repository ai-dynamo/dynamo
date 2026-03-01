# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for disaggregated params encoding, decoding, and manipulation.

These functions handle the serialization and state management of
DisaggregatedParams objects for prefill/decode disaggregation.
"""

import base64
import dataclasses
import logging
from dataclasses import asdict
from typing import Any, Optional

from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.trtllm.constants import DisaggregationMode


class DisaggregatedParamsCodec:
    """
    Codec for encoding and decoding disaggregated params for network transfer.
    """

    @staticmethod
    def decode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        opaque_state = disaggregated_params.opaque_state
        if isinstance(opaque_state, str):
            opaque_state = base64.b64decode(opaque_state)
        return dataclasses.replace(disaggregated_params, opaque_state=opaque_state)

    @staticmethod
    def encode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        opaque_state = disaggregated_params.opaque_state
        if isinstance(opaque_state, (bytes, bytearray)):
            opaque_state = base64.b64encode(opaque_state).decode("utf-8")
        return dataclasses.replace(disaggregated_params, opaque_state=opaque_state)


class DisaggregatedParamsUtils:
    """
    Utility functions for disaggregated params manipulation.

    These are pure functions that handle the logic of setting up,
    encoding, and decoding disaggregated params for P/D disaggregation.
    """

    @staticmethod
    def decode_from_prefill(
        prefill_result: dict,
    ) -> tuple[DisaggregatedParams, dict]:
        """
        Extract and decode disaggregated params from prefill_result.

        Args:
            prefill_result: Result from prefill worker containing encoded disaggregated params

        Returns:
            Tuple of (disaggregated_params, epd_metadata) where:
            - disaggregated_params: Decoded DisaggregatedParams object with request_type="generation_only"
            - epd_metadata: Dictionary containing EPD-specific metadata
        """
        params_dict = prefill_result["disaggregated_params"].copy()

        # Remove worker_id if present (added by prefill worker, not needed for decode)
        params_dict.pop("worker_id", None)

        # Extract EPD metadata that was packed by prefill worker
        epd_metadata = {}
        if "_epd_metadata" in params_dict:
            epd_metadata = params_dict.pop("_epd_metadata")
            logging.debug(
                f"DECODE: Extracted _epd_metadata with {len(epd_metadata)} fields"
            )

        # Decode the disaggregated params
        disaggregated_params = DisaggregatedParamsCodec.decode(
            DisaggregatedParams(**params_dict)
        )
        # Set to generation_only mode for decode phase
        disaggregated_params.request_type = "generation_only"

        # In generation-only mode, multimodal embeddings are already processed and in KV cache
        # Remove multimodal_embedding_handles to avoid TRT-LLM validation error
        if (
            hasattr(disaggregated_params, "multimodal_embedding_handles")
            and disaggregated_params.multimodal_embedding_handles
        ):
            disaggregated_params.multimodal_embedding_handles = None

        logging.debug("DECODE: Set request_type to generation_only")

        return disaggregated_params, epd_metadata

    @staticmethod
    def encode_and_pack(
        output: Any,
        disaggregated_params: Any,
        request: dict,
        res: Any,
        processed_input: Any = None,
    ) -> Optional[dict]:
        """
        Encode and pack disaggregated params for PREFILL mode response.

        Handles:
        - Choosing between output and input disaggregated params
        - Preserving multimodal_embedding_handles in EPD flow
        - Encoding params for transmission
        - Packing prefill metadata for DECODE optimization

        Args:
            output: GenerationResult from the engine (has disaggregated_params attribute)
            disaggregated_params: Input disaggregated params
            request: Original request dict
            res: RequestOutput object with prompt and prompt_token_ids attributes
            processed_input: The processed input dict from process_openai_request

        Returns:
            Dictionary with encoded disaggregated params, or None if encoding failed
        """
        # In EPD flow, output.disaggregated_params might be None, use the input params
        params_to_encode = (
            output.disaggregated_params
            if output.disaggregated_params is not None
            else disaggregated_params
        )

        # In EPD flow, manually preserve multimodal_embedding_handles from input
        # because TRT-LLM engine may not propagate them through prefill
        if params_to_encode is not None and disaggregated_params is not None:
            input_handles = getattr(
                disaggregated_params,
                "multimodal_embedding_handles",
                None,
            )
            output_handles = getattr(
                params_to_encode, "multimodal_embedding_handles", None
            )

            if input_handles is not None and output_handles is None:
                params_to_encode.multimodal_embedding_handles = input_handles
                # Also preserve hashes if they exist
                input_hashes = getattr(disaggregated_params, "multimodal_hashes", None)
                if input_hashes is not None:
                    params_to_encode.multimodal_hashes = input_hashes

        encoded_params = DisaggregatedParamsCodec.encode(params_to_encode)

        if encoded_params is None:
            logging.error("PREFILL: encoded_params is None - decode worker will fail!")
            return None

        logging.debug("PREFILL: Successfully encoded disaggregated params")
        params_dict = asdict(encoded_params)

        # Pack prefill metadata for DECODE worker optimization
        prefill_metadata = DisaggregatedParamsUtils.build_prefill_metadata(
            request, res, processed_input
        )

        # Add metadata to the disaggregated_params dict
        if prefill_metadata:
            params_dict["_epd_metadata"] = prefill_metadata

        return params_dict

    @staticmethod
    def build_prefill_metadata(
        request: dict,
        res: Any,
        processed_input: Any,
    ) -> dict:
        """
        Build prefill metadata for DECODE worker optimization.

        Args:
            request: Original request dict
            res: RequestOutput object with prompt and prompt_token_ids attributes
            processed_input: The processed input dict from process_openai_request

        Returns:
            Dictionary with prefill metadata
        """
        prefill_metadata = {}

        # ALWAYS pack prompt info for DECODE to skip re-processing
        # Per TRT-LLM team: DECODE never needs to reload images - KV cache has the context
        if (
            processed_input
            and isinstance(processed_input, dict)
            and processed_input.get("prompt")
        ):
            prefill_metadata["_prefill_prompt"] = processed_input["prompt"]
        elif res.prompt:
            prefill_metadata["_prefill_prompt"] = res.prompt

        if res.prompt_token_ids:
            prefill_metadata["_prefill_prompt_token_ids"] = list(res.prompt_token_ids)

        # EPD-specific: use encoder's prompt if available
        if "_epd_processed_prompt" in request and res.prompt:
            prefill_metadata["_epd_processed_prompt"] = res.prompt
        if "_epd_prompt_token_ids" in request and res.prompt_token_ids:
            prefill_metadata["_epd_prompt_token_ids"] = list(res.prompt_token_ids)

        return prefill_metadata

    @staticmethod
    def setup_for_mode(
        disaggregation_mode: DisaggregationMode,
        request: dict,
        ep_disaggregated_params: Optional[Any],
    ) -> tuple[Any, Any, dict]:
        """
        Setup disaggregated_params based on PREFILL/DECODE mode.

        For PREFILL mode:
        - Uses ep_disaggregated_params from encode worker if available
        - Otherwise creates new DisaggregatedParams with request_type="context_only"

        For DECODE mode:
        - Decodes disaggregated_params from prefill_result
        - Extracts EPD metadata for prompt optimization

        Args:
            disaggregation_mode: Current mode (PREFILL, DECODE, etc.)
            request: Request dictionary (may contain prefill_result)
            ep_disaggregated_params: Optional params from encode worker (EPD flow)

        Returns:
            Tuple of (disaggregated_params, ep_disaggregated_params, epd_metadata)
        """
        disaggregated_params = None
        epd_metadata = {}

        # PREFILL mode: setup context_only params
        if disaggregation_mode == DisaggregationMode.PREFILL:
            if ep_disaggregated_params:
                ep_disaggregated_params.request_type = "context_only"
                disaggregated_params = ep_disaggregated_params
            else:
                disaggregated_params = DisaggregatedParams(request_type="context_only")

        # DECODE mode: decode params from prefill_result
        prefill_result = request.get("prefill_result")
        if prefill_result and "disaggregated_params" in prefill_result:
            (
                disaggregated_params,
                epd_metadata,
            ) = DisaggregatedParamsUtils.decode_from_prefill(prefill_result)
            # For full EPD flow, make decoded params available to multimodal processor
            ep_disaggregated_params = disaggregated_params

        return disaggregated_params, ep_disaggregated_params, epd_metadata
