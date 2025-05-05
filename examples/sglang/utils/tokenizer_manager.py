# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Dict, List, Optional, Union

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    SamplingParams,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs


class PatchedTokenizerManager(TokenizerManager):
    # TODO: THIS IS A SIMPLE POC MOVE THIS TO RUST WHEN WE SHIP
    """
    This class acts as a patch for the `TokenizerManager` class in SGLang. Because
    Dynamo has it's own communication primatives - we don't need to use the
    ZMQ initialization and communication. We also don't currently support multimodel
    inputs or embedding APIs.

    Ideally much of this (if not all) can and should get handled in rust land very soon. But
    in order to use sglang helper functions like `v1_chat_generate_request` we need to be able to pass
    in a `TokenizerManager` instance (even though it's not expressly typed as such).
    """

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args

        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            enable_multimodal=server_args.enable_multimodal,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
        self.tokenizer = get_tokenizer(
            tokenizer_name=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )

        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
            input_ids = self.tokenizer.encode(input_text)

        self._validate_token_len(obj, input_ids)
        return self._create_tokenized_object(obj, input_text, input_ids, input_embeds)

    def _validate_token_len(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""

        input_token_num = len(input_ids) if input_ids is not None else 0
        # Check if input alone exceeds context length
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            max_new_tokens is not None
            and (max_new_tokens + input_token_num) >= self.context_len
        ):
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: List[int],
        input_embeds: Optional[Union[List[float], None]] = None,
        image_inputs: Optional[Dict] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters."""

        if self.is_generation:
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            token_ids_logprob = obj.token_ids_logprob
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                lora_path=obj.lora_path,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
            )

        return tokenized_obj
