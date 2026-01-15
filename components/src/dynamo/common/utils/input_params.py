#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Any, List


class InputParamManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_input_param(self, request: dict, use_tokenizer: bool):
        """
        Get the input parameter for the request.

        If token_ids is present in the request, use it directly (for token input).
        Otherwise, use the tokenizer to process text input.
        """
        # Check for token_ids first - if present, use directly regardless of use_tokenizer
        if "token_ids" in request:
            return request["token_ids"]

        if use_tokenizer:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not available")

            if "messages" in request:
                return self.tokenizer.apply_chat_template(
                    request["messages"], tokenize=False, add_generation_prompt=True
                )
            elif "prompt" in request:
                return self.tokenizer.encode(request["prompt"])
            elif "text" in request:
                return self.tokenizer.encode(request["text"])
            else:
                raise ValueError("No input parameter found in request")
        raise ValueError("No input parameter found in request")

    def tokenize_chat_request(self, request: Any) -> List[int]:
        """
        Tokenize a chat completion request and return the token IDs.

        This helper function takes an OpenAI-style chat completion request
        (either as a dict or ChatCompletionRequest object), applies the chat
        template using the tokenizer, and returns the resulting token IDs.

        Args:
            request: A chat completion request containing a "messages" field.
                     Can be either a dict with a "messages" key or a
                     ChatCompletionRequest object from sglang.

        Returns:
            List[int]: The token IDs resulting from applying the chat template
                       to the messages.

        Raises:
            ValueError: If the tokenizer is not available or if no messages
                        are found in the request.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available")

        # Handle both dict and object forms
        if hasattr(request, "messages"):
            messages = request.messages
        elif isinstance(request, dict) and "messages" in request:
            messages = request["messages"]
        else:
            raise ValueError("No 'messages' field found in request")

        # Apply chat template with tokenization enabled
        token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        return token_ids
