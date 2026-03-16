#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional


class InputParamManager:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def get_input_param(self, request: dict, use_tokenizer: bool) -> Optional[Any]:
        """
        Get the input parameter for the request.
        """

        if use_tokenizer:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not available")

            if "messages" in request:
                # Forward chat_template_args / chat_template_kwargs to the
                # template so model-specific variables (e.g. enable_thinking)
                # are available during rendering.
                extra_kwargs = {}
                if "chat_template_kwargs" in request:
                    extra_kwargs.update(request["chat_template_kwargs"])
                if "chat_template_args" in request:
                    extra_kwargs.update(request["chat_template_args"])
                return self.tokenizer.apply_chat_template(
                    request["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
            elif "prompt" in request:
                return self.tokenizer.encode(request["prompt"])
            elif "text" in request:
                return self.tokenizer.encode(request["text"])
            else:
                raise ValueError("No input parameter found in request")
        return request.get("token_ids")
