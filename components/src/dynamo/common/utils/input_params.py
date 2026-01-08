#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


class InputParamManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_input_param(self, request: dict, use_tokenizer: bool):
        """
        Get the input parameter for the request.
        """

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
        return request.get("token_ids")
