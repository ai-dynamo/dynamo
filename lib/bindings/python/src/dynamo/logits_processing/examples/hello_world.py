# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizerBase

from .forced_sequence import ForcedSequenceLogitsProcessor

RESPONSE = "Hello world!"


class HelloWorldLogitsProcessor(ForcedSequenceLogitsProcessor):
    """
    Sample Logits Processor that always outputs a hardcoded
    response (`RESPONSE`), no matter the input.

    Thin wrapper over `ForcedSequenceLogitsProcessor`: resolves the
    forced token IDs from the tokenizer at construction time, then
    reuses the shared forced-sequence masking and per-request state.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id; HelloWorldLogitsProcessor requires one."
            )
        super().__init__(tokenizer.encode(RESPONSE, add_special_tokens=False), eos_id)
