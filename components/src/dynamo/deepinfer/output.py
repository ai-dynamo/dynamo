# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned subset of DeepInfer RequestOutput, safe across engine steps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt_len: int
    token_id: int
    num_output_tokens: int
    finished: bool
    finish_reason: str
