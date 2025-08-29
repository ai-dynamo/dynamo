# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Tuple

# import from __init__.py in the same directory
from dynamo.reasoning_parser import BaseReasoningParser


class BasicReasoningParser(BaseReasoningParser):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
    ):
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self._in_reasoning = False
        self.stream_reasoning = True

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse_reasoning(
        self, text: str, _token_ids: Sequence[int]
    ) -> Tuple[str, str]:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        start_idx = text.find(self.think_start_token)
        if start_idx == -1:
            return (text, "")
        normal_prefix = text[:start_idx]
        after_start = text[start_idx + len(self.think_start_token) :]
        end_idx = after_start.find(self.think_end_token)
        if end_idx == -1:
            # Reasoning started but not closed yet
            return (normal_prefix, after_start)
        reasoning_text = after_start[:end_idx]
        normal_suffix = after_start[end_idx + len(self.think_end_token) :]
        return (normal_prefix + normal_suffix, reasoning_text)

    def parse_reasoning_streaming_incremental(
        self, new_text: str, _token_ids: Sequence[int]
    ) -> Tuple[str, str]:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current = self._buffer
        normal_out = ""

        # If not in reasoning, emit normal prefix up to `<think>`
        if not self._in_reasoning:
            start_idx = current.find(self.think_start_token)
            if start_idx == -1:
                self._buffer = ""
                return (current, "")
            normal_out = current[:start_idx]
            current = current[start_idx + len(self.think_start_token) :]
            self._in_reasoning = True
            self.stripped_think_start = True

        # In reasoning: check for `</think>`
        end_idx = current.find(self.think_end_token)
        if end_idx != -1:
            reasoning_delta = current[:end_idx]
            normal_suffix = current[end_idx + len(self.think_end_token) :]
            self._buffer = ""
            self._in_reasoning = False
            self.stripped_think_start = False
            return (normal_out + normal_suffix, reasoning_delta.rstrip())

        # No end yet
        if self.stream_reasoning:
            self._buffer = ""
            return (normal_out, current)
        return (normal_out, "")
