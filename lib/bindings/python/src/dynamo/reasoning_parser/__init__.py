# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Sequence, Tuple


class BaseReasoningParser(Protocol):
    def __init__(self):
        """Initialize the reasoning parser.

        This method should set up any necessary internal state or configurations.

        Signature must not change and must not take any arguments other than self.

        """
        ...

    def detect_and_parse_reasoning(
        self, text: str, token_ids: Sequence[int]
    ) -> Tuple[str, str]:
        """Detect and parse reasoning from the given text and token IDs.

        Args:
            text (str): The input text to analyze.
            token_ids (Sequence[int]): The corresponding token IDs for the text.

        Returns:
            Tuple[str, str]: A tuple containing the parsed  normal text and reasoning if detected.
            Either or both strings can be empty if no reasoning or no normal text is found.
            (normal_text, reasoning_text)
        """
        ...

    def parse_reasoning_streaming_incremental(
        self, text: str, token_ids: Sequence[int]
    ) -> Tuple[str, str]:
        """Parse reasoning from the given text and token IDs in a streaming incremental manner.

        Args:
            text (str): The input text to analyze.
            token_ids (Sequence[int]): The corresponding token IDs for the text.

        Returns:
            Tuple[str, str]: A tuple containing the parsed  normal text and reasoning if detected.
            Either or both strings can be empty if no reasoning or no normal text is found.
            (normal_text, reasoning_text)
        """
        ...
