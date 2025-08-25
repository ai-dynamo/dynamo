# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common base classes and utilities for engine tests (vLLM, TRT-LLM, etc.)"""

from dataclasses import dataclass
from typing import Any, Callable, List

from tests.utils.deployment_graph import Payload

# Common text prompt used across tests
TEXT_PROMPT = "Tell me a short joke about AI."


@dataclass
class EngineConfig:
    """Base configuration for engine test scenarios"""

    name: str
    directory: str
    script_name: str
    marks: List[Any]
    endpoints: List[str]
    response_handlers: List[Callable[[Any], str]]
    model: str
    timeout: int = 600
    delayed_start: int = 0


def create_payload_for_config(config: EngineConfig) -> Payload:
    """Create a standard payload using the model from the engine config.

    This provides the default implementation for text-only models.
    """
    return Payload(
        payload_chat={
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": TEXT_PROMPT,
                }
            ],
            "max_tokens": 150,
            "temperature": 0.1,
            "stream": False,
        },
        payload_completions={
            "model": config.model,
            "prompt": TEXT_PROMPT,
            "max_tokens": 150,
            "temperature": 0.1,
            "stream": False,
        },
        repeat_count=3,
        expected_log=[],
        expected_response=["AI"],
    )
