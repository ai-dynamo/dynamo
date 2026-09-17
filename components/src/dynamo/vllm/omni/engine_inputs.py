# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine-input contract shared by every request-building path.

Lives in its own module, not in ``omni_handler``, so the per-modality builders
(``audio_handler``, and the model-specific adapters behind them) can annotate
what they return without importing the handler that consumes it.
"""

from dataclasses import dataclass
from typing import Any, Dict, Union

from vllm.lora.request import LoRARequest
from vllm_omni.inputs.data import OmniTextPrompt

from dynamo.common.utils.output_modalities import RequestType


@dataclass
class EngineInputs:
    """Parsed engine inputs ready for AsyncOmni.generate().

    Attributes:
        prompt: OmniTextPrompt dict for the engine.
        sampling_params_list: Per-stage sampling parameters, or None for defaults.
        request_type: The resolved request type (may differ from the initial parse
            when a chat completion request carries video params).
        fps: Frames per second, only meaningful for video requests.
        response_format: Desired response format (e.g. "url" or "b64_json" for
            image requests). None means use the default for the request type.
        output_format: The output format to use for the response.
            None means use the default for the request type.
    """

    prompt: Union[OmniTextPrompt, Dict[str, Any]]
    sampling_params_list: list | None = None
    request_type: RequestType = RequestType.CHAT_COMPLETION
    fps: int = 0
    speed: float = 1.0
    response_format: str | None = None
    output_format: str | None = None
    lora_request: LoRARequest | None = None
    stream_audio: bool = False
