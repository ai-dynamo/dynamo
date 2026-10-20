# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock only DeepInfer's add_request/step/abort_request token interface.

No DeepInfer, Torch, CUDA, model weights, or OpenAI response objects are used.
The RequestOutput fields mirror DeepInfer's csrc/engine/outputs.hpp. Unlike the
native pooled pointers, these Python records own their values.
"""

from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt_len: int
    token_id: int
    num_output_tokens: int
    finished: bool
    finish_reason: str


@dataclass
class PendingRequest:
    prompt_len: int
    token_ids: list[int]
    finish_reason: str
    emitted: int = 0


class MockDeepInfer:
    def __init__(self, model: str):
        self.tokenizer = Tokenizer.from_file(f"{model}/tokenizer.json")
        self.requests: dict[str, PendingRequest] = {}

    def add_request(
        self,
        *,
        user_id: str,
        request_id: str,
        sampling_params: dict,
        prompt_token_ids: np.ndarray,
    ) -> None:
        del user_id
        if request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request_id}")
        prompt = self.tokenizer.decode(
            prompt_token_ids.tolist(), skip_special_tokens=False
        )
        output = mock_response(prompt)
        if prompt.endswith("<think>"):
            # The renderer has already supplied the reasoning opening token.
            output = (
                output.removeprefix("<think>")
                if output.startswith("<think>")
                else "</think>" + output
            )
        elif prompt.endswith("</think>") and output.startswith("<think>"):
            output = output.split("</think>", 1)[1]
        token_ids = self.tokenizer.encode(output, add_special_tokens=False).ids
        max_tokens = sampling_params["max_tokens"]
        finish_reason = "length" if len(token_ids) > max_tokens else "stop"
        self.requests[request_id] = PendingRequest(
            len(prompt_token_ids), token_ids[:max_tokens], finish_reason
        )

    def step(self) -> list[RequestOutput]:
        outputs = []
        for request_id, request in list(self.requests.items()):
            token_id = request.token_ids[request.emitted]
            request.emitted += 1
            finished = request.emitted == len(request.token_ids)
            outputs.append(
                RequestOutput(
                    request_id,
                    request.prompt_len,
                    token_id,
                    request.emitted,
                    finished,
                    request.finish_reason if finished else "",
                )
            )
            if finished:
                del self.requests[request_id]
        return outputs

    def abort_request(self, request_id: str) -> None:
        self.requests.pop(request_id, None)


def mock_response(prompt: str) -> str:
    """Explicit markers select fixtures; this mock does not understand language."""
    prompt = prompt.rsplit("<｜User｜>", 1)[-1]
    # Tool results precede the final assistant prefix in the rendered prompt.
    if "DEEPINFER_TOOL_RESULT" in prompt:
        return "<think>The tool returned the weather.</think>It is 21 C in Paris."
    if "DEEPINFER_PARALLEL" in prompt or "DEEPINFER_TOOL" in prompt:
        cities = ["Paris", "Tokyo"] if "DEEPINFER_PARALLEL" in prompt else ["Paris"]
        calls = "\n".join(
            '<｜DSML｜ invoke name="get_weather">'
            '<｜DSML｜ parameter name="city" string="true">'
            f"{city}</｜DSML｜ parameter></｜DSML｜ invoke>"
            for city in cities
        )
        return (
            "<think>I need the weather tool.</think>"
            f"<｜DSML｜ calls>{calls}</｜DSML｜ calls>"
        )
    if "DEEPINFER_REASONING" in prompt:
        return "<think>I will compute six times seven.</think>The answer is 42."
    return "Hello from the DeepInfer mock."
