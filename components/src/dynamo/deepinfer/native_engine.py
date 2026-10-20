# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loaded only for native inference, after the owner thread is started."""

# Load libtorch's shared libraries before importing DeepInfer's extension.
import torch  # noqa: F401
from deepinfer._C.engine import RequestOutput as NativeRequestOutput
from deepinfer.engine.arg_utils import EngineArgs
from deepinfer.engine.llm_engine import LLMEngine

from dynamo.deepinfer.args import Config
from dynamo.deepinfer.output import RequestOutput


class NativeDeepInfer:
    def __init__(self, config: Config):
        args = EngineArgs(
            model=config.model,
            served_model_name=config.served_model_name,
            max_model_len=config.max_model_len,
            **{"data_parallel_size": 1, **config.engine_args},
        )
        self.engine = LLMEngine.from_engine_args(args)

    def add_request(self, **kwargs) -> None:
        # Match DeepInfer's text-only admission path for vision-capable models.
        if self.engine.multi_modal_processor is not None:
            kwargs["multi_modal_data"] = {"image_bytes": [], "image_sizes": []}
        self.engine.add_request(**kwargs)

    def step(self) -> list[RequestOutput]:
        self.has_work = self.engine.has_unfinished_requests()
        raw = self.engine.step() if self.has_work else self.engine.flush_inflight_reqs()
        # Read only needed scalar fields; cloning also copies unused logprob vectors.
        return [
            RequestOutput(
                request_id=NativeRequestOutput.get_request_id(int(ptr)),
                prompt_len=NativeRequestOutput.get_prompt_len(int(ptr)),
                token_id=NativeRequestOutput.get_token_id(int(ptr)),
                num_output_tokens=NativeRequestOutput.get_num_output_tokens(int(ptr)),
                finished=NativeRequestOutput.get_finished(int(ptr)),
                finish_reason=NativeRequestOutput.get_finish_reason(int(ptr)),
            )
            for ptr in raw
        ]

    def abort_request(self, request_id: str) -> None:
        self.engine.abort_request(request_id)

    def close(self) -> None:
        self.engine.close()
