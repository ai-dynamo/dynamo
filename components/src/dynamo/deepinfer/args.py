# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for the experimental DeepInfer backend."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.runtime import parse_endpoint

DEFAULT_ENDPOINT_COMPONENT = "backend"
DEFAULT_ENDPOINT_NAME = "generate"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


class Config(DynamoRuntimeConfig):
    component: str
    disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED
    use_kv_events: bool = False

    model: str
    served_model_name: Optional[str] = None
    expert_parallel_size: int = 8
    context_parallel_size: int = 8
    max_model_len: int = 32768
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 32768
    gpu_memory_utilization: float = 0.85
    device_offset: int = 0

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        if self.disaggregation_mode != DisaggregationMode.AGGREGATED:
            raise ValueError(
                "The DeepInfer backend only supports aggregated serving; "
                f"got disaggregation mode {self.disaggregation_mode.value!r}"
            )
        for name in (
            "expert_parallel_size",
            "context_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
        if self.context_parallel_size % self.expert_parallel_size:
            raise ValueError(
                "--context-parallel-size must be a multiple of --expert-parallel-size "
                "so that a single attention data-parallel rank owns every GPU"
            )
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("--gpu-memory-utilization must be in (0, 1]")


def _add_deepinfer_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("DeepInfer Engine Options")
    group.add_argument(
        "--model",
        default=os.environ.get("DEEPINFER_HF_MODEL_DIR", DEFAULT_MODEL),
        help="Hugging Face model id or local checkpoint directory.",
    )
    group.add_argument(
        "--served-model-name",
        default=None,
        help="Model name advertised to clients. Defaults to --model.",
    )
    group.add_argument(
        "--expert-parallel-size",
        type=int,
        default=Config.expert_parallel_size,
        help="MoE expert-parallel size. Must divide --context-parallel-size.",
    )
    group.add_argument(
        "--context-parallel-size",
        type=int,
        default=Config.context_parallel_size,
        help=(
            "Balanced context-parallel size. The single attention data-parallel "
            "rank spawns this many workers, one per GPU."
        ),
    )
    group.add_argument("--max-model-len", type=int, default=Config.max_model_len)
    group.add_argument("--max-num-seqs", type=int, default=Config.max_num_seqs)
    group.add_argument(
        "--max-num-batched-tokens", type=int, default=Config.max_num_batched_tokens
    )
    group.add_argument(
        "--gpu-memory-utilization", type=float, default=Config.gpu_memory_utilization
    )
    group.add_argument(
        "--device-offset",
        type=int,
        default=int(os.environ.get("DEVICE_OFFSET", "0")),
        help="First physical GPU index used by this worker.",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    cli_args = list(argv) if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Dynamo DeepInfer worker (experimental, aggregated only)",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    _add_deepinfer_arguments(parser)

    config = Config.from_cli_args(parser.parse_args(cli_args))
    config.served_model_name = config.served_model_name or config.model
    config.validate()

    endpoint = (
        config.endpoint
        or f"dyn://{config.namespace}.{DEFAULT_ENDPOINT_COMPONENT}.{DEFAULT_ENDPOINT_NAME}"
    )
    namespace, component, endpoint_name = parse_endpoint(endpoint)
    config.namespace = namespace
    config.component = component
    config.endpoint = endpoint_name
    return config
