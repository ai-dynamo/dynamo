# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Use the same runtime arguments and Worker configuration as TokenSpeed."""

import argparse
import json
from pathlib import Path

from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint


class Config(DynamoRuntimeConfig):
    model: str
    served_model_name: str
    component: str
    use_kv_events: bool = False
    mock_token_delay: float
    max_model_len: int
    mock: bool
    engine_args: dict
    rank_worker: bool


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Dynamo DeepInfer worker")
    DynamoRuntimeArgGroup().add_arguments(parser)
    parser.add_argument(
        "--model",
        required=True,
        help="Local DeepInfer checkpoint (tokenizer/config only with --mock)",
    )
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use CPU-only synthetic inference instead of DeepInfer",
    )
    parser.add_argument(
        "--engine-args",
        type=json.loads,
        default={},
        help="JSON object of DeepInfer EngineArgs options",
    )
    parser.add_argument("--mock-token-delay", type=float, default=0.001)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--rank-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not Path(args.model, "tokenizer.json").is_file():
        parser.error(
            "--model must contain tokenizer.json; run the example's prepare_model.py"
        )
    if not args.mock and not Path(args.model, "config.json").is_file():
        parser.error(
            "native DeepInfer requires a local model checkpoint with config.json"
        )
    if not isinstance(args.engine_args, dict):
        parser.error("--engine-args must be a JSON object")
    reserved = {
        "model",
        "tokenizer",
        "tokenizer_revision",
        "served_model_name",
        "max_model_len",
        "engine_type",
        "offline_inference",
    }
    if reserved.intersection(args.engine_args):
        parser.error(
            "use the model directory's tokenizer and Dynamo model/name/context flags; engine_type and offline_inference are fixed"
        )
    if args.mock and args.engine_args:
        parser.error("--engine-args cannot be used with --mock")
    dp_size = args.engine_args.get("data_parallel_size", 1)
    cp_size = args.engine_args.get("context_parallel_size", 1)
    ep_size = args.engine_args.get("expert_parallel_size", 1)
    enp_size = args.engine_args.get("engram_parallel_size", 0)
    rank = args.engine_args.get("engine_rank", 0)
    if any(type(v) is not int or v < 1 for v in (dp_size, cp_size, ep_size)):
        parser.error("DP, CP and EP sizes must be positive integers")
    world_size = dp_size * cp_size
    if world_size % ep_size:
        parser.error("expert_parallel_size must divide DP * CP")
    if type(enp_size) is not int or enp_size not in (0, world_size):
        parser.error("engram_parallel_size must be 0 or DP * CP")
    if type(rank) is not int or not 0 <= rank < dp_size:
        parser.error("engine_rank must be in [0, data_parallel_size)")
    if not args.rank_worker and rank != 0:
        parser.error("the launcher assigns engine_rank; omit it")
    if args.mock_token_delay < 0 or args.max_model_len <= 0:
        parser.error(
            "--mock-token-delay must be nonnegative and --max-model-len positive"
        )
    config = Config.from_cli_args(args)
    config.served_model_name = config.served_model_name or (
        "deepinfer-mock" if config.mock else "deepinfer"
    )
    if config.dyn_structural_tag is None and config.dyn_enable_structural_tag is None:
        config.dyn_structural_tag = False
    config.validate()
    config.namespace, config.component, config.endpoint = parse_endpoint(
        config.endpoint or f"dyn://{config.namespace}.deepinfer.generate"
    )
    # These overrides are published with model registration and run on the frontend.
    config.dyn_tool_call_parser = config.dyn_tool_call_parser or "deepseek_v41"
    config.dyn_reasoning_parser = config.dyn_reasoning_parser or "deepseek_v41"
    if config.structural_tag is not None:
        parser.error("this adapter does not implement constrained decoding")
    return config
