# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig


@dataclass
class LLMAPIConfig:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        pytorch_backend_config: PyTorchConfig | None = None,
        kv_cache_config: KvCacheConfig | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.pytorch_backend_config = pytorch_backend_config
        self.kv_cache_config = kv_cache_config
        self.extra_args = kwargs

        # Hardcoded to skip tokenizer init for now.
        # We will handle the tokenization/detokenization
        # in the base engine.
        if "skip_tokenizer_init" in self.extra_args:
            self.extra_args.pop("skip_tokenizer_init")
        self.skip_tokenizer_init = True

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pytorch_backend_config": self.pytorch_backend_config,
            "kv_cache_config": self.kv_cache_config,
        }
        if self.extra_args:
            data.update(self.extra_args)
        return data

    def update_sub_configs(self, other_config: Dict[str, Any]):
        if "pytorch_backend_config" in other_config:
            self.pytorch_backend_config = PyTorchConfig(
                **other_config["pytorch_backend_config"]
            )
            self.extra_args.pop("pytorch_backend_config", None)

        if "kv_cache_config" in other_config:
            self.kv_cache_config = KvCacheConfig(**other_config["kv_cache_config"])
            self.extra_args.pop("kv_cache_config", None)


def _get_llm_args(engine_config):
    # Only do model validation checks and leave other checks to LLMAPI
    if "model_name" not in engine_config:
        raise ValueError("Model name is required in the TRT-LLM engine config.")

    if engine_config.get("model_path", ""):
        if os.path.exists(engine_config.get("model_path", "")):
            engine_config["model_path"] = Path(engine_config["model_path"])
        else:
            raise ValueError(f"Model path {engine_config['model_path']} does not exist")

    model_name = engine_config["model_name"]
    model_path = engine_config.get("model_path", None)

    engine_config.pop("model_name")
    engine_config.pop("model_path", None)

    # Store all other args as kwargs
    llm_api_config = LLMAPIConfig(
        model_name=model_name,
        model_path=model_path,
        **engine_config,
    )
    # Parse supported sub configs and remove from kwargs
    llm_api_config.update_sub_configs(engine_config)

    return llm_api_config


def _init_engine_args(engine_args_filepath):
    """Initialize engine arguments from config file."""
    if not os.path.isfile(engine_args_filepath):
        raise ValueError(
            "'YAML file containing TRT-LLM engine args must be provided in when launching the worker."
        )

    try:
        with open(engine_args_filepath) as file:
            trtllm_engine_config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse engine config: {e}")

    return _get_llm_args(trtllm_engine_config)


def parse_tensorrt_llm_args(
    config_args,
) -> Tuple[Any, Tuple[Dict[str, Any], Dict[str, Any]]]:
    parser = argparse.ArgumentParser(description="A TensorRT-LLM Worker parser")
    parser.add_argument(
        "--engine_args", type=str, required=True, help="Path to the engine args file"
    )
    parser.add_argument(
        "--llmapi-disaggregated-config",
        "-c",
        type=str,
        help="Path to the llmapi disaggregated config file",
        default=None,
    )
    parser.add_argument(
        "--router",
        type=str,
        choices=["random", "round-robin", "kv"],
        default="random",
        help="Router type to use for scheduling requests to workers",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers for aggregated (monolith) server",
    )
    parser.add_argument(
        "--min-prefill-workers",
        type=int,
        default=1,
        help="Minimum number of prefill workers for disaggregated server",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Number of tokens per KV block in TRTLLM worker. Default is 32 for pytorch backend.",
    )
    parser.add_argument(
        "--remote-prefill",
        action="store_true",
        help="Use remote prefill workers for generation server in Disaggregated mode.",
    )

    args = parser.parse_args(config_args)
    return (args, _init_engine_args(args.engine_args))


def parse_dynamo_run_args() -> Tuple[Any, Tuple[Dict[str, Any], Dict[str, Any]]]:
    parser = argparse.ArgumentParser(
        description="A TensorRT-LLM Dynamo-run engine parser"
    )
    parser.add_argument(
        "--engine_args", type=str, required=True, help="Path to the engine args file"
    )
    parser.add_argument(
        "--publish-kv-cache-events",
        action="store_true",
        help="Publish KV cache events from TensorRT-LLM. Currently, only supported for context worker in Disaggregated mode.",
    )

    args, _ = parser.parse_known_args()
    return (args, _init_engine_args(args.engine_args))
