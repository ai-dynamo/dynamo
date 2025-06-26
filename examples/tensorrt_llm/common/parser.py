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
from typing import Any, Dict, Tuple


def parse_tensorrt_llm_args(
    config_args,
) -> Tuple[Any, Tuple[Dict[str, Any], Dict[str, Any]]]:
    parser = argparse.ArgumentParser(description="A TensorRT-LLM Worker parser")
    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a YAML file containing additional keyword arguments to pass to the TRTLLM engine.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to disk model or HuggingFace model identifier to load.",
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        help="Name to serve the model under.",
    )
    parser.add_argument(
        "--router",
        type=str,
        choices=["random", "round-robin", "kv"],
        default="random",
        help="Router type to use for scheduling requests to workers",
    )

    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=32,
        help="Number of tokens per KV block in TRTLLM worker. Default is 32 for pytorch backend.",
    )

    parser.add_argument(
        "--enable-disagg",
        action="store_true",
        help="Enable remote prefill for the worker",
    )

    args = parser.parse_args(config_args)
    return args
