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

import logging
from copy import deepcopy
from typing import Literal

from dynamo.planner.defaults import WORKER_COMPONENT_NAMES

from .defaults import DEFAULT_MODEL_NAME, DYNAMO_RUN_DEFAULT_PORT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def break_arguments(args: list[str]) -> list[str]:
    ans = []
    if isinstance(args, str):
        ans = args.split(" ")
    else:
        for arg in args:
            ans.extend(arg.split(" "))
    return ans


def join_arguments(args: list[str]) -> list[str]:
    return [" ".join(args)]


def append_argument(args: list[str], to_append) -> list[str]:
    idx = find_arg_index(args)
    if isinstance(to_append, list):
        args[idx:idx] = to_append
    else:
        args.insert(idx, to_append)
    return args


def find_arg_index(args: list[str]) -> int:
    # find the correct index to insert an argument
    idx = len(args)

    try:
        new_idx = args.index("|")
        idx = min(idx, new_idx)
    except ValueError:
        pass

    try:
        new_idx = args.index("2>&1")
        idx = min(idx, new_idx)
    except ValueError:
        pass

    return idx


class VllmV1ConfigModifier:
    @classmethod
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        config = deepcopy(config)

        # set metadata name
        config["metadata"]["name"] = "vllm-v1-agg"

        # disable planner
        if "Planner" in config["spec"]["services"]:
            del config["spec"]["services"]["Planner"]

        if target == "prefill":
            # convert prefill worker into decode worker
            config["spec"]["services"][
                WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker
            ] = config["spec"]["services"][
                WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker
            ]
            del config["spec"]["services"][
                WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker
            ]

            args = config["spec"]["services"][
                WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker
            ]["extraPodSpec"]["mainContainer"]["args"]

            args = break_arguments(args)

            # remove --is-prefill-worker flag
            args.remove("--is-prefill-worker")

            # disable prefix caching
            if "--enable-prefix-caching" in args:
                args.remove("--enable-prefix-caching")
            if "--no-enable-prefix-caching" not in args:
                args = append_argument(args, "--no-enable-prefix-caching")

            config["spec"]["services"][WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
                "extraPodSpec"
            ]["mainContainer"]["args"] = join_arguments(args)

        elif target == "decode":
            # delete prefill worker
            del config["spec"]["services"][
                WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker
            ]

            args = config["spec"]["services"][
                WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker
            ]["extraPodSpec"]["mainContainer"]["args"]

            args = break_arguments(args)

            # enable prefix caching
            if "--enable-prefix-caching" not in args:
                args = append_argument(args, "--enable-prefix-caching")
            if "--no-enable-prefix-caching" in args:
                args.remove("--no-enable-prefix-caching")

            config["spec"]["services"][WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
                "extraPodSpec"
            ]["mainContainer"]["args"] = join_arguments(args)

        # set num workers to 1
        decode_worker_config = config["spec"]["services"][
            WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker
        ]
        decode_worker_config["replicas"] = 1

        return config

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        config = deepcopy(config)

        config["spec"]["services"][WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
            "resources"
        ]["requests"]["gpu"] = str(tp_size)
        config["spec"]["services"][WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
            "resources"
        ]["limits"]["gpu"] = str(tp_size)

        args = config["spec"]["services"][
            WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker
        ]["extraPodSpec"]["mainContainer"]["args"]

        args = break_arguments(args)

        try:
            idx = args.index("--tensor-parallel-size")
            args[idx + 1] = str(tp_size)
        except ValueError:
            args = append_argument(args, ["--tensor-parallel-size", str(tp_size)])

        config["spec"]["services"][WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
            "extraPodSpec"
        ]["mainContainer"]["args"] = join_arguments(args)

        return config

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        worker_name = WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker
        args = config["spec"]["services"][worker_name]["extraPodSpec"]["mainContainer"][
            "args"
        ]

        args = break_arguments(args)
        for i, arg in enumerate(args):
            if arg == "--model" and i + 1 < len(args):
                return args[i + 1]

        logger.warning(
            f"Model name not found in configuration args, using default model name: {DEFAULT_MODEL_NAME}"
        )
        return DEFAULT_MODEL_NAME

    @classmethod
    def get_port(cls, config: dict) -> int:
        args = config["spec"]["services"]["Frontend"]["extraPodSpec"]["mainContainer"][
            "args"
        ]
        args = break_arguments(args)
        try:
            idx = args.index("--http-port")
            return int(args[idx + 1])
        except ValueError:
            logger.warning(
                f"Port not found in configuration args, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

    @classmethod
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
        # TODO
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "Maximum concurrency for" in line:
                        line = line.strip().split("Maximum concurrency for ")[1]
                        token_count = int(
                            line.split(" tokens per request: ")[0].replace(",", "")
                        )
                        concurrency = float(line.split(" tokens per request: ")[1][:-1])

                        logger.info(
                            f"Found KV cache info: {token_count} x {concurrency} = {int(token_count * concurrency)}"
                        )
                        return int(token_count * concurrency)
        except Exception as e:
            logger.warning(
                f"Failed to parse KV cache size from line: {line}. Error: {e}"
            )
        return 0


CONFIG_MODIFIERS = {
    "vllm_v1": VllmV1ConfigModifier,
}
