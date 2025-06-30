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
from typing import Literal

from dynamo.planner.defaults import WORKER_COMPONENT_NAMES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class VllmV0ConfigModifier:
    @classmethod
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        config = config.copy()

        # disable planner
        if "Planner" in config:
            config["Planner"]["no-operation"] = True

        if target == "prefill":
            if WORKER_COMPONENT_NAMES["vllm_v0"].prefill_worker in config:
                # make PrefillWorker into VllmWorker
                del config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker]
                config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker] = config[
                    WORKER_COMPONENT_NAMES["vllm_v0"].prefill_worker
                ]
                del config[WORKER_COMPONENT_NAMES["vllm_v0"].prefill_worker]

            # to profile prefill, we disable prefix caching
            config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
                "enable-prefix-caching"
            ] = False
        elif target == "decode":
            if WORKER_COMPONENT_NAMES["vllm_v0"].prefill_worker in config:
                del config[WORKER_COMPONENT_NAMES["vllm_v0"].prefill_worker]

            # to profile prefill, we enable prefix caching to pass the prefill stage
            config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
                "enable-prefix-caching"
            ] = True

        # set num workers to 1
        config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker]["ServiceArgs"][
            "workers"
        ] = 1

        # set PP to 1
        if (
            "pipeline-parallel-size"
            in config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker]
            and config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
                "pipeline-parallel-size"
            ]
            > 1
        ):
            logger.warning("Currently we only support TP, setting PP to 1")
            config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
                "pipeline-parallel-size"
            ] = 1

        # always local prefill
        config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
            "remote-prefill"
        ] = False
        config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
            "conditional-disagg"
        ] = False

        return config

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker][
            "tensor-parallel-size"
        ] = tp_size
        config[WORKER_COMPONENT_NAMES["vllm_v0"].decode_worker]["ServiceArgs"][
            "resources"
        ]["gpu"] = tp_size
        return config

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        if "Common" in config and "served_model_name" in config["Common"]:
            return config["Common"]["served_model_name"]
        else:
            return config["Frontend"]["served_model_name"]

    @classmethod
    def get_port(cls, config: dict) -> int:
        if "Common" in config and "port" in config["Common"]:
            return config["Common"]["port"]
        else:
            return config["Frontend"]["port"]

    @classmethod
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "Maximum concurrency for" in line:
                        line = line.strip().split("Maximum concurrency for ")[1]
                        token_count = int(line.split(" tokens per request: ")[0])
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


class VllmV1ConfigModifier:
    @classmethod
    def convert_config(cls, config: dict, target: Literal["prefill", "decode"]) -> dict:
        config = config.copy()

        # disable planner
        if "Planner" in config:
            config["Planner"]["no-operation"] = True

        # turn-off disagg
        config["SimpleLoadBalancer"]["enable_disagg"] = False

        if target == "prefill":
            if WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker in config:
                # make VllmPrefillWorker into VllmDecodeWorker
                del config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker]
                config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker] = config[
                    WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker
                ]
                del config[WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker]

            # to profile prefill, we disable prefix caching
            config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
                "enable-prefix-caching"
            ] = False
        elif target == "decode":
            if WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker in config:
                del config[WORKER_COMPONENT_NAMES["vllm_v1"].prefill_worker]

            # to profile prefill, we enable prefix caching to pass the prefill stage
            config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
                "enable-prefix-caching"
            ] = True

        # set num workers to 1
        config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker]["ServiceArgs"][
            "workers"
        ] = 1

        # set PP to 1
        if (
            "pipeline-parallel-size"
            in config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker]
            and config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
                "pipeline-parallel-size"
            ]
            > 1
        ):
            logger.warning("Currently we only support TP, setting PP to 1")
            config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
                "pipeline-parallel-size"
            ] = 1

        return config

    @classmethod
    def set_config_tp_size(cls, config: dict, tp_size: int):
        config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker][
            "tensor-parallel-size"
        ] = tp_size
        config[WORKER_COMPONENT_NAMES["vllm_v1"].decode_worker]["ServiceArgs"][
            "resources"
        ]["gpu"] = tp_size
        return config

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        if "Common" in config and "served_model_name" in config["Common"]:
            return config["Common"]["served_model_name"]
        else:
            return config["Frontend"]["served_model_name"]

    @classmethod
    def get_port(cls, config: dict) -> int:
        if "Common" in config and "port" in config["Common"]:
            return config["Common"]["port"]
        else:
            return config["Frontend"]["port"]

    @classmethod
    def get_kv_cache_size_from_dynamo_log(cls, dynamo_log_fn: str) -> int:
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
    "vllm_v0": VllmV0ConfigModifier,
    "vllm_v1": VllmV1ConfigModifier,
}
