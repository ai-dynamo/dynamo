# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Literal

import yaml

from benchmarks.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    get_service_name_by_type,
    get_worker_service_from_config,
    remove_valued_arguments,
    set_argument_value,
    setup_worker_service_resources,
    validate_and_get_worker_args,
)
from benchmarks.profiler.utils.defaults import (
    DEFAULT_MODEL_NAME,
    DYNAMO_RUN_DEFAULT_PORT,
)
from dynamo.planner.defaults import SubComponentType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


DEFAULT_SGLANG_CONFIG_PATH = "components/backends/sglang/deploy/disagg.yaml"


class SGLangConfigModifier:
    @classmethod
    def load_default_config(cls) -> dict:
        with open(DEFAULT_SGLANG_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def update_model(cls, config, model_name: str) -> dict:
        # change the model to serve
        cfg = Config.model_validate(config)

        # Update model for both prefill and decode workers
        for sub_component_type in [SubComponentType.PREFILL, SubComponentType.DECODE]:
            try:
                worker_service = get_worker_service_from_config(
                    cfg, backend="sglang", sub_component_type=sub_component_type
                )
                args = validate_and_get_worker_args(worker_service, backend="sglang")
                args = break_arguments(args)

                # Update both --model-path and --served-model-name
                args = set_argument_value(args, "--model-path", model_name)
                args = set_argument_value(args, "--served-model-name", model_name)

                worker_service.extraPodSpec.mainContainer.args = args
            except (ValueError, KeyError):
                # Service might not exist (e.g., in aggregated mode)
                logger.debug(
                    f"Skipping {sub_component_type} service as it doesn't exist"
                )
                continue

        return cfg.model_dump()

    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: Literal["prefill", "decode"],
        is_moe_model: bool = False,
    ) -> dict:
        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "sglang-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == "prefill":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                cfg, "sglang", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                cfg, "sglang", SubComponentType.DECODE
            )

            # convert prefill worker into decode worker
            cfg.spec.services[decode_service_name] = cfg.spec.services[
                prefill_service_name
            ]
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated mode (using decode worker for prefill-only)
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg,
                backend="sglang",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="sglang")
            args = break_arguments(args)

            # remove disagg flags
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")
            args = remove_valued_arguments(args, "--disaggregation-bootstrap-port")

            # disable prefix caching
            if "--disable-radix-cache" not in args:
                args = append_argument(args, "--disable-radix-cache")

            worker_service.extraPodSpec.mainContainer.args = args

        elif target == "decode":
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                cfg, "sglang", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                cfg, "sglang", SubComponentType.DECODE
            )

            # delete prefill worker
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated decode-only mode
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg,
                backend="sglang",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="sglang")
            args = break_arguments(args)

            # remove disagg flags
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")
            args = remove_valued_arguments(args, "--disaggregation-bootstrap-port")

            # enable prefix caching
            if "--disable-radix-cache" in args:
                args.remove("--disable-radix-cache")

            if is_moe_model:
                # need to use round_robin dp attention routing for MoE models to ensure kv reuse can skip prefill
                if "--load-balance-method" in args:
                    idx = args.index("--load-balance-method")
                    args[idx + 1] = "round_robin"
                else:
                    args = append_argument(
                        args, ["--load-balance-method", "round_robin"]
                    )

            worker_service.extraPodSpec.mainContainer.args = args

        # set num workers to 1
        # Use the inferred decode service name
        final_decode_service_name = get_service_name_by_type(
            cfg, "sglang", SubComponentType.DECODE
        )
        decode_worker_config = cfg.spec.services[final_decode_service_name]
        decode_worker_config.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(
        cls,
        config: dict,
        tp_size: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )

        # Set up resources
        setup_worker_service_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # Set --tp argument
        args = set_argument_value(args, "--tp", str(tp_size))

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(
        cls,
        config: dict,
        tep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, tep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # 1. Set --tp=tep_size, if not present add it
        args = set_argument_value(args, "--tp", str(tep_size))

        # 2. Set --ep-size=tep_size, if not present add it
        args = set_argument_value(args, "--ep-size", str(tep_size))

        # 3. Remove --dp if present
        args = remove_valued_arguments(args, "--dp")

        # 4. Remove --enable-dp-attention if present
        if "--enable-dp-attention" in args:
            args.remove("--enable-dp-attention")

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, dep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # 1. Set --tp=dep_size
        args = set_argument_value(args, "--tp", str(dep_size))

        # 2. Set --dp=dep_size (data parallelism across experts)
        args = set_argument_value(args, "--dp", str(dep_size))

        # 3. Enable --enable-dp-attention
        if "--enable-dp-attention" not in args:
            args = append_argument(args, "--enable-dp-attention")

        # 4. Set --ep-size=dep_size (expert parallelism size)
        args = set_argument_value(args, "--ep-size", str(dep_size))

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        cfg = Config.model_validate(config)
        try:
            worker_service = get_worker_service_from_config(cfg, backend="sglang")
            args = validate_and_get_worker_args(worker_service, backend="sglang")
        except (ValueError, KeyError):
            logger.warning(
                f"Worker service missing or invalid, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME

        args = break_arguments(args)
        for i, arg in enumerate(args):
            if arg == "--served-model-name" and i + 1 < len(args):
                return args[i + 1]

        logger.warning(
            f"Model name not found in configuration args, using default model name: {DEFAULT_MODEL_NAME}"
        )
        return DEFAULT_MODEL_NAME

    @classmethod
    def get_port(cls, config: dict) -> int:
        cfg = Config.model_validate(config)
        frontend_service = cfg.spec.services.get("Frontend")
        if (
            not frontend_service
            or not frontend_service.extraPodSpec
            or not frontend_service.extraPodSpec.mainContainer
        ):
            logger.warning(
                f"Frontend service or container not found, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = frontend_service.extraPodSpec.mainContainer.args
        if not args:
            logger.warning(
                f"No args found in Frontend configuration, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = break_arguments(args)
        try:
            idx = args.index("--http-port")
            return int(args[idx + 1])
        except (ValueError, IndexError):
            logger.warning(
                f"Port not found in configuration args, using default port: {DYNAMO_RUN_DEFAULT_PORT}"
            )
            return DYNAMO_RUN_DEFAULT_PORT

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "KV Cache is allocated" in line and "#tokens:" in line:
                        # Extract the number after "#tokens:"
                        match = re.search(r"#tokens:\s*(\d+)", line)
                        if match:
                            return int(match.group(1)) * attention_dp_size
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")
        return 0
