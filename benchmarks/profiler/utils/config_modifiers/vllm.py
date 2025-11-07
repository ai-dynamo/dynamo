# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import yaml

from benchmarks.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    get_service_name_by_type,
    get_worker_service_from_config,
    set_argument_value,
    setup_worker_service_resources,
    update_image,
    validate_and_get_worker_args,
)
from benchmarks.profiler.utils.defaults import (
    DEFAULT_MODEL_NAME,
    DYNAMO_RUN_DEFAULT_PORT,
    EngineType,
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


DEFAULT_VLLM_CONFIG_PATH = "examples/backends/vllm/deploy/disagg.yaml"


class VllmV1ConfigModifier:
    @classmethod
    def load_default_config(cls) -> dict:
        with open(DEFAULT_VLLM_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def update_model(cls, config, model_name: str) -> dict:
        # change the model to serve
        cfg = Config.model_validate(config)

        # Update model for both prefill and decode workers
        for sub_component_type in [SubComponentType.PREFILL, SubComponentType.DECODE]:
            try:
                worker_service = get_worker_service_from_config(
                    cfg, backend="vllm", sub_component_type=sub_component_type
                )
                args = validate_and_get_worker_args(worker_service, backend="vllm")
                args = break_arguments(args)

                # Update --model (vllm uses --model instead of --model-path and --served-model-name)
                args = set_argument_value(args, "--model", model_name)

                worker_service.extraPodSpec.mainContainer.args = args
            except (ValueError, KeyError):
                # Service might not exist (e.g., in aggregated mode)
                logger.debug(
                    f"Skipping {sub_component_type} service as it doesn't exist"
                )
                continue

        return cfg.model_dump()

    @classmethod
    def update_image(cls, config, image: str) -> dict:
        """Update container image for all DGD services (frontend, planner, workers)."""
        return update_image(config, image)

    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: EngineType,
        is_moe_model: bool = False,
    ) -> dict:
        if is_moe_model:
            raise NotImplementedError(
                "MoE model support is not implemented for VLLM backend"
            )

        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = "vllm-agg"

        # disable planner
        if "Planner" in cfg.spec.services:
            del cfg.spec.services["Planner"]

        if target == EngineType.PREFILL:
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                cfg, "vllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                cfg, "vllm", SubComponentType.DECODE
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
                backend="vllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="vllm")
            args = break_arguments(args)

            # remove --is-prefill-worker flag
            args.remove("--is-prefill-worker")

            # disable prefix caching
            if "--enable-prefix-caching" in args:
                args.remove("--enable-prefix-caching")
            if "--no-enable-prefix-caching" not in args:
                args = append_argument(args, "--no-enable-prefix-caching")

            worker_service.extraPodSpec.mainContainer.args = args

        elif target == EngineType.DECODE:
            # Get service names by inferring from subComponentType first
            prefill_service_name = get_service_name_by_type(
                cfg, "vllm", SubComponentType.PREFILL
            )
            decode_service_name = get_service_name_by_type(
                cfg, "vllm", SubComponentType.DECODE
            )

            # delete prefill worker
            del cfg.spec.services[prefill_service_name]

            # Set subComponentType for aggregated decode-only mode
            cfg.spec.services[decode_service_name].subComponentType = "decode"

            worker_service = get_worker_service_from_config(
                cfg,
                backend="vllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="vllm")
            args = break_arguments(args)

            # enable prefix caching
            if "--enable-prefix-caching" not in args:
                args = append_argument(args, "--enable-prefix-caching")
            if "--no-enable-prefix-caching" in args:
                args.remove("--no-enable-prefix-caching")

            worker_service.extraPodSpec.mainContainer.args = args

        # set num workers to 1
        # Use the inferred decode service name
        final_decode_service_name = get_service_name_by_type(
            cfg, "vllm", SubComponentType.DECODE
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
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources
        setup_worker_service_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        try:
            idx = args.index("--tensor-parallel-size")
            args[idx + 1] = str(tp_size)
        except ValueError:
            args = append_argument(args, ["--tensor-parallel-size", str(tp_size)])

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
        raise NotImplementedError(
            "TEP (Tensor Expert Parallelism) is not implemented for VLLM backend"
        )

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        raise NotImplementedError(
            "DEP (Data Expert Parallelism) is not implemented for VLLM backend"
        )

    @classmethod
    def get_model_name(cls, config: dict) -> str:
        cfg = Config.model_validate(config)
        try:
            worker_service = get_worker_service_from_config(cfg, backend="vllm")
            args = validate_and_get_worker_args(worker_service, backend="vllm")
        except (ValueError, KeyError):
            logger.warning(
                f"Worker service missing or invalid, using default model name: {DEFAULT_MODEL_NAME}"
            )
            return DEFAULT_MODEL_NAME

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
