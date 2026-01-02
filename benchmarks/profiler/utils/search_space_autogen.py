# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import os

import yaml

from benchmarks.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from benchmarks.profiler.utils.model_info import ModelInfo, get_model_info
from deploy.utils.gpu_inventory import get_gpu_summary

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

MODEL_GPU_MEM_FRAC_MAX = 0.9

# for MoE models, we sweep up to number of GPUs that can hold 8x the model weights
MOE_MODEL_MAX_NUM_GPU_FACTOR = 8


def auto_generate_search_space(args: argparse.Namespace) -> None:
    config_modifier = CONFIG_MODIFIERS[
        args.backend
    ]  # args.backend is already validated in argparse

    # first get the config
    if not args.config:
        # modify config file from default config file
        logger.info("DGD config file not provided, using default config file")
        config = config_modifier.load_default_config()
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    if args.model:
        logger.info(f"Updating model in DGD config file to {args.model}")
        config = config_modifier.update_model(config, args.model)
        if args.dgd_image:
            logger.info(f"Updating DGD image to {args.dgd_image}")
            config = config_modifier.update_image(config, args.dgd_image)

        config_fn = f"{args.output_dir}/disagg_config.yaml"
        logger.info(f"Saving generated disagg DGD config for profiling to {config_fn}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(config_fn, "w") as f:
            yaml.dump(config, f)
        args.config = config_fn

    # get model info and update args
    model_info: ModelInfo | None = None
    if not args.model:
        # get the model name from config
        args.model = config_modifier.get_model_name(config)
    logger.info(f"Getting model info for {args.model}...")
    model_info = get_model_info(args.model)

    num_experts_str = (
        f", num_experts={model_info.num_experts}"
        if model_info.num_experts is not None
        else ""
    )
    logger.info(
        f"Model {args.model} has size {model_info.model_size}, is_moe={model_info.is_moe}, and max_context_length={model_info.max_context_length}{num_experts_str}"
    )
    args.model_info = model_info

    # now determine the search space
    if args.enable_gpu_discovery:
        if (
            args.min_num_gpus_per_engine == 0
            or args.max_num_gpus_per_engine == 0
            or args.num_gpus_per_node == 0
        ):
            if not args.model:
                # TODO: get model info provided DGD config
                error_msg = "No model provided, cannot auto-generate GPU search space. Please provide `--model` or GPU info"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info("Getting GPU info from k8s cluster...")
            gpu_info = get_gpu_summary()
            logger.info(
                f"Cluster has {gpu_info['gpus_per_node']}x{gpu_info['model']} GPUs per node with {gpu_info['vram']} VRAM"
            )

            # model_info should be set by now (checked above), but mypy needs explicit verification
            assert (
                model_info is not None
            ), "model_info must be set when model is provided"

            vram_mib = int(gpu_info["vram"])  # type: ignore[call-overload]
            gpus_per_node = int(gpu_info["gpus_per_node"])  # type: ignore[call-overload]

            min_gpu = math.ceil(
                model_info.model_size / MODEL_GPU_MEM_FRAC_MAX / vram_mib
            )
            if not model_info.is_moe:
                max_gpu = gpus_per_node
            else:
                max_gpu = max(min_gpu * MOE_MODEL_MAX_NUM_GPU_FACTOR, gpus_per_node)
            if min_gpu > max_gpu:
                error_msg = f"No valid GPU configuration found for model {args.model} on the cluster with {gpu_info['gpus_per_node']}x{gpu_info['model']} GPUs per node"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(
                f"Auto-generated search space for model {args.model} on the cluster with {gpu_info['gpus_per_node']}x{gpu_info['model']} GPUs per node: {min_gpu} to {max_gpu}"
            )
            args.min_num_gpus_per_engine = min_gpu
            args.max_num_gpus_per_engine = max_gpu
            args.num_gpus_per_node = gpus_per_node  # type: ignore[assignment]
    else:
        # use default values for GPUs
        if args.min_num_gpus_per_engine == 0:
            logger.warning(
                "GPU discover is disabled and min_num_gpus_per_engine is not specified, setting to 1"
            )
            args.min_num_gpus_per_engine = 1
        if args.max_num_gpus_per_engine == 0:
            logger.warning(
                "GPU discover is disabled and max_num_gpus_per_engine is not specified, setting to 4"
            )
            args.max_num_gpus_per_engine = 4
        if args.num_gpus_per_node == 0:
            logger.warning(
                "GPU discover is disabled and num_gpus_per_node is not specified, setting to 8"
            )
            args.num_gpus_per_node = 8
    return
