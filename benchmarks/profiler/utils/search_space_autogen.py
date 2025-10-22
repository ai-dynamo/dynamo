# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import os

import yaml

from benchmarks.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from benchmarks.profiler.utils.model_info import get_model_info
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
MOE_MODEL_MAX_NUM_GPUS = 32


def auto_generate_search_space(args: argparse.Namespace) -> None:
    config_modifier = CONFIG_MODIFIERS[
        args.backend
    ]  # args.backend is already validated in argparse

    # first check if config file exists
    if args.model is not None:
        if not args.config:
            # modify config file from default config file
            logger.info("DGD config file not provided, using default config file")
            config = config_modifier.load_default_config()

        else:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

        logger.info(f"Updating model in DGD config file to {args.model}")
        config = config_modifier.update_model(config, args.model)
        config_fn = f"{args.output_dir}/disagg_config.yaml"
        logger.info(f"Saving generated disagg DGD config for profiling to {config_fn}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(config_fn, "w") as f:
            yaml.dump(config, f)
        args.config = config_fn

    # now determine the search space
    if args.model is not None:
        model_info = get_model_info(args.model)
        gpu_info = get_gpu_summary()

        logger.info(
            f"Model {args.model} has size {model_info['model_size']}, is_moe={model_info['is_moe']}, and max_context_length={model_info['max_context_length']}"
        )
        logger.info(
            f"Cluster has {gpu_info['gpus_per_node']}x{gpu_info['model']} GPUs per node with {gpu_info['vram']} VRAM"
        )

        min_gpu = math.ceil(
            model_info["model_size"] / MODEL_GPU_MEM_FRAC_MAX / gpu_info["vram"]  # type: ignore[operator]
        )
        max_gpu = (
            gpu_info["gpus_per_node"]  # type: ignore[misc]
            if not model_info["is_moe"]
            else MOE_MODEL_MAX_NUM_GPUS
        )
        if min_gpu > max_gpu:
            error_msg = f"No valid GPU configuration found for model {args.model} on the cluster with {gpu_info['gpus_per_node']}x{gpu_info['model']} GPUs per node"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"Auto-generated search space for model {args.model} on the cluster with {gpu_info['gpus_per_node']}x{gpu_info['model']} GPUs per node: {min_gpu} to {max_gpu}"
        )
        args.min_num_gpus_per_engine = min_gpu
        args.max_num_gpus_per_engine = max_gpu
        args.is_moe_model = model_info["is_moe"]  # type: ignore[assignment]
        args.max_context_length = model_info["max_context_length"]  # type: ignore[assignment]
        args.num_gpus_per_node = gpu_info["gpus_per_node"]  # type: ignore[assignment]

    return
