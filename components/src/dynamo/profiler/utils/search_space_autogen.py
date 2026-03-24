# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import os

import yaml

from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from dynamo.profiler.utils.model_info import ModelInfo, get_model_info

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
        if args.model_cache_pvc_name:
            config = config_modifier.update_model_from_pvc(
                config,
                args.model,
                args.model_cache_pvc_name,
                args.model_cache_pvc_mount_path,
                args.model_cache_pvc_path,
            )
        else:
            # Non-PVC: workers download from HF, so model_path == model_name
            config = config_modifier.update_model(config, args.model, args.model)
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
    model_name_or_path = ""
    if args.model:
        # prioritize using model cache in PVC over downloading from HF
        if args.model_cache_pvc_name:
            # Keep consistent path normalization with config mutation logic
            model_name_or_path = config_modifier._normalize_model_path(
                args.model_cache_pvc_mount_path, args.model_cache_pvc_path
            )
        else:
            model_name_or_path = args.model
    else:
        # get the model name from config
        args.model, args.model_path = config_modifier.get_model_name(config)
        model_name_or_path = args.model_path
    logger.info(f"Getting model info for {args.model} at {model_name_or_path}...")
    try:
        model_info = get_model_info(model_name_or_path)
    except Exception as e:
        # Common in dry-run mode when the PVC isn't mounted locally.
        logger.warning(
            f"Failed to load model info from local path '{model_name_or_path}': {e}. "
            f"Trying to download from HF for '{args.model}'."
        )
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

    # Determine the search space for profiling
    # User-provided min/max values take precedence; auto-calculate missing bounds
    # based on GPU hardware info and model size
    user_specified_ranges = (
        args.min_num_gpus_per_engine != 0 and args.max_num_gpus_per_engine != 0
    )

    if user_specified_ranges:
        logger.info(
            f"Using user-specified GPU search space: {args.min_num_gpus_per_engine} to {args.max_num_gpus_per_engine}"
        )
        # Ensure num_gpus_per_node is set (needed for multi-node configs)
        if args.num_gpus_per_node == 0:
            logger.warning("num_gpus_per_node not specified, setting to 8")
            args.num_gpus_per_node = 8
    else:
        # Auto-calculate search space (honor partial user overrides)
        # NOTE: will be handled in AIC
        if args.num_gpus_per_node != 0 and args.gpu_vram_mib != 0:
            # Have GPU hardware info - calculate based on model size
            if not args.model:
                error_msg = "No model provided, cannot auto-generate GPU search space. Please provide --model"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            assert (
                model_info is not None
            ), "model_info must be set when model is provided"

            logger.info(
                f"Auto-generating search space: {args.num_gpus_per_node}x {args.gpu_model} GPUs with {args.gpu_vram_mib} MiB VRAM per GPU"
            )
            if args.system:
                logger.info(f"Hardware system: {args.system}")

            # Calculate minimum GPUs needed for model
            min_gpu = math.ceil(
                model_info.model_size / MODEL_GPU_MEM_FRAC_MAX / args.gpu_vram_mib
            )

            # Calculate maximum GPUs to profile
            if not model_info.is_moe:
                max_gpu = args.num_gpus_per_node
            else:
                # MoE models can benefit from more GPUs
                max_gpu = max(
                    min_gpu * MOE_MODEL_MAX_NUM_GPU_FACTOR, args.num_gpus_per_node
                )

            # Honor partial user overrides
            final_min = args.min_num_gpus_per_engine or min_gpu
            final_max = args.max_num_gpus_per_engine or max_gpu

            # Validate final_min <= final_max
            if final_min > final_max:
                error_msg = f"Invalid GPU range: min_num_gpus_per_engine ({final_min}) > max_num_gpus_per_engine ({final_max})"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Clamp to valid range [1, args.num_gpus_per_node]
            final_min = max(1, min(final_min, args.num_gpus_per_node))
            final_max = max(1, min(final_max, args.num_gpus_per_node))

            logger.info(f"Auto-generated search space: {final_min} to {final_max} GPUs")
            args.min_num_gpus_per_engine = final_min
            args.max_num_gpus_per_engine = final_max
        else:
            # No GPU info available - use defaults
            logger.warning("GPU hardware info not available, using default values")
            args.min_num_gpus_per_engine = args.min_num_gpus_per_engine or 1
            args.max_num_gpus_per_engine = args.max_num_gpus_per_engine or 4
            args.num_gpus_per_node = args.num_gpus_per_node or 8
            logger.info(
                f"Default search space: {args.min_num_gpus_per_engine} to {args.max_num_gpus_per_engine} GPUs, {args.num_gpus_per_node} GPUs per node"
            )
    return
