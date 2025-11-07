# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
from typing import Any, Dict

import yaml

from benchmarks.profiler.utils.planner_utils import add_planner_arguments_to_parser
from benchmarks.profiler.utils.search_space_autogen import auto_generate_search_space


def parse_config_string(config_str: str) -> Dict[str, Any]:
    """Parse configuration string as Python dict literal, YAML, or JSON.

    Supports multiple input formats:
    1. Python dict literal: "{'engine': {'backend': 'vllm'}, 'sla': {'isl': 3000}}"
    2. YAML string: "engine:\n  backend: vllm\nsla:\n  isl: 3000"
    3. JSON string: '{"engine": {"backend": "vllm"}, "sla": {"isl": 3000}}'

    Args:
        config_str: Configuration string in one of the supported formats

    Returns:
        Dictionary containing the configuration

    Raises:
        ValueError: If config cannot be parsed or is not a dictionary
    """
    config = None

    # Try 1: Parse as Python dict literal (most direct for CLI)
    try:
        config = ast.literal_eval(config_str)
        if isinstance(config, dict):
            return config
    except (ValueError, SyntaxError):
        pass

    # Try 2: Parse as YAML/JSON (for K8s ConfigMaps and files)
    try:
        config = yaml.safe_load(config_str)
        if config is not None and isinstance(config, dict):
            return config
    except yaml.YAMLError:
        pass

    # If we got here, parsing failed
    raise ValueError(
        "Failed to parse config string. Expected Python dict literal, YAML, or JSON format. "
        f"Examples:\n"
        f"  Python dict: \"{'engine': {'backend': 'vllm'}}\"\n"
        f'  YAML: "engine:\\n  backend: vllm"\n'
        f'  JSON: \'{{"engine": {{"backend": "vllm"}}}}\''
    )


def create_profiler_parser() -> argparse.Namespace:
    """
    Create argument parser with support for YAML config string.

    Config structure:
        output_dir: String (path to the output results directory, default: profiling_results)
        deployment:
            namespace: String (kubernetes namespace, default: dynamo-sla-profiler)
            service_name: String (service name, default: "")
            model: String (model to serve, can be HF model name or local model path)
        engine:
            backend: String (backend type, currently support [vllm, sglang, trtllm], default: vllm)
            config: String (path to the DynamoGraphDeployment config file, default: "")
            max_context_length: Int (maximum context length supported by the served model, default: 0)
            is_moe_model: Boolean (enable MoE (Mixture of Experts) model support, use TEP for prefill and DEP for decode, default: False)
        hardware:
            min_num_gpus_per_engine: Int (minimum number of GPUs per engine, default: 0)
            max_num_gpus_per_engine: Int (maximum number of GPUs per engine, default: 0)
            num_gpus_per_node: Int (number of GPUs per node for MoE models - this will be the granularity when searching for the best TEP/DEP size, default: 0)
        sweep:
            prefill_interpolation_granularity: Int (how many samples to benchmark to interpolate TTFT under different ISL, default: 16)
            decode_interpolation_granularity: Int (how many samples to benchmark to interpolate ITL under different active kv cache size and decode context length, default: 6)
            use_ai_configurator: Boolean (use ai-configurator to estimate benchmarking results instead of running actual deployment, default: False)
            aic_system: String (target system for use with aiconfigurator, default: None)
            aic_model_name: String (aiconfigurator name of the target model, default: None)
            aic_backend: String (aiconfigurator backend of the target model, if not provided, will use args.backend, default: "")
            aic_backend_version: String (specify backend version when using aiconfigurator to estimate perf, default: None)
            dry_run: Boolean (dry run the profile job, default: False)
        sla:
            isl: Int (target input sequence length, default: 3000)
            osl: Int (target output sequence length, default: 500)
            ttft: Float (target Time To First Token in milliseconds, default: 50)
            itl: Float (target Inter Token Latency in milliseconds, default: 10)
        planner: (planner-bypass arguments, use hyphens or underscores)
            i.e., planner-min-endpoint: 2  # or planner_min_endpoint: 2 (both work)
    """
    # Step 1: Pre-parse to check if --profile-config is provided
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--profile-config", type=str)
    pre_args, _ = pre_parser.parse_known_args()

    # Step 2: Parse config if provided
    config = {}
    if pre_args.profile_config:
        config = parse_config_string(pre_args.profile_config)

    # Step 3: Create main parser with config-aware defaults
    parser = argparse.ArgumentParser(
        description="Profile the TTFT and ITL of the Prefill and Decode engine with different parallelization mapping. When profiling prefill we mock/fix decode,when profiling decode we mock/fix prefill."
    )

    parser.add_argument(
        "--profile-config",
        type=str,
        help="Configuration as Python dict literal, YAML, or JSON string. CLI args override config values. "
        "Example: \"{'engine': {'backend': 'vllm', 'config': '/path'}, 'sla': {'isl': 3000}}\"",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("deployment", {}).get("model", ""),
        help="Model to serve, can be HF model name or local model path",
    )
    parser.add_argument(
        "--dgd-image",
        type=str,
        default=config.get("deployment", {}).get("dgd_image", ""),
        help="Container image to use for DGD components (frontend, planner, workers). Overrides images in config file.",
    )

    # CLI arguments with config-aware defaults (using nested .get() for cleaner code)
    parser.add_argument(
        "--namespace",
        type=str,
        default=config.get("deployment", {}).get("namespace", "dynamo-sla-profiler"),
        help="Kubernetes namespace to deploy the DynamoGraphDeployment",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=config.get("engine", {}).get("backend", "vllm"),
        choices=["vllm", "sglang", "trtllm"],
        help="backend type, currently support [vllm, sglang, trtllm]",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=config.get("engine", {}).get("config", ""),
        required=False,
        help="Path to the DynamoGraphDeployment config file (required, can be provided via CLI or config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.get("output_dir", "profiling_results"),
        help="Path to the output results directory",
    )
    parser.add_argument(
        "--min-num-gpus-per-engine",
        type=int,
        default=config.get("hardware", {}).get("min_num_gpus_per_engine", 0),
        help="minimum number of GPUs per engine",
    )
    parser.add_argument(
        "--max-num-gpus-per-engine",
        type=int,
        default=config.get("hardware", {}).get("max_num_gpus_per_engine", 0),
        help="maximum number of GPUs per engine",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=config.get("hardware", {}).get("num_gpus_per_node", 0),
        help="Number of GPUs per node for MoE models - this will be the granularity when searching for the best TEP/DEP size",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=config.get("sla", {}).get("isl", 3000),
        help="target input sequence length",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=config.get("sla", {}).get("osl", 500),
        help="target output sequence length",
    )
    parser.add_argument(
        "--ttft",
        type=float,
        default=config.get("sla", {}).get("ttft", 50.0),
        help="target Time To First Token (float, in milliseconds)",
    )
    parser.add_argument(
        "--itl",
        type=float,
        default=config.get("sla", {}).get("itl", 10.0),
        help="target Inter Token Latency (float, in milliseconds)",
    )

    # arguments used for interpolating TTFT and ITL under different ISL/OSL
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=config.get("engine", {}).get("max_context_length", 0),
        help="maximum context length supported by the served model",
    )
    parser.add_argument(
        "--prefill-interpolation-granularity",
        type=int,
        default=config.get("sweep", {}).get("prefill_interpolation_granularity", 16),
        help="how many samples to benchmark to interpolate TTFT under different ISL",
    )
    parser.add_argument(
        "--decode-interpolation-granularity",
        type=int,
        default=config.get("sweep", {}).get("decode_interpolation_granularity", 6),
        help="how many samples to benchmark to interpolate ITL under different active kv cache size and decode context length",
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default=config.get("deployment", {}).get("service_name", ""),
        help="Service name for port forwarding (default: {deployment_name}-frontend)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=config.get("sweep", {}).get("dry_run", False),
        help="Dry run the profile job",
    )
    parser.add_argument(
        "--enable-gpu-discovery",
        action="store_true",
        default=config.get("hardware", {}).get("enable_gpu_discovery", False),
        help="Enable automatic GPU discovery from Kubernetes cluster nodes. When enabled, overrides any manually specified hardware configuration. Requires cluster-wide node access permissions.",
    )

    # Dynamically add all planner arguments from planner_argparse.py
    add_planner_arguments_to_parser(parser, prefix="planner-")
    # Set defaults for any planner arguments found in config.planner
    # Note: argparse converts hyphens to underscores, so we need to normalize keys
    planner_config = config.get("planner", {})
    if planner_config:
        # Convert hyphens to underscores to match argparse's internal naming
        normalized_planner_config = {
            key.replace("-", "_"): value for key, value in planner_config.items()
        }
        parser.set_defaults(**normalized_planner_config)

    # arguments if using aiconfigurator
    parser.add_argument(
        "--use-ai-configurator",
        action="store_true",
        default=config.get("sweep", {}).get("use_ai_configurator", False),
        help="Use ai-configurator to estimate benchmarking results instead of running actual deployment.",
    )
    parser.add_argument(
        "--aic-system",
        type=str,
        default=config.get("sweep", {}).get("aic_system"),
        help="Target system for use with aiconfigurator (e.g. h100_sxm, h200_sxm)",
    )
    parser.add_argument(
        "--aic-model-name",
        type=str,
        default=config.get("sweep", {}).get("aic_model_name"),
        help="aiconfigurator name of the target model (e.g. QWEN3_32B, DEEPSEEK_V3)",
    )
    parser.add_argument(
        "--aic-backend",
        type=str,
        default=config.get("sweep", {}).get("aic_backend", ""),
        help="aiconfigurator backend of the target model, if not provided, will use args.backend",
    )
    parser.add_argument(
        "--aic-backend-version",
        type=str,
        default=config.get("sweep", {}).get("aic_backend_version"),
        help="Specify backend version when using aiconfigurator to estimate perf.",
    )

    # Parse arguments
    args = parser.parse_args()

    # remove --profile-config from args
    if hasattr(args, "profile_config"):
        delattr(args, "profile_config")

    # Validate required arguments
    # Either --model or --config (or both) must be provided
    if not args.model and not args.config:
        parser.error("--model or --config is required (provide at least one)")

    auto_generate_search_space(args)
    return args
