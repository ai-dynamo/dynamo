# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import os
from typing import Any, Dict

import yaml

from benchmarks.profiler.utils.planner_utils import add_planner_arguments_to_parser
from benchmarks.profiler.utils.search_space_autogen import auto_generate_search_space


def _get(cfg: Dict[str, Any], camel: str, snake: str, default: Any = None) -> Any:
    """Get config value with camelCase preferred, snake_case fallback."""
    if camel in cfg:
        return cfg[camel]
    return cfg.get(snake, default)


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    import re

    # Insert underscore before uppercase letters and lowercase
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


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

    Config structure (camelCase preferred, snake_case supported for backwards compat):
        outputDir: String (path to the output results directory, default: profiling_results)
        deployment:
            namespace: String (kubernetes namespace, default: dynamo-sla-profiler)
            serviceName: String (service name, default: "")
            model: String (served model name)
            dgdImage: String (container image to use for DGD components (frontend, planner, workers), overrides images in config file)
            modelCache:
                pvcName: String (name of the PVC to mount the model cache,
                    if not provided, model must be HF name and will download from HF, default: "")
                pvcPath: String (path to the model cache in the PVC, default: "")
                mountPath: String (path to the model cache in the container,
                    note that the PVC must be mounted to the same path for the profiling job,
                    default: "/opt/model-cache")
        engine:
            backend: String (backend type, currently support [vllm, sglang, trtllm], default: vllm)
            config: String (path to the DynamoGraphDeployment config file, default: "")
            maxContextLength: Int (maximum context length supported by the served model, default: 0)
            isMoeModel: Boolean (enable MoE (Mixture of Experts) model support, use TEP for prefill and DEP for decode, default: False)
        hardware:
            minNumGpusPerEngine: Int (minimum number of GPUs per engine, default: 0)
            maxNumGpusPerEngine: Int (maximum number of GPUs per engine, default: 0)
            numGpusPerNode: Int (number of GPUs per node for MoE models - this will be the granularity when searching for the best TEP/DEP size, default: 0)
            enableGpuDiscovery: Boolean (enable automatic GPU discovery from Kubernetes cluster nodes, when enabled overrides any manually specified hardware configuration, requires cluster-wide node access permissions, default: False)
        sweep:
            prefillInterpolationGranularity: Int (how many samples to benchmark to interpolate TTFT under different ISL, default: 16)
            decodeInterpolationGranularity: Int (how many samples to benchmark to interpolate ITL under different active kv cache size and decode context length, default: 6)
            useAiConfigurator: Boolean (use ai-configurator to estimate benchmarking results instead of running actual deployment, default: False)
            aicSystem: String (target system for use with aiconfigurator, default: None)
            aicHfId: String (aiconfigurator huggingface id of the target model, default: None)
            aicBackend: String (aiconfigurator backend of the target model, if not provided, will use args.backend, default: "")
            aicBackendVersion: String (specify backend version when using aiconfigurator to estimate perf, default: None)
            dryRun: Boolean (dry run the profile job, default: False)
            pickWithWebui: Boolean (pick the best parallelization mapping using webUI, default: False)
            webuiPort: Int (webUI port, default: $PROFILER_WEBUI_PORT or 8000)
        sla:
            isl: Int (target input sequence length, default: 3000)
            osl: Int (target output sequence length, default: 500)
            ttft: Float (target Time To First Token in milliseconds, default: 50)
            itl: Float (target Inter Token Latency in milliseconds, default: 10)
        planner: (planner arguments)
            e.g., plannerMinEndpoint: 2
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

    # CLI arguments with config-aware defaults (using nested .get() for cleaner code)
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("deployment", {}).get("model", ""),
        help="Served model name",
    )
    model_cache_config = config.get("deployment", {}).get("modelCache", {})
    parser.add_argument(
        "--model-cache-pvc-name",
        type=str,
        default=model_cache_config.get("pvcName", ""),
        help="Name of the PVC that contains the model weights. If not provided, args.model must be a HF model name and will download from HF",
    )
    parser.add_argument(
        "--model-cache-pvc-path",
        type=str,
        default=model_cache_config.get("pvcPath", ""),
        help="Path to the model cache in the PVC",
    )
    parser.add_argument(
        "--model-cache-pvc-mount-path",
        type=str,
        default=model_cache_config.get("mountPath", "/opt/model-cache"),
        help="Path to the model cache in the container, note that the PVC must be mounted to the same path for the profiling job",
    )
    deployment_cfg = config.get("deployment", {})
    parser.add_argument(
        "--dgd-image",
        type=str,
        default=_get(deployment_cfg, "dgdImage", "dgd_image", ""),
        help="Container image to use for DGD components (frontend, planner, workers). Overrides images in config file.",
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default=deployment_cfg.get("namespace", "dynamo-sla-profiler"),
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
        default=_get(config, "outputDir", "output_dir", "profiling_results"),
        help="Path to the output results directory",
    )
    hardware_cfg = config.get("hardware", {})
    parser.add_argument(
        "--min-num-gpus-per-engine",
        type=int,
        default=_get(hardware_cfg, "minNumGpusPerEngine", "min_num_gpus_per_engine", 0),
        help="minimum number of GPUs per engine",
    )
    parser.add_argument(
        "--max-num-gpus-per-engine",
        type=int,
        default=_get(hardware_cfg, "maxNumGpusPerEngine", "max_num_gpus_per_engine", 0),
        help="maximum number of GPUs per engine",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=_get(hardware_cfg, "numGpusPerNode", "num_gpus_per_node", 0),
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
    engine_cfg = config.get("engine", {})
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=_get(engine_cfg, "maxContextLength", "max_context_length", 0),
        help="maximum context length supported by the served model",
    )
    sweep_cfg = config.get("sweep", {})
    parser.add_argument(
        "--prefill-interpolation-granularity",
        type=int,
        default=_get(
            sweep_cfg,
            "prefillInterpolationGranularity",
            "prefill_interpolation_granularity",
            16,
        ),
        help="how many samples to benchmark to interpolate TTFT under different ISL",
    )
    parser.add_argument(
        "--decode-interpolation-granularity",
        type=int,
        default=_get(
            sweep_cfg,
            "decodeInterpolationGranularity",
            "decode_interpolation_granularity",
            6,
        ),
        help="how many samples to benchmark to interpolate ITL under different active kv cache size and decode context length",
    )
    parser.add_argument(
        "--service-name",
        type=str,
        default=_get(deployment_cfg, "serviceName", "service_name", ""),
        help="Service name for port forwarding (default: {deployment_name}-frontend)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=_get(sweep_cfg, "dryRun", "dry_run", False),
        help="Dry run the profile job",
    )
    parser.add_argument(
        "--enable-gpu-discovery",
        action="store_true",
        default=_get(hardware_cfg, "enableGpuDiscovery", "enable_gpu_discovery", False),
        help="Enable automatic GPU discovery from Kubernetes cluster nodes. When enabled, overrides any manually specified hardware configuration. Requires cluster-wide node access permissions.",
    )
    parser.add_argument(
        "--pick-with-webui",
        action="store_true",
        default=_get(sweep_cfg, "pickWithWebui", "pick_with_webui", False),
        help="Pick the best parallelization mapping using webUI",
    )

    default_webui_port = 8000
    webui_port_env = os.environ.get("PROFILER_WEBUI_PORT")
    if webui_port_env:
        default_webui_port = int(webui_port_env)
    parser.add_argument(
        "--webui-port",
        type=int,
        default=_get(sweep_cfg, "webuiPort", "webui_port", default_webui_port),
        help="WebUI port",
    )

    # Dynamically add all planner arguments from planner_argparse.py
    add_planner_arguments_to_parser(parser, prefix="planner-")
    # Set defaults for any planner arguments found in config.planner
    # Normalize keys: camelCase -> snake_case, hyphens -> underscores
    planner_config = config.get("planner", {})
    if planner_config:
        normalized_planner_config = {
            _camel_to_snake(key).replace("-", "_"): value
            for key, value in planner_config.items()
        }
        parser.set_defaults(**normalized_planner_config)

    # arguments if using aiconfigurator
    parser.add_argument(
        "--use-ai-configurator",
        action="store_true",
        default=_get(sweep_cfg, "useAiConfigurator", "use_ai_configurator", False),
        help="Use ai-configurator to estimate benchmarking results instead of running actual deployment.",
    )
    parser.add_argument(
        "--aic-system",
        type=str,
        default=_get(sweep_cfg, "aicSystem", "aic_system", None),
        help="Target system for use with aiconfigurator (e.g. h100_sxm, h200_sxm)",
    )
    parser.add_argument(
        "--aic-hf-id",
        type=str,
        default=_get(sweep_cfg, "aicHfId", "aic_hf_id", None),
        help="aiconfigurator name of the target model (e.g. Qwen/Qwen3-32B, meta-llama/Llama-3.1-405B)",
    )
    parser.add_argument(
        "--aic-backend",
        type=str,
        default=_get(sweep_cfg, "aicBackend", "aic_backend", ""),
        help="aiconfigurator backend of the target model, if not provided, will use args.backend",
    )
    parser.add_argument(
        "--aic-backend-version",
        type=str,
        default=_get(sweep_cfg, "aicBackendVersion", "aic_backend_version", None),
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
