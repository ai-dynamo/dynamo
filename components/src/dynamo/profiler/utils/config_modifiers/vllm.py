# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

import yaml

from dynamo.planner.config.defaults import SubComponentType
from dynamo.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    get_service_name_by_type,
    get_worker_service_from_config,
    remove_valued_arguments,
    set_argument_value,
    setup_worker_service_resources,
    update_image,
    validate_and_get_worker_args,
)
from dynamo.profiler.utils.config_modifiers.protocol import BaseConfigModifier
from dynamo.profiler.utils.defaults import (
    DYNAMO_RUN_DEFAULT_PORT,
    EngineType,
    resolve_deploy_path,
)
from dynamo.profiler.utils.dgdr_v1beta1_types import DeviceType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DEFAULT_VLLM_DISAGG_CONFIG_PATH = resolve_deploy_path(
    "examples/backends/vllm/deploy/disagg.yaml"
)
DEFAULT_VLLM_AGG_CONFIG_PATH = resolve_deploy_path(
    "examples/backends/vllm/deploy/agg.yaml"
)


class VllmV1ConfigModifier(BaseConfigModifier):
    BACKEND = "vllm"
    # vllm uses a different arg for model path
    WORKER_MODEL_PATH_ARG = "--model"

    @classmethod
    def load_default_config(cls, mode: str = "disagg") -> dict:
        path = (
            DEFAULT_VLLM_AGG_CONFIG_PATH
            if mode == "agg"
            else DEFAULT_VLLM_DISAGG_CONFIG_PATH
        )
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def update_image(cls, config, image: str) -> dict:
        """Update container image for all DGD services (frontend, planner, workers)."""
        return update_image(config, image)

    @classmethod
    def set_device_type(cls, config: dict, device_type: str) -> dict:
        """Inject XPU-specific configuration when device_type is 'xpu'.

        For NVIDIA CUDA (default), this method is a no-op.
        For Intel XPU:
        - Adds ``VLLM_TARGET_DEVICE=xpu`` to each worker's env.
        - Removes generic ``gpu`` resource limits/requests (not used by Intel DRA).
        - Adds a ``ResourceClaimTemplate`` for Intel DRA (``gpu.intel.com``) and
          the matching ``resourceClaims`` + ``claims`` references to each worker
          pod spec, following the pattern in
          ``examples/backends/vllm/deploy/agg_xpu_dra.yaml``.
        """
        device_str = (
            device_type.value
            if isinstance(device_type, DeviceType)
            else str(device_type).lower()
        )
        if device_str != DeviceType.Xpu.value:
            return config

        cfg = Config.model_validate(config)
        worker_names: list[str] = []
        for name, service in cfg.spec.services.items():
            if name in cls._NON_WORKER_SERVICES:
                continue
            if not service.extraPodSpec or not service.extraPodSpec.mainContainer:
                continue

            container = service.extraPodSpec.mainContainer

            # 1. Inject VLLM_TARGET_DEVICE=xpu env var
            env: list = list(container.model_extra.get("env") or [])
            if not any(
                isinstance(e, dict) and e.get("name") == "VLLM_TARGET_DEVICE"
                for e in env
            ):
                env.append({"name": "VLLM_TARGET_DEVICE", "value": "xpu"})
            container.model_extra["env"] = env

            # 2. Remove NVIDIA-style gpu resource requests/limits (incompatible with DRA)
            if service.resources:
                for bucket in ("requests", "limits"):
                    res_dict = getattr(service.resources, bucket, None) or {}
                    if isinstance(res_dict, dict):
                        res_dict.pop("gpu", None)
                        setattr(service.resources, bucket, res_dict or None)

            worker_names.append(name)

        # 3. Add a single shared ResourceClaimTemplate for Intel DRA and wire
        #    each worker pod to use it.  Derive template name from the DGD
        #    metadata to avoid collisions across deployments in the same namespace.
        template_name = (
            f"{cfg.metadata.name}-gpu-template" if cfg.metadata.name else "gpu-template"
        )
        if worker_names:
            # Attach resourceClaims + container claims to each worker's extraPodSpec
            for name in worker_names:
                service = cfg.spec.services[name]
                pod_extra = service.extraPodSpec.model_extra
                # resourceClaims at pod level
                pod_extra.setdefault("resourceClaims", [])
                existing_claim_names = [
                    rc.get("name")
                    for rc in pod_extra["resourceClaims"]
                    if isinstance(rc, dict)
                ]
                if "gpu" not in existing_claim_names:
                    pod_extra["resourceClaims"].append(
                        {"name": "gpu", "resourceClaimTemplateName": template_name}
                    )
                # claims at container level (grants access to the pod-level claim)
                container = service.extraPodSpec.mainContainer
                container_resources = container.model_extra.get("resources") or {}
                claims: list = container_resources.get("claims") or []
                if not any(
                    isinstance(c, dict) and c.get("name") == "gpu" for c in claims
                ):
                    claims.append({"name": "gpu"})
                container_resources["claims"] = claims
                container.model_extra["resources"] = container_resources

        dumped = cfg.model_dump()

        # 4. Prepend the ResourceClaimTemplate as a separate K8s document.
        #    Callers that write multi-doc YAML will naturally pick this up.
        #    We tag it in a wrapper key so dgd_generation can detect and
        #    emit it as a separate YAML document.
        resource_claim_template = {
            "apiVersion": "resource.k8s.io/v1",
            "kind": "ResourceClaimTemplate",
            "metadata": {"name": template_name},
            "spec": {
                "spec": {
                    "devices": {
                        "requests": [
                            {
                                "name": "gpu",
                                "exactly": {
                                    "deviceClassName": "gpu.intel.com",
                                    "count": 1,
                                },
                            }
                        ]
                    }
                }
            },
        }
        dumped["_xpu_resource_claim_template"] = resource_claim_template
        return dumped

    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: EngineType,
        is_moe_model: bool = False,
    ) -> dict:
        cfg = Config.model_validate(config)

        # MoE flags (--enable-expert-parallel) are set in set_config_tep_size/set_config_dep_size

        # set metadata name
        cfg.metadata.name = "agg"

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

            # remove --disaggregation-mode and its value (or legacy --is-prefill-worker)
            args = remove_valued_arguments(args, "--disaggregation-mode")
            if "--is-prefill-worker" in args:
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
    ) -> dict:
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources
        setup_worker_service_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Remove --tp alias if present, use --tensor-parallel-size as canonical form
        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", str(tp_size))

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
        """
        Set Tensor Expert Parallelism (TEP) for vLLM MoE models.

        vLLM derives expert parallelism size automatically:
        expert_parallel_size = tensor_parallel_size * data_parallel_size

        For TEP: TP=tep_size, DP=1 → EP size = tep_size
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, tep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Remove aliases, use canonical forms
        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", str(tep_size))
        args = remove_valued_arguments(args, "--dp")
        args = set_argument_value(args, "--data-parallel-size", "1")

        # Remove hybrid load balancing flags - not compatible with DP=1
        args = remove_valued_arguments(args, "--data-parallel-size-local")
        if "--data-parallel-hybrid-lb" in args:
            args.remove("--data-parallel-hybrid-lb")

        # Enable expert parallel for MoE
        if "--enable-expert-parallel" not in args:
            args = append_argument(args, "--enable-expert-parallel")

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
        """
        Set Data Expert Parallelism (DEP) for vLLM MoE models.

        vLLM derives expert parallelism size automatically:
        expert_parallel_size = tensor_parallel_size * data_parallel_size

        For DEP: TP=1, DP=dep_size → EP size = dep_size
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_service_resources(worker_service, dep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Remove aliases, use canonical forms
        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", "1")
        args = remove_valued_arguments(args, "--dp")
        args = set_argument_value(args, "--data-parallel-size", str(dep_size))

        # Handle hybrid load balancing for multinode DEP
        # If dep_size > num_gpus_per_node, we need multinode and can use hybrid-lb
        if dep_size > num_gpus_per_node and "--data-parallel-hybrid-lb" in args:
            # Set local DP size to GPUs per node for hybrid load balancing
            args = set_argument_value(
                args, "--data-parallel-size-local", str(num_gpus_per_node)
            )
        else:
            # Remove hybrid-lb flags if not needed or not multinode
            args = remove_valued_arguments(args, "--data-parallel-size-local")
            if "--data-parallel-hybrid-lb" in args:
                args.remove("--data-parallel-hybrid-lb")

        # Enable expert parallel for MoE
        if "--enable-expert-parallel" not in args:
            args = append_argument(args, "--enable-expert-parallel")

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> Tuple[str, str]:
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(cfg, backend="vllm")
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)
        return cls._get_model_name_and_path_from_args(args)

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
                        try:
                            line = line.strip().split("Maximum concurrency for ")[1]
                            token_count = int(
                                line.split(" tokens per request: ")[0].replace(",", "")
                            )
                            concurrency = float(
                                line.split(" tokens per request: ")[1][:-1]
                            )

                            # Log shows per-rank KV cache; multiply by attention_dp_size for total
                            kv_cache_per_rank = int(token_count * concurrency)
                            total_kv_cache = kv_cache_per_rank * attention_dp_size
                            logger.info(
                                f"Found KV cache: {kv_cache_per_rank} per rank x {attention_dp_size} = {total_kv_cache} total"
                            )
                            return total_kv_cache
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse KV cache size from line: {line}. Error: {e}"
                            )
        except FileNotFoundError:
            logger.warning(f"Log file not found: {dynamo_log_fn}")
        except Exception as e:
            logger.warning(f"Failed to read log file {dynamo_log_fn}: {e}")
        return 0

    @classmethod
    def set_prefill_config(
        cls,
        config: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """
        Configure prefill-related limits for aggregated prefill runs.
        vLLM uses --max-num-seqs to limit concurrency and
        --max-num-batched-tokens to cap total tokens per step.

        In vLLM, --max-num-batched-tokens controls per-GPU buffer allocation
        during memory profiling. For DEP (DP > 1), we must use the base token
        limit per GPU, not the multiplied total, to avoid OOM during profiling.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_service_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Get DP size from args (check both --dp and --data-parallel-size aliases)
        dp_size = 1
        for i, arg in enumerate(args):
            if arg in ("--dp", "--data-parallel-size") and i + 1 < len(args):
                dp_size = int(args[i + 1])
                break

        # For DEP (DP > 1), compute per-GPU token limit to avoid OOM
        per_gpu_max_tokens = (
            max_num_tokens // dp_size if dp_size > 1 else max_num_tokens
        )

        args = set_argument_value(args, "--max-num-seqs", str(max_batch_size))
        args = set_argument_value(
            args, "--max-num-batched-tokens", str(per_gpu_max_tokens)
        )

        worker_service.extraPodSpec.mainContainer.args = args
        return cfg.model_dump()
