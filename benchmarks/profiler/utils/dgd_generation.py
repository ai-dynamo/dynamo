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

import copy
import json
import os
from typing import Any, Optional

import numpy as np
import yaml

from benchmarks.profiler.utils.config import (
    Config,
    DgdPlannerServiceConfig,
    set_argument_value,
)
from benchmarks.profiler.utils.config_modifiers.parallelization_mapping import (
    ParallelizationMapping,
)
from benchmarks.profiler.utils.planner_utils import build_planner_args_from_namespace
from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.defaults import MockerComponentName, SubComponentType

# Path to mocker disagg config relative to workspace
MOCKER_DISAGG_CONFIG_PATH = "examples/backends/mocker/deploy/disagg.yaml"


def generate_dgd_config_with_planner(
    config_path: str,
    config_modifier,
    output_dir: str,
    args,
    best_prefill_mapping: ParallelizationMapping,
    best_decode_mapping: ParallelizationMapping,
    num_gpus_per_node: int = 8,
) -> tuple[list[dict] | dict, list[dict] | dict]:
    """Generate DGD config with planner based on profiling results.

    Args:
        config_path: Path to the YAML config file
        config_modifier: Config modifier instance (e.g., SGLangConfigModifier)
        output_dir: Output directory for profile results
        args: Parsed arguments namespace from profile_sla
        best_prefill_mapping: Parallel mapping for prefill (TP/TEP/DEP)
        best_decode_mapping: Parallel mapping for decode (TP/TEP/DEP)
        num_gpus_per_node: Number of GPUs per node (for TEP/DEP models)

    Returns:
        tuple: (dgd_config, mocker_config) where:
            - dgd_config: list[dict] | dict - If a ConfigMap is generated for planner data,
              returns a list of two YAML documents [ConfigMap, DGD]; otherwise returns a single DGD dict.
            - mocker_config: list[dict] | dict - Mocker DGD config with planner for testing purposes.
              If a ConfigMap is generated, returns [ConfigMap, DGD]; otherwise returns a single DGD dict.
    """

    # Load config from file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update container image if provided
    # This overrides the default image in the config file for all DGD components
    if args.dgd_image:
        config = config_modifier.update_image(config, args.dgd_image)

    # Apply prefill parallelization based on the actual mapping used in profiling
    if best_prefill_mapping.tp is not None:
        # Dense model or TP for prefill
        config = config_modifier.set_config_tp_size(
            config, best_prefill_mapping.tp, SubComponentType.PREFILL
        )
    elif best_prefill_mapping.tep is not None:
        # MoE model with TEP for prefill
        config = config_modifier.set_config_tep_size(
            config,
            best_prefill_mapping.tep,
            num_gpus_per_node,
            SubComponentType.PREFILL,
        )
    elif best_prefill_mapping.dep is not None:
        # MoE model with DEP for prefill
        config = config_modifier.set_config_dep_size(
            config,
            best_prefill_mapping.dep,
            num_gpus_per_node,
            SubComponentType.PREFILL,
        )

    # Apply decode parallelization based on the actual mapping used in profiling
    if best_decode_mapping.tp is not None:
        # Dense model or TP for decode
        config = config_modifier.set_config_tp_size(
            config, best_decode_mapping.tp, SubComponentType.DECODE
        )
    elif best_decode_mapping.tep is not None:
        # MoE model with TEP for decode
        config = config_modifier.set_config_tep_size(
            config,
            best_decode_mapping.tep,
            num_gpus_per_node,
            SubComponentType.DECODE,
        )
    elif best_decode_mapping.dep is not None:
        # MoE model with DEP for decode
        config = config_modifier.set_config_dep_size(
            config,
            best_decode_mapping.dep,
            num_gpus_per_node,
            SubComponentType.DECODE,
        )

    config = Config.model_validate(config)

    # add the planner service
    planner_config = DgdPlannerServiceConfig()
    frontend_service = config.spec.services["Frontend"]
    planner_config.dynamoNamespace = getattr(frontend_service, "dynamoNamespace", "dynamo")  # type: ignore[attr-defined]
    frontend_image: Optional[str] = None
    if frontend_service.extraPodSpec and frontend_service.extraPodSpec.mainContainer:
        frontend_image = frontend_service.extraPodSpec.mainContainer.image
        if frontend_image and planner_config.extraPodSpec.mainContainer:
            planner_config.extraPodSpec.mainContainer.image = frontend_image

    # Build planner args dynamically from parsed arguments
    # This includes shared args (ttft, itl, backend, namespace) from profile_sla
    # and planner-specific args (with planner_ prefix)
    planner_args = build_planner_args_from_namespace(args, prefix="planner_")

    # Override profiling-specific arguments with results from profiling
    # Remove and re-add to ensure correct values from profiling context
    # Note: --namespace is NOT added here; planner gets it from DYN_NAMESPACE env var
    # which is automatically injected by the operator based on dynamoNamespace
    planner_args = [
        arg
        for arg in planner_args
        if not any(
            arg.startswith(f"--{key}=")
            for key in [
                "namespace",
                "prefill-engine-num-gpu",
                "decode-engine-num-gpu",
                "profile-results-dir",
            ]
        )
    ]

    # Add arguments determined by profiling results
    cm_mount_path = f"{get_workspace_dir()}/profiling_results"
    planner_args.extend(
        [
            f"--prefill-engine-num-gpu={best_prefill_mapping.get_num_gpus()}",
            f"--decode-engine-num-gpu={best_decode_mapping.get_num_gpus()}",
            f"--profile-results-dir={cm_mount_path}",
        ]
    )

    if (
        planner_config.extraPodSpec.mainContainer
        and planner_config.extraPodSpec.mainContainer.args is not None
    ):
        planner_config.extraPodSpec.mainContainer.args.extend(planner_args)
    # Convert planner config to dict first, then the entire config to dict
    planner_dict = planner_config.model_dump(exclude_unset=False)
    config_dict = config.model_dump(exclude_unset=False)

    # Build a ConfigMap from NPZ profiling outputs and mount it into the Planner
    # We store data as plain JSON (lists/float/int) to avoid binary artifacts.
    prefill_npz = f"{output_dir}/selected_prefill_interpolation/raw_data.npz"
    decode_npz = f"{output_dir}/selected_decode_interpolation/raw_data.npz"

    config_map_obj: Optional[dict] = None
    try:
        with np.load(prefill_npz) as p_raw:
            prefill_json = {
                "prefill_isl": p_raw["prefill_isl"].tolist(),
                "prefill_ttft": p_raw["prefill_ttft"].tolist(),
                "prefill_thpt_per_gpu": p_raw["prefill_thpt_per_gpu"].tolist(),
            }
    except FileNotFoundError:
        prefill_json = None

    try:
        with np.load(decode_npz) as d_raw:
            # max_kv_tokens saved as array; convert to int
            max_kv_tokens = d_raw["max_kv_tokens"]
            if hasattr(max_kv_tokens, "tolist"):
                max_kv_tokens_val = max_kv_tokens.tolist()
                # Handle [value] vs value
                if isinstance(max_kv_tokens_val, list):
                    max_kv_tokens_val = (
                        int(max_kv_tokens_val[0]) if max_kv_tokens_val else 0
                    )
                else:
                    max_kv_tokens_val = int(max_kv_tokens_val)
            else:
                max_kv_tokens_val = int(max_kv_tokens)

            decode_json = {
                "x_kv_usage": d_raw["x_kv_usage"].tolist(),
                "y_context_length": d_raw["y_context_length"].tolist(),
                "z_itl": d_raw["z_itl"].tolist(),
                "z_thpt_per_gpu": d_raw["z_thpt_per_gpu"].tolist(),
                "max_kv_tokens": max_kv_tokens_val,
            }
    except FileNotFoundError:
        decode_json = None

    if prefill_json is not None and decode_json is not None:
        config_map_obj = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "planner-profile-data"},
            "data": {
                "prefill_raw_data.json": json.dumps(prefill_json),
                "decode_raw_data.json": json.dumps(decode_json),
            },
        }

        # Attach the ConfigMap as a volume in the Planner service
        planner_volumes = planner_dict.setdefault("extraPodSpec", {}).setdefault(
            "volumes", []
        )
        planner_volumes.append(
            {
                "name": "planner-profile-data",
                "configMap": {"name": "planner-profile-data"},
            }
        )
        mc_dict = planner_dict.setdefault("extraPodSpec", {}).setdefault(
            "mainContainer", {}
        )
        mc_mounts = mc_dict.setdefault("volumeMounts", [])
        mc_mounts.append(
            {
                "name": "planner-profile-data",
                "mountPath": cm_mount_path,
                "readOnly": True,
            }
        )

    # Finalize DGD services
    config_dict["spec"]["services"]["Planner"] = planner_dict

    # Generate mocker config with planner for testing purposes
    mocker_config = _generate_mocker_config_with_planner(
        args=args,
        cm_mount_path=cm_mount_path,
        config_map_obj=config_map_obj,
        planner_dict=planner_dict,
    )

    # Return multi-doc YAML (ConfigMap + DGD) when ConfigMap is created; else DGD only
    dgd_config: list[dict[str, Any]] | dict[str, Any]
    if config_map_obj is not None:
        dgd_config = [config_map_obj, config_dict]
    else:
        dgd_config = config_dict

    return dgd_config, mocker_config


def _generate_mocker_config_with_planner(
    args,
    cm_mount_path: str,
    config_map_obj: Optional[dict],
    planner_dict: dict,
) -> list[dict] | dict:
    """Generate mocker DGD config with planner for testing purposes.

    This loads the mocker disagg.yaml, updates the worker image, mounts the
    profiling ConfigMap, sets --planner-profile-data for workers, and adds the planner service.

    Args:
        args: Parsed arguments namespace from profile_sla
        cm_mount_path: Mount path for the ConfigMap containing profiling data
        config_map_obj: The ConfigMap object (if created)
        planner_dict: The planner service dict to reuse (includes planner image)

    Returns:
        list[dict] | dict: If a ConfigMap is generated, returns [ConfigMap, DGD];
            otherwise returns a single DGD dict.
    """
    # Load mocker disagg config
    workspace_dir = get_workspace_dir()
    mocker_config_path = os.path.join(workspace_dir, MOCKER_DISAGG_CONFIG_PATH)

    with open(mocker_config_path, "r") as f:
        mocker_config = yaml.safe_load(f)

    # Update worker image if provided
    if args.dgd_image:
        for service_name, service_config in (
            mocker_config.get("spec", {}).get("services", {}).items()
        ):
            if service_config.get("extraPodSpec") and service_config[
                "extraPodSpec"
            ].get("mainContainer"):
                service_config["extraPodSpec"]["mainContainer"][
                    "image"
                ] = args.dgd_image

    # Update worker args: --planner-profile-data, --model-path, --model-name
    mocker_worker_names = [
        MockerComponentName.prefill_worker_k8s_name,
        MockerComponentName.decode_worker_k8s_name,
    ]
    for worker_name in mocker_worker_names:
        service_config = (
            mocker_config.get("spec", {}).get("services", {}).get(worker_name)
        )
        if service_config:
            main_container = service_config.get("extraPodSpec", {}).get(
                "mainContainer", {}
            )
            args_list = main_container.get("args", [])
            args_list = set_argument_value(
                args_list, "--planner-profile-data", cm_mount_path
            )
            # Update model path and name if available in args
            args_list = set_argument_value(args_list, "--model-path", args.model)
            args_list = set_argument_value(args_list, "--model-name", args.model)
            main_container["args"] = args_list

    # Mount the ConfigMap if it exists
    if config_map_obj is not None:
        for worker_name in mocker_worker_names:
            service_config = (
                mocker_config.get("spec", {}).get("services", {}).get(worker_name)
            )
            if service_config:
                extra_pod_spec = service_config.setdefault("extraPodSpec", {})

                # Add volume
                volumes = extra_pod_spec.setdefault("volumes", [])
                volumes.append(
                    {
                        "name": "planner-profile-data",
                        "configMap": {"name": "planner-profile-data"},
                    }
                )

                # Add volume mount
                main_container = extra_pod_spec.setdefault("mainContainer", {})
                volume_mounts = main_container.setdefault("volumeMounts", [])
                volume_mounts.append(
                    {
                        "name": "planner-profile-data",
                        "mountPath": cm_mount_path,
                        "readOnly": True,
                    }
                )

    # Add planner service (reuse the same planner config but with mocker backend)
    mocker_planner_dict = copy.deepcopy(planner_dict)

    # Get the mocker's dynamoNamespace from Frontend service
    mocker_namespace = (
        mocker_config.get("spec", {})
        .get("services", {})
        .get("Frontend", {})
        .get("dynamoNamespace", "mocker-disagg")
    )

    # Update planner's dynamoNamespace to match mocker's namespace
    mocker_planner_dict["dynamoNamespace"] = mocker_namespace

    # Planner args use --key=value format, so we need to find and replace
    planner_main_container = mocker_planner_dict.get("extraPodSpec", {}).get(
        "mainContainer", {}
    )
    planner_args = planner_main_container.get("args", [])
    updated_planner_args = []
    for arg in planner_args:
        if arg.startswith("--backend="):
            updated_planner_args.append("--backend=mocker")
        else:
            updated_planner_args.append(arg)
    planner_main_container["args"] = updated_planner_args

    mocker_config["spec"]["services"]["Planner"] = mocker_planner_dict

    # Return multi-doc YAML (ConfigMap + DGD) when ConfigMap is created; else DGD only
    if config_map_obj is not None:
        return [config_map_obj, mocker_config]
    return mocker_config
