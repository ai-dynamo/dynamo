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

import json
from typing import Optional

import numpy as np
import yaml

from benchmarks.profiler.utils.config import Config, DgdPlannerServiceConfig
from benchmarks.profiler.utils.planner_utils import build_planner_args_from_namespace
from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.defaults import SubComponentType


def generate_dgd_config_with_planner(
    config_path: str,
    config_modifier,
    best_prefill_gpus: int,
    best_decode_gpus: int,
    output_dir: str,
    args,
    is_moe_model: bool = False,
    num_gpus_per_node: int = 8,
):
    """Generate DGD config with planner based on profiling results.

    Args:
        config_path: Path to the YAML config file
        config_modifier: Config modifier instance (e.g., SGLangConfigModifier)
        best_prefill_gpus: Number of GPUs for prefill engine
        best_decode_gpus: Number of GPUs for decode engine
        output_dir: Output directory for profile results
        args: Parsed arguments namespace from profile_sla
        is_moe_model: Whether this is an MoE model
        num_gpus_per_node: Number of GPUs per node (for MoE models)

    Returns:
        list[dict] | dict: If a ConfigMap is generated for planner data, returns a list
        of two YAML documents [ConfigMap, DGD]; otherwise returns a single DGD dict.
    """

    # Load config from file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update container image if provided
    # This overrides the default image in the config file for all DGD components
    if args.dgd_image:
        config = config_modifier.update_image(config, args.dgd_image)

    if not is_moe_model:
        # dense model, use TP for both prefill and decode
        config = config_modifier.set_config_tp_size(
            config, best_prefill_gpus, SubComponentType.PREFILL
        )
        config = config_modifier.set_config_tp_size(
            config, best_decode_gpus, SubComponentType.DECODE
        )
    else:
        # MoE model, use TEP for prefill and DEP for decode
        config = config_modifier.set_config_tep_size(
            config,
            best_prefill_gpus,
            num_gpus_per_node,
            SubComponentType.PREFILL,
        )
        config = config_modifier.set_config_dep_size(
            config,
            best_decode_gpus,
            num_gpus_per_node,
            SubComponentType.DECODE,
        )
    config = Config.model_validate(config)

    # add the planner service
    planner_config = DgdPlannerServiceConfig()
    frontend_service = config.spec.services["Frontend"]
    planner_config.dynamoNamespace = getattr(frontend_service, "dynamoNamespace", "dynamo")  # type: ignore[attr-defined]
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
    frontend_namespace = getattr(config.spec.services["Frontend"], "dynamoNamespace", "dynamo")  # type: ignore[attr-defined]
    cm_mount_path = f"{get_workspace_dir()}/profiling_results"
    planner_args.extend(
        [
            f"--namespace={frontend_namespace}",
            f"--prefill-engine-num-gpu={best_prefill_gpus}",
            f"--decode-engine-num-gpu={best_decode_gpus}",
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
                "mountPoint": cm_mount_path,
                "readOnly": True,
            }
        )

    # Finalize DGD services
    config_dict["spec"]["services"]["Planner"] = planner_dict

    # Return multi-doc YAML (ConfigMap + DGD) when ConfigMap is created; else DGD only
    if config_map_obj is not None:
        return [config_map_obj, config_dict]
    return config_dict
