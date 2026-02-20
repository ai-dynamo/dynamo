# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.defaults import MockerComponentName
from dynamo.planner.utils.planner_config import PlannerConfig
from dynamo.profiler.utils.config import (
    Config,
    DgdPlannerServiceConfig,
    set_argument_value,
)

# Path to mocker disagg config relative to workspace
MOCKER_DISAGG_CONFIG_PATH = "examples/backends/mocker/deploy/disagg.yaml"


def generate_dgd_config_with_planner(
    dgdr,
    config_path: str,
    output_dir: str | None,
    best_prefill_mapping=None,
    best_decode_mapping=None,
) -> tuple[list[dict] | dict, list[dict] | dict]:
    """Generate DGD config with planner based on profiling results.

    The ``config_path`` should point to a DGD YAML that already has the
    correct parallelization and image applied (produced by AIC's generator
    pipeline).  This function loads it, adds the planner service (with
    profiling data ConfigMap if available), and produces the final
    deployable DGD.

    Args:
        dgdr: DynamoGraphDeploymentRequestSpec.
        config_path: Path to the picked DGD YAML config file (already has
            correct parallelization, replicas, and image).
        output_dir: Output directory containing profiling interpolation data.
        best_prefill_mapping: Picked prefill parallel config (PickedParallelConfig).
            Used only for ``prefill_engine_num_gpu`` in PlannerConfig.
        best_decode_mapping: Picked decode parallel config (PickedParallelConfig).
            Used only for ``decode_engine_num_gpu`` in PlannerConfig.

    Returns:
        tuple: (dgd_config, mocker_config)
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config.model_validate(raw)

    # --- Build PlannerConfig ---
    planner_cfg = _build_planner_config(
        dgdr,
        best_prefill_mapping,
        best_decode_mapping,
    )

    # --- Add planner service to DGD ---
    planner_service = DgdPlannerServiceConfig()
    frontend_service = config.spec.services["Frontend"]
    if frontend_service.extraPodSpec and frontend_service.extraPodSpec.mainContainer:
        frontend_image = frontend_service.extraPodSpec.mainContainer.image
        if frontend_image and planner_service.extraPodSpec.mainContainer:
            planner_service.extraPodSpec.mainContainer.image = frontend_image

    planner_dict = planner_service.model_dump(exclude_unset=False)
    config_dict = config.model_dump(exclude_unset=False)

    profile_data_mount = f"{get_workspace_dir()}/profiling_results"
    planner_config_mount = f"{get_workspace_dir()}/planner_config"

    # --- ConfigMap 1: profiling interpolation data ---
    profile_data_cm: Optional[dict] = None
    profiling_data = _load_profiling_data(output_dir) if output_dir else {}
    if profiling_data:
        planner_cfg.profile_results_dir = profile_data_mount

        profile_cm_data: dict[str, str] = {}
        if profiling_data.get("prefill"):
            profile_cm_data["prefill_raw_data.json"] = json.dumps(
                profiling_data["prefill"]
            )
        if profiling_data.get("decode"):
            profile_cm_data["decode_raw_data.json"] = json.dumps(
                profiling_data["decode"]
            )

        profile_data_cm = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "planner-profile-data"},
            "data": profile_cm_data,
        }

    # --- ConfigMap 2: planner config ---
    planner_config_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "planner-config"},
        "data": {
            "planner_config.json": planner_cfg.model_dump_json(),
        },
    }

    # --- Mount both ConfigMaps into the planner service ---
    planner_volumes = planner_dict.setdefault("extraPodSpec", {}).setdefault(
        "volumes", []
    )
    mc_dict = planner_dict.setdefault("extraPodSpec", {}).setdefault(
        "mainContainer", {}
    )
    mc_mounts = mc_dict.setdefault("volumeMounts", [])

    # Planner config volume
    planner_volumes.append(
        {
            "name": "planner-config",
            "configMap": {"name": "planner-config"},
        }
    )
    mc_mounts.append(
        {
            "name": "planner-config",
            "mountPath": planner_config_mount,
            "readOnly": True,
        }
    )

    # Profiling data volume (only if data exists)
    if profile_data_cm is not None:
        planner_volumes.append(
            {
                "name": "planner-profile-data",
                "configMap": {"name": "planner-profile-data"},
            }
        )
        mc_mounts.append(
            {
                "name": "planner-profile-data",
                "mountPath": profile_data_mount,
                "readOnly": True,
            }
        )

    # Planner reads its config from the mounted planner-config ConfigMap
    mc_args = mc_dict.setdefault("args", [])
    mc_args.extend(["--config", f"{planner_config_mount}/planner_config.json"])

    config_dict["spec"]["services"]["Planner"] = planner_dict

    # --- Generate mocker config ---
    mocker_config = _generate_mocker_config_with_planner(
        dgdr=dgdr,
        profile_data_mount=profile_data_mount,
        planner_config_mount=planner_config_mount,
        profile_data_cm=profile_data_cm,
        planner_config_cm=planner_config_cm,
        planner_dict=planner_dict,
    )

    # Collect all ConfigMaps + DGD into multi-doc output
    config_maps = [cm for cm in [profile_data_cm, planner_config_cm] if cm is not None]
    if config_maps:
        dgd_config: list[dict[str, Any]] = config_maps + [config_dict]
    else:
        dgd_config = config_dict

    return dgd_config, mocker_config


def _build_planner_config(
    dgdr,
    best_prefill_mapping,
    best_decode_mapping,
) -> PlannerConfig:
    """Build a PlannerConfig from the DGDR spec and picked parallel configs."""
    if dgdr.features and dgdr.features.planner:
        planner_cfg = dgdr.features.planner.model_copy(deep=True)
    else:
        planner_cfg = PlannerConfig()

    if best_prefill_mapping is not None:
        planner_cfg.prefill_engine_num_gpu = best_prefill_mapping.num_gpus

    if best_decode_mapping is not None:
        planner_cfg.decode_engine_num_gpu = best_decode_mapping.num_gpus

    return planner_cfg


def _load_profiling_data(output_dir: str) -> dict:
    """Load interpolation profiling data from NPZ files."""
    result: dict = {}

    prefill_npz = f"{output_dir}/selected_prefill_interpolation/raw_data.npz"
    try:
        with np.load(prefill_npz) as p_raw:
            result["prefill"] = {
                "prefill_isl": p_raw["prefill_isl"].tolist(),
                "prefill_ttft": p_raw["prefill_ttft"].tolist(),
                "prefill_thpt_per_gpu": p_raw["prefill_thpt_per_gpu"].tolist(),
            }
    except FileNotFoundError:
        pass

    decode_npz = f"{output_dir}/selected_decode_interpolation/raw_data.npz"
    try:
        with np.load(decode_npz) as d_raw:
            max_kv_tokens = d_raw["max_kv_tokens"]
            if hasattr(max_kv_tokens, "tolist"):
                max_kv_tokens_val = max_kv_tokens.tolist()
                if isinstance(max_kv_tokens_val, list):
                    max_kv_tokens_val = (
                        int(max_kv_tokens_val[0]) if max_kv_tokens_val else 0
                    )
                else:
                    max_kv_tokens_val = int(max_kv_tokens_val)
            else:
                max_kv_tokens_val = int(max_kv_tokens)

            result["decode"] = {
                "x_kv_usage": d_raw["x_kv_usage"].tolist(),
                "y_context_length": d_raw["y_context_length"].tolist(),
                "z_itl": d_raw["z_itl"].tolist(),
                "z_thpt_per_gpu": d_raw["z_thpt_per_gpu"].tolist(),
                "max_kv_tokens": max_kv_tokens_val,
            }
    except FileNotFoundError:
        pass

    return result


def _generate_mocker_config_with_planner(
    dgdr,
    profile_data_mount: str,
    planner_config_mount: str,
    profile_data_cm: Optional[dict],
    planner_config_cm: dict,
    planner_dict: dict,
) -> list[dict] | dict:
    """Generate mocker DGD config with planner for testing purposes."""
    workspace_dir = get_workspace_dir()
    mocker_config_path = os.path.join(workspace_dir, MOCKER_DISAGG_CONFIG_PATH)

    with open(mocker_config_path, "r") as f:
        mocker_config = yaml.safe_load(f)

    image = dgdr.image
    if image:
        for service_config in (
            mocker_config.get("spec", {}).get("services", {}).values()
        ):
            if service_config.get("extraPodSpec") and service_config[
                "extraPodSpec"
            ].get("mainContainer"):
                service_config["extraPodSpec"]["mainContainer"]["image"] = image

    model = dgdr.model
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
            if profile_data_cm is not None:
                args_list = set_argument_value(
                    args_list, "--planner-profile-data", profile_data_mount
                )
            args_list = set_argument_value(args_list, "--model-path", model)
            args_list = set_argument_value(args_list, "--model-name", model)
            main_container["args"] = args_list

    # Mount profiling data ConfigMap into mocker workers
    if profile_data_cm is not None:
        for worker_name in mocker_worker_names:
            service_config = (
                mocker_config.get("spec", {}).get("services", {}).get(worker_name)
            )
            if service_config:
                extra_pod_spec = service_config.setdefault("extraPodSpec", {})
                volumes = extra_pod_spec.setdefault("volumes", [])
                volumes.append(
                    {
                        "name": "planner-profile-data",
                        "configMap": {"name": "planner-profile-data"},
                    }
                )
                main_container = extra_pod_spec.setdefault("mainContainer", {})
                volume_mounts = main_container.setdefault("volumeMounts", [])
                volume_mounts.append(
                    {
                        "name": "planner-profile-data",
                        "mountPath": profile_data_mount,
                        "readOnly": True,
                    }
                )

    # Reuse planner service dict (already has both ConfigMaps mounted + --config arg)
    mocker_planner_dict = copy.deepcopy(planner_dict)
    mocker_config["spec"]["services"]["Planner"] = mocker_planner_dict

    config_maps = [cm for cm in [profile_data_cm, planner_config_cm] if cm is not None]
    if config_maps:
        return config_maps + [mocker_config]
    return mocker_config
