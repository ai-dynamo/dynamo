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

import json
import logging
import os
import uuid
from typing import Any, Optional

import numpy as np
import yaml

from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.backend_components import (
    MockerComponentName,
    VllmComponentName,
)
from dynamo.planner.config.parallelization import (
    PickedParallelConfig,
    picked_to_aic_model_config_kwargs,
)
from dynamo.planner.config.planner_config import (
    PlannerConfig,
    PlannerPreDeploymentSweepMode,
)
from dynamo.profiler.utils.config import DgdPlannerServiceConfig, set_argument_value
from dynamo.profiler.utils.profile_common import (
    ProfilerOperationalConfig,
    derive_planner_image,
    is_mocker_enabled,
    is_planner_enabled,
    needs_profile_data,
)

logger = logging.getLogger(__name__)

# Path to mocker disagg config relative to workspace
MOCKER_DISAGG_CONFIG_PATH = "examples/backends/mocker/deploy/disagg.yaml"

# ConfigMap name prefixes (a 4-char UUID suffix is appended at runtime
# so that multiple deployments in the same namespace don't collide)
PLANNER_CONFIG_PREFIX = "planner-config"
PLANNER_PROFILE_DATA_PREFIX = "planner-profile-data"

# Well-known mount paths inside pods
PROFILE_DATA_MOUNT = f"{get_workspace_dir()}/profiling_results"
PLANNER_CONFIG_MOUNT = f"{get_workspace_dir()}/planner_config"


def _make_cm_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_final_config(
    dgdr,
    ops: ProfilerOperationalConfig,
    dgd_config: dict | None,
    best_prefill_config=None,
    best_decode_config=None,
    aic_spec: Optional[AICInterpolationSpec] = None,
    resolved_backend: Optional[str] = None,
) -> Any:
    """Apply Dynamo features to the picked DGD config via composable layers.

    1. **Mocker** — swap the base to the mocker DGD template if enabled.
    2. **vLLM self-benchmark** — when the resolved backend is vLLM, set
       ``DYN_BENCHMARK_MODE`` on each worker so the ``get_perf_metrics``
       endpoint is populated at runtime. The planner consumes this as
       priority 1 of its bootstrap chain, superseding AIC and files.
    3. **Planner** — inject the Planner service + planner-config ConfigMap.
       When ``aic_spec`` is given (rapid mode), it is embedded in the
       planner config so the planner runs AIC interpolation at bootstrap
       if the endpoint is unavailable.
    4. **Profile data** — attach interpolation-data ConfigMap when mocker
       or planner-thorough is enabled.
    """
    if not dgd_config:
        return dgd_config

    mocker = is_mocker_enabled(dgdr)
    planner = is_planner_enabled(dgdr)
    profile = needs_profile_data(dgdr)

    if not mocker and not planner:
        return dgd_config

    # Save picked config for auditing
    dgd_config_path = f"{ops.output_dir}/picked_dgd_config.yaml"
    with open(dgd_config_path, "w") as f:
        yaml.safe_dump(dgd_config, f, sort_keys=False)

    # Step 1: choose base config
    if mocker:
        logger.info("Mocker enabled — using mocker DGD as base.")
        base = generate_mocker_config(dgdr, aic_spec=aic_spec)
    else:
        base = dgd_config

    # Step 2: for vLLM deployments, turn on the per-worker self-benchmark so
    # the get_perf_metrics endpoint is available to the planner. Mocker
    # workers don't use DYN_BENCHMARK_MODE, so skip when mocker is active.
    if not mocker and resolved_backend == "vllm":
        enable_vllm_benchmark_mode(base)

    # Steps 3-4: layer features, collecting ConfigMaps
    config_maps: list[dict] = []

    if planner:
        planner_cm = add_planner_to_config(
            dgdr,
            base,
            best_prefill_mapping=best_prefill_config,
            best_decode_mapping=best_decode_config,
            aic_spec=aic_spec,
        )
        config_maps.append(planner_cm)

    if profile:
        output_dir = ops.output_dir if not ops.dry_run else None
        profile_cm = add_profile_data_to_config(base, output_dir, mocker_enabled=mocker)
        if profile_cm:
            config_maps.append(profile_cm)

    if config_maps:
        return config_maps + [base]
    return base


def _vllm_worker_roles() -> dict[str, str]:
    """Canonical DGD service name → DYN_BENCHMARK_MODE role.

    Sourced from :class:`VllmComponentName` so we stay in sync with the
    rest of the planner/profiler if the k8s service names are ever
    renamed.
    """
    return {
        VllmComponentName.prefill_worker_k8s_name: "prefill",
        VllmComponentName.decode_worker_k8s_name: "decode",
        VllmComponentName.agg_worker_k8s_name: "agg",
    }


def enable_vllm_benchmark_mode(config_dict: dict) -> None:
    """Set ``DYN_BENCHMARK_MODE`` on every vLLM worker in *config_dict*.

    Mutates ``config_dict`` in place. Each recognised worker service
    (``VllmPrefillWorker`` / ``VllmDecodeWorker`` / ``VllmWorker``) gets the
    mode matching its role so its startup self-benchmark publishes
    ForwardPassMetrics via the ``get_perf_metrics`` endpoint.

    Idempotent: if ``DYN_BENCHMARK_MODE`` is already set (e.g. via user
    overrides) the existing entry is replaced with the role-correct value.
    """
    services = config_dict.get("spec", {}).get("services", {})
    for svc_name, mode in _vllm_worker_roles().items():
        svc = services.get(svc_name)
        if svc is None:
            continue
        main_container = svc.setdefault("extraPodSpec", {}).setdefault(
            "mainContainer", {}
        )
        env_list = main_container.setdefault("env", [])
        # Strip any existing DYN_BENCHMARK_MODE; append canonical value.
        env_list[:] = [
            e
            for e in env_list
            if not (isinstance(e, dict) and e.get("name") == "DYN_BENCHMARK_MODE")
        ]
        env_list.append({"name": "DYN_BENCHMARK_MODE", "value": mode})
        logger.info(
            "Enabled vLLM self-benchmark on service %s (DYN_BENCHMARK_MODE=%s)",
            svc_name,
            mode,
        )


def generate_mocker_config(
    dgdr, aic_spec: Optional[AICInterpolationSpec] = None
) -> dict:
    """Load the mocker DGD template and apply DGDR images and model paths.

    When ``aic_spec`` is provided (planner-rapid with an AIC-supported backend),
    inject ``--aic-perf-model`` plus related flags onto the prefill/decode
    workers so each mocker pod pulls its latency model directly from the
    AIConfigurator SDK at runtime — no NPZ round-trip through the profiler.

    Returns:
        The mocker DGD config dict (no planner, no ConfigMaps).
    """
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
    aic_workers = _mocker_aic_worker_picks(aic_spec)
    for worker_name in _mocker_worker_names():
        service_config = (
            mocker_config.get("spec", {}).get("services", {}).get(worker_name)
        )
        if service_config:
            main_container = service_config.get("extraPodSpec", {}).get(
                "mainContainer", {}
            )
            args_list = main_container.get("args", [])
            args_list = set_argument_value(args_list, "--model-path", model)
            args_list = set_argument_value(args_list, "--model-name", model)
            pick = aic_workers.get(worker_name) if aic_workers else None
            if pick is not None and aic_spec is not None:
                args_list = _inject_mocker_aic_args(args_list, aic_spec, pick)
            main_container["args"] = args_list

    return mocker_config


def _mocker_aic_worker_picks(
    aic_spec: Optional[AICInterpolationSpec],
) -> Optional[dict[str, PickedParallelConfig]]:
    if aic_spec is None:
        return None
    return {
        MockerComponentName.prefill_worker_k8s_name: aic_spec.prefill_pick,
        MockerComponentName.decode_worker_k8s_name: aic_spec.decode_pick,
    }


def _inject_mocker_aic_args(
    args_list: list,
    aic_spec: AICInterpolationSpec,
    pick: PickedParallelConfig,
) -> list:
    """Inject ``--aic-*`` flags onto a single mocker worker's args list.

    The mocker simulates vllm/sglang scheduling; for trtllm AIC data we keep
    the default ``--engine-type`` and only override ``--aic-backend`` so the
    perf-model lookups point at the correct database.
    """
    kwargs = picked_to_aic_model_config_kwargs(pick)
    if "--aic-perf-model" not in args_list:
        args_list.append("--aic-perf-model")
    args_list = set_argument_value(args_list, "--aic-backend", aic_spec.backend)
    args_list = set_argument_value(args_list, "--aic-system", aic_spec.system)
    args_list = set_argument_value(args_list, "--aic-tp-size", str(kwargs["tp_size"]))
    args_list = set_argument_value(
        args_list, "--aic-moe-tp-size", str(kwargs["moe_tp_size"])
    )
    args_list = set_argument_value(
        args_list, "--aic-moe-ep-size", str(kwargs["moe_ep_size"])
    )
    args_list = set_argument_value(
        args_list, "--aic-attention-dp-size", str(kwargs["attention_dp_size"])
    )
    if aic_spec.backend in ("vllm", "sglang"):
        args_list = set_argument_value(args_list, "--engine-type", aic_spec.backend)
    return args_list


def add_planner_to_config(
    dgdr,
    config_dict: dict,
    best_prefill_mapping=None,
    best_decode_mapping=None,
    aic_spec: Optional[AICInterpolationSpec] = None,
) -> dict:
    """Add a Planner service and its planner-config ConfigMap to *config_dict*.

    The planner's ``profile_results_dir`` is always set to the well-known
    mount path so the pod knows where to look when profile data is
    mounted separately by :func:`add_profile_data_to_config`.

    Args:
        dgdr: DynamoGraphDeploymentRequestSpec.
        config_dict: The base DGD config (real or mocker) — mutated in place.
        best_prefill_mapping: Picked prefill parallel config.
        best_decode_mapping: Picked decode parallel config.
        aic_spec: AIC interpolation spec (rapid mode). When set, the planner
            runs AIC in-process at bootstrap instead of reading NPZ files.

    Returns:
        The ``planner_config_cm`` ConfigMap dict.
    """
    planner_cfg = _build_planner_config(
        dgdr, best_prefill_mapping, best_decode_mapping, aic_spec
    )
    planner_cfg.profile_results_dir = PROFILE_DATA_MOUNT

    planner_service = DgdPlannerServiceConfig()
    if planner_service.extraPodSpec.mainContainer and dgdr.image:
        planner_service.extraPodSpec.mainContainer.image = derive_planner_image(
            dgdr.image
        )

    planner_dict = planner_service.model_dump(exclude_unset=False)

    planner_config_cm_name = _make_cm_name(PLANNER_CONFIG_PREFIX)

    # --- ConfigMap: planner config ---
    planner_config_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": planner_config_cm_name},
        "data": {
            "planner_config.json": planner_cfg.model_dump_json(),
        },
    }

    # --- Mount planner-config ConfigMap into the planner service ---
    planner_volumes = planner_dict.setdefault("extraPodSpec", {}).setdefault(
        "volumes", []
    )
    mc_dict = planner_dict.setdefault("extraPodSpec", {}).setdefault(
        "mainContainer", {}
    )
    mc_mounts = mc_dict.setdefault("volumeMounts", [])

    planner_volumes.append(
        {
            "name": planner_config_cm_name,
            "configMap": {"name": planner_config_cm_name},
        }
    )
    mc_mounts.append(
        {
            "name": planner_config_cm_name,
            "mountPath": PLANNER_CONFIG_MOUNT,
            "readOnly": True,
        }
    )

    mc_args = mc_dict.setdefault("args", [])
    mc_args.extend(["--config", f"{PLANNER_CONFIG_MOUNT}/planner_config.json"])

    config_dict["spec"]["services"]["Planner"] = planner_dict

    return planner_config_cm


def add_profile_data_to_config(
    config_dict: dict,
    output_dir: str | None,
    mocker_enabled: bool = False,
) -> Optional[dict]:
    """Create a profile-data ConfigMap and mount it into consumers in *config_dict*.

    Consumers are auto-detected:
    - The **Planner** service (if present) gets the volume mounted.
    - **Mocker workers** (when *mocker_enabled*) get the volume mounted and
      ``--planner-profile-data`` set.

    Args:
        config_dict: The DGD config dict — mutated in place.
        output_dir: Directory containing profiling interpolation NPZ files.
        mocker_enabled: Only inject ``--planner-profile-data`` into workers
            when the mocker backend is active.  Non-mocker backends (vllm,
            sglang, trtllm) do not recognise this argument.

    Returns:
        The ``profile_data_cm`` ConfigMap dict, or ``None`` if no profiling
        data was found.
    """
    profiling_data = _load_profiling_data(output_dir) if output_dir else {}
    if not profiling_data:
        return None

    profile_data_cm_name = _make_cm_name(PLANNER_PROFILE_DATA_PREFIX)

    profile_cm_data: dict[str, str] = {}
    # TODO: use enums
    if profiling_data.get("prefill"):
        profile_cm_data["prefill_raw_data.json"] = json.dumps(profiling_data["prefill"])
    if profiling_data.get("decode"):
        profile_cm_data["decode_raw_data.json"] = json.dumps(profiling_data["decode"])

    profile_data_cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": profile_data_cm_name},
        "data": profile_cm_data,
    }

    # Mount into Planner service if it exists
    planner_svc = config_dict.get("spec", {}).get("services", {}).get("Planner")
    if planner_svc is not None:
        _mount_volume_into_service(
            planner_svc, profile_data_cm_name, PROFILE_DATA_MOUNT
        )

    # Mount into mocker workers only when the mocker backend is active.
    # Non-mocker backends (vllm, sglang, trtllm) share the same service
    # names ("prefill", "decode") but do not accept --planner-profile-data.
    if mocker_enabled:
        services = config_dict.get("spec", {}).get("services", {})
        for worker_name in _mocker_worker_names():
            worker_svc = services.get(worker_name)
            if worker_svc is not None:
                main_container = worker_svc.get("extraPodSpec", {}).get(
                    "mainContainer", {}
                )
                args_list = main_container.get("args", [])
                args_list = set_argument_value(
                    args_list, "--planner-profile-data", PROFILE_DATA_MOUNT
                )
                main_container["args"] = args_list
                _mount_volume_into_service(
                    worker_svc, profile_data_cm_name, PROFILE_DATA_MOUNT
                )

    return profile_data_cm


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mocker_worker_names() -> list[str]:
    return [
        MockerComponentName.prefill_worker_k8s_name,
        MockerComponentName.decode_worker_k8s_name,
    ]


def _mount_volume_into_service(
    service_dict: dict, cm_name: str, mount_path: str
) -> None:
    """Add a ConfigMap volume + volumeMount to a service's extraPodSpec."""
    extra_pod_spec = service_dict.setdefault("extraPodSpec", {})
    volumes = extra_pod_spec.setdefault("volumes", [])
    volumes.append(
        {
            "name": cm_name,
            "configMap": {"name": cm_name},
        }
    )
    main_container = extra_pod_spec.setdefault("mainContainer", {})
    volume_mounts = main_container.setdefault("volumeMounts", [])
    volume_mounts.append(
        {
            "name": cm_name,
            "mountPath": mount_path,
            "readOnly": True,
        }
    )


def _build_planner_config(
    dgdr,
    best_prefill_mapping,
    best_decode_mapping,
    aic_spec: Optional[AICInterpolationSpec] = None,
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

    if aic_spec is not None:
        planner_cfg.aic_interpolation = aic_spec

    return planner_cfg


def build_aic_interpolation_spec(
    dgdr,
    best_prefill_pick: Optional[PickedParallelConfig],
    best_decode_pick: Optional[PickedParallelConfig],
    isl: int,
    osl: int,
    sweep_max_context_length: int,
    resolved_backend: str,
    system: str,
    prefill_interpolation_granularity: int,
    decode_interpolation_granularity: int,
) -> Optional[AICInterpolationSpec]:
    """Build an ``AICInterpolationSpec`` for rapid-mode AIC consumers.

    Consumed by both the planner (to bootstrap perf models in-process) and
    the mocker (via ``--aic-perf-model`` flags injected into worker args).
    Returns ``None`` when any of the following hold:

    * no AIC consumer needs it — planner is disabled or has
      ``enable_throughput_scaling=False``, **and** mocker is disabled
    * ``pre_deployment_sweeping_mode`` is not ``Rapid``
    * picks are missing
    * ``resolved_backend`` is not one AIC supports
    """
    planner = (
        dgdr.features.planner  # type: ignore[union-attr]
        if dgdr.features is not None and dgdr.features.planner is not None
        else None
    )
    mocker_enabled = is_mocker_enabled(dgdr)
    planner_needs_aic = (
        is_planner_enabled(dgdr)
        and planner is not None
        and planner.enable_throughput_scaling
    )
    if not planner_needs_aic and not mocker_enabled:
        return None
    sweep_mode = planner.pre_deployment_sweeping_mode if planner is not None else None
    if sweep_mode != PlannerPreDeploymentSweepMode.Rapid:
        return None
    if best_prefill_pick is None or best_decode_pick is None:
        logger.info(
            "Rapid mode but picks are missing; skipping aic_interpolation spec."
        )
        return None
    if resolved_backend not in ("trtllm", "vllm", "sglang"):
        logger.info(
            "Rapid mode but backend %r is not supported by AIC; skipping spec.",
            resolved_backend,
        )
        return None

    return AICInterpolationSpec(
        hf_id=dgdr.model,
        system=system,
        backend=resolved_backend,
        isl=isl,
        osl=osl,
        sweep_max_context_length=sweep_max_context_length,
        prefill_interpolation_granularity=prefill_interpolation_granularity,
        decode_interpolation_granularity=decode_interpolation_granularity,
        prefill_pick=best_prefill_pick,
        decode_pick=best_decode_pick,
    )


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
