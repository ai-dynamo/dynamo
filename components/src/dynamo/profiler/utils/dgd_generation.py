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

import importlib
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import yaml

from dynamo.common.utils.paths import get_workspace_dir
from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.backend_components import MockerComponentName
from dynamo.planner.config.parallelization import (
    PickedParallelConfig,
    picked_to_aic_model_config_kwargs,
)
from dynamo.planner.config.planner_config import (
    AICPerfModelSpec,
    PlannerConfig,
    PlannerPreDeploymentSweepMode,
)
from dynamo.profiler.utils.config import (
    DgdPlannerComponentConfig,
    get_component_dict,
    get_main_container,
    get_main_container_dict,
    set_argument_value,
)
from dynamo.profiler.utils.config_modifiers.trtllm import enable_trtllm_chunked_prefill
from dynamo.profiler.utils.dgd_template import load_dgd_template
from dynamo.profiler.utils.profile_common import (
    ProfilerOperationalConfig,
    derive_planner_image,
    is_mocker_enabled,
    is_planner_enabled,
    needs_mocker_aic_perf_model,
    needs_profile_data,
)
from dynamo.profiler.utils.replay_optimize.constants import AIC_BACKEND_VERSIONS

logger = logging.getLogger(__name__)

_MOCKER_AIC_BACKEND_VERSIONS = {
    **AIC_BACKEND_VERSIONS,
    "trtllm": "1.3.0rc10",
}


def _load_latest_database_version() -> Optional[Callable[..., Optional[str]]]:
    try:
        perf_database = importlib.import_module("aiconfigurator_core.sdk.perf_database")
    except ModuleNotFoundError as e:
        if e.name != "aiconfigurator_core":
            raise
        return None
    return perf_database.get_latest_database_version


get_latest_database_version = _load_latest_database_version()

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
    aic_perf_model: Optional[AICPerfModelSpec] = None,
    resolved_backend: Optional[str] = None,
) -> Any:
    """Apply Dynamo features to the picked DGD config via composable layers.

    1. **TRT-LLM runtime defaults** — enable chunked prefill on generated
       TRT-LLM workers so their token budget may be smaller than the request ISL.
    2. **Mocker** — swap the base to the mocker DGD template if enabled.
    3. **vLLM self-benchmark** — when the resolved backend is vLLM, set
       ``DYN_BENCHMARK_MODE`` on each worker so the ``get_perf_metrics``
       endpoint is populated at runtime. The planner consumes this as
       priority 1 of its bootstrap chain, superseding AIC and files.
    4. **Planner** — inject the Planner service + planner-config ConfigMap.
       When ``aic_perf_model`` is given, it is embedded so the planner can
       initialize its direct AIC core model with native identity. When
       ``aic_spec`` is given (rapid mode), it is embedded so the planner can
       run AIC interpolation at bootstrap if the endpoint is unavailable.
    5. **Profile data** — attach interpolation-data ConfigMap when mocker
       or planner-thorough is enabled. The ConfigMap is only emitted when
       the picked config is disaggregated AND the interpolation NPZ files
       were produced on disk; rapid-mode deployments never emit it (the
       planner uses AIC in-process or ``get_perf_metrics`` instead), and
       agg picks skip interpolation entirely.
    """
    if not dgd_config:
        return dgd_config

    mocker = is_mocker_enabled(dgdr)
    planner = is_planner_enabled(dgdr)
    profile = needs_profile_data(dgdr)

    if not mocker and resolved_backend == "trtllm":
        enable_trtllm_chunked_prefill(dgd_config)

    if not mocker and not planner:
        apply_runtime_version_override(dgdr, dgd_config)
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
        planner_cfg = dgdr.features.planner if dgdr.features else None
        if planner_cfg is not None:
            enable_planner_worker_scaling_adapters(base, planner_cfg)
        planner_cm = add_planner_to_config(
            dgdr,
            base,
            best_prefill_mapping=best_prefill_config,
            best_decode_mapping=best_decode_config,
            aic_spec=aic_spec,
            aic_perf_model=aic_perf_model,
        )
        config_maps.append(planner_cm)

    if profile:
        output_dir = ops.output_dir if not ops.dry_run else None
        profile_cm = add_profile_data_to_config(base, output_dir, mocker_enabled=mocker)
        if profile_cm:
            config_maps.append(profile_cm)

    apply_runtime_version_override(dgdr, base)
    if config_maps:
        return config_maps + [base]
    return base


def apply_runtime_version_override(dgdr, config_dict: dict) -> None:
    """Apply the DGDR runtime version to every generated DGD component."""
    override = dgdr.runtimeVersionOverride
    if not override:
        return

    components = config_dict.get("spec", {}).get("components", [])
    if not isinstance(components, list):
        return
    for component in components:
        if isinstance(component, dict):
            component["runtimeVersionOverride"] = override


_VLLM_BENCHMARK_MODE_BY_COMPONENT_TYPE = {
    "prefill": "prefill",
    "decode": "decode",
    "worker": "agg",
}


def enable_vllm_benchmark_mode(config_dict: dict) -> None:
    """Set ``DYN_BENCHMARK_MODE`` on every typed vLLM worker.

    The caller invokes this only for vLLM deployments. Worker roles are resolved
    from the v1beta1 ``spec.components[].type`` field, so custom and legacy
    component names receive the same benchmark mode as canonical names.

    Mutates ``config_dict`` in place. The operation is idempotent: an existing
    ``DYN_BENCHMARK_MODE`` entry is replaced with the role-correct value.
    """
    components = config_dict.get("spec", {}).get("components", [])
    if not isinstance(components, list):
        return

    for component in components:
        if not isinstance(component, dict):
            continue
        mode = _VLLM_BENCHMARK_MODE_BY_COMPONENT_TYPE.get(component.get("type"))
        if mode is None:
            continue
        main_container = get_main_container_dict(component)
        if main_container is None:
            continue
        env_list = main_container.get("env") or []
        main_container["env"] = env_list
        # Strip any existing DYN_BENCHMARK_MODE; append the type-derived value.
        env_list[:] = [
            entry
            for entry in env_list
            if not (
                isinstance(entry, dict) and entry.get("name") == "DYN_BENCHMARK_MODE"
            )
        ]
        env_list.append({"name": "DYN_BENCHMARK_MODE", "value": mode})
        logger.info(
            "Enabled vLLM self-benchmark on component %s (DYN_BENCHMARK_MODE=%s)",
            component.get("name", "<unnamed>"),
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
    mocker_config = load_dgd_template("mocker", "disagg")

    image = dgdr.image
    if image:
        components = mocker_config.get("spec", {}).get("components", [])
        for component in components:
            if not isinstance(component, dict):
                continue
            main_container = get_main_container_dict(component)
            if main_container is not None:
                main_container["image"] = image

    model = dgdr.model
    aic_workers = _mocker_aic_worker_picks(aic_spec)
    for worker_name in _mocker_worker_names():
        component = get_component_dict(mocker_config, worker_name)
        if component:
            main_container = get_main_container_dict(component)
            if main_container is None:
                continue
            args_list = main_container.get("args", [])
            args_list = set_argument_value(args_list, "--model-path", model)
            args_list = set_argument_value(args_list, "--model-name", model)
            pick = aic_workers.get(worker_name) if aic_workers else None
            if pick is not None and aic_spec is not None:
                args_list = _inject_mocker_aic_args(args_list, aic_spec, pick)
            main_container["args"] = args_list

    return mocker_config


def enable_planner_worker_scaling_adapters(
    config_dict: dict, planner_config: PlannerConfig
) -> None:
    """Opt worker components into DGDSA when Planner manages replicas."""
    if planner_config.advisory:
        return

    components = config_dict.get("spec", {}).get("components", [])
    if not isinstance(components, list):
        return

    target_subcomponents = _planner_scaling_subcomponents(planner_config.mode)
    untyped_worker_count = sum(
        1
        for component in components
        if isinstance(component, dict)
        and component.get("type") == "worker"
        and _infer_subcomponent_from_component_name(component.get("name", "")) is None
    )

    for component in components:
        if not isinstance(component, dict):
            continue
        if not _is_planner_scalable_worker_component(
            component,
            target_subcomponents,
            planner_config.mode,
            untyped_worker_count,
        ):
            continue
        scaling_adapter = component.setdefault("scalingAdapter", {})
        if not isinstance(scaling_adapter, dict):
            component["scalingAdapter"] = {"enabled": True}
            continue
        scaling_adapter["enabled"] = True


def _is_planner_scalable_worker_component(
    component: dict,
    target_subcomponents: set[str],
    planner_mode: str,
    untyped_worker_count: int,
) -> bool:
    component_type = component.get("type")
    if component_type in target_subcomponents:
        return True
    if component_type != "worker":
        return False

    inferred_type = _infer_subcomponent_from_component_name(component.get("name", ""))
    if inferred_type is not None:
        if inferred_type in target_subcomponents:
            component["type"] = inferred_type
            return True
        return False

    if planner_mode == "agg" and untyped_worker_count == 1:
        component["type"] = "decode"
        return True

    return False


def _planner_scaling_subcomponents(planner_mode: str) -> set[str]:
    if planner_mode == "prefill":
        return {"prefill"}
    if planner_mode in {"decode", "agg"}:
        return {"decode"}
    if planner_mode == "disagg":
        return {"prefill", "decode"}
    return set()


def _infer_subcomponent_from_component_name(component_name: str) -> Optional[str]:
    normalized = component_name.lower()
    if "prefill" in normalized:
        return "prefill"
    if "decode" in normalized:
        return "decode"
    return None


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
    backend_version = _MOCKER_AIC_BACKEND_VERSIONS.get(aic_spec.backend)
    if backend_version is not None:
        args_list = set_argument_value(
            args_list, "--aic-backend-version", backend_version
        )
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
    aic_perf_model: Optional[AICPerfModelSpec] = None,
) -> dict:
    """Add a Planner component and its planner-config ConfigMap to *config_dict*.

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
        aic_perf_model: Native AIC forward-pass perf model identity for
            real-time Planner engine queries.

    Returns:
        The ``planner_config_cm`` ConfigMap dict.
    """
    planner_cfg = _build_planner_config(
        dgdr,
        best_prefill_mapping,
        best_decode_mapping,
        aic_spec,
        aic_perf_model,
    )
    planner_cfg.profile_results_dir = PROFILE_DATA_MOUNT

    planner_component = DgdPlannerComponentConfig()
    if dgdr.image:
        get_main_container(planner_component).image = derive_planner_image(dgdr.image)

    planner_dict = planner_component.model_dump(exclude_unset=False)

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

    # --- Mount planner-config ConfigMap into the planner component ---
    pod_spec = planner_dict.setdefault("podTemplate", {}).setdefault("spec", {})
    planner_volumes = pod_spec.get("volumes") or []
    pod_spec["volumes"] = planner_volumes
    mc_dict = get_main_container_dict(planner_dict)
    if mc_dict is None:
        raise ValueError("Generated Planner component has no main container")
    mc_mounts = mc_dict.get("volumeMounts") or []
    mc_dict["volumeMounts"] = mc_mounts

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

    mc_args = mc_dict.get("args") or []
    mc_dict["args"] = mc_args
    mc_args.extend(["--config", f"{PLANNER_CONFIG_MOUNT}/planner_config.json"])

    components = config_dict["spec"].setdefault("components", [])
    components[:] = [
        component
        for component in components
        if not (isinstance(component, dict) and component.get("name") == "Planner")
    ]
    components.append(planner_dict)

    return planner_config_cm


def add_profile_data_to_config(
    config_dict: dict,
    output_dir: str | None,
    mocker_enabled: bool = False,
) -> Optional[dict]:
    """Create a profile-data ConfigMap and mount it into consumers in *config_dict*.

    Consumers are auto-detected:
    - The **Planner** component (if present) gets the volume mounted.
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

    planner_component = get_component_dict(config_dict, "Planner")
    if planner_component is not None:
        _mount_volume_into_component(
            planner_component, profile_data_cm_name, PROFILE_DATA_MOUNT
        )

    # Mount into mocker workers only when the mocker backend is active.
    # Non-mocker backends (vllm, sglang, trtllm) share the same service
    # names ("prefill", "decode") but do not accept --planner-profile-data.
    if mocker_enabled:
        for worker_name in _mocker_worker_names():
            worker_component = get_component_dict(config_dict, worker_name)
            if worker_component is not None:
                main_container = get_main_container_dict(worker_component)
                if main_container is None:
                    continue
                args_list = main_container.get("args", [])
                args_list = set_argument_value(
                    args_list, "--planner-profile-data", PROFILE_DATA_MOUNT
                )
                main_container["args"] = args_list
                _mount_volume_into_component(
                    worker_component, profile_data_cm_name, PROFILE_DATA_MOUNT
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


def _mount_volume_into_component(
    component: dict, cm_name: str, mount_path: str
) -> None:
    """Add a ConfigMap volume and main-container mount to a component."""
    pod_spec = component.setdefault("podTemplate", {}).setdefault("spec", {})
    volumes = pod_spec.get("volumes") or []
    pod_spec["volumes"] = volumes
    volumes.append(
        {
            "name": cm_name,
            "configMap": {"name": cm_name},
        }
    )
    main_container = get_main_container_dict(component)
    if main_container is None:
        raise ValueError(f"Component {component.get('name')!r} has no main container")
    volume_mounts = main_container.get("volumeMounts") or []
    main_container["volumeMounts"] = volume_mounts
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
    aic_perf_model: Optional[AICPerfModelSpec] = None,
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
    if aic_perf_model is not None:
        planner_cfg.aic_perf_model = aic_perf_model

    # Propagate SLA targets from spec.sla so the post-deployment planner enforces
    # the same SLA used at sweep time. Without this, the planner silently uses
    # SLAPlannerDefaults ttft_ms=500 / itl_ms=50.
    #
    # Gate on model_fields_set: run_profile() calls valid_dgdr_spec() first, which
    # injects a defaulted SLASpec() (ttft=2000, itl=30) when spec.sla is omitted.
    # Only values the user explicitly set are in model_fields_set, so a defaulted
    # SLASpec falls through and keeps the prior planner defaults.
    #
    # Explicit user overrides on features.planner.{ttft_ms, itl_ms} take precedence.

    sla = dgdr.sla
    if (
        sla is not None
        and sla.e2eLatency is None
        and ("ttft" in sla.model_fields_set or "itl" in sla.model_fields_set)
    ):
        explicit = (
            dgdr.features.planner.model_fields_set
            if dgdr.features and dgdr.features.planner
            else set()
        )
        if "ttft_ms" not in explicit:
            planner_cfg.ttft_ms = float(sla.ttft)
        if "itl_ms" not in explicit:
            planner_cfg.itl_ms = float(sla.itl)

    return planner_cfg


def build_aic_perf_model_spec(
    dgdr,
    best_prefill_pick: Optional[PickedParallelConfig],
    best_decode_pick: Optional[PickedParallelConfig],
    resolved_backend: str,
    system: str,
) -> Optional[AICPerfModelSpec]:
    """Build native AIC identity for the Planner's AIC core integration.

    This is intentionally independent from AIC interpolation. It does not
    request a sweep; it only gives the Planner enough identity and parallelism
    data to try native forward-pass estimation before falling back to
    observed-FPM regression.
    """
    planner = (
        dgdr.features.planner  # type: ignore[union-attr]
        if dgdr.features is not None and dgdr.features.planner is not None
        else None
    )
    if (
        not is_planner_enabled(dgdr)
        or planner is None
        or planner.optimization_target != "sla"
    ):
        return None
    if resolved_backend not in ("trtllm", "vllm", "sglang"):
        return None

    mode = planner.mode
    if mode in ("prefill", "disagg") and best_prefill_pick is None:
        return None
    if mode in ("decode", "agg", "disagg") and best_decode_pick is None:
        return None

    if get_latest_database_version is None:
        logger.warning(
            "aiconfigurator-core is unavailable; Planner will use FPM regression "
            "instead of native AIC estimates."
        )
        return None

    backend_version = get_latest_database_version(
        system=system,
        backend=resolved_backend,
    )
    if backend_version is None:
        logger.warning(
            "No AIC performance database is available for system=%s, backend=%s; "
            "Planner will use FPM regression instead of native AIC estimates.",
            system,
            resolved_backend,
        )
        return None

    return AICPerfModelSpec(
        hf_id=dgdr.model,
        system=system,
        backend=resolved_backend,
        backend_version=backend_version,
        prefill_pick=best_prefill_pick,
        decode_pick=best_decode_pick,
    )


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

    * neither a throughput-scaling Planner with a rapid sweep nor a mocker in
      rapid mode needs it
    * picks are missing
    * ``resolved_backend`` is not one AIC supports

    .. note::
        The spec only carries ``prefill_pick`` + ``decode_pick``, so the
        caller in ``profile_sla.py`` gates this on a disaggregated pick
        (``is_disagg_config``). When rapid AIC picks an aggregated config
        and the override to disagg fails, ``aic_spec`` is ``None`` and the
        planner has no AIC fallback — it relies solely on the
        ``get_perf_metrics`` endpoint (``DYN_BENCHMARK_MODE``).

        TODO: extend ``AICInterpolationSpec`` with an ``agg_pick`` so
        throughput-scaling on an aggregated deployment has a matching
        AIC bootstrap path (planner + mocker + thorough NPZ). Tracking
        via the wider agg+throughput-scaling rework.
    """
    planner = (
        dgdr.features.planner  # type: ignore[union-attr]
        if dgdr.features is not None and dgdr.features.planner is not None
        else None
    )
    mocker_needs_aic = needs_mocker_aic_perf_model(dgdr)
    planner_needs_aic = (
        is_planner_enabled(dgdr)
        and planner is not None
        and planner.enable_throughput_scaling
        and planner.pre_deployment_sweeping_mode == PlannerPreDeploymentSweepMode.Rapid
    )
    if not planner_needs_aic and not mocker_needs_aic:
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
