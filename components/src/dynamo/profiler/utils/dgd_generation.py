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
    derive_epp_image,
    derive_planner_image,
    is_inference_gateway_enabled,
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

# --- Inference-gateway (GAIE/EPP) injection ------------------------------
# Service key for the injected Endpoint Picker Plugin component.
EPP_SERVICE_NAME = "Epp"
# Annotations the DGD controller reads to emit the HTTPRoute (mirror
# deploy/operator/internal/consts.KubeAnnotationInferenceGatewayName /
# ...InferenceGatewayNamespace). The name gates emission; the namespace
# targets a shared, cross-namespace Gateway (defaults to the DGD namespace).
INFERENCE_GATEWAY_NAME_ANNOTATION = "nvidia.com/inference-gateway-name"
INFERENCE_GATEWAY_NAMESPACE_ANNOTATION = "nvidia.com/inference-gateway-namespace"
# Pod label the EPP label-filter selects on to separate prefill/decode pools.
DYNAMO_SUB_COMPONENT_LABEL = "nvidia.com/dynamo-sub-component-type"
# Each worker runs a direct-mode frontend sidecar so the gateway can route
# straight to it; the EPP (not a standalone frontend) picks the endpoint.
FRONTEND_SIDECAR_ARGS = ["-m", "dynamo.frontend", "--router-mode", "direct"]
# KV-cache block size advertised to the EPP when it cannot be inferred from
# worker args (aggregated vs disaggregated defaults, matching dynamo-gaie).
DEFAULT_EPP_KV_BLOCK_SIZE_AGG = 128
DEFAULT_EPP_KV_BLOCK_SIZE_DISAGG = 16


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
    igw = is_inference_gateway_enabled(dgdr)
    profile = needs_profile_data(dgdr)

    if not mocker and not planner and not igw:
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
    # the get_perf_metrics endpoint is available to the planner. Gated on the
    # planner (its only consumer): mocker workers don't use DYN_BENCHMARK_MODE,
    # and an inference-gateway-only deployment shouldn't benchmark either.
    if planner and not mocker and resolved_backend == "vllm":
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
        )
        config_maps.append(planner_cm)

    if profile:
        output_dir = ops.output_dir if not ops.dry_run else None
        profile_cm = add_profile_data_to_config(base, output_dir, mocker_enabled=mocker)
        if profile_cm:
            config_maps.append(profile_cm)

    # Step 5: front the deployment with the GAIE inference gateway (EPP). This
    # mutates `base` in place (adds the Epp service + frontend sidecars, drops
    # the standalone Frontend) and is what makes `inferenceGateway.enabled` emit
    # the InferencePool + HTTPRoute on the operator side. No ConfigMap is needed
    # here: the EPP config is inlined under eppConfig.config and the operator
    # materializes the ConfigMap itself.
    if igw:
        add_inference_gateway_to_config(dgdr, base)

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


def enable_planner_worker_scaling_adapters(
    config_dict: dict, planner_config: PlannerConfig
) -> None:
    """Opt worker services into DGDSA when non-advisory Planner manages replicas."""
    if planner_config.advisory:
        return

    services = config_dict.get("spec", {}).get("services", {})
    if not isinstance(services, dict):
        return

    target_subcomponents = _planner_scaling_subcomponents(planner_config.mode)
    untyped_worker_count = sum(
        1
        for service_name, service_config in services.items()
        if isinstance(service_config, dict)
        and service_config.get("componentType") == "worker"
        and not service_config.get("subComponentType")
        and _infer_subcomponent_from_service_name(service_name) is None
    )

    for service_name, service_config in services.items():
        if not isinstance(service_config, dict):
            continue
        if not _is_planner_scalable_worker_service(
            service_name,
            service_config,
            target_subcomponents,
            planner_config.mode,
            untyped_worker_count,
        ):
            continue
        scaling_adapter = service_config.setdefault("scalingAdapter", {})
        if not isinstance(scaling_adapter, dict):
            service_config["scalingAdapter"] = {"enabled": True}
            continue
        scaling_adapter["enabled"] = True


def _is_planner_scalable_worker_service(
    service_name: str,
    service_config: dict,
    target_subcomponents: set[str],
    planner_mode: str,
    untyped_worker_count: int,
) -> bool:
    if service_config.get("componentType") != "worker":
        return False

    sub_component_type = service_config.get("subComponentType")
    if sub_component_type:
        return sub_component_type in target_subcomponents

    inferred_type = _infer_subcomponent_from_service_name(service_name)
    if inferred_type is not None:
        if inferred_type in target_subcomponents:
            service_config["subComponentType"] = inferred_type
            return True
        return False

    # Some agg templates have one generic worker name and no subComponentType.
    # Mark it as decode so the Kubernetes planner can rediscover the target.
    if planner_mode == "agg" and untyped_worker_count == 1:
        service_config["subComponentType"] = "decode"
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


def _infer_subcomponent_from_service_name(service_name: str) -> Optional[str]:
    normalized = service_name.lower()
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


# ---------------------------------------------------------------------------
# Inference gateway (GAIE / EPP)
# ---------------------------------------------------------------------------


def _is_disaggregated(services: dict) -> bool:
    """True when any worker service is a prefill worker (disagg topology)."""
    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        if (
            svc.get("subComponentType") == "prefill"
            or svc.get("componentType") == "prefill"
        ):
            return True
    return False


def _worker_service_names(services: dict) -> list[str]:
    """Names of the inference worker services (excludes frontend/planner/epp)."""
    workers = []
    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        ctype = svc.get("componentType")
        if ctype in ("frontend", "planner", "epp"):
            continue
        if ctype == "worker" or svc.get("subComponentType") in (
            "prefill",
            "decode",
            "agg",
        ):
            workers.append(name)
    return workers


def _extract_block_size(services: dict, default: int) -> int:
    """Read ``--block-size`` from any worker's main-container args, else *default*.

    Worker args may be a flat token list or a single shell-string element, so
    both forms are tokenized before scanning.
    """
    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        mc = svc.get("extraPodSpec", {}).get("mainContainer", {})
        tokens: list[str] = []
        for arg in mc.get("args", []) or []:
            tokens.extend(str(arg).split())
        for i, tok in enumerate(tokens):
            if tok == "--block-size" and i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    pass
            if tok.startswith("--block-size="):
                try:
                    return int(tok.split("=", 1)[1])
                except ValueError:
                    pass
    return default


def _routing_profile_router_env(routing_profile) -> list[dict]:
    """Map a ``RoutingProfile`` preset to Dynamo KV-router env knobs.

    The EPP scorer (``dyn-decode-scorer`` / ``dyn-prefill-scorer``) delegates
    endpoint selection to the Dynamo KV router via FFI, and that router reads
    these env vars on the EPP container (see ``lib/bindings/c/src/lib.rs``):

    * ``DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT`` in ``[0, 1]`` (default ``1.0``) —
      credit for device-local prefix-cache overlap; higher packs requests onto
      cache-warm workers.
    * ``DYN_ROUTER_PREFILL_LOAD_SCALE`` >= ``0`` (default ``1.0``) — how
      strongly prefill load penalizes a worker; higher spreads load to cut
      queueing.

    ``balanced`` leaves the router at its defaults (no env emitted). The
    magnitudes here are initial heuristics, not tuned constants.
    """
    profile = getattr(routing_profile, "value", routing_profile)
    if profile == "throughput":
        # Maximize KV-cache reuse and tolerate load -> pack onto warm workers.
        return [
            {"name": "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "value": "1.0"},
            {"name": "DYN_ROUTER_PREFILL_LOAD_SCALE", "value": "0.5"},
        ]
    if profile == "latency":
        # De-emphasize reuse and penalize loaded workers -> spread, cut queueing.
        return [
            {"name": "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT", "value": "0.5"},
            {"name": "DYN_ROUTER_PREFILL_LOAD_SCALE", "value": "2.0"},
        ]
    return []


def _build_epp_endpoint_picker_config(disaggregated: bool) -> dict:
    """Build the EndpointPickerConfig, mirroring the dynamo-gaie presets.

    Aggregated: a single ``decode`` profile whose label-filter allows unlabeled
    pods (no prefill profile, so the decode scorer runs with full KV overlap).
    Disaggregated: separate ``prefill`` and ``decode`` profiles, each filtered
    to its sub-component label and scored by the matching dyn scorer.
    """
    plugins: list[dict] = [
        {"type": "disagg-profile-handler"},
        {
            "name": "decode-filter",
            "type": "label-filter",
            "parameters": {
                "label": DYNAMO_SUB_COMPONENT_LABEL,
                "validValues": ["decode"],
                # No prefill profile in agg, so accept unlabeled pods.
                "allowsNoLabel": not disaggregated,
            },
        },
        {"name": "picker", "type": "max-score-picker"},
        {"name": "dyn-decode", "type": "dyn-decode-scorer"},
    ]
    scheduling_profiles: list[dict] = [
        {
            "name": "decode",
            "plugins": [
                {"pluginRef": "decode-filter", "weight": 1},
                {"pluginRef": "dyn-decode", "weight": 1},
                {"pluginRef": "picker", "weight": 1},
            ],
        }
    ]

    if disaggregated:
        plugins.insert(
            2,
            {
                "name": "prefill-filter",
                "type": "label-filter",
                "parameters": {
                    "label": DYNAMO_SUB_COMPONENT_LABEL,
                    "validValues": ["prefill"],
                    "allowsNoLabel": False,
                },
            },
        )
        plugins.append({"name": "dyn-prefill", "type": "dyn-prefill-scorer"})
        scheduling_profiles.insert(
            0,
            {
                "name": "prefill",
                "plugins": [
                    {"pluginRef": "prefill-filter", "weight": 1},
                    {"pluginRef": "dyn-prefill", "weight": 1},
                    {"pluginRef": "picker", "weight": 1},
                ],
            },
        )

    return {"plugins": plugins, "schedulingProfiles": scheduling_profiles}


def add_inference_gateway_to_config(dgdr, config_dict: dict) -> None:
    """Front the generated DGD with the GAIE inference gateway (EPP), in place.

    Transforms a router-style deployment (standalone Frontend + workers) into
    the GAIE topology the operator expects:

      1. Inject an ``Epp`` service carrying the EndpointPickerConfig (agg vs
         disagg, inferred from worker sub-component types), the EPP image, and
         the routing-profile router knobs.
      2. Give every worker a direct-mode frontend sidecar and drop the
         standalone ``Frontend`` service, so the gateway routes straight to the
         workers and the EPP picks the endpoint.
      3. Set the ``nvidia.com/inference-gateway-name`` annotation when a gateway
         name is supplied -- the signal the DGD controller gates on to emit the
         HTTPRoute binding the model to the EPP InferencePool.

    See ``deploy/operator/internal/dynamo/epp`` for the operator side.
    """
    igw = dgdr.features.inferenceGateway
    spec = config_dict.setdefault("spec", {})
    services = spec.setdefault("services", {})

    disaggregated = _is_disaggregated(services)
    block_size = _extract_block_size(
        services,
        DEFAULT_EPP_KV_BLOCK_SIZE_DISAGG
        if disaggregated
        else DEFAULT_EPP_KV_BLOCK_SIZE_AGG,
    )

    worker_names = _worker_service_names(services)

    # Reuse a worker's image-pull secret for the EPP and the sidecars.
    env_from_secret = None
    for name in worker_names:
        ref = services[name].get("envFromSecret")
        if ref:
            env_from_secret = ref
            break

    # --- 1. Epp service -----------------------------------------------------
    epp_env = [
        {"name": "DYN_KV_CACHE_BLOCK_SIZE", "value": str(block_size)},
        {"name": "DYN_MODEL_NAME", "value": dgdr.model},
        {"name": "DYN_ENFORCE_DISAGG", "value": "true" if disaggregated else "false"},
    ]
    epp_env.extend(_routing_profile_router_env(igw.routingProfile))

    main_container: dict = {"env": epp_env}
    if dgdr.image:
        main_container["image"] = derive_epp_image(dgdr.image)

    epp_service: dict = {
        "componentType": "epp",
        "replicas": 1,
        "extraPodSpec": {"mainContainer": main_container},
        "eppConfig": {"config": _build_epp_endpoint_picker_config(disaggregated)},
    }
    if env_from_secret:
        epp_service["envFromSecret"] = env_from_secret
    services[EPP_SERVICE_NAME] = epp_service

    # --- 2. frontend sidecars + drop the standalone Frontend ----------------
    frontend_image = None
    for name in list(services.keys()):
        svc = services[name]
        if isinstance(svc, dict) and svc.get("componentType") == "frontend":
            frontend_image = (
                svc.get("extraPodSpec", {}).get("mainContainer", {}).get("image")
            )
            del services[name]

    for name in worker_names:
        svc = services[name]
        if "frontendSidecar" in svc:
            continue
        sidecar_image = frontend_image or svc.get("extraPodSpec", {}).get(
            "mainContainer", {}
        ).get("image")
        sidecar: dict = {"args": list(FRONTEND_SIDECAR_ARGS)}
        if sidecar_image:
            sidecar["image"] = sidecar_image
        if svc.get("envFromSecret"):
            sidecar["envFromSecret"] = svc["envFromSecret"]
        svc["frontendSidecar"] = sidecar

    # --- 3. annotation handoff so the operator emits the HTTPRoute ----------
    # The Gateway is shared, pre-provisioned infrastructure: we attach to it by
    # name (and namespace, since it usually lives in its own namespace), never
    # create it.
    if igw.gatewayName:
        metadata = config_dict.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations[INFERENCE_GATEWAY_NAME_ANNOTATION] = igw.gatewayName
        if igw.gatewayNamespace:
            annotations[INFERENCE_GATEWAY_NAMESPACE_ANNOTATION] = igw.gatewayNamespace
        logger.info(
            "Inference gateway enabled: EPP injected (%s), HTTPRoute will bind to "
            "Gateway %r in namespace %r.",
            "disaggregated" if disaggregated else "aggregated",
            igw.gatewayName,
            igw.gatewayNamespace or "<deployment namespace>",
        )
    else:
        logger.warning(
            "inferenceGateway.gatewayName is not set: emitting the EPP InferencePool "
            "but no HTTPRoute. Set gatewayName to attach to the shared Gateway."
        )


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
