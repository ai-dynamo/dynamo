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

"""RAPID search strategy: AIC simulation + picking + DGD generation."""

import logging

import pandas as pd
import yaml
from aiconfigurator.cli.main import _execute_task_configs, build_default_task_configs
from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.generator.naive import build_naive_generator_params
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    OptimizationType,
)
from dynamo.profiler.utils.optimization_type import (
    OptimizationTypeConfig,
    resolve_optimization_type_config,
)
from dynamo.profiler.utils.profile_common import (
    derive_backend_image,
    needs_profile_data,
)

logger = logging.getLogger(__name__)


def _build_k8s_overrides(
    dgdr: DynamoGraphDeploymentRequestSpec,
    backend: str,
) -> dict:
    """Extract K8s overrides (image, PVC) from a DGDR spec."""
    overrides: dict = {
        "k8s_image": derive_backend_image(dgdr.image, backend),
    }
    if dgdr.modelCache:
        if dgdr.modelCache.pvcName:
            overrides["k8s_pvc_name"] = dgdr.modelCache.pvcName
        if dgdr.modelCache.pvcMountPath:
            overrides["k8s_pvc_mount_path"] = dgdr.modelCache.pvcMountPath
        if dgdr.modelCache.pvcModelPath:
            overrides["k8s_model_path_in_pvc"] = dgdr.modelCache.pvcModelPath
    return overrides


def _generate_dgd_from_pick(
    dgdr: DynamoGraphDeploymentRequestSpec,
    best_config_df: pd.DataFrame,
    chosen_exp: str,
    task_configs: dict[str, TaskConfig],
) -> dict | None:
    """Generate a DGD config dict from the rank-1 picked result via AIC's generator."""
    if best_config_df is None or best_config_df.empty:
        return None

    row = best_config_df.iloc[0]

    tc = task_configs.get(chosen_exp)
    # TODO: temporary workaround — when backend="auto", AIC's
    # merge_experiment_results_by_mode collapses e.g. "agg_vllm" into "agg",
    # but task_configs retains the original keys. Reconstruct the key from
    # the winning row's backend column. Proper fix: AIC should return the
    # original task config key alongside the merged chosen experiment name.
    if tc is None and "backend" in row.index:
        tc = task_configs.get(f"{chosen_exp}_{row['backend']}")
    if tc is None:
        return None

    original_total_gpus = tc.total_gpus
    if "total_gpus_needed" in row.index and row["total_gpus_needed"] > 0:
        tc.total_gpus = int(row["total_gpus_needed"])

    k8s_overrides = _build_k8s_overrides(dgdr, tc.backend_name)
    cfg = task_config_to_generator_config(
        task_config=tc,
        result_df=row,
        generator_overrides={"K8sConfig": k8s_overrides} if k8s_overrides else None,
    )
    tc.total_gpus = original_total_gpus

    artifacts = generate_backend_artifacts(
        params=cfg,
        backend=tc.backend_name,
        backend_version=tc.backend_version,
        use_dynamo_generator=True,
    )
    dgd_yaml = artifacts.get("k8s_deploy.yaml", "")
    if dgd_yaml:
        return yaml.safe_load(dgd_yaml)
    return None


# Fallback backend when AIC simulation is unavailable and no concrete backend is specified.
_DEFAULT_NAIVE_BACKEND = "vllm"


def _run_naive_fallback(
    dgdr: DynamoGraphDeploymentRequestSpec,
    model: str,
    total_gpus: int,
    system: str,
    backend: str,
) -> dict:
    """Handle the AIC-unsupported path via naive config generation.

    When ``optimizationType`` is active the GPU count is computed from model
    weight and VRAM (1.5× rule) via :func:`resolve_optimization_type_config`,
    and the appropriate parallelization strategy (TP/TEP/DEP) and serving
    mode (agg/disagg) are passed to the AIC naive generator.
    """
    if backend == "auto":
        backend = _DEFAULT_NAIVE_BACKEND
        logger.info("Auto backend resolved to '%s' for naive fallback.", backend)
    logger.info(
        "AIC does not support this combo — falling back to naive config generation."
    )

    # When optimizationType is active, use VRAM-based GPU count instead of
    # totalGpus so the naive config fits the model properly.
    effective_gpus = total_gpus
    if dgdr.sla and dgdr.sla.optimizationType is not None:
        from dynamo.profiler.utils.model_info import get_model_info

        try:
            model_info = get_model_info(model)
            opt_config = resolve_optimization_type_config(dgdr, model_info)
            effective_gpus = opt_config.num_gpus
            logger.info(
                "optimizationType=%s: naive fallback using %d GPUs "
                "(target parallelization: %s).",
                dgdr.sla.optimizationType.value,
                effective_gpus,
                opt_config.prefill_mapping.label(),
            )
        except Exception:
            logger.warning(
                "Could not resolve optimization type config for naive fallback; "
                "using totalGpus=%d.",
                total_gpus,
            )

    # Determine serving mode: prefer disagg when we have enough GPUs
    opt_type_value = None
    naive_mode = "agg"
    if dgdr.sla and dgdr.sla.optimizationType is not None:
        opt_type_value = dgdr.sla.optimizationType.value
        if effective_gpus >= 2:
            naive_mode = "disagg"

    generator_params = build_naive_generator_params(
        model_name=model,
        total_gpus=effective_gpus,
        system_name=system,
        backend_name=backend,
        mode=naive_mode,
        optimization_type=opt_type_value,
    )

    k8s_overrides = _build_k8s_overrides(dgdr, backend)
    generator_params.setdefault("K8sConfig", {}).update(k8s_overrides)

    # Generate DGD through the dynamo config modifier (build_dgd_config),
    # which loads the clean base YAML and produces proper command/args arrays.
    artifacts = generate_backend_artifacts(
        params=generator_params,
        backend=backend,
        use_dynamo_generator=True,
    )
    dgd_yaml = artifacts.get("k8s_deploy.yaml", "")
    dgd_config = yaml.safe_load(dgd_yaml) if dgd_yaml else None

    return {
        "best_config_df": pd.DataFrame(),
        "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
        "dgd_config": dgd_config,
        "chosen_exp": "agg",
        "resolved_backend": backend,
    }


def _run_autoscale_sim(
    dgdr: DynamoGraphDeploymentRequestSpec,
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
    target_ttft: float,
    target_tpot: float,
    request_latency: float | None,
) -> dict:
    """Build a TaskConfig, run autoscale simulation, collect latencies, generate DGD."""
    # TODO(AIC): the autoscale path constructs TaskConfig directly; BackendName("auto")
    # is not a valid enum value, so resolve "auto" to a concrete backend here.
    # AIC should add native auto-backend support in the autoscale path.
    if backend == "auto":
        backend = _DEFAULT_NAIVE_BACKEND
        logger.info("Auto backend resolved to '%s' for autoscale simulation.", backend)

    planner_cfg = dgdr.features.planner if dgdr.features else None
    if planner_cfg and planner_cfg.enable_throughput_scaling:
        logger.warning(
            "Throughput-based scaling enabled — only disagg mode is supported."
        )

    task = TaskConfig(
        serving_mode="disagg",
        model_path=model,
        system_name=system,
        backend_name=backend,
        total_gpus=total_gpus,
        isl=isl,
        osl=osl,
        ttft=target_ttft,
        tpot=target_tpot,
        request_latency=request_latency,
    )
    runner = TaskRunner()
    sim_result = runner.run(task, autoscale=True)
    pareto_df = sim_result.get("pareto_df", pd.DataFrame())
    best_latencies = {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    if pareto_df is not None and not pareto_df.empty:
        row = pareto_df.iloc[0]
        best_latencies["ttft"] = float(row.get("ttft", 0.0))
        best_latencies["tpot"] = float(row.get("tpot", 0.0))
        best_latencies["request_latency"] = float(row.get("request_latency", 0.0))

    task_configs = {"disagg": task}
    dgd_config = _generate_dgd_from_pick(dgdr, pareto_df, "disagg", task_configs)
    return {
        "best_config_df": pareto_df,
        "best_latencies": best_latencies,
        "dgd_config": dgd_config,
        "chosen_exp": "disagg",
        "task_configs": task_configs,
        "resolved_backend": backend,
    }


def _run_default_sim(
    dgdr: DynamoGraphDeploymentRequestSpec,
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
    target_ttft: float,
    target_tpot: float,
    request_latency: float | None,
    picking_mode: str,
) -> dict:
    """Build default task_configs, apply load_match kwargs, run simulation, generate DGD."""
    task_configs = build_default_task_configs(
        model_path=model,
        total_gpus=total_gpus,
        system=system,
        backend=backend,
        isl=isl,
        osl=osl,
        ttft=target_ttft,
        tpot=target_tpot,
        request_latency=request_latency,
    )

    load_kwargs: dict = {}
    if picking_mode == "load_match" and dgdr.workload is not None:
        load_kwargs["target_request_rate"] = dgdr.workload.requestRate
        load_kwargs["target_concurrency"] = dgdr.workload.concurrency
        load_kwargs["max_total_gpus"] = total_gpus

    chosen, best_configs, _, _, best_latencies_map = _execute_task_configs(
        task_configs,
        mode="default",
        top_n=5,
        **load_kwargs,
    )

    # When interpolation data is needed (mocker or throughput-scaling), a
    # disaggregated config is required.  If AIC picked an aggregated config,
    # override to the best available disaggregated alternative so that
    # run_interpolation() can run successfully downstream.
    if chosen == "agg" and needs_profile_data(dgdr):
        disagg_key = next(
            (k for k in best_configs if "disagg" in k and not best_configs[k].empty),
            None,
        )
        if disagg_key:
            logger.info(
                "AIC picked aggregated config but interpolation data is required — "
                "overriding to '%s' to support mocker/throughput-scaling.",
                disagg_key,
            )
            chosen = disagg_key
        else:
            logger.warning(
                "AIC picked aggregated config and no disaggregated alternative "
                "is available; interpolation data will be skipped."
            )

    best_config_df = best_configs.get(chosen, pd.DataFrame())
    best_latencies = best_latencies_map.get(
        chosen, {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    )

    dgd_config = _generate_dgd_from_pick(dgdr, best_config_df, chosen, task_configs)

    # When backend="auto" AIC expands to per-backend task configs; the winning
    # row carries the concrete backend name so downstream consumers (e.g.
    # run_interpolation) can use it without re-encountering "auto".
    resolved_backend = backend
    if (
        backend == "auto"
        and not best_config_df.empty
        and "backend" in best_config_df.columns
    ):
        resolved_backend = best_config_df.iloc[0]["backend"]

    return {
        "best_config_df": best_config_df,
        "best_latencies": best_latencies,
        "dgd_config": dgd_config,
        "chosen_exp": chosen,
        "task_configs": task_configs,
        "resolved_backend": resolved_backend,
    }


def _pick_by_optimization_type(
    results_df: pd.DataFrame,
    optimization_type: OptimizationType,
) -> pd.DataFrame:
    """Pick the best row from simulation results based on optimization type.

    For **throughput**: sort by ``tokens/s/gpu`` descending (highest concurrency).
    For **latency**: sort by ``tpot`` ascending (lowest latency).

    Returns a single-row DataFrame (or empty if input is empty).
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    if optimization_type == OptimizationType.Throughput:
        # Prefer highest throughput per GPU
        sort_col = "tokens/s/gpu" if "tokens/s/gpu" in results_df.columns else "tpot"
        ascending = sort_col == "tpot"
    else:
        # Prefer lowest latency
        sort_col = "tpot" if "tpot" in results_df.columns else "tokens/s/gpu"
        ascending = sort_col != "tokens/s/gpu"

    sorted_df = results_df.sort_values(sort_col, ascending=ascending)
    return sorted_df.head(1).reset_index(drop=True)


def _run_optimization_type_sim(
    dgdr: DynamoGraphDeploymentRequestSpec,
    opt_config: OptimizationTypeConfig,
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
) -> dict:
    """Run AIC simulation restricted to the target parallelization for optimizationType.

    This is similar to ``_run_default_sim`` but:
    1. Restricts the AIC task config parallelization lists to only the resolved
       mapping (e.g. TEP=8 → tp_list=[1], moe_ep_list=[8]).
    2. Picks by optimization type (throughput → highest tps/gpu, latency → lowest tpot)
       instead of SLA-based picking.
    """
    if backend == "auto":
        backend = _DEFAULT_NAIVE_BACKEND
        logger.info(
            "Auto backend resolved to '%s' for optimization-type simulation.", backend
        )

    opt_type = dgdr.sla.optimizationType

    # Use very permissive SLA targets — the picking is done by optimization type,
    # not by SLA filtering
    target_ttft = float("inf")
    target_tpot = float("inf")

    task_configs = build_default_task_configs(
        model_path=model,
        total_gpus=total_gpus,
        system=system,
        backend=backend,
        isl=isl,
        osl=osl,
        ttft=target_ttft,
        tpot=target_tpot,
        request_latency=None,
    )

    # Restrict parallelization to the resolved mapping
    # Convert from ParallelizationMapping (tp/tep/dep) to AIC worker config lists
    mapping = opt_config.prefill_mapping
    if mapping.tp is not None:
        aic_tp = mapping.tp
        aic_dp, aic_moe_tp, aic_moe_ep = 1, 1, 1
    elif mapping.tep is not None:
        aic_tp, aic_dp = 1, 1
        aic_moe_tp, aic_moe_ep = mapping.tep, 1
    elif mapping.dep is not None:
        aic_tp, aic_moe_tp = 1, 1
        aic_dp, aic_moe_ep = mapping.dep, mapping.dep
    else:
        aic_tp, aic_dp, aic_moe_tp, aic_moe_ep = 1, 1, 1, 1

    for _key, tc in task_configs.items():
        if not hasattr(tc, "config") or tc.config is None:
            continue
        for worker_cfg_attr in (
            "prefill_worker_config",
            "decode_worker_config",
            "agg_worker_config",
        ):
            worker_cfg = getattr(tc.config, worker_cfg_attr, None)
            if worker_cfg is None:
                continue
            worker_cfg.num_gpu_per_worker = [opt_config.num_gpus]
            worker_cfg.tp_list = [aic_tp]
            worker_cfg.dp_list = [aic_dp]
            if hasattr(worker_cfg, "moe_tp_list"):
                worker_cfg.moe_tp_list = [aic_moe_tp]
            if hasattr(worker_cfg, "moe_ep_list"):
                worker_cfg.moe_ep_list = [aic_moe_ep]
            if hasattr(worker_cfg, "pp_list"):
                worker_cfg.pp_list = [1]

    logger.info(
        "Running optimization-type simulation: type=%s, parallelization=%s, "
        "num_gpus=%d",
        opt_type.value,
        mapping.label(),
        opt_config.num_gpus,
    )

    chosen, best_configs, _, _, best_latencies_map = _execute_task_configs(
        task_configs,
        mode="default",
        top_n=5,
    )

    # Merge all experiment results and pick by optimization type
    all_dfs = [df for df in best_configs.values() if df is not None and not df.empty]
    merged_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    best_config_df = _pick_by_optimization_type(merged_df, opt_type)

    best_latencies = {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    if not best_config_df.empty:
        row = best_config_df.iloc[0]
        best_latencies["ttft"] = float(row.get("ttft", 0.0))
        best_latencies["tpot"] = float(row.get("tpot", 0.0))
        best_latencies["request_latency"] = float(row.get("request_latency", 0.0))

    # Determine the chosen experiment for DGD generation
    if not best_config_df.empty:
        # Prefer disagg when available
        disagg_keys = [k for k in best_configs if "disagg" in k and not best_configs[k].empty]
        chosen = disagg_keys[0] if disagg_keys else chosen

    dgd_config = _generate_dgd_from_pick(dgdr, best_config_df, chosen, task_configs)

    resolved_backend = backend
    if (
        backend == "auto"
        and not best_config_df.empty
        and "backend" in best_config_df.columns
    ):
        resolved_backend = best_config_df.iloc[0]["backend"]

    return {
        "best_config_df": best_config_df,
        "best_latencies": best_latencies,
        "dgd_config": dgd_config,
        "chosen_exp": chosen,
        "task_configs": task_configs,
        "resolved_backend": resolved_backend,
    }


def run_rapid(
    dgdr: DynamoGraphDeploymentRequestSpec,
    picking_mode: str,
    aic_supported: bool,
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
    target_ttft: float,
    target_tpot: float,
    request_latency: float | None,
) -> dict:
    """Run AIC simulation and picking.  Returns a result dict with
    ``best_config_df``, ``best_latencies``, and ``dgd_config``.
    """
    if not aic_supported:
        return _run_naive_fallback(dgdr, model, total_gpus, system, backend)

    # optimizationType mode — restricted enumeration + optimization-type picking
    if dgdr.sla and dgdr.sla.optimizationType is not None:
        from dynamo.profiler.utils.model_info import get_model_info

        model_info = get_model_info(model)
        opt_config = resolve_optimization_type_config(dgdr, model_info)
        return _run_optimization_type_sim(
            dgdr,
            opt_config,
            model,
            system,
            backend,
            total_gpus,
            isl,
            osl,
        )

    if picking_mode == "autoscale":
        return _run_autoscale_sim(
            dgdr,
            model,
            system,
            backend,
            total_gpus,
            isl,
            osl,
            target_ttft,
            target_tpot,
            request_latency,
        )
    return _run_default_sim(
        dgdr,
        model,
        system,
        backend,
        total_gpus,
        isl,
        osl,
        target_ttft,
        target_tpot,
        request_latency,
        picking_mode,
    )
