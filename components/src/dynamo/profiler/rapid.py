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
from aiconfigurator.generator.api import (
    generate_backend_artifacts,
    generate_naive_config,
)
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec
from dynamo.profiler.utils.profile_common import derive_backend_image

logger = logging.getLogger(__name__)


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

    generator_overrides: dict = {}

    k8s_overrides: dict = {}
    k8s_overrides["k8s_image"] = derive_backend_image(dgdr.image, tc.backend_name)
    if dgdr.modelCache:
        if dgdr.modelCache.pvcName:
            k8s_overrides["k8s_pvc_name"] = dgdr.modelCache.pvcName
        if dgdr.modelCache.pvcMountPath:
            k8s_overrides["k8s_pvc_mount_path"] = dgdr.modelCache.pvcMountPath
        if dgdr.modelCache.pvcModelPath:
            k8s_overrides["k8s_model_path_in_pvc"] = dgdr.modelCache.pvcModelPath
    if k8s_overrides:
        generator_overrides["K8sConfig"] = k8s_overrides

    cfg = task_config_to_generator_config(
        task_config=tc,
        result_df=row,
        generator_overrides=generator_overrides or None,
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


# in naive mode, use vllm as the default backend
_DEFAULT_NAIVE_BACKEND = "vllm"


def _run_naive_fallback(
    dgdr: DynamoGraphDeploymentRequestSpec,
    model: str,
    total_gpus: int,
    system: str,
    backend: str,
) -> dict:
    """Handle the AIC-unsupported path via naive config generation."""
    if backend == "auto":
        backend = _DEFAULT_NAIVE_BACKEND
        logger.info(
            "Auto backend resolved to '%s' for naive fallback.",
            backend,
        )
    logger.info(
        "AIC does not support this combo — falling back to naive config generation."
    )
    naive_result = generate_naive_config(model, total_gpus, system, backend)

    dgd_yaml = naive_result.get("artifacts", {}).get("k8s_deploy.yaml", "")
    dgd_config = yaml.safe_load(dgd_yaml) if dgd_yaml else None
    if dgd_config:
        config_modifier = CONFIG_MODIFIERS[backend]
        dgd_config = config_modifier.update_image(
            dgd_config, derive_backend_image(dgdr.image, backend)
        )
        if dgdr.modelCache and dgdr.modelCache.pvcName:
            dgd_config = config_modifier.update_model_from_pvc(
                dgd_config,
                model_name=model,
                pvc_name=dgdr.modelCache.pvcName,
                pvc_mount_path=dgdr.modelCache.pvcMountPath,
                pvc_path=dgdr.modelCache.pvcModelPath or "",
            )

    return {
        "best_config_df": pd.DataFrame(),
        "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
        "dgd_config": dgd_config,
        "chosen_exp": None,
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

    best_config_df = best_configs.get(chosen, pd.DataFrame())
    best_latencies = best_latencies_map.get(
        chosen, {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    )

    dgd_config = _generate_dgd_from_pick(dgdr, best_config_df, chosen, task_configs)

    return {
        "best_config_df": best_config_df,
        "best_latencies": best_latencies,
        "dgd_config": dgd_config,
        "chosen_exp": chosen,
        "task_configs": task_configs,
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
