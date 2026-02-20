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

import logging
import os
from dataclasses import dataclass

import pandas as pd
import yaml
from aiconfigurator.cli.main import _execute_task_configs, build_default_task_configs
from aiconfigurator.generator.api import (
    generate_backend_artifacts,
    generate_naive_config,
)
from aiconfigurator.generator.enumerate import (
    check_model_hardware_support,
    enumerate_profiling_configs,
)
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk.picking import pick_autoscale, pick_default, pick_load_match
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

from deploy.utils.dynamo_deployment import (
    DynamoDeploymentClient,
    cleanup_remaining_deployments,
)
from dynamo.planner.defaults import SubComponentType
from dynamo.planner.utils.planner_config import PlannerPreDeploymentSweepMode
from dynamo.profiler.utils.aic_dataframe import (
    build_decode_row,
    build_disagg_df_from_static,
    build_prefill_row,
    make_parallel_label,
)
from dynamo.profiler.utils.aiperf import (
    get_decode_itl_and_thpt_per_gpu,
    get_prefill_ttft,
)
from dynamo.profiler.utils.config import Config, get_service_name_by_type
from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    PickedParallelConfig,
)
from dynamo.profiler.utils.defaults import EngineType, SearchStrategy
from dynamo.profiler.utils.dgd_generation import generate_dgd_config_with_planner
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec
from dynamo.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from dynamo.profiler.utils.profile_decode import (
    get_num_request_range,
    profile_decode,
    profile_decode_aiconfigurator,
)
from dynamo.profiler.utils.profile_prefill import (
    profile_prefill,
    profile_prefill_aiconfigurator,
)
from dynamo.profiler.utils.profiler_status import ProfilerStatus, write_profiler_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operational defaults not part of DynamoGraphDeploymentRequestSpec
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "profiling_results"
DEFAULT_NAMESPACE = os.environ.get("DGDR_NAMESPACE", "dynamo-sla-profiler")
DEFAULT_DEPLOYMENT_TIMEOUT = 3600
DEFAULT_PREFILL_INTERPOLATION_GRANULARITY = 16
DEFAULT_DECODE_INTERPOLATION_GRANULARITY = 6
DEFAULT_DRY_RUN = False


@dataclass
class ProfilerOperationalConfig:
    """Operational knobs that are not part of the DGDR spec."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    deployment_timeout: int = DEFAULT_DEPLOYMENT_TIMEOUT
    prefill_interpolation_granularity: int = DEFAULT_PREFILL_INTERPOLATION_GRANULARITY
    decode_interpolation_granularity: int = DEFAULT_DECODE_INTERPOLATION_GRANULARITY
    dry_run: bool = DEFAULT_DRY_RUN


def _picked_config_from_row(prefix: str, row: pd.Series) -> PickedParallelConfig:
    """Extract a PickedParallelConfig from a picked ColumnsDisagg DataFrame row."""
    return PickedParallelConfig(
        tp=int(row.get(f"{prefix}tp", 1)),
        pp=int(row.get(f"{prefix}pp", 1)),
        dp=int(row.get(f"{prefix}dp", 1)),
        moe_tp=int(row.get(f"{prefix}moe_tp", 1)),
        moe_ep=int(row.get(f"{prefix}moe_ep", 1)),
    )


def _is_planner_enabled(dgdr: DynamoGraphDeploymentRequestSpec) -> bool:
    """True when the DGDR spec has a planner config with scaling enabled."""
    return (
        dgdr.features is not None
        and dgdr.features.planner is not None
        and dgdr.features.planner.scaling_enabled()
    )


def _determine_picking_mode(dgdr: DynamoGraphDeploymentRequestSpec) -> str:
    target_load_provided = dgdr.workload is not None and (
        dgdr.workload.requestRate is not None or dgdr.workload.concurrency is not None
    )
    if _is_planner_enabled(dgdr):
        return "autoscale"
    elif target_load_provided:
        return "load_match"
    return "default"


def _warn_and_update_sla(
    best_latencies: dict,
    target_ttft: float,
    target_tpot: float,
) -> tuple[float, float]:
    """Warn if SLA is unachievable; return (possibly updated) targets."""
    achieved_ttft = best_latencies.get("ttft", 0.0)
    achieved_tpot = best_latencies.get("tpot", 0.0)

    if achieved_ttft > target_ttft:
        logger.warning(
            "TTFT SLA %.1fms is unachievable. Best achievable: %.1fms. Updating SLA.",
            target_ttft,
            achieved_ttft,
        )
        target_ttft = achieved_ttft

    if achieved_tpot > target_tpot:
        logger.warning(
            "ITL SLA %.1fms is unachievable. Best achievable: %.1fms. Updating SLA.",
            target_tpot,
            achieved_tpot,
        )
        target_tpot = achieved_tpot

    return target_ttft, target_tpot


def _warn_gpu_shortage(
    picking_mode: str,
    best_latencies: dict,
    total_gpus: int,
) -> None:
    if picking_mode != "load_match":
        return
    gpus_needed = best_latencies.get("total_gpus_needed")
    if gpus_needed is not None and gpus_needed > total_gpus:
        logger.warning(
            "Load target requires %d GPUs but only %d available. "
            "Consider adding more GPUs or reducing the load target.",
            gpus_needed,
            total_gpus,
        )


# ---------------------------------------------------------------------------
# RAPID path
# ---------------------------------------------------------------------------


def _generate_dgd_from_pick(
    dgdr: DynamoGraphDeploymentRequestSpec,
    best_config_df: pd.DataFrame,
    chosen_exp: str,
    task_configs: dict[str, TaskConfig],
) -> dict | None:
    """Generate a DGD config dict from the rank-1 picked result via AIC's generator."""
    if best_config_df is None or best_config_df.empty:
        return None

    tc = task_configs.get(chosen_exp)
    if tc is None:
        return None

    row = best_config_df.iloc[0]

    original_total_gpus = tc.total_gpus
    if "total_gpus_needed" in row.index and row["total_gpus_needed"] > 0:
        tc.total_gpus = int(row["total_gpus_needed"])

    generator_overrides: dict = {}

    k8s_overrides: dict = {}
    if dgdr.image:
        k8s_overrides["k8s_image"] = dgdr.image
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


def _run_rapid(
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
        logger.info(
            "AIC does not support this combo — falling back to naive config generation."
        )
        naive_result = generate_naive_config(model, total_gpus, system, backend)

        # Extract DGD and apply DGDR overrides (image, PVC) that generate_naive_config doesn't handle
        dgd_yaml = naive_result.get("artifacts", {}).get("k8s_deploy.yaml", "")
        dgd_config = yaml.safe_load(dgd_yaml) if dgd_yaml else None
        if dgd_config:
            config_modifier = CONFIG_MODIFIERS[backend]
            if dgdr.image:
                dgd_config = config_modifier.update_image(dgd_config, dgdr.image)
            if dgdr.modelCache and dgdr.modelCache.pvcName:
                dgd_config = config_modifier.update_model_from_pvc(
                    dgd_config,
                    model_name=model,
                    pvc_name=dgdr.modelCache.pvcName,
                    pvc_mount_path=dgdr.modelCache.pvcMountPath or "/opt/model-cache",
                    pvc_path=dgdr.modelCache.pvcModelPath or "",
                )

        return {
            "best_config_df": pd.DataFrame(),
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": dgd_config,
            "chosen_exp": None,
        }

    if picking_mode == "autoscale":
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

    # default or load_match: build both agg + disagg task configs
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


# ---------------------------------------------------------------------------
# THOROUGH path
# ---------------------------------------------------------------------------


async def _run_thorough(
    dgdr: DynamoGraphDeploymentRequestSpec,
    ops: ProfilerOperationalConfig,
    picking_mode: str,
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
    target_ttft: float,
    target_tpot: float,
    request_latency: float | None,
    deployment_clients: list,
) -> dict:
    """Enumerate candidates, deploy + benchmark each, build DataFrames, pick."""
    logger.warning("THOROUGH mode: only disagg configurations are supported.")

    prefill_candidates, decode_candidates = enumerate_profiling_configs(
        model_path=model,
        system=system,
        backend=backend,
        image=dgdr.image,
        isl=isl,
        osl=osl,
        num_gpus_per_node=dgdr.hardware.numGpusPerNode,
        k8s_pvc_name=dgdr.modelCache.pvcName if dgdr.modelCache else None,
        k8s_pvc_mount_path=dgdr.modelCache.pvcMountPath
        if dgdr.modelCache
        else "/workspace/model_cache",
        k8s_model_path_in_pvc=dgdr.modelCache.pvcModelPath if dgdr.modelCache else None,
    )

    logger.info(
        "Enumerated %d prefill candidates, %d decode candidates",
        len(prefill_candidates),
        len(decode_candidates),
    )

    config_modifier = CONFIG_MODIFIERS[backend]

    # --- Benchmark prefill candidates ---
    prefill_rows: list[dict] = []
    for candidate in prefill_candidates:
        num_gpus = candidate.num_gpus
        label = make_parallel_label(
            candidate.tp,
            candidate.pp,
            candidate.dp,
            candidate.moe_tp,
            candidate.moe_ep,
        )
        tag = label.replace("=", "").replace("/", "_")
        work_dir = f"{ops.output_dir}/prefill_{num_gpus}gpus_{tag}"
        os.makedirs(work_dir, exist_ok=True)

        config_fn = f"{work_dir}/config.yaml"
        with open(config_fn, "w") as f:
            yaml.dump(candidate.dgd_config, f)

        model_name, model_path = config_modifier.get_model_name(candidate.dgd_config)
        frontend_port = config_modifier.get_port(candidate.dgd_config)

        logger.info("Profiling prefill candidate %s with %d GPUs...", label, num_gpus)

        client = DynamoDeploymentClient(
            namespace=DEFAULT_NAMESPACE,
            base_log_dir=work_dir,
            model_name=model_name,
            frontend_port=frontend_port,
            deployment_name=candidate.dgd_config["metadata"]["name"],
        )
        deployment_clients.append(client)
        await client.create_deployment(config_fn)
        logger.info("Waiting for prefill deployment to be ready...")
        try:
            await client.wait_for_deployment_ready(timeout=ops.deployment_timeout)
        except TimeoutError:
            logger.error("Prefill deployment %s timed out, skipping", label)
            logger.error("Prefill %s with %d GPUs timed out", label, num_gpus)
            await client.delete_deployment()
            deployment_clients.remove(client)
            continue
        logger.info("Prefill deployment ready")

        await client.get_deployment_logs()

        base_url = client.get_service_url()
        ai_perf_dir = f"{work_dir}/aiperf_isl{isl}"
        ttft = get_prefill_ttft(
            isl,
            ai_perf_dir,
            model_name,
            model_path,
            base_url,
            attention_dp_size=candidate.dp,
        )

        await client.delete_deployment()
        deployment_clients.remove(client)

        if ttft is not None:
            prefill_rows.append(
                build_prefill_row(
                    model=model,
                    isl=isl,
                    osl=osl,
                    ttft=ttft,
                    tp=candidate.tp,
                    pp=candidate.pp,
                    dp=candidate.dp,
                    moe_tp=candidate.moe_tp,
                    moe_ep=candidate.moe_ep,
                    backend=backend,
                    system=system,
                )
            )

    prefill_df = pd.DataFrame(prefill_rows) if prefill_rows else pd.DataFrame()

    # --- Benchmark decode candidates ---
    decode_rows: list[dict] = []
    for candidate in decode_candidates:
        num_gpus = candidate.num_gpus
        label = make_parallel_label(
            candidate.tp,
            candidate.pp,
            candidate.dp,
            candidate.moe_tp,
            candidate.moe_ep,
        )
        tag = label.replace("=", "").replace("/", "_")
        work_dir = f"{ops.output_dir}/decode_{num_gpus}gpus_{tag}"
        os.makedirs(work_dir, exist_ok=True)

        config_fn = f"{work_dir}/config.yaml"
        with open(config_fn, "w") as f:
            yaml.dump(candidate.dgd_config, f)

        model_name, model_path = config_modifier.get_model_name(candidate.dgd_config)
        frontend_port = config_modifier.get_port(candidate.dgd_config)

        logger.info("Profiling decode candidate %s with %d GPUs...", label, num_gpus)

        client = DynamoDeploymentClient(
            namespace=DEFAULT_NAMESPACE,
            base_log_dir=work_dir,
            model_name=model_name,
            frontend_port=frontend_port,
            deployment_name=candidate.dgd_config["metadata"]["name"],
        )
        deployment_clients.append(client)
        await client.create_deployment(config_fn)
        logger.info("Waiting for decode deployment to be ready...")
        try:
            await client.wait_for_deployment_ready(timeout=ops.deployment_timeout)
        except TimeoutError:
            logger.error("Decode deployment %s timed out, skipping", label)
            logger.error("Decode %s with %d GPUs timed out", label, num_gpus)
            await client.delete_deployment()
            deployment_clients.remove(client)
            continue
        logger.info("Decode deployment ready")

        await client.get_deployment_logs()

        decode_cfg = Config.model_validate(candidate.dgd_config)
        decode_service_name = get_service_name_by_type(
            decode_cfg, backend, SubComponentType.DECODE
        ).lower()
        max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
            f"{work_dir}/{client.deployment_name}/{decode_service_name}/0.log",
            attention_dp_size=candidate.dp,
        )
        max_concurrency = max_kv_tokens // (isl + osl)

        sweep_num_request = get_num_request_range(
            candidate.dp,
            max_concurrency,
            ops.decode_interpolation_granularity,
        )
        logger.info("Sweeping num_request: %s", sweep_num_request)

        base_url = client.get_service_url()
        for num_request in sweep_num_request:
            ai_perf_dir = f"{work_dir}/aiperf_request{num_request}_isl{isl}_osl{osl}_n{num_request}"
            itl, thpt_per_gpu = get_decode_itl_and_thpt_per_gpu(
                isl,
                osl,
                num_request,
                ai_perf_dir,
                model_name,
                model_path,
                base_url=base_url,
                num_gpus=num_gpus,
                attention_dp_size=candidate.dp,
            )
            if itl is not None and thpt_per_gpu is not None:
                decode_rows.append(
                    build_decode_row(
                        tpot=itl,
                        thpt_per_gpu=thpt_per_gpu,
                        num_request=num_request,
                        num_gpus=num_gpus,
                        osl=osl,
                        tp=candidate.tp,
                        pp=candidate.pp,
                        dp=candidate.dp,
                        moe_tp=candidate.moe_tp,
                        moe_ep=candidate.moe_ep,
                        backend=backend,
                        system=system,
                    )
                )

        await client.delete_deployment()
        deployment_clients.remove(client)

    decode_df = pd.DataFrame(decode_rows) if decode_rows else pd.DataFrame()

    # --- Picking ---
    if prefill_df.empty:
        error_msg = "No prefill results produced in THOROUGH mode."
        logger.error(error_msg)
        return {
            "best_config_df": pd.DataFrame(),
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": None,
            "chosen_exp": None,
        }
    if decode_df.empty:
        error_msg = "No decode results produced in THOROUGH mode."
        logger.error(error_msg)
        return {
            "best_config_df": pd.DataFrame(),
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": None,
            "chosen_exp": None,
        }

    if picking_mode == "autoscale":
        result = pick_autoscale(prefill_df, decode_df, target_ttft, target_tpot)
    elif picking_mode == "load_match":
        disagg_df = build_disagg_df_from_static(prefill_df, decode_df)
        lm_kwargs: dict = {
            "pareto_df": disagg_df,
            "serving_mode": "disagg",
            "top_n": 5,
        }
        if request_latency is not None:
            lm_kwargs["target_request_latency"] = request_latency
        else:
            lm_kwargs["target_tpot"] = target_tpot

        if dgdr.workload:
            lm_kwargs["target_request_rate"] = dgdr.workload.requestRate
            lm_kwargs["target_concurrency"] = dgdr.workload.concurrency
        if total_gpus:
            lm_kwargs["max_total_gpus"] = total_gpus

        result = pick_load_match(**lm_kwargs)
    else:
        disagg_df = build_disagg_df_from_static(prefill_df, decode_df)
        pk_kwargs: dict = {
            "pareto_df": disagg_df,
            "total_gpus": total_gpus,
            "serving_mode": "disagg",
            "top_n": 5,
        }
        if request_latency is not None:
            pk_kwargs["target_request_latency"] = request_latency
        else:
            pk_kwargs["target_tpot"] = target_tpot

        result = pick_default(**pk_kwargs)

    best_config_df = result.get("best_config_df", pd.DataFrame())

    # Generate DGD via AIC's generator pipeline
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
    dgd_config = _generate_dgd_from_pick(
        dgdr, best_config_df, "disagg", {"disagg": task}
    )

    return {
        "best_config_df": best_config_df,
        "best_latencies": result.get(
            "best_latencies", {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
        ),
        "dgd_config": dgd_config,
        "chosen_exp": "disagg",
    }


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


async def _run_interpolation(
    dgdr: DynamoGraphDeploymentRequestSpec,
    ops: ProfilerOperationalConfig,
    disagg_config: dict,
    best_prefill_config: PickedParallelConfig,
    best_decode_config: PickedParallelConfig,
    model: str,
    system: str,
    backend: str,
    isl: int,
    osl: int,
    sweep_max_context_length: int,
    deployment_clients: list,
):
    """Generate interpolation curves for the planner based on sweep mode.

    Takes the output disagg DGD config and uses ``convert_config`` to strip
    it down to standalone prefill / decode engines for profiling.
    """
    planner_cfg = (
        dgdr.features.planner if (dgdr.features and dgdr.features.planner) else None
    )
    sweep_mode = PlannerPreDeploymentSweepMode.None_
    if planner_cfg and planner_cfg.pre_deployment_sweeping_mode:
        sweep_mode = planner_cfg.pre_deployment_sweeping_mode

    if sweep_mode == PlannerPreDeploymentSweepMode.None_:
        logger.info(
            "Planner pre-deployment sweeping is disabled — skipping interpolation."
        )
        return

    config_modifier = CONFIG_MODIFIERS[backend]
    model_name, model_path = config_modifier.get_model_name(disagg_config)

    best_prefill_gpus = best_prefill_config.num_gpus
    best_decode_gpus = best_decode_config.num_gpus

    # --- Prefill interpolation ---
    prefill_config = config_modifier.convert_config(disagg_config, EngineType.PREFILL)

    work_dir = f"{ops.output_dir}/selected_prefill_interpolation"
    os.makedirs(work_dir, exist_ok=True)
    prefill_config_fn = f"{work_dir}/config.yaml"
    with open(prefill_config_fn, "w") as f:
        yaml.dump(prefill_config, f)

    if sweep_mode == PlannerPreDeploymentSweepMode.Rapid:
        logger.info("Using AIC simulation for prefill interpolation.")
        estimator = AIConfiguratorPerfEstimator(
            hf_id=model,
            system=system.lower(),
            backend=backend,
        )
        profile_prefill_aiconfigurator(
            work_dir,
            best_prefill_gpus,
            sweep_max_context_length,
            ops.prefill_interpolation_granularity,
            estimator,
            tp_size=best_prefill_config.tp_size,
        )
    elif sweep_mode == PlannerPreDeploymentSweepMode.Thorough:
        logger.info("Using real GPUs for prefill interpolation.")
        frontend_port = config_modifier.get_port(prefill_config)
        client = DynamoDeploymentClient(
            namespace=DEFAULT_NAMESPACE,
            base_log_dir=work_dir,
            model_name=model_name,
            frontend_port=frontend_port,
            deployment_name=prefill_config["metadata"]["name"],
        )
        deployment_clients.append(client)
        await client.create_deployment(prefill_config_fn)
        logger.info("Waiting for prefill interpolation deployment...")
        try:
            await client.wait_for_deployment_ready(timeout=ops.deployment_timeout)
        except TimeoutError:
            logger.error("Prefill interpolation deployment timed out, skipping.")
            await client.delete_deployment()
            deployment_clients.remove(client)
            return

        await client.get_deployment_logs()
        base_url = client.get_service_url()

        profile_prefill(
            work_dir,
            model_name,
            model_path,
            base_url,
            best_prefill_gpus,
            sweep_max_context_length,
            ops.prefill_interpolation_granularity,
            attention_dp_size=best_prefill_config.dp,
        )

        await client.delete_deployment()
        deployment_clients.remove(client)

    # --- Decode interpolation ---
    decode_config = config_modifier.convert_config(disagg_config, EngineType.DECODE)

    work_dir = f"{ops.output_dir}/selected_decode_interpolation"
    os.makedirs(work_dir, exist_ok=True)
    decode_config_fn = f"{work_dir}/config.yaml"
    with open(decode_config_fn, "w") as f:
        yaml.dump(decode_config, f)

    if sweep_mode == PlannerPreDeploymentSweepMode.Rapid:
        logger.info("Using AIC simulation for decode interpolation.")
        estimator = AIConfiguratorPerfEstimator(
            hf_id=model,
            system=system.lower(),
            backend=backend,
        )
        attention_dp_size = best_decode_config.dp
        max_kv_tokens = estimator.get_max_kv_tokens(
            isl,
            osl,
            tp_size=best_decode_config.tp_size,
        )
        profile_decode_aiconfigurator(
            work_dir,
            best_decode_gpus,
            max_kv_tokens,
            sweep_max_context_length,
            ops.decode_interpolation_granularity,
            estimator,
            attention_dp_size,
            tp_size=best_decode_config.tp_size,
        )
    elif sweep_mode == PlannerPreDeploymentSweepMode.Thorough:
        logger.info("Using real GPUs for decode interpolation.")
        frontend_port = config_modifier.get_port(decode_config)
        client = DynamoDeploymentClient(
            namespace=DEFAULT_NAMESPACE,
            base_log_dir=work_dir,
            model_name=model_name,
            frontend_port=frontend_port,
            deployment_name=decode_config["metadata"]["name"],
        )
        deployment_clients.append(client)
        await client.create_deployment(decode_config_fn)
        logger.info("Waiting for decode interpolation deployment...")
        try:
            await client.wait_for_deployment_ready(timeout=ops.deployment_timeout)
        except TimeoutError:
            logger.error("Decode interpolation deployment timed out, skipping.")
            await client.delete_deployment()
            deployment_clients.remove(client)
            return

        await client.get_deployment_logs()

        attention_dp_size = best_decode_config.dp
        decode_cfg = Config.model_validate(decode_config)
        decode_service_name = get_service_name_by_type(
            decode_cfg, backend, SubComponentType.DECODE
        ).lower()
        max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
            f"{work_dir}/{client.deployment_name}/{decode_service_name}/0.log",
            attention_dp_size=attention_dp_size,
        )
        base_url = client.get_service_url()

        profile_decode(
            work_dir,
            model_name,
            model_path,
            base_url,
            best_decode_gpus,
            max_kv_tokens,
            sweep_max_context_length,
            ops.decode_interpolation_granularity,
            attention_dp_size,
        )

        await client.delete_deployment()
        deployment_clients.remove(client)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_profile(
    dgdr: DynamoGraphDeploymentRequestSpec,
    ops: ProfilerOperationalConfig | None = None,
):
    """Run the profiling pipeline.

    Args:
        dgdr: The DynamoGraphDeploymentRequest spec describing the model,
              hardware, workload, SLA, and feature configuration.
        ops:  Operational knobs (output dir, namespace, granularity, etc.).
              Uses defaults when ``None``.
    """
    if ops is None:
        ops = ProfilerOperationalConfig()

    deployment_clients: list = []

    os.makedirs(ops.output_dir, exist_ok=True)
    write_profiler_status(
        ops.output_dir,
        status=ProfilerStatus.RUNNING,
        message="Profiler job started",
    )

    try:
        model = dgdr.model
        backend = dgdr.backend.value.lower()
        system = dgdr.hardware.gpuSku.lower()
        total_gpus = dgdr.hardware.totalGpus
        isl = dgdr.workload.isl
        osl = dgdr.workload.osl
        target_ttft = dgdr.sla.ttft
        target_tpot = dgdr.sla.itl
        request_latency = dgdr.sla.e2eLatency
        search_strategy = SearchStrategy(dgdr.searchStrategy.value)

        picking_mode = _determine_picking_mode(dgdr)
        logger.info(
            "Profiler config: model=%s, backend=%s, system=%s, total_gpus=%s, "
            "isl=%d, osl=%d, ttft=%.1f, itl=%.1f, strategy=%s, picking=%s",
            model,
            backend,
            system,
            total_gpus,
            isl,
            osl,
            target_ttft,
            target_tpot,
            search_strategy.value,
            picking_mode,
        )

        aic_supported = check_model_hardware_support(model, system, backend)

        if _is_planner_enabled(dgdr) and not aic_supported:
            planner_cfg = dgdr.features.planner
            if planner_cfg.enable_throughput_scaling:
                raise ValueError(
                    "Throughput-based planner scaling requires AIC support, but "
                    f"{model} on {system}/{backend} is not supported by AIC. "
                    "Use a supported model/hardware/backend combination or disable throughput scaling."
                )
            if (
                planner_cfg.pre_deployment_sweeping_mode
                == PlannerPreDeploymentSweepMode.Rapid
            ):
                logger.warning(
                    "Planner pre-deployment sweeping mode is 'rapid' but AIC does not support "
                    "%s on %s/%s. Falling back to 'none' (no pre-deployment sweeping).",
                    model,
                    system,
                    backend,
                )
                planner_cfg.pre_deployment_sweeping_mode = (
                    PlannerPreDeploymentSweepMode.None_
                )

        if search_strategy == SearchStrategy.THOROUGH and backend == "auto":
            raise ValueError(
                "THOROUGH search strategy does not support 'auto' backend. "
                "Please specify a concrete backend (trtllm, vllm, sglang)."
            )

        # ---------------------------------------------------------------
        # Dryrun: skip all deployment / simulation
        # ---------------------------------------------------------------
        if ops.dry_run:
            logger.info("Dry run mode — skipping deployment and benchmarking.")
            min_gpus = (
                dgdr.hardware.minNumGpusPerEngine
                if dgdr.hardware and dgdr.hardware.minNumGpusPerEngine
                else 1
            )
            best_prefill_config = PickedParallelConfig(tp=min_gpus)
            best_decode_config = PickedParallelConfig(tp=min_gpus)
        else:
            # Pick initial config
            if search_strategy == SearchStrategy.RAPID:
                pick_result = _run_rapid(
                    dgdr,
                    picking_mode,
                    aic_supported,
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
            else:
                pick_result = await _run_thorough(
                    dgdr,
                    ops,
                    picking_mode,
                    model,
                    system,
                    backend,
                    total_gpus,
                    isl,
                    osl,
                    target_ttft,
                    target_tpot,
                    request_latency,
                    deployment_clients,
                )

            best_config_df = pick_result["best_config_df"]
            best_latencies = pick_result["best_latencies"]

            # SLA / GPU warnings
            target_ttft, target_tpot = _warn_and_update_sla(
                best_latencies,
                target_ttft,
                target_tpot,
            )
            _warn_gpu_shortage(picking_mode, best_latencies, total_gpus or 0)

            # Extract best prefill/decode mappings from picked result
            if best_config_df is not None and not best_config_df.empty:
                row = best_config_df.iloc[0]
                best_prefill_config = _picked_config_from_row("(p)", row)
                best_decode_config = _picked_config_from_row("(d)", row)
            else:
                min_gpus = (
                    dgdr.hardware.minNumGpusPerEngine
                    if dgdr.hardware and dgdr.hardware.minNumGpusPerEngine
                    else 1
                )
                best_prefill_config = PickedParallelConfig(tp=min_gpus)
                best_decode_config = PickedParallelConfig(tp=min_gpus)

        logger.info(
            "Selected prefill: %s (%d GPUs, tp=%d pp=%d dp=%d moe_tp=%d moe_ep=%d), "
            "decode: %s (%d GPUs, tp=%d pp=%d dp=%d moe_tp=%d moe_ep=%d)",
            best_prefill_config.label(),
            best_prefill_config.num_gpus,
            best_prefill_config.tp,
            best_prefill_config.pp,
            best_prefill_config.dp,
            best_prefill_config.moe_tp,
            best_prefill_config.moe_ep,
            best_decode_config.label(),
            best_decode_config.num_gpus,
            best_decode_config.tp,
            best_decode_config.pp,
            best_decode_config.dp,
            best_decode_config.moe_tp,
            best_decode_config.moe_ep,
        )

        # DGD config produced by the picking step (with correct parallelization + replicas)
        dgd_config = pick_result.get("dgd_config") if not ops.dry_run else None

        # ---------------------------------------------------------------
        # Interpolation curves (must run before final DGD assembly
        # because profiling data is saved into ConfigMaps)
        # ---------------------------------------------------------------
        if not ops.dry_run and _is_planner_enabled(dgdr) and dgd_config:
            sweep_max_context_length = isl * 2

            await _run_interpolation(
                dgdr,
                ops,
                dgd_config,
                best_prefill_config,
                best_decode_config,
                model,
                system,
                backend,
                isl,
                osl,
                sweep_max_context_length,
                deployment_clients,
            )

        # ---------------------------------------------------------------
        # Final DGD assembly
        # ---------------------------------------------------------------
        mocker_enabled = (
            dgdr.features is not None
            and dgdr.features.mocker is not None
            and dgdr.features.mocker.enabled
        )

        if dgd_config and (_is_planner_enabled(dgdr) or mocker_enabled):
            dgd_config_path = f"{ops.output_dir}/picked_dgd_config.yaml"
            with open(dgd_config_path, "w") as f:
                yaml.safe_dump(dgd_config, f, sort_keys=False)

            real_config, mocker_config = generate_dgd_config_with_planner(
                dgdr=dgdr,
                config_path=dgd_config_path,
                output_dir=ops.output_dir if not ops.dry_run else None,
                best_prefill_mapping=best_prefill_config,
                best_decode_mapping=best_decode_config,
            )

            if mocker_enabled:
                logger.info("Mocker enabled — using mocker DGD config.")
                final_config = mocker_config
            else:
                final_config = real_config
        else:
            final_config = dgd_config

        output_file = f"{ops.output_dir}/final_config.yaml"
        if not final_config:
            if ops.dry_run:
                logger.warning("Dry run mode — no DGD config produced (expected).")
                yaml.safe_dump({}, open(output_file, "w"), sort_keys=False)
            else:
                error_msg = "Profiler did not produce a DGD config."
                logger.error(error_msg)
                write_profiler_status(
                    ops.output_dir,
                    status=ProfilerStatus.FAILED,
                    error=error_msg,
                    message=error_msg,
                )
                return
        else:
            with open(output_file, "w") as f:
                if isinstance(final_config, list):
                    yaml.safe_dump_all(final_config, f, sort_keys=False)
                else:
                    yaml.safe_dump(final_config, f, sort_keys=False)
            logger.info("Final DGD config saved to %s", output_file)

        write_profiler_status(
            ops.output_dir,
            status=ProfilerStatus.SUCCESS,
            message="Profiler completed successfully",
            outputs={
                "final_config": "final_config.yaml",
            },
        )

    except Exception as e:
        logger.exception("Profile job failed with error")
        write_profiler_status(
            ops.output_dir,
            status=ProfilerStatus.FAILED,
            error=str(e),
            message=f"Profiler failed with exception: {type(e).__name__}",
        )
        raise
    finally:
        logger.info("Performing final cleanup of any remaining deployments...")
        await cleanup_remaining_deployments(deployment_clients, DEFAULT_NAMESPACE)
        logger.info("Final cleanup completed.")
