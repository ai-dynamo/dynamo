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
from aiconfigurator.cli.main import _execute_tasks, build_default_tasks
from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.generator.naive import build_naive_generator_params
from aiconfigurator.sdk.task_v2 import Task

from dynamo.profiler.utils.config import clamp_total_gpus_to_budget
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec
from dynamo.profiler.utils.model_cache_paths import model_cache_path_in_pvc
from dynamo.profiler.utils.profile_common import (
    DEFAULT_BACKEND,
    derive_backend_image,
    needs_mocker_aic_perf_model,
    needs_profile_data,
    resolve_model_path,
)

logger = logging.getLogger(__name__)


def _build_k8s_overrides(
    dgdr: DynamoGraphDeploymentRequestSpec,
    backend: str,
) -> dict:
    """Extract K8s overrides (image, PVC) from a DGDR spec."""
    backend_image = derive_backend_image(dgdr.image, backend)
    # derive_backend_image keeps the registry, tag and digest, so a difference
    # means a sibling repository that may be absent from the registry.
    if backend_image != dgdr.image:
        logger.warning(
            "Backend '%s' needs image '%s', so the generated deployment does "
            "not use the requested image '%s'. Ensure the derived image exists "
            "in the registry, or request a concrete backend.",
            backend,
            backend_image,
            dgdr.image,
        )
    overrides: dict = {
        "k8s_image": backend_image,
    }
    if dgdr.modelCache:
        if dgdr.modelCache.pvcName:
            overrides["k8s_pvc_name"] = dgdr.modelCache.pvcName
        if dgdr.modelCache.pvcMountPath:
            overrides["k8s_pvc_mount_path"] = dgdr.modelCache.pvcMountPath
        if dgdr.modelCache.pvcModelPath:
            overrides["k8s_model_path_in_pvc"] = model_cache_path_in_pvc(
                dgdr.modelCache.pvcMountPath,
                dgdr.modelCache.pvcModelPath,
            )
    return overrides


def _winning_backend(best_config_df: pd.DataFrame | None) -> str | None:
    """The concrete backend of a result bucket's rank-1 row.

    AIC labels every row of a merged bucket with the backend of the task that
    produced it. Returns ``None`` for an empty or unlabelled bucket, which is
    what a single-backend search produces.
    """
    if best_config_df is None or best_config_df.empty:
        return None
    if "backend" not in best_config_df.columns:
        return None
    return str(best_config_df.iloc[0]["backend"])


def _resolve_task_config(
    row: pd.Series,
    chosen_exp: str,
    task_configs: dict[str, Task],
    row_backend: str | None,
) -> Task | None:
    """Find the task config that produced the winning row.

    Under ``backend='auto'`` AIC runs one task per backend, keyed e.g.
    ``agg_vllm``, then merges their results into the bare mode buckets ``agg``
    and ``disagg``. ``chosen_exp`` is therefore a merged bucket name that is
    absent from ``task_configs``, and the originating key has to be recovered.
    Three sources are tried, most authoritative first:

    1. ``_task_key``, which AIC publishes on every merged row.
    2. ``chosen_exp`` itself, which is the key when one concrete backend was
       requested and no merge happened.
    3. ``f"{chosen_exp}_{row_backend}"``, reconstructing AIC's key scheme for
       dependency versions that do not publish ``_task_key``.

    Returns:
        The task config, or ``None`` if no source resolves one.
    """
    if "_task_key" in row.index and pd.notna(row["_task_key"]):
        tc = task_configs.get(str(row["_task_key"]))
        if tc is not None:
            return tc

    tc = task_configs.get(chosen_exp)
    if tc is not None:
        return tc

    if row_backend is not None:
        return task_configs.get(f"{chosen_exp}_{row_backend}")
    return None


def _generate_dgd_from_pick(
    dgdr: DynamoGraphDeploymentRequestSpec,
    best_config_df: pd.DataFrame,
    chosen_exp: str,
    task_configs: dict[str, Task],
    picking_mode: str = "default",
) -> dict | None:
    """Generate a DGD config dict from the rank-1 picked result via AIC's generator."""
    if best_config_df is None or best_config_df.empty:
        return None

    row = best_config_df.iloc[0]
    row_backend = str(row["backend"]) if "backend" in row.index else None

    tc = _resolve_task_config(row, chosen_exp, task_configs, row_backend)
    if tc is None:
        logger.error(
            "No task config resolves the picked result: experiment='%s', "
            "winning backend=%s, available keys=%s. No deployment generated.",
            chosen_exp,
            row_backend,
            sorted(task_configs),
        )
        return None

    # One backend for the whole generated deployment: the image and the
    # command line are both selected from this local.
    backend = tc.primary_backend_name
    if row_backend is not None and row_backend != backend:
        logger.error(
            "Winning configuration reports backend '%s' but the resolved task "
            "config '%s' is for backend '%s'. Refusing to generate a "
            "deployment that would mix the two.",
            row_backend,
            chosen_exp,
            backend,
        )
        return None

    original_total_gpus = tc.total_gpus
    try:
        if picking_mode == "autoscale":
            # pick_autoscale returns rows with (p)workers=1 / (d)workers=1 by
            # construction; the planner handles runtime scaling. AIC's
            # module_bridge rescales workers by total_gpus // gpus_per_replica
            # whenever total_gpus is truthy, which would override the picker's
            # intent. Zeroing total_gpus here disables that rescale so the
            # picker's workers=1 flows through unchanged.
            tc.total_gpus = 0
        elif "total_gpus_needed" in row.index:
            clamped_total_gpus, was_clamped = clamp_total_gpus_to_budget(
                row["total_gpus_needed"],
                original_total_gpus,
            )
            # Enforce DGDR hardware budget as a hard cap in rapid mode.
            # Some AIC pickers expose total_gpus_needed as a ranking signal rather
            # than a strict feasibility constraint.
            if was_clamped:
                logger.warning(
                    "Picked config requests %d GPUs but DGDR budget is %d; "
                    "clamping generated deployment to budget.",
                    int(row["total_gpus_needed"]),
                    original_total_gpus,
                )
            tc.total_gpus = clamped_total_gpus

        k8s_overrides = _build_k8s_overrides(dgdr, backend)
        logger.info(
            "Generating deployment for experiment='%s' with backend='%s' and "
            "image='%s'.",
            chosen_exp,
            backend,
            k8s_overrides["k8s_image"],
        )
        cfg = task_config_to_generator_config(
            task_config=tc,
            result_df=row,
            generator_overrides={"K8sConfig": k8s_overrides} if k8s_overrides else None,
        )
    finally:
        tc.total_gpus = original_total_gpus

    service_cfg = cfg.get("ServiceConfig")
    if isinstance(service_cfg, dict):
        service_cfg["model_path"] = dgdr.model
        service_cfg["served_model_path"] = dgdr.model

    artifacts = generate_backend_artifacts(
        params=cfg,
        backend=backend,
        backend_version=tc.primary_backend_version,
        use_dynamo_generator=True,
    )
    dgd_yaml = artifacts.get("k8s_deploy.yaml", "")
    if dgd_yaml:
        return yaml.safe_load(dgd_yaml)
    return None


# Fallback backend when AIC simulation is unavailable and no concrete backend is specified.
_DEFAULT_NAIVE_BACKEND = DEFAULT_BACKEND


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
        logger.info("Auto backend resolved to '%s' for naive fallback.", backend)
    logger.info(
        "AIC does not support this combo — falling back to naive config generation."
    )

    sla = dgdr.sla
    if sla is not None and sla.e2eLatency is not None:
        requested_sla = f"e2eLatency={sla.e2eLatency:.1f}ms"
    elif sla is not None and sla.ttft is not None and sla.itl is not None:
        requested_sla = f"ttft={sla.ttft:.1f}ms, itl={sla.itl:.1f}ms"
    else:
        requested_sla = "requested SLA"
    logger.warning(
        "SLA is unverified (%s): no performance estimates are available for "
        "model=%s, system=%s, backend=%s. Naive fallback will generate a default "
        "configuration that may not meet the requested SLA.",
        requested_sla,
        model,
        system,
        backend,
    )

    generator_params = build_naive_generator_params(
        model_name=model,
        total_gpus=total_gpus,
        system_name=system,
        backend_name=backend,
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
    """Build a Task, run autoscale simulation, collect latencies, generate DGD."""
    # TODO(AIC): the autoscale path constructs Task directly; BackendName("auto")
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

    local_or_hf_model = resolve_model_path(dgdr)
    task = Task(
        serving_mode="disagg",
        prefill_model_path=local_or_hf_model,
        decode_model_path=local_or_hf_model,
        prefill_system_name=system,
        decode_system_name=system,
        prefill_backend_name=backend,
        decode_backend_name=backend,
        total_gpus=total_gpus,
        isl=isl,
        osl=osl,
        ttft=target_ttft,
        tpot=target_tpot,
        request_latency=request_latency,
    )
    pareto_df = task.run(autoscale=True)
    best_latencies = {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    if pareto_df is not None and not pareto_df.empty:
        row = pareto_df.iloc[0]
        best_latencies["ttft"] = float(row.get("ttft", 0.0))
        best_latencies["tpot"] = float(row.get("tpot", 0.0))
        best_latencies["request_latency"] = float(row.get("request_latency", 0.0))

    task_configs = {"disagg": task}
    dgd_config = _generate_dgd_from_pick(
        dgdr, pareto_df, "disagg", task_configs, "autoscale"
    )
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
    local_or_hf_model = resolve_model_path(dgdr)
    task_configs = build_default_tasks(
        model_path=local_or_hf_model,
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

    chosen, best_configs, _, _, best_latencies_map, _ = _execute_tasks(
        task_configs,
        mode="default",
        top_n=5,
        **load_kwargs,
    )

    # File-based interpolation and rapid mocker AIC specs both require separate
    # prefill/decode picks. If AIC picked an aggregated config, override to the
    # best available disaggregated alternative for the downstream consumer.
    requires_disagg = needs_profile_data(dgdr) or needs_mocker_aic_perf_model(dgdr)
    if chosen == "agg" and requires_disagg:
        disagg_key = next(
            (k for k in best_configs if "disagg" in k and not best_configs[k].empty),
            None,
        )
        if disagg_key:
            logger.info(
                "AIC picked aggregated config but separate prefill/decode picks "
                "are required — "
                "overriding to '%s' to support mocker/throughput-scaling.",
                disagg_key,
            )
            # Each bucket is merged across backends under backend="auto", so the
            # override can change the deployment's backend, not just its mode.
            summarized_backend = _winning_backend(best_configs.get(chosen))
            override_backend = _winning_backend(best_configs.get(disagg_key))
            if (
                summarized_backend is not None
                and override_backend is not None
                and summarized_backend != override_backend
            ):
                logger.warning(
                    "The override also changes the backend: the summarized "
                    "winner used '%s' but the generated deployment uses '%s', "
                    "so it runs the '%s' runtime image and command line.",
                    summarized_backend,
                    override_backend,
                    override_backend,
                )
            chosen = disagg_key
        else:
            logger.warning(
                "AIC picked aggregated config and no disaggregated alternative "
                "is available; separate prefill/decode performance data will "
                "be unavailable."
            )

    best_config_df = best_configs.get(chosen, pd.DataFrame())
    best_latencies = best_latencies_map.get(
        chosen, {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    )

    # When backend="auto" AIC expands to per-backend task configs; the winning
    # row carries the concrete backend name so downstream consumers (e.g.
    # run_interpolation) can use it without re-encountering "auto". Resolved
    # before generation so both are read from the same row.
    resolved_backend = backend
    if backend == "auto":
        row_backend = _winning_backend(best_config_df)
        if row_backend is not None:
            resolved_backend = row_backend

    dgd_config = _generate_dgd_from_pick(
        dgdr, best_config_df, chosen, task_configs, picking_mode
    )

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
