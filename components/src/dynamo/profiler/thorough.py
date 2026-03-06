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

"""THOROUGH search strategy: enumerate candidates, deploy, benchmark, pick."""

import logging
import os

import pandas as pd
import yaml
from aiconfigurator.generator.enumerate import enumerate_profiling_configs
from aiconfigurator.sdk.picking import pick_autoscale, pick_default, pick_load_match
from aiconfigurator.sdk.task import TaskConfig

from deploy.utils.dynamo_deployment import DynamoDeploymentClient
from dynamo.planner.defaults import SubComponentType
from dynamo.profiler.rapid import _generate_dgd_from_pick
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
from dynamo.profiler.utils.config_modifiers.protocol import apply_dgd_overrides
from dynamo.profiler.utils.dgdr_v1beta1_types import (
    DynamoGraphDeploymentRequestSpec,
    ModelCacheSpec,
)
from dynamo.profiler.utils.model_info import ModelInfo, get_model_info
from dynamo.profiler.utils.profile_common import (
    ProfilerOperationalConfig,
    derive_backend_image,
    get_profiling_job_tolerations,
    inject_tolerations_into_dgd,
)
from dynamo.profiler.utils.profile_decode import get_num_request_range

logger = logging.getLogger(__name__)

# Fraction of GPU VRAM assumed usable for model weights (remainder is KV cache
# and activation workspace).
_MODEL_GPU_MEM_FRAC = 0.9


def _estimate_dense_size_mib(model_info: ModelInfo) -> float:
    """Estimate the size of dense (non-expert) weights in MiB.

    For non-MoE models, returns the full model size.
    For MoE models, estimates only the attention projection and embedding
    weights that are replicated per TP rank and are *not* distributed by EP.
    The expert FFN weights are excluded because EP shards them.

    The estimate uses the formula::

        dense_params = num_hidden_layers × 4 × hidden_size²   (attention)
                     + 2 × vocab_size × hidden_size            (embeddings + LM head)

    This is intentionally conservative (may under-count some dense weights
    such as shared experts and router networks) so that infeasible candidates
    are pruned while over-large TP requirements are not imposed.

    Returns:
        Estimated dense weight size in MiB (model_size units).
    """
    if not model_info.is_moe or model_info.num_experts is None:
        return model_info.model_size

    hidden_size = model_info.hidden_size
    num_layers = model_info.num_hidden_layers
    vocab_size = model_info.vocab_size
    num_experts = model_info.num_experts
    intermediate_size = model_info.intermediate_size

    if not (hidden_size and num_layers):
        # Insufficient architecture info — fall back to full model size.
        logger.debug(
            "Missing hidden_size or num_hidden_layers; using full model size as "
            "dense-layer estimate (conservative)."
        )
        return model_info.model_size

    # Attention + embedding dense params.
    attention_params = num_layers * 4 * hidden_size * hidden_size
    embedding_params = 2 * (vocab_size or 0) * hidden_size
    dense_params = attention_params + embedding_params

    # Expert FFN params (sharded by EP).
    if intermediate_size and num_experts > 0:
        expert_params = num_layers * num_experts * 3 * hidden_size * intermediate_size
    else:
        expert_params = 0

    total_estimated_params = dense_params + expert_params
    if total_estimated_params == 0:
        return model_info.model_size

    dense_fraction = dense_params / total_estimated_params
    return model_info.model_size * dense_fraction


def _prune_infeasible_candidates(
    candidates: list,
    model_info: ModelInfo,
    vram_mib: float,
    label: str = "candidate",
) -> list:
    """Remove candidates whose estimated per-GPU memory exceeds available VRAM.

    For MoE models, expert weights are distributed via EP across ``moe_ep``
    GPUs, but dense weights (attention, embeddings) are *not* — they must fit
    within ``vram_mib / tp``.  Candidates that cannot satisfy the memory
    budget are logged and dropped, preventing guaranteed OOM deployments.

    For non-MoE models all weights are TP-sharded uniformly.

    Args:
        candidates: List of profiling candidate objects (each with ``.tp``
            and ``.moe_ep`` attributes).
        model_info: Model architecture and weight-size metadata.
        vram_mib: VRAM per GPU in MiB.
        label: Human-readable name used in log messages (e.g. "prefill").

    Returns:
        Filtered list containing only memory-feasible candidates.
    """
    if vram_mib <= 0:
        return candidates

    dense_size_mib = _estimate_dense_size_mib(model_info)
    expert_size_mib = max(0.0, model_info.model_size - dense_size_mib)
    usable_vram_mib = vram_mib * _MODEL_GPU_MEM_FRAC

    feasible: list = []
    pruned: list = []
    for candidate in candidates:
        tp = candidate.tp
        ep = getattr(candidate, "moe_ep", 1) or 1

        if model_info.is_moe:
            # Dense weights: split by TP only (EP does not distribute them).
            # Expert weights: split by TP × EP.
            per_gpu_mib = dense_size_mib / tp + expert_size_mib / (tp * ep)
        else:
            per_gpu_mib = model_info.model_size / tp

        if per_gpu_mib <= usable_vram_mib:
            feasible.append(candidate)
        else:
            pruned.append((candidate, per_gpu_mib))

    if pruned:
        logger.warning(
            "Pruned %d infeasible %s candidate(s): estimated per-GPU memory "
            "exceeds %.1f GiB × %.0f%% = %.1f GiB usable VRAM:",
            len(pruned),
            label,
            vram_mib / 1024,
            _MODEL_GPU_MEM_FRAC * 100,
            usable_vram_mib / 1024,
        )
        for candidate, per_gpu_mib in pruned:
            logger.warning(
                "  Pruned %s tp=%d ep=%d: ~%.1f GiB estimated > %.1f GiB usable",
                label,
                candidate.tp,
                getattr(candidate, "moe_ep", 1) or 1,
                per_gpu_mib / 1024,
                usable_vram_mib / 1024,
            )

    return feasible


async def _benchmark_prefill_candidates(
    prefill_candidates,
    ops: ProfilerOperationalConfig,
    isl: int,
    osl: int,
    model: str,
    system: str,
    backend: str,
    deployment_clients: list,
    config_modifier,
) -> pd.DataFrame:
    """Deploy each prefill candidate, measure TTFT, return prefill_df."""
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
            namespace=ops.k8s_namespace,
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

    return pd.DataFrame(prefill_rows) if prefill_rows else pd.DataFrame()


async def _benchmark_decode_candidates(
    decode_candidates,
    ops: ProfilerOperationalConfig,
    isl: int,
    osl: int,
    model: str,
    system: str,
    backend: str,
    deployment_clients: list,
    config_modifier,
) -> pd.DataFrame:
    """Deploy each decode candidate, sweep num_request, return decode_df."""
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
            namespace=ops.k8s_namespace,
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

    return pd.DataFrame(decode_rows) if decode_rows else pd.DataFrame()


def _pick_thorough_best_config(
    prefill_df: pd.DataFrame,
    decode_df: pd.DataFrame,
    picking_mode: str,
    target_ttft: float,
    target_tpot: float,
    request_latency: float | None,
    total_gpus: int,
    dgdr: DynamoGraphDeploymentRequestSpec,
) -> dict:
    """Dispatch to pick_autoscale / pick_load_match / pick_default, return result dict."""
    if picking_mode == "autoscale":
        return pick_autoscale(prefill_df, decode_df, target_ttft, target_tpot)
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
        if dgdr.workload and dgdr.workload.requestRate is not None:
            lm_kwargs["target_request_rate"] = dgdr.workload.requestRate
        if dgdr.workload and dgdr.workload.concurrency is not None:
            lm_kwargs["target_concurrency"] = dgdr.workload.concurrency
        if total_gpus:
            lm_kwargs["max_total_gpus"] = total_gpus
        return pick_load_match(**lm_kwargs)
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
        return pick_default(**pk_kwargs)


async def run_thorough(
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

    # --- Stage 1: Enumeration ---
    model_cache = dgdr.modelCache or ModelCacheSpec()
    prefill_candidates, decode_candidates = enumerate_profiling_configs(
        model_path=model,
        system=system,
        backend=backend,
        image=derive_backend_image(dgdr.image, backend),
        isl=isl,
        osl=osl,
        num_gpus_per_node=dgdr.hardware.numGpusPerNode,
        k8s_pvc_name=model_cache.pvcName,
        k8s_pvc_mount_path=model_cache.pvcMountPath,
        k8s_model_path_in_pvc=model_cache.pvcModelPath,
    )

    logger.info(
        "Enumerated %d prefill candidates, %d decode candidates",
        len(prefill_candidates),
        len(decode_candidates),
    )

    # --- Stage 1.5: Prune candidates that cannot fit in GPU VRAM ---
    # For MoE models with EP, expert weights are sharded across moe_ep GPUs,
    # but dense layers (attention, embeddings) are NOT — they must fit within
    # vram_mib / tp.  Skipping this check wastes cluster time on 1-hour
    # timeouts per OOM candidate.
    vram_mib = (dgdr.hardware and dgdr.hardware.vramMb) or 0.0
    if vram_mib > 0:
        try:
            model_info = get_model_info(model)
            prefill_candidates = _prune_infeasible_candidates(
                prefill_candidates, model_info, vram_mib, "prefill"
            )
            decode_candidates = _prune_infeasible_candidates(
                decode_candidates, model_info, vram_mib, "decode"
            )
            logger.info(
                "After VRAM pruning: %d prefill candidates, %d decode candidates",
                len(prefill_candidates),
                len(decode_candidates),
            )
        except Exception as exc:
            logger.warning(
                "Could not load model info for VRAM feasibility check (%s). "
                "Skipping candidate pruning — infeasible candidates may be deployed.",
                exc,
            )
    else:
        logger.debug(
            "hardware.vramMb not set; skipping per-candidate VRAM feasibility check."
        )

    if dgdr.overrides and dgdr.overrides.dgd:
        for candidate in prefill_candidates:
            candidate.dgd_config = apply_dgd_overrides(
                candidate.dgd_config, dgdr.overrides.dgd
            )
        for candidate in decode_candidates:
            candidate.dgd_config = apply_dgd_overrides(
                candidate.dgd_config, dgdr.overrides.dgd
            )
        logger.info(
            "Applied DGD overrides to %d prefill + %d decode candidates.",
            len(prefill_candidates),
            len(decode_candidates),
        )

    # Propagate profiling-job tolerations to candidate DGDs
    job_tolerations = get_profiling_job_tolerations(dgdr)
    if job_tolerations:
        for candidate in prefill_candidates:
            candidate.dgd_config = inject_tolerations_into_dgd(
                candidate.dgd_config, job_tolerations
            )
        for candidate in decode_candidates:
            candidate.dgd_config = inject_tolerations_into_dgd(
                candidate.dgd_config, job_tolerations
            )
        logger.debug(
            "Propagated %d profiling-job toleration(s) to %d prefill + %d decode candidates.",
            len(job_tolerations),
            len(prefill_candidates),
            len(decode_candidates),
        )

    config_modifier = CONFIG_MODIFIERS[backend]

    # --- Stage 2: Benchmarking ---
    prefill_df = await _benchmark_prefill_candidates(
        prefill_candidates,
        ops,
        isl,
        osl,
        model,
        system,
        backend,
        deployment_clients,
        config_modifier,
    )
    decode_df = await _benchmark_decode_candidates(
        decode_candidates,
        ops,
        isl,
        osl,
        model,
        system,
        backend,
        deployment_clients,
        config_modifier,
    )

    # --- Stage 3: Picking ---
    if prefill_df.empty:
        logger.error("No prefill results produced in THOROUGH mode.")
        return {
            "best_config_df": pd.DataFrame(),
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": None,
            "chosen_exp": None,
        }
    if decode_df.empty:
        logger.error("No decode results produced in THOROUGH mode.")
        return {
            "best_config_df": pd.DataFrame(),
            "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
            "dgd_config": None,
            "chosen_exp": None,
        }

    result = _pick_thorough_best_config(
        prefill_df,
        decode_df,
        picking_mode,
        target_ttft,
        target_tpot,
        request_latency,
        total_gpus,
        dgdr,
    )

    best_config_df = result.get("best_config_df", pd.DataFrame())

    # --- Stage 4: DGD generation ---
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
