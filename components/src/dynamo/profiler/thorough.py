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
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec
from dynamo.profiler.utils.profile_common import (
    ProfilerOperationalConfig,
    derive_backend_image,
)
from dynamo.profiler.utils.profile_decode import get_num_request_range

logger = logging.getLogger(__name__)


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
    prefill_candidates, decode_candidates = enumerate_profiling_configs(
        model_path=model,
        system=system,
        backend=backend,
        image=derive_backend_image(dgdr.image, backend),
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

    # --- Stage 2: Benchmarking ---
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

    decode_df = pd.DataFrame(decode_rows) if decode_rows else pd.DataFrame()

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
