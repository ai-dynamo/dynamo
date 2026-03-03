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

"""Interpolation curve generation for planner pre-deployment sweeping."""

import logging
import os

import yaml

from deploy.utils.dynamo_deployment import DynamoDeploymentClient
from dynamo.planner.defaults import SubComponentType
from dynamo.planner.utils.planner_config import PlannerPreDeploymentSweepMode
from dynamo.profiler.utils.config import Config, get_service_name_by_type
from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    PickedParallelConfig,
)
from dynamo.profiler.utils.defaults import EngineType
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec
from dynamo.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from dynamo.profiler.utils.profile_common import ProfilerOperationalConfig
from dynamo.profiler.utils.profile_decode import (
    profile_decode,
    profile_decode_aiconfigurator,
)
from dynamo.profiler.utils.profile_prefill import (
    profile_prefill,
    profile_prefill_aiconfigurator,
)

logger = logging.getLogger(__name__)


async def run_interpolation(
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
            namespace=ops.k8s_namespace,
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
            namespace=ops.k8s_namespace,
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
