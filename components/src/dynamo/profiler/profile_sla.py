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

"""Profiler main entry point."""

import logging
import os
from typing import Any

import yaml
from aiconfigurator.generator.enumerate import check_model_hardware_support
from aiconfigurator.sdk.utils import get_model_config_from_model_path

from deploy.utils.dynamo_deployment import cleanup_remaining_deployments
from dynamo.profiler.interpolation import run_interpolation
from dynamo.profiler.rapid import run_rapid
from dynamo.profiler.thorough import run_thorough
from dynamo.profiler.utils.config_modifiers.parallelization_mapping import (
    PickedParallelConfig,
)
from dynamo.profiler.utils.config_modifiers.protocol import apply_dgd_overrides
from dynamo.profiler.utils.defaults import SearchStrategy
from dynamo.profiler.utils.dgd_generation import generate_dgd_config_with_planner
from dynamo.profiler.utils.dgdr_v1beta1_types import DynamoGraphDeploymentRequestSpec
from dynamo.profiler.utils.dgdr_validate import (
    run_gate_checks,
    validate_dgdr_for_profiler,
)
from dynamo.profiler.utils.profile_common import (
    ProfilerOperationalConfig,
    determine_picking_mode,
    is_planner_enabled,
    picked_config_from_row,
    resolve_model_path,
    warn_and_update_sla,
    warn_gpu_shortage,
)
from dynamo.profiler.utils.profiler_status import ProfilerStatus, write_profiler_status

logger = logging.getLogger(__name__)


def _extract_profiler_params(dgdr: DynamoGraphDeploymentRequestSpec) -> tuple:
    """Pull all profiler parameters from dgdr and log them."""
    model = dgdr.model
    backend = dgdr.backend.value.lower()
    system = dgdr.hardware.gpuSku.lower()
    total_gpus = dgdr.hardware.totalGpus
    isl = dgdr.workload.isl
    osl = dgdr.workload.osl
    request_latency = dgdr.sla.e2eLatency
    if request_latency is not None:
        target_ttft = request_latency
        target_tpot = request_latency
    else:
        target_ttft = dgdr.sla.ttft
        target_tpot = dgdr.sla.itl
    search_strategy = SearchStrategy(dgdr.searchStrategy)
    picking_mode = determine_picking_mode(dgdr)
    logger.info(
        "Profiler config: model=%s, backend=%s, system=%s, total_gpus=%s, "
        "isl=%d, osl=%d, ttft=%.1f, itl=%.1f, e2e_latency=%s, strategy=%s, picking=%s",
        model,
        backend,
        system,
        total_gpus,
        isl,
        osl,
        target_ttft,
        target_tpot,
        request_latency,
        search_strategy.value,
        picking_mode,
    )
    return (
        model,
        backend,
        system,
        total_gpus,
        isl,
        osl,
        request_latency,
        target_ttft,
        target_tpot,
        search_strategy,
        picking_mode,
    )


async def _execute_strategy(
    dgdr: DynamoGraphDeploymentRequestSpec,
    ops: ProfilerOperationalConfig,
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
    deployment_clients: list,
    search_strategy: SearchStrategy,
) -> tuple[dict, PickedParallelConfig, PickedParallelConfig, float, float]:
    """Dispatch dry-run / RAPID / THOROUGH; extract configs; update SLA targets."""
    if ops.dry_run:
        logger.info("Dry run mode — skipping deployment and benchmarking.")
        best_prefill_config = PickedParallelConfig(tp=1)
        best_decode_config = PickedParallelConfig(tp=1)
        pick_result: dict = {}
    else:
        if search_strategy == SearchStrategy.RAPID:
            pick_result = run_rapid(
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
            pick_result = await run_thorough(
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

        target_ttft, target_tpot = warn_and_update_sla(
            best_latencies,
            target_ttft,
            target_tpot,
        )
        warn_gpu_shortage(picking_mode, best_latencies, total_gpus or 0)

        if best_config_df is not None and not best_config_df.empty:
            row = best_config_df.iloc[0]
            best_prefill_config = picked_config_from_row("(p)", row)
            best_decode_config = picked_config_from_row("(d)", row)
        else:
            best_prefill_config = PickedParallelConfig(tp=1)
            best_decode_config = PickedParallelConfig(tp=1)

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
    return (
        pick_result,
        best_prefill_config,
        best_decode_config,
        target_ttft,
        target_tpot,
    )


def _assemble_final_config(
    dgdr: DynamoGraphDeploymentRequestSpec,
    ops: ProfilerOperationalConfig,
    dgd_config: dict | None,
    best_prefill_config: PickedParallelConfig,
    best_decode_config: PickedParallelConfig,
) -> Any:
    """Handle mocker/planner branching and return the final DGD config."""
    mocker_enabled = (
        dgdr.features is not None
        and dgdr.features.mocker is not None
        and dgdr.features.mocker.enabled
    )

    if dgd_config and (is_planner_enabled(dgdr) or mocker_enabled):
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
            return mocker_config
        return real_config
    return dgd_config


def _write_final_output(ops: ProfilerOperationalConfig, final_config: Any) -> bool:
    """Write final_config.yaml and profiler status. Returns False on unrecoverable failure."""
    output_file = f"{ops.output_dir}/final_config.yaml"
    if not final_config:
        if ops.dry_run:
            logger.warning("Dry run mode — no DGD config produced (expected).")
            with open(output_file, "w") as f:
                yaml.safe_dump(None, f, sort_keys=False)
        else:
            error_msg = "Profiler did not produce a DGD config."
            logger.error(error_msg)
            write_profiler_status(
                ops.output_dir,
                status=ProfilerStatus.FAILED,
                error=error_msg,
                message=error_msg,
            )
            return False
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
    return True


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
        # Validate and normalise — after this, required fields are guaranteed non-None
        validate_dgdr_for_profiler(dgdr)

        (
            model,
            backend,
            system,
            total_gpus,
            isl,
            osl,
            request_latency,
            target_ttft,
            target_tpot,
            search_strategy,
            picking_mode,
        ) = _extract_profiler_params(dgdr)

        aic_supported = check_model_hardware_support(model, system, backend)
        run_gate_checks(dgdr, aic_supported, search_strategy, backend)

        (
            pick_result,
            best_prefill_config,
            best_decode_config,
            target_ttft,
            target_tpot,
        ) = await _execute_strategy(
            dgdr,
            ops,
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
            deployment_clients,
            search_strategy,
        )

        dgd_config = pick_result.get("dgd_config") if not ops.dry_run else None

        # ---------------------------------------------------------------
        # Interpolation curves
        # ---------------------------------------------------------------
        if not ops.dry_run and is_planner_enabled(dgdr) and dgd_config:
            try:
                model_cfg = get_model_config_from_model_path(resolve_model_path(dgdr))
                sweep_max_context_length = model_cfg.get("max_position_embeddings", 0)
            except Exception:
                logger.warning("Could not fetch model max context length.")
                sweep_max_context_length = 0
            if not sweep_max_context_length:
                sweep_max_context_length = isl * 2 if isl > 0 else 8192

            await run_interpolation(
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
        final_config = _assemble_final_config(
            dgdr, ops, dgd_config, best_prefill_config, best_decode_config
        )

        # --- Apply DGD overrides (user-supplied partial DGD) ---
        if final_config and dgdr.overrides and dgdr.overrides.dgd:
            if isinstance(final_config, list):
                final_config[-1] = apply_dgd_overrides(
                    final_config[-1], dgdr.overrides.dgd
                )
            elif isinstance(final_config, dict):
                final_config = apply_dgd_overrides(final_config, dgdr.overrides.dgd)
            logger.info("Applied DGD overrides to the final config.")
        if not _write_final_output(ops, final_config):
            return

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
        await cleanup_remaining_deployments(deployment_clients, ops.k8s_namespace)
        logger.info("Final cleanup completed.")
