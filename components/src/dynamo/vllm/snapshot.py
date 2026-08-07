# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
from collections.abc import Callable

from vllm.config import VllmConfig

from dynamo.common.snapshot.constants import SNAPSHOT_FAILOVER_SOURCE_ENV
from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)
from dynamo.common.snapshot.restore_context import (
    parse_snapshot_restore_runtime_config,
    refresh_snapshot_restore_config,
)
from dynamo.common.utils.env import env_bool

from . import envs
from .args import Config
from .constants import DisaggregationMode
from .handlers import VllmEnginePauseController
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)


def validate_snapshot_failover_clone_profile(
    config: Config,
    *,
    shadow: bool,
) -> None:
    engine_args = config.engine_args
    violations = []

    if config.gms_shadow_mode != shadow:
        violations.append(f"gms_shadow_mode must be {shadow}")
    if config.disaggregation_mode != DisaggregationMode.AGGREGATED:
        violations.append("disaggregation_mode must resolve to aggregated")
    if config.request_plane != "tcp":
        violations.append("request_plane must be tcp")
    if engine_args.load_format != "gms":
        violations.append("load_format must be gms")
    for name, value in (
        ("tensor_parallel_size", engine_args.tensor_parallel_size),
        ("pipeline_parallel_size", engine_args.pipeline_parallel_size),
        ("data_parallel_size", engine_args.data_parallel_size),
    ):
        if value != 1:
            violations.append(f"{name} must be 1")

    disabled_config = {
        "embedding_worker": config.embedding_worker,
        "realtime": config.realtime,
        "headless": config.headless,
        "route_to_encoder": config.route_to_encoder,
        "enable_multimodal": config.enable_multimodal,
        "enable_rl": config.enable_rl,
        "benchmark_mode": config.benchmark_mode is not None,
        "fpm_trace": config.fpm_trace,
    }
    violations.extend(
        f"{name} must be disabled"
        for name, enabled in disabled_config.items()
        if enabled
    )

    if engine_args.enable_lora:
        violations.append("enable_lora must be disabled")
    if engine_args.kv_transfer_config is not None:
        violations.append("kv_transfer_config must be disabled")
    if config.use_kv_events or engine_args.kv_events_config is not None:
        violations.append("KV events must be disabled")
    if envs.is_set("DYN_FORWARDPASS_METRIC_PORT"):
        violations.append("forward-pass metric listener must be disabled")

    if violations:
        raise ValueError(
            "automatic snapshot failover clone profile is invalid: "
            + "; ".join(violations)
        )


def validate_snapshot_failover_clone_runner(vllm_config: VllmConfig) -> None:
    if vllm_config.model_config.runner_type != "generate":
        raise ValueError(
            "automatic snapshot failover clone profile is invalid: "
            "resolved runner_type must be generate"
        )


async def refresh_vllm_snapshot_restore_config(
    config: Config,
    argv: list[str] | None,
    *,
    restore_paused: bool,
) -> Config:
    config = await refresh_snapshot_restore_config(
        config,
        lambda: parse_snapshot_restore_runtime_config(argv),
    )
    if restore_paused:
        config.gms_shadow_mode = env_bool("DYN_VLLM_GMS_SHADOW_MODE")
    return config


async def prepare_snapshot_engine(
    config: Config,
    setup_vllm_engine: Callable[[Config], EngineSetupResult],
) -> EngineSnapshotController[EngineSetupResult] | None:
    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return None

    failover_source = os.environ.get(SNAPSHOT_FAILOVER_SOURCE_ENV)
    if failover_source not in (None, "1"):
        raise ValueError(f"{SNAPSHOT_FAILOVER_SOURCE_ENV} must be 1 when set")
    if failover_source == "1":
        validate_snapshot_failover_clone_profile(config, shadow=False)

    if config.headless:
        raise ValueError(
            "--headless is incompatible with snapshot mode "
            "(DYN_SNAPSHOT_CONTROL_DIR is set). "
            "Remove --headless or unset DYN_SNAPSHOT_CONTROL_DIR."
        )

    configure_snapshot_capture_env()
    logger.info("Snapshot mode enabled (watcher-driven signals)")
    config.engine_args.enable_sleep_mode = True

    engine = setup_vllm_engine(config)
    if failover_source == "1":
        validate_snapshot_failover_clone_runner(engine[1])
    gc.collect()
    snapshot_controller = EngineSnapshotController(
        engine=engine,
        pause_controller=VllmEnginePauseController(engine[0]),
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        logger.info("vLLM snapshot captured successfully")
        os._exit(0)

    return snapshot_controller
