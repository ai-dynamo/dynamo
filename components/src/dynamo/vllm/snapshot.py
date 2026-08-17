# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
from collections.abc import Callable

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)

from .args import Config
from .constants import DisaggregationMode
from .handlers import VllmEnginePauseController
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)


async def prepare_snapshot_engine(
    config: Config,
    setup_vllm_engine: Callable[[Config], EngineSetupResult],
) -> EngineSnapshotController[EngineSetupResult] | None:
    snapshot_config = SnapshotConfig.from_env()
    if snapshot_config is None:
        return None

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
    if (
        config.disaggregation_mode == DisaggregationMode.AGGREGATED
        and not config.realtime
        and not config.enable_multimodal
        and not config.route_to_encoder
        and engine[1].model_config.runner_type == "generate"
    ):
        logger.info("Running vLLM snapshot warmup generation")
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3, 4])
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
        )
        async for _ in engine[0].generate(
            prompt,
            sampling_params,
            request_id="dynamo-snapshot-warmup",
        ):
            pass
        logger.info("vLLM snapshot warmup generation completed")
    else:
        logger.info(
            "Skipping vLLM snapshot warmup: only ordinary aggregated "
            "generation workers are supported"
        )
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
