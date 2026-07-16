# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import logging
import uuid
from collections.abc import Callable

from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

from dynamo.common.snapshot.lifecycle import (
    EngineSnapshotController,
    SnapshotConfig,
    configure_snapshot_capture_env,
)

from .args import Config
from .constants import DisaggregationMode
from .handlers import VllmEnginePauseController, get_dp_range_for_worker
from .worker_factory import EngineSetupResult

logger = logging.getLogger(__name__)

_WARMUP_INPUT_IDS = (1, 2, 3)


async def warmup_engine(engine_setup: EngineSetupResult) -> None:
    """Warm the direct vLLM generation path before snapshot capture."""
    engine, vllm_config, *_ = engine_setup
    runner_type = vllm_config.model_config.runner_type
    if runner_type != "generate":
        logger.info("Skipping vLLM snapshot warmup for non-generation model")
        return

    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    _, managed_dp_size = get_dp_range_for_worker(vllm_config)

    async def consume_generation(local_dp_rank: int) -> None:
        async for _ in engine.generate(
            TokensPrompt(prompt_token_ids=list(_WARMUP_INPUT_IDS)),
            sampling_params,
            str(uuid.uuid4()),
            data_parallel_rank=local_dp_rank,
        ):
            pass

    logger.info("vLLM snapshot warmup starting")
    await asyncio.gather(*(consume_generation(rank) for rank in range(managed_dp_size)))
    logger.info("vLLM snapshot warmup complete")


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
    # Embedding and encode workers do not serve generation through this engine.
    if (
        not config.embedding_worker
        and config.disaggregation_mode != DisaggregationMode.ENCODE
    ):
        await warmup_engine(engine)
    gc.collect()
    snapshot_controller = EngineSnapshotController(
        engine=engine,
        pause_controller=VllmEnginePauseController(engine[0]),
        snapshot_config=snapshot_config,
        pause_args=(None,),
    )
    if not await snapshot_controller.wait_for_restore():
        raise SystemExit(0)

    return snapshot_controller
