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
_WARMUP_PREFIX = "dynamo-snapshot-warmup-"
_WARMUP_TIMEOUT_SEC = 600.0


async def warmup_engine(engine_setup: EngineSetupResult) -> None:
    """Warm the direct vLLM generation path before snapshot capture."""
    engine, vllm_config, *_ = engine_setup
    runner_type = vllm_config.model_config.runner_type
    if runner_type != "generate":
        raise ValueError(
            "vLLM Dynamo Snapshot generation warmup requires a generation model; "
            f"got runner_type={runner_type!r}"
        )

    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    _, managed_dp_size = get_dp_range_for_worker(vllm_config)
    request_ids = [f"{_WARMUP_PREFIX}{uuid.uuid4()}" for _ in range(managed_dp_size)]
    cache_salts = [f"{_WARMUP_PREFIX}{uuid.uuid4()}" for _ in range(managed_dp_size)]

    async def consume_generation(
        local_dp_rank: int,
        request_id: str,
        cache_salt: str,
    ) -> None:
        prompt = TokensPrompt(
            prompt_token_ids=list(_WARMUP_INPUT_IDS),
            cache_salt=cache_salt,
        )
        final_output = None
        async for output in engine.generate(
            prompt,
            sampling_params,
            request_id,
            data_parallel_rank=local_dp_rank,
        ):
            final_output = output

        if final_output is None or not final_output.finished:
            raise RuntimeError("vLLM snapshot warmup ended without a final output")
        if not final_output.outputs or not final_output.outputs[0].token_ids:
            raise RuntimeError("vLLM snapshot warmup generated no tokens")

    logger.info("vLLM snapshot warmup starting")
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    consume_generation(rank, request_id, cache_salt)
                    for rank, (request_id, cache_salt) in enumerate(
                        zip(request_ids, cache_salts)
                    )
                ),
                return_exceptions=True,
            ),
            timeout=_WARMUP_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        # Cancellation before vLLM add_request() returns cannot self-abort.
        abort_results = await asyncio.gather(
            *(engine.abort(request_id) for request_id in request_ids),
            return_exceptions=True,
        )
        for request_id, abort_error in zip(request_ids, abort_results):
            if isinstance(abort_error, BaseException):
                logger.error(
                    "Failed to abort timed-out vLLM snapshot warmup request %s: %s",
                    request_id,
                    abort_error,
                )
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
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

    if config.embedding_worker:
        raise ValueError(
            "--embedding-worker is incompatible with snapshot mode because "
            "pooling engines cannot run the required generation warmup"
        )
    if config.disaggregation_mode == DisaggregationMode.ENCODE:
        raise ValueError(
            "vLLM multimodal encode workers are incompatible with snapshot mode "
            "because they do not use the snapshotted generation engine"
        )

    configure_snapshot_capture_env()
    logger.info("Snapshot mode enabled (watcher-driven signals)")
    config.engine_args.enable_sleep_mode = True

    engine = setup_vllm_engine(config)
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
