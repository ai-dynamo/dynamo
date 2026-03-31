# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import logging
import os
import tempfile
from typing import Any, AsyncGenerator

import yaml

from dynamo import prometheus_names
from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine

logger = logging.getLogger(__name__)


class OmniStageWorker:
    """Single-stage worker — runs engine, yields serializable output.

    PIL images can't cross Dynamo's process boundary. For final-output stages,
    images are encoded to raw bytes so the router can reconstruct and format them.
    Formatting (NvVideosResponse, video encoding) stays in the router.
    """

    def __init__(
        self,
        engine: StageEngine,
        stage_config: Any,
        connectors: dict,
        stage_id: int,
    ) -> None:
        self.engine = engine
        self.stage_id = stage_id
        self.connectors = connectors
        self.final_output: bool = getattr(stage_config, "final_output", False)
        self.final_output_type: str = getattr(stage_config, "final_output_type", "text")

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        request_id = request.get("request_id") or context.id()

        # Reconstruct typed objects — AsyncOmni rejects raw dicts
        raw_sp = request.get("sampling_params_list")
        if raw_sp and isinstance(raw_sp[0], dict):
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams

            sampling_params_list = [OmniDiffusionSamplingParams(**sp) for sp in raw_sp]
        else:
            sampling_params_list = raw_sp

        if request.get("from_connector"):
            from vllm_omni.distributed.omni_connectors.adapter import (
                try_recv_via_connector,
            )

            prompt, _ = try_recv_via_connector(request, self.connectors, self.stage_id)
        elif "engine_inputs" in request:
            prompt = request["engine_inputs"]
        else:
            # Direct frontend → stage (single-stage, no router)
            prompt = request

        from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes

        last_result = None
        try:
            async for chunk in self.engine.generate(
                prompt, request_id, sampling_params_list=sampling_params_list
            ):
                last_result = chunk
        except Exception as e:
            logger.error("Stage %d engine error: %s", self.stage_id, e)
            yield {"error": str(e), "finished": True}
            return

        shm_meta = shm_write_bytes(serialize_obj(last_result), name=request_id)
        yield {
            "shm_meta": shm_meta,
            "final_output": self.final_output,
            "finished": True,
        }


async def init_omni_stage(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
) -> None:
    """Initialize a single omni stage worker.

    Mirrors init_omni() setup pattern exactly to avoid routing/handler issues.
    """
    from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
    from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

    from dynamo.common.utils.output_modalities import get_output_modalities
    from dynamo.vllm.health_check import VllmOmniHealthCheckPayload

    stage_id: int = config.stage_id  # type: ignore[assignment]
    stage_configs = load_stage_configs_from_yaml(config.stage_configs_path)  # type: ignore[arg-type]
    if stage_id >= len(stage_configs):
        raise ValueError(
            f"--stage-id {stage_id} out of range (YAML has {len(stage_configs)} stages)"
        )
    my_config = stage_configs[stage_id]
    stage_type: str = getattr(my_config, "stage_type", "llm")

    # Mirror init_omni: create endpoint FIRST
    # Stage worker registers at {ns}.{model_stage}.generate — NOT {ns}.backend.generate.
    # The router registers at {ns}.backend.generate and discovers workers by model_stage.
    # If both used the same endpoint, the frontend would round-robin between them.
    model_stage = getattr(my_config.engine_args, "model_stage", f"stage{stage_id}")
    generate_endpoint = runtime.endpoint(f"{config.namespace}.{model_stage}.generate")
    shutdown_endpoints[:] = [generate_endpoint]

    engine = _create_engine(config.model, my_config, stage_type)
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    _, connectors = initialize_orchestrator_connectors(config.stage_configs_path)  # type: ignore[arg-type]

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        stage_id=stage_id,
    )

    setup_metrics_collection(config, generate_endpoint, logger)

    if not getattr(config.engine_args, "data_parallel_rank", 0):
        model_type = get_output_modalities(config.output_modalities, config.model)
        if model_type is None:
            final_output_type = getattr(my_config, "final_output_type", "text")
            model_type = _resolve_model_type(final_output_type)

        await register_model(
            ModelInput.Text,
            model_type,
            generate_endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=getattr(config.engine_args, "block_size", None),
        )
        logger.info("Stage %d: registered endpoint '%s'", stage_id, generate_endpoint)

        health_check_payload = (
            await VllmOmniHealthCheckPayload.create(engine)  # type: ignore[arg-type]
        ).to_dict()

        try:
            await generate_endpoint.serve_endpoint(
                worker.generate,
                graceful_shutdown=True,
                metrics_labels=[
                    (
                        prometheus_names.labels.MODEL,
                        config.served_model_name or config.model,
                    ),
                    (
                        prometheus_names.labels.MODEL_NAME,
                        config.served_model_name or config.model,
                    ),
                ],
                health_check_payload=health_check_payload,
            )
        except Exception as e:
            logger.error("Stage %d: endpoint failed: %s", stage_id, e)
            raise


def _create_engine(model: str, stage_config: Any, stage_type: str) -> StageEngine:
    """Create AsyncOmni with a single-stage YAML."""
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    single_stage_config = {
        "stage_args": [_stage_config_to_dict(stage_config, stage_type)],
        "runtime": {"edges": []},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(single_stage_config, tmp)
        tmp_path = tmp.name

    try:
        return AsyncOmni(model=model, stage_configs_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def _stage_config_to_dict(stage_config: Any, stage_type: str) -> dict:
    from omegaconf import OmegaConf

    engine_args = stage_config.engine_args
    if OmegaConf.is_config(engine_args):
        engine_args_dict = OmegaConf.to_container(engine_args, resolve=True)
    elif hasattr(engine_args, "__dict__"):
        engine_args_dict = dict(vars(engine_args))
    else:
        engine_args_dict = dict(engine_args)

    return {
        "stage_id": 0,
        "stage_type": stage_type,
        "engine_args": engine_args_dict,
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
