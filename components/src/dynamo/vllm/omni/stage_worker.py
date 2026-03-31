# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import logging
import tempfile
from typing import Any, AsyncGenerator

import yaml

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine

logger = logging.getLogger(__name__)


class OmniStageWorker:
    """Single-stage worker. Not model-specific — behaviour driven by stage YAML.

    Uses AsyncOmni for both LLM and diffusion stage types. Single-stage YAML
    means final_stage_id=0 so orchestrator sends output directly to caller.
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
        self.stage_type: str = getattr(stage_config, "stage_type", "llm")
        self.final_output: bool = getattr(stage_config, "final_output", False)
        self.final_output_type: str = getattr(stage_config, "final_output_type", "text")

    async def generate(
        self, request: dict, context
    ) -> AsyncGenerator[dict, None]:
        from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector

        request_id = request.get("request_id") or context.id()
        sampling_params_list = request.get("sampling_params_list")

        if request.get("from_connector"):
            # Router → stage: payload arrives via SHM connector
            prompt, _ = try_recv_via_connector(request, self.connectors, self.stage_id)
        elif "engine_inputs" in request:
            # Router → stage: payload sent inline (small payloads, no connector)
            prompt = request["engine_inputs"]
        else:
            # Frontend → stage directly (single-stage, no router):
            # pass the whole request dict — AsyncOmni's diffusion pipeline
            # extracts prompt text from request["prompt"] internally
            prompt = request

        last_result = None
        try:
            async for chunk in self.engine.generate(
                prompt, request_id, sampling_params_list=sampling_params_list
            ):
                last_result = chunk
                if self.final_output:
                    yield chunk
        except Exception as e:
            logger.error("Stage %d engine error: %s", self.stage_id, e)
            yield {"error": str(e), "finished": True}
            return

        logger.info(
            "Stage %d output type=%s attrs=%s",
            self.stage_id,
            type(last_result).__name__,
            [a for a in dir(last_result) if not a.startswith("_")] if last_result else "None",
        )
        yield {"stage_output": last_result, "finished": True}


async def init_omni_stage(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_event,
) -> None:
    from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
    from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

    stage_id: int = config.stage_id  # type: ignore[assignment]  # dispatch guarantees non-None
    stage_configs = load_stage_configs_from_yaml(config.stage_configs_path)  # type: ignore[arg-type]
    if stage_id >= len(stage_configs):
        raise ValueError(
            f"--stage-id {stage_id} out of range (YAML has {len(stage_configs)} stages)"
        )
    my_config = stage_configs[stage_id]

    stage_type: str = getattr(my_config, "stage_type", "llm")
    model_stage: str = getattr(my_config.engine_args, "model_stage", f"stage{stage_id}")

    # Create endpoint first — matches init_omni() ordering where endpoint is
    # created before engine init to avoid internal ID mismatches in the runtime
    endpoint_name = f"{config.namespace}.{config.component}.{config.endpoint or 'generate'}"
    generate_endpoint = runtime.endpoint(endpoint_name)

    engine = _create_engine(config.model, my_config, stage_type)
    logger.info("Stage %d (%s): engine created", stage_id, model_stage)

    _, connectors = initialize_orchestrator_connectors(config.stage_configs_path)

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        stage_id=stage_id,
    )

    if not getattr(config.engine_args, "data_parallel_rank", 0):
        from dynamo.common.utils.output_modalities import get_output_modalities
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
        )
        logger.info("Stage %d: registered endpoint '%s'", stage_id, endpoint_name)

    try:
        await generate_endpoint.serve_endpoint(worker.generate, graceful_shutdown=True)
    except Exception as e:
        logger.error("Stage %d: endpoint failed: %s", stage_id, e)
        raise


def _create_engine(model: str, stage_config: Any, stage_type: str) -> StageEngine:
    """Create AsyncOmni with a single-stage YAML derived from stage_config.

    AsyncOmni is used for both LLM and diffusion — it is the only vllm_omni engine
    with a consistent async generator generate() interface for both stage types.
    """
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    # Build a minimal single-stage YAML from this stage's config
    # so AsyncOmni initializes with only this stage (final_stage_id=0)
    single_stage_config = {
        "stage_args": [_stage_config_to_dict(stage_config, stage_type)],
        "runtime": {"edges": []},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(single_stage_config, tmp)
        tmp_path = tmp.name

    return AsyncOmni(model=model, stage_configs_path=tmp_path)


def _stage_config_to_dict(stage_config: Any, stage_type: str) -> dict:
    """Convert a vllm_omni stage config dataclass to a plain dict for YAML."""
    from omegaconf import OmegaConf

    engine_args = stage_config.engine_args
    if hasattr(engine_args, "__dict__"):
        engine_args_dict = OmegaConf.to_container(engine_args, resolve=True) if OmegaConf.is_config(engine_args) else dict(vars(engine_args))
    else:
        engine_args_dict = dict(engine_args)

    return {
        "stage_id": 0,  # renumbered as 0 for single-stage YAML
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
