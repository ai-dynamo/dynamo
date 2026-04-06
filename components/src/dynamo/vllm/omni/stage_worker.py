# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import importlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, AsyncGenerator, cast

import yaml

from dynamo import prometheus_names
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed

logger = logging.getLogger(__name__)


@dataclass
class _Proxy:
    """Satisfies stage_list[i].engine_outputs for processor functions.

    Processor functions (e.g. ar2diffusion) access stage_list[i].engine_outputs
    as a list of OmniRequestOutput objects.
    """

    engine_outputs: Any = None


class OmniStageWorker:
    """Single-stage worker: fetches inputs → runs processor → runs engine → writes output.

    For stage 0: gets engine_inputs directly from request.
    For stage N > 0: fetches previous stage outputs from connectors via stage_connector_refs,
    runs the pre-processor (e.g. thinker2talker) to produce this stage's engine inputs,
    then runs the engine.

    Non-final stages write output to a connector and yield stage_connector_refs for the router.
    Final stages write to SHM and yield shm_meta for the router to format.
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
        self.connectors = connectors  # {(from_stage, to_stage): vllm_omni connector}
        self.final_output: bool = getattr(stage_config, "final_output", False)

        # TODO: use per-request sampling_params_list from request when the router
        # forwards it (see router TODO). Until then, YAML defaults apply for all requests.
        self._default_sp = _build_default_sampling_params(stage_config)

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes

        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()
        original_prompt = req.original_prompt
        # JSON sends dict keys as strings; normalize to int for stage_connector_refs.
        stage_connector_refs = _int_keyed(req.stage_connector_refs)

        # --- Resolve engine inputs ---
        if stage_connector_refs:
            # Stage N > 0: fetch previous stage outputs from connectors, run pre-processor.
            stage_list = self._fetch_stage_inputs(stage_connector_refs, request_id)
            if stage_list is None:
                yield {
                    "error": "Failed to fetch inputs from connectors",
                    "finished": True,
                }
                return

            if self._processor is not None:
                prompt = self._processor(
                    stage_list,
                    self._engine_input_source,
                    [original_prompt],
                    self._requires_mm,
                )
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
            else:
                # No processor: use the most recent fetched stage output directly.
                prompt = stage_list[-1].engine_outputs[0]
        elif req.engine_inputs is not None:
            # Stage 0: engine inputs come directly from the router.
            prompt = req.engine_inputs
        else:
            # Direct frontend → stage (single-stage, no router).
            prompt = request

        logger.debug(
            "Stage %d: engine.generate for %s — prompt type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )

        # --- Run engine ---
        last_result = None
        try:
            async for chunk in self.engine.generate(
                prompt, request_id, sampling_params_list=self._default_sp
            ):
                last_result = chunk
        except Exception as e:
            logger.error(
                "Stage %d engine error for %s: %s",
                self.stage_id,
                request_id,
                e,
                exc_info=True,
            )
            yield {"error": str(e), "finished": True}
            return

        # --- Write output ---
        if not self.final_output:
            from_s, to_s = _connector_key(self.stage_id, self.stage_id + 1)
            connector = self.connectors.get((from_s, to_s))
            if connector is not None:
                try:
                    ok, _, metadata = connector.put(  # type: ignore[arg-type]
                        from_s, to_s, request_id, last_result
                    )
                except Exception as e:
                    logger.error(
                        "Stage %d: connector.put() raised %s: %s",
                        self.stage_id,
                        type(e).__name__,
                        e,
                        exc_info=True,
                    )
                    yield {"error": f"connector.put() raised: {e}", "finished": True}
                    return
                if not ok:
                    yield {"error": "connector.put() failed", "finished": True}
                    return
                yield {
                    "original_prompt": original_prompt,
                    "stage_connector_refs": {
                        **{str(k): v for k, v in stage_connector_refs.items()},
                        str(self.stage_id): metadata,
                    },
                    "finished": True,
                }
                return
            logger.warning(
                "Stage %d: no connector found for edge (%s→%s), falling through to SHM",
                self.stage_id,
                from_s,
                to_s,
            )

        # Final stage → router: write to SHM (no YAML edge for this leg).
        shm_meta = shm_write_bytes(serialize_obj(last_result), name=request_id)
        yield {"shm_meta": shm_meta, "finished": True}

    def _fetch_stage_inputs(
        self, stage_connector_refs: dict[int, Any], request_id: str
    ) -> list[_Proxy] | None:
        """Fetch previous stage outputs from connectors for the processor/engine.

        Fetches only the stages listed in engine_input_source (or all refs if empty).
        Returns _Proxy objects in engine_input_source order, or None on any error.
        """
        sources = self._engine_input_source or sorted(stage_connector_refs.keys())
        stage_list = []
        for stage_k in sources:
            if (meta_k := stage_connector_refs.get(stage_k)) is None:
                logger.error(
                    "Stage %d: no connector ref for source stage %d",
                    self.stage_id,
                    stage_k,
                )
                return None
            if (
                connector := self.connectors.get(_connector_key(stage_k, self.stage_id))
            ) is None:
                logger.error(
                    "Stage %d: no connector for edge (%s→%s)",
                    self.stage_id,
                    stage_k,
                    self.stage_id,
                )
                return None
            try:
                payload = connector.get(
                    str(stage_k), str(self.stage_id), request_id, metadata=meta_k
                )
            except Exception as e:
                logger.error(
                    "Stage %d: connector.get() failed: %s",
                    self.stage_id,
                    e,
                    exc_info=True,
                )
                return None
            payload_data = payload[0] if isinstance(payload, tuple) else payload
            if not payload_data:
                logger.error(
                    "Stage %d: empty payload from connector (%s→%s)",
                    self.stage_id,
                    stage_k,
                    self.stage_id,
                )
                return None
            engine_inputs = (
                payload_data.get("engine_inputs")
                if isinstance(payload_data, dict)
                else payload_data
            )
            stage_list.append(_Proxy(engine_outputs=[engine_inputs]))
        return stage_list


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

    from dynamo.vllm.health_check import VllmOmniHealthCheckPayload

    assert config.stage_id is not None  # dispatch in main.py guarantees this
    stage_id: int = config.stage_id
    stage_configs = load_stage_configs_from_yaml(config.stage_configs_path)  # type: ignore[arg-type]
    if stage_id >= len(stage_configs):
        raise ValueError(
            f"--stage-id {stage_id} out of range (YAML has {len(stage_configs)} stages)"
        )
    my_config = stage_configs[stage_id]
    stage_type: str = getattr(my_config, "stage_type", "llm")

    # Stage worker registers at {ns}.{model_stage}.generate — NOT {ns}.backend.generate.
    # Router registers at {ns}.backend.generate and discovers workers by model_stage.
    model_stage = getattr(my_config.engine_args, "model_stage", f"stage{stage_id}")
    generate_endpoint = runtime.endpoint(f"{config.namespace}.{model_stage}.generate")
    shutdown_endpoints[:] = [generate_endpoint]

    engine = _create_engine(config.model, my_config, stage_type)
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    # Connectors for inter-stage output transfer — type determined by YAML config
    # (SharedMemoryConnector, MooncakeConnector, etc.)
    _, connectors = initialize_orchestrator_connectors(config.stage_configs_path)  # type: ignore[arg-type]

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        stage_id=stage_id,
    )

    setup_metrics_collection(config, generate_endpoint, logger)

    if not getattr(config.engine_args, "data_parallel_rank", 0):
        logger.info(
            "Stage %d: serving internal stage endpoint '%s' (not registering model)",
            stage_id,
            generate_endpoint,
        )
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


def _connector_key(from_stage: int, to_stage: int) -> tuple[str, str]:
    """Build the connector dict key used by initialize_orchestrator_connectors."""
    return (str(from_stage), str(to_stage))


def _load_processor(func_path: str | None) -> Any:
    """Load a processor function from a dotted module path, or return None."""
    if not func_path:
        return None
    module_path, func_name = func_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), func_name)


def _build_default_sampling_params(stage_config: Any) -> list | None:
    """Construct typed sampling params from YAML default_sampling_params."""
    defaults = getattr(stage_config, "default_sampling_params", None)
    if not defaults:
        return None

    from omegaconf import OmegaConf

    if OmegaConf.is_config(defaults):
        params = OmegaConf.to_container(defaults, resolve=True)
    else:
        params = dict(defaults)
    params_dict = cast(dict[str, Any], params)

    stage_type = getattr(stage_config, "stage_type", "llm")
    if stage_type == "diffusion":
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        return [OmniDiffusionSamplingParams(**params_dict)]

    from vllm.sampling_params import SamplingParams

    return [SamplingParams(**params_dict)]


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
    """Convert a parsed stage config to a single-stage YAML dict."""
    from omegaconf import OmegaConf

    def _to_plain(obj: Any) -> Any:
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        return obj

    result: dict = {
        "stage_id": 0,
        "stage_type": stage_type,
        "engine_args": _to_plain(stage_config.engine_args),
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }

    for key in ("default_sampling_params", "is_comprehension"):
        val = getattr(stage_config, key, None)
        if val is not None:
            result[key] = _to_plain(val)

    runtime = getattr(stage_config, "runtime", None)
    if runtime is not None:
        rt = _to_plain(runtime)
        rt["devices"] = "0"
        result["runtime"] = rt

    return result


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
