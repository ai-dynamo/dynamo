# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import asyncio
import atexit
import copy
import importlib
import inspect
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import yaml
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors

try:
    from vllm_omni.distributed.omni_connectors.adapter import (
        compute_talker_prompt_ids_length,
    )
except ImportError:  # pragma: no cover - adapter is absent in lightweight stubs

    def compute_talker_prompt_ids_length(prompt_ids: list[int]) -> int:
        return len(prompt_ids)


try:
    from vllm_omni.engine.async_omni_engine import _apply_omni_final_stage_metadata
except ImportError:  # pragma: no cover - only used by lightweight unit-test stubs

    def _apply_omni_final_stage_metadata(request, final_stage_id):
        return request


from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs
from vllm_omni.inputs.data import OmniTokensPrompt

from dynamo import prometheus_names
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed
from dynamo.vllm.omni.utils import _build_sampling_params, parse_omni_request

logger = logging.getLogger(__name__)

_ASYNC_PREPARE_KEY = "__dynamo_omni_prepare"
_ASYNC_PREWARM_KEY = "__dynamo_omni_async_prewarm"
_ASYNC_PREWARM_READY_KEY = "__dynamo_omni_async_prewarm_ready"


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
        output_modalities: list | None = None,
        default_video_fps: int = 16,
        pipeline_stage_configs: list[Any] | None = None,
    ) -> None:
        self.engine = engine
        self.stage_id = stage_id
        self.connectors = connectors  # {(from_stage, to_stage): vllm_omni connector}
        self._output_modalities = output_modalities or []
        self._default_video_fps = default_video_fps
        self.stage_config = stage_config
        self._pipeline_stage_configs = pipeline_stage_configs or [stage_config]
        self._async_chunk = _stage_config_uses_async_chunk(stage_config)
        self._background_tasks: set[asyncio.Task] = set()

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()
        original_prompt = req.original_prompt
        stage_connector_refs = _int_keyed(req.stage_connector_refs)
        final_stage_id = req.final_stage_id
        async_prewarm = bool(request.get(_ASYNC_PREWARM_KEY))

        sampling_params_list_override: dict | None = None
        if request.get(_ASYNC_PREPARE_KEY):
            try:
                prepared = await self._prepare_router_request(request)
            except Exception as e:
                logger.error(
                    "Stage %d prepare error for %s: %s",
                    self.stage_id,
                    request_id,
                    e,
                    exc_info=True,
                )
                yield {"error": str(e), "finished": True}
                return
            yield {**prepared, "finished": True}
            return

        if async_prewarm:
            sampling_params_list_override = req.sampling_params_list
            prompt_token_ids = request.get("prompt_token_ids", [])
            if prompt_token_ids is None:
                prompt_token_ids = []
            try:
                prompt = self._build_async_prewarm_request(
                    list(prompt_token_ids),
                    original_prompt,
                    request_id,
                    sampling_params_list_override,
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return
        elif stage_connector_refs:
            sampling_params_list_override = req.sampling_params_list
            try:
                stage_list = self._fetch_stage_inputs(stage_connector_refs, request_id)
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return

            if len(stage_list) != len(
                self._engine_input_source or stage_connector_refs
            ):
                logger.warning(
                    "Stage %d: expected %d stage inputs, got %d",
                    self.stage_id,
                    len(self._engine_input_source or stage_connector_refs),
                    len(stage_list),
                )

            if self._processor is not None:
                prompt = self._process_stage_inputs(stage_list, original_prompt)
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
            else:
                upstream = stage_list[-1].engine_outputs[0]
                if hasattr(upstream, "outputs") and upstream.outputs:
                    try:
                        prompt = self._build_engine_core_request_from_upstream(
                            stage_list, request_id, sampling_params_list_override
                        )
                    except RuntimeError as e:
                        yield {"error": str(e), "finished": True}
                        return
                else:
                    prompt = upstream
        elif req.request_id is not None:
            final_stage_id = self._resolve_final_stage_id(request, final_stage_id)
            parsed = await parse_omni_request(
                request,
                self._output_modalities,
                self._default_video_fps,
                tokenizer_getter=self.engine.get_tokenizer,
            )
            prompt = parsed["engine_inputs"]
            original_prompt = parsed["original_prompt"]
            prompt_token_ids = request.get("prompt_token_ids")
            if prompt_token_ids:
                prompt = _with_prompt_token_ids(prompt, list(prompt_token_ids))
                original_prompt = _with_prompt_token_ids(
                    original_prompt, list(prompt_token_ids)
                )
            sampling_params_list_override = parsed["sampling_params_list"]
        else:
            final_stage_id = self._resolve_final_stage_id(request, final_stage_id)
            prompt = request

        logger.debug(
            "Stage %d: engine.generate for %s — prompt type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )

        sp = _build_sampling_params(self.stage_config, sampling_params_list_override)
        try:
            prompt = self._prepare_downstream_prompt(
                prompt,
                request_id=request_id,
                sampling_params_list=sp,
                final_stage_id=final_stage_id,
            )
        except RuntimeError as e:
            yield {"error": str(e), "finished": True}
            return

        if async_prewarm:
            if (
                self._async_chunk
                and final_stage_id is not None
                and final_stage_id > self.stage_id
            ):
                task = asyncio.create_task(
                    self._drain_async_request(prompt, request_id, sp)
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                await asyncio.sleep(0)
                yield {_ASYNC_PREWARM_READY_KEY: True, "finished": True}
                return
            yield {_ASYNC_PREWARM_READY_KEY: True}

        last_result = None
        try:
            async for chunk in self.engine.generate(
                prompt, request_id=request_id, sampling_params_list=sp
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

        _ensure_cumulative_token_ids(last_result)

        from_s, to_s = _connector_key(self.stage_id, self.stage_id + 1)
        connector = self.connectors.get((from_s, to_s))
        is_final_stage_for_request = (
            final_stage_id is not None and final_stage_id <= self.stage_id
        )
        if connector is not None and not (
            self._async_chunk and is_final_stage_for_request
        ):
            try:
                ok, _, metadata = connector.put(
                    from_s,
                    to_s,
                    request_id,
                    _prepare_connector_payload(last_result),
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
            out: dict = {
                "original_prompt": original_prompt,
                "stage_connector_refs": {
                    **{str(k): v for k, v in stage_connector_refs.items()},
                    str(self.stage_id): metadata,
                },
                "finished": True,
            }
            if final_stage_id is not None:
                out["final_stage_id"] = final_stage_id
            if sampling_params_list_override is not None:
                out["sampling_params_list"] = sampling_params_list_override
            yield out
            return

        shm_name = (
            f"{request_id}-stage-{self.stage_id}" if self._async_chunk else request_id
        )
        shm_meta = shm_write_bytes(serialize_obj(last_result), name=shm_name)
        out = {"shm_meta": shm_meta, "finished": True}
        if final_stage_id is not None:
            out["final_stage_id"] = final_stage_id
        yield out

    async def _prepare_router_request(self, request: dict) -> dict:
        frontend_request = _strip_internal_fields(request)
        parsed = await parse_omni_request(
            frontend_request,
            self._output_modalities,
            self._default_video_fps,
            tokenizer_getter=self.engine.get_tokenizer,
        )
        final_stage_id = self._resolve_final_stage_id(frontend_request, None)
        prompt_token_ids: list[int] = []
        messages = frontend_request.get("messages")
        if isinstance(messages, list):
            prompt_token_ids = await _render_chat_prompt_token_ids(
                frontend_request, self.engine
            ) or await _extract_chat_prompt_token_ids(frontend_request, self.engine)
        if not prompt_token_ids:
            prompt_token_ids = await _extract_prompt_token_ids(
                parsed["engine_inputs"], self.engine
            )
        return {
            "original_prompt": parsed["original_prompt"],
            "sampling_params_list": parsed["sampling_params_list"],
            "prompt_token_ids": prompt_token_ids,
            "final_stage_id": final_stage_id,
        }

    async def _drain_async_request(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list | None,
    ) -> None:
        try:
            async for _ in self.engine.generate(
                prompt, request_id=request_id, sampling_params_list=sampling_params_list
            ):
                pass
        except Exception:
            logger.exception(
                "Stage %d async prewarm background error for %s",
                self.stage_id,
                request_id,
            )

    def _build_engine_core_request_from_upstream(
        self,
        stage_list: list[_Proxy],
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        """Build an OmniEngineCoreRequest from the upstream stage output.

        Used for stages without a custom processor (e.g. code2wav).  Mirrors
        what the native orchestrator does via ``build_engine_core_request_from_tokens``
        and ``_forward_to_next_stage``.  Building an ``EngineCoreRequest``
        bypasses ``InputProcessor.process_inputs()`` which would fail for
        non-autoregressive stages (``worker_type: generation``) with
        "This model does not support generation".

        Raises RuntimeError on unexpected upstream output structure.
        """
        try:
            # engine_outputs[0]: first (and only) RequestOutput — Dynamo
            # processes one request at a time per stage.
            # outputs[0]: first CompletionOutput (n=1 sampling).
            # Matches native orchestrator's process_engine_inputs pattern.
            upstream = stage_list[-1].engine_outputs[0]
            token_ids = upstream.outputs[0].token_ids
        except (IndexError, AttributeError) as e:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot extract token_ids from "
                f"upstream output: {e}"
            ) from e

        return self._build_engine_core_request_from_tokens(
            list(token_ids), request_id, sampling_params_list_override
        )

    def _build_engine_core_request_from_tokens(
        self,
        token_ids: list[int],
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        tokens_prompt = OmniTokensPrompt(prompt_token_ids=list(token_ids))
        return self._build_engine_core_request_from_prompt(
            tokens_prompt, request_id, sampling_params_list_override
        )

    def _build_async_prewarm_request(
        self,
        prompt_token_ids: list[int],
        original_prompt: Any,
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        prompt = (
            copy.deepcopy(original_prompt) if isinstance(original_prompt, dict) else {}
        )
        prompt["prompt_token_ids"] = self._async_prewarm_prompt_token_ids(
            prompt_token_ids
        )
        prompt["multi_modal_data"] = None
        prompt["mm_processor_kwargs"] = None
        return self._build_engine_core_request_from_prompt(
            prompt, request_id, sampling_params_list_override
        )

    def _async_prewarm_prompt_token_ids(self, prompt_token_ids: list[int]) -> list[int]:
        if _stage_config_uses_generation_worker(self.stage_config):
            return []

        if not prompt_token_ids:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot async prewarm AR stage without prompt_token_ids"
            )

        try:
            prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
        except Exception:
            logger.debug("Failed to compute async prewarm prompt length", exc_info=True)
            prompt_len = max(1, len(prompt_token_ids))
        return [0] * prompt_len

    def _build_engine_core_request_from_prompt(
        self,
        prompt_input: Any,
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        sp_list = _build_sampling_params(
            self.stage_config, sampling_params_list_override
        )
        params = sp_list[0] if sp_list else None
        if params is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot build engine request without default sampling params"
            )

        kwargs = {
            "request_id": request_id,
            "prompt": prompt_input,
            "params": params,
        }
        if (
            "model_config"
            in inspect.signature(build_engine_core_request_from_tokens).parameters
        ):
            model_config = _engine_model_config(self.engine)
            if model_config is not None:
                kwargs["model_config"] = model_config
        prompt = build_engine_core_request_from_tokens(**kwargs)
        if getattr(prompt, "external_req_id", None) is None:
            prompt.external_req_id = prompt.request_id
        return prompt

    def _resolve_final_stage_id(self, request: dict, existing: int | None) -> int:
        if existing is not None:
            return int(existing)
        requested = request.get("modalities")
        if isinstance(requested, list):
            wanted = {str(modality).lower() for modality in requested}
            for stage_config in reversed(self._pipeline_stage_configs):
                final_type = str(
                    getattr(stage_config, "final_output_type", "") or ""
                ).lower()
                if final_type and final_type in wanted:
                    return int(getattr(stage_config, "stage_id", 0) or 0)
        return max(
            int(getattr(stage_config, "stage_id", idx) or 0)
            for idx, stage_config in enumerate(self._pipeline_stage_configs)
        )

    def _prepare_downstream_prompt(
        self,
        prompt: Any,
        *,
        request_id: str,
        sampling_params_list: list | None,
        final_stage_id: int | None,
    ) -> Any:
        if final_stage_id is None or final_stage_id <= self.stage_id:
            return prompt

        if hasattr(prompt, "request_id") and hasattr(prompt, "prompt_token_ids"):
            return _apply_omni_final_stage_metadata(prompt, final_stage_id)

        build_message = getattr(
            getattr(self.engine, "engine", None), "_build_add_request_message", None
        )
        if not callable(build_message):
            raise RuntimeError(
                f"Stage {self.stage_id}: engine cannot prebuild request with final_stage_id={final_stage_id}"
            )
        message = build_message(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
        )
        return message["prompt"] if isinstance(message, dict) else message.prompt

    def _process_stage_inputs(self, stage_list: list[_Proxy], original_prompt: Any):
        """Call vLLM-Omni stage processors using the v0.20 transition API."""
        if self._processor is None:
            raise RuntimeError(f"Stage {self.stage_id}: no processor configured")

        signature = inspect.signature(self._processor)
        positional_params = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        parameter_names = [parameter.name for parameter in positional_params]

        if parameter_names[:2] == ["stage_list", "engine_input_source"]:
            logger.debug(
                "Stage %d: processor dispatch branch=stage_list parameters=%s",
                self.stage_id,
                parameter_names,
            )
            return self._processor(
                stage_list,
                self._engine_input_source,
                [original_prompt],
                self._requires_mm,
            )

        source_outputs = [
            output
            for stage_input in stage_list
            for output in (stage_input.engine_outputs or [])
        ]
        if _accepts_source_outputs_processor(parameter_names):
            logger.debug(
                "Stage %d: processor dispatch branch=source_outputs parameters=%s",
                self.stage_id,
                parameter_names,
            )
            if len(parameter_names) >= 4:
                return self._processor(
                    source_outputs,
                    original_prompt,
                    self._requires_mm,
                    None,
                )
            return self._processor(
                source_outputs,
                original_prompt,
                self._requires_mm,
            )

        raise TypeError(
            f"Stage {self.stage_id}: unsupported processor signature for "
            f"{self._processor!r}; expected stage-list parameters "
            "('stage_list', 'engine_input_source', ...) or source-output "
            "parameters ('source_outputs', 'original_prompt', ...), got "
            f"{parameter_names}"
        )

    def _fetch_stage_inputs(
        self, stage_connector_refs: dict[int, Any], request_id: str
    ) -> list[_Proxy]:
        """Fetch previous stage outputs from connectors for the processor/engine.

        Fetches only the stages listed in engine_input_source (or all refs if empty).
        Returns _Proxy objects in engine_input_source order.
        Raises RuntimeError on any failure so the caller can propagate it as an error chunk.
        """
        sources = self._engine_input_source or sorted(stage_connector_refs.keys())
        stage_list = []
        for stage_k in sources:
            if (meta_k := stage_connector_refs.get(stage_k)) is None:
                raise RuntimeError(
                    f"Stage {self.stage_id}: no connector ref for source stage {stage_k}"
                )
            if (
                connector := self.connectors.get(_connector_key(stage_k, self.stage_id))
            ) is None:
                raise RuntimeError(
                    f"Stage {self.stage_id}: no connector for edge ({stage_k}→{self.stage_id})"
                )
            try:
                payload = connector.get(
                    str(stage_k), str(self.stage_id), request_id, metadata=meta_k
                )
            except Exception as e:
                raise RuntimeError(
                    f"Stage {self.stage_id}: connector.get() failed: {e}"
                ) from e
            payload_data = payload[0] if isinstance(payload, tuple) else payload
            if not payload_data:
                raise RuntimeError(
                    f"Stage {self.stage_id}: empty payload from connector ({stage_k}→{self.stage_id})"
                )
            if isinstance(payload_data, dict) and "engine_inputs" in payload_data:
                engine_inputs = payload_data["engine_inputs"]
                _restore_completion_output_attrs(
                    engine_inputs,
                    payload_data.get("_dynamo_completion_output_attrs"),
                )
            else:
                engine_inputs = payload_data
            _ensure_cumulative_token_ids(engine_inputs)
            stage_list.append(_Proxy(engine_outputs=[engine_inputs]))
        return stage_list


async def init_omni_stage(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Initialize a single omni stage worker.

    Mirrors init_omni() setup pattern exactly to avoid routing/handler issues.
    """
    if config.stage_id is None:
        raise ValueError("--stage-id is required for stage worker initialization")
    stage_id: int = config.stage_id
    resolved_stage_configs_path, stage_configs = load_and_resolve_stage_configs(
        config.model,
        config.stage_configs_path,
        kwargs={},
    )
    use_async_chunk = any(_stage_config_uses_async_chunk(cfg) for cfg in stage_configs)
    connector_configs_path = (
        resolved_stage_configs_path
        if use_async_chunk
        else _ensure_stage_connectors(resolved_stage_configs_path, stage_configs)
    )
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

    engine = _create_engine(
        config.model,
        my_config,
        stage_type,
        source_config_path=resolved_stage_configs_path,
        async_chunk=use_async_chunk,
    )
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    # Connectors for inter-stage output transfer — type determined by YAML config
    # (SharedMemoryConnector, MooncakeConnector, etc.)
    _, connectors = initialize_orchestrator_connectors(connector_configs_path)  # type: ignore[arg-type]

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        output_modalities=config.output_modalities,
        default_video_fps=config.default_video_fps,
        stage_id=stage_id,
        pipeline_stage_configs=stage_configs,
    )

    setup_metrics_collection(config, generate_endpoint, logger)

    if config.engine_args.data_parallel_rank:
        logger.info(
            "Stage %d: non-leader DP rank %d; waiting for shutdown",
            stage_id,
            config.engine_args.data_parallel_rank,
        )
        if shutdown_event is not None:
            await shutdown_event.wait()
        return

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


def _ensure_stage_connectors(stage_configs_path: str, stage_configs: list[Any]) -> str:
    """Add default SHM connector edges for stage configs that omit them."""
    try:
        with open(stage_configs_path) as f:
            deploy_config = yaml.safe_load(f) or {}
    except OSError:
        logger.warning(
            "Could not read stage config %s; using it without connector synthesis",
            stage_configs_path,
        )
        return stage_configs_path

    if not isinstance(deploy_config, dict):
        return stage_configs_path

    stages = deploy_config.get("stages")
    if not isinstance(stages, list):
        return stage_configs_path

    stages_by_id = {
        int(stage.get("stage_id", idx)): stage
        for idx, stage in enumerate(stages)
        if isinstance(stage, dict)
    }
    connector_name = "connector_of_shared_memory"
    changed = False

    for stage_config in stage_configs:
        to_stage = int(getattr(stage_config, "stage_id", -1))
        if to_stage < 0:
            continue
        stage = stages_by_id.get(to_stage)
        if stage is None:
            continue
        input_connectors = stage.setdefault("input_connectors", {})
        if not isinstance(input_connectors, dict):
            continue
        for from_stage in getattr(stage_config, "engine_input_source", []) or []:
            connector_key = f"from_stage_{int(from_stage)}"
            if connector_key not in input_connectors:
                input_connectors[connector_key] = connector_name
                changed = True

    if not changed:
        return stage_configs_path

    connectors = deploy_config.setdefault("connectors", {})
    if not isinstance(connectors, dict):
        raise ValueError(
            f"'connectors' in {stage_configs_path} must be a mapping to "
            f"synthesize {connector_name}; got {type(connectors).__name__}"
        )
    connectors.setdefault(
        connector_name,
        {
            "name": "SharedMemoryConnector",
            "extra": {},
        },
    )

    tmp_dir = tempfile.mkdtemp(prefix=f"dynamo_omni_stage_{os.getpid()}_")
    tmp_path = os.path.join(tmp_dir, "stage_config.yaml")
    with open(tmp_path, "w") as tmp:
        yaml.safe_dump(deploy_config, tmp, sort_keys=False)

    atexit.register(_cleanup_temp_stage_config, tmp_dir)
    logger.info(
        "Synthesized default SharedMemoryConnector edges in %s from %s",
        tmp_path,
        stage_configs_path,
    )
    return tmp_path


def _cleanup_temp_stage_config(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    except OSError:
        pass


def _strip_internal_fields(request: dict) -> dict:
    return {
        key: value
        for key, value in request.items()
        if not key.startswith("__dynamo_omni_")
    }


def _with_prompt_token_ids(prompt: Any, prompt_token_ids: list[int]) -> Any:
    if isinstance(prompt, dict):
        updated = dict(prompt)
        updated["prompt_token_ids"] = list(prompt_token_ids)
        return updated
    if isinstance(prompt, str):
        return {"prompt": prompt, "prompt_token_ids": list(prompt_token_ids)}
    return prompt


async def _extract_prompt_token_ids(prompt: Any, engine: StageEngine) -> list[int]:
    token_ids = _get_prompt_token_ids(prompt)
    if token_ids is not None:
        return token_ids
    text = prompt if isinstance(prompt, str) else None
    if text is None and isinstance(prompt, dict):
        value = prompt.get("prompt")
        if isinstance(value, str):
            text = value
    if not text:
        return []
    tokenizer = await _get_tokenizer(engine)
    if tokenizer is None:
        return []
    return _encode_text(tokenizer, text)


async def _extract_chat_prompt_token_ids(
    request: dict, engine: StageEngine
) -> list[int]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    tokenizer = await _get_tokenizer(engine)
    if tokenizer is None:
        return []
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return []
    try:
        return list(
            apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            or []
        )
    except Exception:
        logger.debug("Tokenizer chat template unavailable", exc_info=True)
        return []


async def _render_chat_prompt_token_ids(
    request: dict, engine: StageEngine
) -> list[int]:
    renderer = getattr(engine, "renderer", None)
    engine_impl = getattr(engine, "engine", None)
    input_processor = getattr(engine_impl, "input_processor", None)
    model_config = getattr(input_processor, "model_config", None)
    if renderer is None or model_config is None:
        return []

    try:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )
    except ImportError:
        return []

    try:
        chat_request = ChatCompletionRequest.model_validate(request)
        tok_params = chat_request.build_tok_params(model_config)
        chat_params = chat_request.build_chat_params(
            getattr(chat_request, "chat_template", None),
            "auto",
        )
        (_,), (engine_prompt,) = await renderer.render_chat_async(
            [chat_request.messages],
            chat_params,
            tok_params,
            prompt_extras={
                key: value
                for key in ("mm_processor_kwargs", "cache_salt")
                if (value := getattr(chat_request, key, None)) is not None
            },
        )
    except Exception:
        logger.debug("Native chat rendering unavailable", exc_info=True)
        return []

    return await _extract_prompt_token_ids(engine_prompt, engine)


async def _get_tokenizer(engine: StageEngine) -> Any:
    tokenizer = engine.get_tokenizer()
    return await tokenizer if inspect.isawaitable(tokenizer) else tokenizer


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return list(encode(text, add_special_tokens=False))
        except TypeError:
            return list(encode(text))
    result = tokenizer(text)
    input_ids = getattr(result, "input_ids", None)
    if input_ids is None and isinstance(result, dict):
        input_ids = result.get("input_ids")
    return list(input_ids or [])


def _get_prompt_token_ids(prompt: Any) -> list[int] | None:
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
    else:
        token_ids = getattr(prompt, "prompt_token_ids", None)
    return None if token_ids is None else list(token_ids)


def _engine_model_config(engine: StageEngine) -> Any:
    try:
        stage_configs = getattr(engine.engine, "stage_vllm_configs", None)
        if stage_configs:
            return getattr(stage_configs[0], "model_config", None)
    except Exception:
        logger.debug("Failed to read vLLM model config", exc_info=True)
    return None


def _prepare_connector_payload(engine_inputs: Any) -> Any:
    """Preserve dynamic CompletionOutput attrs that Omni's msgpack codec drops."""
    _promote_request_multimodal_output(engine_inputs)
    output_attrs = _collect_completion_output_attrs(engine_inputs)
    if len(output_attrs) == 0:
        return engine_inputs
    return {
        "engine_inputs": engine_inputs,
        "_dynamo_completion_output_attrs": output_attrs,
    }


def _collect_completion_output_attrs(engine_inputs: Any) -> list[dict[str, Any]]:
    output_attrs: list[dict[str, Any]] = []
    for output in _iter_completion_outputs(engine_inputs):
        attrs: dict[str, Any] = {}
        cumulative_token_ids = getattr(output, "cumulative_token_ids", None)
        if cumulative_token_ids is not None:
            attrs["cumulative_token_ids"] = list(cumulative_token_ids)
        multimodal_output = getattr(output, "multimodal_output", None)
        if multimodal_output:
            attrs["multimodal_output"] = multimodal_output
        output_attrs.append(attrs)
    return output_attrs


def _promote_request_multimodal_output(engine_inputs: Any) -> None:
    """Expose request-level multimodal payloads on the sole completion output."""
    request_multimodal_output = getattr(engine_inputs, "multimodal_output", None)
    if not request_multimodal_output:
        return

    outputs = _iter_completion_outputs(engine_inputs)
    if len(outputs) != 1:
        return

    completion = outputs[0]
    if not getattr(completion, "multimodal_output", None):
        completion.multimodal_output = request_multimodal_output


def _restore_completion_output_attrs(
    engine_inputs: Any, output_attrs: Any | None
) -> None:
    if not isinstance(output_attrs, list):
        return
    for output, attrs in zip(
        _iter_completion_outputs(engine_inputs), output_attrs, strict=False
    ):
        if not isinstance(attrs, dict):
            continue
        if "cumulative_token_ids" in attrs:
            output.cumulative_token_ids = list(attrs["cumulative_token_ids"])
        if "multimodal_output" in attrs:
            output.multimodal_output = attrs["multimodal_output"]


def _ensure_cumulative_token_ids(engine_inputs: Any) -> None:
    """Bridge vLLM 0.20 CompletionOutput into vLLM-Omni stage processors."""
    for output in _iter_completion_outputs(engine_inputs):
        if not hasattr(output, "cumulative_token_ids") and hasattr(output, "token_ids"):
            output.cumulative_token_ids = list(output.token_ids)


def _iter_completion_outputs(engine_inputs: Any):
    outputs = getattr(engine_inputs, "outputs", None)
    if outputs is None:
        request_output = getattr(engine_inputs, "request_output", None)
        outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return []
    return list(outputs)


def _accepts_source_outputs_processor(parameter_names: list[str]) -> bool:
    if len(parameter_names) < 3:
        return False
    prompt_param_name = parameter_names[1].lstrip("_")
    requires_mm_param_name = parameter_names[2].lstrip("_")
    return (
        parameter_names[0] == "source_outputs"
        and prompt_param_name in {"original_prompt", "prompt"}
        and requires_mm_param_name in {"requires_mm", "requires_multimodal_data"}
    )


def _stage_config_uses_async_chunk(stage_config: Any) -> bool:
    engine_args = getattr(stage_config, "engine_args", None)
    return bool(getattr(engine_args, "async_chunk", False))


def _stage_config_uses_generation_worker(stage_config: Any) -> bool:
    engine_args = getattr(stage_config, "engine_args", None)
    worker_type = str(getattr(engine_args, "worker_type", "") or "").lower()
    stage_type = str(getattr(stage_config, "stage_type", "") or "").lower()
    return worker_type == "generation" or stage_type == "diffusion"


def _create_engine(
    model: str,
    stage_config: Any,
    stage_type: str,
    *,
    source_config_path: str | None = None,
    async_chunk: bool | None = None,
) -> StageEngine:
    """Create AsyncOmni with a single-stage YAML."""
    use_async_chunk = (
        _stage_config_uses_async_chunk(stage_config)
        if async_chunk is None
        else bool(async_chunk)
    )
    source_stage_id = int(getattr(stage_config, "stage_id", 0) or 0)
    stage_arg = _stage_config_to_dict(
        stage_config,
        stage_type,
        preserve_stage_id=use_async_chunk,
    )
    _normalize_single_stage_runtime_devices(stage_arg)
    source_config = _load_stage_config(source_config_path) if use_async_chunk else {}
    if use_async_chunk:
        _inject_output_connectors_from_source(stage_arg, source_config, source_stage_id)
    single_stage_config = {
        "async_chunk": use_async_chunk,
        "stage_args": [stage_arg],
        "runtime": _runtime_config_for_single_stage(
            source_config, preserve_edges=use_async_chunk
        ),
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(single_stage_config, tmp)
        tmp_path = tmp.name

    try:
        engine = AsyncOmni(model=model, stage_configs_path=tmp_path)
        if use_async_chunk:
            _preserve_external_request_ids(engine)
        return engine
    finally:
        os.unlink(tmp_path)


def _preserve_external_request_ids(engine: Any) -> None:
    """Keep Dynamo's request ids stable inside single-stage AsyncOmni workers."""

    def _stable_request_id(external_request_id: str) -> str:
        return external_request_id or str(uuid.uuid4())

    if hasattr(engine, "_get_unique_request_id"):
        engine._get_unique_request_id = _stable_request_id


def _stage_config_to_dict(
    stage_config: Any,
    stage_type: str,
    preserve_stage_id: bool = False,
) -> dict:
    """Convert a parsed stage config to a single-stage YAML dict."""
    stage_id = int(getattr(stage_config, "stage_id", 0) or 0)
    result: dict = {
        "stage_id": stage_id if preserve_stage_id else 0,
        "stage_type": stage_type,
        "engine_args": _to_plain(stage_config.engine_args),
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }

    for key in (
        "default_sampling_params",
        "is_comprehension",
        "custom_process_input_func",
        "requires_multimodal_data",
        "input_connectors",
        "output_connectors",
    ):
        val = getattr(stage_config, key, None)
        if val is not None:
            result[key] = _to_plain(val)

    engine_input_source = getattr(
        stage_config,
        "engine_input_source",
        getattr(stage_config, "input_sources", None),
    )
    if engine_input_source is not None:
        result["engine_input_source"] = _to_plain(engine_input_source)

    runtime = getattr(stage_config, "runtime", None)
    if runtime is not None:
        rt = _to_plain(runtime)
        rt.setdefault("devices", "0")
        result["runtime"] = rt

    return result


def _to_plain(obj: Any) -> Any:
    if _is_omegaconf_config(obj):
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        return _to_plain(OmegaConf.to_container(obj, resolve=True))
    if isinstance(obj, (list, tuple)):
        return [_to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_plain(value) for key, value in obj.items()}
    if hasattr(obj, "__dict__"):
        return {key: _to_plain(value) for key, value in vars(obj).items()}
    return copy.deepcopy(obj)


def _is_omegaconf_config(obj: Any) -> bool:
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
    except ImportError:
        return False
    return bool(OmegaConf.is_config(obj))


def _load_stage_config(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            source = yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Failed to read source omni stage config", exc_info=True)
        return {}
    return source if isinstance(source, dict) else {}


def _runtime_config_for_single_stage(
    source_config: dict, *, preserve_edges: bool
) -> dict:
    if not preserve_edges or not source_config:
        return {"edges": []}

    runtime = copy.deepcopy(source_config.get("runtime") or {})
    if "connectors" in source_config and "connectors" not in runtime:
        runtime["connectors"] = copy.deepcopy(source_config["connectors"])
    if "edges" in source_config and "edges" not in runtime:
        runtime["edges"] = copy.deepcopy(source_config["edges"])
    runtime.setdefault("edges", [])
    return runtime


def _inject_output_connectors_from_source(
    stage_dict: dict,
    source_config: dict,
    source_stage_id: int,
) -> None:
    if not source_config:
        return

    output_connectors = dict(stage_dict.get("output_connectors") or {})
    for downstream in (
        source_config.get("stage_args") or source_config.get("stages") or []
    ):
        if not isinstance(downstream, dict):
            continue
        to_stage = downstream.get("stage_id")
        input_connectors = downstream.get("input_connectors") or {}
        connector_ref = input_connectors.get(f"from_stage_{source_stage_id}")
        if connector_ref is not None and to_stage is not None:
            output_connectors.setdefault(f"to_stage_{to_stage}", connector_ref)
    if output_connectors:
        stage_dict["output_connectors"] = output_connectors


def _normalize_single_stage_runtime_devices(stage_arg: dict) -> None:
    """Map stage-local device visibility to vLLM-Omni logical device IDs."""
    runtime = stage_arg.get("runtime")
    if not isinstance(runtime, dict):
        return

    devices = runtime.get("devices")
    visible_devices = _get_visible_devices()
    if devices in (None, "cpu") or not visible_devices:
        return

    requested_devices = _parse_runtime_devices(devices)
    if requested_devices != visible_devices:
        return

    # Dynamo starts each stage worker with the process visibility already
    # narrowed to that stage's devices. vLLM-Omni then interprets runtime.devices
    # as logical indexes inside that visible set.
    runtime["devices"] = ",".join(str(i) for i in range(len(requested_devices)))


def _get_visible_devices() -> list[str]:
    for env_var in (
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    ):
        if devices := os.environ.get(env_var):
            return _parse_runtime_devices(devices)
    return []


def _parse_runtime_devices(devices: Any) -> list[str]:
    if isinstance(devices, int):
        return [str(devices)]
    if isinstance(devices, str):
        return [device.strip() for device in devices.split(",") if device.strip()]
    if isinstance(devices, (list, tuple)):
        return [str(device).strip() for device in devices if str(device).strip()]
    return []


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
