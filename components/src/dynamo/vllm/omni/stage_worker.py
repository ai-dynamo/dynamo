# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import asyncio
import copy
import importlib
import inspect
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import yaml
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.inputs.data import OmniTokensPrompt

from dynamo import prometheus_names
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed
from dynamo.vllm.omni.utils import (
    _build_sampling_params,
    load_omni_stage_configs,
    parse_omni_request,
    stage_configs_use_async_chunk,
)

logger = logging.getLogger(__name__)

_ASYNC_PREPARE_KEY = "__dynamo_omni_prepare"
_ASYNC_PREWARM_KEY = "__dynamo_omni_async_prewarm"
_ASYNC_PREWARM_READY_KEY = "__dynamo_omni_async_prewarm_ready"
_MEDIA_CONTENT_KEYS = frozenset(
    (
        "image_url",
        "image_pil",
        "image_embeds",
        "audio_url",
        "input_audio",
        "audio_embeds",
        "video_url",
    )
)


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
    ) -> None:
        self.engine = engine
        self.stage_id = stage_id
        self.connectors = connectors  # {(from_stage, to_stage): vllm_omni connector}
        self._output_modalities = output_modalities or []
        self._default_video_fps = default_video_fps
        self.stage_config = stage_config
        self._async_chunk = _stage_config_uses_async_chunk(stage_config)

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config,
            "engine_input_source",
            getattr(stage_config, "input_sources", []),
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()
        original_prompt = req.original_prompt
        # JSON sends dict keys as strings; normalize to int for stage_connector_refs.
        stage_connector_refs = _int_keyed(req.stage_connector_refs)

        # --- Resolve engine inputs ---
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
        if request.get(_ASYNC_PREWARM_KEY):
            prompt_token_ids = request.get("prompt_token_ids") or [0]
            sampling_params_list_override = req.sampling_params_list
            try:
                prompt = self._build_engine_core_request_from_tokens(
                    list(prompt_token_ids),
                    request_id,
                    sampling_params_list_override,
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return
            yield {_ASYNC_PREWARM_READY_KEY: True}
        elif stage_connector_refs:
            # Stage N > 0: fetch previous stage outputs from connectors, run pre-processor.
            sampling_params_list_override = req.sampling_params_list
            try:
                stage_list = self._fetch_stage_inputs(stage_connector_refs, request_id)
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return

            filled = sum(1 for p in stage_list if p.engine_outputs is not None)
            expected = len(self._engine_input_source or stage_connector_refs)
            if filled != expected:
                logger.warning(
                    "Stage %d: expected %d stage inputs, got %d",
                    self.stage_id,
                    expected,
                    filled,
                )

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
                # No processor: check if the upstream output has the
                # structure needed to build an OmniEngineCoreRequest
                # (e.g. code2wav receiving token_ids from talker).
                # Otherwise fall back to passing the raw data directly.
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
            # Stage 0 via router: raw request forwarded with request_id — parse it.
            try:
                parsed = await parse_omni_request(
                    request,
                    self._output_modalities,
                    self._default_video_fps,
                    tokenizer_getter=self.engine.get_tokenizer,
                    renderer=getattr(self.engine, "renderer", None),
                    model_config=getattr(self.engine, "model_config", None),
                )
            except Exception as e:
                logger.error(
                    "Stage %d request parse error for %s: %s",
                    self.stage_id,
                    request_id,
                    e,
                    exc_info=True,
                )
                yield {"error": str(e), "finished": True}
                return
            prompt = parsed["engine_inputs"]
            original_prompt = parsed["original_prompt"]
            sampling_params_list_override = parsed["sampling_params_list"]
        else:
            # Direct frontend → stage (single-stage, no router).
            prompt = request

        logger.debug(
            "Stage %d: engine.generate for %s — prompt type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )

        sp = _build_sampling_params(self.stage_config, sampling_params_list_override)
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

        # --- Write output ---
        # Check for a downstream connector first, regardless of final_output.
        # In vllm-omni's native mode, multiple stages can set final_output=True
        # (meaning "produces user-visible output"). In Dynamo's disaggregated
        # mode the actual pipeline topology — connector edges from the YAML —
        # determines whether output should go to a connector or to SHM.
        from_s, to_s = _connector_key(self.stage_id, self.stage_id + 1)
        connector = self.connectors.get((from_s, to_s))
        if connector is not None and not self._async_chunk:
            try:
                connector_payload = _prepare_connector_payload(last_result)
                ok, _, metadata = connector.put(  # type: ignore[arg-type]
                    from_s, to_s, request_id, connector_payload
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
            if sampling_params_list_override is not None:
                out["sampling_params_list"] = sampling_params_list_override
            yield out
            return

        # Final stage → router: write output to shared memory and return the SHM handle.
        # The router reads it back via shm_deserialize() to format the response.
        #
        # NOTE: This is a single-node-only workaround — SHM requires the final stage
        # worker and the router to reside on the same machine. A proper multi-node
        # solution would use a connector edge (like inter-stage connectors) instead.
        # Tracked in TODO: shm_meta should be replaced by a YAML-configured connector edge.
        shm_name = (
            f"{request_id}-stage-{self.stage_id}" if self._async_chunk else request_id
        )
        shm_meta = shm_write_bytes(serialize_obj(last_result), name=shm_name)
        yield {"shm_meta": shm_meta, "finished": True}

    async def _prepare_router_request(self, request: dict) -> dict:
        frontend_request = _with_prepare_multimodal_uuids(
            _strip_internal_fields(request),
            request.get("request_id") or "",
        )
        parsed = await parse_omni_request(
            frontend_request,
            self._output_modalities,
            self._default_video_fps,
            tokenizer_getter=self.engine.get_tokenizer,
            renderer=getattr(self.engine, "renderer", None),
            model_config=getattr(self.engine, "model_config", None),
        )
        prompt_token_ids = await _extract_prompt_token_ids(
            parsed["engine_inputs"],
            self.engine,
        )
        return {
            "original_prompt": parsed["original_prompt"],
            "sampling_params_list": parsed["sampling_params_list"],
            "prompt_token_ids": prompt_token_ids,
        }

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
            list(token_ids),
            request_id,
            sampling_params_list_override,
        )

    def _build_engine_core_request_from_tokens(
        self,
        token_ids: list[int],
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        tokens_prompt = OmniTokensPrompt(prompt_token_ids=token_ids)
        sp_list = _build_sampling_params(
            self.stage_config, sampling_params_list_override
        )
        params = sp_list[0] if sp_list else None
        if params is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot build engine request without "
                "default sampling params"
            )
        model_config = _engine_model_config(self.engine)
        prompt = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=tokens_prompt,
            params=params,
            model_config=model_config,
        )
        # Pre-built EngineCoreRequests skip the output processor registration
        # in _build_add_request_message (the isinstance(prompt, EngineCoreRequest)
        # branch bypasses that block).  Register manually so that the engine's
        # output processor can match the response back to this request.
        prompt.external_req_id = prompt.request_id
        self.engine.engine.output_processors[0].add_request(
            request=prompt,
            prompt=None,
            parent_req=None,
            request_index=0,
            queue=None,
        )
        return prompt

    def _fetch_stage_inputs(
        self, stage_connector_refs: dict[int, Any], request_id: str
    ) -> list[_Proxy]:
        """Fetch previous stage outputs from connectors for the processor/engine.

        Fetches only the stages listed in engine_input_source (or all refs if empty).
        Returns a sparse list indexed by stage_id. vLLM-Omni processors use
        engine_input_source values as indexes into stage_list.
        Raises RuntimeError on any failure so the caller can propagate it as an error chunk.
        """
        sources = self._engine_input_source or sorted(stage_connector_refs.keys())
        max_stage_id = max(sources) if sources else 0
        stage_list = [_Proxy() for _ in range(max_stage_id + 1)]
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
            engine_inputs = (
                payload_data.get("engine_inputs")
                if isinstance(payload_data, dict)
                else payload_data
            )
            _ensure_cumulative_token_ids(engine_inputs)
            stage_list[stage_k] = _Proxy(engine_outputs=[engine_inputs])
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
    stage_configs = load_omni_stage_configs(config.model, config.stage_configs_path)
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
        source_config_path=config.stage_configs_path,
        async_chunk=stage_configs_use_async_chunk(stage_configs),
    )
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    # Connectors for inter-stage output transfer — type determined by YAML config
    # (SharedMemoryConnector, MooncakeConnector, etc.)
    _, connectors = initialize_orchestrator_connectors(config.stage_configs_path)  # type: ignore[arg-type]

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        output_modalities=config.output_modalities,
        default_video_fps=config.default_video_fps,
        stage_id=stage_id,
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


def _ensure_cumulative_token_ids(result: Any) -> None:
    """Bridge vLLM 0.20 CompletionOutput into vLLM-Omni rc1 processors."""
    for output in getattr(result, "outputs", []) or []:
        if not hasattr(output, "cumulative_token_ids") and hasattr(output, "token_ids"):
            output.cumulative_token_ids = list(output.token_ids)


def _strip_internal_fields(request: dict) -> dict:
    return {k: v for k, v in request.items() if not k.startswith("__dynamo_omni_")}


def _with_prepare_multimodal_uuids(request: dict, request_id: str) -> dict:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return request

    prepared = copy.deepcopy(request)
    for message_index, message in enumerate(prepared.get("messages") or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part_index, part in enumerate(parts):
            if _is_media_content_part(part) and part.get("uuid") is None:
                prepare_uuid = (
                    f"__dynamo_prepare_{request_id}_{message_index}_{part_index}"
                )
                part["uuid"] = prepare_uuid
    return prepared


def _is_media_content_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    return part_type in _MEDIA_CONTENT_KEYS or any(
        key in part for key in _MEDIA_CONTENT_KEYS
    )


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

    tokenizer_result = engine.get_tokenizer()
    tokenizer = (
        await tokenizer_result
        if inspect.isawaitable(tokenizer_result)
        else tokenizer_result
    )
    if tokenizer is None:
        return []
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
    if token_ids is None:
        return None
    return list(token_ids)


def _prepare_connector_payload(result: Any) -> Any:
    """Return the object to transfer to the next stage via connector.

    AsyncOmni yields OmniRequestOutput wrappers, while vLLM-Omni's native
    orchestrator forwards the underlying RequestOutput to stage processors.
    Also make cumulative token ids survive connector serialization: vLLM-Omni
    stores them as a dynamic CompletionOutput field, but the rc1 msgpack serde
    only serializes dataclass fields such as token_ids.
    """
    payload = getattr(result, "request_output", None) or result
    for output in getattr(payload, "outputs", []) or []:
        cumulative = getattr(output, "cumulative_token_ids", None)
        if cumulative is not None:
            output.token_ids = list(cumulative)
            continue
        if hasattr(output, "token_ids"):
            output.cumulative_token_ids = list(output.token_ids)
    return payload


def _engine_model_config(engine: StageEngine) -> Any:
    try:
        stage_configs = getattr(engine.engine, "stage_vllm_configs", None)
        if stage_configs:
            return getattr(stage_configs[0], "model_config", None)
    except Exception:
        logger.debug(
            "Failed to read vLLM model config from stage engine", exc_info=True
        )
    return None


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
    if use_async_chunk and source_stage_id > 0:
        _patch_headless_stage_finalize()
    stage_dict = _stage_config_to_dict(stage_config, stage_type, use_async_chunk)
    if use_async_chunk:
        _inject_output_connectors_from_source(
            stage_dict,
            source_config_path,
            source_stage_id,
        )
    single_stage_config = {
        "async_chunk": use_async_chunk,
        "stage_args": [stage_dict],
        "runtime": _runtime_config_for_single_stage(
            source_config_path,
            preserve_edges=use_async_chunk,
        ),
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(single_stage_config, tmp)
        tmp_path = tmp.name

    try:
        return AsyncOmni(model=model, stage_configs_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def _stage_config_to_dict(
    stage_config: Any, stage_type: str, preserve_stage_id: bool = False
) -> dict:
    """Convert a parsed stage config to a single-stage YAML dict."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    def _to_plain(obj: Any) -> Any:
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        return obj

    stage_id = int(getattr(stage_config, "stage_id", 0) or 0)
    result: dict = {
        "stage_id": stage_id if preserve_stage_id else 0,
        "stage_type": stage_type,
        "engine_args": _to_plain(stage_config.engine_args),
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }

    input_sources = getattr(
        stage_config,
        "engine_input_source",
        getattr(stage_config, "input_sources", None),
    )
    if input_sources is not None:
        result["engine_input_source"] = _to_plain(input_sources)

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

    runtime = getattr(stage_config, "runtime", None)
    if runtime is not None:
        rt = _to_plain(runtime)
        rt["devices"] = "0"
        result["runtime"] = rt

    return result


def _stage_config_uses_async_chunk(stage_config: Any) -> bool:
    engine_args = getattr(stage_config, "engine_args", None)
    return bool(getattr(engine_args, "async_chunk", False))


def _runtime_config_for_single_stage(
    source_config_path: str | None,
    *,
    preserve_edges: bool,
) -> dict:
    if not preserve_edges or not source_config_path:
        return {"edges": []}

    try:
        with open(source_config_path, encoding="utf-8") as f:
            source = yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Failed to read source omni stage config", exc_info=True)
        return {"edges": []}

    runtime = copy.deepcopy(source.get("runtime") or {})
    if "connectors" in source and "connectors" not in runtime:
        runtime["connectors"] = copy.deepcopy(source["connectors"])
    if "edges" in source and "edges" not in runtime:
        runtime["edges"] = copy.deepcopy(source["edges"])
    runtime.setdefault("edges", [])
    return runtime


def _inject_output_connectors_from_source(
    stage_dict: dict,
    source_config_path: str | None,
    source_stage_id: int,
) -> None:
    if not source_config_path:
        return

    try:
        with open(source_config_path, encoding="utf-8") as f:
            source = yaml.safe_load(f) or {}
    except Exception:
        logger.debug("Failed to read source omni stage config", exc_info=True)
        return

    source_stages = source.get("stage_args") or source.get("stages") or []
    output_connectors = dict(stage_dict.get("output_connectors") or {})
    for downstream in source_stages:
        to_stage = downstream.get("stage_id")
        input_connectors = downstream.get("input_connectors") or {}
        connector_ref = input_connectors.get(f"from_stage_{source_stage_id}")
        if connector_ref is not None and to_stage is not None:
            output_connectors.setdefault(f"to_stage_{to_stage}", connector_ref)

    if output_connectors:
        stage_dict["output_connectors"] = output_connectors


def _patch_headless_stage_finalize() -> None:
    try:
        from vllm_omni.engine import async_omni_engine, stage_init_utils
    except Exception:
        logger.debug("Failed to import vLLM-Omni stage finalizer", exc_info=True)
        return

    if getattr(async_omni_engine, "_dynamo_headless_finalize_patch", False):
        return

    def _finalize_initialized_stages(stage_clients, input_processor):
        if any(stage_client is None for stage_client in stage_clients):
            raise RuntimeError(
                "Stage initialization completed with missing stage clients"
            )
        initialized_stage_clients = [
            stage_client for stage_client in stage_clients if stage_client is not None
        ]
        default_sampling_params_list = [
            stage_client.default_sampling_params
            for stage_client in initialized_stage_clients
        ]
        stage_metadata = [
            {
                "final_output": stage_client.final_output,
                "final_output_type": stage_client.final_output_type,
                "stage_type": stage_client.stage_type,
            }
            for stage_client in initialized_stage_clients
        ]
        return initialized_stage_clients, default_sampling_params_list, stage_metadata

    stage_init_utils.finalize_initialized_stages = _finalize_initialized_stages
    async_omni_engine.finalize_initialized_stages = _finalize_initialized_stages
    async_omni_engine._dynamo_headless_finalize_patch = True


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
