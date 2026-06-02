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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import torch
import yaml
from vllm.entrypoints.chat_utils import MM_PARSER_MAP
from vllm.v1.request import EngineCoreRequest
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine import async_omni_engine, stage_init_utils
from vllm_omni.engine.async_omni_engine import _apply_omni_final_stage_metadata
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.inputs.data import OmniTokensPrompt

from dynamo import prometheus_names
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed
from dynamo.vllm.omni.utils import (
    OmniChatPreprocessor,
    _build_sampling_params,
    load_omni_stage_configs,
    parse_omni_request,
    stage_configs_use_async_chunk,
)

logger = logging.getLogger(__name__)

_ASYNC_PREPARE_KEY = "__dynamo_omni_prepare"
_ASYNC_PREWARM_KEY = "__dynamo_omni_async_prewarm"
_ASYNC_PREWARM_READY_KEY = "__dynamo_omni_async_prewarm_ready"
_ASYNC_DRAIN_INLINE_KEY = "__dynamo_omni_async_drain_inline"
_TEXT_CONTENT_TYPES = frozenset(
    ("text", "input_text", "output_text", "refusal", "thinking")
)
_MEDIA_CONTENT_KEYS = frozenset(MM_PARSER_MAP) - _TEXT_CONTENT_TYPES


@dataclass
class _Proxy:
    """Satisfies stage_list[i].engine_outputs for processor functions.

    Processor functions (e.g. ar2diffusion) access stage_list[i].engine_outputs
    as a list of OmniRequestOutput objects.
    """

    engine_outputs: Any = None
    bridge_states: dict[str, Any] | None = None


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

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )
        model_config = getattr(engine, "model_config", None)
        self._omni_chat_preprocessor = (
            OmniChatPreprocessor(model_config) if model_config is not None else None
        )
        self._background_tasks: set[asyncio.Task] = set()

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()
        original_prompt = req.original_prompt
        # JSON sends dict keys as strings; normalize to int for stage_connector_refs.
        stage_connector_refs = _int_keyed(req.stage_connector_refs)
        final_stage_id: int | None = None
        async_prewarm_request = False

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
            async_prewarm_request = True
            final_stage_id = req.final_stage_id
            prompt_token_ids = (
                request["prompt_token_ids"] if "prompt_token_ids" in request else [0]
            )
            if prompt_token_ids is None:
                prompt_token_ids = [0]
            sampling_params_list_override = req.sampling_params_list
            try:
                prompt = self._build_engine_core_request_from_tokens(
                    list(prompt_token_ids),
                    request_id,
                    sampling_params_list_override,
                    register=not _requires_downstream_final_stage_metadata(
                        final_stage_id,
                        self.stage_id,
                    ),
                    base_prompt=req.original_prompt,
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return
        elif stage_connector_refs:
            # Stage N > 0: fetch previous stage outputs from connectors, run pre-processor.
            sampling_params_list_override = req.sampling_params_list
            final_stage_id = req.final_stage_id
            try:
                stage_list = self._fetch_stage_inputs(stage_connector_refs, request_id)
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return

            filled = sum(1 for stage_input in stage_list if stage_input.engine_outputs)
            expected = len(self._engine_input_source or stage_connector_refs)
            if filled != expected:
                logger.warning(
                    "Stage %d: expected %d stage inputs, got %d",
                    self.stage_id,
                    expected,
                    filled,
                )

            processor_output = False
            if self._processor is not None:
                prompt = self._process_stage_inputs(stage_list, original_prompt)
                if isinstance(prompt, list) and len(prompt) == 0:
                    logger.error(
                        "Stage %d: processor returned no engine inputs",
                        self.stage_id,
                    )
                    yield {
                        "error": f"Stage {self.stage_id}: processor returned no engine inputs",
                        "finished": True,
                    }
                    return
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
                processor_output = True
            else:
                # No processor: check if the upstream output has the
                # structure needed to build an OmniEngineCoreRequest
                # (e.g. code2wav receiving token_ids from talker).
                # Otherwise fall back to passing the raw data directly.
                upstream = stage_list[-1].engine_outputs[0]
                if hasattr(upstream, "outputs") and upstream.outputs:
                    try:
                        prompt = self._build_engine_core_request_from_upstream(
                            stage_list,
                            request_id,
                            sampling_params_list_override,
                            register=not _requires_downstream_final_stage_metadata(
                                final_stage_id,
                                self.stage_id,
                            ),
                        )
                    except RuntimeError as e:
                        yield {"error": str(e), "finished": True}
                        return
                else:
                    prompt = upstream
        elif req.request_id is not None:
            # Stage 0 via router: raw request forwarded with request_id — parse it.
            final_stage_id = self._resolve_final_stage_id(request, None)
            parsed = await parse_omni_request(
                request,
                self._output_modalities,
                self._default_video_fps,
                tokenizer_getter=self.engine.get_tokenizer,
                chat_preprocessor=self._omni_chat_preprocessor,
                renderer=getattr(self.engine, "renderer", None),
            )
            await _attach_chat_prompt_token_ids(parsed, request, self.engine)
            prompt = parsed["engine_inputs"]
            original_prompt = parsed["original_prompt"]
            sampling_params_list_override = parsed["sampling_params_list"]
        else:
            # Direct frontend → stage (single-stage, no router).
            final_stage_id = self._resolve_final_stage_id(request, None)
            prompt = request

        logger.debug(
            "Stage %d: engine.generate for %s — prompt type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )

        sp = _build_sampling_params(self.stage_config, sampling_params_list_override)
        if stage_connector_refs and processor_output:
            try:
                prompt = self._build_engine_core_request_from_processor_prompt(
                    prompt,
                    request_id=request_id,
                    sampling_params_list=sp,
                    register=not _requires_downstream_final_stage_metadata(
                        final_stage_id,
                        self.stage_id,
                    ),
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return
        prompt = self._prepare_downstream_payload_prompt(
            prompt,
            request_id=request_id,
            sampling_params_list=sp,
            final_stage_id=final_stage_id,
        )
        if async_prewarm_request:
            if (
                self._async_chunk
                and final_stage_id is not None
                and final_stage_id > self.stage_id
                and not request.get(_ASYNC_DRAIN_INLINE_KEY)
            ):
                task = asyncio.create_task(
                    self._drain_nonfinal_async_request(prompt, request_id, sp)
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                await asyncio.sleep(0)
                yield {_ASYNC_PREWARM_READY_KEY: True, "finished": True}
                return
            yield {_ASYNC_PREWARM_READY_KEY: True}

        last_result = None
        intermediate_bridge_multimodal_outputs: dict[str, Any] = {}
        final_results: list[Any] = []

        try:
            async for chunk in self.engine.generate(
                prompt, request_id=request_id, sampling_params_list=sp
            ):
                chunk_multimodal_output = _chunk_multimodal_output(chunk)
                if not getattr(chunk, "finished", False):
                    _accumulate_bridge_multimodal_output_from_payload(
                        intermediate_bridge_multimodal_outputs,
                        getattr(chunk, "request_id", None),
                        chunk_multimodal_output,
                    )
                if (
                    final_stage_id is not None
                    and final_stage_id <= self.stage_id
                    and isinstance(chunk_multimodal_output, dict)
                    and chunk_multimodal_output
                ):
                    final_results.append(chunk)
                last_result = chunk
                logger.debug(
                    "Stage %d engine chunk for %s finished=%s final_output_type=%s type=%s",
                    self.stage_id,
                    request_id,
                    getattr(chunk, "finished", None),
                    getattr(chunk, "final_output_type", None),
                    type(chunk).__name__,
                )
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
        is_final_stage_for_request = (
            final_stage_id is not None and final_stage_id <= self.stage_id
        )
        if (
            connector is not None
            and not self._async_chunk
            and not is_final_stage_for_request
        ):
            try:
                ok, _, metadata = connector.put(  # type: ignore[arg-type]
                    from_s,
                    to_s,
                    request_id,
                    _prepare_connector_payload(
                        last_result,
                        bridge_states=_bridge_states_from_multimodal_outputs(
                            intermediate_bridge_multimodal_outputs
                        ),
                    ),
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

        if (
            self._async_chunk
            and self.stage_id > 0
            and connector is not None
            and not is_final_stage_for_request
        ):
            yield {"finished": True}
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
        serializable_result = (
            final_results
            if len(final_results) > 1
            else final_results[0]
            if final_results
            else last_result
        )
        shm_meta = shm_write_bytes(serialize_obj(serializable_result), name=shm_name)
        out = {"shm_meta": shm_meta, "finished": True}
        if final_stage_id is not None:
            out["final_stage_id"] = final_stage_id
        yield out

    async def _drain_nonfinal_async_request(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list | None,
    ) -> None:
        chunks = 0
        last_type = None
        logger.debug(
            "Stage %d async prewarm background start for %s prompt_type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )
        try:
            async for output in self.engine.generate(
                prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            ):
                chunks += 1
                last_type = type(output).__name__
                request_output = getattr(output, "request_output", output)
                finished = getattr(request_output, "finished", None)
                logger.debug(
                    "Stage %d async prewarm background chunk for %s count=%d output_type=%s finished=%s",
                    self.stage_id,
                    request_id,
                    chunks,
                    last_type,
                    finished,
                )
            logger.debug(
                "Stage %d async prewarm background done for %s chunks=%d last_type=%s",
                self.stage_id,
                request_id,
                chunks,
                last_type,
            )
        except Exception as e:
            logger.error(
                "Stage %d async prewarm background error for %s after chunks=%d: %s",
                self.stage_id,
                request_id,
                chunks,
                e,
                exc_info=True,
            )

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
            chat_preprocessor=self._omni_chat_preprocessor,
            renderer=getattr(self.engine, "renderer", None),
        )
        await _attach_chat_prompt_token_ids(parsed, frontend_request, self.engine)
        prompt_token_ids = await _extract_prompt_token_ids(
            parsed["engine_inputs"],
            self.engine,
        )
        if len(prompt_token_ids) < 2 and isinstance(
            frontend_request.get("messages"), list
        ):
            prompt_token_ids = await _extract_chat_prompt_token_ids(
                frontend_request,
                self.engine,
            )
        return {
            "original_prompt": parsed["original_prompt"],
            "sampling_params_list": parsed["sampling_params_list"],
            "prompt_token_ids": prompt_token_ids,
            "final_stage_id": self._resolve_final_stage_id(frontend_request, None),
        }

    def _build_engine_core_request_from_upstream(
        self,
        stage_list: list[_Proxy],
        request_id: str,
        sampling_params_list_override: dict | None,
        *,
        register: bool = True,
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
            register=register,
        )

    def _build_engine_core_request_from_tokens(
        self,
        token_ids: list[int],
        request_id: str,
        sampling_params_list_override: dict | None,
        *,
        register: bool = True,
        base_prompt: Any | None = None,
    ):
        if isinstance(base_prompt, dict):
            tokens_prompt = copy.deepcopy(base_prompt)
            tokens_prompt["prompt_token_ids"] = token_ids
            tokens_prompt["multi_modal_data"] = None
            tokens_prompt["mm_processor_kwargs"] = None
        else:
            tokens_prompt = OmniTokensPrompt(prompt_token_ids=token_ids)
        sp_list = _build_sampling_params(
            self.stage_config, sampling_params_list_override
        )
        return self._build_engine_core_request(
            tokens_prompt,
            request_id,
            sp_list,
            register=register,
        )

    def _build_engine_core_request(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list | None,
        *,
        register: bool = True,
    ):
        params = sampling_params_list[0] if sampling_params_list else None
        if params is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot build engine request without "
                "default sampling params"
            )

        kwargs = {
            "request_id": request_id,
            "prompt": prompt,
            "params": params,
        }
        if (
            "model_config"
            in inspect.signature(build_engine_core_request_from_tokens).parameters
        ):
            model_config = _engine_model_config(self.engine)
            if model_config is not None:
                kwargs["model_config"] = model_config

        request = build_engine_core_request_from_tokens(**kwargs)
        if register:
            self._register_engine_core_request(request)
        return request

    def _build_engine_core_request_from_processor_prompt(
        self,
        prompt: Any,
        *,
        request_id: str,
        sampling_params_list: list | None,
        register: bool = True,
    ) -> Any:
        """Mirror native inter-stage admission for processor outputs.

        vLLM-Omni wraps non-diffusion downstream processor outputs in an
        OmniEngineCoreRequest before submitting them to the stage engine. This
        matters for non-AR stages such as code2wav, whose model does not pass
        normal vLLM input validation from a raw OmniTokensPrompt.
        """
        if isinstance(prompt, EngineCoreRequest):
            if register:
                self._register_engine_core_request(prompt)
            return prompt
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise RuntimeError(
                    f"Stage {self.stage_id}: expected one processor output, "
                    f"got {len(prompt)}"
                )
            prompt = prompt[0]
        prompt_token_ids = _get_prompt_token_ids(prompt)
        if prompt_token_ids is not None and len(prompt_token_ids) == 0:
            raise RuntimeError(
                f"Stage {self.stage_id}: processor returned zero prompt tokens"
            )
        if prompt_token_ids is None:
            return prompt

        return self._build_engine_core_request(
            prompt,
            request_id,
            sampling_params_list,
            register=register,
        )

    def _register_engine_core_request(self, request: EngineCoreRequest) -> None:
        if request.external_req_id is None:
            request.external_req_id = request.request_id
        self.engine.engine.output_processors[0].add_request(
            request=request,
            prompt=None,
            parent_req=None,
            request_index=0,
            queue=None,
        )

    def _resolve_final_stage_id(
        self,
        request: dict,
        existing: int | None,
    ) -> int:
        if existing is not None:
            return int(existing)
        return get_final_stage_id_for_e2e(
            request.get("modalities"),
            self._output_modalities,
            self._pipeline_stage_configs,
        )

    def _prepare_downstream_payload_prompt(
        self,
        prompt: Any,
        *,
        request_id: str,
        sampling_params_list: list | None,
        final_stage_id: int | None,
    ) -> Any:
        if final_stage_id is None or final_stage_id <= self.stage_id:
            return prompt

        # vLLM-Omni's AR runner only emits inter-stage multimodal payloads
        # (hidden states, audio codes, etc.) when omni_final_stage_id points to
        # a downstream stage. Dynamo runs each stage in a one-stage AsyncOmni,
        # so the normal generate() path would stamp final_stage_id=0 and drop
        # those payloads. Prebuilding the EngineCoreRequest preserves the
        # pipeline-level final stage metadata while keeping this local
        # one-stage engine from routing to nonexistent downstream stages.
        if isinstance(prompt, EngineCoreRequest):
            prompt = _apply_omni_final_stage_metadata(prompt, final_stage_id)
            self._register_engine_core_request(prompt)
            return prompt

        inner_engine = getattr(self.engine, "engine", None)
        build_message = getattr(inner_engine, "_build_add_request_message", None)
        if not callable(build_message):
            raise RuntimeError(
                f"Stage {self.stage_id}: engine cannot prebuild request with "
                f"final_stage_id={final_stage_id}"
            )

        message = build_message(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
        )
        if isinstance(message, dict):
            return message["prompt"]
        return message.prompt

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
                    _build_streaming_context(stage_list),
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
            "parameters ('source_outputs', 'original_prompt|prompt', ...), got "
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
            if isinstance(payload_data, dict) and "engine_inputs" in payload_data:
                engine_inputs = payload_data["engine_inputs"]
                _restore_completion_output_attrs(
                    engine_inputs,
                    payload_data.get("_dynamo_completion_output_attrs"),
                )
                bridge_states = payload_data.get("_dynamo_streaming_bridge_states")
            else:
                engine_inputs = payload_data
                bridge_states = None
            _ensure_cumulative_token_ids(engine_inputs)
            stage_list[stage_k] = _Proxy(
                engine_outputs=[engine_inputs],
                bridge_states=bridge_states
                if isinstance(bridge_states, dict)
                else None,
            )
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
    use_async_chunk = stage_configs_use_async_chunk(stage_configs)
    connector_configs_path = (
        config.stage_configs_path
        if use_async_chunk
        else _ensure_stage_connectors(config.stage_configs_path, stage_configs)
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
        source_config_path=config.stage_configs_path,
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


def _requires_downstream_final_stage_metadata(
    final_stage_id: int | None,
    stage_id: int,
) -> bool:
    return final_stage_id is not None and final_stage_id > stage_id


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
                part[
                    "uuid"
                ] = f"__dynamo_prepare_{request_id}_{message_index}_{part_index}"
    return prepared


def _is_media_content_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    return part_type in _MEDIA_CONTENT_KEYS or any(
        key in part for key in _MEDIA_CONTENT_KEYS
    )


async def _attach_chat_prompt_token_ids(
    parsed: dict[str, Any],
    request: dict,
    engine: StageEngine,
) -> None:
    if not isinstance(request.get("messages"), list):
        return
    prompt_token_ids = _get_prompt_token_ids(parsed.get("engine_inputs")) or []
    chat_prompt_token_ids = await _extract_chat_prompt_token_ids(request, engine)
    if len(prompt_token_ids) < 2 or (
        _has_chatml_start(chat_prompt_token_ids)
        and not _has_chatml_start(prompt_token_ids)
    ):
        prompt_token_ids = chat_prompt_token_ids
    if len(prompt_token_ids) < 2:
        return
    for key in ("engine_inputs", "original_prompt"):
        value = parsed.get(key)
        if not isinstance(value, dict):
            wrapped = OmniTokensPrompt(prompt_token_ids=list(prompt_token_ids))
            if isinstance(value, str):
                wrapped["prompt"] = value
            parsed[key] = wrapped
            value = wrapped
        if value.get("prompt_token_ids") != list(prompt_token_ids):
            value["prompt_token_ids"] = list(prompt_token_ids)
        additional = value.setdefault("additional_information", {})
        if isinstance(additional, dict):
            ids = additional.setdefault("ids", {})
            if isinstance(ids, dict):
                ids["prompt"] = list(prompt_token_ids)
                ids["all"] = list(prompt_token_ids)


def _has_chatml_start(token_ids: list[int]) -> bool:
    return 151644 in token_ids


async def _extract_chat_prompt_token_ids(
    request: dict, engine: StageEngine
) -> list[int]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    tokenizer_result = engine.get_tokenizer()
    tokenizer = (
        await tokenizer_result
        if inspect.isawaitable(tokenizer_result)
        else tokenizer_result
    )
    if tokenizer is None:
        return _synthetic_chatml_prompt_token_ids(messages)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            token_ids = list(
                apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                or []
            )
            if len(token_ids) >= 2:
                return token_ids
        except Exception:
            logger.debug(
                "Tokenizer chat template unavailable; using ChatML token-id fallback",
                exc_info=True,
            )
    text = _render_chatml_prompt(messages)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            token_ids = list(encode(text, add_special_tokens=False))
        except TypeError:
            token_ids = list(encode(text))
        return (
            token_ids
            if len(token_ids) >= 2
            else _synthetic_chatml_prompt_token_ids(messages)
        )
    result = tokenizer(text)
    input_ids = getattr(result, "input_ids", None)
    if input_ids is None and isinstance(result, dict):
        input_ids = result.get("input_ids")
    token_ids = list(input_ids or [])
    return (
        token_ids
        if len(token_ids) >= 2
        else _synthetic_chatml_prompt_token_ids(messages)
    )


def _render_chatml_prompt(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        if role == "developer":
            role = "system"
        parts.append(
            f"<|im_start|>{role}\n{_message_content_text(message.get('content'))}<|im_end|>"
        )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _synthetic_chatml_prompt_token_ids(messages: list[Any]) -> list[int]:
    role_token_ids = {
        "system": 8948,
        "user": 872,
        "assistant": 77091,
        "developer": 8948,
    }
    ids: list[int] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        role_id = role_token_ids.get(role, 872)
        token_budget = max(
            1, len(_message_content_text(message.get("content")).split())
        )
        ids.extend([151644, role_id, 198])
        ids.extend([0] * token_budget)
        ids.extend([151645, 198])
    ids.extend([151644, 77091, 198])
    return ids


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in _TEXT_CONTENT_TYPES and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
    return "\n".join(text_parts)


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


def _prepare_connector_payload(
    engine_inputs: Any,
    bridge_states: dict[str, Any] | None = None,
) -> Any:
    """Preserve dynamic CompletionOutput attrs that Omni's msgpack codec drops."""
    _promote_request_multimodal_output(engine_inputs)
    output_attrs = _collect_completion_output_attrs(engine_inputs)
    if len(output_attrs) == 0 and not bridge_states:
        return engine_inputs
    payload = {
        "engine_inputs": engine_inputs,
        "_dynamo_completion_output_attrs": output_attrs,
    }
    if bridge_states:
        payload["_dynamo_streaming_bridge_states"] = bridge_states
    return payload


def _build_streaming_context(stage_list: list[_Proxy]) -> Any | None:
    bridge_states: dict[str, Any] = {}
    for stage_input in stage_list:
        if isinstance(stage_input.bridge_states, dict):
            _merge_bridge_states(bridge_states, stage_input.bridge_states)
    if not bridge_states:
        return None
    return SimpleNamespace(
        enabled=False,
        segment_finished=False,
        new_prompt_len_snapshot=None,
        bridge_states=bridge_states,
    )


def _merge_bridge_states(
    target: dict[str, Any],
    incoming: dict[str, Any],
) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_bridge_states(target[key], value)
        else:
            target[key] = value


def _bridge_states_from_multimodal_outputs(
    by_req: dict[str, Any],
) -> dict[str, Any] | None:
    if not by_req:
        return None
    return {"pd_prefill_multimodal_output_by_req": by_req}


def _accumulate_bridge_multimodal_output(
    by_req: dict[str, Any],
    chunk: Any,
) -> None:
    req_id = getattr(chunk, "request_id", None)
    _accumulate_bridge_multimodal_output_from_payload(
        by_req,
        req_id,
        _chunk_multimodal_output(chunk),
    )


def _accumulate_bridge_multimodal_output_from_payload(
    by_req: dict[str, Any],
    req_id: Any,
    multimodal_output: Any | None,
) -> None:
    if req_id is None:
        return
    if not isinstance(multimodal_output, dict) or not multimodal_output:
        return
    req_id = str(req_id)
    existing = by_req.get(req_id)
    by_req[req_id] = (
        multimodal_output
        if existing is None
        else _merge_multimodal_payload(existing, multimodal_output)
    )


def _chunk_multimodal_output(chunk: Any) -> Any | None:
    multimodal_output = getattr(chunk, "multimodal_output", None)
    if multimodal_output:
        return multimodal_output
    request_output = getattr(chunk, "request_output", None)
    if request_output is not None:
        multimodal_output = getattr(request_output, "multimodal_output", None)
        if multimodal_output:
            return multimodal_output
    outputs = _iter_completion_outputs(chunk)
    if len(outputs) == 1:
        return getattr(outputs[0], "multimodal_output", None)
    return None


def _merge_multimodal_payload(old: Any, new: Any) -> Any:
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            if key in merged:
                merged[key] = _merge_multimodal_payload(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
        concatenated = _concat_compatible_tensors(old, new)
        if concatenated is not None:
            return concatenated
    if isinstance(old, list) and isinstance(new, list):
        return old + new
    return new


def _concat_compatible_tensors(
    old: torch.Tensor, new: torch.Tensor
) -> torch.Tensor | None:
    if old.dim() == 0 or new.dim() == 0 or old.dim() != new.dim():
        return None
    if old.shape[1:] == new.shape[1:]:
        return torch.cat([old, new], dim=0)
    if old.shape[:-1] == new.shape[:-1]:
        return torch.cat([old, new], dim=-1)
    return None


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
        request_output = getattr(engine_inputs, "request_output", None)
        request_multimodal_output = getattr(request_output, "multimodal_output", None)
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
    source_outputs_param_name = parameter_names[0]
    prompt_param_name = parameter_names[1].lstrip("_")
    requires_mm_param_name = parameter_names[2].lstrip("_")
    return (
        source_outputs_param_name == "source_outputs"
        and prompt_param_name in {"original_prompt", "prompt"}
        and requires_mm_param_name in {"requires_mm", "requires_multimodal_data"}
    )


def _has_prompt_token_ids(prompt: Any) -> bool:
    try:
        prompt["prompt_token_ids"]
    except (KeyError, TypeError):
        return False
    return True


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

    stage_arg = _stage_config_to_dict(
        stage_config,
        stage_type,
        preserve_stage_id=use_async_chunk,
    )
    _normalize_single_stage_runtime_devices(stage_arg)
    if use_async_chunk:
        _inject_output_connectors_from_source(
            stage_arg,
            source_config_path,
            source_stage_id,
        )
    single_stage_config = {
        "async_chunk": use_async_chunk,
        "stage_args": [stage_arg],
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
        if not isinstance(downstream, dict):
            continue
        to_stage = downstream.get("stage_id")
        input_connectors = downstream.get("input_connectors") or {}
        connector_ref = input_connectors.get(f"from_stage_{source_stage_id}")
        if connector_ref is not None and to_stage is not None:
            output_connectors.setdefault(f"to_stage_{to_stage}", connector_ref)

    if output_connectors:
        stage_dict["output_connectors"] = output_connectors


def _patch_headless_stage_finalize() -> None:
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
