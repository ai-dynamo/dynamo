# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import asyncio
import importlib
import inspect
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, AsyncGenerator, cast

import yaml
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.entrypoints.utils import load_stage_configs_from_model
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

_DYNAMO_PIPELINE_FINAL_STAGE_KEY = "_dynamo_pipeline_final_stage_id"
_DYNAMO_NATIVE_ASYNC_REF_KEY = "_dynamo_native_async"
_DYNAMO_NATIVE_ASYNC_PROMPT_IDS_KEY = "prompt_token_ids"


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

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )
        self._native_async = _stage_async_chunk_enabled(stage_config)

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()
        original_prompt = req.original_prompt
        final_stage_id = req.final_stage_id
        stage_text_output = req.stage_text_output
        # JSON sends dict keys as strings; normalize to int for stage_connector_refs.
        stage_connector_refs = _int_keyed(req.stage_connector_refs)

        # --- Resolve engine inputs ---
        sampling_params_list_override: dict | None = None
        if (
            stage_connector_refs
            and self._native_async
            and _native_async_input_ref(stage_connector_refs, self.stage_id) is not None
        ):
            # Native vLLM-Omni async-chunk stages receive real tensors through
            # their connector on the model-runner thread. Dynamo only needs to
            # submit a placeholder request with the same external request id.
            sampling_params_list_override = req.sampling_params_list
            prompt = self._build_native_async_request(
                stage_connector_refs,
                request_id,
                sampling_params_list_override,
            )
        elif stage_connector_refs:
            # Stage N > 0: fetch previous stage outputs from connectors, run pre-processor.
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
            parsed = await parse_omni_request(
                request,
                self._output_modalities,
                self._default_video_fps,
                tokenizer_getter=self.engine.get_tokenizer,
                engine=self.engine,
            )
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

        prompt = _attach_pipeline_final_stage(prompt, final_stage_id)
        output_modalities = _request_output_modalities(request, original_prompt)
        sp = _build_sampling_params(self.stage_config, sampling_params_list_override)
        last_result = None

        try:
            async for chunk in self.engine.generate(
                prompt,
                request_id=request_id,
                sampling_params_list=sp,
                output_modalities=output_modalities,
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
        # Check for a downstream connector first, regardless of final_output.
        # In vllm-omni's native mode, multiple stages can set final_output=True
        # (meaning "produces user-visible output"). In Dynamo's disaggregated
        # mode the actual pipeline topology — connector edges from the YAML —
        # determines whether output should go to a connector or to SHM.
        from_s, to_s = _connector_key(self.stage_id, self.stage_id + 1)
        connector = self.connectors.get((from_s, to_s))
        if connector is not None and (
            final_stage_id is None or self.stage_id < final_stage_id
        ):
            connector_result = last_result
            stage_text_output = stage_text_output or _extract_text_output(
                connector_result
            )
            if self._native_async:
                native_out: dict = {
                    "original_prompt": original_prompt,
                    "stage_connector_refs": {
                        **{str(k): v for k, v in stage_connector_refs.items()},
                        str(self.stage_id): _native_async_output_ref(
                            connector_result, stage_connector_refs
                        ),
                    },
                    "finished": True,
                }
                if stage_text_output:
                    native_out["stage_text_output"] = stage_text_output
                if sampling_params_list_override is not None:
                    native_out["sampling_params_list"] = sampling_params_list_override
                yield native_out
                return

            try:
                ok, _, metadata = connector.put(  # type: ignore[arg-type]
                    from_s,
                    to_s,
                    request_id,
                    _prepare_connector_payload(connector_result),
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
            connector_out: dict = {
                "original_prompt": original_prompt,
                "stage_connector_refs": {
                    **{str(k): v for k, v in stage_connector_refs.items()},
                    str(self.stage_id): metadata,
                },
                "finished": True,
            }
            if stage_text_output:
                connector_out["stage_text_output"] = stage_text_output
            if sampling_params_list_override is not None:
                connector_out["sampling_params_list"] = sampling_params_list_override
            yield connector_out
            return

        # Final stage → router: write output to shared memory and return the SHM handle.
        # The router reads it back via shm_deserialize() to format the response.
        #
        # NOTE: This is a single-node-only workaround — SHM requires the final stage
        # worker and the router to reside on the same machine. A proper multi-node
        # solution would use a connector edge (like inter-stage connectors) instead.
        # Tracked in TODO: shm_meta should be replaced by a YAML-configured connector edge.
        stage_text_output = stage_text_output or _extract_text_output(last_result)
        shm_meta = shm_write_bytes(serialize_obj(last_result), name=request_id)
        out = {"shm_meta": shm_meta, "finished": True}
        if stage_text_output:
            out["stage_text_output"] = stage_text_output
        yield out

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

        tokens_prompt = OmniTokensPrompt(prompt_token_ids=list(token_ids))
        sp_list = _build_sampling_params(
            self.stage_config, sampling_params_list_override
        )
        params = sp_list[0] if sp_list else None
        prompt = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=tokens_prompt,
            params=params,
        )
        # vLLM-Omni registers pre-built EngineCoreRequests through
        # StagePool.submit_initial(), but vLLM still requires an external ID
        # on the request before it reaches the output processor.
        prompt.external_req_id = prompt.request_id
        return prompt

    def _build_native_async_request(
        self,
        stage_connector_refs: dict[int, Any],
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        ref = _native_async_input_ref(stage_connector_refs, self.stage_id)
        prompt_token_ids = (
            ref.get(_DYNAMO_NATIVE_ASYNC_PROMPT_IDS_KEY, [])
            if isinstance(ref, dict)
            else []
        )
        prompt_len = _native_async_placeholder_len(prompt_token_ids)
        tokens_prompt = OmniTokensPrompt(prompt_token_ids=[0] * prompt_len)
        sp_list = _build_sampling_params(
            self.stage_config, sampling_params_list_override
        )
        params = sp_list[0] if sp_list else None
        prompt = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=tokens_prompt,
            params=params,
        )
        prompt.external_req_id = request_id
        return prompt

    def _process_stage_inputs(self, stage_list: list[_Proxy], original_prompt: Any):
        """Call vLLM-Omni stage processors with the v0.20 transition API."""
        if self._processor is None:
            raise RuntimeError(f"Stage {self.stage_id}: no processor configured")

        signature = inspect.signature(self._processor)
        parameter_names = list(signature.parameters)

        # ming_flash_omni still uses the stage-list transition signature in
        # vLLM-Omni v0.20. Most other processors use direct source_outputs.
        if parameter_names[:2] == ["stage_list", "engine_input_source"]:
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
    stage_configs = load_stage_configs_from_model(
        config.model, deploy_config_path=config.stage_configs_path
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

    transfer_config, connectors = initialize_orchestrator_connectors(  # type: ignore[arg-type]
        config.stage_configs_path
    )

    engine = _create_engine(config.model, my_config, stage_type, transfer_config)
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    # Connectors for inter-stage output transfer — type determined by YAML config
    # (SharedMemoryConnector, MooncakeConnector, etc.)

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


def _request_output_modalities(request: dict, original_prompt: Any) -> list[str] | None:
    """Resolve requested output modalities from the raw request or original prompt."""
    modalities = request.get("modalities")
    if modalities is None and isinstance(original_prompt, dict):
        modalities = original_prompt.get("modalities")
    return modalities if isinstance(modalities, list) else None


def _stage_async_chunk_enabled(stage_config: Any) -> bool:
    engine_args = getattr(stage_config, "engine_args", None)
    return bool(_config_value(engine_args, "async_chunk", False))


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except TypeError:
            pass
    return getattr(config, name, default)


def _native_async_input_ref(
    stage_connector_refs: dict[int, Any], stage_id: int
) -> dict[str, Any] | None:
    ref = stage_connector_refs.get(stage_id - 1)
    if isinstance(ref, dict) and ref.get(_DYNAMO_NATIVE_ASYNC_REF_KEY):
        return ref
    return None


def _native_async_output_ref(
    engine_outputs: Any, stage_connector_refs: dict[int, Any]
) -> dict[str, Any]:
    prompt_token_ids = _native_async_prompt_token_ids(
        engine_outputs, stage_connector_refs
    )
    return {
        _DYNAMO_NATIVE_ASYNC_REF_KEY: True,
        _DYNAMO_NATIVE_ASYNC_PROMPT_IDS_KEY: prompt_token_ids,
    }


def _native_async_prompt_token_ids(
    engine_outputs: Any, stage_connector_refs: dict[int, Any]
) -> list[int]:
    for ref in stage_connector_refs.values():
        if isinstance(ref, dict) and ref.get(_DYNAMO_NATIVE_ASYNC_REF_KEY):
            prompt_token_ids = ref.get(_DYNAMO_NATIVE_ASYNC_PROMPT_IDS_KEY)
            if isinstance(prompt_token_ids, list):
                return list(prompt_token_ids)

    prompt_token_ids = getattr(engine_outputs, "prompt_token_ids", None)
    if prompt_token_ids is None:
        return []
    return list(prompt_token_ids)


def _native_async_placeholder_len(prompt_token_ids: list[int]) -> int:
    try:
        from vllm_omni.distributed.omni_connectors.adapter import (
            compute_talker_prompt_ids_length,
        )

        return max(1, int(compute_talker_prompt_ids_length(prompt_token_ids)))
    except Exception:
        logger.debug("Could not compute native async placeholder length", exc_info=True)
        return max(1, len(prompt_token_ids))


def _attach_pipeline_final_stage(prompt: Any, final_stage_id: int | None) -> Any:
    """Attach Dynamo's pipeline final stage without changing local Omni routing.

    Each Dynamo worker runs a single-stage AsyncOmni instance, whose local
    orchestrator must still finish at stage 0. vLLM-Omni also uses
    omni_final_stage_id in request metadata to decide whether AR stages should
    emit downstream multimodal payloads. This private marker lets the patched
    metadata helper preserve the pipeline-level final stage for that purpose.
    """
    if final_stage_id is None:
        return prompt
    if isinstance(prompt, dict):
        additional_information = prompt.setdefault("additional_information", {})
        if isinstance(additional_information, dict):
            additional_information[_DYNAMO_PIPELINE_FINAL_STAGE_KEY] = final_stage_id
        return prompt

    payload = getattr(prompt, "additional_information", None)
    if payload is None and not hasattr(prompt, "request_id"):
        return prompt

    async_omni_engine = importlib.import_module("vllm_omni.engine.async_omni_engine")
    try:
        info = (
            async_omni_engine.deserialize_additional_information(payload)
            if payload is not None
            else {}
        )
        info["omni_final_stage_id"] = int(final_stage_id)
        serialized = async_omni_engine.serialize_additional_information(info)
        try:
            prompt.additional_information = serialized
            return prompt
        except AttributeError:
            return async_omni_engine.OmniEngineCoreRequest.from_request(
                prompt,
                additional_information=serialized,
            )
    except Exception:
        logger.debug("Could not attach Omni final stage metadata", exc_info=True)
        return prompt


def _patch_omni_final_stage_metadata() -> None:
    """Preserve Dynamo's pipeline final stage in vLLM-Omni request metadata."""
    async_omni_engine = cast(
        Any, importlib.import_module("vllm_omni.engine.async_omni_engine")
    )

    if getattr(async_omni_engine, "_dynamo_final_stage_patch", False):
        return

    original_apply = async_omni_engine._apply_omni_final_stage_metadata

    def patched_apply(request, final_stage_id):
        result = original_apply(request, final_stage_id)
        payload = getattr(result, "additional_information", None)
        if payload is None:
            return result

        info = async_omni_engine.deserialize_additional_information(payload)
        pipeline_final_stage_id = info.pop(_DYNAMO_PIPELINE_FINAL_STAGE_KEY, None)
        if pipeline_final_stage_id is None:
            return result

        try:
            info["omni_final_stage_id"] = int(pipeline_final_stage_id)
        except (TypeError, ValueError):
            return result

        return async_omni_engine.OmniEngineCoreRequest.from_request(
            result,
            additional_information=async_omni_engine.serialize_additional_information(
                info
            ),
        )

    async_omni_engine._apply_omni_final_stage_metadata = patched_apply
    async_omni_engine._dynamo_final_stage_patch = True


def _load_processor(func_path: str | None) -> Any:
    """Load a processor function from a dotted module path, or return None."""
    if not func_path:
        return None
    module_path, func_name = func_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), func_name)


def _prepare_connector_payload(engine_inputs: Any) -> Any:
    """Preserve dynamic CompletionOutput attrs that Omni's msgpack codec drops."""
    _promote_request_multimodal_output(engine_inputs)
    output_attrs = _collect_completion_output_attrs(engine_inputs)
    if not output_attrs:
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
    return output_attrs if any(output_attrs) else []


def _promote_request_multimodal_output(engine_inputs: Any) -> None:
    """Expose request-level multimodal payloads on the sole completion output.

    Some Omni outputs carry multimodal payloads on the request wrapper while
    v0.20 stage processors read them from ``outputs[0].multimodal_output``.
    Dynamo currently handles one completion per staged request, so promote the
    request-level payload when the completion does not already carry one.
    """
    request_multimodal_output = getattr(engine_inputs, "multimodal_output", None)
    if not request_multimodal_output:
        return

    outputs = _iter_completion_outputs(engine_inputs)
    if len(outputs) != 1:
        return

    completion = outputs[0]
    if not getattr(completion, "multimodal_output", None):
        setattr(completion, "multimodal_output", request_multimodal_output)


def _restore_completion_output_attrs(
    engine_inputs: Any, output_attrs: Any | None
) -> None:
    if not isinstance(output_attrs, list):
        return
    for output, attrs in zip(_iter_completion_outputs(engine_inputs), output_attrs):
        if not isinstance(attrs, dict):
            continue
        if "cumulative_token_ids" in attrs:
            setattr(output, "cumulative_token_ids", list(attrs["cumulative_token_ids"]))
        if "multimodal_output" in attrs:
            setattr(output, "multimodal_output", attrs["multimodal_output"])


def _iter_completion_outputs(engine_inputs: Any):
    outputs = getattr(engine_inputs, "outputs", None)
    if outputs is None:
        request_output = getattr(engine_inputs, "request_output", None)
        outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return []
    return list(outputs)


def _extract_text_output(engine_inputs: Any) -> str | None:
    for output in _iter_completion_outputs(engine_inputs):
        text = getattr(output, "text", None)
        if isinstance(text, str) and text:
            return text
    return None


def _create_engine(
    model: str, stage_config: Any, stage_type: str, transfer_config: Any | None = None
) -> StageEngine:
    """Create AsyncOmni with a single-stage YAML."""
    _patch_omni_final_stage_metadata()
    native_async = _stage_async_chunk_enabled(stage_config)

    single_stage_config = {
        "async_chunk": native_async,
        "stage_args": [
            _stage_config_to_dict(
                stage_config,
                stage_type,
                native_async=native_async,
                transfer_config=transfer_config,
            )
        ],
        "runtime": {"edges": []},
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
    *,
    native_async: bool = False,
    transfer_config: Any | None = None,
) -> dict:
    """Convert a parsed stage config to a single-stage YAML dict."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    def _to_plain(obj: Any) -> Any:
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        return obj

    result: dict = {
        "stage_id": int(getattr(stage_config, "stage_id", 0)) if native_async else 0,
        "stage_type": stage_type,
        "engine_args": _single_stage_engine_args(stage_config.engine_args),
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }

    for key in ("default_sampling_params", "is_comprehension"):
        val = getattr(stage_config, key, None)
        if val is not None:
            result[key] = _to_plain(val)

    engine_input_source = getattr(stage_config, "engine_input_source", None)
    if engine_input_source is not None:
        result["engine_input_source"] = _to_plain(engine_input_source)

    if native_async:
        _attach_native_async_connectors(result, stage_config, transfer_config)

    runtime = getattr(stage_config, "runtime", None)
    if runtime is not None:
        rt = _to_plain(runtime)
        rt["devices"] = "0"
        result["runtime"] = rt

    return result


def _attach_native_async_connectors(
    stage_config_dict: dict, stage_config: Any, transfer_config: Any | None
) -> None:
    connectors = getattr(transfer_config, "connectors", None)
    if not connectors:
        return

    stage_id = str(getattr(stage_config, "stage_id", stage_config_dict["stage_id"]))
    input_connectors: dict[str, Any] = {}
    output_connectors: dict[str, Any] = {}

    for (from_stage, to_stage), spec in connectors.items():
        connector = {"name": spec.name, "extra": dict(spec.extra or {})}
        if str(to_stage) == stage_id:
            input_connectors[f"from_stage_{from_stage}"] = connector
        if str(from_stage) == stage_id:
            output_connectors[f"to_stage_{to_stage}"] = connector

    if input_connectors:
        stage_config_dict["input_connectors"] = input_connectors
    if output_connectors:
        stage_config_dict["output_connectors"] = output_connectors


def _single_stage_engine_args(engine_args: Any) -> Any:
    """Build engine args for Dynamo's one-stage AsyncOmni wrapper."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    if OmegaConf.is_config(engine_args):
        result = OmegaConf.to_container(engine_args, resolve=True)
    elif hasattr(engine_args, "__dict__"):
        result = dict(vars(engine_args))
    else:
        result = engine_args

    if isinstance(result, dict):
        result = dict(result)

    return result


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
