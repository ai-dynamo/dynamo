# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List

from dynamo import prometheus_names
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.llm import ModelInput, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.output_formatter import OutputFormatter
from dynamo.vllm.omni.stage_worker import (
    _ASYNC_PREPARE_KEY,
    _ASYNC_PREWARM_KEY,
    _ASYNC_PREWARM_READY_KEY,
    _resolve_model_type,
)
from dynamo.vllm.omni.types import StageOutput
from dynamo.vllm.omni.utils import (
    load_omni_stage_configs,
    shm_deserialize,
    stage_configs_use_async_chunk,
)

try:
    from vllm_omni.distributed.omni_connectors.adapter import (
        compute_talker_prompt_ids_length,
    )
except Exception:  # pragma: no cover - depends on vLLM-Omni version
    compute_talker_prompt_ids_length = None

logger = logging.getLogger(__name__)


class OmniStageRouter:
    """Pure message broker for multi-stage omni pipelines."""

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        self.config = config
        self.stage_configs = load_omni_stage_configs(config.model, stage_configs_path)
        self._async_chunk = stage_configs_use_async_chunk(self.stage_configs)
        self.stage_clients: Dict[str, Any] = {}

        media_fs = (
            get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
        )
        self._formatter = OutputFormatter(
            model_name=config.served_model_name or config.model,
            media_fs=media_fs,
            media_http_url=config.media_output_http_url,
            default_fps=config.default_video_fps,
        )

    def set_stage_client(self, model_stage: str, client: Any) -> None:
        self.stage_clients[model_stage] = client
        logger.info("Registered stage client: %s", model_stage)

    async def generate(
        self,
        request: dict,
        context,  # noqa: ARG002 — context unused; router generates its own request_id
    ) -> AsyncGenerator[dict, None]:
        request_id = str(uuid.uuid4())
        _, request_type = parse_request_type(request, self.config.output_modalities)

        if self._async_chunk and len(self.stage_configs) > 1:
            async for chunk in self._generate_async_chunk(
                request,
                request_id,
                request_type,
            ):
                yield chunk
            return

        stage_outputs: List[StageOutput] = []
        for stage_idx, stage_cfg in enumerate(self.stage_configs):
            model_stage = _model_stage_name(stage_cfg, stage_idx)
            client = self.stage_clients.get(model_stage)
            if client is None:
                yield {
                    "error": f"No client for stage '{model_stage}'",
                    "finished": True,
                }
                return

            if stage_idx == 0:
                # This is a workaround for now to pass in the raw request to stage 0. StageRequest validates it but ignores any unknown keys, so it gets passed through.
                stage_request = {"request_id": request_id, **request}
            else:
                stage_request = stage_outputs[-1].to_next_stage_request(request_id)

            raw_stage_output = {}
            logger.info(
                "Router: stage %d request keys=%s",
                stage_idx,
                list(stage_request.keys()),
            )
            # For now, it is just one chunk output from the stage. Keeping the loop style in mind if in future we decide to stream multiple chunks from the stage.
            async for chunk in await client.round_robin(stage_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                raw_stage_output.update(data)
            stage_outputs.append(StageOutput.model_validate(raw_stage_output))

            if stage_outputs[-1].error:
                yield {"error": stage_outputs[-1].error, "finished": True}
                return

        final = stage_outputs[-1]
        async for chunk in self._format_final_output(
            final,
            request,
            request_id,
            request_type,
        ):
            yield chunk

    async def _generate_async_chunk(
        self,
        request: dict,
        request_id: str,
        request_type: RequestType,
    ) -> AsyncGenerator[dict, None]:
        stage0_cfg = self.stage_configs[0]
        stage0_client = self.stage_clients.get(_model_stage_name(stage0_cfg, 0))
        if stage0_client is None:
            yield {"error": "No client for stage 'stage0'", "finished": True}
            return

        prepare_output = await self._call_stage_raw(
            stage0_client,
            {"request_id": request_id, **request, _ASYNC_PREPARE_KEY: True},
        )
        if prepare_output.get("error"):
            yield {"error": prepare_output["error"], "finished": True}
            return

        prompt_len = _compute_async_prewarm_prompt_len(
            prepare_output.get("prompt_token_ids") or []
        )
        prewarm_request = {
            "request_id": request_id,
            "original_prompt": prepare_output.get("original_prompt"),
            "sampling_params_list": prepare_output.get("sampling_params_list"),
            "prompt_token_ids": [0] * prompt_len,
            _ASYNC_PREWARM_KEY: True,
        }

        downstream_tasks: dict[int, asyncio.Task[StageOutput]] = {}
        prewarm_ready: dict[int, asyncio.Future[str | None]] = {}
        try:
            for stage_idx, stage_cfg in enumerate(self.stage_configs[1:], start=1):
                model_stage = _model_stage_name(stage_cfg, stage_idx)
                client = self.stage_clients.get(model_stage)
                if client is None:
                    await _cancel_downstream_tasks(downstream_tasks)
                    yield {
                        "error": f"No client for stage '{model_stage}'",
                        "finished": True,
                    }
                    return
                ready_future = asyncio.get_running_loop().create_future()
                prewarm_ready[stage_idx] = ready_future
                downstream_tasks[stage_idx] = asyncio.create_task(
                    self._call_stage(
                        client,
                        dict(prewarm_request),
                        prewarm_ready=ready_future,
                    )
                )

            try:
                ready_errors = await asyncio.gather(*prewarm_ready.values())
                if ready_error := next((e for e in ready_errors if e), None):
                    await _cancel_downstream_tasks(downstream_tasks)
                    yield {"error": ready_error, "finished": True}
                    return
                stage0_output = await self._call_stage(
                    stage0_client,
                    {"request_id": request_id, **request},
                )
            except Exception as e:
                await _cancel_downstream_tasks(downstream_tasks)
                yield {"error": str(e), "finished": True}
                return
            if stage0_output.error:
                await _cancel_downstream_tasks(downstream_tasks)
                yield {"error": stage0_output.error, "finished": True}
                return
            stage0_text = _stage_output_text(stage0_output)

            final: StageOutput | None = None
            for stage_idx in range(1, len(self.stage_configs)):
                try:
                    output = await downstream_tasks[stage_idx]
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await _cancel_downstream_tasks(downstream_tasks)
                    yield {"error": str(e), "finished": True}
                    return
                if output.error:
                    await _cancel_downstream_tasks(downstream_tasks)
                    yield {"error": output.error, "finished": True}
                    return
                final = output

            if final is None:
                yield {"error": "No final stage output", "finished": True}
                return

            async for chunk in self._format_final_output(
                final,
                request,
                request_id,
                request_type,
                extra_ctx={"text": stage0_text} if stage0_text else None,
            ):
                yield chunk
        finally:
            await _cancel_downstream_tasks(downstream_tasks)

    async def _call_stage(
        self,
        client: Any,
        stage_request: dict,
        *,
        prewarm_ready: asyncio.Future[str | None] | None = None,
    ) -> StageOutput:
        return StageOutput.model_validate(
            await self._call_stage_raw(
                client,
                stage_request,
                prewarm_ready=prewarm_ready,
            )
        )

    async def _call_stage_raw(
        self,
        client: Any,
        stage_request: dict,
        *,
        prewarm_ready: asyncio.Future[str | None] | None = None,
    ) -> dict:
        raw_stage_output = {}
        try:
            async for chunk in await client.round_robin(stage_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                if data.pop(_ASYNC_PREWARM_READY_KEY, False):
                    if prewarm_ready is not None and not prewarm_ready.done():
                        prewarm_ready.set_result(None)
                raw_stage_output.update(data)
        except Exception as e:
            if prewarm_ready is not None and not prewarm_ready.done():
                prewarm_ready.set_result(str(e))
            raise
        if prewarm_ready is not None and not prewarm_ready.done():
            error = raw_stage_output.get("error")
            if error:
                prewarm_ready.set_result(str(error))
            else:
                prewarm_ready.set_result("Stage prewarm completed without ready ack")
        return raw_stage_output

    async def _format_final_output(
        self,
        final: StageOutput,
        request: dict,
        request_id: str,
        request_type: RequestType,
        extra_ctx: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        if not final.shm_meta:
            yield {"error": "No SHM output from final stage", "finished": True}
            return

        fmt_ctx = _format_context(request, request_type)
        if extra_ctx:
            fmt_ctx.update(extra_ctx)

        async for chunk in self._format_output(
            final, request_id, request_type, fmt_ctx
        ):
            yield chunk

    async def _format_output(
        self,
        stage_output: StageOutput,
        request_id: str,
        request_type: RequestType,
        ctx: dict,
    ) -> AsyncGenerator[dict, None]:
        """Read OmniRequestOutput from SHM and format via OutputFormatter."""
        shm_meta = stage_output.shm_meta
        if not shm_meta:
            logger.warning("Router: no shm_meta in stage output")
            return

        result = shm_deserialize(shm_meta)
        chunk = await self._formatter.format(
            result, request_id, request_type=request_type, **ctx
        )
        if chunk:
            yield chunk
        else:
            final_output_type = getattr(result, "final_output_type", "unknown")
            logger.warning(
                "Router: formatter returned None, final_output_type=%s",
                final_output_type,
            )
            yield {
                "error": f"Formatter returned no output for type '{final_output_type}'",
                "finished": True,
            }


async def init_omni_stage_router(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
) -> None:
    """Initialize OmniStageRouter as a Dynamo backend endpoint."""
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint or 'generate'}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    router = OmniStageRouter(config, config.stage_configs_path)  # type: ignore[arg-type]

    setup_metrics_collection(config, generate_endpoint, logger)

    # Discover stage endpoints
    for stage_cfg in router.stage_configs:
        model_stage = _model_stage_name(stage_cfg, stage_cfg.stage_id)
        client = await runtime.endpoint(
            f"{config.namespace}.{model_stage}.generate"
        ).client()
        await client.wait_for_instances()
        router.set_stage_client(model_stage, client)

    final_cfg = router.stage_configs[-1]
    final_output_type = getattr(final_cfg, "final_output_type", "image")
    model_type = get_output_modalities(config.output_modalities, config.model)
    if model_type is None:
        model_type = _resolve_model_type(final_output_type)

    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
    )
    logger.info("OmniStageRouter registered at '%s'", generate_endpoint)

    try:
        await generate_endpoint.serve_endpoint(
            router.generate,
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
        )
    except Exception as e:
        logger.error("OmniStageRouter endpoint failed: %s", e)
        raise


def _model_stage_name(stage_cfg: Any, stage_idx: int) -> str:
    engine_args = getattr(stage_cfg, "engine_args", None)
    return getattr(engine_args, "model_stage", f"stage{stage_idx}")


def _compute_async_prewarm_prompt_len(prompt_token_ids: list[int]) -> int:
    if compute_talker_prompt_ids_length is not None and prompt_token_ids:
        try:
            return max(1, int(compute_talker_prompt_ids_length(prompt_token_ids)))
        except Exception:
            logger.debug("Failed to compute Qwen3 talker prompt length", exc_info=True)
    return max(1, len(prompt_token_ids), 1)


async def _cancel_downstream_tasks(tasks: dict[int, asyncio.Task[StageOutput]]) -> None:
    if not tasks:
        return
    for task in tasks.values():
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)


def _stage_output_text(stage_output: StageOutput) -> str:
    if not stage_output.shm_meta:
        return ""
    try:
        return _extract_result_text(shm_deserialize(stage_output.shm_meta))
    except Exception:
        logger.debug(
            "Failed to read text output from async chunk stage 0", exc_info=True
        )
        return ""


def _extract_result_text(result: Any) -> str:
    request_output = getattr(result, "request_output", None) or result
    outputs = getattr(request_output, "outputs", None) or []
    if not outputs:
        return ""
    text = getattr(outputs[0], "text", "")
    return text if isinstance(text, str) else ""


def _format_context(request: dict, request_type: RequestType) -> Dict[str, Any]:
    nvext = request.get("nvext") or {}
    fmt_ctx: Dict[str, Any] = {}
    if nvext.get("fps") is not None:
        fmt_ctx["fps"] = nvext["fps"]
    if nvext.get("speed") is not None:
        fmt_ctx["speed"] = nvext["speed"]
    response_format = (
        request.get("data_source")
        if request_type == RequestType.AUDIO_GENERATION
        else request.get("response_format")
    )
    output_format = (
        request.get("response_format")
        if request_type == RequestType.AUDIO_GENERATION
        else request.get("output_format")
    )
    if response_format is not None:
        fmt_ctx["response_format"] = response_format
    if output_format is not None:
        fmt_ctx["output_format"] = output_format
    return fmt_ctx
