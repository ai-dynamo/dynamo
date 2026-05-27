# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List

import torch
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs

from dynamo import prometheus_names
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.llm import ModelInput, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.connectors import register_dynamoomni_nixl_connector
from dynamo.vllm.omni.output_formatter import OutputFormatter
from dynamo.vllm.omni.stage_worker import (
    _connector_key,
    _ensure_stage_connectors,
    _resolve_model_type,
    _restore_completion_output_attrs,
    _restore_stage_to_router_tensor_payload,
    _uses_nixl_connector,
)
from dynamo.vllm.omni.types import StageOutput
from dynamo.vllm.omni.utils import (
    coerce_token_ids_to_list,
    ensure_awaited,
    is_tensor_payload,
    shm_deserialize,
    unwrap_connector_payload,
)

logger = logging.getLogger(__name__)


class OmniStageRouter:
    """Pure message broker for multi-stage omni pipelines."""

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        self.config = config
        (
            resolved_stage_configs_path,
            self.stage_configs,
        ) = load_and_resolve_stage_configs(
            config.model,
            stage_configs_path,
            kwargs={},
        )
        self.stage_clients: Dict[str, Any] = {}

        # Initialize connectors so the router can fetch final-stage output
        # via connector.get() instead of SHM -- enabling multi-node deployments.
        connector_configs_path = _ensure_stage_connectors(
            resolved_stage_configs_path,
            self.stage_configs,
        )
        # Only register NixlConnector if it's actually used in stage configs
        if _uses_nixl_connector(connector_configs_path, self.stage_configs):
            try:
                register_dynamoomni_nixl_connector()
            except Exception as e:
                logger.error("Router: failed to register NixlConnector: %s", e)
                raise

        _, self.connectors = initialize_orchestrator_connectors(connector_configs_path)  # type: ignore[arg-type]
        logger.info("Router: initialized %d connector(s)", len(self.connectors))

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

        stage_outputs: List[StageOutput] = []
        for stage_idx, stage_cfg in enumerate(self.stage_configs):
            model_stage = getattr(
                stage_cfg.engine_args, "model_stage", f"stage{stage_idx}"
            )
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
        # Accept either connector-based output (multi-node) or SHM (single-node legacy).
        # Connector path: final stage wrote via connector.put(to_stage="router") and
        # returned stage_connector_refs[last_stage_id] = metadata.
        final_stage_id = str(len(self.stage_configs) - 1)
        has_connector_output = (
            final.stage_connector_refs is not None
            and final_stage_id in final.stage_connector_refs
            and self.connectors.get(_connector_key(int(final_stage_id), "router"))
            is not None
        )
        if not has_connector_output and not final.shm_meta:
            yield {
                "error": "No output from final stage (no connector ref and no SHM)",
                "finished": True,
            }
            return

        # Build formatting context from the original request
        nvext = request.get("nvext") or {}
        fmt_ctx: Dict[str, Any] = {}
        if nvext.get("fps") is not None:
            fmt_ctx["fps"] = nvext["fps"]
        if nvext.get("speed") is not None:
            fmt_ctx["speed"] = nvext["speed"]
        # If the request type is AUDIO_GENERATION,
        # we need to normalize the data_source and response_format to
        # align with other modalities.
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

        async for chunk in self._format_output(
            final,
            request_id,
            request_type,
            fmt_ctx,
            final_stage_id=len(self.stage_configs) - 1,
        ):
            yield chunk

    async def _format_output(
        self,
        stage_output: StageOutput,
        request_id: str,
        request_type: RequestType,
        ctx: dict,
        final_stage_id: int = 0,
    ) -> AsyncGenerator[dict, None]:
        """Read OmniRequestOutput from connector (multi-node) or SHM (single-node) and format."""
        # --- Connector path (multi-node: router and final stage on different machines) ---
        router_connector = self.connectors.get(_connector_key(final_stage_id, "router"))
        if router_connector is not None and stage_output.stage_connector_refs:
            meta_k = stage_output.stage_connector_refs.get(str(final_stage_id))
            if meta_k is not None:
                try:
                    get_result = await ensure_awaited(
                        router_connector.get(
                            str(final_stage_id),
                            "router",
                            request_id,
                            metadata=meta_k,
                        )
                    )
                    payload_data = unwrap_connector_payload(get_result)

                    if is_tensor_payload(payload_data):
                        payload_data = _restore_stage_to_router_tensor_payload(
                            payload_data
                        )

                    if (
                        isinstance(payload_data, dict)
                        and payload_data.get("_dynamo_payload_type")
                        == "stage_to_router_output"
                    ):
                        result = _payload_dict_to_result(payload_data)
                    elif (
                        isinstance(payload_data, dict)
                        and "engine_inputs" in payload_data
                    ):
                        result = payload_data["engine_inputs"]
                        _restore_completion_output_attrs(
                            result,
                            payload_data.get("_dynamo_completion_output_attrs"),
                        )
                    else:
                        result = payload_data
                    logger.debug(
                        "Router: fetched final output via connector for %s", request_id
                    )
                except Exception as e:
                    logger.error(
                        "Router: connector.get() failed for %s: %s", request_id, e
                    )
                    yield {
                        "error": f"Router connector.get() failed: {e}",
                        "finished": True,
                    }
                    return
            else:
                logger.warning(
                    "Router: no connector ref for final stage %d", final_stage_id
                )
                yield {"error": "No connector ref for final stage", "finished": True}
                return
        else:
            # --- SHM fallback (single-node: router and final stage on same machine) ---
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


def _payload_dict_to_result(payload_data: dict[str, Any]) -> Any:
    kind = payload_data.get("kind")
    if kind == "images":
        return SimpleNamespace(
            images=_coerce_images(payload_data.get("images")),
            final_output_type="image",
        )

    if kind == "text":
        outputs_raw = payload_data.get("outputs")
        if not isinstance(outputs_raw, list):
            raise RuntimeError("Invalid stage->router payload: missing text outputs")

        outputs = [
            SimpleNamespace(
                token_ids=coerce_token_ids_to_list(item.get("token_ids")),
                text=item.get("text", ""),
                finish_reason=item.get("finish_reason"),
            )
            for item in outputs_raw
            if isinstance(item, dict)
        ]
        request_output = SimpleNamespace(outputs=outputs)
        result = SimpleNamespace(
            request_output=request_output,
            final_output_type="text",
        )
        if (multimodal_output := payload_data.get("multimodal_output")) is not None:
            request_output.multimodal_output = multimodal_output
        return result

    result = SimpleNamespace(final_output_type="unknown")
    if (images := payload_data.get("images")) is not None:
        result.images = _coerce_images(images)
        result.final_output_type = "image"
        return result

    multimodal_output = payload_data.get("multimodal_output")
    if isinstance(multimodal_output, dict):
        if "audio" in multimodal_output or "sr" in multimodal_output:
            result.multimodal_output = multimodal_output
            result.final_output_type = "audio"
            return result
        if "images" in multimodal_output:
            result.images = _coerce_images(multimodal_output.get("images"))
            result.final_output_type = "image"
            return result
        if "model_outputs" in multimodal_output:
            result.images = _coerce_images(multimodal_output.get("model_outputs"))
            result.multimodal_output = multimodal_output
            result.final_output_type = "image"
            return result

    if kind == "audio":
        if multimodal_output is not None:
            result.multimodal_output = multimodal_output
            result.final_output_type = "audio"
    return result


def _coerce_images(images: Any) -> Any:
    if images is None:
        return []
    if isinstance(images, torch.Tensor):
        return images.detach().cpu().tolist()
    return images


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
        model_stage = getattr(
            stage_cfg.engine_args, "model_stage", f"stage{stage_cfg.stage_id}"
        )
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
        # OmniStageRouter is the user-visible front for an internal
        # multi-stage pipeline; the per-stage workers are private. From
        # the frontend's topology view, the router serves end-to-end as
        # Aggregated with no peer dependencies.
        worker_type=WorkerType.Aggregated,
        needs=[],
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
