# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List

from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

from dynamo import prometheus_names
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.llm import ModelInput, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.args import get_served_model_name
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.output_formatter import OutputFormatter
from dynamo.vllm.omni.stage_worker import _resolve_model_type
from dynamo.vllm.omni.types import StageOutput
from dynamo.vllm.omni.utils import shm_deserialize

logger = logging.getLogger(__name__)


class OmniStageRouter:
    """Pure message broker for multi-stage omni pipelines."""

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        self.config = config
        self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)
        self.stage_clients: Dict[str, Any] = {}

        media_fs = (
            get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
        )
        self._formatter = OutputFormatter(
            model_name=get_served_model_name(config.model, config.served_model_name),
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
        if not final.shm_meta:
            yield {"error": "No SHM output from final stage", "finished": True}
            return

        # Build formatting context from the original request
        nvext = request.get("nvext") or {}
        fmt_ctx: Dict[str, Any] = {}
        if nvext.get("fps") is not None:
            fmt_ctx["fps"] = nvext["fps"]
        if request.get("response_format") is not None:
            fmt_ctx["response_format"] = request["response_format"]
        if nvext.get("speed") is not None:
            fmt_ctx["speed"] = nvext["speed"]

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

    primary_name = get_served_model_name(config.model, config.served_model_name)
    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        primary_name,
    )
    logger.info("OmniStageRouter registered at '%s'", generate_endpoint)

    try:
        await generate_endpoint.serve_endpoint(
            router.generate,
            graceful_shutdown=True,
            metrics_labels=[
                (
                    prometheus_names.labels.MODEL,
                    primary_name,
                ),
                (
                    prometheus_names.labels.MODEL_NAME,
                    primary_name,
                ),
            ],
        )
    except Exception as e:
        logger.error("OmniStageRouter endpoint failed: %s", e)
        raise
