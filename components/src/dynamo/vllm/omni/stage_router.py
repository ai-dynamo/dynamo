# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import dataclasses
import importlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict

from dynamo import prometheus_names
from dynamo.llm import ModelInput, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import StageConnector

logger = logging.getLogger(__name__)


def _shm_deserialize(shm_meta: dict) -> Any:
    """Read and deserialize an OmniRequestOutput from shared memory."""
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
    from vllm_omni.entrypoints.stage_utils import shm_read_bytes

    return OmniSerializer.deserialize(shm_read_bytes(shm_meta))


def _parse_engine_inputs(request: dict, request_type: Any, config: "OmniConfig") -> dict:
    """Convert a raw frontend request into engine_inputs + sampling_params_list.

    Passing the raw request dict as prompt causes the diffusion engine to use
    default values (e.g. num_frames=1). This extracts num_frames, size, etc.
    from nvext and builds proper OmniDiffusionSamplingParams.

    Returns a dict with keys:
      engine_inputs:        OmniTextPrompt-compatible dict (prompt text)
      sampling_params_list: [OmniDiffusionSamplingParams dict] or None
    """
    from dynamo.common.utils.output_modalities import RequestType

    if request_type == RequestType.VIDEO_GENERATION:
        from dynamo.common.utils.video_utils import compute_num_frames, parse_size
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

        nvext = request.get("nvext") or {}
        width, height = parse_size(request.get("size", "832x480"))
        num_frames = compute_num_frames(
            num_frames=nvext.get("num_frames"),
            fps=nvext.get("fps"),
            default_fps=config.default_video_fps,
        )
        sp = OmniDiffusionSamplingParams(height=height, width=width, num_frames=num_frames)
        if nvext.get("num_inference_steps") is not None:
            sp.num_inference_steps = nvext["num_inference_steps"]
        if nvext.get("guidance_scale") is not None:
            sp.guidance_scale = nvext["guidance_scale"]
        if nvext.get("seed") is not None:
            sp.seed = nvext["seed"]

        return {
            "engine_inputs": OmniTextPrompt(prompt=request.get("prompt", "")),
            "sampling_params_list": [dataclasses.asdict(sp)],
        }

    if request_type == RequestType.IMAGE_GENERATION:
        from dynamo.common.utils.video_utils import parse_size
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

        nvext = request.get("nvext") or {}
        width, height = parse_size(request.get("size", "1024x1024"), default_w=1024, default_h=1024)
        sp = OmniDiffusionSamplingParams(height=height, width=width)
        if nvext.get("num_inference_steps") is not None:
            sp.num_inference_steps = nvext["num_inference_steps"]

        return {
            "engine_inputs": OmniTextPrompt(prompt=request.get("prompt", "")),
            "sampling_params_list": [dataclasses.asdict(sp)],
        }

    # Chat / text: pass prompt text, no diffusion params
    messages = request.get("messages", [])
    text = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        request.get("prompt", ""),
    )
    return {"engine_inputs": text, "sampling_params_list": None}


@dataclass
class _Proxy:
    """Satisfies stage_list[i].engine_outputs access in processor functions."""
    engine_outputs: Any = None


class OmniStageRouter:
    """Orchestrates a multi-stage omni pipeline.

    Registered as the backend endpoint the frontend talks to. Owns:
    - Stage orchestration (call stages via Dynamo, connector transfer)
    - Output post-processing (convert OmniRequestOutput → frontend response)

    Stage workers return raw OmniRequestOutput. The router formats them.
    """

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
        from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

        self.config = config
        self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)
        _, self.connectors = initialize_orchestrator_connectors(stage_configs_path)

        self.processors: Dict[int, Any] = {}
        for cfg in self.stage_configs:
            func_path = getattr(cfg, "custom_process_input_func", None)
            if func_path:
                module_path, func_name = func_path.rsplit(".", 1)
                self.processors[cfg.stage_id] = getattr(
                    importlib.import_module(module_path), func_name
                )
                logger.info("Loaded processor for stage %d: %s", cfg.stage_id, func_path)

        self.stage_clients: Dict[str, Any] = {}

        # Media output config for response formatting
        from dynamo.common.storage import get_fs
        self._media_output_fs = (
            get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
        )
        self._media_output_http_url = config.media_output_http_url

    def set_stage_client(self, model_stage: str, client: Any) -> None:
        self.stage_clients[model_stage] = client
        logger.info("Registered stage client: %s", model_stage)

    async def generate(
        self, request: dict, context  # noqa: ARG002 — context unused; router generates its own request_id
    ) -> AsyncGenerator[dict, None]:
        from dynamo.common.utils.output_modalities import parse_request_type

        request_id = str(uuid.uuid4())
        _, request_type = parse_request_type(request, self.config.output_modalities)

        # Parse frontend request into proper engine inputs so num_frames,
        # num_inference_steps etc. reach the diffusion engine correctly.
        # Passing the raw dict as prompt causes the engine to use defaults (1 frame).
        engine_inputs = _parse_engine_inputs(request, request_type, self.config)
        proxies = [_Proxy() for _ in self.stage_configs]

        # --- Call each stage in order ---
        raw: Dict[str, Any] = {}
        stage_result: Any = None
        try:
            for i, stage_cfg in enumerate(self.stage_configs):
                model_stage = getattr(stage_cfg.engine_args, "model_stage", f"stage{i}")
                client = self.stage_clients.get(model_stage)
                if client is None:
                    yield {"error": f"No client for stage '{model_stage}'", "finished": True}
                    return

                if i == 0:
                    stage_request = {**engine_inputs, "request_id": request_id}
                else:
                    # Run processor if defined
                    proxies[i - 1].engine_outputs = stage_result
                    engine_input_source = getattr(stage_cfg, "engine_input_source", [i - 1])
                    requires_mm = getattr(stage_cfg, "requires_multimodal_data", False)
                    if i in self.processors:
                        next_inputs = self.processors[i](
                            proxies, engine_input_source, [request], requires_mm
                        )
                    else:
                        next_inputs = stage_result

                    # Transfer via connector
                    connector: StageConnector | None = self.connectors.get((str(i - 1), str(i)))  # type: ignore[assignment]
                    if connector is not None:
                        ok, _, _ = connector.put(str(i - 1), str(i), request_id, next_inputs)  # type: ignore[arg-type]
                        if not ok:
                            yield {"error": "connector.put() failed", "finished": True}
                            return
                        stage_request = {
                            "from_connector": True,
                            "from_stage": str(i - 1),
                            "to_stage": str(i),
                            "request_id": request_id,
                        }
                    else:
                        stage_request = {"engine_inputs": next_inputs, "request_id": request_id}

                raw = {}
                async for chunk in await client.round_robin(stage_request):
                    data = chunk.data()
                    if isinstance(data, (str, bytes)):
                        data = json.loads(data)
                    raw.update(data)

                if "error" in raw:
                    yield raw
                    return

                # For intermediate stages: deserialize SHM output for the next processor
                if i < len(self.stage_configs) - 1 and raw.get("shm_meta"):
                    stage_result = _shm_deserialize(raw["shm_meta"])
        finally:
            for connector in self.connectors.values():
                try:
                    connector.cleanup(request_id)  # type: ignore[arg-type]
                except Exception:
                    pass

        # --- Format final output ---
        if not raw.get("shm_meta"):
            yield {"error": "No SHM output from final stage", "finished": True}
            return

        async for chunk in self._format_output(raw, request_id, request_type):
            yield chunk

    async def _format_output(
        self, raw: dict, request_id: str, request_type: Any
    ) -> AsyncGenerator[dict, None]:
        """Read OmniRequestOutput from SHM and format into frontend response."""
        from dynamo.common.utils.output_modalities import RequestType

        shm_meta = raw.get("shm_meta")
        if not shm_meta:
            logger.warning("Router: no shm_meta in stage output")
            return

        result = _shm_deserialize(shm_meta)
        images = getattr(result, "images", None)
        request_output = getattr(result, "request_output", None)

        if images and request_type == RequestType.VIDEO_GENERATION:
            chunk = await self._format_video(images, request_id)
            if chunk:
                yield chunk
        elif images:
            chunk = await self._format_image(images)
            if chunk:
                yield chunk
        elif request_output and getattr(request_output, "outputs", None):
            yield {"choices": [{"message": {"content": request_output.outputs[0].text}}]}
        else:
            logger.warning("Router: unrecognized output, final_output_type=%s",
                           getattr(result, "final_output_type", "unknown"))

    async def _format_video(self, images: list, request_id: str) -> dict | None:
        import asyncio
        import tempfile
        import time

        from diffusers.utils.export_utils import export_to_video
        from dynamo.common.protocols.video_protocol import NvVideosResponse, VideoData
        from dynamo.common.storage import upload_to_fs
        from dynamo.common.utils.video_utils import normalize_video_frames

        if not images:
            return None

        try:
            frame_list = normalize_video_frames(images)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                fps = self.config.default_video_fps
                await asyncio.to_thread(export_to_video, frame_list, tmp.name, fps)
                video_bytes = tmp.read()

            if self._media_output_fs is None:
                logger.warning("No media_output_fs configured, cannot upload video")
                return None

            video_url = await upload_to_fs(
                self._media_output_fs,
                f"videos/{request_id}.mp4",
                video_bytes,
                self._media_output_http_url,
            )

            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self.config.served_model_name or self.config.model,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[VideoData(url=video_url)],
                inference_time_s=0.0,
            ).model_dump()
        except Exception as e:
            logger.error("Video formatting failed for %s: %s", request_id, e)
            return None

    async def _format_image(self, images: list) -> dict | None:
        # TODO: use OmniHandler._format_image_chunk once formatting is extracted
        # to a shared utility. For now, returns base64 PNG of the first image only.
        import base64
        import io

        if not images:
            return None
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}


async def init_omni_stage_router(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
) -> None:
    """Initialize OmniStageRouter as a Dynamo backend endpoint.

    Mirrors init_omni() setup pattern exactly.
    """
    from dynamo.common.utils.output_modalities import get_output_modalities
    from dynamo.vllm.omni.stage_worker import _resolve_model_type

    # Mirror init_omni: create endpoint FIRST
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint or 'generate'}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    router = OmniStageRouter(config, config.stage_configs_path)  # type: ignore[arg-type]

    setup_metrics_collection(config, generate_endpoint, logger)

    # Discover stage endpoints
    for stage_cfg in router.stage_configs:
        model_stage = getattr(stage_cfg.engine_args, "model_stage", f"stage{stage_cfg.stage_id}")
        client = await runtime.endpoint(
            f"{config.namespace}.{model_stage}.generate"
        ).client()
        await client.wait_for_instances()
        router.set_stage_client(model_stage, client)

    if not getattr(config.engine_args, "data_parallel_rank", 0):
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
                    (prometheus_names.labels.MODEL, config.served_model_name or config.model),
                    (prometheus_names.labels.MODEL_NAME, config.served_model_name or config.model),
                ],
            )
        except Exception as e:
            logger.error("OmniStageRouter endpoint failed: %s", e)
            raise
