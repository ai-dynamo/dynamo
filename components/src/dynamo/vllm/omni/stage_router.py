# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import dataclasses
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict

from dynamo import prometheus_names
from dynamo.llm import ModelInput, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.types import OmniInterStageRequest, StageOutput

logger = logging.getLogger(__name__)


# TODO: Passing raw prompts to InputProcessor is deprecated and will be removed in v0.18. You should instead pass the outputs of Renderer.render_cmpl() or Renderer.render_chat().
def _build_original_prompt(request: dict, nvext: dict, height: int, width: int) -> Any:
    """Build the rich prompt dict that processor functions (ar2diffusion etc.) read.

    Processors access height/width for geometry, sampling keys for diffusion,
    and multi_modal_data for I2V — all extracted from the frontend request here
    so workers receive a complete, structured prompt without re-parsing.
    """
    from vllm_omni.inputs.data import OmniTextPrompt

    prompt = OmniTextPrompt(prompt=request.get("prompt", ""))
    prompt["height"] = height  # type: ignore[literal-required]
    prompt["width"] = width  # type: ignore[literal-required]
    for key in (
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "negative_prompt",
        "boundary_ratio",
        "guidance_scale_2",
    ):
        if nvext.get(key) is not None:
            prompt[key] = nvext[key]  # type: ignore[literal-required]
    if request.get("multi_modal_data"):
        prompt["multi_modal_data"] = request["multi_modal_data"]
    return prompt


def _shm_deserialize(shm_meta: dict) -> Any:
    """Read and deserialize an OmniRequestOutput from shared memory."""
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer
    from vllm_omni.entrypoints.stage_utils import shm_read_bytes

    return OmniSerializer.deserialize(shm_read_bytes(shm_meta))


def _parse_engine_inputs(
    request: dict, request_type: Any, config: "OmniConfig"
) -> dict:
    """Convert a raw frontend request into engine_inputs + sampling_params_list.

    Passing the raw request dict as prompt causes the diffusion engine to use
    default values (e.g. num_frames=1). This extracts num_frames, size, etc.
    from nvext and builds proper OmniDiffusionSamplingParams.

    Returns a dict with keys:
      engine_inputs:        OmniTextPrompt (prompt text only) for the stage 0 engine
      original_prompt:      rich prompt dict with geometry/params for processor functions
      sampling_params_list: [OmniDiffusionSamplingParams dict] or None
    """
    from dynamo.common.utils.output_modalities import RequestType

    if request_type == RequestType.VIDEO_GENERATION:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

        from dynamo.common.utils.video_utils import compute_num_frames, parse_size

        nvext = request.get("nvext") or {}
        width, height = parse_size(request.get("size", "832x480"))
        num_frames = compute_num_frames(
            num_frames=nvext.get("num_frames"),
            fps=nvext.get("fps"),
            default_fps=config.default_video_fps,
        )
        sp = OmniDiffusionSamplingParams(
            height=height, width=width, num_frames=num_frames
        )
        for attr in (
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "boundary_ratio",
            "guidance_scale_2",
        ):
            if nvext.get(attr) is not None:
                setattr(sp, attr, nvext[attr])

        return {
            "engine_inputs": OmniTextPrompt(prompt=request.get("prompt", "")),
            "original_prompt": _build_original_prompt(request, nvext, height, width),
            "sampling_params_list": [dataclasses.asdict(sp)],
        }

    if request_type == RequestType.IMAGE_GENERATION:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

        from dynamo.common.utils.video_utils import parse_size

        nvext = request.get("nvext") or {}
        width, height = parse_size(
            request.get("size", "1024x1024"), default_w=1024, default_h=1024
        )
        sp = OmniDiffusionSamplingParams(height=height, width=width)
        if nvext.get("num_inference_steps") is not None:
            sp.num_inference_steps = nvext["num_inference_steps"]

        return {
            "engine_inputs": OmniTextPrompt(prompt=request.get("prompt", "")),
            "original_prompt": _build_original_prompt(request, nvext, height, width),
            "sampling_params_list": [dataclasses.asdict(sp)],
        }

    # Chat / text
    messages = request.get("messages", [])
    text = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        request.get("prompt", ""),
    )
    return {
        "engine_inputs": text,
        "original_prompt": {"prompt": text},
        "sampling_params_list": None,
    }


class OmniStageRouter:
    """Pure message broker for multi-stage omni pipelines."""

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        from vllm_omni.entrypoints.utils import load_stage_configs_from_yaml

        self.config = config
        self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)
        self.stage_clients: Dict[str, Any] = {}

        from dynamo.common.storage import get_fs
        from dynamo.vllm.omni.output_formatter import OutputFormatter

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
        from dynamo.common.utils.output_modalities import parse_request_type

        request_id = str(uuid.uuid4())
        _, request_type = parse_request_type(request, self.config.output_modalities)

        # Parse frontend request into proper engine inputs so num_frames,
        # num_inference_steps etc. reach the diffusion engine correctly.
        # Passing the raw dict as prompt causes the engine to use defaults (1 frame).
        engine_inputs = _parse_engine_inputs(request, request_type, self.config)

        # Build inter-stage protocol — original_prompt set once, passed unchanged.
        inter_stage_req = OmniInterStageRequest(
            request_id=request_id,
            original_prompt=engine_inputs["original_prompt"],
        )

        # --- Call each stage in order ---
        raw: Dict[str, Any] = {}
        for i, stage_cfg in enumerate(self.stage_configs):
            logger.info("Router: calling stage %d", i)
            model_stage = getattr(stage_cfg.engine_args, "model_stage", f"stage{i}")
            client = self.stage_clients.get(model_stage)
            if client is None:
                yield {
                    "error": f"No client for stage '{model_stage}'",
                    "finished": True,
                }
                return

            if i == 0:
                # Stage 0: send engine_inputs + inter-stage protocol
                # TODO: forward engine_inputs["sampling_params_list"] to stage workers
                # so per-request params (num_inference_steps, guidance_scale, seed, etc.)
                # override YAML defaults instead of being silently dropped.
                stage_request = {
                    **inter_stage_req.to_dict(),
                    "engine_inputs": engine_inputs["engine_inputs"],
                }
            else:
                logger.info(
                    "Router: stage %d received keys=%s", i - 1, list(raw.keys())
                )
                # Subsequent stages: validate + filter to known protocol fields only.
                # StageOutput drops unknown keys; router never inspects stage_connector_refs.
                stage_request = StageOutput.model_validate(raw).to_next_stage_request(
                    request_id
                )

            raw = {}
            logger.info(
                "Router: stage %d request keys=%s", i, list(stage_request.keys())
            )
            async for chunk in await client.round_robin(stage_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                raw.update(data)

            if "error" in raw:
                yield raw
                return

        # --- Format final output ---
        if not raw.get("shm_meta"):
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

        async for chunk in self._format_output(raw, request_id, request_type, fmt_ctx):
            yield chunk

    async def _format_output(
        self, raw: dict, request_id: str, request_type: Any, ctx: dict
    ) -> AsyncGenerator[dict, None]:
        """Read OmniRequestOutput from SHM and format via OutputFormatter."""
        shm_meta = raw.get("shm_meta")
        if not shm_meta:
            logger.warning("Router: no shm_meta in stage output")
            return

        result = _shm_deserialize(shm_meta)
        chunk = await self._formatter.format(
            result, request_id, request_type=request_type, **ctx
        )
        if chunk:
            yield chunk
        else:
            logger.warning(
                "Router: formatter returned None, final_output_type=%s",
                getattr(result, "final_output_type", "unknown"),
            )


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
        model_stage = getattr(
            stage_cfg.engine_args, "model_stage", f"stage{stage_cfg.stage_id}"
        )
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
