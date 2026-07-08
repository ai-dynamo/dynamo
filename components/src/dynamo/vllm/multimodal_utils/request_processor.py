# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM multimodal request preparation.

The legacy handler and unified backend both receive Dynamo's
``PreprocessedRequest`` wire shape. This module owns the engine-facing
translation: media loading, frontend-transferred ``mm_kwargs``, stable
multimodal UUIDs, and the model-specific prefill/decode handoff.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import torch
from vllm.inputs import TokensPrompt
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange

from dynamo.common.constants import DisaggregationMode
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlReceiver,
    MmKwargsReceiver,
    MmKwargsShmReceiver,
    MmKwargsShmTransferMetadata,
    MmKwargsTransferMetadata,
)
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils import nvtx_utils as _nvtx

from .hash_utils import compute_mm_uuids_from_images
from .model import ModelFamily, construct_qwen_decode_mm_data, resolve_model_family
from .models.qwen import (
    QwenGridParams,
    build_qwen_embedding_params,
    load_qwen_grid_params,
)

logger = logging.getLogger(__name__)

IMAGE_URL_KEY = "image_url"
VIDEO_URL_KEY = "video_url"
AUDIO_URL_KEY = "audio_url"
URL_VARIANT_KEY = "Url"


def pad_mm_hashes_to_64(mm_hashes: list[str]) -> list[str]:
    """Pad frontend hashes to vLLM's 64-character UUID representation."""
    return [
        value.ljust(64, "0") if isinstance(value, str) and len(value) < 64 else value
        for value in mm_hashes
    ]


def _normalize_forwarded_mm_modality(
    modality: str,
    use_unified_vision_chunk: bool,
) -> str:
    """Map frontend modality names to the names expected by the model."""
    if use_unified_vision_chunk and modality == "image":
        return "vision_chunk"
    return modality


def _build_forwarded_mm_uuids(
    extra_args: dict[str, Any],
    use_unified_vision_chunk: bool,
) -> Optional[dict[str, Any]]:
    """Preserve frontend cache identities, including mixed modalities."""
    grouped_hashes = extra_args.get("mm_hashes_by_modality")
    if isinstance(grouped_hashes, dict):
        mm_uuids: dict[str, Any] = {}
        for modality, hashes in grouped_hashes.items():
            if not hashes:
                continue
            modality_key = _normalize_forwarded_mm_modality(
                str(modality),
                use_unified_vision_chunk,
            )
            mm_uuids.setdefault(modality_key, []).extend(
                pad_mm_hashes_to_64(list(hashes))
            )
        if mm_uuids:
            return mm_uuids

    forwarded_hashes = extra_args.get("mm_hashes")
    if forwarded_hashes:
        modality_key = _normalize_forwarded_mm_modality(
            "image",
            use_unified_vision_chunk,
        )
        return {modality_key: pad_mm_hashes_to_64(list(forwarded_hashes))}

    return None


def _get_modality_extra_values(
    extra_args: dict[str, Any],
    grouped_key: str,
    flat_key: str,
    metadata_modality: str,
    backend_modality: str,
) -> Any:
    """Read grouped transfer metadata with the legacy image-only fallback."""
    grouped_values = extra_args.get(grouped_key)
    if isinstance(grouped_values, dict):
        for key in (metadata_modality, backend_modality):
            values = grouped_values.get(key)
            if values:
                return values
    if metadata_modality != "image":
        return None
    return extra_args.get(flat_key)


def _placeholder_range_from_extra_arg(value: Any) -> PlaceholderRange:
    """Restore placeholder ranges and optional partial-embedding masks."""
    if isinstance(value, dict):
        offset = int(value["offset"])
        length = int(value["length"])
        is_embed_raw = value.get("is_embed")
        is_embed = (
            None
            if is_embed_raw is None
            else torch.as_tensor(is_embed_raw, dtype=torch.bool)
        )
        if is_embed is not None and is_embed.numel() != length:
            raise ValueError(
                "forwarded mm placeholder is_embed length "
                f"{is_embed.numel()} does not match placeholder length {length}"
            )
        return PlaceholderRange(offset=offset, length=length, is_embed=is_embed)

    offset, length = value
    return PlaceholderRange(offset=offset, length=length)


def compute_mm_uuids(
    multi_modal_data: Optional[dict[str, Any]],
) -> Optional[dict[str, list[str]]]:
    """Compute image UUIDs when the frontend did not provide canonical hashes."""
    if not multi_modal_data:
        return None

    modality = "image"
    images = multi_modal_data.get(modality)
    if images is None and "vision_chunk" in multi_modal_data:
        modality = "vision_chunk"
        chunks = multi_modal_data[modality]
        if not isinstance(chunks, list):
            chunks = [chunks]
        images = [
            chunk.get("image")
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("image") is not None
        ]
    elif isinstance(images, dict):
        # Pre-computed embedding dictionaries do not have a stable raw-image
        # preimage here. Their identity is carried by the upstream encoder/cache.
        return None

    if images is None:
        return None
    if not isinstance(images, list):
        images = [images]
    if not images:
        return None
    return {modality: compute_mm_uuids_from_images(images)}


def get_mm_processor_kwargs(request: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Read processor kwargs from the canonical or router-compatible location."""
    value = request.get("mm_processor_kwargs")
    if value is None:
        extra_args = request.get("extra_args")
        if isinstance(extra_args, dict):
            value = extra_args.get("mm_processor_kwargs")
    return value


@dataclass
class PreparedMultimodalPrompt:
    """Engine-ready prompt plus data needed for a prefill handoff."""

    prompt: Any
    request: dict[str, Any]
    multi_modal_data: Optional[dict[str, Any]] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None


@dataclass
class PreparedMultimodalInput:
    """Mode-aware request state before the engine prompt is constructed."""

    request: dict[str, Any]
    multi_modal_data: Optional[dict[str, Any]]
    mm_processor_kwargs: Optional[dict[str, Any]]
    pre_rendered_prompt: Any = None


class MissingMultimodalHandoffError(ValueError):
    """Prefill did not provide metadata required by multimodal decode."""


class VllmMultimodalRequestProcessor:
    """Translate Dynamo multimodal requests into vLLM engine inputs."""

    def __init__(
        self,
        *,
        model: str,
        engine_client: Any = None,
        enable_multimodal: bool = False,
        enable_frontend_decoding: bool = False,
        embedding_loader: Any = None,
        image_loader: Optional[ImageLoader] = None,
        video_loader: Optional[VideoLoader] = None,
        audio_loader: Optional[AudioLoader] = None,
        use_unified_vision_chunk: Optional[bool] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model = model
        self.engine_client = engine_client
        self.enable_multimodal = enable_multimodal
        self.trust_remote_code = trust_remote_code
        self.embedding_loader = embedding_loader
        self.image_loader = image_loader or ImageLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.video_loader = video_loader or VideoLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.audio_loader = audio_loader or AudioLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self._mm_kwargs_receiver: Optional[MmKwargsNixlReceiver] = None
        self._model_family = resolve_model_family(model)
        self._qwen_grid_params: Optional[QwenGridParams] = None

        if use_unified_vision_chunk is None:
            model_config = getattr(
                getattr(engine_client, "vllm_config", None), "model_config", None
            )
            use_unified_vision_chunk = bool(
                getattr(
                    getattr(model_config, "hf_config", None),
                    "use_unified_vision_chunk",
                    False,
                )
            )
        self.use_unified_vision_chunk = use_unified_vision_chunk

    @staticmethod
    def _multimodal_disabled_error() -> ValueError:
        return ValueError(
            "Received multimodal data but multimodal processing is not enabled. "
            "Use --enable-multimodal flag to enable multimodal processing."
        )

    def validate_multimodal_request(self, request: dict[str, Any]) -> None:
        """Enforce the multimodal opt-in on the unmodified inbound request."""
        extra_args = request.get("extra_args")
        has_transfer = isinstance(extra_args, dict) and any(
            extra_args.get(key) is not None
            for key in ("mm_kwargs_shm", "mm_kwargs_nixl")
        )
        if (
            request.get("multi_modal_data") is not None or has_transfer
        ) and not self.enable_multimodal:
            raise self._multimodal_disabled_error()

    def initialize_prefill_handoff(self) -> None:
        """Load model policy needed to construct the P/D decode handoff."""
        if not self.enable_multimodal or self._model_family is not ModelFamily.QWEN_VL:
            return
        self._qwen_grid_params = load_qwen_grid_params(
            self.model, trust_remote_code=self.trust_remote_code
        )
        if self._qwen_grid_params is None and self.embedding_loader is None:
            raise RuntimeError(
                "Qwen-VL grid parameters could not be loaded and no encode "
                "worker is configured. Multimodal P/D requests cannot "
                "initialize decode mRoPE."
            )

    def build_prefill_handoff(
        self,
        *,
        multi_modal_data: Optional[dict[str, Any]],
        prompt_token_ids: list[int],
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Build the model-specific multimodal portion of a P/D handoff."""
        if not multi_modal_data:
            return None
        if self._model_family is ModelFamily.QWEN_VL:
            return build_qwen_embedding_params(
                multi_modal_data,
                self._qwen_grid_params,
                mm_processor_kwargs,
            )
        return {"expanded_prompt_token_ids": prompt_token_ids}

    async def extract_multimodal_data(
        self,
        request: dict[str, Any],
        request_id: str,
        context: Any,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Load Dynamo URL/decoded media into vLLM's modality dictionary."""
        rng = _nvtx.start_range("mm_backend:extract_multimodal_data", color="orange")
        try:
            mm_map = request.get("multi_modal_data")
            if mm_map is None:
                return None

            vllm_mm_data: dict[str, Any] = {}

            # A separate encoder currently supports URL-based images only. Keep
            # processing other modalities locally so mixed image/video requests
            # preserve all of their inputs.
            if self.embedding_loader is not None:
                image_urls: list[str] = []
                supported = True
                for item in mm_map.get(IMAGE_URL_KEY, []):
                    if isinstance(item, dict) and URL_VARIANT_KEY in item:
                        image_urls.append(item[URL_VARIANT_KEY])
                    elif isinstance(item, dict) and "Decoded" in item:
                        supported = False
                if supported:
                    vllm_mm_data = (
                        await self.embedding_loader.load_multimodal_embeddings(
                            image_urls,
                            request_id,
                            model=self.model,
                            context=context,
                        )
                    )

            image_items = mm_map.get(IMAGE_URL_KEY, [])
            image_key = "vision_chunk" if self.use_unified_vision_chunk else "image"
            if image_key not in vllm_mm_data and image_items:
                with _nvtx.annotate("mm_backend:image_download", color="green"):
                    images = await self.image_loader.load_image_batch(image_items)
                if images:
                    if self.use_unified_vision_chunk:
                        chunks = [
                            {"type": "image", "image": image, "uuid": None}
                            for image in images
                        ]
                        vllm_mm_data[image_key] = (
                            chunks[0] if len(chunks) == 1 else chunks
                        )
                    else:
                        vllm_mm_data[image_key] = (
                            images[0] if len(images) == 1 else images
                        )

            video_items = mm_map.get(VIDEO_URL_KEY, [])
            if video_items:
                videos = await self.video_loader.load_video_batch(video_items)
                if videos:
                    vllm_mm_data["video"] = videos[0] if len(videos) == 1 else videos

            audio_items = mm_map.get(AUDIO_URL_KEY, [])
            if audio_items:
                audios = await self.audio_loader.load_audio_batch(audio_items)
                if audios:
                    vllm_mm_data["audio"] = audios[0] if len(audios) == 1 else audios

            if (
                video_items
                and mm_processor_kwargs
                and mm_processor_kwargs.get("use_audio_in_video", False)
            ):
                video_audios = []
                for item in video_items:
                    url = item.get(URL_VARIANT_KEY) if isinstance(item, dict) else None
                    if not url:
                        raise ValueError(
                            "use_audio_in_video requires all video items to be "
                            "URL-based. Got a non-URL video item (e.g. frontend-"
                            "decoded). Audio extraction from decoded video data "
                            "is not yet supported."
                        )
                    try:
                        video_audios.append(await self.audio_loader.load_audio(url))
                    except Exception:
                        logger.error(
                            "Request %s failed to extract audio from video. "
                            "use_audio_in_video requires every video to "
                            "contain an audio stream.",
                            request_id,
                        )
                        raise
                if video_audios:
                    existing = vllm_mm_data.get("audio")
                    existing_items = (
                        existing
                        if isinstance(existing, list)
                        else ([existing] if existing is not None else [])
                    )
                    all_audios = existing_items + video_audios
                    vllm_mm_data["audio"] = (
                        all_audios[0] if len(all_audios) == 1 else all_audios
                    )

            return vllm_mm_data or None
        finally:
            _nvtx.end_range(rng)

    async def try_receive_mm_kwargs(
        self, request: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Build a pre-rendered vLLM input from frontend SHM/NIXL metadata."""
        extra_args = request.get("extra_args") or {}
        shm_meta_raw = extra_args.get("mm_kwargs_shm")
        nixl_meta_raw = extra_args.get("mm_kwargs_nixl")
        try:
            if shm_meta_raw:
                shm_metadata = MmKwargsShmTransferMetadata.model_validate(shm_meta_raw)
                return await self._receive_mm_kwargs(
                    extra_args, "shm", MmKwargsShmReceiver(), shm_metadata
                )

            if nixl_meta_raw:
                nixl_metadata = MmKwargsTransferMetadata.model_validate(nixl_meta_raw)
                if self._mm_kwargs_receiver is None:
                    self._mm_kwargs_receiver = MmKwargsNixlReceiver()
                return await self._receive_mm_kwargs(
                    extra_args, "nixl", self._mm_kwargs_receiver, nixl_metadata
                )
        except Exception:
            logger.exception(
                "Multimodal transfer setup failed; falling back to raw media"
            )
        return None

    async def _receive_mm_kwargs(
        self,
        extra_args: dict[str, Any],
        transport: str,
        receiver: MmKwargsReceiver,
        metadata: MmKwargsShmTransferMetadata | MmKwargsTransferMetadata,
    ) -> Optional[dict[str, Any]]:
        color = "magenta" if transport == "nixl" else "cyan"
        rng = _nvtx.start_range(f"mm_backend:{transport}_receive", color=color)
        try:
            backend_modality = _normalize_forwarded_mm_modality(
                metadata.modality,
                self.use_unified_vision_chunk,
            )
            mm_hashes = (
                _get_modality_extra_values(
                    extra_args,
                    "mm_hashes_by_modality",
                    "mm_hashes",
                    metadata.modality,
                    backend_modality,
                )
                or metadata.mm_hashes
            )
            mm_placeholders = _get_modality_extra_values(
                extra_args,
                "mm_placeholders_by_modality",
                "mm_placeholders",
                metadata.modality,
                backend_modality,
            )
            expanded_token_ids = extra_args.get("expanded_token_ids")
            if not mm_hashes or not mm_placeholders or not expanded_token_ids:
                logger.warning(
                    "%s multimodal transfer metadata is incomplete; falling back",
                    transport,
                )
                return None

            results = await receiver.receive(metadata)
            pickled_items = results.get("__pickled_kwargs_item__")
            if not pickled_items:
                return None

            kwargs_items = []
            for payload in pickled_items:
                # The sender is Dynamo's internal frontend transfer service,
                # which deliberately serializes vLLM's Python-only kwargs
                # objects. External request payloads never supply these bytes.
                item = pickle.loads(payload)
                if not isinstance(item, MultiModalKwargsItem):
                    logger.warning(
                        "%s transfer produced %s instead of MultiModalKwargsItem",
                        transport,
                        type(item).__name__,
                    )
                    return None
                kwargs_items.append(item)

            if not (len(kwargs_items) == len(mm_hashes) == len(mm_placeholders)):
                logger.warning(
                    "%s multimodal transfer item/hash/placeholder counts differ; "
                    "falling back",
                    transport,
                )
                return None

            padded_hashes = pad_mm_hashes_to_64(list(mm_hashes))
            mm_hashes_dict = {backend_modality: padded_hashes}
            mm_kwargs_dict = {backend_modality: kwargs_items}
            engine_input = {
                "type": "multimodal",
                "prompt_token_ids": expanded_token_ids,
                "mm_kwargs": mm_kwargs_dict,
                "mm_hashes": mm_hashes_dict,
                "mm_placeholders": {
                    backend_modality: [
                        _placeholder_range_from_extra_arg(placeholder)
                        for placeholder in mm_placeholders
                    ]
                },
            }

            input_processor = getattr(self.engine_client, "input_processor", None)
            if input_processor is not None:
                try:
                    input_processor.inject_into_mm_cache(mm_hashes_dict, mm_kwargs_dict)
                except Exception:
                    logger.debug(
                        "Failed to inject transferred mm_kwargs into vLLM cache",
                        exc_info=True,
                    )
            return engine_input
        except Exception:
            logger.exception("%s multimodal transfer failed; falling back", transport)
            return None
        finally:
            _nvtx.end_range(rng)

    def build_tokens_prompt(
        self,
        request: dict[str, Any],
        multi_modal_data: Optional[dict[str, Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
    ) -> TokensPrompt:
        """Create a TokensPrompt with stable multimodal UUIDs."""
        extra_args = request.get("extra_args") or {}
        mm_uuids = _build_forwarded_mm_uuids(
            extra_args,
            self.use_unified_vision_chunk,
        )
        if mm_uuids is None and self.embedding_loader is None:
            mm_uuids = compute_mm_uuids(multi_modal_data)
            if mm_uuids is not None:
                logger.warning(
                    "No frontend multimodal hashes were provided; recomputed "
                    "image UUIDs may not match routing decisions"
                )

        prompt_kwargs: dict[str, Any] = {
            "prompt_token_ids": request["token_ids"],
            "multi_modal_data": multi_modal_data,
        }
        if mm_uuids is not None:
            prompt_kwargs["multi_modal_uuids"] = mm_uuids
        if mm_processor_kwargs is not None:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        return TokensPrompt(**prompt_kwargs)

    async def prepare_input(
        self,
        request: dict[str, Any],
        request_id: str,
        context: Any,
        mode: DisaggregationMode,
    ) -> PreparedMultimodalInput:
        """Apply aggregated/P/D media policy to a validated request.

        Entry points must call :meth:`validate_multimodal_request` on the raw
        request before invoking this transformation. ``prepare_prompt`` does
        that for unified engines; the legacy handler validates at ``generate``
        so text and token modes share the same security boundary.
        """
        mm_processor_kwargs = get_mm_processor_kwargs(request)
        request_for_prompt = dict(request)
        has_mm_data = request.get("multi_modal_data") is not None
        multi_modal_data: Optional[dict[str, Any]] = None
        pre_rendered = None

        if mode == DisaggregationMode.DECODE:
            prefill_result = request.get("prefill_result") or {}
            disaggregated_params = prefill_result.get("disaggregated_params") or {}
            embedding_params = disaggregated_params.get("embedding_params") or None
            if self._model_family is ModelFamily.QWEN_VL:
                if embedding_params is not None:
                    multi_modal_data = construct_qwen_decode_mm_data(
                        embedding_params.get("image_grid_thw"),
                        embedding_params.get("embeddings_shape"),
                        request_id,
                    )
                elif has_mm_data and request["multi_modal_data"].get(IMAGE_URL_KEY):
                    prefill_result = request.get("prefill_result")
                    message = (
                        "Decode worker received multimodal request without "
                        "prefill result"
                        if prefill_result is None
                        else "Prefill did not produce required multimodal "
                        "embedding metadata (image_grid_thw) for Qwen-VL decode"
                    )
                    raise MissingMultimodalHandoffError(message)
            elif embedding_params and embedding_params.get("expanded_prompt_token_ids"):
                request_for_prompt["token_ids"] = embedding_params[
                    "expanded_prompt_token_ids"
                ]
                has_mm_data = False

            # Preserve the legacy fallback: video/audio media is loaded again
            # on decode because the handoff currently carries image metadata only.
            if multi_modal_data is None and has_mm_data:
                mm_map = request["multi_modal_data"]
                if mm_map.get(VIDEO_URL_KEY) or mm_map.get(AUDIO_URL_KEY):
                    multi_modal_data = await self.extract_multimodal_data(
                        request,
                        request_id,
                        context,
                        mm_processor_kwargs,
                    )
        elif mode == DisaggregationMode.AGGREGATED:
            pre_rendered = await self.try_receive_mm_kwargs(request)
            if pre_rendered is None:
                multi_modal_data = await self.extract_multimodal_data(
                    request,
                    request_id,
                    context,
                    mm_processor_kwargs,
                )
        else:
            # P/D prefill still needs the raw media object after generation to
            # construct model-specific decode metadata. The transferred
            # pre-rendered input intentionally remains an aggregated-only fast
            # path until that handoff can be derived from vLLM's processed
            # feature data.
            multi_modal_data = await self.extract_multimodal_data(
                request,
                request_id,
                context,
                mm_processor_kwargs,
            )

        return PreparedMultimodalInput(
            request=request_for_prompt,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            pre_rendered_prompt=pre_rendered,
        )

    async def prepare_prompt(
        self,
        request: dict[str, Any],
        request_id: str,
        context: Any,
        mode: DisaggregationMode,
    ) -> PreparedMultimodalPrompt:
        """Prepare the complete engine prompt for the unified backend."""
        self.validate_multimodal_request(request)
        prepared = await self.prepare_input(request, request_id, context, mode)
        prompt = prepared.pre_rendered_prompt or self.build_tokens_prompt(
            prepared.request,
            prepared.multi_modal_data,
            prepared.mm_processor_kwargs,
        )
        return PreparedMultimodalPrompt(
            prompt=prompt,
            request=prepared.request,
            multi_modal_data=prepared.multi_modal_data,
            mm_processor_kwargs=prepared.mm_processor_kwargs,
        )
