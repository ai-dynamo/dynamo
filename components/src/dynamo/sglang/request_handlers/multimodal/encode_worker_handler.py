# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import torch
from blake3 import blake3

# MMEncoder chain imports compiled CUDA ops; may fail in CPU-only environments.
try:
    from sglang.srt.disaggregation.encode_server import MMEncoder
    from sglang.srt.managers.schedule_batch import Modality
except (ImportError, OSError):
    MMEncoder = None  # type: ignore[assignment]
    Modality = None  # type: ignore[assignment]
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

from dynamo._core import Client, Context
from dynamo.common.http import fetch_bytes
from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import EMBEDDING_SENDER_FACTORIES
from dynamo.common.multimodal.cache_uuid import reject_unsupported_multimodal_uuids
from dynamo.common.multimodal.codec_errors import (
    MissingMediaDecoderError,
    video_decoder_missing,
)
from dynamo.common.multimodal.media_source import (
    is_local_media_url,
    read_local_media_bytes,
)
from dynamo.common.multimodal.nvdec_decoder import (
    DISABLE_ENV,
    nvdec_available,
    probe_video_codec,
    should_use_nvdec,
)
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.env import env_bool
from dynamo.llm import MultimodalEmbeddingCachePublisher
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SglangMultimodalRequest,
)
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.request_handlers.multimodal.nvdec_video_decoder import (
    SGLANG_VIDEO_DECODER_AVAILABLE,
    NvdecVideoDecoder,
)

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

IMAGE_URL_KEY = "image_url"
VIDEO_URL_KEY = "video_url"

# SGLang model types whose video preprocessing needs per-frame timestamps from
# ``video_metadata``. For these, ``_process_video_items`` runs
# ``for m in video_metadata: m.get("fps")`` (sglang.srt.disaggregation.encode_server,
# v0.5.15). With the metadata shim below every pre-decoded frame carries a valid
# metadata dict (so that branch no longer crashes on ``None``), but routing these
# models through NVDEC is still gated pending end-to-end timestamp validation. The
# qwen2.5 family takes the ``video_grid_thw`` branch and does not read the metadata,
# so it is unaffected either way. SGLang has no in-image software decoder for the
# gated models either, so keeping them on the URL path is no worse than status quo.
_NVDEC_UNSAFE_MODEL_TYPES = frozenset(
    {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "intern_s2_preview"}
)

# Fallback source fps stamped into the synthesized ``video_metadata`` for
# pre-decoded NVDEC frames (see ``_install_nvdec_video_metadata_shim``). The
# supported qwen2_5_vl model_type ignores ``video_metadata`` entirely; the value
# only feeds the qwen3_vl timestamp branch, which is gated off today.
_NVDEC_SHIM_FPS = 24.0


def _software_video_decoder_imports() -> bool:
    """True when SGLang's software video decoder (torchcodec or decord)
    actually imports -- not merely resolves to a spec."""
    for module in ("torchcodec", "decord"):
        try:
            importlib.import_module(module)
            return True
        except Exception:  # noqa: BLE001 - broken installs count as absent
            continue
    return False


def _install_load_video_passthrough() -> None:
    """Let a pre-constructed decoder survive SGLang's ``load_video``.

    ``load_video`` (sglang.srt.utils.common, v0.5.16) returns
    ``list``/``tuple``/``Tensor``/``ndarray`` untouched, but has no case for an
    object that is already a ``VideoDecoderWrapper``: it falls through to
    ``_normalize_video_input``, which returns ``None`` for that type, and raises
    ``ValueError: Unsupported video input type``. Dynamo hands the encode path an
    ``NvdecVideoDecoder`` so SGLang can drive frame selection itself, so teach
    ``load_video`` to pass such objects through.

    Patch the name **as bound in each importing module**: they do
    ``from sglang.srt.utils import load_video``, so rebinding
    ``sglang.srt.utils.load_video`` alone leaves those call sites untouched.
    ``encode_server`` is the one this handler actually goes through -- its
    ``_flatten_and_load_videos`` calls its own binding -- and omitting it is why
    an earlier revision still raised ``ValueError: Unsupported video input type``
    end to end while every unit test passed.

    Verified against ``v0.5.16``: ``encode_server``, ``base_processor`` and
    ``utils`` are the ``srt`` modules that bind the name. Patch every one that
    imports successfully and log which, so a future SGLang bump that moves the
    call site shows up as a missing module rather than silently reverting to the
    URL path. Idempotent; a no-op when SGLang is unavailable.
    """
    if not SGLANG_VIDEO_DECODER_AVAILABLE:
        return
    try:
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper
    except (ImportError, OSError):
        return

    def _wrap(module, orig):
        def _load_video_passthrough(video_file, *args, **kwargs):
            if isinstance(video_file, VideoDecoderWrapper):
                return video_file
            return orig(video_file, *args, **kwargs)

        _load_video_passthrough._dynamo_nvdec_passthrough = True  # type: ignore[attr-defined]
        module.load_video = _load_video_passthrough

    patched: list[str] = []
    for module_path in (
        # The encode worker's own call site -- the one that matters here.
        "sglang.srt.disaggregation.encode_server",
        "sglang.srt.multimodal.processors.base_processor",
        "sglang.srt.utils",
    ):
        try:
            module = importlib.import_module(module_path)
        except (ImportError, OSError):
            continue
        orig = getattr(module, "load_video", None)
        if orig is None:
            continue
        if getattr(orig, "_dynamo_nvdec_passthrough", False):
            patched.append(module_path)
            continue
        _wrap(module, orig)
        patched.append(module_path)

    if "sglang.srt.disaggregation.encode_server" not in patched:
        # Without this one the decoder reaches load_video unpatched and the
        # request fails with "Unsupported video input type" rather than falling
        # back, so make the gap visible instead of waiting for a 400.
        logger.warning(
            "load_video passthrough not installed on encode_server (patched: %s); "
            "NVDEC video decoding will not work on this SGLang version",
            patched or "none",
        )
    else:
        logger.debug("load_video passthrough installed on: %s", patched)


def _install_nvdec_video_metadata_shim() -> None:
    """Let SGLang accept pre-decoded frames under transformers >= 5.12.

    SGLang's ``preprocess_video`` returns ``(frames, None)`` for a pre-decoded
    ndarray, and ``_flatten_and_load_videos`` then sets
    ``video_processor_kwargs["video_metadata"] = [None]`` -- its ``if
    video_metadata:`` treats the ``[None]`` list as truthy. transformers >= 5.12
    strict-validates that kwarg and rejects ``[None]`` (it accepts
    ``VideoMetadata`` / ``dict`` / ``list[dict]`` / ``None``), so
    ``MMEncoder._encode`` raises ``BadRequestError``. Wrap ``preprocess_video`` so
    a pre-decoded ndarray instead carries a valid metadata dict -- the same shape
    SGLang builds on its own torchvision path -- which ``list[dict]`` validation
    accepts. This unblocks any pre-decoded-pixel path -- frontend decoding
    (``--frontend-decoding``) being the one that reaches it -- and supplies the
    ``fps`` / ``frames_indices`` the qwen3_vl branch reads. Idempotent; a no-op
    when SGLang is unavailable or already patched. Validated on GPU hardware
    against the real ``MMEncoder._encode`` (before FAIL -> after PASS).

    This is **not** the NVDEC path. NVDEC hands SGLang an ``NvdecVideoDecoder``,
    which takes ``preprocess_video``'s decoder branch and returns real metadata,
    so the synthesized values below never apply to it. The shim remains only for
    genuine pre-decoded arrays.
    """
    try:
        from sglang.srt.disaggregation import encode_server as es
    except (ImportError, OSError):
        return

    orig = getattr(es, "preprocess_video", None)
    if orig is None or getattr(orig, "_dynamo_nvdec_shim", False):
        return

    async def _preprocess_video_with_metadata(vr, *args, **kwargs):
        video, metadata = await orig(vr, *args, **kwargs)
        if metadata is None and isinstance(video, np.ndarray) and video.ndim >= 1:
            num_frames = int(video.shape[0])
            fps = _NVDEC_SHIM_FPS
            metadata = {
                "fps": fps,
                "duration": (num_frames / fps) if fps else 0.0,
                "total_num_frames": num_frames,
                "frames_indices": list(range(num_frames)),
                "video_backend": "nvdec",
            }
        return video, metadata

    _preprocess_video_with_metadata._dynamo_nvdec_shim = True  # type: ignore[attr-defined]
    es.preprocess_video = _preprocess_video_with_metadata


class MultimodalEncodeWorkerHandler(BaseWorkerHandler[SglangMultimodalRequest, str]):
    """
    Handler for multimodal encode worker component that processes images/videos
    and forwards them to the downstream worker.

    Receives pre-tokenized requests from the Rust frontend (ModelInput.Tokens)
    with token_ids and multi_modal_data containing image URLs. Encodes images
    via MMEncoder, expands placeholder tokens, transfers embeddings via NIXL,
    and forwards to the PD worker.
    """

    def __init__(
        self,
        config: Config,
        pd_worker_client: Client,
        cache_publisher: MultimodalEmbeddingCachePublisher | None = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        super().__init__(engine=None, config=config, shutdown_event=shutdown_event)
        self.pd_worker_client = pd_worker_client
        self._cache_publisher = cache_publisher
        self.model = config.server_args.model_path
        self._missing_video_cache_key_config_warned = False

        # NVDEC hardware decode for H.264/H.265 video input. #11836 strips the
        # software video decoders (av/decord/torchcodec) from the SGLang image,
        # so these codecs otherwise have no decoder. Reuse the shared default so
        # this worker and the vLLM/TRT-LLM backends agree on
        # DYN_MM_VIDEO_NUM_FRAMES; VP8/VP9/AV1 stay on the frontend path.
        self.num_video_frames = max(1, VideoLoader.NUM_FRAMES_DEFAULT)
        self._url_policy = UrlValidationPolicy.from_env()

        if MMEncoder is None:
            raise RuntimeError(
                "MMEncoder is not available. "
                "Multimodal encode worker requires a CUDA environment."
            )

        # torch.distributed requires a dist_init_method even for tp=1;
        # port 0 lets the OS assign a free port.
        self.encoder = MMEncoder(
            server_args=config.server_args,
            dist_init_method="tcp://127.0.0.1:0",
            rank=0,
        )
        self._max_input_token_id = self._resolve_max_input_token_id_from_model_config(
            self.encoder.model_config
        )

        # Let SGLang accept the NVDEC-backed decoder this handler builds, so it
        # keeps ownership of frame selection (see _install_load_video_passthrough).
        _install_load_video_passthrough()

        # Make SGLang's video processor accept pre-decoded frame arrays on
        # transformers >= 5.12 (see _install_nvdec_video_metadata_shim). Safe to
        # install unconditionally: it only rewrites the ``None`` metadata SGLang
        # emits for pre-decoded pixels, leaving the software-decoder path intact.
        _install_nvdec_video_metadata_shim()

        # Load tokenizer to convert image token string to integer ID
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=config.server_args.trust_remote_code
        )

        # Get image/video token strings and resolve them to integer IDs
        template = chat_templates[getattr(config.server_args, "chat_template")].copy()
        image_token_str = template.image_token

        image_token_id = self._resolve_mm_token_id(
            image_token_str, preferred_token="<|image_pad|>"
        )
        if image_token_id is None:
            raise ValueError("image token is not defined in chat template")
        self.image_token_id = image_token_id

        self.video_token_id: Optional[int] = self._resolve_mm_token_id(
            getattr(template, "video_token", None), preferred_token="<|video_pad|>"
        )

        self.min_workers = 1

        sender = EMBEDDING_SENDER_FACTORIES.get(
            config.dynamo_args.embedding_transfer_mode
        )
        if sender is None:
            raise ValueError(
                "Invalid embedding transfer mode: "
                f"{config.dynamo_args.embedding_transfer_mode}"
            )
        self.embedding_sender = sender()

        # Optional CPU-side LRU embedding cache
        self._embedding_cache: MultimodalEmbeddingCacheManager | None = None
        capacity_gb = config.dynamo_args.multimodal_embedding_cache_capacity_gb
        if capacity_gb > 0:
            capacity_bytes = int(capacity_gb * 1024**3)
            self._embedding_cache = MultimodalEmbeddingCacheManager(capacity_bytes)
            logger.info("Multimodal embedding cache enabled: %.2f GB", capacity_gb)

    def _publish_cache_delta(
        self, added_keys: list[str], removed_keys: list[str]
    ) -> None:
        if self._cache_publisher is None or (not added_keys and not removed_keys):
            return
        try:
            self._cache_publisher.publish_delta(added_keys, removed_keys)
        except Exception:
            logger.warning(
                "Failed to publish embedding cache delta; "
                "routing cache state may be stale",
                exc_info=True,
            )

    def cleanup(self) -> None:
        pass

    @staticmethod
    def _url_hash(url: str) -> str:
        """Stable blake3 hash of a media URL, used as embedding cache key."""
        return blake3(url.encode()).hexdigest()

    @classmethod
    def _normalize_cache_key_value(cls, value: Any) -> Any:
        """Convert nested config values into a stable JSON-serializable form."""
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_cache_key_value(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_cache_key_value(item) for item in value]
        if isinstance(value, torch.Tensor):
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _media_cache_key(self, url: str, modality: Any, encoder: Any) -> str:
        """Build a cache key that is URL-stable for images and config-aware for video."""
        if modality != Modality.VIDEO:
            return self._url_hash(url)

        video_config = {}
        missing_config_fields: list[str] = []
        vision_config = getattr(encoder, "vision_config", None)
        if isinstance(vision_config, dict):
            video_config = self._normalize_cache_key_value(
                vision_config.get("video", {})
            )
        else:
            missing_config_fields.append("vision_config")

        if missing_config_fields and not getattr(
            self, "_missing_video_cache_key_config_warned", False
        ):
            logger.warning(
                "Video embedding cache key could not include encoder %s; "
                "cache reuse may not reflect all video processor settings.",
                ", ".join(missing_config_fields),
            )
            self._missing_video_cache_key_config_warned = True

        cache_key_payload = {
            "url": url,
            "video_config": video_config,
        }
        return self._url_hash(
            json.dumps(cache_key_payload, sort_keys=True, separators=(",", ":"))
        )

    def _resolve_mm_token_id(
        self, token_str: Optional[str], preferred_token: Optional[str] = None
    ) -> Optional[int]:
        if not token_str:
            return None

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            return token_id

        # For templates like qwen2-vl, modality placeholders are composite
        # strings and need to be resolved to inner pad-token IDs.
        candidates: list[str] = []
        if preferred_token and preferred_token in token_str:
            candidates.append(preferred_token)

        for marker in ("<|image_pad|>", "<|video_pad|>"):
            if marker in token_str and marker not in candidates:
                candidates.append(marker)

        for candidate in candidates:
            candidate_id = self.tokenizer.convert_tokens_to_ids(candidate)
            if isinstance(candidate_id, int) and candidate_id >= 0:
                return candidate_id

        return None

    @staticmethod
    def _ensure_batched_grid(grid_dim: Any, item_count: int) -> list:
        grid_list = (
            grid_dim.tolist() if isinstance(grid_dim, torch.Tensor) else grid_dim
        )
        if (
            item_count == 1
            and isinstance(grid_list, list)
            and len(grid_list) == 3
            and not isinstance(grid_list[0], list)
        ):
            # SGLang may squeeze the batch dimension for a single media item.
            # Normalize that flat THW grid to the batched shape used below.
            return [grid_list]
        return grid_list

    @staticmethod
    def _grid_units(grid_item: Any, modality: str) -> int:
        if modality not in ("IMAGE", "VIDEO"):
            raise ValueError(f"Unsupported modality for grid units: {modality}")
        if not isinstance(grid_item, list) or len(grid_item) != 3:
            raise ValueError(f"Invalid {modality.lower()} grid: {grid_item}")
        return int(grid_item[0] * grid_item[1] * grid_item[2])

    def _split_token_counts(
        self, grid_list: list, total_tokens: int, modality: str
    ) -> list[int]:
        """Compute per-item token counts for a modality from encoder grid metadata."""
        if total_tokens <= 0:
            raise ValueError("Invalid token count for embeddings")

        grid_sizes = [self._grid_units(item, modality) for item in grid_list]
        total_grid_tokens = sum(grid_sizes)
        if total_grid_tokens <= 0:
            raise ValueError("Invalid grid statistics for embeddings")
        if total_grid_tokens % total_tokens != 0:
            raise ValueError(
                "Cannot infer merge factor: grid token total is not divisible "
                "by embedding token total"
            )
        merge_factor = total_grid_tokens // total_tokens
        token_counts = []
        for grid_count in grid_sizes:
            if grid_count % merge_factor != 0:
                raise ValueError(
                    "Cannot split embeddings: per-item grid token count not "
                    "divisible by inferred merge factor"
                )
            token_counts.append(grid_count // merge_factor)
        if sum(token_counts) != total_tokens:
            raise ValueError(
                "Cannot split embeddings: per-item token counts do not match "
                "embedding token total"
            )
        return token_counts

    @staticmethod
    def _jsonable_media_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.item() if value.ndim == 0 else value.tolist()
        return value

    @classmethod
    def _aux_value_for_item(
        cls,
        value: Any,
        index: int,
        item_count: int,
    ) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) == item_count:
                return cls._jsonable_media_value(value[index])
            if item_count == 1:
                return cls._jsonable_media_value(value)
            raise ValueError(
                "Auxiliary media metadata length mismatch: "
                f"expected {item_count} items, got {len(value)}"
            )
        return cls._jsonable_media_value(value)

    def _nvdec_video_enabled(self) -> bool:
        """Whether to route video URLs through NVDEC for this encoder.

        Off when explicitly disabled, when PyNvVideoCodec/CUDA is unavailable,
        or when the model's SGLang video preprocessing cannot accept
        pre-decoded frames (see ``_NVDEC_UNSAFE_MODEL_TYPES``).
        """
        # env_bool, not a truthiness test on the raw value: everywhere else this
        # switch is read that way, and a bare os.environ.get makes
        # DYN_DISABLE_NVDEC=0 *disable* NVDEC here while leaving it enabled in
        # vLLM and TensorRT-LLM -- the same setting meaning opposite things
        # depending on the backend.
        if env_bool(DISABLE_ENV):
            return False
        if not nvdec_available():
            return False
        model_type = getattr(self.encoder, "model_type", "") or ""
        return model_type not in _NVDEC_UNSAFE_MODEL_TYPES

    async def _maybe_nvdec_decoder(self, url: str) -> Optional[Any]:
        """Resolve a video URL to what SGLang should decode.

        Reads the URL (SSRF-validated for http(s); policy-gated for ``file://``
        and ``data:``), probes the container codec, and returns:

        * an ``NvdecVideoDecoder`` for H.264/H.265 -- hardware decode;
        * the **fetched bytes** for any other codec, or if building the decoder
          fails -- SGLang decodes what we already have;
        * ``None`` only when nothing was fetched (unsupported scheme, or the
          fetch itself failed), leaving the caller to pass the URL through.

        Returning the bytes rather than the URL matters for three reasons, all
        reported by Codex on #11836. SGLang would otherwise download the same
        payload a second time, doubling ingress and latency for every non-NVDEC
        video. A one-use signed URL would fail on that second fetch. And SGLang
        applies no URL policy of its own -- ``_normalize_video_input`` fetches
        http(s) straight through ``get_mm_http_session`` -- so the bytes it
        decoded were never the validated ones, and a redirect could resolve
        differently between the two fetches. Handing over the bytes we validated
        and already hold closes all three.

        ``load_video`` takes ``Union[str, bytes, VideoData]`` and
        ``_normalize_video_input`` returns ``bytes`` untouched (sglang v0.5.16),
        so this needs no shim on the SGLang side.

        A decoder rather than decoded frames: SGLang's ``preprocess_video`` reads
        the source frame count and fps off the decoder and applies the model's
        own ``vision_config.video`` policy to choose which frames to pull. Handing
        it pre-sampled pixels takes the ``return vr, None`` branch instead, which
        silently replaces that policy with a backend-global frame count and
        fabricated metadata. Passing the decoder keeps frame selection in SGLang
        and reports true source values.

        The fetch stays here rather than in SGLang because SGLang's
        ``_normalize_video_input`` downloads http(s) with no policy check, which
        would drop our SSRF validation.

        ``file://`` and ``data:`` are included because SGLang's own decoders are
        stripped from the codec-compliant image: without hardware decode those
        schemes have no decoder at all, so excluding them here would drop local
        and inline video entirely rather than merely skipping acceleration.
        """
        content: bytes | None = None
        try:
            normalized = await validate_media_url(url, self._url_policy)
            scheme = urlparse(normalized).scheme
            if scheme in ("http", "https"):
                content = await fetch_bytes(normalized, 30.0, policy=self._url_policy)
            elif is_local_media_url(normalized):
                content = await read_local_media_bytes(normalized, self._url_policy)
            else:
                return None
            codec = probe_video_codec(content)
            if not should_use_nvdec(codec):
                # Not going to hardware. SGLang's software path needs
                # torchcodec or decord, which the codec-compliant image strips
                # -- without this preflight the failure happens deep inside
                # SGLang as a bare "No module named 'decord'" with the whole
                # video payload repr embedded in the message. Fail here, where
                # the codec is known and the message can be actionable.
                #
                # Real import, not find_spec: a package whose files exist but
                # whose native libraries cannot load has a spec and would pass
                # a find_spec preflight only to fail deep in SGLang anyway.
                if not _software_video_decoder_imports():
                    raise video_decoder_missing("sglang", "decord2", "decord", codec)
                # A software decoder exists; the bytes are already here and
                # were fetched under policy. Hand them over instead of the URL.
                return content
            # Constructing the decoder opens the container and reads its frame
            # index, so keep it off the event loop.
            return await asyncio.to_thread(NvdecVideoDecoder, content)
        except UrlValidationError:
            # A policy refusal is not a decode failure and must not degrade to
            # the URL fallback. SGLang applies no policy of its own: it fetches
            # http(s) through get_mm_http_session and resolves file:// to a bare
            # path, so passing a rejected URL on turns "denied" into an
            # unvalidated fetch or local read.
            #
            # Confirmed on GPU hardware before this guard existed: a loopback
            # URL the policy refused was served to SGLang (38128 bytes fetched
            # from a blocked address), and a refused file:// path resolved to a
            # readable local file.
            #
            # UrlValidationError subclasses ValueError, which is how this
            # handler already reports a bad request, so the caller surfaces it
            # as one instead of silently widening what the deployment accepts.
            raise
        except MissingMediaDecoderError:
            # The preflight above is the actionable error this path exists to
            # raise. Letting the broad handler below catch it would return the
            # bytes anyway and reproduce exactly the deep-SGLang failure it
            # replaces.
            raise
        except Exception as exc:  # noqa: BLE001 - additive; never blocks the path
            # If the fetch itself failed there are no bytes and the URL is all
            # the caller has. If it succeeded and only the decoder construction
            # failed, pass the validated bytes on rather than making SGLang
            # fetch them again -- but only if SGLang can actually decode them:
            # this fallback leg reaches SGLang's software path exactly like the
            # non-hardware-codec leg above, so it needs the same preflight, or
            # a host with broken NVDEC and no software decoder gets the deep
            # payload-blob error back.
            if content is not None and not _software_video_decoder_imports():
                raise video_decoder_missing(
                    "sglang", "decord2", "decord", probe_video_codec(content)
                ) from exc
            logger.warning(
                "NVDEC decode failed for video URL (%s); falling back to %s",
                exc,
                "the fetched bytes" if content is not None else "URL passthrough",
            )
            return content

    async def _build_encode_inputs(
        self, urls: list[str], modality_name: str
    ) -> list[Any]:
        """Map video URLs to NVDEC-backed decoders where applicable.

        Returns a list positionally aligned with ``urls``: each entry is an
        ``NvdecVideoDecoder`` (H.264/H.265), the fetched bytes (any other codec,
        so SGLang does not re-download what we already validated and hold), or
        the original URL string when nothing was fetched.
        Non-video modalities return the URLs unchanged. When NVDEC is disabled
        or ineligible the URLs are returned policy-validated and normalized,
        since SGLang fetches them with its own session.

        Called from both the cached and uncached encode paths. The embedding
        cache is disabled by default, so routing this only through the cached
        path would leave hardware decode unreachable in a stock deployment.
        """
        if modality_name != "VIDEO":
            return list(urls)
        if not self._nvdec_video_enabled():
            # NVDEC off (CPU image, DYN_DISABLE_NVDEC, or a gated model type):
            # these URLs go straight to SGLang's software path, which fetches
            # them with its own session and never consults our url policy. Run
            # the policy here so a source we would refuse is refused before
            # SGLang can reach it -- and before we answer with anything about
            # this deployment, since a request we reject is not the place to
            # report which decoders are installed.
            validated = [
                await validate_media_url(url, self._url_policy) for url in urls
            ]
            # Without this preflight these deployments -- the ones MOST likely
            # to lack a decoder entirely -- still get the deep
            # "No module named 'decord'" with the payload repr embedded. No
            # bytes were fetched here, so the codec cannot be named.
            if urls and not _software_video_decoder_imports():
                raise video_decoder_missing("sglang", "decord2", "decord", None)
            return validated
        encode_inputs: list[Any] = []
        for url in urls:
            decoder = await self._maybe_nvdec_decoder(url)
            encode_inputs.append(decoder if decoder is not None else url)
        return encode_inputs

    async def _encode_with_cache(
        self, media_urls: list[str], modality: Any
    ) -> tuple[Any, torch.Tensor, list[CachedEmbedding]]:
        """Cache-aware multimodal encoding.

        Checks the CPU LRU cache per media URL. Uncached URLs are batch-encoded,
        split per item, stored in cache, then reassembled with the cached hits in
        the original URL order.
        """
        assert self._embedding_cache is not None

        modality_name = getattr(modality, "name", str(modality))
        cached: dict[int, CachedEmbedding] = {}
        uncached_indices: list[int] = []
        uncached_urls: list[str] = []

        cache_keys = [
            self._media_cache_key(url, modality, self.encoder) for url in media_urls
        ]

        for i, (url, cache_key) in enumerate(zip(media_urls, cache_keys)):
            hit = self._embedding_cache.get(cache_key)
            if hit is not None:
                logger.info("Embedding cache hit for %s URL index %d", modality_name, i)
                cached[i] = hit
            else:
                uncached_indices.append(i)
                uncached_urls.append(url)

        new_entries: dict[int, CachedEmbedding] = {}
        # SGLang's _encode outputs are already on CPU; use CPU as target for consistency
        target_device = torch.device("cpu")
        if uncached_urls:
            # H.264/H.265 URLs are hardware-decoded to frames here; SGLang's
            # _encode accepts pre-decoded ndarrays in place of URLs. Cache keys
            # above remain keyed on the original URL, so caching is unaffected.
            encode_inputs = await self._build_encode_inputs(
                uncached_urls, modality_name
            )
            grid_dim, new_embeddings, aux_data = await self.encoder._encode(
                encode_inputs, modality
            )
            # Verify SGLang output is on CPU as expected
            if new_embeddings.device != target_device:
                logger.warning(
                    f"SGLang _encode returned embeddings on {new_embeddings.device}, "
                    f"expected CPU. Moving to CPU."
                )
                new_embeddings = new_embeddings.to(target_device)
            grid_list = self._ensure_batched_grid(grid_dim, len(uncached_urls))
            if not (
                isinstance(new_embeddings, torch.Tensor) and new_embeddings.ndim == 2
            ):
                raise ValueError(
                    f"Unsupported embeddings type from encoder: {type(new_embeddings)}"
                )
            token_counts = self._split_token_counts(
                grid_list, new_embeddings.shape[0], modality_name
            )
            split_tensors = torch.split(new_embeddings, token_counts, dim=0)
            item_count = len(grid_list)
            for local_idx, (orig_idx, _url, tensor, grid_thw) in enumerate(
                zip(uncached_indices, uncached_urls, split_tensors, grid_list)
            ):
                entry_kwargs: dict[str, Any] = {"tensor": tensor.contiguous()}
                if modality_name == "IMAGE":
                    entry_kwargs["image_grid_thw"] = grid_thw
                elif modality_name == "VIDEO":
                    entry_kwargs["video_grid_thw"] = grid_thw
                    if aux_data:
                        entry_kwargs["second_per_grid_ts"] = self._aux_value_for_item(
                            aux_data.get("second_per_grid_ts"),
                            local_idx,
                            item_count,
                        )
                        entry_kwargs["video_timestamps"] = self._aux_value_for_item(
                            aux_data.get("video_timestamps"),
                            local_idx,
                            item_count,
                        )
                else:
                    raise ValueError(f"Unsupported multimodal modality: {modality}")
                entry = CachedEmbedding(**entry_kwargs)
                mutation = self._embedding_cache.set_with_delta(
                    cache_keys[orig_idx], entry
                )
                self._publish_cache_delta(mutation.added_keys, mutation.removed_keys)
                new_entries[orig_idx] = entry

        # Reassemble results in original URL order
        all_grid_thw: list = []
        all_entries: list[CachedEmbedding] = []
        embedding_parts: list[torch.Tensor] = []
        for i in range(len(media_urls)):
            entry = cached[i] if i in cached else new_entries[i]
            grid_thw = (
                entry.image_grid_thw
                if modality_name == "IMAGE"
                else entry.video_grid_thw
            )
            if grid_thw is None:
                raise ValueError(
                    f"{modality_name.lower()}_grid_thw is required for cached item"
                )
            all_grid_thw.append(grid_thw)
            all_entries.append(entry)
            embedding_parts.append(entry.tensor)

        full_embeddings = torch.cat(embedding_parts, dim=0)
        return torch.tensor(all_grid_thw), full_embeddings, all_entries

    def _extract_media_urls(
        self, request: Dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """
        Extract image/video URLs from the multi_modal_data field of a PreprocessedRequest.

        The Rust frontend populates multi_modal_data with the format:
            {"image_url": [{"Url": "https://..."}, ...], "video_url": [{"Url": "https://..."}, ...]}

        Multimodal cache UUIDs are rejected before URL extraction because
        SGLang cannot resolve payload-free media slots.

        Returns:
            Tuple of (image_urls, video_urls) lists.
        """
        reject_unsupported_multimodal_uuids(request.get("multi_modal_uuids"))
        mm_data = request.get("multi_modal_data")
        if not mm_data:
            raise ValueError("multi_modal_data is required for the encode worker.")

        image_items = mm_data.get(IMAGE_URL_KEY, [])
        video_items = mm_data.get(VIDEO_URL_KEY, [])

        if not image_items and not video_items:
            raise ValueError(
                "multi_modal_data must contain image_url or video_url entries."
            )

        image_urls: list[str] = []
        video_urls: list[str] = []

        # Extract image URLs
        for item in image_items:
            if isinstance(item, str):
                image_urls.append(item)
            elif isinstance(item, dict) and "Url" in item:
                image_urls.append(item["Url"])
            elif isinstance(item, dict) and "Decoded" in item:
                raise ValueError(
                    "Frontend-decoded media (Decoded variant) is incompatible "
                    "with the multimodal encode worker. The encode worker "
                    "requires image URLs to run vision encoding via MMEncoder. "
                    "Disable --frontend-decoding when using EPD serving."
                )
            else:
                raise ValueError(f"Unsupported image data variant: {item}")

        # Extract video URLs
        for item in video_items:
            if isinstance(item, str):
                video_urls.append(item)
            elif isinstance(item, dict) and "Url" in item:
                video_urls.append(item["Url"])
            elif isinstance(item, dict) and "Decoded" in item:
                raise ValueError(
                    "Frontend-decoded media (Decoded variant) is incompatible "
                    "with the current SGLang EPD video path. Video inputs are "
                    "URL passthrough only in EPD mode and do not accept Decoded "
                    "payloads in the encode worker. Disable --frontend-decoding "
                    "or use URL-based video_url inputs."
                )
            else:
                raise ValueError(f"Unsupported video data variant: {item}")

        return image_urls, video_urls

    @_nvtx.range_decorator("mm:enc:generate", color="blue")
    async def generate(
        self, raw_request: Dict[str, Any], context: Context
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Encode images from a pre-tokenized multimodal request, expand placeholder
        tokens, transfer embeddings via NIXL, and stream PD worker responses.

        The Rust frontend (ModelInput.Tokens) sends a PreprocessedRequest dict
        with token_ids and multi_modal_data. This handler:
        1. Extracts image URLs from multi_modal_data.
        2. Runs vision encoding via MMEncoder.
        3. Expands image placeholder tokens to match patch counts.
        4. Creates a NIXL descriptor for embedding transfer.
        5. Forwards the request to the PD worker and streams responses back.

        Args:
            raw_request: PreprocessedRequest dict from the Rust frontend.
            context: Context object for cancellation handling.
        """
        if isinstance(raw_request, str):
            raw_request = json.loads(raw_request)

        # Extract image/video URLs from the frontend's multi_modal_data
        image_urls, video_urls = self._extract_media_urls(raw_request)
        preprocessed_request = PreprocessedRequest.model_validate(raw_request)
        allowed_oov_ids = frozenset(
            token_id
            for media_inputs, token_id in (
                (image_urls, self.image_token_id),
                (video_urls, self.video_token_id),
            )
            if media_inputs and token_id is not None
        )
        self._validate_token_ids(preprocessed_request.token_ids, allowed_oov_ids)

        # Build MultiModalGroup objects for the downstream SglangMultimodalRequest.
        multimodal_groups = [
            MultiModalGroup(multimodal_input=MultiModalInput(image_url=url))
            for url in image_urls
        ] + [
            MultiModalGroup(multimodal_input=MultiModalInput(video_url=url))
            for url in video_urls
        ]
        # Build SglangMultimodalRequest from the pre-tokenized request
        request = SglangMultimodalRequest(
            request=preprocessed_request,
            multimodal_inputs=multimodal_groups,
        )

        try:
            transfer_future = None
            combined_embeddings_parts: list[torch.Tensor] = []

            # Build modality-local metadata in the same order as multimodal_groups.
            modality_specs = [
                (
                    "IMAGE",
                    image_urls,
                    Modality.IMAGE,
                    self.image_token_id,
                    "image_grid_thw",
                    "image_url",
                ),
                (
                    "VIDEO",
                    video_urls,
                    Modality.VIDEO,
                    self.video_token_id,
                    "video_grid_thw",
                    "video_url",
                ),
            ]

            group_offset = 0
            for (
                modality_name,
                urls,
                modality_enum,
                token_id,
                grid_attr,
                url_attr,
            ) in modality_specs:
                if not urls:
                    continue
                if token_id is None:
                    raise ValueError(
                        f"{modality_name.lower()} token is not defined in chat template"
                    )

                aux_data: dict[str, Any] | None = None
                cached_entries: list[CachedEmbedding] | None = None
                with _nvtx.annotate("mm:enc:vision_encode", color="red"):
                    if self._embedding_cache is not None:
                        (
                            grid_dim,
                            embeddings,
                            cached_entries,
                        ) = await self._encode_with_cache(urls, modality_enum)
                    else:
                        # The embedding cache is off by default, so this is the
                        # path a stock deployment takes. It must apply the same
                        # NVDEC conversion as the cached branch -- otherwise raw
                        # URLs reach SGLang, which has no video decoder in the
                        # codec-compliant image, and every H.264/H.265 request
                        # fails. Not hoisted above the branch because
                        # _encode_with_cache converts only its uncached subset.
                        encode_inputs = await self._build_encode_inputs(
                            urls, modality_name
                        )
                        grid_dim, embeddings, aux_data = await self.encoder._encode(
                            encode_inputs, modality_enum
                        )

                grid_list = self._ensure_batched_grid(grid_dim, len(urls))

                if not isinstance(grid_list, list) or len(grid_list) != len(urls):
                    raise ValueError(
                        f"{modality_name.lower()} grid size mismatch: "
                        f"expected {len(urls)} items, got {grid_list}"
                    )

                if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                    raise ValueError(
                        f"Unsupported embeddings type from encoder: {type(embeddings)}"
                    )

                token_counts = self._split_token_counts(
                    grid_list, embeddings.shape[0], modality_name
                )

                placeholder_count = request.request.token_ids.count(token_id)
                if placeholder_count < len(urls):
                    raise ValueError(
                        f"Not enough {modality_name.lower()} placeholders in token_ids"
                    )

                group_slice = multimodal_groups[group_offset : group_offset + len(urls)]
                for idx, (mm_group, grid_item, token_count) in enumerate(
                    zip(group_slice, grid_list, token_counts)
                ):
                    setattr(mm_group, grid_attr, grid_item)
                    mm_group.num_mm_tokens = int(token_count)
                    if modality_name == "VIDEO":
                        if cached_entries is not None:
                            mm_group.second_per_grid_ts = cached_entries[
                                idx
                            ].second_per_grid_ts
                            mm_group.video_timestamps = cached_entries[
                                idx
                            ].video_timestamps
                        elif aux_data:
                            mm_group.second_per_grid_ts = self._aux_value_for_item(
                                aux_data.get("second_per_grid_ts"),
                                idx,
                                len(urls),
                            )
                            mm_group.video_timestamps = self._aux_value_for_item(
                                aux_data.get("video_timestamps"),
                                idx,
                                len(urls),
                            )
                    if mm_group.multimodal_input is not None:
                        setattr(mm_group.multimodal_input, url_attr, None)

                search_start = 0
                for num_tokens in token_counts:
                    try:
                        token_index = request.request.token_ids.index(
                            token_id, search_start
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Not enough {modality_name.lower()} tokens found for provided inputs"
                        ) from e

                    request.request.token_ids = (
                        request.request.token_ids[:token_index]
                        + [token_id] * num_tokens
                        + request.request.token_ids[token_index + 1 :]
                    )
                    search_start = token_index + num_tokens

                combined_embeddings_parts.append(embeddings)
                group_offset += len(urls)

            if combined_embeddings_parts:
                precomputed_embeddings = torch.cat(combined_embeddings_parts, dim=0)
                request.embeddings_shape = tuple(precomputed_embeddings.shape)  # type: ignore[assignment]
                request.transfer_payload = None

                with _nvtx.annotate("mm:enc:embedding_transfer", color="purple"):
                    (
                        transfer_request,
                        transfer_future,
                    ) = await self.embedding_sender.send_embeddings(
                        precomputed_embeddings
                    )
                    request.transfer_payload = transfer_request
            # Get the response generator from downstream worker
            payload = request.model_dump_json()
            response_generator = await self.pd_worker_client.round_robin(
                payload, context=context
            )

            # Parse PD worker responses and yield as LLMEngineOutput-
            # compatible dicts for the Rust frontend to post-process.
            async for response in response_generator:
                raw = response.data() if hasattr(response, "data") else str(response)
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    logger.warning("Non-JSON response from PD worker: %r", raw[:200])
                    data = {"token_ids": [], "text": raw}
                # Strip the internal 'finished' flag — the Rust frontend
                # uses 'finish_reason' (present when finished=True).
                data.pop("finished", None)
                # Remove empty 'text' so the Rust frontend detokenizes
                # from token_ids instead of using the empty string.
                if not data.get("text"):
                    data.pop("text", None)
                yield data

            if transfer_future is not None:
                await transfer_future

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
