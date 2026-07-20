# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shim for SGLang internal APIs.

SGLang is pre-1.0 and routinely moves, renames, or introduces APIs between
releases. This module is the single place where we handle those differences
so the rest of the component can import from here without version-specific
try/except blocks.

Policy: support current SGLang release + 1 version back (N and N-1). Each
fallback branch must document which version it covers and when it can be
removed. When the old version falls outside the support window, delete the
fallback and any associated polyfills.

Runtime data-contract notes (not code-level shims):

* ``meta_info["routed_experts"]`` is a base64 UTF-8 string from sglang
  >= 0.5.11. Pass through; do not re-encode.
"""

import inspect
import logging
from collections.abc import Mapping
from functools import lru_cache, wraps
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warn_require_reasoning_unsupported() -> None:
    logger.warning(
        "Dropping require_reasoning=true because SGLang Engine.async_generate "
        "does not support it; reasoning-aware guided decoding may fail. "
        "Upgrade SGLang to enable this request mode."
    )


# ---------------------------------------------------------------------------
# Top-level sglang exports: Engine, ServerArgs
#
# Some SGLang dev builds (including 0.5.x snapshots) do not re-export these
# from sglang/__init__.py, while Dynamo historically uses `import sglang as sgl`
# followed by `sgl.Engine(...)` throughout this backend.
# ---------------------------------------------------------------------------
def ensure_sglang_top_level_exports() -> None:
    """Restore top-level SGLang exports omitted by some install flavors."""
    import sglang as sgl

    if not hasattr(sgl, "Engine"):
        from sglang.srt.entrypoints.engine import Engine

        sgl.Engine = Engine

    if not hasattr(sgl, "ServerArgs"):
        from sglang.srt.server_args import ServerArgs

        sgl.ServerArgs = ServerArgs


ensure_sglang_top_level_exports()


def ensure_sglang_tensor_image_size() -> None:
    """Allow SGLang's image-token resolver to handle decoded image tensors.

    SGLang 0.5.13 through 0.5.15 assume every decoded image exposes the PIL
    ``height``/``width`` attributes. Its CUDA JPEG decoder instead returns a
    CHW tensor, causing multimodal requests to fall back to retokenization.

    Remove this compatibility override once the minimum supported SGLang
    release handles tensor image dimensions itself.
    """
    import torch
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

    original = getattr(BaseMultimodalProcessor, "resolve_image_token_counts", None)
    if original is None or getattr(
        original, "_dynamo_tensor_image_size_support", False
    ):
        return

    @wraps(original)
    def resolve_image_token_counts(self: Any, images: list[Any]) -> list[int]:
        if not any(isinstance(image, torch.Tensor) for image in images):
            return original(self, images)

        image_sizes: list[tuple[int, int]] = []
        for image in images:
            if isinstance(image, torch.Tensor):
                if image.ndim < 2:
                    raise ValueError(f"Invalid image tensor shape: {image.shape}")
                height, width = image.shape[-2:]
            else:
                height, width = image.height, image.width
            image_sizes.append((int(height), int(width)))

        token_counts = self._processor._get_num_multimodal_tokens(
            image_sizes=image_sizes
        ).num_image_tokens
        return [int(count) for count in token_counts]

    resolve_image_token_counts._dynamo_tensor_image_size_support = True  # type: ignore[attr-defined]
    BaseMultimodalProcessor.resolve_image_token_counts = resolve_image_token_counts


def ensure_sglang_external_mm_hash_updates_pad_value() -> None:
    """Keep caller-supplied MM hashes authoritative for processor outputs.

    SGLang 0.5.15 may compute ``pad_value`` from a precomputed embedding before
    its tokenizer manager applies ``mm_hashes``. Later ``set_pad_value`` calls
    then return early with the stale value. Refresh that value from the current
    hash so E/PD processor-output requests use the same prefix-cache identity as
    Dynamo's frontend router. Remove this shim once SGLang clears/recomputes the
    pad value when it accepts a caller hash.
    """
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    from dynamo.common.multimodal.routing_utils import pad_value_for_mm_hash

    original = MultimodalDataItem.set_pad_value
    if getattr(original, "_dynamo_external_mm_hash_refresh", False):
        return

    @wraps(original)
    def set_pad_value(self: Any) -> None:
        if self.hash is not None and self.pad_value is not None:
            expected = pad_value_for_mm_hash(self.hash)
            if self.pad_value != expected:
                self.pad_value = expected
                return
        original(self)

    set_pad_value._dynamo_external_mm_hash_refresh = True  # type: ignore[attr-defined]
    MultimodalDataItem.set_pad_value = set_pad_value


def ensure_sglang_frontend_decoded_video_support() -> None:
    """Keep frontend-sampled videos unchanged in SGLang's encode server.

    SGLang 0.5.15 treats every ``VideoDecoderWrapper`` as an unsampled source:
    it samples and resizes the frames before invoking the model's HF video
    processor. Dynamo has already selected the model-visible frames, so that
    path can change both timestamps and the vision token grid. Preserve the
    transferred RGB frames and their source indices until SGLang exposes a
    native pre-sampled-video contract.
    """
    from sglang.srt.disaggregation import encode_server
    from sglang.srt.multimodal.processors import qwen_vl

    from dynamo.sglang.request_handlers.multimodal.video_input import (
        FrontendDecodedVideo,
    )

    original = qwen_vl.preprocess_video
    if getattr(original, "_dynamo_frontend_decoded_video_support", False):
        encode_server.preprocess_video = original
        return

    @wraps(original)
    async def preprocess_video(video: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(video, FrontendDecodedVideo):
            return video.as_processor_input()
        return await original(video, *args, **kwargs)

    preprocess_video._dynamo_frontend_decoded_video_support = True  # type: ignore[attr-defined]
    qwen_vl.preprocess_video = preprocess_video
    encode_server.preprocess_video = preprocess_video


@lru_cache(maxsize=32)
def _get_async_generate_supported_kwarg_names(
    async_generate: Any,
) -> frozenset[str] | None:
    """Return supported async_generate keyword names, or None for **kwargs."""
    try:
        signature = inspect.signature(async_generate)
    except (TypeError, ValueError):
        logger.debug(
            "Could not inspect SGLang Engine.async_generate signature; "
            "dropping optional compatibility kwargs"
        )
        return frozenset()

    names: set[str] = set()
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(name)

    return frozenset(names)


def filter_supported_async_generate_kwargs(
    engine: Any, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return only async_generate kwargs accepted by this SGLang engine.

    SGLang occasionally adds optional Engine.async_generate kwargs before every
    supported install flavor has them. Keep the compatibility boundary narrow:
    callers decide which kwargs are optional, and this helper only drops those
    optional kwargs when the installed engine cannot accept them.
    """
    async_generate = engine.async_generate
    signature_source = getattr(async_generate, "__func__", async_generate)

    try:
        supported_kwarg_names = _get_async_generate_supported_kwarg_names(
            signature_source
        )
    except TypeError:
        supported_kwarg_names = _get_async_generate_supported_kwarg_names.__wrapped__(
            signature_source
        )

    if supported_kwarg_names is None:
        return kwargs

    return {key: value for key, value in kwargs.items() if key in supported_kwarg_names}


def require_reasoning_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Build the optional SGLang per-request reasoning-gate argument."""
    require_reasoning = bool(request.get("require_reasoning", False))
    kwargs = filter_supported_async_generate_kwargs(
        engine,
        {"require_reasoning": require_reasoning},
    )
    if require_reasoning and "require_reasoning" not in kwargs:
        _warn_require_reasoning_unsupported()
    return kwargs


@lru_cache(maxsize=32)
def _start_profile_accepts_request_object(start_profile: Any) -> bool:
    """Return whether TokenizerManager.start_profile expects a ProfileReq."""
    try:
        signature = inspect.signature(start_profile)
    except (TypeError, ValueError):
        logger.debug(
            "Could not inspect SGLang TokenizerManager.start_profile signature; "
            "using the legacy keyword-argument API"
        )
        return False

    return "req" in signature.parameters


def _build_profile_request(body: dict[str, Any]) -> Any:
    from sglang.srt.managers.io_struct import ProfileReq

    return ProfileReq(**body)


async def start_profile_compat(tokenizer_manager: Any, body: dict[str, Any]) -> None:
    """Start profiling across SGLang's old and new control APIs.

    SGLang 0.5.14 accepts profiling fields as keyword arguments. SGLang 0.5.15
    accepts one ``ProfileReq`` object instead.
    """
    start_profile = tokenizer_manager.start_profile
    signature_source = getattr(start_profile, "__func__", start_profile)

    try:
        accepts_request_object = _start_profile_accepts_request_object(signature_source)
    except TypeError:
        accepts_request_object = _start_profile_accepts_request_object.__wrapped__(
            signature_source
        )

    if accepts_request_object:
        await start_profile(_build_profile_request(body))
    else:
        await start_profile(**body)


def enable_disjoint_streaming_output(server_args: Any) -> None:
    """Enable SGLang's disjoint streaming output.

    Diffusion workers pass a ``SimpleNamespace`` stub that does not carry the
    field, so this is a no-op when the attribute is absent.
    """
    if hasattr(server_args, "incremental_streaming_output"):
        server_args.incremental_streaming_output = True


__all__ = [
    "enable_disjoint_streaming_output",
    "ensure_sglang_external_mm_hash_updates_pad_value",
    "ensure_sglang_frontend_decoded_video_support",
    "ensure_sglang_tensor_image_size",
    "ensure_sglang_top_level_exports",
    "filter_supported_async_generate_kwargs",
    "require_reasoning_kwargs",
    "start_profile_compat",
]
