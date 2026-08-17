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


def set_pre_publish_server_args(server_args: Any, **fields: Any) -> None:
    """Update resolved SGLang ServerArgs before runtime publication."""
    # SGLang development releases after 0.5.16 require pre-publish updates here.
    late_resolution = getattr(server_args, "_late_resolution", None)
    if callable(late_resolution):
        late_resolution("dynamo", **fields)
        return

    # SGLang 0.5.16 exposes override() as its post-resolution mutation API;
    # remove when 0.5.16 leaves the support window.
    override = getattr(server_args, "override", None)
    if callable(override):
        override("dynamo", **fields)
        return

    # Fallback for SGLang <= 0.5.15; remove when it leaves the support window.
    for name, value in fields.items():
        setattr(server_args, name, value)


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

    SGLang 0.5.13 through 0.5.16 assume every decoded image exposes the PIL
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


__all__ = [
    "ensure_sglang_tensor_image_size",
    "ensure_sglang_top_level_exports",
    "filter_supported_async_generate_kwargs",
    "require_reasoning_kwargs",
    "set_pre_publish_server_args",
]
