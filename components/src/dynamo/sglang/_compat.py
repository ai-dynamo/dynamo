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

try:
    from sglang.srt.arg_groups.overrides import declare_late_resolution
except ImportError:
    # SGLang 0.5.17 and the XPU 0.5.11 pin predate declarations.
    declare_late_resolution = None

try:
    from sglang.srt.arg_groups.overrides import resolved_view as sglang_resolved_view
except ImportError:
    # SGLang #36255 exposes ServerArgs._resolved() instead.
    sglang_resolved_view = None

try:
    from sglang.srt.arg_groups.overrides import (
        model_config_of as sglang_model_config_of,
    )
except ImportError:
    # Fallback for sglang <= 0.5.18, which exposes ServerArgs.get_model_config().
    # Remove when min supported version has the accessor move (sgl #36972).
    sglang_model_config_of = None

try:
    from sglang.srt.arg_groups.overrides import (
        use_mla_backend as sglang_use_mla_backend,
    )
except ImportError:
    # Fallback for sglang <= 0.5.18, which exposes ServerArgs.use_mla_backend().
    # Remove when min supported version has the accessor move (sgl #36972).
    sglang_use_mla_backend = None

logger = logging.getLogger(__name__)

try:
    from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger
except ModuleNotFoundError as exc:
    if exc.name == "sglang.srt.utils.server_args_config_parser":
        # Keep the CUDA 0.5.18 and XPU 0.5.11 pins working until both move here.
        from sglang.srt.server_args_config_parser import ConfigArgumentMerger
    elif exc.name == "sglang" or (exc.name or "").startswith("sglang."):
        # SGLang absent entirely: degrade to None like every other SGLang import
        # here, so this module's SGLang-free code paths stay importable.
        ConfigArgumentMerger = None  # type: ignore[assignment,misc]
    else:
        raise


def get_sglang_model_config(server_args: Any) -> Any:
    """Return the resolved model config across SGLang ServerArgs APIs.

    SGLang #36972 moved ``ServerArgs.get_model_config()`` to the module-level
    ``model_config_of()``. Remove the legacy branch when the minimum supported
    SGLang release contains that move.
    """
    legacy_getter = getattr(server_args, "get_model_config", None)
    if legacy_getter is not None:
        return legacy_getter()
    if sglang_model_config_of is None:
        raise AttributeError("SGLang does not expose a model config accessor")
    return sglang_model_config_of(server_args)


def sglang_uses_mla_backend(server_args: Any) -> bool:
    """Return whether this configuration selects SGLang's MLA attention backend.

    SGLang #36972 moved ``ServerArgs.use_mla_backend()`` to the module-level
    ``use_mla_backend()``. Remove the legacy branch when the minimum supported
    SGLang release contains that move.
    """
    legacy_getter = getattr(server_args, "use_mla_backend", None)
    if legacy_getter is not None:
        return bool(legacy_getter())
    if sglang_use_mla_backend is None:
        raise AttributeError("SGLang does not expose an MLA backend accessor")
    return bool(sglang_use_mla_backend(server_args))


@lru_cache(maxsize=1)
def _warn_require_reasoning_unsupported() -> None:
    logger.warning(
        "Dropping require_reasoning=true because SGLang Engine.async_generate "
        "does not support it; reasoning-aware guided decoding may fail. "
        "Upgrade SGLang to enable this request mode."
    )


def ensure_sglang_tensor_image_size() -> None:
    """Allow SGLang's image-token resolver to handle decoded image tensors.

    SGLang 0.5.13 through 0.5.18 assume every decoded image exposes the PIL
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


def _already_normalized() -> None:
    """Stand in for a request's normalizer once the shim has already run it."""


def _normalize_generate_request_once(obj: Any) -> None:
    """Run SGLang's request normalization exactly once for this request object.

    ``TokenizerManager.generate_request`` calls ``normalize_batch_and_arguments()``
    unconditionally, and that call is not idempotent: for parallel sampling a
    second pass re-derives the batch size from the already-expanded input list
    and expands it again, turning ``n`` sequences into ``n * n``. Normalizing
    early is therefore only safe if SGLang's own later call is neutralized.
    """
    if hasattr(obj, "batch_size"):
        # Already normalized, or this release assigns batch_size itself.
        # Re-normalizing here would cause the double expansion described above.
        return

    normalize = getattr(obj, "normalize_batch_and_arguments", None)
    if normalize is None:
        return

    # A failed pass has already mutated the request, so let the error propagate
    # rather than leave SGLang to normalize that partial state a second time.
    normalize()

    obj.normalize_batch_and_arguments = _already_normalized


def _report_rejected_request(
    bridge: Any, exc: Exception, args: Any, kwargs: Any
) -> bool:
    """Answer a rejected request through the bridge's native error callback.

    Reports ``False`` when the release exposes no way to do so, leaving the
    caller to raise: worse for the client, which then waits out its deadline,
    but the only remaining way to make the failure visible at all.
    """
    send_native_error = getattr(bridge, "_send_native_error", None)
    chunk_callback = args[0] if args else kwargs.get("chunk_callback")
    if send_native_error is None or chunk_callback is None:
        return False

    send_native_error(chunk_callback, str(exc))
    return True


_GRPC_BRIDGE_MODULE = "sglang.srt.entrypoints.grpc_bridge"


def _is_absent_grpc_bridge(exc: ImportError) -> bool:
    """Report whether an import failure means the native gRPC bridge is absent.

    A release that does not ship the bridge, and one that renamed
    ``RuntimeHandle``, both name the bridge module in ``exc.name``; an
    environment without SGLang at all names one of its parent packages. Any
    other name -- a dependency the bridge itself imports -- means the bridge is
    present but failed to load, which is a broken install rather than a release
    that does not need the override.
    """
    name = getattr(exc, "name", None)
    if not name:
        return False
    return name == _GRPC_BRIDGE_MODULE or _GRPC_BRIDGE_MODULE.startswith(f"{name}.")


def ensure_sglang_grpc_bridge_batch_size(bridge_class: Any = None) -> None:
    """Normalize gRPC generate requests before SGLang reads their batch size.

    SGLang 0.5.17 and 0.5.18 read ``obj.batch_size`` in the streaming branch of
    the native gRPC bridge's ``_run_generate``, one statement after calling
    ``TokenizerManager.generate_request``. That call only builds an async
    generator, so its body -- and with it the ``normalize_batch_and_arguments()``
    call that assigns ``batch_size`` -- has not run yet. Every streaming request
    over the native gRPC server therefore fails with ``'GenerateReqInput' object
    has no attribute 'batch_size'``, which the sidecar surfaces as HTTP 500.
    Normalizing first restores the ordering the upstream code assumes.

    This only affects engines launched through ``dynamo.sglang.launch_server``.
    A stock engine that does not need the shim must still start, so an absent
    or already-fixed bridge is logged at debug level and left alone. An SGLang
    that is installed but fails to import raises: that is a broken environment,
    not a release outside the support window.

    ``bridge_class`` patches an explicit class instead of importing SGLang's;
    tests use it to exercise the wrapper without SGLang installed.

    Remove this override once the minimum supported SGLang release normalizes
    before reading ``batch_size`` -- that is, once both 0.5.17 and 0.5.18 fall
    outside the support window.
    """
    if bridge_class is None:
        try:
            from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
        except ImportError as exc:
            if not _is_absent_grpc_bridge(exc):
                raise
            logger.debug(
                "SGLang does not expose a native gRPC bridge; "
                "skipping the batch_size normalization override"
            )
            return
        bridge_class = RuntimeHandle

    original = getattr(bridge_class, "_run_generate", None)
    if original is None or getattr(
        original, "_dynamo_grpc_bridge_batch_size_support", False
    ):
        return

    @wraps(original)
    async def _run_generate(self: Any, obj: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            _normalize_generate_request_once(obj)
        except Exception as exc:
            # A rejected request is normally answered from inside _run_generate's
            # own handler. Normalizing early moves that failure outside it, and an
            # exception escaping this coroutine would only reach the scheduler's
            # logger -- leaving the client to wait out its deadline instead of
            # receiving the error. Report it the same way, and do not run the
            # request: the failed pass has already half-expanded it.
            if not _report_rejected_request(self, exc, args, kwargs):
                raise
            logger.error(
                "gRPC generate error for rid=%s: %s", getattr(obj, "rid", None), exc
            )
            return None

        return await original(self, obj, *args, **kwargs)

    _run_generate._dynamo_grpc_bridge_batch_size_support = True  # type: ignore[attr-defined]
    bridge_class._run_generate = _run_generate


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Declare launcher-stage SGLang configuration fields.

    SGLang 0.5.18+ resolves its effective configuration separately from raw
    ``ServerArgs`` input. Declare pre-engine changes through its resolution API
    so the engine's resolved projection observes them. SGLang 0.5.17 exposes
    ``ServerArgs.override`` instead. The separately pinned XPU image still uses
    SGLang 0.5.11, which predates both APIs; preserve its legacy assignment
    behavior until its engine pin is upgraded.
    """
    if declare_late_resolution is not None:
        declare_late_resolution(server_args, source, **fields)
        return

    late_resolution = getattr(server_args, "_late_resolution", None)
    if callable(late_resolution):
        late_resolution(source, **fields)
        return

    # Fallback for SGLang 0.5.17. Remove when minimum supported SGLang is 0.5.18+.
    override = getattr(server_args, "override", None)
    if callable(override):
        override(source, **fields)
        return

    # XPU compatibility for SGLang 0.5.11. Remove when the XPU SGLang pin is
    # upgraded to 0.5.16+.
    for name, value in fields.items():
        setattr(server_args, name, value)


def resolved_server_args(server_args: Any) -> Any:
    """Return SGLang's effective configuration for one initialized engine.

    SGLang #36255 exposes ``ServerArgs._resolved()``. Current SGLang keeps
    ``ServerArgs`` raw and exposes the same projection through
    ``resolved_view()``. Older supported releases and Dynamo's non-LLM argument
    stubs retain effective values on the object itself.
    """
    resolve = getattr(server_args, "_resolved", None)
    if callable(resolve):
        return resolve()
    if sglang_resolved_view is not None:
        return sglang_resolved_view(server_args)
    return server_args


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

    Both supported CUDA releases accept Dynamo's optional kwargs. The separately
    pinned XPU image still uses SGLang 0.5.11, which predates ``mm_hashes`` and
    ``require_reasoning``. Keep the compatibility boundary narrow: callers
    decide which kwargs are optional, and this helper only drops those optional
    kwargs when the installed engine cannot accept them. Remove this filtering
    when the XPU SGLang pin is upgraded to 0.5.16+.
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
    "ConfigArgumentMerger",
    "ensure_sglang_grpc_bridge_batch_size",
    "ensure_sglang_tensor_image_size",
    "filter_supported_async_generate_kwargs",
    "get_sglang_model_config",
    "override_server_args",
    "require_reasoning_kwargs",
    "resolved_server_args",
    "sglang_uses_mla_backend",
]
