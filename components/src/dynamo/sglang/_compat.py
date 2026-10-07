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

import argparse
import importlib
import inspect
import logging
import uuid
from collections.abc import Mapping
from functools import lru_cache
from types import ModuleType
from typing import Any

from dynamo.llm.exceptions import InvalidArgument

try:
    from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger
except ModuleNotFoundError as exc:
    if exc.name != "sglang.srt.utils.server_args_config_parser":
        raise
    # Fallback for the separately pinned XPU SGLang 0.5.11.
    # Remove when the XPU pin is upgraded to 0.5.19+.
    from sglang.srt.server_args_config_parser import ConfigArgumentMerger

try:
    from sglang.srt.arg_groups.overrides import (
        model_config_of as sglang_model_config_of,
    )
except ImportError:
    # Fallback for XPU SGLang 0.5.11, which exposes ServerArgs.get_model_config().
    # Remove when the XPU pin is upgraded to 0.5.19+.
    sglang_model_config_of = None

try:
    from sglang.srt.arg_groups.overrides import (
        use_mla_backend as sglang_use_mla_backend,
    )
except ImportError:
    # Fallback for XPU SGLang 0.5.11, which exposes ServerArgs.use_mla_backend().
    # Remove when the XPU pin is upgraded to 0.5.19+.
    sglang_use_mla_backend = None

try:
    from sglang.srt.runtime_context import publish as _sglang_publish
except ImportError:
    # Fallback for the XPU SGLang 0.5.11 pin.
    # Remove when the XPU pin is upgraded to 0.5.19+.
    _sglang_publish = None

try:
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
except ImportError:
    # Fail closed for downstream SGLang builds that omit request-time
    # statistics or dispatch timestamps.
    APIServerReqTimeStats = None


def supports_disagg_prefill_cancel_anytime(engine: Any) -> bool:
    """Return whether aborts can be ordered after scheduler dispatch."""
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    if not isinstance(getattr(tokenizer_manager, "rid_to_state", None), Mapping):
        return False
    if APIServerReqTimeStats is None:
        return False
    fields = getattr(APIServerReqTimeStats, "__dataclass_fields__", {})
    return "api_server_dispatch_finish_time" in fields


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


def publish_server_args(server_args: Any, *, role: str) -> None:
    """Publish process-wide SGLang configuration when the API is available."""
    if _sglang_publish is not None:
        _sglang_publish(server_args, role=role)


try:
    from sglang.srt.arg_groups.overrides import declare_resolution
except ImportError:
    # The separately pinned XPU SGLang 0.5.11 predates declarations.
    # Remove when that pin is upgraded to 0.5.19+.
    declare_resolution = None

try:
    from sglang.srt.arg_groups.model_override_base import (
        resolved_view as sglang_resolved_view,
    )
except ImportError:
    # The separately pinned XPU SGLang 0.5.11 stores effective values on
    # ServerArgs directly. Remove when that pin is upgraded to 0.5.19+.
    sglang_resolved_view = None

logger = logging.getLogger(__name__)


def add_sglang_cli_compat(parser: argparse.ArgumentParser) -> None:
    """Keep launch scripts compatible with SGLang's renamed graph options."""
    legacy_flag = "--disable-piecewise-cuda-graph"
    options = parser._option_string_actions
    if legacy_flag in options or "--cuda-graph-backend-prefill" not in options:
        return
    # SGLang 0.5.20 removed the alias supplied by 0.5.19. Preserve its exact
    # translation while launch scripts also support the XPU 0.5.11 pin.
    # Remove when all supported pins accept --cuda-graph-backend-prefill and
    # the launch scripts have migrated to that spelling.
    parser.add_argument(
        legacy_flag,
        dest="cuda_graph_backend_prefill",
        action="store_const",
        const="disabled",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def get_mm_encoder_class() -> type[Any]:
    """Load MMEncoder from the supported SGLang package layout.

    Keep this import deferred because the encoder module imports compiled CUDA
    operators and this compatibility module is also collected on CPU-only CI
    hosts.
    """
    try:
        from sglang.srt.disaggregation.encoder.server import MMEncoder
    except ImportError:
        # Fallback for XPU SGLang 0.5.11.
        # Remove when the XPU pin is upgraded to 0.5.19+.
        from sglang.srt.disaggregation.encode_server import MMEncoder

    return MMEncoder


def get_encoder_preprocessor_modules() -> tuple[ModuleType, ...]:
    """Return importable encoder modules that bind video preprocessing APIs."""
    modules: list[ModuleType] = []
    for module_path in (
        "sglang.srt.disaggregation.encoder.preprocessor",
        # Fallback for XPU SGLang 0.5.11.
        # Remove when the XPU pin is upgraded to 0.5.19+.
        "sglang.srt.disaggregation.encode_server",
    ):
        try:
            modules.append(importlib.import_module(module_path))
        except (ImportError, OSError):
            continue
    return tuple(modules)


async def mm_encode(
    encoder: Any, media_inputs: list[Any], modality: Any
) -> tuple[Any, Any, dict[str, Any]]:
    """Encode media across the supported SGLang MMEncoder APIs."""
    legacy_encode = getattr(encoder, "_encode", None)
    if callable(legacy_encode):
        # Fallback for XPU SGLang 0.5.11.
        # Remove when the XPU pin is upgraded to 0.5.19+.
        return await legacy_encode(media_inputs, modality)

    prepare = getattr(encoder, "_prepare_encode_context", None)
    compute = getattr(encoder, "_compute_embedding", None)
    if not callable(prepare) or not callable(compute):
        raise RuntimeError("SGLang MMEncoder does not expose an encode API")

    request = {
        "req_id": f"dynamo-direct-{uuid.uuid4()}",
        "num_parts": 1,
        "part_idx": 0,
        "mm_items": media_inputs,
        "hashes": None,
    }
    encode_context = await prepare(
        [request],
        modality,
        use_global_cache=False,
    )
    embeddings = await compute(encode_context, keep_on_gpu=False)
    if embeddings is None:
        raise RuntimeError("SGLang MMEncoder returned no embeddings")
    return (
        encode_context.preprocess_result.grid_thw,
        embeddings,
        encode_context.aux_data,
    )


@lru_cache(maxsize=1)
def _warn_require_reasoning_unsupported() -> None:
    logger.warning(
        "Dropping require_reasoning=true because SGLang Engine.async_generate "
        "does not support it; reasoning-aware guided decoding may fail. "
        "Upgrade SGLang to enable this request mode."
    )


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Declare launcher-stage SGLang configuration fields.

    SGLang 0.5.18+ resolves its effective configuration separately from raw
    ``ServerArgs`` input. Declare pre-engine changes through its resolution API
    so the engine's resolved projection observes them. The separately pinned
    XPU image still uses SGLang 0.5.11, which predates that API; preserve its
    legacy assignment behavior until its engine pin is upgraded.
    """
    if declare_resolution is not None:
        declare_resolution(server_args, source, **fields)
        return

    # XPU compatibility for SGLang 0.5.11. Remove when the XPU SGLang pin is
    # upgraded to 0.5.16+.
    for name, value in fields.items():
        setattr(server_args, name, value)


def resolved_server_args(server_args: Any) -> Any:
    """Return SGLang's effective configuration for one initialized engine.

    SGLang 0.5.20 and 0.5.21 keep ``ServerArgs`` raw and expose the effective
    projection through ``resolved_view()``. The separately pinned XPU release
    and Dynamo's non-LLM argument stubs retain effective values on the object
    itself.
    """
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


def supports_external_mm_hashes(engine: Any) -> bool:
    """Enable safe caller-provided MM hashes when this SGLang accepts them.

    Supported SGLang releases apply caller hashes after some processors have
    already built ``padded_input_ids``. Rebuild that derived field after
    tokenization so the external hash, item pad value, and padded IDs remain
    consistent. The repair is idempotent if upstream already rebuilt them.
    """
    if "mm_hashes" not in filter_supported_async_generate_kwargs(
        engine, {"mm_hashes": None}
    ):
        return False

    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    tokenize_one = getattr(tokenizer_manager, "_tokenize_one_request", None)
    if tokenize_one is None or getattr(
        tokenize_one, "_dynamo_rebuilds_external_mm_padding", False
    ):
        return True

    # Deferred: schedule_batch imports torch and other SGLang runtime modules.
    from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput

    async def tokenize_with_consistent_mm_padding(obj):
        tokenized = await tokenize_one(obj)
        mm_inputs = getattr(tokenized, "mm_inputs", None)
        if getattr(obj, "mm_hashes", None) and mm_inputs is not None:
            padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
                tokenized.input_ids,
                mm_inputs.mm_items,
            )
            if padded_input_ids is not None:
                mm_inputs.padded_input_ids = padded_input_ids
        return tokenized

    setattr(
        tokenize_with_consistent_mm_padding,
        "_dynamo_rebuilds_external_mm_padding",
        True,
    )
    assert tokenizer_manager is not None
    tokenizer_manager._tokenize_one_request = tokenize_with_consistent_mm_padding

    return True


def cache_salt_kwargs(engine: Any, cache_salt: str | None) -> dict[str, Any]:
    """Preserve cache isolation, rejecting salts an older engine cannot honor."""
    if not cache_salt:
        return {}
    # SGLang 0.5.11 in the XPU image lacks cache_salt. Remove this check when
    # the XPU pin supports the explicit cache_salt argument (0.5.18+).
    kwargs = filter_supported_async_generate_kwargs(engine, {"cache_salt": cache_salt})
    if "cache_salt" not in kwargs:
        raise ValueError("cache_salt is not supported by the installed SGLang engine")
    return kwargs


def prefill_dp_rank_kwargs(engine: Any, prefill_dp_rank: Any) -> dict[str, Any]:
    """Hand SGLang's decode the prefill DP rank the router already chose.

    Without ``disagg_prefill_dp_rank`` the decode scheduler parks the request
    and resolves the rank over HTTP against the prefill bootstrap server, which
    only learns the room once the prefill scheduler has created its KV sender;
    the prefill forward cannot start before that round trip completes.
    """
    if prefill_dp_rank is None:
        return {}
    return filter_supported_async_generate_kwargs(
        engine, {"disagg_prefill_dp_rank": int(prefill_dp_rank)}
    )


def require_reasoning_kwargs(
    engine: Any,
    request: Mapping[str, Any],
    *,
    thinking_budget_requested: bool = False,
) -> dict[str, Any]:
    """Build the optional SGLang per-request reasoning-gate argument."""
    require_reasoning = bool(request.get("require_reasoning", False))
    kwargs = filter_supported_async_generate_kwargs(
        engine,
        {"require_reasoning": require_reasoning},
    )
    if require_reasoning and "require_reasoning" not in kwargs:
        # The XPU SGLang 0.5.11 pin predates ``require_reasoning``. Keep
        # non-budget requests compatible until that pin is upgraded to 0.5.16+.
        if thinking_budget_requested:
            raise InvalidArgument(
                "thinking_token_budget requires an SGLang engine that supports "
                "per-request require_reasoning"
            )
        _warn_require_reasoning_unsupported()
    return kwargs


__all__ = [
    "ConfigArgumentMerger",
    "add_sglang_cli_compat",
    "cache_salt_kwargs",
    "filter_supported_async_generate_kwargs",
    "get_encoder_preprocessor_modules",
    "get_mm_encoder_class",
    "get_sglang_model_config",
    "mm_encode",
    "override_server_args",
    "publish_server_args",
    "require_reasoning_kwargs",
    "resolved_server_args",
    "sglang_uses_mla_backend",
    "supports_external_mm_hashes",
]
