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


def _run_scheduler_process_with_gemma4_compat(*args: Any, **kwargs: Any) -> Any:
    """Install the grammar shim inside SGLang's spawned scheduler."""
    ensure_gemma4_implicit_tool_reasoning_boundary()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def ensure_gemma4_implicit_tool_reasoning_boundary() -> None:
    """Let SGLang's reasoning grammar hand off on Gemma 4 tool start.

    SGLang 0.5.14 transitions its guided grammar only on ``<channel|>``.
    Gemma 4 may instead emit ``<|tool_call>`` directly from the thought
    channel. Attach that model-specific token ID to each reasoning grammar
    object and preserve it across request-local copies.

    Remove when the minimum supported SGLang release treats Gemma 4's tool
    start token as an implicit reasoning end.
    """
    from sglang.srt.constrained.reasoner_grammar_backend import (
        ReasonerGrammarBackend,
        ReasonerGrammarObject,
    )
    from sglang.srt.entrypoints.engine import Engine

    # Engine uses multiprocessing spawn, so parent class mutations are not
    # inherited. Make the pickled scheduler target install the same shim in
    # the child before constructing its grammar backend.
    Engine.run_scheduler_process_func = staticmethod(
        _run_scheduler_process_with_gemma4_compat
    )

    original_transfer_state = ReasonerGrammarObject.transfer_state
    if getattr(original_transfer_state, "_dynamo_gemma4_tool_boundary", False):
        return

    original_backend_init = ReasonerGrammarBackend.__init__
    original_make_grammar = ReasonerGrammarBackend._make_grammar_object
    original_copy = ReasonerGrammarObject.copy

    @wraps(original_backend_init)
    def backend_init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = inspect.signature(original_backend_init).bind(self, *args, **kwargs)
        original_backend_init(self, *args, **kwargs)

        reasoning_parser = bound.arguments["reasoning_parser"]
        tokenizer = bound.arguments["tokenizer"]
        if type(reasoning_parser.detector).__name__ != "Gemma4Detector":
            return

        tool_start_ids = tokenizer.encode("<|tool_call>", add_special_tokens=False)
        if len(tool_start_ids) == 1:
            self._dynamo_implicit_think_end_id = tool_start_ids[0]

    @wraps(original_make_grammar)
    def make_grammar(self: Any, *args: Any, **kwargs: Any) -> Any:
        grammar = original_make_grammar(self, *args, **kwargs)
        grammar._dynamo_implicit_think_end_id = getattr(
            self, "_dynamo_implicit_think_end_id", None
        )
        return grammar

    @wraps(original_transfer_state)
    def transfer_state(self: Any, token: int) -> None:
        if self._is_thinking() and token == getattr(
            self, "_dynamo_implicit_think_end_id", None
        ):
            self.tokens_after_end = 0
            return
        original_transfer_state(self, token)

    @wraps(original_copy)
    def copy(self: Any) -> Any:
        grammar = original_copy(self)
        grammar._dynamo_implicit_think_end_id = getattr(
            self, "_dynamo_implicit_think_end_id", None
        )
        return grammar

    transfer_state._dynamo_gemma4_tool_boundary = True  # type: ignore[attr-defined]
    ReasonerGrammarBackend.__init__ = backend_init
    ReasonerGrammarBackend._make_grammar_object = make_grammar
    ReasonerGrammarObject.transfer_state = transfer_state
    ReasonerGrammarObject.copy = copy


def ensure_sglang_tensor_image_size() -> None:
    """Allow SGLang's image-token resolver to handle decoded image tensors.

    SGLang 0.5.13 and 0.5.14 assume every decoded image exposes the PIL
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

    SGLang 0.5.11 accepts profiling fields as keyword arguments. Newer builds
    accept one ``ProfileReq`` object instead.
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
    "ensure_gemma4_implicit_tool_reasoning_boundary",
    "ensure_sglang_tensor_image_size",
    "ensure_sglang_top_level_exports",
    "filter_supported_async_generate_kwargs",
    "require_reasoning_kwargs",
    "start_profile_compat",
]
