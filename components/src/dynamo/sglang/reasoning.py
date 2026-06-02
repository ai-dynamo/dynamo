# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextvars
import functools
import threading
from contextlib import contextmanager
from typing import Any, Dict

_DYN_REQUIRE_REASONING_CV: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "dynamo_sglang_require_reasoning", default=False
)
_REQUIRE_REASONING_PROXY_LOCK = threading.Lock()


def install_require_reasoning_proxy(engine: Any) -> None:
    """Set SGLang GenerateReqInput.require_reasoning from a per-request context."""
    tm = getattr(engine, "tokenizer_manager", None)
    if tm is None or getattr(tm, "_dynamo_require_reasoning_wrapped", False):
        return

    with _REQUIRE_REASONING_PROXY_LOCK:
        if getattr(tm, "_dynamo_require_reasoning_wrapped", False):
            return

        original = tm.generate_request

        @functools.wraps(original)
        def _wrapped(obj, request):
            if _DYN_REQUIRE_REASONING_CV.get():
                obj.require_reasoning = True
            return original(obj, request)

        tm._dynamo_require_reasoning_wrapped = True  # type: ignore[attr-defined]
        tm.generate_request = _wrapped  # type: ignore[assignment]


def request_requires_reasoning(
    has_reasoning_parser: bool, request: Dict[str, Any], input_param: Dict[str, Any]
) -> bool:
    if not has_reasoning_parser:
        return False

    extra_args = request.get("extra_args") or {}
    if isinstance(extra_args, dict) and (
        extra_args.get("require_reasoning")
        or extra_args.get("prompt_injected_reasoning")
    ):
        return True

    prompt = input_param.get("prompt")
    return isinstance(prompt, str) and prompt.rstrip().endswith("<think>")


@contextmanager
def require_reasoning_context(
    has_reasoning_parser: bool, request: Dict[str, Any], input_param: Dict[str, Any]
):
    token = _DYN_REQUIRE_REASONING_CV.set(
        request_requires_reasoning(has_reasoning_parser, request, input_param)
    )
    try:
        yield
    finally:
        _DYN_REQUIRE_REASONING_CV.reset(token)
