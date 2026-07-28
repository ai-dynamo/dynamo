# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning enablement on the sglang chat-processor path.

Regression coverage for the bug where ``resolve_request_force_reasoning``
decided enablement from two local sets duplicating a table SGLang already
publishes as each detector's ``reasoning_default``. A parser in neither set
fell through to ``template_default``, which is ``False`` for any model shipping
no Jinja chat template -- so reasoning silently never ran.

Kimi-K3 is exactly that case: ``reasoning_default='thinking'`` and no chat
template, so ``reasoning_content`` came back null and the raw
``<|close|>think<|sep|>`` marker leaked into ``content``.
"""

import pytest

from dynamo.frontend.sglang_prepost import (
    _sglang_reasoning_default,
    resolve_request_force_reasoning,
)

# Needs sglang packages (gpu_1 container) but allocates no GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]


def _resolve(parser, kwargs=None, template_default=False, request=None):
    req = dict(request or {})
    if kwargs is not None:
        req["chat_template_kwargs"] = kwargs
    return resolve_request_force_reasoning(req, parser, template_default)


def _skip_if_parser_absent(parser):
    """Skip when this SGLang build does not expose the parser.

    The fallback tables still cover the older names, but asserting SGLang-derived
    behaviour is only meaningful when SGLang actually knows the parser.
    """
    if _sglang_reasoning_default(parser) is None:
        pytest.skip(f"sglang build does not expose reasoning parser {parser!r}")


class TestSglangDerivedDefault:
    """The regression: a parser SGLang knows but the local tables do not."""

    def test_kimi_k3_reasons_by_default_with_no_chat_template(self):
        # The exact failing case. template_default=False stands in for "model
        # ships no Jinja template"; before the fix this returned False.
        _skip_if_parser_absent("kimi_k3")
        assert _resolve("kimi_k3", {}, template_default=False) is True

    def test_kimi_k3_opts_out_via_thinking_false(self):
        _skip_if_parser_absent("kimi_k3")
        assert _resolve("kimi_k3", {"thinking": False}, template_default=False) is False

    def test_kimi_k3_ignores_template_default(self):
        # SGLang's answer must win over the statically detected template default.
        _skip_if_parser_absent("kimi_k3")
        assert _resolve("kimi_k3", {}, template_default=True) is True


class TestExistingBehaviourUnchanged:
    """Parsers already covered by the static tables must not shift."""

    @pytest.mark.parametrize(
        "parser, flag",
        [
            ("kimi_k2", "thinking"),
            ("qwen3", "enable_thinking"),
        ],
    )
    def test_opt_out_families(self, parser, flag):
        assert _resolve(parser, {}) is True
        assert _resolve(parser, {flag: False}) is False

    @pytest.mark.parametrize(
        "parser, flag",
        [
            ("deepseek-v3", "thinking"),
            ("gemma4", "enable_thinking"),
        ],
    )
    def test_opt_in_families(self, parser, flag):
        assert _resolve(parser, {}) is False
        assert _resolve(parser, {flag: True}) is True

    def test_minimax_m3_keeps_explicit_handling(self):
        # Handled before the SGLang lookup, so it must be unaffected by it.
        assert _resolve("minimax-m3", {}) is True
        assert _resolve("minimax-m3", {"thinking_mode": "disabled"}) is False

    @pytest.mark.parametrize(
        "request_body, expected",
        [
            ({}, False),
            ({"reasoning_effort": "none"}, False),
            ({"reasoning_effort": "low"}, True),
        ],
    )
    def test_mistral_keeps_explicit_handling(self, request_body, expected):
        assert _resolve("mistral", request=request_body) is expected


class TestFallback:
    """Parsers SGLang does not expose fall back to the static tables."""

    def test_unknown_parser_follows_template_default(self):
        assert _resolve("not-a-real-parser", {}, template_default=True) is True
        assert _resolve("not-a-real-parser", {}, template_default=False) is False

    def test_unknown_parser_lookup_returns_none(self):
        assert _sglang_reasoning_default("not-a-real-parser") is None

    def test_no_parser_disables_reasoning(self):
        assert _resolve(None, {}, template_default=True) is False
