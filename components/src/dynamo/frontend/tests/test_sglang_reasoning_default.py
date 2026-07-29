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

import logging

import pytest
from sglang.srt.parser.reasoning_parser import ReasoningParser

from dynamo.frontend.sglang_prepost import (
    _SGLANG_REASONING_MODES,
    _force_reasoning_from_sglang_default,
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

    @pytest.mark.parametrize(
        "effort, expected",
        [
            (None, False),
            ("none", False),
            ("no_think", False),
            ("low", True),
            ("high", True),
        ],
    )
    def test_hunyuan_gates_on_reasoning_effort(self, effort, expected):
        """hunyuan declares reasoning_default='always' but must not honour it.

        The Hy3-preview template emits no <think> unless reasoning_effort asks
        for it, so taking 'always' at face value would route the whole response
        into reasoning_content. SGLang special-cases this ahead of its own
        reasoning_default dispatch; we mirror that.
        """
        body = {} if effort is None else {"reasoning_effort": effort}
        assert _resolve("hunyuan", request=body) is expected

    def test_hunyuan_reasoning_default_is_always(self):
        """Documents why the branch above is needed, not redundant."""
        _skip_if_parser_absent("hunyuan")
        assert _sglang_reasoning_default("hunyuan") == "always"


class TestDispatchHelper:
    """Direct coverage of every mode branch, independent of any real parser."""

    @pytest.mark.parametrize(
        "mode, kwargs, request_body, expected",
        [
            # always: unconditional
            ("always", {}, {}, True),
            ("always", {"enable_thinking": False}, {}, True),
            # mistral: gated on reasoning_effort, top-level or nested
            ("mistral", {}, {}, False),
            ("mistral", {}, {"reasoning_effort": "none"}, False),
            ("mistral", {}, {"reasoning_effort": "low"}, True),
            ("mistral", {"reasoning_effort": "low"}, {}, True),
            # top-level wins over the nested value
            (
                "mistral",
                {"reasoning_effort": "low"},
                {"reasoning_effort": "none"},
                False,
            ),
            (
                "mistral",
                {"reasoning_effort": "none"},
                {"reasoning_effort": "low"},
                True,
            ),
            # opt-out modes: on unless the matching kwarg is exactly False
            ("thinking", {}, {}, True),
            ("thinking", {"thinking": False}, {}, False),
            ("thinking", {"thinking": True}, {}, True),
            ("enable_thinking", {}, {}, True),
            ("enable_thinking", {"enable_thinking": False}, {}, False),
            # opt-in modes: off unless the toggle is exactly True
            ("explicit_thinking", {}, {}, False),
            ("explicit_thinking", {"thinking": True}, {}, True),
            ("explicit_thinking", {"thinking": "yes"}, {}, False),
            ("explicit_enable_thinking", {}, {}, False),
            ("explicit_enable_thinking", {"enable_thinking": True}, {}, True),
            # no mode: not applicable, caller falls back
            (None, {}, {}, None),
        ],
    )
    def test_mode_dispatch(self, mode, kwargs, request_body, expected):
        assert _force_reasoning_from_sglang_default(mode, kwargs, request_body) is (
            expected
        )

    def test_every_implemented_mode_has_a_branch(self):
        """No member of the set may fall through to the AssertionError."""
        for mode in _SGLANG_REASONING_MODES:
            result = _force_reasoning_from_sglang_default(mode, {}, {})
            assert isinstance(result, bool), f"mode {mode!r} returned {result!r}"


class TestFallback:
    """Parsers SGLang does not expose fall back to the static tables."""

    def test_unknown_parser_follows_template_default(self):
        assert _resolve("not-a-real-parser", {}, template_default=True) is True
        assert _resolve("not-a-real-parser", {}, template_default=False) is False

    def test_unknown_parser_lookup_returns_none(self):
        assert _sglang_reasoning_default("not-a-real-parser") is None

    def test_no_parser_disables_reasoning(self):
        assert _resolve(None, {}, template_default=True) is False


class TestModeCoverage:
    """Guard the assumption that the dispatch covers what SGLang declares.

    Without this, SGLang adding a new ``reasoning_default`` value would silently
    route every affected model back to the static tables -- reintroducing the
    exact silent miss this change exists to remove. This test fails loudly
    instead.
    """

    def test_every_registered_detector_mode_is_implemented(self):
        detector_map = getattr(ReasoningParser, "DetectorMap", None)
        if not detector_map:
            pytest.skip("this sglang build exposes no DetectorMap")

        unimplemented = {}
        for name in sorted(detector_map):
            try:
                mode = ReasoningParser(model_type=name).detector.reasoning_default
            except Exception:  # pragma: no cover - detector needs extra deps
                continue
            if mode not in _SGLANG_REASONING_MODES:
                unimplemented[name] = mode

        assert not unimplemented, (
            "sglang declares reasoning_default values this frontend does not "
            f"implement: {unimplemented}. Add them to _SGLANG_REASONING_MODES "
            "and give each a branch in _force_reasoning_from_sglang_default."
        )

    def test_unimplemented_mode_warns_and_falls_back(self, monkeypatch, caplog):
        """A mode outside the implemented set must be reported, not swallowed."""

        class _Detector:
            reasoning_default = "some_future_mode"

        class _Parser:
            def __init__(self, *args, **kwargs):
                self.detector = _Detector()

        monkeypatch.setattr("dynamo.frontend.sglang_prepost.ReasoningParser", _Parser)
        _sglang_reasoning_default.cache_clear()
        try:
            with caplog.at_level(logging.WARNING):
                assert _sglang_reasoning_default("pretend-parser") is None
            assert "some_future_mode" in caplog.text
        finally:
            _sglang_reasoning_default.cache_clear()

    def test_non_valueerror_propagates(self, monkeypatch):
        """Only ValueError means 'unknown parser'; other failures must surface."""

        class _Parser:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("sglang is broken")

        monkeypatch.setattr("dynamo.frontend.sglang_prepost.ReasoningParser", _Parser)
        _sglang_reasoning_default.cache_clear()
        try:
            with pytest.raises(RuntimeError, match="sglang is broken"):
                _sglang_reasoning_default("pretend-parser-2")
        finally:
            _sglang_reasoning_default.cache_clear()
