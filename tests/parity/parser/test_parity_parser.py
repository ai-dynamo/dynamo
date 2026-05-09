# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser-mode parity tests — Method 2 of the cross-impl parity (parser) effort.

For each fixture, run the same input through Dynamo's Rust parser (via PyO3)
and the upstream Python parsers (vLLM + SGLang). Diff against the recorded
expected output and against each other.

Run:
    pytest tests/parity/parser/test_parity_parser.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.parity import common

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


# Impl name → optional package whose absence is a legitimate skip. Anything
# else that fails inside the wrapper module (e.g. a stale `from vllm.X import
# Y` after vLLM renamed Y) is a real bug and must surface as a test ERROR,
# not a green skip — so we use `pytest.importorskip` inline rather than a
# blanket `try: import X except ImportError`.
IMPL_NAMES: tuple[str, ...] = ("dynamo", "vllm", "sglang")
_PACKAGE: dict[str, str] = {
    "dynamo": "dynamo._core",
    "vllm": "vllm",
    "sglang": "sglang",
}


# (impl, family, case_id) -> reason. Surfaced findings from running the harness;
# each entry is a real behavioral divergence we've chosen to document rather
# than fix. Promote to a tracked bug before removing.
#
# Pattern: vLLM and SGLang both truncate normal_text at the *end* of the
# tool-call wrapper for some parser families. Dynamo preserves text that
# appears after the closing wrapper token for the families still listed below.
_TRAILING_NORMAL_TEXT_DROP = "drops trailing normal_text after tool-call wrapper end"
_HARMONY_REQUIRES_ASSISTANT_PREFIX = (
    "SGLang's GptOssDetector bot_token requires '<|start|>assistant<|channel|>commentary' "
    "envelope; Dynamo accepts the bare '<|channel|>commentary' variant"
)
# PARSER.batch.4 (malformed JSON) and PARSER.batch.5 (missing end-token) are
# impl-defined recovery contracts per PARSER_CASES.md. Each parser picks its
# own behavior (drop, recover-best-effort, fall back to string args, surface
# error). Cross-impl parity is not expected; we record divergences as a map
# of "what each impl does" rather than asserting one truth.
_RECOVERY_CONTRACT = "impl-defined recovery contract (see PARSER_CASES.md)"

KNOWN_DIVERGENCES: dict[tuple[str, str, str], str] = {
    # vLLM and SGLang both truncate normal_text at the *end* of the tool-call
    # wrapper for these parser families. Only Dynamo preserves text after
    # the closing wrapper token.
    ("vllm", "kimi_k2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "glm47", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "kimi_k2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    # SGLang's GptOssDetector requires a strict '<|start|>assistant<|channel|>commentary'
    # bot_token; bare '<|channel|>commentary' variants (PARSER.batch.1, .6, .13)
    # are not detected at all.
    ("sglang", "harmony", "PARSER.batch.1"): _HARMONY_REQUIRES_ASSISTANT_PREFIX,
    ("sglang", "harmony", "PARSER.batch.6"): _HARMONY_REQUIRES_ASSISTANT_PREFIX,
    ("sglang", "harmony", "PARSER.batch.8"): _HARMONY_REQUIRES_ASSISTANT_PREFIX,
    # SGLang's GptOssDetector drops the analysis-channel preamble entirely;
    # Dynamo surfaces it as normal_text.
    (
        "sglang",
        "harmony",
        "PARSER.batch.7",
    ): "drops analysis-channel content from normal_text",
    # Inverse of the bare-envelope finding: when the input *does* have the
    # assistant prefix and contains back-to-back commentary blocks, SGLang
    # extracts both calls — Dynamo's harmony parser drops them and treats the
    # raw input as normal_text. Dynamo bug class; see harmony_parser.rs comment
    # block above test_parse_harmony_multiple_calls_recovers.
    (
        "sglang",
        "harmony",
        "PARSER.batch.2",
    ): "Dynamo drops back-to-back commentary blocks; SGLang extracts them",
    (
        "sglang",
        "harmony",
        "PARSER.batch.10",
    ): "Dynamo drops back-to-back commentary blocks; SGLang extracts them",
    # Whitespace handling on text immediately preceding the bot_token:
    # - SGLang on DeepSeek variants trims one trailing space; Dynamo keeps it.
    (
        "sglang",
        "deepseek_v3_1",
        "PARSER.batch.8",
    ): "trims trailing space from preceding normal_text",
    (
        "sglang",
        "deepseek_v3",
        "PARSER.batch.8",
    ): "trims trailing space from preceding normal_text",
    (
        "sglang",
        "pythonic",
        "PARSER.batch.8",
    ): _TRAILING_NORMAL_TEXT_DROP,
    (
        "sglang",
        "deepseek_v3",
        "PARSER.batch.5",
    ): _RECOVERY_CONTRACT,
    (
        "vllm",
        "deepseek_v3",
        "PARSER.batch.4",
    ): _RECOVERY_CONTRACT,
    (
        "vllm",
        "deepseek_v3",
        "PARSER.batch.5",
    ): _RECOVERY_CONTRACT,
    (
        "vllm",
        "gemma4",
        "PARSER.batch.8",
    ): _TRAILING_NORMAL_TEXT_DROP,
    # vLLM's pythonic parser rejects the whole input when the bracket-call form
    # is surrounded by interleaved text — calls=[], normal_text=raw input.
    # Dynamo's pythonic parser extracts the call and surfaces the surrounding
    # text in normal_text (matches the upstream Llama 4 / pythonic_detector
    # behavior — vLLM is the outlier here).
    (
        "vllm",
        "pythonic",
        "PARSER.batch.8",
    ): "vLLM rejects bracket-call form when surrounded by interleaved text; Dynamo extracts the call",
    (
        "vllm",
        "deepseek_v4",
        "PARSER.batch.5",
    ): _RECOVERY_CONTRACT,
    (
        "vllm",
        "deepseek_v4",
        "PARSER.batch.7",
    ): "vLLM emits JSON-typed parameter values as raw strings; Dynamo coerces nested object/array types",
    # PARSER.batch.4 (malformed) — impl-defined recovery contract.
    ("vllm", "deepseek_v3_1", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "minimax_m2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "kimi_k2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "harmony", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    # PARSER.batch.5 (missing end-token recovery) — impl-defined. Dynamo's
    # PyO3 batch binding now uses `detect_and_parse_tool_call_with_recovery`
    # so the registered `expected` reflects the recovered call; impls that
    # still drop on the missing end-token diverge here.
    # vllm/sglang qwen3_coder.batch.5 both recover to the same call as
    # Dynamo and match the new expected — intentionally NOT registered.
    ("vllm", "glm47", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("vllm", "minimax_m2", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("vllm", "deepseek_v3_1", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "glm47", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "minimax_m2", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "deepseek_v3_1", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "harmony", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    # ----- new families: hermes / qwen25 / mistral / jamba / llama3_json / phi4 / nemotron_nano / deepseek_v3_2 -----
    ("vllm", "deepseek_v3_2", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    (
        "vllm",
        "deepseek_v3_2",
        "PARSER.batch.7",
    ): "emits JSON-typed parameter values as raw strings; Dynamo coerces nested object/array types",
    ("vllm", "hermes", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "hermes", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "jamba", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("vllm", "jamba", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    (
        "vllm",
        "llama3_json",
        "PARSER.batch.2",
    ): "vLLM llama3_json does not parse semicolon-separated calls the same way Dynamo does",
    ("vllm", "llama3_json", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "llama3_json", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    (
        "vllm",
        "llama3_json",
        "PARSER.batch.10",
    ): "vLLM llama3_json does not parse semicolon-separated calls the same way Dynamo does",
    ("vllm", "mistral", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "mistral", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "phi4", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    (
        "vllm",
        "phi4",
        "PARSER.batch.7",
    ): "emits JSON-typed parameter values as raw strings; Dynamo coerces nested object/array types",
    ("vllm", "phi4", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "deepseek_v3_2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "hermes", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    # SGLang's MistralDetector and Qwen25Detector both return empty `calls`
    # on the bare wire formats Dynamo emits — format-detection failure rather
    # than a recovery-contract divergence. The exact reason (different start
    # token, schema-shape mismatch, missing chat-format envelope) is not
    # investigated here; the cells are recorded as `X` so the matrix reflects
    # the observed behavior.
    (
        "sglang",
        "mistral",
        "PARSER.batch.1",
    ): "SGLang MistralDetector returns calls=[] on the bare [TOOL_CALLS]...[/TOOL_CALLS] format Dynamo emits",
    (
        "sglang",
        "mistral",
        "PARSER.batch.2",
    ): "SGLang MistralDetector returns calls=[] on the bare [TOOL_CALLS]...[/TOOL_CALLS] format Dynamo emits",
    ("sglang", "mistral", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "mistral", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    (
        "sglang",
        "mistral",
        "PARSER.batch.6",
    ): "SGLang MistralDetector returns calls=[] on the bare [TOOL_CALLS]...[/TOOL_CALLS] format Dynamo emits",
    (
        "sglang",
        "mistral",
        "PARSER.batch.7",
    ): "SGLang MistralDetector returns calls=[] on the bare [TOOL_CALLS]...[/TOOL_CALLS] format Dynamo emits",
    ("sglang", "mistral", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    (
        "sglang",
        "mistral",
        "PARSER.batch.10",
    ): "SGLang MistralDetector returns calls=[] on the bare [TOOL_CALLS]...[/TOOL_CALLS] format Dynamo emits",
    (
        "sglang",
        "qwen25",
        "PARSER.batch.1",
    ): "SGLang Qwen25Detector returns calls=[] on the hermes-style <tool_call>...</tool_call> format Dynamo emits",
    (
        "sglang",
        "qwen25",
        "PARSER.batch.2",
    ): "SGLang Qwen25Detector returns calls=[] on the hermes-style <tool_call>...</tool_call> format Dynamo emits",
    ("sglang", "qwen25", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "qwen25", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    (
        "sglang",
        "qwen25",
        "PARSER.batch.6",
    ): "SGLang Qwen25Detector returns calls=[] on the hermes-style <tool_call>...</tool_call> format Dynamo emits",
    (
        "sglang",
        "qwen25",
        "PARSER.batch.7",
    ): "SGLang Qwen25Detector returns calls=[] on the hermes-style <tool_call>...</tool_call> format Dynamo emits",
    ("sglang", "qwen25", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    (
        "sglang",
        "qwen25",
        "PARSER.batch.10",
    ): "SGLang Qwen25Detector returns calls=[] on the hermes-style <tool_call>...</tool_call> format Dynamo emits",
}


def _load_fixtures() -> list[tuple[str, str, dict[str, Any]]]:
    """Yields (family, case_id, case_dict) for every case across all families.

    Schema: <family>/PARSER.<mode>.yaml holds
        {family: "...", mode: "batch", cases: {PARSER.batch.1: {...}, ...}}
    Case keys are the full PARSER_CASES.md ID (e.g. `PARSER.batch.1`)
    so they match KNOWN_DIVERGENCES keys and parametrize IDs directly.
    """
    out = []
    for fp in sorted(FIXTURES_ROOT.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        for case_id, case in doc["cases"].items():
            out.append((doc["family"], case_id, case))
    out.sort(key=lambda t: (t[0], int(t[1].rsplit(".", 1)[1])))
    return out


FIXTURES = _load_fixtures()


def _marks_for(impl: str) -> list:
    """Per-param marks. Only the impl marker (`vllm`/`sglang`) — used by CI
    shards to filter via `-m vllm` / `-m sglang`. `dynamo` gets no marker so
    it runs in every shard (it's the reference we compare against).

    Known-divergence xfail handling is intentionally NOT a parametrize-time
    mark — it lives inline in the test body (see `XPASS-strict` block) so a
    parser/runtime crash on a known-divergence case still hits `pytest.fail`
    hard instead of being swallowed by an outer `xfail` wrapper."""
    if impl in ("vllm", "sglang"):
        return [getattr(pytest.mark, impl)]
    return []


@pytest.mark.parametrize(
    "family,case_id,fixture,impl_name",
    [
        pytest.param(
            f,
            c,
            fx,
            impl,
            marks=_marks_for(impl),
            id=f"{f}/{c}#{impl}",
        )
        for (f, c, fx) in FIXTURES
        for impl in IMPL_NAMES
    ],
)
def test_parity(
    family: str,
    case_id: str,
    fixture: dict[str, Any],
    impl_name: str,
) -> None:
    # Skip ONLY when the optional package itself is missing — wrapper-internal
    # ImportError (e.g. a stale upstream API ref after a vLLM/SGLang rename)
    # propagates as a real test ERROR rather than a silent green skip.
    pytest.importorskip(_PACKAGE[impl_name])
    parse_mod = importlib.import_module(f"tests.parity.parser.{impl_name}")
    got = parse_mod.parse(family, fixture["model_text"], fixture.get("tools"))

    # Runtime / parser errors fail HARD. Wrappers prefix the env-shaped case
    # ("impl has no parser registered for this family") with `UNAVAILABLE:`
    # so we can still skip those cleanly. This check runs *before* the
    # known-divergence xfail block below, so a crash on a known-divergence
    # case is not masked.
    if got.error:
        if got.error.startswith("UNAVAILABLE:"):
            pytest.skip(f"{impl_name} unavailable for {family}: {got.error}")
        pytest.fail(f"{impl_name} crashed on {family}/{case_id}: {got.error}")

    expected = common.ParseResult(
        calls=fixture["expected"]["calls"],
        normal_text=fixture["expected"].get("normal_text"),
    )
    got_canonical = common.canonical(got.to_dict())
    expected_canonical = common.canonical(expected.to_dict())

    # Known divergences: xfail the value-diff assertion only. If the values
    # now match, surface that as a registry-staleness failure (XPASS-strict
    # equivalent) instead of silently passing.
    div = KNOWN_DIVERGENCES.get((impl_name, family, case_id))
    if div is not None:
        if got_canonical == expected_canonical:
            pytest.fail(
                f"XPASS-strict: known divergence ({impl_name},{family},{case_id}) "
                f"now matches expected — remove from KNOWN_DIVERGENCES. "
                f"Reason was: {div}"
            )
        pytest.xfail(f"known divergence: {div}")

    assert got_canonical == expected_canonical, (
        f"\nimpl:     {impl_name}\n"
        f"family:   {family}\n"
        f"case:     {case_id}\n"
        f"expected: {expected_canonical}\n"
        f"got:      {got_canonical}\n"
    )
