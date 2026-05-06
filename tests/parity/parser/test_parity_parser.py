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
from typing import Any, Callable

import pytest
import yaml

from tests.parity import common

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _maybe_import(impl_name: str) -> Callable | None:
    """Import an impl wrapper lazily; return None if its dependency package
    (vllm / sglang / dynamo._core) isn't installed in this container."""
    try:
        mod = importlib.import_module(f"tests.parity.parser.{impl_name}")
        return mod.parse
    except ImportError as e:
        print(f"[parity] skipping impl {impl_name!r}: {e}")
        return None


IMPLS: dict[str, Callable | None] = {
    "dynamo": _maybe_import("dynamo"),
    "vllm": _maybe_import("vllm"),
    "sglang": _maybe_import("sglang"),
}


# (impl, family, case_id) -> reason. Surfaced findings from running the harness;
# each entry is a real behavioral divergence we've chosen to document rather
# than fix. Promote to a tracked bug before removing.
#
# Pattern: vLLM and SGLang both truncate normal_text at the *end* of the
# tool-call wrapper across all XML-style parsers (kimi_k2, qwen3_coder, glm47).
# Only Dynamo preserves text that appears after the closing wrapper token.
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
    # wrapper across all XML-style parsers. Only Dynamo preserves text after
    # the closing wrapper token.
    ("vllm", "kimi_k2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "qwen3_coder", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "glm47", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "kimi_k2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "qwen3_coder", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
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
    # - SGLang on deepseek_v3_1: trims one trailing space; Dynamo keeps it
    # - vLLM on minimax_m2: keeps trailing space; Dynamo trims it (opposite
    #   direction from deepseek)
    (
        "sglang",
        "deepseek_v3_1",
        "PARSER.batch.8",
    ): "trims trailing space from preceding normal_text",
    (
        "vllm",
        "minimax_m2",
        "PARSER.batch.8",
    ): "preserves trailing space; Dynamo trims it",
    # PARSER.batch.4 (malformed) — impl-defined recovery contract.
    ("vllm", "deepseek_v3_1", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "minimax_m2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "kimi_k2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "harmony", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    # PARSER.batch.5 (missing end-token recovery) — impl-defined.
    ("vllm", "qwen3_coder", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("vllm", "deepseek_v3_1", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "qwen3_coder", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "deepseek_v3_1", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "harmony", "PARSER.batch.5"): _RECOVERY_CONTRACT,
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


def _marks_for(family: str, case_id: str, impl: str) -> list:
    """Per-param marks for a parametrized case. Includes the impl marker
    (so CI marker filters `pre_merge and vllm and gpu_0` / `... sglang ...`
    pick up the right subset per container; `dynamo` gets no impl marker).

    Adds `pytest.mark.xfail(strict=True, reason=...)` for known divergences
    so the assertion still runs — `XPASS` (strict) flags a fix the registry
    hasn't been updated for, instead of silently masking the new pass."""
    marks = []
    if impl in ("vllm", "sglang"):
        marks.append(getattr(pytest.mark, impl))
    div = KNOWN_DIVERGENCES.get((impl, family, case_id))
    if div is not None:
        marks.append(pytest.mark.xfail(strict=True, reason=f"known divergence: {div}"))
    return marks


@pytest.mark.parametrize(
    "family,case_id,fixture,impl_name",
    [
        pytest.param(
            f,
            c,
            fx,
            impl,
            marks=_marks_for(f, c, impl),
            id=f"{f}/{c}#{impl}",
        )
        for (f, c, fx) in FIXTURES
        for impl in IMPLS
    ],
)
def test_parity(
    family: str,
    case_id: str,
    fixture: dict[str, Any],
    impl_name: str,
) -> None:
    impl_fn = IMPLS[impl_name]
    if impl_fn is None:
        pytest.skip(f"{impl_name} not installed in this container")

    got = impl_fn(family, fixture["model_text"], fixture.get("tools"))

    # Distinguish "impl has no parser registered for this family" (env-shaped:
    # skip) from "parser raised on input" (runtime: a regression we want to
    # see). Wrappers prefix the env case with `UNAVAILABLE:`.
    if got.error:
        if got.error.startswith("UNAVAILABLE:"):
            pytest.skip(f"{impl_name} unavailable for {family}: {got.error}")
        pytest.fail(f"{impl_name} crashed on {family}/{case_id}: {got.error}")

    expected = common.ParseResult(
        calls=fixture["expected"]["calls"],
        normal_text=fixture["expected"].get("normal_text"),
    )

    assert common.canonical(got.to_dict()) == common.canonical(expected.to_dict()), (
        f"\nimpl:     {impl_name}\n"
        f"family:   {family}\n"
        f"case:     {case_id}\n"
        f"expected: {common.canonical(expected.to_dict())}\n"
        f"got:      {common.canonical(got.to_dict())}\n"
    )
