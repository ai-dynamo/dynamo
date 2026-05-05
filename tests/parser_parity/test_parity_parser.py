# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser-mode parity tests — Method 2 of DIS-1906.

For each fixture, run the same input through Dynamo's Rust parser (via PyO3)
and the upstream Python parsers (vLLM + SGLang). Diff against the recorded
expected output and against each other.

Run:
    pytest tests/parser-parity/test_parity_parser.py -v
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.parser_parity.impls import common

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


def _maybe_import(impl_name: str) -> Callable | None:
    """Import an impl wrapper lazily; return None if its dependency package
    (vllm / sglang / dynamo._core) isn't installed in this container."""
    try:
        mod = importlib.import_module(f"tests.parser_parity.impls.parser.{impl_name}")
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
    ("sglang", "glm47", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
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
    ("vllm", "qwen3_coder", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "deepseek_v3_1", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "minimax_m2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "kimi_k2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "qwen3_coder", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "harmony", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    # PARSER.batch.5 (missing end-token recovery) — impl-defined.
    ("vllm", "qwen3_coder", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("vllm", "deepseek_v3_1", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "qwen3_coder", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "deepseek_v3_1", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "harmony", "PARSER.batch.5"): _RECOVERY_CONTRACT,
}


def _load_fixtures() -> list[tuple[str, str, dict[str, Any]]]:
    """Yields (family, case_id, fixture_dict) for every case across all families.

    Each <family>/PARSER.<mode>.json holds all cases for that mode in one file:

        {"family": "...", "mode": "batch", "cases": {"1": {...}, "2": {...}, ...}}

    `case_id` is reconstructed as e.g. "PARSER.batch.1" so test IDs and the
    KNOWN_DIVERGENCES registry stay aligned with the per-case naming.
    """
    out = []
    for fp in sorted(FIXTURES_ROOT.glob("*/PARSER.*.json")):
        doc = json.loads(fp.read_text())
        family = doc["family"]
        mode = doc["mode"]
        for case_num, case in doc["cases"].items():
            case_id = f"PARSER.{mode}.{case_num}"
            fixture = {
                "case": case_id,
                "family": family,
                "description": case["description"],
                "model_text": case["model_text"],
                "tools": case["tools"],
                "expected": case["expected"],
            }
            out.append((family, case_id, fixture))
    # Stable order: family first, then numeric case index.
    out.sort(key=lambda t: (t[0], int(t[1].rsplit(".", 1)[1])))
    return out


FIXTURES = _load_fixtures()


@pytest.mark.parametrize(
    "family,case_id,fixture,impl_name",
    [(f, c, fx, impl) for (f, c, fx) in FIXTURES for impl in IMPLS],
    ids=[f"{f}/{c}/{impl}" for (f, c, fx) in FIXTURES for impl in IMPLS],
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

    divergence = KNOWN_DIVERGENCES.get((impl_name, family, case_id))
    if divergence is not None:
        pytest.xfail(f"known divergence: {divergence}")

    got = impl_fn(family, fixture["model_text"], fixture.get("tools"))

    if got.error:
        pytest.skip(f"{impl_name} unavailable for {family}: {got.error}")

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
