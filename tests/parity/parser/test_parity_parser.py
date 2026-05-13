# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser-mode parity tests — Method 2 of the cross-impl parity (parser) effort.

For each fixture, run the same input through Dynamo's Rust parser (via PyO3)
and the upstream Python parsers (vLLM + SGLang). Each case's YAML carries an
`expected:` block keyed by impl name (Variant D schema):

    expected:
      dynamo:                          # oracle, always required
        calls: [...]
        normal_text: '...'
      vllm:                            # OPTIONAL: only when vLLM diverges
        diverges: true
        reason: "<why>"
        expected_error_pattern: "..."  # OPTIONAL: with EXPECTS_ERROR contract
      sglang:                          # OPTIONAL: only when SGLang diverges
        diverges: true
        reason: "..."

If an impl key is absent, the impl is expected to match `dynamo`. If
`diverges: true` is set, the case xfails for that impl (and XPASS-strict
turns on if it later starts matching `dynamo`).

Run:
    pytest tests/parity/parser/test_parity_parser.py -v
"""

from __future__ import annotations

import importlib
import re
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


def _case_sort_key(case_id: str) -> tuple[int, str]:
    """Sort key for case IDs that may carry a sub-letter.

    `PARSER.batch.5`   → (5, "")
    `PARSER.batch.8.a` → (8, "a")
    `PARSER.batch.8.b` → (8, "b")
    """
    parts = case_id.split(".")
    top = int(parts[2])
    sub = parts[3] if len(parts) > 3 else ""
    return (top, sub)


def _load_fixtures() -> list[tuple[str, str, dict[str, Any]]]:
    """Yields (family, case_id, case_dict) for every case across all families.

    Two file layouts coexist:
      <family>/PARSER.<mode>.yaml       — legacy flat: holds 1, 2, ..., 10
      <family>/PARSER.<mode>.<n>.yaml   — per-top-level-case: holds n.a, n.b, ...

    Both schemas are
        {family: "...", mode: "batch", cases: {PARSER.batch.<id>: {...}, ...}}
    Case keys are the full PARSER_CASES.md ID (e.g. `PARSER.batch.1` or
    `PARSER.batch.8.a`) so they match parametrize IDs directly. The merge
    across the two layouts is conflict-free as long as a given case ID
    lives in exactly one file.
    """
    out = []
    for fp in sorted(FIXTURES_ROOT.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        for case_id, case in doc["cases"].items():
            out.append((doc["family"], case_id, case))
    out.sort(key=lambda t: (t[0], *_case_sort_key(t[1])))
    return out


FIXTURES = _load_fixtures()


def _marks_for(impl: str) -> list:
    """Per-param marks. Only the impl marker (`vllm`/`sglang`) — used by CI
    shards to filter via `-m vllm` / `-m sglang`. `dynamo` gets no marker so
    it runs in every shard (it's the reference we compare against)."""
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

    # Per-impl expected from YAML. Missing peer key → peer must match dynamo.
    expected_map = fixture["expected"]
    spec = expected_map.get(impl_name) or expected_map["dynamo"]
    is_divergent = isinstance(spec, dict) and spec.get("diverges") is True

    # Wrapper-level "no peer parser for this family" — clean skip.
    if got.error and got.error.startswith("UNAVAILABLE:"):
        pytest.skip(f"{impl_name} unavailable for {family}: {got.error}")

    # Errors absorbed only when the YAML opts in via `expected_error_pattern`
    # and the actual error matches the pattern. Otherwise, errors are real
    # bugs and fail HARD (an unrelated wrapper/runtime crash must not hide
    # behind a divergence entry that's only supposed to differ on values).
    if got.error:
        if is_divergent and (pat := spec.get("expected_error_pattern")):
            if re.search(pat, got.error):
                pytest.xfail(
                    f"known divergence (error matched /{pat}/): {spec.get('reason', '')}"
                )
        pytest.fail(f"{impl_name} crashed on {family}/{case_id}: {got.error}")

    got_canonical = common.canonical(got.to_dict())
    dynamo_spec = expected_map["dynamo"]
    dynamo_canonical = common.canonical(
        common.ParseResult(
            calls=dynamo_spec["calls"],
            normal_text=dynamo_spec.get("normal_text"),
        ).to_dict()
    )

    # Divergent peer (no concrete expected output recorded yet): the peer
    # must produce SOMETHING different from dynamo. If it now matches dynamo,
    # the YAML override is stale — surface as XPASS-strict failure.
    if is_divergent:
        if got_canonical == dynamo_canonical:
            pytest.fail(
                f"XPASS-strict: ({impl_name},{family},{case_id}) now matches dynamo — "
                f"remove the `expected.{impl_name}` block from the fixture. "
                f"Reason was: {spec.get('reason', '<no reason>')}"
            )
        pytest.xfail(f"known divergence: {spec.get('reason', '<no reason>')}")

    # Non-divergent peer (or dynamo itself): must match the recorded expected.
    expected_canonical = common.canonical(
        common.ParseResult(
            calls=spec["calls"],
            normal_text=spec.get("normal_text"),
        ).to_dict()
    )
    assert got_canonical == expected_canonical, (
        f"\nimpl:     {impl_name}\n"
        f"family:   {family}\n"
        f"case:     {case_id}\n"
        f"expected: {expected_canonical}\n"
        f"got:      {got_canonical}\n"
    )
