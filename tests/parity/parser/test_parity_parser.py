# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parser-mode parity tests — Method 2 of the cross-impl parity (parser) effort.

For each fixture, run the same input through Dynamo's Rust parser (via PyO3)
and the upstream Python parsers (vLLM + SGLang). Each case's YAML carries an
`expected:` block with ALL THREE impl keys (Variant A):

    expected:
      dynamo: &d_8_a                   # always present (oracle)
        calls: [...]
        normal_text: '...'
      vllm: *d_8_a                     # anchor ref when engine matches dynamo
      sglang:                          # concrete override when engine differs
        calls: [...]
        normal_text: '...'

Per-impl spec is one of:
  - `{calls, normal_text}` — concrete expected output. The test asserts the
    impl produces this exact output (whether via anchor ref to dynamo or as
    an inline divergent block, the test is the same).
  - `{unavailable: <msg>}` — impl has no parser for this family; skip.
  - `{error: <msg>}` — impl is expected to crash with this error string
    (substring match).

If the impl's actual output drifts away from the recorded spec, the assertion
fails noisily — the YAML edit needed is obvious from the diff. There's no
xfail/XPASS-strict bookkeeping because the spec IS the truth.

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

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]

FIXTURES_ROOT = Path(__file__).parent / "fixtures"

# Impl name → optional package whose absence is a legitimate skip. Anything
# else that fails inside the wrapper module (e.g. a stale `from vllm.X import
# Y` after vLLM renamed Y) is a real bug and must surface as a test ERROR,
# not a green skip — so we use `pytest.importorskip` inline rather than a
# blanket `try: import X except ImportError`.
_PACKAGE: dict[str, str] = {
    "dynamo": "dynamo._core",
    "vllm": "vllm",
    "sglang": "sglang",
}


def _case_sort_key(case_id: str) -> tuple[int, str]:
    """Sort key for case IDs that may carry a sub-letter.

    `PARSER.batch.5`   → (5, "")
    `PARSER.batch.5.d` → (5, "d")
    `PARSER.batch.8.a` → (8, "a")
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
    """
    out = []
    for fp in sorted(FIXTURES_ROOT.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        for case_id, case in doc["cases"].items():
            out.append((doc["family"], case_id, case))
    out.sort(key=lambda t: (t[0], *_case_sort_key(t[1])))
    return out


FIXTURES = _load_fixtures()


def _is_na_stub(fixture: dict[str, Any]) -> bool:
    return set(fixture) == {"description", "reason"} and all(
        isinstance(fixture[key], str) and fixture[key].strip()
        for key in ("description", "reason")
    )


def _run_parity_case(
    family: str,
    case_id: str,
    fixture: dict[str, Any],
    impl_name: str,
) -> None:
    """Single-case parity assertion shared by all three entry points
    (the all-impls `test_parity` plus the impl-specific
    `test_parity_vllm` / `test_parity_sglang`)."""
    # Skip ONLY when the optional package itself is missing — wrapper-internal
    # ImportError (e.g. a stale upstream API ref after a vLLM/SGLang rename)
    # propagates as a real test ERROR rather than a silent green skip.
    pytest.importorskip(_PACKAGE[impl_name])
    if "expected" not in fixture:
        if not _is_na_stub(fixture):
            pytest.fail(
                f"{impl_name}/{family}/{case_id}: fixture is missing expected: "
                "but is not an explicit n/a stub. n/a stubs must contain only "
                "non-empty description: and reason: fields."
            )
        pytest.skip(
            f"{impl_name}/{family}/{case_id}: n/a stub (no expected: block): "
            f"{fixture['reason']}"
        )
    parse_mod = importlib.import_module(f"tests.parity.parser.{impl_name}")
    got = parse_mod.parse(family, fixture["model_text"], fixture.get("tools"))

    spec = fixture["expected"][impl_name]

    if "unavailable" in spec:
        pytest.skip(f"{impl_name} unavailable for {family}: {spec['unavailable']}")

    if "error" in spec:
        if got.error and spec["error"] in got.error:
            return  # PASS — engine emitted the recorded error as expected
        pytest.fail(
            f"{impl_name}/{family}/{case_id}: expected error {spec['error']!r}, "
            f"got {got.error!r}"
        )

    if got.error:
        pytest.fail(f"{impl_name} crashed on {family}/{case_id}: {got.error}")

    expected = common.ParseResult(
        calls=spec["calls"],
        normal_text=spec.get("normal_text"),
    )
    got_canonical = common.canonical(got.to_dict())
    expected_canonical = common.canonical(expected.to_dict())
    assert got_canonical == expected_canonical, (
        f"\nimpl:     {impl_name}\n"
        f"family:   {family}\n"
        f"case:     {case_id}\n"
        f"expected: {expected_canonical}\n"
        f"got:      {got_canonical}\n"
    )


# Dynamo-only entry. No engine marker → runs in the default (dynamo-runtime)
# CI lane. vLLM and SGLang have their own dedicated test functions below.
@pytest.mark.parametrize(
    "family,case_id,fixture",
    [pytest.param(f, c, fx, id=f"{f}/{c}") for (f, c, fx) in FIXTURES],
)
def test_parity(
    family: str,
    case_id: str,
    fixture: dict[str, Any],
) -> None:
    _run_parity_case(family, case_id, fixture, "dynamo")


# vLLM-only entry: function-level `vllm` marker lets the vllm-runtime CI
# lane select this with `pytest -m vllm`. Each parametrized case targets
# the same fixture set but only the vllm path; dynamo/sglang are not
# exercised here.
@pytest.mark.vllm
@pytest.mark.parametrize(
    "family,case_id,fixture",
    [pytest.param(f, c, fx, id=f"{f}/{c}") for (f, c, fx) in FIXTURES],
)
def test_parity_vllm(
    family: str,
    case_id: str,
    fixture: dict[str, Any],
) -> None:
    _run_parity_case(family, case_id, fixture, "vllm")


# SGLang-only entry (sibling of test_parity_vllm; same shape).
@pytest.mark.sglang
@pytest.mark.parametrize(
    "family,case_id,fixture",
    [pytest.param(f, c, fx, id=f"{f}/{c}") for (f, c, fx) in FIXTURES],
)
def test_parity_sglang(
    family: str,
    case_id: str,
    fixture: dict[str, Any],
) -> None:
    _run_parity_case(family, case_id, fixture, "sglang")
