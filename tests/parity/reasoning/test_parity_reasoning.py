# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning parser parity tests.

For each fixture, run the same model text through Dynamo's Rust reasoning
parser and peer reasoning parser implementations from vLLM / SGLang when
available. The YAML fixture is the contract:

    expected:
      dynamo: &d_1
        reasoning_text: "..."
        normal_text: "..."
      vllm: *d_1
      sglang:
        unavailable: "..."

Run:
    python3 -m pytest tests/parity/reasoning/test_parity_reasoning.py -v
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

_PACKAGE: dict[str, str] = {
    "dynamo": "dynamo._core",
    "vllm": "vllm",
    "sglang": "sglang",
}


def _case_sort_key(case_id: str) -> tuple[int, int, str]:
    parts = case_id.split(".")
    mode = 0 if parts[1] == "batch" else 1
    top = int(parts[2])
    sub = parts[3] if len(parts) > 3 else ""
    return (mode, top, sub)


def _load_fixtures() -> list[tuple[str, str, str, dict[str, Any]]]:
    out = []
    for fp in sorted(FIXTURES_ROOT.glob("*/REASONING.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        mode = doc["mode"]
        for case_id, case in doc["cases"].items():
            out.append((doc["family"], mode, case_id, case))
    out.sort(key=lambda t: (t[0], *_case_sort_key(t[2])))
    return out


FIXTURES = tuple(_load_fixtures())


def _is_na_stub(fixture: dict[str, Any]) -> bool:
    return set(fixture) <= {"description", "reason", "ref", "spec_ref"} and all(
        isinstance(fixture[key], str) and fixture[key].strip()
        for key in ("description", "reason")
    )


def _run_parity_case(
    family: str,
    mode: str,
    case_id: str,
    fixture: dict[str, Any],
    impl_name: str,
) -> None:
    pytest.importorskip(_PACKAGE[impl_name])
    if "expected" not in fixture:
        if not _is_na_stub(fixture):
            pytest.fail(
                f"{impl_name}/{family}/{case_id}: fixture is missing expected: "
                "but is not an explicit n/a stub"
            )
        pytest.skip(
            f"{impl_name}/{family}/{case_id}: n/a stub (no expected block): "
            f"{fixture['reason']}"
        )

    spec = fixture["expected"][impl_name]
    if "unavailable" in spec:
        pytest.skip(f"{impl_name} unavailable for {family}: {spec['unavailable']}")

    parse_mod = importlib.import_module(f"tests.parity.reasoning.{impl_name}")
    got = parse_mod.parse(family, fixture, mode)

    if "error" in spec:
        if got.error and spec["error"] in got.error:
            return
        pytest.fail(
            f"{impl_name}/{family}/{case_id}: expected error {spec['error']!r}, "
            f"got {got.error!r}"
        )

    if got.error:
        pytest.fail(f"{impl_name} crashed on {family}/{case_id}: {got.error}")

    expected = common.ReasoningResult(
        reasoning_text=spec.get("reasoning_text"),
        normal_text=spec.get("normal_text"),
    )
    got_canonical = common.canonical(got.to_dict())
    expected_canonical = common.canonical(expected.to_dict())
    assert got_canonical == expected_canonical, (
        f"\nimpl:     {impl_name}\n"
        f"family:   {family}\n"
        f"mode:     {mode}\n"
        f"case:     {case_id}\n"
        f"expected: {expected_canonical}\n"
        f"got:      {got_canonical}\n"
    )


@pytest.mark.parametrize(
    "family,mode,case_id,fixture",
    [pytest.param(f, m, c, fx, id=f"{f}/{c}") for (f, m, c, fx) in FIXTURES],
)
def test_reasoning_parity(
    family: str,
    mode: str,
    case_id: str,
    fixture: dict[str, Any],
) -> None:
    _run_parity_case(family, mode, case_id, fixture, "dynamo")


@pytest.mark.vllm
@pytest.mark.parametrize(
    "family,mode,case_id,fixture",
    [pytest.param(f, m, c, fx, id=f"{f}/{c}") for (f, m, c, fx) in FIXTURES],
)
def test_reasoning_parity_vllm(
    family: str,
    mode: str,
    case_id: str,
    fixture: dict[str, Any],
) -> None:
    _run_parity_case(family, mode, case_id, fixture, "vllm")


@pytest.mark.sglang
@pytest.mark.parametrize(
    "family,mode,case_id,fixture",
    [pytest.param(f, m, c, fx, id=f"{f}/{c}") for (f, m, c, fx) in FIXTURES],
)
def test_reasoning_parity_sglang(
    family: str,
    mode: str,
    case_id: str,
    fixture: dict[str, Any],
) -> None:
    _run_parity_case(family, mode, case_id, fixture, "sglang")
