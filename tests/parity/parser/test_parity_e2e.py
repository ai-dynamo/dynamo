# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E parity tests against real `vllm serve` / `sglang.launch_server`
HTTP processes.

For each fixture, force the server to emit `model_text` verbatim via
constrained decoding, then assert the parsed `tool_calls` JSON matches
`expected`.

Gated behind the `e2e` pytest marker; run with:

    pytest tests/parity/parser/test_parity_e2e.py -m e2e -v

Per-impl skip rules:
- `vllm` tests skip when the `vllm` CLI is not on PATH.
- `sglang` tests skip when the `sglang` Python package is not importable.

So a single-framework devcontainer (vllm container or sglang container)
only runs the tests for the impl it has, and reports the others as
skipped — not as errors.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.parity import common
from tests.parity.parser import client as e2e_client
from tests.parity.parser.server import get_or_boot_server

pytestmark = [
    pytest.mark.e2e,  # Test Type: integration / end-to-end (boots a real server)
    pytest.mark.pre_merge,  # Lifecycle: runs in pre-merge CI
    pytest.mark.gpu_1,  # Hardware: `--load-format dummy` still allocates ~2.5 GiB on a CUDA device
]

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


# Per-impl availability gate. Each entry: (impl_name, lambda check).
# `True` means tests for that impl run; `False` skips them.
_IMPL_AVAILABLE: dict[str, bool] = {
    "vllm": shutil.which("vllm") is not None,
    "sglang": importlib.util.find_spec("sglang") is not None,
}


# Families with a vLLM/SGLang parser registered AND a tokenizer-compatible
# default model — POC scope. Other families (deepseek_v3_1, minimax_m2,
# harmony) need per-family tokenizer overrides; deferred.
_POC_FAMILIES = ("kimi_k2", "qwen3_coder", "glm47")
_IMPLS = ("vllm", "sglang")


# (impl, family, case_id) -> reason. Real cross-impl divergences that
# reproduce through the HTTP stack — recorded rather than asserted-against.
_TRAILING_NORMAL_TEXT_DROP = "drops trailing normal_text after tool-call wrapper end"
_RECOVERY_CONTRACT = "impl-defined recovery contract (see PARSER_CASES.md)"
KNOWN_DIVERGENCES: dict[tuple[str, str, str], str] = {
    # vLLM findings
    ("vllm", "kimi_k2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "qwen3_coder", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("vllm", "qwen3_coder", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("vllm", "qwen3_coder", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("vllm", "glm47", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    # SGLang findings: trailing-text drop mirrors vLLM on the same families,
    # plus a wider recovery-contract footprint — SGLang's kimi_k2 parser
    # drops the call entirely on malformed JSON args (vLLM and Dynamo both
    # recover by surfacing the truncated args as a raw string).
    ("sglang", "kimi_k2", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "kimi_k2", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "qwen3_coder", "PARSER.batch.4"): _RECOVERY_CONTRACT,
    ("sglang", "qwen3_coder", "PARSER.batch.5"): _RECOVERY_CONTRACT,
    ("sglang", "qwen3_coder", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
    ("sglang", "glm47", "PARSER.batch.8"): _TRAILING_NORMAL_TEXT_DROP,
}


def _load_cases(family: str) -> list[tuple[str, dict[str, Any]]]:
    fp = FIXTURES_ROOT / family / "PARSER.batch.json"
    doc = json.loads(fp.read_text())
    return sorted(
        [(f"PARSER.batch.{n}", c) for n, c in doc["cases"].items()],
        key=lambda kv: int(kv[0].rsplit(".", 1)[1]),
    )


# Per-param impl markers so CI marker filters
# (`pre_merge and vllm and gpu_1`, `pre_merge and sglang and gpu_1`) pick up
# only the right subset in each container.
@pytest.mark.parametrize(
    "impl,family,case_id,fixture",
    [
        pytest.param(impl, family, case_id, case, marks=getattr(pytest.mark, impl))
        for impl in _IMPLS
        for family in _POC_FAMILIES
        for case_id, case in _load_cases(family)
    ],
    ids=[
        f"{family}/{case_id}/{impl}"
        for impl in _IMPLS
        for family in _POC_FAMILIES
        for case_id, _ in _load_cases(family)
    ],
)
def test_e2e(
    impl: str,
    family: str,
    case_id: str,
    fixture: dict[str, Any],
    e2e_server_cache: dict,
    e2e_server_lifecycles: list,
) -> None:
    if not _IMPL_AVAILABLE[impl]:
        pytest.skip(f"{impl} not installed in this container")

    divergence = KNOWN_DIVERGENCES.get((impl, family, case_id))
    if divergence is not None:
        pytest.xfail(f"known divergence: {divergence}")

    base_url = get_or_boot_server(
        impl=impl,
        parser_family=family,
        cache=e2e_server_cache,
        lifecycles=e2e_server_lifecycles,
    )
    got = e2e_client.parse(
        impl,
        family,
        fixture["model_text"],
        fixture["tools"],
        base_url=base_url,
    )
    if got.error:
        pytest.fail(f"{impl} e2e errored: {got.error}")

    expected = common.ParseResult(
        calls=fixture["expected"]["calls"],
        normal_text=fixture["expected"].get("normal_text"),
    )
    assert common.canonical(got.to_dict()) == common.canonical(expected.to_dict()), (
        f"\nimpl:     {impl} (e2e)\n"
        f"family:   {family}\n"
        f"case:     {case_id}\n"
        f"expected: {common.canonical(expected.to_dict())}\n"
        f"got:      {common.canonical(got.to_dict())}\n"
    )
