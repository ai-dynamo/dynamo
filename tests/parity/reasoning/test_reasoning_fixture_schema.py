# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_IMPLS = ("dynamo", "vllm", "sglang")
MODES = {"batch", "stream"}
STUB_KEYS = {"description", "reason", "ref", "spec_ref"}
CASE_KEYS = {
    "description",
    "ref",
    "spec_ref",
    "model_text",
    "chunks",
    "chat_template_kwargs",
    "expected",
    "reason",
    "dynamo_leak",
}
BLOCK_KEYS = {"reasoning_text", "normal_text", "reason", "error", "unavailable"}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{path}: fixture must be a mapping"
    return data


def _location(path: Path, case_id: str) -> str:
    return f"{path.relative_to(FIXTURES_DIR.parent.parent.parent)}:{case_id}"


def _assert_expected_block(path: Path, case_id: str, impl: str, block: object) -> None:
    where = f"{_location(path, case_id)} expected.{impl}"
    assert isinstance(block, dict), f"{where}: must be a mapping"
    unknown_keys = set(block) - BLOCK_KEYS
    assert not unknown_keys, f"{where}: unknown keys {sorted(unknown_keys)}"

    terminal_keys = {"error", "unavailable"} & set(block)
    assert len(terminal_keys) <= 1, f"{where}: use only one of error/unavailable"
    if terminal_keys:
        assert (
            set(block) <= terminal_keys
        ), f"{where}: terminal block must only contain {terminal_keys}"
        value = block[next(iter(terminal_keys))]
        assert (
            isinstance(value, str) and value
        ), f"{where}: terminal message must be non-empty"
        return

    assert "reasoning_text" in block, f"{where}: missing reasoning_text"
    assert "normal_text" in block, f"{where}: missing normal_text"
    assert isinstance(
        block["reasoning_text"], str
    ), f"{where}: reasoning_text must be a string"
    assert isinstance(
        block["normal_text"], str
    ), f"{where}: normal_text must be a string"
    if "reason" in block:
        assert (
            isinstance(block["reason"], str) and block["reason"]
        ), f"{where}: reason must be non-empty"


def test_reasoning_fixture_schema() -> None:
    paths = sorted(FIXTURES_DIR.glob("*/REASONING.*.yaml"))
    assert paths, "reasoning fixtures must exist"

    seen_case_ids: set[tuple[str, str]] = set()
    for path in paths:
        data = _load_yaml(path)
        family = data.get("family")
        mode = data.get("mode")
        cases = data.get("cases")

        assert (
            isinstance(family, str) and family
        ), f"{path}: family must be a non-empty string"
        assert mode in MODES, f"{path}: mode must be one of {sorted(MODES)}"
        assert (
            isinstance(cases, dict) and cases
        ), f"{path}: cases must be a non-empty mapping"

        for case_id, case in cases.items():
            where = _location(path, case_id)
            assert isinstance(case_id, str) and case_id.startswith(
                f"REASONING.{mode}."
            ), f"{where}: case id must match fixture mode"
            seen_key = (family, case_id)
            assert (
                seen_key not in seen_case_ids
            ), f"{where}: duplicate case id for family"
            seen_case_ids.add(seen_key)

            assert isinstance(case, dict), f"{where}: case must be a mapping"
            unknown_keys = set(case) - CASE_KEYS
            assert not unknown_keys, f"{where}: unknown keys {sorted(unknown_keys)}"
            assert (
                isinstance(case.get("description"), str) and case["description"]
            ), f"{where}: description must be non-empty"

            if "expected" not in case:
                assert (
                    set(case) <= STUB_KEYS
                ), f"{where}: stub case must only use {sorted(STUB_KEYS)}"
                assert (
                    isinstance(case.get("reason"), str) and case["reason"]
                ), f"{where}: stub case must document why it is not applicable"
                continue

            if mode == "batch":
                assert "model_text" in case, f"{where}: batch case missing model_text"
                assert isinstance(
                    case["model_text"], str
                ), f"{where}: model_text must be a string"
            else:
                assert "chunks" in case, f"{where}: stream case missing chunks"
                assert isinstance(
                    case["chunks"], list
                ), f"{where}: chunks must be a list"

            dynamo_leak = case.get("dynamo_leak", False)
            assert isinstance(
                dynamo_leak, bool
            ), f"{where}: dynamo_leak must be boolean"
            if dynamo_leak:
                assert not case_id.startswith(
                    "REASONING.batch.3."
                ), f"{where}: downstream tool call handoff must not be marked as a reasoning leak"

            expected = case["expected"]
            assert isinstance(expected, dict), f"{where}: expected must be a mapping"
            assert set(expected) == set(
                EXPECTED_IMPLS
            ), f"{where}: expected must include dynamo/vllm/sglang"
            for impl in EXPECTED_IMPLS:
                _assert_expected_block(path, case_id, impl, expected[impl])
