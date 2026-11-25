# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import sys
from pathlib import Path
from urllib.error import URLError

import case_runner
import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


def _write_case(path: Path, value: dict) -> case_runner.GoldenCase:
    path.mkdir(parents=True)
    case_path = path / "case.yaml"
    case_path.write_text(yaml.safe_dump(value, sort_keys=False))
    return case_runner.load_case(case_path)


def test_load_case_rejects_unknown_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(case_runner, "_CASES_ROOT", tmp_path)
    path = tmp_path / "features" / "invalid"
    path.mkdir(parents=True)
    case_path = path / "case.yaml"
    case_path.write_text("command: [echo]\ngoldens: [result.json]\nbackend: vllm\n")

    with pytest.raises(ValueError, match="unknown field.*backend"):
        case_runner.load_case(case_path)


def test_load_case_requires_complete_outcome_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(case_runner, "_CASES_ROOT", tmp_path)
    path = tmp_path / "features" / "invalid"
    path.mkdir(parents=True)
    case_path = path / "case.yaml"
    case_path.write_text("command: [echo]\n")

    with pytest.raises(ValueError, match="successful cases require goldens"):
        case_runner.load_case(case_path)


def test_fetch_mirror_reuses_atomic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(case_runner, "_CACHE_ROOT", tmp_path / "cache")
    calls = 0

    def urlopen(_source: str, *, timeout: int):
        nonlocal calls
        calls += 1
        assert timeout == 60
        return io.BytesIO(b'{"model_type": "fixture"}')

    monkeypatch.setattr(case_runner.urllib.request, "urlopen", urlopen)

    first = case_runner.fetch_mirror("https://example.invalid/revision/config.json")
    second = case_runner.fetch_mirror("https://example.invalid/revision/config.json")

    assert first == second
    assert first.read_bytes() == b'{"model_type": "fixture"}'
    assert calls == 1


def test_fetch_mirror_propagates_failure_without_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(case_runner, "_CACHE_ROOT", tmp_path / "cache")

    def urlopen(_source: str, *, timeout: int):
        raise URLError("unavailable")

    monkeypatch.setattr(case_runner.urllib.request, "urlopen", urlopen)

    with pytest.raises(URLError, match="unavailable"):
        case_runner.fetch_mirror("https://example.invalid/revision/config.json")

    assert not list((tmp_path / "cache").glob("**/*"))


def test_run_case_executes_generic_command_and_compares_goldens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_root = tmp_path / "cases"
    monkeypatch.setattr(case_runner, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(case_runner, "_FIXTURES_ROOT", tmp_path / "fixtures")
    case = _write_case(
        cases_root / "features" / "success",
        {
            "command": [
                "{python}",
                "-c",
                (
                    "from pathlib import Path; "
                    "Path(__import__('sys').argv[1]).write_text('value: 1\\n')"
                ),
                "{output}/result.yaml",
            ],
            "goldens": ["result.yaml"],
        },
    )
    expected = case.path / "expected"
    expected.mkdir()
    (expected / "result.yaml").write_text("value: 1\n")

    case_runner.run_case(case, tmp_path / "run")


def test_run_case_accepts_declared_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_root = tmp_path / "cases"
    monkeypatch.setattr(case_runner, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(case_runner, "_FIXTURES_ROOT", tmp_path / "fixtures")
    case = _write_case(
        cases_root / "errors" / "unsupported",
        {
            "command": [
                "{python}",
                "-c",
                "import sys; print('unsupported topology', file=sys.stderr); sys.exit(2)",
            ],
            "exit_code": "nonzero",
            "diagnostic": "unsupported topology",
        },
    )

    case_runner.run_case(case, tmp_path / "run")


def test_run_case_rejects_unexpected_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_root = tmp_path / "cases"
    monkeypatch.setattr(case_runner, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(case_runner, "_FIXTURES_ROOT", tmp_path / "fixtures")
    case = _write_case(
        cases_root / "features" / "unexpected",
        {
            "command": [sys.executable, "-c", "raise SystemExit(3)"],
            "goldens": ["result.json"],
        },
    )

    with pytest.raises(AssertionError, match="returned 3, expected 0"):
        case_runner.run_case(case, tmp_path / "run")
