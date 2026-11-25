# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml

from dynamo.aisimulate.tests.dgd.golden import (
    assert_structured_goldens,
    repository_root,
)

_ROOT = Path(__file__).resolve().parent
_CASES_ROOT = _ROOT / "cases"
_FIXTURES_ROOT = _ROOT / "fixtures"
_CACHE_ROOT = repository_root() / ".cache" / "aisimulate-golden-tests"
_FIELDS = {"command", "diagnostic", "exit_code", "goldens", "mirrors"}
_SUBSTITUTIONS = {"python", "output"}
_PLACEHOLDER = re.compile(r"{([^{}]+)}")


@dataclass(frozen=True)
class Mirror:
    destination: PurePosixPath
    source: str


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    path: Path
    command: tuple[str, ...]
    exit_code: int | str
    goldens: tuple[PurePosixPath, ...]
    diagnostic: str | None
    mirrors: tuple[Mirror, ...]


def _relative_path(value: object, *, field: str, case_path: Path) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{case_path}: {field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{case_path}: {field} must stay within its case tree")
    return path


def load_case(case_path: Path) -> GoldenCase:
    value = yaml.safe_load(case_path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{case_path} must contain one mapping")
    unknown = value.keys() - _FIELDS
    if unknown:
        raise ValueError(f"{case_path}: unknown field(s): {', '.join(sorted(unknown))}")

    command = value.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ValueError(f"{case_path}: command must be a non-empty string list")
    placeholders = {
        placeholder
        for argument in command
        for placeholder in _PLACEHOLDER.findall(argument)
    }
    unsupported = placeholders - _SUBSTITUTIONS
    if unsupported:
        raise ValueError(
            f"{case_path}: unsupported command placeholder(s): "
            f"{', '.join(sorted(unsupported))}"
        )

    exit_code = value.get("exit_code", 0)
    if isinstance(exit_code, bool) or not (
        isinstance(exit_code, int) or exit_code == "nonzero"
    ):
        raise ValueError(f"{case_path}: exit_code must be an integer or 'nonzero'")

    raw_goldens = value.get("goldens", [])
    if not isinstance(raw_goldens, list):
        raise TypeError(f"{case_path}: goldens must be a list")
    goldens = tuple(
        _relative_path(item, field="goldens", case_path=case_path)
        for item in raw_goldens
    )
    if len(set(goldens)) != len(goldens):
        raise ValueError(f"{case_path}: goldens must not contain duplicates")

    diagnostic = value.get("diagnostic")
    if diagnostic is not None and (not isinstance(diagnostic, str) or not diagnostic):
        raise ValueError(f"{case_path}: diagnostic must be a non-empty regex")
    if diagnostic is not None:
        re.compile(diagnostic)

    raw_mirrors = value.get("mirrors", {})
    if not isinstance(raw_mirrors, dict):
        raise TypeError(f"{case_path}: mirrors must be a mapping")
    mirrors = []
    for destination, source in raw_mirrors.items():
        if not isinstance(source, str) or not source.startswith("https://"):
            raise ValueError(f"{case_path}: mirror sources must be HTTPS URLs")
        mirrors.append(
            Mirror(
                destination=_relative_path(
                    destination, field="mirrors", case_path=case_path
                ),
                source=source,
            )
        )

    if exit_code == 0:
        if not goldens:
            raise ValueError(f"{case_path}: successful cases require goldens")
        if diagnostic is not None:
            raise ValueError(
                f"{case_path}: successful cases must not declare a diagnostic"
            )
    else:
        if goldens:
            raise ValueError(f"{case_path}: failing cases must not declare goldens")
        if diagnostic is None:
            raise ValueError(f"{case_path}: failing cases require a diagnostic")

    relative = case_path.parent.relative_to(_CASES_ROOT)
    return GoldenCase(
        case_id=relative.as_posix(),
        path=case_path.parent,
        command=tuple(command),
        exit_code=exit_code,
        goldens=goldens,
        diagnostic=diagnostic,
        mirrors=tuple(mirrors),
    )


def discover_cases() -> tuple[GoldenCase, ...]:
    cases = tuple(load_case(path) for path in sorted(_CASES_ROOT.glob("**/case.yaml")))
    if not cases:
        raise RuntimeError(f"no golden cases found under {_CASES_ROOT}")
    return cases


def _cache_path(source: str) -> Path:
    digest = hashlib.sha256(source.encode()).hexdigest()
    filename = Path(urlparse(source).path).name or "mirror"
    return _CACHE_ROOT / digest / filename


def fetch_mirror(source: str) -> Path:
    cached = _cache_path(source)
    if cached.is_file():
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(source, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=cached.parent, delete=False
            ) as temporary:
                shutil.copyfileobj(response, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
        os.replace(temporary_path, cached)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if not cached.exists():
            try:
                cached.parent.rmdir()
            except OSError:
                pass
    return cached


def _ignore_staged_files(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "expected"
        or name == "__pycache__"
        or ".new." in name
        or name.endswith((".pyc", ".pyo"))
    }


def stage_case(case: GoldenCase, workspace: Path) -> Path:
    staged_root = workspace / "e2e"
    staged_case = staged_root / "cases" / case.path.relative_to(_CASES_ROOT)
    shutil.copytree(case.path, staged_case, ignore=_ignore_staged_files)
    if _FIXTURES_ROOT.is_dir():
        shutil.copytree(
            _FIXTURES_ROOT,
            staged_root / "fixtures",
            ignore=_ignore_staged_files,
        )
    for mirror in case.mirrors:
        destination = staged_case / mirror.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fetch_mirror(mirror.source), destination)
    return staged_case


def run_case(case: GoldenCase, tmp_path: Path) -> None:
    staged_case = stage_case(case, tmp_path / "workspace")
    output = tmp_path / "output"
    output.mkdir()
    substitutions = {"python": sys.executable, "output": str(output)}
    command = [argument.format(**substitutions) for argument in case.command]
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    try:
        result = subprocess.run(
            command,
            cwd=staged_case,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"{case.case_id}: command timed out: {command}") from exc

    expected_code = case.exit_code
    if expected_code == "nonzero":
        exit_matches = result.returncode > 0
    else:
        exit_matches = result.returncode == expected_code
    if not exit_matches:
        raise AssertionError(
            f"{case.case_id}: command returned {result.returncode}, "
            f"expected {expected_code}\ncommand: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    if case.diagnostic is not None:
        diagnostic = f"{result.stdout}\n{result.stderr}"
        if re.search(case.diagnostic, diagnostic) is None:
            raise AssertionError(
                f"{case.case_id}: expected diagnostic {case.diagnostic!r}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return

    missing = [str(path) for path in case.goldens if not (output / path).is_file()]
    if missing:
        raise AssertionError(
            f"{case.case_id}: command did not produce declared output(s): "
            f"{', '.join(missing)}\ncommand: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert_structured_goldens(
        (case.path / "expected" / path, output / path) for path in case.goldens
    )
