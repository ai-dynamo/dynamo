# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visibility narrowing for the ``components/`` Python surface.

Only symbols a module positively declares public (literal ``__all__``
membership, or a curated ``.pyi`` stub, or an explicit annotation) may reach
``stable``; everything else floors at ``experimental``.
"""

from __future__ import annotations

from pathlib import Path

from api_surface.models import SurfaceSnapshot
from api_surface.snapshot import build_snapshot


def _repo_with_module(tmp_path: Path, source: str, module: str = "mod") -> Path:
    repo = tmp_path / "repo"
    package = repo / "components/src/dynamo"
    package.mkdir(parents=True)
    package.joinpath(f"{module}.py").write_text(source, encoding="utf-8")
    return repo


def _symbols(repo: Path) -> dict[str, object]:
    result = build_snapshot(repo, "1.4.0", stubs=[])
    snapshot = SurfaceSnapshot.from_dict(result.data["snapshot"])
    return {symbol.id: symbol for symbol in snapshot.symbols}


def test_unlisted_function_is_experimental(tmp_path: Path) -> None:
    repo = _repo_with_module(tmp_path, "def helper() -> None:\n    pass\n")

    symbols = _symbols(repo)

    assert symbols["python:dynamo.mod.helper"].stability == "experimental"


def test_all_listed_function_is_stable(tmp_path: Path) -> None:
    repo = _repo_with_module(
        tmp_path,
        '__all__ = ["handler"]\n\ndef handler() -> None:\n    pass\n',
    )

    symbols = _symbols(repo)

    assert symbols["python:dynamo.mod.handler"].stability == "stable"


def test_member_of_all_listed_class_is_stable(tmp_path: Path) -> None:
    repo = _repo_with_module(
        tmp_path,
        '__all__ = ["Client"]\n\n'
        "class Client:\n"
        "    def run(self) -> None:\n"
        "        pass\n",
    )

    symbols = _symbols(repo)

    assert symbols["python:dynamo.mod.Client"].stability == "stable"
    assert symbols["python:dynamo.mod.Client.run"].stability == "stable"


def test_computed_all_is_treated_as_absent(tmp_path: Path) -> None:
    repo = _repo_with_module(
        tmp_path,
        '__all__ = [name for name in dir() if not name.startswith("_")]\n\n'
        "def handler() -> None:\n"
        "    pass\n",
    )

    symbols = _symbols(repo)

    assert symbols["python:dynamo.mod.handler"].stability == "experimental"


def test_stub_symbol_stays_stable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("api.pyi").write_text(
        "def runtime() -> None: ...\n", encoding="utf-8"
    )

    result = build_snapshot(repo, "1.4.0", stubs=[("api.pyi", "dynamo.api")])
    snapshot = SurfaceSnapshot.from_dict(result.data["snapshot"])
    symbols = {symbol.id: symbol for symbol in snapshot.symbols}

    assert symbols["python:dynamo.api.runtime"].stability == "stable"


def test_annotation_promotes_unlisted_symbol_to_stable(tmp_path: Path) -> None:
    repo = _repo_with_module(tmp_path, "def helper() -> None:\n    pass\n")
    annotations = repo / ".github/api-surface"
    annotations.mkdir(parents=True)
    annotations.joinpath("annotations.yaml").write_text(
        "symbols:\n" '  "python:dynamo.mod.helper":\n' "    stability: stable\n",
        encoding="utf-8",
    )

    symbols = _symbols(repo)

    assert symbols["python:dynamo.mod.helper"].stability == "stable"
