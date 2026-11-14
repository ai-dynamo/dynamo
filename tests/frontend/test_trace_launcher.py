# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The repository launcher must use its own checkout from any working directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).parent / "trace_replay" / "frontend-replay"
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.parallel,
    pytest.mark.timeout(30),
]


def _package(root: Path, code: str) -> Path:
    package = root
    for name in ("tests", "frontend", "trace_replay"):
        package /= name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
    (package / "build_corpus.py").write_text(code)
    return package


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "checkout with spaces"
    package = _package(
        root,
        "import json, sys\n"
        "def main(argv):\n"
        "    print(json.dumps({'origin': __file__, 'argv': argv, 'sys_path': sys.path}))\n"
        "    return 0\n",
    )
    launcher = package / LAUNCHER.name
    shutil.copyfile(LAUNCHER, launcher)
    launcher.chmod(0o755)
    return root, launcher


def test_launcher_uses_its_checkout_over_cwd_and_pythonpath(tmp_path: Path) -> None:
    root, launcher = _checkout(tmp_path)
    decoy = "raise RuntimeError('a different checkout was imported')\n"
    cwd = tmp_path / "caller"
    _package(cwd, decoy)
    other = tmp_path / "other checkout"
    _package(other, decoy)
    environment = {**os.environ, "PYTHONPATH": str(other), "PYTHONNOUSERSITE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "build-corpus",
            "--requests",
            "requests with spaces.jsonl",
        ],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert Path(observed["origin"]).is_relative_to(root)
    assert observed["argv"] == ["--requests", "requests with spaces.jsonl"]
    assert observed["sys_path"][0] == str(root)
    assert str(launcher.parent) not in observed["sys_path"]


def test_executable_launcher_preserves_arguments_and_uv_isolation(
    tmp_path: Path,
) -> None:
    _, launcher = _checkout(tmp_path)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_uv = binaries / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['DYN_REPLAY_TEST_UV_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
        "index = sys.argv.index('--script') + 1\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[index:]])\n"
    )
    fake_uv.chmod(0o755)
    log = tmp_path / "uv-arguments.json"
    args = [
        "build-corpus",
        "--requests",
        "input with spaces.jsonl",
        "--output",
        "output with spaces",
    ]
    environment = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "DYN_REPLAY_TEST_UV_LOG": str(log),
    }
    result = subprocess.run(
        [str(launcher), *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(log.read_text()) == [
        "run",
        "--no-project",
        "--script",
        str(launcher),
        *args,
    ]
    assert json.loads(result.stdout)["argv"] == args[1:]
    metadata_text = (
        LAUNCHER.read_text().split("# /// script\n", 1)[1].split("# ///", 1)[0]
    )
    metadata = tomllib.loads(
        "\n".join(line.removeprefix("# ") for line in metadata_text.splitlines())
    )
    assert metadata["requires-python"] == "==3.12.*"
    assert metadata["dependencies"] == ["tokenizers==0.21.4", "requests==2.33.1"]
    assert metadata["tool"]["uv"]["exclude-newer"].startswith("2026-11-11")


def test_launcher_build_help_from_another_directory(tmp_path: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    result = subprocess.run(
        [sys.executable, str(LAUNCHER.resolve()), "build-corpus", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--reference-url" in result.stdout
    assert "--profile" in result.stdout
    assert "--cache-dir" in result.stdout


def test_launcher_rejects_unknown_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(LAUNCHER.resolve()), "unknown-command"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "build-corpus" in result.stderr
