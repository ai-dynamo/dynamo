# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for copyable CLI commands in the Dynamo documentation."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]

REPO_ROOT = Path(__file__).resolve().parents[1]
FERN_PAGES = REPO_ROOT / "docs" / "fern" / "pages"
QUICKSTART = FERN_PAGES / "cli" / "getting-started" / "quickstart.mdx"
TRANSLATED_QUICKSTARTS = [
    REPO_ROOT
    / "docs"
    / "fern"
    / "translations"
    / "zh-CN"
    / "pages"
    / "cli"
    / relative_path
    for relative_path in (
        Path("getting-started/quickstart.mdx"),
        Path("installation/install-dynamo.mdx"),
    )
]
TRANSLATED_LOCAL_INSTALL = TRANSLATED_QUICKSTARTS[1]
DISAGGREGATED_SERVING = FERN_PAGES / "cli" / "disaggregated-serving" / "overview.mdx"
KV_AWARE_ROUTING = FERN_PAGES / "cli" / "kv-aware-routing" / "overview.mdx"
MODEL_DEPLOYMENT = FERN_PAGES / "cli" / "model-deployment" / "introduction.mdx"
RUNTIME_CONFIGURATION = (
    FERN_PAGES / "reference" / "components" / "runtime-configuration.mdx"
)
CONDITIONAL_DISAGGREGATION = (
    FERN_PAGES
    / "developer-guide"
    / "advanced-customizations"
    / "conditional-disaggregation.md"
)
FREEZE_INSTALL_SELECTOR = (
    REPO_ROOT / ".github" / "scripts" / "freeze_install_selector.py"
)
VLLM_RUNTIME_DOCKERFILE = (
    REPO_ROOT / "container" / "templates" / "vllm_runtime.Dockerfile"
)
SGLANG_RUNTIME_DOCKERFILE = (
    REPO_ROOT / "container" / "templates" / "sglang_runtime.Dockerfile"
)
MODEL_ID = "Qwen/Qwen3-0.6B"


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash(?:[^\n]*)\n(.*?)```", text, flags=re.DOTALL)


def _first_bash_block_after(path: Path, marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    _, separator, tail = text.partition(marker)
    assert separator, f"marker not found in {path}: {marker}"
    blocks = _bash_blocks(tail)
    assert blocks, f"no bash block follows {marker!r} in {path}"
    return blocks[0].strip()


def _tab_bash_blocks(path: Path, title: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tabs = re.findall(
        rf'<Tab title="{re.escape(title)}"[^>]*>(.*?)</Tab>',
        text,
        flags=re.DOTALL,
    )
    blocks = [block for tab in tabs for block in _bash_blocks(tab)]
    assert blocks, f"no bash blocks found for the {title} tab in {path}"
    return blocks


def _command_tokens(command: str) -> list[str]:
    return shlex.split(command.replace("\\\n", " "), comments=True)


def _frontend_config(command: str) -> tuple[FrontendConfig, list[str]]:
    tokens = _command_tokens(command)
    assert tokens[:3] == ["python", "-m", "dynamo.frontend"]

    parser = argparse.ArgumentParser(allow_abbrev=False)
    FrontendArgGroup().add_arguments(parser)
    args, unknown = parser.parse_known_args(tokens[3:])
    return FrontendConfig.from_cli_args(args), unknown


def _clear_dynamo_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("DYN_"):
            monkeypatch.delenv(name)


def test_release_quickstart_vllm_container_starts_a_shell(tmp_path: Path) -> None:
    """A release snapshot must not publish a vLLM image with no command."""
    snapshot = tmp_path / "quickstart.mdx"
    snapshot.write_text(QUICKSTART.read_text(encoding="utf-8"), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(FREEZE_INSTALL_SELECTOR), str(snapshot), "9.9.9"],
        check=True,
        capture_output=True,
        text=True,
    )

    generated = snapshot.read_text(encoding="utf-8")
    vllm_command = next(
        line.strip() for line in generated.splitlines() if "vllm-runtime:9.9.9" in line
    )
    assert _command_tokens(vllm_command)[-1] == "bash"

    xpu_vllm_command = next(
        line.strip()
        for line in generated.splitlines()
        if "container/run.sh" in line and "vllm-xpu-runtime" in line
    )
    assert _command_tokens(xpu_vllm_command)[-2:] == ["--", "bash"]

    xpu_sglang_command = next(
        line.strip()
        for line in generated.splitlines()
        if "container/run.sh" in line and "sglang-xpu-runtime" in line
    )
    assert _command_tokens(xpu_sglang_command)[-1] == "-it"


@pytest.mark.parametrize("path", TRANSLATED_QUICKSTARTS)
def test_translated_vllm_container_examples_start_a_shell(path: Path) -> None:
    """Static translated examples must launch the commandless vLLM image."""
    command = next(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "docker run " in line and "/vllm-runtime:" in line
    )

    assert _command_tokens(command)[-1] == "bash"


def test_translated_xpu_vllm_commands_start_a_shell() -> None:
    """Static XPU vLLM examples must pass a command through run.sh."""
    lines = TRANSLATED_LOCAL_INSTALL.read_text(encoding="utf-8").splitlines()
    vllm_commands = [
        line.strip()
        for line in lines
        if "container/run.sh" in line and "vllm-xpu-runtime" in line
    ]
    sglang_commands = [
        line.strip()
        for line in lines
        if "container/run.sh" in line and "sglang-xpu-runtime" in line
    ]

    assert vllm_commands
    assert all(
        _command_tokens(command)[-2:] == ["--", "bash"] for command in vllm_commands
    )
    assert sglang_commands
    assert all(_command_tokens(command)[-1] == "-it" for command in sglang_commands)


@pytest.mark.parametrize(
    ("path", "title", "occurrence", "dynamo_home", "runtime_dockerfile"),
    [
        (DISAGGREGATED_SERVING, "vLLM", 0, "/opt/dynamo", VLLM_RUNTIME_DOCKERFILE),
        (DISAGGREGATED_SERVING, "SGLang", 0, None, SGLANG_RUNTIME_DOCKERFILE),
        (KV_AWARE_ROUTING, "vLLM", 0, "/opt/dynamo", VLLM_RUNTIME_DOCKERFILE),
        (KV_AWARE_ROUTING, "SGLang", 0, None, SGLANG_RUNTIME_DOCKERFILE),
        (KV_AWARE_ROUTING, "vLLM", 1, "/opt/dynamo", VLLM_RUNTIME_DOCKERFILE),
        (KV_AWARE_ROUTING, "SGLang", 1, None, SGLANG_RUNTIME_DOCKERFILE),
    ],
)
def test_vllm_and_sglang_example_directories_resolve_in_runtime_images(
    path: Path,
    title: str,
    occurrence: int,
    dynamo_home: str | None,
    runtime_dockerfile: Path,
) -> None:
    """Documented paths must resolve against the shipped image filesystem."""
    block = _tab_bash_blocks(path, title)[occurrence]
    cd_line = next(
        line.strip() for line in block.splitlines() if line.strip().startswith("cd ")
    )
    cd_operand = shlex.split(cd_line)[1]

    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if dynamo_home is None:
        env.pop("DYNAMO_HOME", None)
    else:
        env["DYNAMO_HOME"] = dynamo_home
    resolved = subprocess.run(
        ["bash", "-u", "-c", f'printf "%s" {cd_operand}'],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout

    dockerfile = runtime_dockerfile.read_text(encoding="utf-8")
    assert re.search(
        r"^COPY\b.*\bexamples\s+/workspace/examples$", dockerfile, re.MULTILINE
    )
    assert resolved.startswith("/workspace/examples/")

    documented_directory = REPO_ROOT / Path(resolved).relative_to("/workspace")
    launch_line = next(
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("cd ")
    )
    launch_script = _command_tokens(launch_line)[-1].removeprefix("./")
    assert documented_directory.is_dir()
    assert (documented_directory / launch_script).is_file()


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_worker_commands_are_complete_in_a_fresh_terminal(
    backend: str, tmp_path: Path
) -> None:
    """A fresh second terminal must not depend on MODEL from the first one."""
    text = MODEL_DEPLOYMENT.read_text(encoding="utf-8")
    _, separator, worker_step = text.partition('<Step title="Start the worker">')
    assert separator
    worker_step = worker_step.partition("</Step>")[0]
    command = next(
        block.strip()
        for block in _bash_blocks(worker_step)
        if f"python -m dynamo.{backend}" in block
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURED_ARGS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    captured_args = tmp_path / "args.txt"
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "CAPTURED_ARGS": str(captured_args),
    }
    result = subprocess.run(
        ["bash", "-u", "-c", command],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    args = captured_args.read_text(encoding="utf-8").splitlines()
    model_flag = "--model" if backend == "vllm" else "--model-path"
    assert args[args.index(model_flag) + 1] == MODEL_ID


def test_frontend_runtime_example_uses_only_supported_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Local frontend example must pass the frontend's real parser."""
    _clear_dynamo_env(monkeypatch)
    local_block = _first_bash_block_after(RUNTIME_CONFIGURATION, '<Tab title="Local">')
    frontend_command = local_block.partition("# Frontend\n")[2].partition("\n\n")[0]

    config, unknown = _frontend_config(frontend_command)

    assert unknown == []
    config.validate()


def test_conditional_disaggregation_example_passes_real_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-aware policy example must include its required threshold."""
    _clear_dynamo_env(monkeypatch)
    command = _first_bash_block_after(
        CONDITIONAL_DISAGGREGATION,
        "Tune the policy with `--router-conditional-disagg-config`.",
    )

    config, unknown = _frontend_config(command)

    assert unknown == []
    config.validate()
