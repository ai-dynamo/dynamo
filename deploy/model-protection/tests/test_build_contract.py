# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
]
DEPLOY_DIR = Path(__file__).resolve().parents[1]


def test_runtime_check_reports_version_and_tpm_separately(monkeypatch, capsys):
    package = ModuleType("dynamo")
    core = ModuleType("dynamo._core")
    core.model_protection_tpm_enabled = True
    core.model_protection_runtime_enabled = True
    core.__file__ = "/test/_core.so"
    package._core = core
    monkeypatch.setitem(sys.modules, "dynamo", package)
    monkeypatch.setitem(sys.modules, "dynamo._core", core)
    vllm_package = ModuleType("dynamo.vllm")
    loader = ModuleType("dynamo.vllm.protection_bootstrap")
    loader.SUPPORTED_VLLM_VERSION = "0.30.0"
    vllm_package.protection_bootstrap = loader
    monkeypatch.setitem(sys.modules, "dynamo.vllm", vllm_package)
    monkeypatch.setitem(sys.modules, "dynamo.vllm.protection_bootstrap", loader)
    spec = importlib.util.spec_from_file_location(
        "protected_image_check", DEPLOY_DIR / "check-runtime.py"
    )
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.setenv("EXPECTED_VLLM_VERSION", "0.30.0")
    monkeypatch.setattr(check, "version", lambda _: "0.29.0")
    with pytest.raises(SystemExit, match="vLLM version mismatch"):
        check.main()
    assert "vLLM=0.29.0" in capsys.readouterr().out
    monkeypatch.setattr(check, "version", lambda _: "0.30.0")
    core.model_protection_tpm_enabled = False
    with pytest.raises(SystemExit, match="TPM feature missing"):
        check.main()
    core.model_protection_tpm_enabled = True
    core.model_protection_runtime_enabled = False
    with pytest.raises(SystemExit, match="Layer runtime missing"):
        check.main()
    core.model_protection_runtime_enabled = True
    loader.SUPPORTED_VLLM_VERSION = "0.29.0"
    with pytest.raises(SystemExit, match="Protected loader version mismatch"):
        check.main()
    loader.SUPPORTED_VLLM_VERSION = "0.30.0"
    check.main()


@pytest.mark.timeout(30)
@pytest.mark.parametrize("skip_base", ["false", "true"])
def test_builder_pins_base_and_checks_before_building_wheel(tmp_path, skip_base):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    stubs = {
        "python3": "exit 0\n",
        "docker": 'printf "docker %s\\n" "$*" >> "$BUILD_CALLS"\n',
        "maturin": 'printf "maturin %s\\n" "$*" >> "$BUILD_CALLS"\nwhile [ "$#" -gt 0 ]; do\n  if [ "$1" = "--out" ]; then shift; touch "$1/ai_dynamo_runtime-test.whl"; exit 0; fi\n  shift\ndone\nexit 1\n',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\n" + body)
        stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        BUILD_CALLS=str(calls),
        SKIP_BASE_BUILD=skip_base,
        TMPDIR=str(tmp_path),
    )
    env.pop("VLLM_PROTECTED_VERSION", None)
    subprocess.run(
        ["bash", str(DEPLOY_DIR / "build-protected-image.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    log = calls.read_text()
    if skip_base == "false":
        assert "RUNTIME_IMAGE_TAG=v0.30.0-ubuntu2404" in log
        assert "VLLM_OMNI_REF=v0.30.0rc1" in log
    else:
        assert "RUNTIME_IMAGE_TAG=" not in log
    assert log.index("docker run") < log.index("maturin build")
    assert "--features model-protection-tpm2" in log
    assert "--build-arg VLLM_PROTECTED_VERSION=0.30.0" in log


@pytest.mark.timeout(30)
def test_builder_rejects_wrong_base_before_building_wheel(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    stubs = {
        "python3": "exit 0\n",
        "docker": 'printf "docker %s\\n" "$*" >> "$BUILD_CALLS"\n'
        'if [ "$1" = "run" ]; then echo "Base image vLLM version mismatch" >&2; exit 1; fi\n',
        "maturin": 'printf "maturin %s\\n" "$*" >> "$BUILD_CALLS"\n',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\n" + body)
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(DEPLOY_DIR / "build-protected-image.sh")],
        env=dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            BUILD_CALLS=str(calls),
            SKIP_BASE_BUILD="true",
            VLLM_PROTECTED_VERSION="0.30.0",
            TMPDIR=str(tmp_path),
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "Base image vLLM version mismatch" in result.stderr
    assert "maturin" not in calls.read_text()
