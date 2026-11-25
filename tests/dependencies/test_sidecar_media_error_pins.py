# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require a media-error contract review when engine or protocol pins change.

These are upgrade tripwires, not live engine-wording tests. The checkout hook
runs them because component images omit container/context.yaml.
"""

import sys
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:
    import tomli as tomllib

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.parallel,
]

ROOT = Path(__file__).resolve().parents[2]

# If a guard fails, inspect the upgraded engine's media-fetch error wording.
# Fix is_media_fetch_error in lib/sidecar/common/src/media_error.rs and its
# fixtures as needed BEFORE updating these reviewed pins; do not just bump them.
# SGLang upgrades must also retain requests exception/cause ancestry: rerun
# test_sglang_deployment's mm_pd_qwen3-vl-2b requests with unreachable image URLs
# and invalid image data, in both unary and streaming modes.
RECHECK = (
    "Engine/protocol pin changed: recheck media-fetch error wording and update "
    "lib/sidecar/common/src/media_error.rs matching and fixtures as needed "
    "before updating the reviewed pins in this test. For SGLang upgrades, also "
    "recheck the Python media loader's requests exception/cause ancestry and "
    "rerun tests/serve/test_sglang.py::test_sglang_deployment for "
    "mm_pd_qwen3-vl-2b, including unary/SSE requests with unreachable image URLs "
    "and invalid image data."
)


def _load(relative: str) -> dict:
    """Read real declarations; image-only pytest skips are covered by the hook."""
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is absent from this component image")
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) if path.suffix == ".yaml" else tomllib.loads(text)


def _reviewed(actual: object, expected: object) -> None:
    """Keep the standalone pre-commit check effective under python -O too."""
    if actual != expected:
        raise AssertionError(f"{RECHECK}\nExpected: {expected!r}\nActual: {actual!r}")


def _python_pin(extra: str, package: str) -> str:
    """Parse requirements so stale comments or changed extras cannot hide a bump."""
    requirements = _load("pyproject.toml")["project"]["optional-dependencies"][extra]
    matches = [Requirement(value) for value in requirements]
    matches = [value for value in matches if value.name == package]
    _reviewed(len(matches), 1)
    return str(matches[0].specifier)


def test_vllm_media_error_version() -> None:
    _reviewed(_python_pin("vllm", "vllm"), "==0.33.0")


def test_sglang_media_error_version() -> None:
    _reviewed(_python_pin("sglang", "sglang"), "==0.5.23")


def test_trtllm_media_error_version() -> None:
    _reviewed(_python_pin("trtllm", "tensorrt-llm"), "==1.4.0rc3")


def test_stock_runtime_media_error_versions() -> None:
    context = _load("container/context.yaml")
    # Stock CUDA/XPU/CPU images; enterprise/custom overrides need their own review.
    actual = {
        (engine, device, config["runtime_image"], config["runtime_image_tag"])
        for engine in ("vllm", "sglang", "trtllm")
        for device, config in context[engine].items()
        if isinstance(config, dict) and "runtime_image" in config
    }
    _reviewed(
        actual,
        {
            ("vllm", "cuda13.0", "vllm/vllm-openai", "v0.33.0"),
            ("vllm", "xpu", "vllm/vllm-openai-xpu", "v0.33.0"),
            ("vllm", "cpu", "vllm/vllm-openai-cpu", "v0.33.0"),
            ("sglang", "cuda13.0", "lmsysorg/sglang", "v0.5.23-cu130-runtime"),
            ("sglang", "xpu", "lmsysorg/sglang", "v0.5.23-xpu"),
            ("trtllm", "cuda13.4", "nvcr.io/nvidia/tensorrt-llm/release", "1.4.0rc3"),
        },
    )


def test_native_protocol_media_error_versions() -> None:
    dependencies = _load("Cargo.toml")["workspace"]["dependencies"]
    _reviewed(
        {name: dependencies[name] for name in ("vllm-proto", "sglang-api-types")},
        {
            "vllm-proto": {"version": "=0.6.0"},
            "sglang-api-types": {"version": "=0.1.1"},
        },
    )


def main() -> int:
    """Run all five checks in a checkout, where missing inputs must fail."""
    for relative in ("pyproject.toml", "container/context.yaml", "Cargo.toml"):
        if not (ROOT / relative).is_file():
            print(f"ERROR: required guard input {relative} is missing")
            return 1
    for check in (
        test_vllm_media_error_version,
        test_sglang_media_error_version,
        test_trtllm_media_error_version,
        test_stock_runtime_media_error_versions,
        test_native_protocol_media_error_versions,
    ):
        check()
    print("All five sidecar media-error upgrade guards passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
