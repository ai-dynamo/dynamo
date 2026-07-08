# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

CONTAINER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTAINER_DIR.parent
INSTALLER = CONTAINER_DIR / "deps" / "vllm" / "install_nightly_overlay.sh"
sys.path.insert(0, str(CONTAINER_DIR / "deps" / "vllm"))

from validate_nightly_overlay import (  # noqa: E402
    ALLOWED_TUPLES,
    verify_source_provenance,
)

CURRENT_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "184914ac7c32e4aa7789bb686bfaa0817dd56dbdc8ee05fc0ec671aa0b1792f0"
)
CURRENT_HEAD = "17355f6f668857d9b85e0e7714529b42757e0730"
CURRENT_REF = "schwinns/exp-cuda-zero-page-234"
CROSSOVER_IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a"
)
CROSSOVER_HEAD = "7e48076f13710677c223daf6e4e1af039c0f016e"
CROSSOVER_REF = "schwinns/exp-93d8-current-overlay-zero-regression-20260708t082747z"
VLLM_URL = "https://github.com/galletas1712/vllm.git"
FLASHINFER_URL = "https://github.com/galletas1712/flashinfer.git"
FLASHINFER_REF = "schwinns/checkpoint-collectives-integration"
FLASHINFER_SHA = "330cc8e1a09f59c1241084459f3df3204b9b8327"


def select_tuple(
    image: str,
    head: str,
    *,
    vllm_url: str = VLLM_URL,
    vllm_ref: str | None = None,
    flashinfer_url: str = FLASHINFER_URL,
    flashinfer_ref: str = FLASHINFER_REF,
) -> subprocess.CompletedProcess[str]:
    if vllm_ref is None:
        vllm_ref = CROSSOVER_REF if head == CROSSOVER_HEAD else CURRENT_REF
    env = os.environ.copy()
    env.update(
        VLLM_RUNTIME_BASE_IMAGE=image,
        VLLM_GIT_URL=vllm_url,
        VLLM_GIT_REF=vllm_ref,
        VLLM_GIT_SHA=head,
        FLASHINFER_GIT_URL=flashinfer_url,
        FLASHINFER_GIT_REF=flashinfer_ref,
        FLASHINFER_GIT_SHA=FLASHINFER_SHA,
    )
    return subprocess.run(
        [str(INSTALLER), "select"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def parse_output(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_selects_current_tuple() -> None:
    selected = parse_output(select_tuple(CURRENT_IMAGE, CURRENT_HEAD))

    assert selected == {
        "vllm_overlay_tuple": "current-697158",
        "vllm_base_index_digest": (
            "sha256:" "184914ac7c32e4aa7789bb686bfaa0817dd56dbdc8ee05fc0ec671aa0b1792f0"
        ),
        "vllm_amd64_digest": (
            "sha256:" "1fd4323d0aafe8d92b4a4b568ad33661ecaf3bfc7f40860c95d09fed4e6ccd58"
        ),
        "vllm_base_commit": "69715823df89b11ee684b84066390cbb9092d5c1",
        "vllm_head": CURRENT_HEAD,
        "vllm_overlay_files": "13",
        "vllm_baseline_sbom": "vllm-openai@184914ac",
    }


def test_selects_old_base_crossover_tuple() -> None:
    selected = parse_output(select_tuple(CROSSOVER_IMAGE, CROSSOVER_HEAD))

    assert selected == {
        "vllm_overlay_tuple": "crossover-93d8",
        "vllm_base_index_digest": (
            "sha256:" "7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9"
        ),
        "vllm_amd64_digest": (
            "sha256:" "5da1eb79b49d3edb3b3601a116273f019adb7cab403e86790f61130f8596810a"
        ),
        "vllm_base_commit": "93d8f834dd8acf33eb0e2a75b2711b628cb6e226",
        "vllm_head": CROSSOVER_HEAD,
        "vllm_overlay_files": "13",
        "vllm_baseline_sbom": "vllm-openai@7c5a10e9",
    }


@pytest.mark.parametrize(
    ("image", "head"),
    [
        (CURRENT_IMAGE, CROSSOVER_HEAD),
        (CROSSOVER_IMAGE, CURRENT_HEAD),
        (CURRENT_IMAGE, ""),
        ("", CURRENT_HEAD),
        ("vllm/vllm-openai:latest", CURRENT_HEAD),
        ("vllm/vllm-openai@sha256:" + "0" * 64, CURRENT_HEAD),
    ],
)
def test_rejects_mixed_and_unknown_tuples(image: str, head: str) -> None:
    result = select_tuple(image, head)

    assert result.returncode != 0
    assert "unknown nightly overlay tuple" in result.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"vllm_url": "https://github.com/vllm-project/vllm.git"},
        {"vllm_ref": "latest"},
        {"flashinfer_url": "https://github.com/flashinfer-ai/flashinfer.git"},
        {"flashinfer_ref": "main"},
    ],
)
def test_rejects_arbitrary_source_refs(overrides: dict[str, str]) -> None:
    result = select_tuple(CURRENT_IMAGE, CURRENT_HEAD, **overrides)

    assert result.returncode != 0
    assert "expected" in result.stderr


@pytest.mark.parametrize("tuple_name", sorted(ALLOWED_TUPLES))
def test_validator_accepts_complete_allowed_provenance(tuple_name: str) -> None:
    source = {
        "install_mode": "python-overlay",
        "vllm_overlay_tuple": tuple_name,
        **ALLOWED_TUPLES[tuple_name],
    }

    verify_source_provenance(source)


def test_validator_rejects_mixed_and_unknown_provenance() -> None:
    mixed = {
        "install_mode": "python-overlay",
        "vllm_overlay_tuple": "crossover-93d8",
        **ALLOWED_TUPLES["crossover-93d8"],
        "vllm_source_sha": CURRENT_HEAD,
    }
    with pytest.raises(RuntimeError, match="Unexpected vLLM overlay provenance"):
        verify_source_provenance(mixed)

    with pytest.raises(RuntimeError, match="Unknown vLLM overlay tuple"):
        verify_source_provenance(
            {
                "install_mode": "python-overlay",
                "vllm_overlay_tuple": "unknown",
            }
        )

    partial = {
        "install_mode": "python-overlay",
        "vllm_overlay_tuple": "current-697158",
    }
    with pytest.raises(RuntimeError, match="Unexpected vLLM overlay provenance"):
        verify_source_provenance(partial)


def test_current_tuple_remains_build_default() -> None:
    template = (CONTAINER_DIR / "templates/vllm_runtime.Dockerfile").read_text()
    workflow = (REPO_ROOT / ".github/workflows/build-on-demand.yml").read_text()

    assert (
        "ARG VLLM_EXPECTED_BASE_COMMIT=" "69715823df89b11ee684b84066390cbb9092d5c1"
    ) in template
    assert "ARG VLLM_EXPECTED_BASELINE_SBOM=vllm-openai@184914ac" in template
    assert (
        "BASELINE_SBOM_FILE=${{ inputs.vllm_install_mode == 'python-overlay' "
        "&& needs.init.outputs.vllm_baseline_sbom || 'cuda@2ab6381d' }}"
    ) in workflow
