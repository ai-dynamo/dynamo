# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/build-on-demand.yml"
BASE_IMAGE = "registry.example.com/ai-dynamo/dynamo@sha256:" + "a" * 64
REQUIRED_BASE_IMAGE = (
    "dynamoci.azurecr.io/ai-dynamo/dynamo:"
    "cuda-zp234-28894200421-runtime"
    "@sha256:694ae15aa5759af86c3c520a4b15ef86e9453a9388a2ab07a029d4e16e133e53"
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _validation_script() -> str:
    steps = _workflow()["jobs"]["snapshot-placeholder-vllm-from-base"]["steps"]
    return next(
        step["run"] for step in steps if step.get("id") == "validate-build-inputs"
    )


def _run_validator(
    tmp_path: Path,
    *,
    base_image: str = BASE_IMAGE,
    criu_repo: str = "https://github.com/galletas1712/criu.git",
    criu_ref: str = "criu-dev",
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    output = tmp_path / "github-output"
    env = {
        **os.environ,
        "BASE_IMAGE_INPUT": base_image,
        "CRIU_REPO_INPUT": criu_repo,
        "CRIU_REF_INPUT": criu_ref,
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            _validation_script(),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    outputs = {}
    if output.exists():
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return result, outputs


@pytest.mark.parametrize(
    "base_image",
    [
        REQUIRED_BASE_IMAGE,
        "repository@sha256:" + "b" * 64,
        "registry.example.com:5000/team/nested/image:release-1.2@sha256:" + "c" * 64,
    ],
)
def test_direct_placeholder_validator_accepts_digest_pinned_base_images(
    tmp_path: Path, base_image: str
):
    result, outputs = _run_validator(tmp_path, base_image=base_image)

    assert result.returncode == 0, result.stderr
    assert outputs["base_image"] == base_image


@pytest.mark.parametrize(
    "base_image",
    [
        "",
        "registry.example.com/ai-dynamo/dynamo:latest",
        "registry.example.com/ai-dynamo/dynamo@md5:" + "a" * 64,
        "registry.example.com/ai-dynamo/dynamo@sha256:not-hex",
        "registry.example.com/ai-dynamo/dynamo@sha256:" + "a" * 63,
        "registry.example.com/ai-dynamo/dynamo@sha256:" + "a" * 65,
        "registry.example.com/ai dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai-dynamo/dynamo@sha256:" + "a" * 64 + "\nevil",
        "registry.example.com/'ai-dynamo'/dynamo@sha256:" + "a" * 64,
        'registry.example.com/"ai-dynamo"/dynamo@sha256:' + "a" * 64,
        "registry.example.com/`id`/dynamo@sha256:" + "a" * 64,
        "registry.example.com/$(id)/dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai-dynamo/dynamo;id@sha256:" + "a" * 64,
        "registry.example.com/ai\\dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai=dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai|dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai>dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai<dynamo@sha256:" + "a" * 64,
        "registry.example.com/ai&dynamo@sha256:" + "a" * 64,
    ],
)
def test_direct_placeholder_validator_rejects_unsafe_base_images(
    tmp_path: Path, base_image: str
):
    result, outputs = _run_validator(tmp_path, base_image=base_image)

    assert result.returncode != 0
    assert outputs == {}


@pytest.mark.parametrize(
    ("criu_repo", "criu_ref"),
    [
        ("https://github.com/galletas1712/criu.git", "criu-dev"),
        ("https://git.example.com/org/criu.git", "release/v1.2_3+fix"),
        ("ssh://git@git.example.com:2222/org/criu.git", "a" * 40),
        ("git@git.example.com:org/criu.git", "refs/tags/v1.2.3-rc+1"),
    ],
)
def test_direct_placeholder_validator_accepts_safe_git_inputs(
    tmp_path: Path, criu_repo: str, criu_ref: str
):
    result, outputs = _run_validator(tmp_path, criu_repo=criu_repo, criu_ref=criu_ref)

    assert result.returncode == 0, result.stderr
    assert outputs == {
        "base_image": BASE_IMAGE,
        "criu_repo": criu_repo,
        "criu_ref": criu_ref,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "criu_repo",
            "https://github.com/galletas1712/criu.git\nBASE_IMAGE=evil:latest",
        ),
        ("criu_repo", "https://github.com/$(id)/criu.git"),
        ("criu_repo", "https://github.com/'org'/criu.git"),
        ("criu_repo", 'https://github.com/"org"/criu.git'),
        ("criu_ref", "`id`"),
        ("criu_ref", "criu-dev;id"),
        ("criu_ref", "BASE_IMAGE=evil"),
        ("criu_ref", "release candidate"),
        ("criu_ref", "release\\candidate"),
        ("criu_ref", "criu-dev&&id"),
        ("criu_ref", "criu-dev|id"),
        ("criu_ref", ""),
    ],
)
def test_direct_placeholder_validator_rejects_shell_sensitive_git_inputs(
    tmp_path: Path, field: str, value: str
):
    result, outputs = _run_validator(tmp_path, **{field: value})

    assert result.returncode != 0
    assert outputs == {}


def test_direct_placeholder_uses_validated_outputs_and_blocks_cleanup():
    jobs = _workflow()["jobs"]
    direct_job = jobs["snapshot-placeholder-vllm-from-base"]
    cleanup_job = jobs["clean-k8s-builder"]
    build_step = next(
        step
        for step in direct_job["steps"]
        if step.get("name") == "Build and push vLLM snapshot placeholder from base"
    )

    assert "snapshot-placeholder-vllm-from-base" in cleanup_job["needs"]
    assert cleanup_job["if"] == "always()"
    assert build_step["with"]["extra_build_args"] == (
        "BASE_IMAGE=${{ steps.validate-build-inputs.outputs.base_image }}\n"
        "CRIU_REPO=${{ steps.validate-build-inputs.outputs.criu_repo }}\n"
        "CRIU_REF=${{ steps.validate-build-inputs.outputs.criu_ref }}\n"
    )
