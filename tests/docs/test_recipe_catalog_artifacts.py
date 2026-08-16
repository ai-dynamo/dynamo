# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "docs/fern/pages/recipes/_catalog"
VALIDATOR_PATH = CATALOG / "validate.py"

if not VALIDATOR_PATH.is_file():
    pytest.skip(
        "recipe catalog sources are not present in this runtime image",
        allow_module_level=True,
    )

_SPEC = importlib.util.spec_from_file_location(
    "recipe_catalog_validate", VALIDATOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
catalog_validate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(catalog_validate)


@pytest.mark.parametrize(
    ("recipe_id", "expected_images"),
    (
        (
            "glm-5-2",
            ("nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-glm-5.2-dev.1",),
        ),
        (
            "inkling",
            ("nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-inkling-dev.1",),
        ),
        (
            "kimi-k2-6",
            ("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-kimi-k2.6-dev.1",),
        ),
        (
            "kimi-k3",
            ("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-kimi-k3-dev.1",),
        ),
        (
            "nemotron-3-5-lightning",
            (
                "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.5.0-nemotron-3.5-lightning-dev.1",
                "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-nemotron-3.5-lightning-dev.1",
            ),
        ),
        (
            "nemotron-3-super",
            ("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-super-dev.1",),
        ),
        (
            "nemotron-3-ultra",
            ("nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1",),
        ),
        (
            "qwen-3-8-2-4t-a95b-fp8",
            (
                "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-qwen-3.8-2.4t-dev.1",
                "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-qwen-3.8-2.4t-dev.1",
            ),
        ),
    ),
)
def test_recipe_specific_images_are_catalog_owned(
    recipe_id: str,
    expected_images: tuple[str, ...],
) -> None:
    document = yaml.safe_load((CATALOG / "recipes" / f"{recipe_id}.yaml").read_text())
    assert tuple(document["artifacts"]["recipe_specific_images"]) == expected_images


def test_recipe_catalog_schema_has_spdx_metadata() -> None:
    schema = json.loads((CATALOG / "schema.json").read_text())

    assert schema["$comment"] == (
        "SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & "
        "AFFILIATES. All rights reserved.\n"
        "SPDX-License-Identifier: Apache-2.0"
    )


@pytest.mark.parametrize(
    "image",
    (
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0/invalid",
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0:invalid",
    ),
)
def test_recipe_catalog_schema_rejects_malformed_image_tags(image: str) -> None:
    schema = json.loads((CATALOG / "schema.json").read_text())
    pattern = schema["properties"]["artifacts"]["properties"]["recipe_specific_images"][
        "items"
    ]["pattern"]

    assert re.fullmatch(pattern, image) is None


@pytest.mark.parametrize(
    ("artifacts", "expected_error"),
    (
        ("not-an-object", "artifacts must be an object"),
        (
            {"recipe_specific_images": ""},
            "artifacts.recipe_specific_images must be an array",
        ),
        (
            {"recipe_specific_images": "not-an-array"},
            "artifacts.recipe_specific_images must be an array",
        ),
    ),
)
def test_recipe_image_validation_reports_malformed_artifacts(
    artifacts: object,
    expected_error: str,
) -> None:
    catalog_validate.ERRORS.clear()

    catalog_validate.check_recipe_specific_images(
        {"artifacts": artifacts}, [], "recipe:malformed"
    )

    assert any(expected_error in error for error in catalog_validate.ERRORS)


def test_recipe_image_validation_ignores_comment_only_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-recipe-dev.1"
    asset = tmp_path / "deploy.yaml"
    asset.write_text(
        "# image: " + declared + "\n"
        "containers:\n"
        "- image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0\n"
    )
    monkeypatch.setattr(catalog_validate, "REPO_ROOT", str(tmp_path))
    catalog_validate.ERRORS.clear()

    catalog_validate.check_recipe_specific_images(
        {"artifacts": {"recipe_specific_images": [declared]}},
        [asset.name],
        "recipe:comment-only",
    )

    assert any(
        "is not referenced by a deploy asset" in error
        for error in catalog_validate.ERRORS
    )


def test_recipe_image_validation_rejects_duplicate_ownership() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-shared-dev.1"
    entries = {
        "recipe-a": {"artifacts": {"recipe_specific_images": [image]}},
        "recipe-b": {"artifacts": {"recipe_specific_images": [image]}},
    }
    catalog_validate.ERRORS.clear()

    catalog_validate.check_recipe_specific_image_ownership(entries)

    assert any(
        "declared by multiple recipes" in error for error in catalog_validate.ERRORS
    )


def test_recipe_catalog_validator_runs_in_pre_merge(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
