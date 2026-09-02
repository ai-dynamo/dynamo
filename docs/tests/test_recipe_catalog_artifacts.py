# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from recipe_catalog_test_utils import CATALOG, VALIDATOR_PATH, load_catalog_validator

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

catalog_validate = load_catalog_validator("recipe_catalog_validate")


@pytest.mark.parametrize(
    ("recipe_id", "expected_images"),
    (
        (
            "deepseek-v4-pro",
            ("nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",),
        ),
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


@pytest.mark.parametrize(
    ("recipe_id", "expected_periods"),
    (
        (
            "deepseek-v4-pro",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
                    "source_revision": "154e76c85233c027565ed3aca220669be1577b6f",
                    "source_kind": "github-release",
                    "release_tag": "v1.3.0-deepseek-v4-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "glm-5-2",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0-glm-5.2-dev.1",
                    "source_revision": "9ab57d7ecefdd2a2af2e2a2c889724a157457cd6",
                    "source_kind": "github-release",
                    "release_tag": "v1.3.0-glm-5.2-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "inkling",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-inkling-dev.1",
                    "source_revision": "73aa2073d12c8cbd5c955f618aed251be920b8cd",
                    "source_kind": "github-release",
                    "release_tag": "v1.4.0-inkling-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "kimi-k2-6",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-kimi-k2.6-dev.1",
                    "source_revision": "66a95d5aa3b5d238fd94b6f8243c4e479cb7242c",
                    "source_kind": "github-release",
                    "release_tag": "v1.3.0-kimi-k2.6-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "kimi-k3",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-kimi-k3-dev.1",
                    "source_revision": "92ec0146e4221c7c9e5013e3bd51db6113f96935",
                    "source_kind": "github-release",
                    "release_tag": "v1.4.0-kimi-k3-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "nemotron-3-5-lightning",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.5.0-nemotron-3.5-lightning-dev.1",
                    "source_revision": "70d2ef431b82ecd42e73c001149ea64bd1816010",
                    "source_kind": "github-release",
                    "release_tag": "v1.5.0-nemotron-3.5-lightning-dev.1",
                    "release_state": "prerelease",
                },
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-nemotron-3.5-lightning-dev.1",
                    "source_revision": "70d2ef431b82ecd42e73c001149ea64bd1816010",
                    "source_kind": "github-release",
                    "release_tag": "v1.5.0-nemotron-3.5-lightning-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "nemotron-3-super",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-super-dev.1",
                    "source_revision": "3424174cb83301418af5a000e9d09f2a4f93261c",
                    "source_kind": "github-release",
                    "release_tag": "v1.3.0-nemotron-super-dev.1",
                    "release_state": "prerelease",
                },
            ),
        ),
        (
            "nemotron-3-ultra",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1",
                    "source_revision": "002cf02ce2b0c3679996526edec89becebe5dd06",
                    "source_kind": "github-release",
                    "release_tag": "v1.3.0-nemotron-ultra-dev.1",
                    "release_state": "draft",
                },
            ),
        ),
        (
            "qwen-3-8-2-4t-a95b-fp8",
            (
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-qwen-3.8-2.4t-dev.1",
                    "source_revision": "c8a33bf20d5478c3fa8fbdb5385d1663af5b496c",
                },
                {
                    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-qwen-3.8-2.4t-dev.1",
                    "source_revision": "c8a33bf20d5478c3fa8fbdb5385d1663af5b496c",
                },
            ),
        ),
    ),
)
def test_recipe_specific_images_publish_effective_periods(
    recipe_id: str,
    expected_periods: tuple[dict[str, str], ...],
) -> None:
    document = yaml.safe_load((CATALOG / "recipes" / f"{recipe_id}.yaml").read_text())

    assert (
        tuple(document["artifacts"]["recipe_specific_image_periods"])
        == expected_periods
    )


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


def _period_artifacts(
    image: str,
    revision: str,
    start: object = "",
    end: str = "",
    current: bool = False,
) -> dict[str, object]:
    period: dict[str, object] = {"image": image, "source_revision": revision}
    if start:
        period["effective_from"] = start
    if end:
        period["effective_to"] = end
    artifacts: dict[str, object] = {"recipe_specific_image_periods": [period]}
    if current:
        artifacts["recipe_specific_images"] = [image]
    return artifacts


def test_recipe_image_validation_allows_effective_dated_ownership_handoff() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-handoff-dev.1"
    entries = {
        "recipe-a": {
            "artifacts": _period_artifacts(image, "a" * 40, "2026-01-01", "2026-01-31")
        },
        "recipe-b": {
            "artifacts": _period_artifacts(image, "b" * 40, "2026-02-01", current=True)
        },
    }
    catalog_validate.ERRORS.clear()

    catalog_validate.check_recipe_specific_image_ownership(entries)

    assert catalog_validate.ERRORS == []


def test_recipe_image_validation_rejects_overlapping_effective_periods() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-overlap-dev.1"
    entries = {
        "recipe-a": {
            "artifacts": _period_artifacts(image, "a" * 40, "2026-01-01", "2026-02-01")
        },
        "recipe-b": {
            "artifacts": _period_artifacts(image, "b" * 40, "2026-02-01", current=True)
        },
    }
    catalog_validate.ERRORS.clear()

    catalog_validate.check_recipe_specific_image_ownership(entries)

    assert any(
        "overlapping ownership periods" in error for error in catalog_validate.ERRORS
    )


def test_recipe_image_validation_rejects_same_start_overlapping_periods() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-overlap-dev.1"
    entries = {
        "recipe-a": {"artifacts": _period_artifacts(image, "a" * 40, "2026-01-01")},
        "recipe-b": {
            "artifacts": _period_artifacts(image, "b" * 40, "2026-01-01", "2026-02-01")
        },
    }
    catalog_validate.ERRORS.clear()

    catalog_validate.check_recipe_specific_image_ownership(entries)

    assert any(
        "overlapping ownership periods" in error for error in catalog_validate.ERRORS
    )


def test_recipe_image_validation_tracks_unquoted_start_for_overlap() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-overlap-dev.1"
    unquoted_start = yaml.safe_load("effective_from: 2026-01-01")["effective_from"]
    entries = {
        "recipe-a": {"artifacts": _period_artifacts(image, "a" * 40, unquoted_start)},
        "recipe-b": {"artifacts": _period_artifacts(image, "b" * 40, "2026-01-01")},
    }

    errors = catalog_validate._image_attribution.recipe_image_ownership_errors(entries)

    assert any("overlapping ownership periods" in error for error in errors)


def test_recipe_image_validation_rejects_empty_ownership_periods() -> None:
    errors = catalog_validate._image_attribution.recipe_image_errors(
        {"recipe_specific_image_periods": []}, [], "empty-periods"
    )

    assert any("must contain at least one item" in error for error in errors)


def test_recipe_image_validation_rejects_invalid_calendar_date() -> None:
    artifacts = {
        "recipe_specific_images": [
            "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-invalid-date-dev.1"
        ],
        "recipe_specific_image_periods": [
            {
                "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-invalid-date-dev.1",
                "effective_from": "2026-99-99",
                "source_revision": "a" * 40,
            }
        ],
    }

    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [], "invalid-date"
    )

    assert any("invalid effective_from" in error for error in errors)


def test_recipe_image_validation_allows_retroactive_open_start(
    tmp_path: Path,
) -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-retroactive-dev.1"
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(f"image: {image}\n")
    artifacts = {
        "recipe_specific_images": [image],
        "recipe_specific_image_periods": [
            {
                "image": image,
                "source_revision": "a" * 40,
            }
        ],
    }

    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [deploy], "retroactive"
    )

    assert errors == []


def test_recipe_image_validation_rejects_closed_period_for_deployed_image(
    tmp_path: Path,
) -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-still-deployed-dev.1"
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(f"image: {image}\n")
    artifacts = _period_artifacts(image, "a" * 40, "2026-01-01", "2026-02-01")

    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [deploy], "still-deployed"
    )

    assert any(
        "deployed recipe-specific image must have an open ownership period" in error
        for error in errors
    )


def test_recipe_image_validation_allows_github_release_provenance_without_current_deploy_asset() -> (
    None
):
    image = "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1"
    artifacts = {
        "recipe_specific_images": [image],
        "recipe_specific_image_periods": [
            {
                "image": image,
                "source_revision": "1" * 40,
                "source_kind": "github-release",
                "release_tag": "v1.3.0-deepseek-v4-dev.1",
                "release_state": "prerelease",
            }
        ],
    }

    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [], "release-provenance"
    )

    assert errors == []


def test_recipe_image_validation_requires_complete_github_release_provenance() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-model-dev.2"
    artifacts = {
        "recipe_specific_images": [image],
        "recipe_specific_image_periods": [
            {
                "image": image,
                "source_revision": "2" * 40,
                "source_kind": "github-release",
            }
        ],
    }

    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [], "missing-release-metadata"
    )

    assert any("missing release_tag" in error for error in errors)
    assert any("invalid release_state" in error for error in errors)


def test_recipe_image_validation_allows_multiple_release_generations() -> None:
    images = [
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0-nemotron-ultra-dev.1",
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-nemotron-ultra-dev.2",
    ]
    artifacts = {
        "recipe_specific_images": images,
        "recipe_specific_image_periods": [
            {
                "image": image,
                "source_revision": str(index) * 40,
                "source_kind": "github-release",
                "release_tag": f"v1.{index + 2}.0-nemotron-ultra-dev.{index}",
                "release_state": "prerelease",
            }
            for index, image in enumerate(images, start=1)
        ],
    }

    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [], "release-series"
    )

    assert errors == []


def test_recipe_catalog_validator_runs_in_pre_merge(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
