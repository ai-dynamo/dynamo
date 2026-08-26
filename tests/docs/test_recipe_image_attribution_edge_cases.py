# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from datetime import date
from pathlib import Path

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "docs/fern/pages/recipes/_catalog/validate.py"

if not VALIDATOR_PATH.is_file():
    pytest.skip(
        "recipe catalog sources are not present in this runtime image",
        allow_module_level=True,
    )

_SPEC = importlib.util.spec_from_file_location(
    "recipe_catalog_validate_edge_cases", VALIDATOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
catalog_validate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(catalog_validate)


def test_recipe_image_validation_rejects_unquoted_end_date_without_crashing() -> None:
    image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-overlap-dev.1"
    periods = [
        ("2026-01-01", date(2026, 2, 1), "recipe-a"),
        ("2026-01-01", None, "recipe-b"),
    ]

    errors = catalog_validate._image_attribution._overlap_errors(image, periods)

    assert any("overlapping ownership periods" in error for error in errors)


@pytest.mark.parametrize(
    ("artifacts", "expected_error"),
    (
        (
            {
                "recipe_specific_images": [{}],
                "recipe_specific_image_periods": [],
            },
            "recipe_specific_images entries must be strings",
        ),
        (
            {
                "recipe_specific_images": [],
                "recipe_specific_image_periods": [
                    {
                        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-dev.1",
                        "source_revision": "a" * 40,
                        "source_kind": [],
                    }
                ],
            },
            "invalid source_kind",
        ),
        (
            {
                "recipe_specific_images": [],
                "recipe_specific_image_periods": [
                    {
                        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0-dev.1",
                        "source_revision": "a" * 40,
                        "source_kind": "github-release",
                        "release_tag": "v1.4.0-dev.1",
                        "release_state": [],
                    }
                ],
            },
            "invalid release_state",
        ),
    ),
)
def test_recipe_image_validation_rejects_unhashable_values_without_crashing(
    artifacts: dict[str, object],
    expected_error: str,
) -> None:
    errors = catalog_validate._image_attribution.recipe_image_errors(
        artifacts, [], "unhashable-value"
    )

    assert any(expected_error in error for error in errors)
