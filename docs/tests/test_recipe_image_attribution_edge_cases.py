# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date

import pytest
from recipe_catalog_test_utils import load_catalog_validator

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

catalog_validate = load_catalog_validator("recipe_catalog_validate_edge_cases")


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
