# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from recipe_catalog_test_utils import CATALOG, load_catalog_validator

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

catalog_validate = load_catalog_validator("recipe_catalog_validate_badges")

BENCH = "recipes/model/perf.yaml"
CERT = {
    "source": "nim-factory",
    "ref": "https://example.com/certification/1",
    "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
    "date": "2026-10-20",
}

# Recipes in the catalog when badges became required. badge_backfill.yaml may
# only shrink from this set; a new recipe ships with its badges instead.
BACKFILL_BASELINE = {
    "ax-k2",
    "deepseek-v4-1-flash",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "gemma4-31b",
    "glm-5-2",
    "glm-5-3-flash",
    "glm-5-nvfp4",
    "gpt-oss-120b",
    "inkling",
    "k-exaone-2",
    "kimi-k2-5",
    "kimi-k2-6",
    "kimi-k3",
    "motif-3",
    "nemotron-3-5-lightning",
    "nemotron-3-super",
    "nemotron-3-ultra",
    "qwen-3-8-2-4t-a95b-fp8",
    "qwen-3-8-flash-next",
    "qwen3-235b-a22b-fp8",
    "qwen3-32b",
    "qwen3-32b-fp8",
    "qwen3-5-122b",
    "qwen3-6-35b",
    "qwen3-6-35b-a3b",
    "qwen3-vl-30b",
}


@pytest.fixture(autouse=True)
def repo_with_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / BENCH).parent.mkdir(parents=True)
    (tmp_path / BENCH).write_text("kind: Job\n")
    monkeypatch.setattr(catalog_validate, "REPO_ROOT", str(tmp_path))


def target(badge: str | None = None, **overrides: object) -> dict[str, object]:
    t: dict[str, object] = {
        "id": "vllm-agg-h200",
        "benchmark": {"tool": "aiperf", "asset": BENCH},
        "expected_performance": {"available": True, "summary": "1 tok/s"},
    }
    if badge is not None:
        t["badge"] = badge
    t.update(overrides)
    return t


def recipe(*targets: dict[str, object], maintainer: str | None = None) -> dict:
    return {"id": "model", "maintainer": maintainer, "targets": list(targets)}


@pytest.mark.parametrize(
    "obj",
    (
        recipe(target("nvidia-validated", benchmark=None)),
        recipe(target("nvidia-optimized")),
        recipe(target("nvidia-certified", certification=CERT)),
        recipe(target("community"), target("community", id="b"), maintainer="Jane Doe"),
    ),
    ids=["validated", "optimized", "certified", "community"],
)
def test_badge_with_its_evidence_passes(obj: dict) -> None:
    assert catalog_validate.recipe_badge_errors(obj, "recipe:model") == []


@pytest.mark.parametrize(
    ("obj", "expected_error"),
    (
        (recipe(target()), "needs a badge"),
        (recipe(target("day-0")), "badge must be one of"),
        (
            recipe(target("nvidia-optimized", benchmark=None)),
            "nvidia-optimized needs a benchmark asset",
        ),
        (
            recipe(
                target("nvidia-optimized", benchmark={"asset": "recipes/missing.yaml"})
            ),
            "nvidia-optimized needs a benchmark asset",
        ),
        (
            recipe(
                target("nvidia-optimized", expected_performance={"available": False})
            ),
            "needs expected_performance.available: true",
        ),
        (
            recipe(target("nvidia-certified")),
            "nvidia-certified needs a certification record",
        ),
        (
            recipe(target("nvidia-certified", certification={**CERT, "ref": ""})),
            "nvidia-certified needs a certification record",
        ),
        (
            recipe(target("nvidia-certified", certification=CERT, benchmark=None)),
            "nvidia-certified needs a benchmark asset",
        ),
        (
            recipe(target("nvidia-optimized", certification=CERT)),
            "has a certification record but badge is nvidia-optimized",
        ),
        (recipe(target("community")), "community badge needs a named maintainer"),
        (
            recipe(
                target("community"),
                target("nvidia-validated", id="b"),
                maintainer="Jane Doe",
            ),
            "community badge must apply to every target or none",
        ),
    ),
    ids=[
        "missing-badge",
        "unknown-badge",
        "optimized-no-benchmark",
        "optimized-missing-benchmark-file",
        "optimized-no-perf",
        "certified-no-record",
        "certified-empty-ref",
        "certified-no-benchmark",
        "certification-on-optimized",
        "community-no-maintainer",
        "community-partial",
    ],
)
def test_badge_without_its_evidence_fails(obj: dict, expected_error: str) -> None:
    errors = catalog_validate.recipe_badge_errors(obj, "recipe:model")

    assert any(expected_error in error for error in errors), errors


def test_pending_recipe_may_omit_badges_but_not_evidence() -> None:
    obj = recipe(target(), target("nvidia-optimized", id="b", benchmark=None))

    errors = catalog_validate.recipe_badge_errors(obj, "recipe:model", pending=True)

    assert errors == [
        "[recipe:model] target b nvidia-optimized needs a benchmark asset"
    ]


@pytest.mark.parametrize(
    ("pending", "expected_error"),
    (
        (["unknown"], "unknown recipe id: unknown"),
        (["done"], "done has a badge on every target; remove it from the list"),
    ),
    ids=["unknown-id", "fully-badged"],
)
def test_badge_backfill_list_rejects_stale_ids(
    pending: list[str], expected_error: str
) -> None:
    entries = {
        "done": recipe(target("nvidia-validated")),
        "open": recipe(target()),
    }

    errors = catalog_validate.badge_backfill_errors(pending, entries)

    assert any(expected_error in error for error in errors), errors


def test_badge_backfill_list_only_shrinks() -> None:
    backfill = yaml.safe_load((CATALOG / "badge_backfill.yaml").read_text())
    pending = set(backfill.get("pending") or [])

    added = sorted(pending - BACKFILL_BASELINE)

    assert not added, (
        "badge_backfill.yaml may only shrink; give these recipes badges instead: %s"
        % ", ".join(added)
    )
