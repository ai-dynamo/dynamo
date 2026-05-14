# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import _collect_models_to_download
from tests.utils import model_registry
from tests.utils.model_registry import (
    DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB,
    MODEL_REGISTRY,
    MODEL_SPECS,
    constant_name_for_repo_id,
    downloadable_model_ids,
    validate_ci_model_ids,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.parallel, pytest.mark.gpu_0]


_QUOTED_REPO_ID_RE = re.compile(
    r"""["']([A-Za-z0-9][A-Za-z0-9_.-]+/[A-Za-z0-9][A-Za-z0-9_.-]+)["']"""
)

_NON_MODEL_REPO_LIKE_STRINGS = {
    "A100/H100",
    "A6000/A40",
    "America/Los_Angeles",
    "Frontend/Router",
    "Offload/Onboard",
    "app.kubernetes.io/name",
    "application/json",
    "bin/compliance-test.ts",
    "cais/mmlu",
    "chaos-mesh.org/v1alpha1",
    "components/src",
    "container/build.sh",
    "data/dev",
    "data/test",
    "edge/embedded",
    "image/png",
    "kubernetes.io/hostname",
    "kubernetes.io/metadata.name",
    "launch/lora",
    "lora/agg_lora.sh",
    "lora/agg_lora_router.sh",
    "networking.k8s.io/v1",
    "node_name/gpu0",
    "nvidia.com/enable-grove",
    "nvidia.com/v1alpha1",
    "nvidia.com/v1beta1",
    "pods/exec",
    "test.dynamo/managed",
    "test.fault-injection/cordoned",
    "test.fault-injection/reason",
    "tests/serve",
    "tests/test_models_dir_flag.py",
    "xpu/agg_lmcache_multiproc_xpu.sh",
    "xpu/agg_lmcache_xpu.sh",
    "xpu/agg_multimodal_xpu.sh",
    "xpu/agg_request_planes_xpu.sh",
    "xpu/agg_router_approx_xpu.sh",
    "xpu/agg_router_xpu.sh",
    "xpu/agg_xpu.sh",
}


class _DummyMark:
    def __init__(self, name: str, *args):
        self.name = name
        self.args = args


class _DummyItem:
    own_markers: list[_DummyMark]

    def __init__(self, model_id: str, *, skipped: bool = False):
        self.own_markers = [_DummyMark("skip")] if skipped else []
        self._model_mark = _DummyMark("model", model_id)

    def iter_markers(self, marker_name: str):
        if marker_name == "model":
            return iter([self._model_mark])
        return iter(())


def test_model_registry_metadata_is_complete():
    assert MODEL_REGISTRY
    assert len(MODEL_REGISTRY) == len(MODEL_SPECS)
    for repo_id, spec in MODEL_REGISTRY.items():
        assert repo_id == spec.repo_id
        assert re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]+/[A-Za-z0-9][A-Za-z0-9_.-]+",
            spec.repo_id,
        )
        assert spec.hf_url == f"https://huggingface.co/{spec.repo_id}"
        assert spec.snapshot_size_gib > 0
        assert spec.kind
        assert spec.architecture_tags
        assert tuple(sorted(set(spec.architecture_tags))) == spec.architecture_tags


def test_model_registry_constant_exports_match_repo_ids():
    for repo_id in MODEL_REGISTRY:
        constant_name = constant_name_for_repo_id(repo_id)
        assert hasattr(model_registry, constant_name), repo_id
        assert getattr(model_registry, constant_name) == repo_id


def test_model_registry_size_policy():
    unapproved = [
        spec.repo_id
        for spec in MODEL_REGISTRY.values()
        if spec.download_required
        and spec.snapshot_size_gib > DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB
        and not spec.over_cap_exception
    ]
    assert unapproved == []

    for spec in MODEL_REGISTRY.values():
        if spec.over_cap_exception:
            assert spec.snapshot_size_gib > DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB
            assert spec.exception_reason
        elif spec.download_required:
            assert spec.snapshot_size_gib <= DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB


def test_downloadable_model_ids_skip_metadata_only_models():
    assert downloadable_model_ids([model_registry.QWEN_QWEN3_32B]) == ()


def test_validate_ci_model_ids_rejects_unknown_models():
    with pytest.raises(ValueError, match="Unregistered CI model"):
        validate_ci_model_ids(["unknown-org/unknown-model"])


def test_collection_model_smoke_validates_and_skips_skipped_items():
    items = [
        _DummyItem(model_registry.QWEN_QWEN3_0_6B),
        _DummyItem("unknown-org/unknown-model", skipped=True),
    ]

    assert _collect_models_to_download(items) == {model_registry.QWEN_QWEN3_0_6B}


def test_ci_test_model_literals_are_registered():
    tests_dir = Path(__file__).resolve().parents[1]
    found: set[str] = set()

    for path in tests_dir.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix not in {".py", ".sh", ".yaml", ".yml"}:
            continue
        text = path.read_text(errors="ignore")
        found.update(match.group(1) for match in _QUOTED_REPO_ID_RE.finditer(text))

    unknown = sorted(found - set(MODEL_REGISTRY) - _NON_MODEL_REPO_LIKE_STRINGS)
    assert unknown == []
