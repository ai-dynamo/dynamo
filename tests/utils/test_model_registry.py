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
    GLOBAL_CI_MODEL_OVERRIDE_ENV_VAR,
    MODEL_PROFILES,
    MODEL_REGISTRY,
    MODEL_SPECS,
    ModelQuery,
    resolve_model_profile,
    select_models,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
]


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
    "application/octet-stream",
    "benchmarks/pyproject.toml",
    "bin/compliance-test.ts",
    "cais/mmlu",
    "chaos-mesh.org/v1alpha1",
    "components/src",
    "components/leaf",
    "components/parent",
    "capacity/fast",
    "capacity/other",
    "capacity/slow",
    "control/sleep",
    "container/build.sh",
    "data/dev",
    "data/test",
    "edge/embedded",
    "examples/custom_encoder",
    "image/png",
    "kubernetes.io/hostname",
    "kubernetes.io/metadata.name",
    "kustomize/base",
    "kvbm_integration/t8.shakespeare.txt",
    "launch/lora",
    "lora/agg_lora.sh",
    "lora/agg_lora_router.sh",
    "networking.k8s.io/v1",
    "node_name/gpu0",
    "nvidia.com/enable-grove",
    "nvidia.com/dynamo-graph-deployment-name",
    "nvidia.com/gpu",
    "nvidia.com/gpu.present",
    "nvidia.com/mig.config",
    "nvidia.com/snapshot-checkpoint-id",
    "nvidia.com/snapshot-is-checkpoint-source",
    "nvidia.com/snapshot-is-restore-target",
    "nvidia.com/snapshot-target-containers",
    "nvidia.com/v1alpha1",
    "nvidia.com/v1beta1",
    "pods/exec",
    "scripts/generate_kustomize_openapi.py",
    "scripts/kustomize-matrix.py",
    "systems/h200_sxm.yaml",
    "test.dynamo/managed",
    "test.fault-injection/cordoned",
    "test.fault-injection/reason",
    "tests/serve",
    "tests/test_models_dir_flag.py",
    "video/mp4",
    # A deliberately fake served-model id used by a frontend protocol unit test.
    "Qwen/Qwen3-Coder",
    "xpu/agg_lmcache_multiproc_xpu.sh",
    "xpu/agg_lmcache_xpu.sh",
    "xpu/agg_multimodal_xpu.sh",
    "xpu/agg_multimodal_router_chat_processor_xpu.sh",
    "xpu/agg_multimodal_router_xpu.sh",
    "xpu/agg_request_planes_xpu.sh",
    "xpu/agg_router_approx_xpu.sh",
    "xpu/agg_router_xpu.sh",
    "xpu/agg_xpu.sh",
    "model_configs/Qwen--Qwen3-32B_config.json",
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


def test_model_registry_invariants_and_size_policy():
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
        assert spec.characteristics
        assert tuple(sorted(set(spec.characteristics))) == spec.characteristics
        assert tuple(sorted(set(spec.backends))) == spec.backends
        constant_name = model_registry.constant_name_for_repo_id(repo_id)
        assert getattr(model_registry, constant_name) == repo_id

    for spec in MODEL_REGISTRY.values():
        if spec.over_cap_exception:
            assert spec.snapshot_size_gib > DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB
            assert spec.exception_reason
        elif spec.download_required:
            assert spec.snapshot_size_gib <= DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB

    for profile_name, profile in MODEL_PROFILES.items():
        assert (
            resolve_model_profile(profile_name, override=profile.default_repo_id)
            == profile.default_repo_id
        )

    assert model_registry.downloadable_model_ids([model_registry.QWEN_QWEN3_32B]) == ()


def test_characteristic_selection_is_smallest_first_and_profiles_are_safe(
    monkeypatch,
):
    candidates = select_models(
        ModelQuery(
            kind="llm",
            required_characteristics=frozenset(
                {"instruction_tuned", "small_ci_candidate", "text_generation"}
            ),
            required_backends=frozenset({"vllm"}),
            max_parameter_count_millions=400,
        )
    )
    assert [spec.repo_id for spec in candidates] == [
        model_registry.IBM_GRANITE_GRANITE_4_0_H_350M,
        model_registry.LIQUIDAI_LFM2_5_350M,
    ]
    assert (
        resolve_model_profile(
            "vllm_smoke", override=model_registry.IBM_GRANITE_GRANITE_4_0_H_350M
        )
        == model_registry.IBM_GRANITE_GRANITE_4_0_H_350M
    )
    assert (
        resolve_model_profile(
            "trtllm_smoke", override=model_registry.GOOGLE_GEMMA_3_270M_IT
        )
        == model_registry.GOOGLE_GEMMA_3_270M_IT
    )

    monkeypatch.setenv(
        GLOBAL_CI_MODEL_OVERRIDE_ENV_VAR, model_registry.GOOGLE_GEMMA_3_270M_IT
    )
    for profile_name in (
        "cross_backend_smoke",
        "sglang_smoke",
        "trtllm_smoke",
        "vllm_smoke",
    ):
        assert (
            resolve_model_profile(profile_name) == model_registry.GOOGLE_GEMMA_3_270M_IT
        )
    assert resolve_model_profile("kv_transfer") == model_registry.QWEN_QWEN3_0_6B

    monkeypatch.setenv(
        "DYN_CI_VLLM_SMOKE_MODEL",
        model_registry.IBM_GRANITE_GRANITE_4_0_H_350M,
    )
    assert (
        resolve_model_profile("vllm_smoke")
        == model_registry.IBM_GRANITE_GRANITE_4_0_H_350M
    )
    with pytest.raises(ValueError, match="does not satisfy"):
        resolve_model_profile(
            "trtllm_smoke",
            override=model_registry.IBM_GRANITE_GRANITE_4_0_H_350M,
        )
    with pytest.raises(ValueError, match="does not satisfy"):
        resolve_model_profile(
            "kv_transfer", override=model_registry.LIQUIDAI_LFM2_5_350M
        )


def test_collection_model_smoke_validates_and_skips_skipped_items():
    items = [
        _DummyItem(model_registry.QWEN_QWEN3_0_6B),
        _DummyItem("unknown-org/unknown-model", skipped=True),
    ]

    assert _collect_models_to_download(items) == {model_registry.QWEN_QWEN3_0_6B}
    with pytest.raises(ValueError, match="Unregistered CI model"):
        _collect_models_to_download([_DummyItem("unknown-org/unknown-model")])


def test_ci_test_model_literals_are_registered():
    tests_dir = Path(__file__).resolve().parents[1]
    found: set[str] = set()

    for path in tests_dir.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path == Path(__file__).resolve():
            continue
        if path.suffix not in {".json", ".py", ".sh", ".toml", ".yaml", ".yml"}:
            continue
        text = path.read_text(errors="ignore")
        found.update(match.group(1) for match in _QUOTED_REPO_ID_RE.finditer(text))

    unknown = sorted(found - set(MODEL_REGISTRY) - _NON_MODEL_REPO_LIKE_STRINGS)
    assert unknown == []
