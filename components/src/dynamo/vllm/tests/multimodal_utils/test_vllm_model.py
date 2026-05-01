# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.model."""

import json

import pytest
import torch

from dynamo.vllm.multimodal_utils.model import (
    ModelFamily,
    construct_qwen_decode_mm_data,
    resolve_model_family,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class TestMultiModalUtils:
    def test_construct_qwen_decode_mm_data(self):
        max_rounds = int(torch.finfo(torch.float16).max) + 2
        expected_image_grid_thw_tensor = torch.tensor([16, 16])
        for i in range(max_rounds):
            # Should not raise any exception
            try:
                mm_data = construct_qwen_decode_mm_data(
                    image_grid_thw=[16, 16],
                    embeddings_shape=[2, 1024],
                    request_id=str(i),
                )
            except Exception as e:
                pytest.fail(
                    f"construct_qwen_decode_mm_data raised {type(e).__name__} on round {i}: {e}"
                )
            assert "image" in mm_data
            assert "image_grid_thw" in mm_data["image"]
            assert "image_embeds" in mm_data["image"]
            assert torch.allclose(
                mm_data["image"]["image_grid_thw"], expected_image_grid_thw_tensor
            )
            # Embedding values are randomly genearted as placehodler, we only check the shape
            assert mm_data["image"]["image_embeds"].shape == (2, 1024)


class TestResolveModelFamily:
    """Cases where resolution is determined entirely by the input string
    (no filesystem state needed). Filesystem-dependent cases live in
    `TestResolveModelFamilyOnDisk`."""

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            pytest.param(
                "Qwen/Qwen2-VL-2B-Instruct",
                ModelFamily.QWEN_VL,
                id="hf-id-qwen2-vl",
            ),
            pytest.param(
                "Qwen/Qwen3-VL-2B-Instruct",
                ModelFamily.QWEN_VL,
                id="hf-id-qwen3-vl",
            ),
            pytest.param(
                "llava-hf/llava-1.5-7b-hf",
                ModelFamily.LLAVA,
                id="hf-id-llava",
            ),
            pytest.param(
                "/root/.cache/huggingface/hub/"
                "models--Qwen--Qwen2-VL-2B-Instruct/snapshots/abc123",
                ModelFamily.QWEN_VL,
                id="hf-cache-snapshot",
            ),
            pytest.param(
                "/local_store/Qwen--Qwen3-VL-2B-Instruct/v2",
                ModelFamily.QWEN_VL,
                id="local_store-parent-with-version",
            ),
            pytest.param(
                "/local_store/qwen2.5-vl-7b-instruct/v3",
                ModelFamily.QWEN_VL,
                id="local_store-org-less",
            ),
            pytest.param("RandomOrg/RandomModel-7B", None, id="unsupported-hf-id"),
        ],
    )
    def test_resolve_string_inputs(self, model_name, expected):
        assert resolve_model_family(model_name) == expected


class TestResolveModelFamilyOnDisk:
    """Cases where resolution depends on filesystem state (`config.json` or
    directory existence)."""

    def write_config(self, model_dir, architectures):
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": architectures})
        )

    def test_metadata_stage_qwen2_vl(self, tmp_path):
        model_dir = tmp_path / "Qwen--Qwen2-VL-2B-Instruct" / "v2"
        self.write_config(model_dir, ["Qwen2VLForConditionalGeneration"])
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL

    def test_metadata_stage_qwen3_vl(self, tmp_path):
        model_dir = tmp_path / "Qwen--Qwen3-VL-2B-Instruct" / "v2"
        self.write_config(model_dir, ["Qwen3VLForConditionalGeneration"])
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL

    def test_metadata_stage_llava(self, tmp_path):
        model_dir = tmp_path / "llava-hf--llava-1.5-7b-hf" / "v1"
        self.write_config(model_dir, ["LlavaForConditionalGeneration"])
        assert resolve_model_family(str(model_dir)) == ModelFamily.LLAVA

    def test_name_stage_no_config_cache_style_parent(self, tmp_path):
        """Directory exists but has no `config.json` — name-stage path-component
        scan should still resolve via the `Qwen--Qwen2-VL-2B-Instruct` parent."""
        model_dir = tmp_path / "Qwen--Qwen2-VL-2B-Instruct" / "v2"
        model_dir.mkdir(parents=True)
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL

    def test_name_stage_no_config_llava(self, tmp_path):
        model_dir = tmp_path / "llava-hf--llava-1.5-7b-hf" / "v1"
        model_dir.mkdir(parents=True)
        assert resolve_model_family(str(model_dir)) == ModelFamily.LLAVA

    def test_unrecognized_arch_falls_through_to_name_stage(self, tmp_path):
        """Metadata stage misses (arch not in registry); name stage catches via
        the `Qwen--Qwen2-VL-2B-Instruct` parent segment."""
        model_dir = tmp_path / "Qwen--Qwen2-VL-2B-Instruct" / "v2"
        self.write_config(model_dir, ["SomeFutureQwenVariantClass"])
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL

    def test_org_less_path_resolves_via_bare_substring(self, tmp_path):
        """Org prefix dropped; verbatim normalization + bare-name substring
        match resolves the family."""
        model_dir = tmp_path / "qwen3-vl-2b-instruct" / "v2"
        model_dir.mkdir(parents=True)
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL

    def test_unsupported_path_returns_none(self, tmp_path):
        model_dir = tmp_path / "random-org--random-model" / "v1"
        model_dir.mkdir(parents=True)
        assert resolve_model_family(str(model_dir)) is None
