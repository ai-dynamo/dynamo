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
                "Qwen/Qwen3.5-9B",
                ModelFamily.QWEN_VL,
                id="hf-id-qwen3.5-unified",
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
    """Cases that genuinely require filesystem state (a real `config.json` to
    exercise the metadata stage). Cases where directory existence is irrelevant
    to the result are covered string-only in `TestResolveModelFamily`."""

    @pytest.mark.parametrize(
        "subdir, architectures, expected",
        [
            pytest.param(
                "Qwen--Qwen2-VL-2B-Instruct/v2",
                ["Qwen2VLForConditionalGeneration"],
                ModelFamily.QWEN_VL,
                id="metadata-qwen2-vl",
            ),
            pytest.param(
                "Qwen--Qwen3-VL-2B-Instruct/v2",
                ["Qwen3VLForConditionalGeneration"],
                ModelFamily.QWEN_VL,
                id="metadata-qwen3-vl",
            ),
            pytest.param(
                "llava-hf--llava-1.5-7b-hf/v1",
                ["LlavaForConditionalGeneration"],
                ModelFamily.LLAVA,
                id="metadata-llava",
            ),
            pytest.param(
                "Qwen--Qwen3.5-9B/v1",
                ["Qwen3_5ForConditionalGeneration"],
                ModelFamily.QWEN_VL,
                id="metadata-qwen3.5-unified",
            ),
        ],
    )
    def test_metadata_stage_resolves_family(
        self, tmp_path, subdir, architectures, expected
    ):
        model_dir = tmp_path / subdir
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": architectures})
        )
        assert resolve_model_family(str(model_dir)) == expected

    def test_unrecognized_arch_falls_through_to_name_stage(self, tmp_path):
        """`config.json` exists but its arch isn't in the registry — the
        resolver must fall through to the name stage rather than return
        None on metadata miss."""
        model_dir = tmp_path / "Qwen--Qwen2-VL-2B-Instruct" / "v2"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": ["SomeFutureQwenVariantClass"]})
        )
        assert resolve_model_family(str(model_dir)) == ModelFamily.QWEN_VL
