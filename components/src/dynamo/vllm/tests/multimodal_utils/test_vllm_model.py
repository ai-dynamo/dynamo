# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.model."""

import json
from importlib import import_module
from unittest.mock import MagicMock

import pytest
import torch

from dynamo.vllm.multimodal_utils import model as model_module
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


class TestLoadVisionModel:
    def test_vllm_encoder_settings_from_environment(self, monkeypatch):
        fake_visual = object()
        fake_llm = MagicMock()
        model_runner = (
            fake_llm.return_value.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        )
        model_runner.model.visual = fake_visual

        monkeypatch.setattr(model_module, "VLLM_ENCODER", 1)
        monkeypatch.setattr(model_module, "LLM", fake_llm)
        monkeypatch.setattr(model_module, "update_environment_variables", MagicMock())
        monkeypatch.delenv("DYN_VLLM_SKIP_ENCODER_ONLY_KERNEL_WARMUP", raising=False)
        monkeypatch.setenv("DYN_VLLM_ENCODER_GPU_MEMORY_UTILIZATION", "0.125")
        monkeypatch.setenv("DYN_VLLM_ENCODER_KV_CACHE_MEMORY_BYTES", "4294967296")
        monkeypatch.setenv("DYN_VLLM_ENCODER_MAX_NUM_SEQS", "64")

        loaded = model_module.load_vision_model("Qwen/Qwen3.5-9B")

        kwargs = fake_llm.call_args.kwargs
        assert kwargs["gpu_memory_utilization"] == 0.125
        assert kwargs["kv_cache_memory_bytes"] == 4294967296
        assert kwargs["max_num_seqs"] == 64
        assert loaded is fake_visual

    def test_encoder_kernel_warmup_patch_is_scoped(self, monkeypatch):
        worker_module = import_module("vllm.v1.worker.gpu_worker")
        original_kernel_warmup = worker_module.kernel_warmup
        monkeypatch.setenv("DYN_VLLM_SKIP_ENCODER_ONLY_KERNEL_WARMUP", "1")

        with pytest.raises(RuntimeError, match="test cleanup"):
            with model_module._maybe_skip_encoder_only_kernel_warmup():
                assert worker_module.kernel_warmup is not original_kernel_warmup
                assert worker_module.kernel_warmup() is None
                raise RuntimeError("test cleanup")

        assert worker_module.kernel_warmup is original_kernel_warmup


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
