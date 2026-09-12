# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.custom_encoder.benchmark.fixed_text_image_workload import (
    BENCHMARK_IMAGE_SIZE_COUNTS,
    CONCURRENCY,
    REQUESTS,
    TARGET_OSL,
    TEXT_ISL,
    _calibrate_prompt,
    _request_schedule,
    generate_workload,
    validate_workload,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> object:
        assert not add_special_tokens
        token_count = 600 + text.count("benchmark")
        return SimpleNamespace(input_ids=list(range(token_count)))


class _FakeImageProcessor:
    merge_size = 2

    def __call__(self, *, images: list, return_tensors: str) -> dict:
        assert return_tensors == "pt"
        grid = 22 if images[0].size == (300, 300) else 36
        return {"image_grid_thw": torch.tensor([[1, grid, grid]])}


def _install_model_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = "examples.custom_encoder.benchmark.fixed_text_image_workload"
    monkeypatch.setattr(
        f"{module}.AutoTokenizer.from_pretrained", lambda _model: _FakeTokenizer()
    )
    monkeypatch.setattr(
        f"{module}.AutoProcessor.from_pretrained",
        lambda _model: SimpleNamespace(image_processor=_FakeImageProcessor()),
    )
    monkeypatch.setattr(
        f"{module}._calculate_custom_isl_components",
        lambda _tokenizer, _processor, _prompt, image: (
            773 if image.size == (300, 300) else 976
        ),
    )


def test_default_contract_is_fixed_text_plus_balanced_unique_images() -> None:
    assert REQUESTS == 1000
    assert CONCURRENCY == 64
    assert TEXT_ISL == 644
    assert TARGET_OSL == 7
    assert BENCHMARK_IMAGE_SIZE_COUNTS == ((300, 300, 500), (500, 500, 500))


def test_prompt_calibration_requires_exact_raw_token_count() -> None:
    prompt, observed = _calibrate_prompt(
        TEXT_ISL, lambda text: 600 + text.count("benchmark")
    )
    assert observed == TEXT_ISL
    assert prompt.count("benchmark") == 44


def test_request_schedule_is_deterministic_and_uses_every_image_once() -> None:
    images = [f"image-{index}" for index in range(10)]
    first = _request_schedule(images, requests=10, seed=42)
    assert first == _request_schedule(images, requests=10, seed=42)
    assert Counter(first) == Counter({image: 1 for image in images})


def test_mixed_workload_keeps_644_raw_tokens_and_reports_decoder_isls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_model_fakes(monkeypatch)
    manifest_path = generate_workload(
        tmp_path,
        requests=2,
        text_isl=644,
        image_size_counts=((300, 300, 1), (500, 500, 1)),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "image_custom_2_textisl644.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["text_isl"] == 644
    assert len({row["text"] for row in rows}) == 1
    assert manifest["unique_encoded_sha256"] == 2
    assert manifest["unique_decoded_rgb_sha256"] == 2
    assert manifest["observed_decoder_isl_by_image_size"] == {
        "300x300": 773,
        "500x500": 976,
    }

    audit = validate_workload(
        tmp_path,
        expected_unique_images=2,
        expected_image_size_counts=((300, 300, 1), (500, 500, 1)),
    )
    assert audit["text_isl"] == 644
    assert audit["observed_decoder_isl_by_image_size"] == {
        "300x300": 773,
        "500x500": 976,
    }
