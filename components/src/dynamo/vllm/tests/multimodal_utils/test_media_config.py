# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import dynamo.vllm.multimodal_utils.media_config as media_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class FakeDecoder:
    def enable_image(self, options):
        self.image_options = options

    def enable_video(self, options):
        self.video_options = options


class FakePreprocessor:
    def enable_video(self, model_type, config_json):
        self.video_config = (model_type, json.loads(config_json))


class FakeFetcher:
    def timeout_ms(self, value):
        self.timeout = value

    def allow_direct_ip(self, value):
        self.direct_ip = value

    def allow_direct_port(self, value):
        self.direct_port = value


def test_video_processor_selection_is_delegated_to_registry(tmp_path, monkeypatch):
    (tmp_path / "preprocessor_config.json").write_text(
        '{"patch_size": 16}', encoding="utf-8"
    )
    monkeypatch.setattr(media_config, "MediaDecoder", FakeDecoder)
    monkeypatch.setattr(media_config, "MediaPreprocessor", FakePreprocessor)
    monkeypatch.setattr(media_config, "MediaFetcher", FakeFetcher)

    decoder, processor, fetcher = media_config.create_frontend_media_config(
        True,
        video_preprocessing=True,
        model=str(tmp_path),
        model_type="qwen3_vl",
    )

    assert decoder.video_options["fps"] == 2.0
    assert processor.video_config == ("qwen3_vl", {"patch_size": 16})
    assert fetcher.timeout == 30000
