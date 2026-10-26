# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.multimodal_utils.models import qwen_video_handoff as mod
from dynamo.vllm.multimodal_utils.request_processor import (
    MissingMultimodalHandoffError,
    VllmMultimodalRequestProcessor,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def processed():
    field = MultiModalFieldConfig.batched("video", keep_on_cpu=True).field
    return {
        "type": "multimodal",
        "prompt_token_ids": list(range(12)),
        "mm_hashes": {"video": ["opaque-native-identity"]},
        "mm_placeholders": {
            "video": [
                PlaceholderRange(
                    offset=2,
                    length=7,
                    is_embed=torch.tensor(
                        [True, False, False, False, True, False, False]
                    ),
                )
            ]
        },
        "mm_kwargs": {
            "video": [
                MultiModalKwargsItem(
                    {
                        "video_grid_thw": MultiModalFieldElem(
                            data=torch.tensor([2, 2, 2]), field=field
                        ),
                        "timestamps": MultiModalFieldElem(data=[0.1, 0.3], field=field),
                        "pixel_values_videos": MultiModalFieldElem(
                            data=torch.ones(8, 16), field=field
                        ),
                    }
                )
            ]
        },
    }


def restore(state, **kwargs):
    return mod.QwenVideoHandoff.restore(
        state,
        max_model_len=128,
        media_counts={"video": 1},
        spatial_merge_size=2,
        **kwargs,
    )


def test_roundtrip_preserves_native_layout_without_pixels(processed):
    state = mod.QwenVideoHandoff().capture(processed)
    restored = restore(json.loads(json.dumps(state)))
    assert restored["prompt_token_ids"] == processed["prompt_token_ids"]
    assert restored["mm_hashes"] == processed["mm_hashes"]
    actual = restored["mm_placeholders"]["video"][0]
    expected = processed["mm_placeholders"]["video"][0]
    assert actual.offset == expected.offset and actual.length == expected.length
    assert torch.equal(actual.is_embed, expected.is_embed)
    item = restored["mm_kwargs"]["video"][0]
    assert set(item) == {"video_grid_thw", "timestamps"}
    assert item["video_grid_thw"].data.tolist() == [2, 2, 2]
    assert item["timestamps"].data.tolist() == [0.1, 0.3]
    assert "pixel_values_videos" not in json.dumps(state)


def test_metadata_only_input_has_identical_native_mrope(processed):
    processed["prompt_token_ids"] = [5, 1001, 1004, 1002, 6, 1001, 1004, 1002, 7]
    processed["mm_placeholders"]["video"] = [
        PlaceholderRange(
            offset=1,
            length=7,
            is_embed=torch.tensor([False, True, False, False, False, True, False]),
        )
    ]
    restored = restore(mod.QwenVideoHandoff().capture(processed))
    config = SimpleNamespace(
        video_token_id=1004,
        vision_start_token_id=1001,
        vision_end_token_id=1002,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )

    def positions(prompt):
        feature = MultiModalFeatureSpec(
            data=prompt["mm_kwargs"]["video"][0],
            modality="video",
            identifier=prompt["mm_hashes"]["video"][0],
            mm_position=prompt["mm_placeholders"]["video"][0],
        )
        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            prompt["prompt_token_ids"], [feature], config
        )

    original_positions, original_delta = positions(processed)
    restored_positions, restored_delta = positions(restored)
    assert torch.equal(original_positions, restored_positions)
    assert original_delta == restored_delta


def test_multiple_videos_keep_separate_native_identities(processed):
    processed["prompt_token_ids"] = list(range(24))
    processed["mm_hashes"]["video"].append("second-video")
    processed["mm_kwargs"]["video"].append(
        copy.deepcopy(processed["mm_kwargs"]["video"][0])
    )
    span = replace(processed["mm_placeholders"]["video"][0], offset=12)
    processed["mm_placeholders"]["video"].append(span)
    state = mod.QwenVideoHandoff().capture(processed)
    restored = mod.QwenVideoHandoff.restore(
        state, max_model_len=128, media_counts={"video": 2}, spatial_merge_size=2
    )
    assert restored["mm_hashes"] == processed["mm_hashes"]
    assert [span.offset for span in restored["mm_placeholders"]["video"]] == [2, 12]


def test_mixed_image_video_keeps_image_metadata(processed):
    processed["mm_hashes"]["image"] = ["image-identity"]
    processed["mm_placeholders"]["image"] = [PlaceholderRange(offset=10, length=1)]
    processed["mm_kwargs"]["image"] = [
        MultiModalKwargsItem(
            {
                "image_grid_thw": MultiModalFieldElem(
                    data=torch.tensor([1, 2, 2]),
                    field=MultiModalFieldConfig.batched(
                        "image", keep_on_cpu=True
                    ).field,
                )
            }
        )
    ]
    state = mod.QwenVideoHandoff().capture(processed)
    restored = mod.QwenVideoHandoff.restore(
        state,
        max_model_len=128,
        media_counts={"video": 1, "image": 1},
        spatial_merge_size=2,
    )
    assert set(restored["mm_kwargs"]) == {"image", "video"}
    assert restored["mm_kwargs"]["image"][0]["image_grid_thw"].data.tolist() == [
        1,
        2,
        2,
    ]


def test_sender_cache_hit_uses_bounded_metadata_cache(processed):
    handoff = mod.QwenVideoHandoff()
    cold = handoff.capture(processed)
    processed["mm_kwargs"]["video"] = [None]
    warm = handoff.capture(processed)
    assert warm == cold
    assert restore(warm)["mm_kwargs"]["video"][0]["video_grid_thw"].data.tolist() == [
        2,
        2,
        2,
    ]


def test_metadata_cache_miss_preserves_raw_media_fallback(processed):
    processed["mm_kwargs"]["video"] = [None]
    assert mod.QwenVideoHandoff().capture(processed) is None


def test_metadata_cache_is_byte_bounded(processed, monkeypatch):
    monkeypatch.setattr(mod, "_MAX_METADATA_CACHE_BYTES", 100)
    handoff = mod.QwenVideoHandoff()
    assert handoff.capture(processed) is not None
    processed["mm_hashes"]["video"] = ["second-identity"]
    assert handoff.capture(processed) is not None
    assert handoff._cache_bytes <= 100
    assert len(handoff._metadata) == 1
    processed["mm_hashes"]["video"] = ["opaque-native-identity"]
    processed["mm_kwargs"]["video"] = [None]
    assert handoff.capture(processed) is None


@pytest.mark.parametrize("state", [None, {}, {"version": 2}, {"version": "1"}])
def test_missing_or_unknown_state_uses_legacy_path(state):
    assert restore(state) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt_token_ids", []),
        ("prompt_token_ids", [True, 1]),
        ("prompt_token_ids", [-1, 1]),
        ("prompt_token_ids", list(range(129))),
        ("mm_hashes", {"video": []}),
        ("mm_hashes", {"video": [None]}),
        ("version", True),
    ],
)
def test_malformed_known_state_is_rejected(processed, key, value):
    state = mod.QwenVideoHandoff().capture(processed)
    state[key] = value
    with pytest.raises(ValueError):
        restore(state)


@pytest.mark.parametrize("grid", [[0, 2, 2], [True, 2, 2], [2, 3, 2], [2, 10**12, 2]])
def test_invalid_grid_is_rejected(processed, grid):
    state = mod.QwenVideoHandoff().capture(processed)
    state["metadata"]["video"][0]["video_grid_thw"] = grid
    with pytest.raises(ValueError):
        restore(state)


@pytest.mark.parametrize(
    "timestamps", [[0.1], [0.1, float("nan")], [True, 0.3], [-1, 0.3]]
)
def test_invalid_timestamps_are_rejected(processed, timestamps):
    state = mod.QwenVideoHandoff().capture(processed)
    state["metadata"]["video"][0]["timestamps"] = timestamps
    with pytest.raises(ValueError):
        restore(state)


@pytest.mark.parametrize(
    "changes", [{"offset": -1}, {"length": 100}, {"is_embed": [True]}]
)
def test_invalid_span_is_rejected(processed, changes):
    state = mod.QwenVideoHandoff().capture(processed)
    state["mm_placeholders"]["video"][0].update(changes)
    with pytest.raises(ValueError):
        restore(state)


def test_wrong_media_count_is_rejected(processed):
    state = mod.QwenVideoHandoff().capture(processed)
    with pytest.raises(ValueError):
        mod.QwenVideoHandoff.restore(
            state, max_model_len=128, media_counts={"video": 2}, spatial_merge_size=2
        )


def test_transport_limit_falls_back_on_capture_and_rejects_known_state(
    processed, monkeypatch
):
    state = mod.QwenVideoHandoff().capture(processed)
    monkeypatch.setattr(mod, "_MAX_HANDOFF_BYTES", 1)
    assert mod.QwenVideoHandoff().capture(processed) is None
    with pytest.raises(ValueError):
        restore(state)


def test_returned_metadata_does_not_mutate_cached_values(processed):
    handoff = mod.QwenVideoHandoff()
    state = handoff.capture(processed)
    expected = copy.deepcopy(state)
    state["metadata"]["video"][0]["timestamps"][0] = 99
    processed["mm_kwargs"]["video"] = [None]
    assert handoff.capture(processed) == expected


@pytest.fixture
def processor(processed):
    renderer = SimpleNamespace(render_cmpl_async=AsyncMock(return_value=[processed]))
    engine = SimpleNamespace(
        input_processor=SimpleNamespace(renderer=renderer),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="qwen3_vl",
                    vision_config=SimpleNamespace(spatial_merge_size=2),
                ),
                max_model_len=128,
            )
        ),
    )
    return VllmMultimodalRequestProcessor(
        model="Qwen/Qwen3-VL-2B-Instruct",
        engine_client=engine,
        use_unified_vision_chunk=False,
        image_loader=SimpleNamespace(load_image_batch=AsyncMock()),
        video_loader=SimpleNamespace(
            load_video_batch=AsyncMock(return_value=[object()])
        ),
        audio_loader=SimpleNamespace(load_audio_batch=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_prefill_renders_once_and_retains_native_layout(processor, processed):
    prompt = {"prompt_token_ids": [1, 2], "multi_modal_data": {"video": object()}}
    rendered, state = await processor.prepare_prefill_video_prompt(
        prompt, {"video": object()}
    )
    assert rendered is processed
    assert state["prompt_token_ids"] == processed["prompt_token_ids"]
    processor.engine_client.input_processor.renderer.render_cmpl_async.assert_awaited_once_with(
        [prompt]
    )


@pytest.mark.parametrize("model_type", ["cosmos3", "cosmos3_omni"])
@pytest.mark.asyncio
async def test_cosmos3_uses_the_qwen_video_handoff(processor, processed, model_type):
    processor.engine_client.vllm_config.model_config.hf_config.model_type = model_type
    rendered, state = await processor.prepare_prefill_video_prompt(
        {"prompt_token_ids": [1, 2]}, {"video": object()}
    )
    assert rendered is processed and state is not None


@pytest.mark.asyncio
async def test_decode_skips_every_media_loader_with_valid_handoff(processor, processed):
    state = mod.QwenVideoHandoff().capture(processed)
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {"video_url": [{"Url": "must-not-be-fetched"}]},
        "prefill_result": {
            "disaggregated_params": {
                "embedding_params": {mod.QWEN_VIDEO_HANDOFF_KEY: state}
            }
        },
    }
    result = await processor.prepare_input(
        request, "decode", None, DisaggregationMode.DECODE
    )
    assert result.multi_modal_data is None
    assert (
        result.pre_rendered_prompt["prompt_token_ids"] == processed["prompt_token_ids"]
    )
    processor.video_loader.load_video_batch.assert_not_awaited()
    processor.image_loader.load_image_batch.assert_not_awaited()
    processor.audio_loader.load_audio_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_version_falls_back_to_real_video(processor):
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {"video_url": [{"Url": "video"}]},
        "prefill_result": {
            "disaggregated_params": {
                "embedding_params": {mod.QWEN_VIDEO_HANDOFF_KEY: {"version": 2}}
            }
        },
    }
    result = await processor.prepare_input(
        request, "decode", None, DisaggregationMode.DECODE
    )
    assert result.pre_rendered_prompt is None
    assert "video" in result.multi_modal_data
    processor.video_loader.load_video_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_known_handoff_does_not_reload_silently(processor, processed):
    state = mod.QwenVideoHandoff().capture(processed)
    state["prompt_token_ids"] = []
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {"video_url": [{"Url": "video"}]},
        "prefill_result": {
            "disaggregated_params": {
                "embedding_params": {mod.QWEN_VIDEO_HANDOFF_KEY: state}
            }
        },
    }
    with pytest.raises(MissingMultimodalHandoffError):
        await processor.prepare_input(
            request, "decode", None, DisaggregationMode.DECODE
        )
    processor.video_loader.load_video_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_keeps_legacy_prefill_path(processor):
    prompt = {"prompt_token_ids": [1, 2]}
    rendered, state = await processor.prepare_prefill_video_prompt(
        prompt, {"video": object(), "audio": object()}
    )
    assert rendered is prompt and state is None
    processor.engine_client.input_processor.renderer.render_cmpl_async.assert_not_awaited()
