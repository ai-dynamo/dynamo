# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from dynamo.common.constants import DisaggregationMode
from dynamo.vllm.multimodal_utils import request_processor as mod
from dynamo.vllm.multimodal_utils.models import qwen as qwen_mod

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _processor(
    *,
    model: str = "Qwen/Qwen3-VL-2B-Instruct",
    enabled: bool = True,
    unified_vision_chunk: bool = False,
) -> mod.VllmMultimodalRequestProcessor:
    return mod.VllmMultimodalRequestProcessor(
        model=model,
        enable_multimodal=enabled,
        image_loader=SimpleNamespace(load_image_batch=AsyncMock(return_value=[])),
        video_loader=SimpleNamespace(load_video_batch=AsyncMock(return_value=[])),
        audio_loader=SimpleNamespace(
            load_audio_batch=AsyncMock(return_value=[]),
            load_audio=AsyncMock(return_value=None),
        ),
        use_unified_vision_chunk=unified_vision_chunk,
    )


@pytest.mark.asyncio
async def test_extracts_mixed_url_data_url_and_decoded_media():
    processor = _processor()
    image = Image.new("RGB", (1, 1))
    video = object()
    audio_a = object()
    audio_b = object()
    image_items = [
        {"Url": "data:image/png;base64,AAAA"},
        {"Decoded": {"shape": [1, 1, 3]}},
    ]
    video_items = [{"Url": "https://example.com/video.mp4"}]
    audio_items = [
        {"Url": "https://example.com/a.wav"},
        {"Url": "https://example.com/b.wav"},
    ]
    processor.image_loader.load_image_batch.return_value = [image]
    processor.video_loader.load_video_batch.return_value = [video]
    processor.audio_loader.load_audio_batch.return_value = [audio_a, audio_b]

    result = await processor.extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": image_items,
                "video_url": video_items,
                "audio_url": audio_items,
            }
        },
        "request-1",
        None,
    )

    assert result == {"image": image, "video": video, "audio": [audio_a, audio_b]}
    processor.image_loader.load_image_batch.assert_awaited_once_with(image_items)
    processor.video_loader.load_video_batch.assert_awaited_once_with(video_items)
    processor.audio_loader.load_audio_batch.assert_awaited_once_with(audio_items)
    processor.audio_loader.load_audio.assert_not_awaited()


@pytest.mark.asyncio
async def test_merges_encoder_images_with_local_video_and_decoded_fallback():
    processor = _processor()
    encoded_image = {"image_embeds": object()}
    video = object()
    processor.embedding_loader = SimpleNamespace(
        load_multimodal_embeddings=AsyncMock(return_value={"image": encoded_image})
    )
    processor.video_loader.load_video_batch.return_value = [video]

    result = await processor.extract_multimodal_data(
        {
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.png"}],
                "video_url": [{"Url": "https://example.com/video.mp4"}],
            }
        },
        "request-encoder",
        None,
    )

    assert result == {"image": encoded_image, "video": video}
    processor.image_loader.load_image_batch.assert_not_awaited()

    decoded_image = object()
    processor.image_loader.load_image_batch.return_value = [decoded_image]
    result = await processor.extract_multimodal_data(
        {"multi_modal_data": {"image_url": [{"Decoded": {"shape": [1, 1, 3]}}]}},
        "request-decoded",
        None,
    )

    assert result == {"image": decoded_image}
    processor.embedding_loader.load_multimodal_embeddings.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejects_media_when_multimodal_is_disabled():
    processor = _processor(enabled=False)

    with pytest.raises(ValueError, match="--enable-multimodal"):
        await processor.prepare_prompt(
            {
                "token_ids": [1, 2],
                "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
            },
            "request-2",
            None,
            DisaggregationMode.AGGREGATED,
        )

    with pytest.raises(ValueError, match="--enable-multimodal"):
        await processor.prepare_prompt(
            {
                "token_ids": [1, 2],
                "extra_args": {"mm_kwargs_shm": {"modality": "image", "items": []}},
            },
            "request-transfer-disabled",
            None,
            DisaggregationMode.AGGREGATED,
        )


@pytest.mark.asyncio
async def test_decode_cannot_hide_disabled_media_with_expanded_tokens():
    processor = _processor(
        model="llava-hf/llava-1.5-7b-hf",
        enabled=False,
    )

    with pytest.raises(ValueError, match="--enable-multimodal"):
        await processor.prepare_prompt(
            {
                "token_ids": [1, 2],
                "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
                "prefill_result": {
                    "disaggregated_params": {
                        "embedding_params": {"expanded_prompt_token_ids": [1, 99, 2]}
                    }
                },
            },
            "request-disabled-decode",
            None,
            DisaggregationMode.DECODE,
        )


@pytest.mark.asyncio
async def test_forwards_use_audio_in_video_to_media_loading():
    processor = _processor()
    video = object()
    audio = object()
    processor.video_loader.load_video_batch.return_value = [video]
    processor.audio_loader.load_audio.return_value = audio

    result = await processor.extract_multimodal_data(
        {"multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]}},
        "request-3",
        None,
        {"use_audio_in_video": True},
    )

    assert result == {"video": video, "audio": audio}
    processor.audio_loader.load_audio.assert_awaited_once_with(
        "https://example.com/video.mp4"
    )


@pytest.mark.asyncio
async def test_reads_processor_kwargs_from_router_extra_args():
    processor = _processor()
    processor.video_loader.load_video_batch.return_value = [object()]
    audio = object()
    processor.audio_loader.load_audio.return_value = audio

    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {
                "video_url": [{"Url": "https://example.com/video.mp4"}]
            },
            "extra_args": {"mm_processor_kwargs": {"use_audio_in_video": True}},
        },
        "request-router-kwargs",
        None,
        DisaggregationMode.AGGREGATED,
    )

    assert prepared.mm_processor_kwargs == {"use_audio_in_video": True}
    assert prepared.multi_modal_data["audio"] is audio


@pytest.mark.asyncio
async def test_audio_in_video_preserves_order_and_merges_standalone_audio():
    processor = _processor()
    video_a, video_b = object(), object()
    standalone_audio, audio_a, audio_b = object(), object(), object()
    processor.video_loader.load_video_batch.return_value = [video_a, video_b]
    processor.audio_loader.load_audio_batch.return_value = [standalone_audio]
    processor.audio_loader.load_audio.side_effect = [audio_a, audio_b]

    result = await processor.extract_multimodal_data(
        {
            "multi_modal_data": {
                "video_url": [
                    {"Url": "https://example.com/a.mp4"},
                    {"Url": "https://example.com/b.mp4"},
                ],
                "audio_url": [{"Url": "https://example.com/narration.wav"}],
            }
        },
        "request-audio-order",
        None,
        {"use_audio_in_video": True},
    )

    assert result == {
        "video": [video_a, video_b],
        "audio": [standalone_audio, audio_a, audio_b],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("video_item", "audio_error", "message"),
    [
        ({"Decoded": {"shape": [2, 4, 4, 3]}}, None, "non-URL video item"),
        (
            {"Url": "https://example.com/silent.mp4"},
            RuntimeError("no audio stream"),
            "no audio stream",
        ),
    ],
)
async def test_audio_in_video_rejects_unusable_audio(video_item, audio_error, message):
    processor = _processor()
    processor.video_loader.load_video_batch.return_value = [object()]
    if audio_error is not None:
        processor.audio_loader.load_audio.side_effect = audio_error

    with pytest.raises((ValueError, RuntimeError), match=message):
        await processor.extract_multimodal_data(
            {"multi_modal_data": {"video_url": [video_item]}},
            "request-audio-error",
            None,
            {"use_audio_in_video": True},
        )


def test_build_tokens_prompt_forwards_hashes_kwargs_and_vision_chunk():
    processor = _processor(unified_vision_chunk=True)
    mm_data = {"vision_chunk": {"type": "image", "image": object(), "uuid": None}}

    prompt = processor.build_tokens_prompt(
        {
            "token_ids": [1, 2, 3],
            "extra_args": {"mm_hashes": ["abcd"]},
        },
        mm_data,
        {"num_crops": 4},
    )

    assert prompt["prompt_token_ids"] == [1, 2, 3]
    assert prompt["multi_modal_data"] is mm_data
    assert prompt["multi_modal_uuids"] == {"vision_chunk": ["abcd".ljust(64, "0")]}
    assert prompt["mm_processor_kwargs"] == {"num_crops": 4}


def test_build_tokens_prompt_preserves_grouped_forwarded_hashes():
    processor = _processor()

    prompt = processor.build_tokens_prompt(
        {
            "token_ids": [1, 2, 3],
            "extra_args": {
                "mm_hashes": ["legacy_image_hash", "legacy_audio_hash"],
                "mm_hashes_by_modality": {
                    "image": ["image_hash"],
                    "audio": ["audio_hash"],
                },
            },
        },
        {"image": object(), "audio": object()},
        None,
    )

    assert prompt["multi_modal_uuids"] == {
        "image": ["image_hash".ljust(64, "0")],
        "audio": ["audio_hash".ljust(64, "0")],
    }


def test_build_tokens_prompt_remaps_grouped_image_hashes_to_vision_chunk():
    processor = _processor(unified_vision_chunk=True)

    prompt = processor.build_tokens_prompt(
        {
            "token_ids": [1, 2, 3],
            "extra_args": {
                "mm_hashes_by_modality": {"image": ["image_hash"]},
            },
        },
        {"vision_chunk": object()},
        None,
    )

    assert prompt["multi_modal_uuids"] == {
        "vision_chunk": ["image_hash".ljust(64, "0")]
    }


def test_build_tokens_prompt_computes_vision_chunk_uuid_without_forwarded_hash():
    processor = _processor(unified_vision_chunk=True)
    image = Image.new("RGB", (1, 1))

    prompt = processor.build_tokens_prompt(
        {"token_ids": [1, 2, 3]},
        {
            "vision_chunk": {
                "type": "image",
                "image": image,
                "uuid": None,
            }
        },
        None,
    )

    assert prompt["multi_modal_uuids"] == {
        "vision_chunk": mod.compute_mm_uuids_from_images([image])
    }


def test_flat_transfer_metadata_fallback_is_image_only():
    extra_args = {
        "mm_hashes": ["image_hash"],
        "mm_placeholders": [(4, 8)],
    }

    assert mod._get_modality_extra_values(
        extra_args,
        "mm_hashes_by_modality",
        "mm_hashes",
        "image",
        "image",
    ) == ["image_hash"]
    assert (
        mod._get_modality_extra_values(
            extra_args,
            "mm_hashes_by_modality",
            "mm_hashes",
            "audio",
            "audio",
        )
        is None
    )


def test_forwarded_placeholder_preserves_is_embed_mask():
    placeholder = mod._placeholder_range_from_extra_arg(
        {"offset": 4, "length": 4, "is_embed": [False, True, True, False]}
    )

    assert placeholder.offset == 4
    assert placeholder.length == 4
    assert placeholder.get_num_embeds() == 2
    assert placeholder.extract_embeds_range() == [(5, 6)]


@pytest.mark.parametrize(
    ("hashes", "expected"),
    [
        ([], []),
        (["0123456789abcdef"], ["0123456789abcdef" + "0" * 48]),
        (["f" * 64], ["f" * 64]),
    ],
)
def test_pad_mm_hashes_to_64(hashes, expected):
    assert mod.pad_mm_hashes_to_64(hashes) == expected


def test_build_tokens_prompt_omits_absent_processor_kwargs():
    prompt = _processor().build_tokens_prompt(
        {"token_ids": [1, 2, 3]},
        None,
        None,
    )

    assert "mm_processor_kwargs" not in prompt


@pytest.mark.asyncio
async def test_aggregated_uses_transferred_prompt_and_falls_back_to_urls():
    processor = _processor()
    transferred = {"type": "multimodal", "prompt_token_ids": [1, 99, 2]}
    processor.try_receive_mm_kwargs = AsyncMock(return_value=transferred)

    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
        },
        "request-4",
        None,
        DisaggregationMode.AGGREGATED,
    )

    assert prepared.prompt is transferred
    processor.image_loader.load_image_batch.assert_not_awaited()

    processor.try_receive_mm_kwargs.return_value = None
    image = Image.new("RGB", (1, 1))
    processor.image_loader.load_image_batch.return_value = [image]
    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
        },
        "request-5",
        None,
        DisaggregationMode.AGGREGATED,
    )

    assert prepared.prompt["multi_modal_data"] == {"image": image}


@pytest.mark.asyncio
async def test_transfer_setup_failure_falls_back_to_raw_media(monkeypatch):
    processor = _processor()
    validate = MagicMock(side_effect=ValueError("invalid metadata"))
    monkeypatch.setattr(mod.MmKwargsShmTransferMetadata, "model_validate", validate)

    assert (
        await processor.try_receive_mm_kwargs(
            {"extra_args": {"mm_kwargs_shm": {"invalid": True}}}
        )
        is None
    )
    validate.assert_called_once()


@pytest.mark.asyncio
async def test_nixl_receiver_initialization_failure_falls_back(monkeypatch):
    processor = _processor()
    monkeypatch.setattr(
        mod.MmKwargsTransferMetadata,
        "model_validate",
        MagicMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        mod,
        "MmKwargsNixlReceiver",
        MagicMock(side_effect=RuntimeError("nixl unavailable")),
    )

    assert (
        await processor.try_receive_mm_kwargs(
            {"extra_args": {"mm_kwargs_nixl": {"modality": "image"}}}
        )
        is None
    )


@pytest.mark.asyncio
async def test_prefill_keeps_raw_media_for_decode_handoff():
    processor = _processor()
    processor.try_receive_mm_kwargs = AsyncMock(
        return_value={"type": "multimodal", "prompt_token_ids": [1, 99, 2]}
    )
    image = Image.new("RGB", (1, 1))
    processor.image_loader.load_image_batch.return_value = [image]

    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
        },
        "request-6",
        None,
        DisaggregationMode.PREFILL,
    )

    processor.try_receive_mm_kwargs.assert_not_awaited()
    assert prepared.multi_modal_data == {"image": image}
    assert prepared.prompt["multi_modal_data"] == {"image": image}


@pytest.mark.asyncio
async def test_qwen_decode_reconstructs_placeholder_embeddings(monkeypatch):
    processor = _processor()
    decode_mm_data = {"image": {"placeholder": object()}}
    monkeypatch.setattr(
        mod,
        "construct_qwen_decode_mm_data",
        lambda grid, shape, request_id: decode_mm_data,
    )

    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
            "prefill_result": {
                "disaggregated_params": {
                    "embedding_params": {
                        "image_grid_thw": [[1, 2, 2]],
                        "embeddings_shape": [1, 16],
                    }
                }
            },
        },
        "request-7",
        None,
        DisaggregationMode.DECODE,
    )

    assert prepared.prompt["multi_modal_data"] is decode_mm_data
    processor.image_loader.load_image_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_qwen_decode_uses_expanded_prompt_tokens():
    processor = _processor(model="llava-hf/llava-1.5-7b-hf")

    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {"image_url": [{"Url": "https://image"}]},
            "prefill_result": {
                "disaggregated_params": {
                    "embedding_params": {"expanded_prompt_token_ids": [1, 99, 99, 2]}
                }
            },
        },
        "request-8",
        None,
        DisaggregationMode.DECODE,
    )

    assert prepared.prompt["prompt_token_ids"] == [1, 99, 99, 2]
    assert prepared.prompt["multi_modal_data"] is None
    processor.image_loader.load_image_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_decode_reloads_video_media():
    processor = _processor()
    video = object()
    processor.video_loader.load_video_batch.return_value = [video]

    prepared = await processor.prepare_prompt(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {
                "video_url": [{"Url": "https://example.com/video.mp4"}]
            },
            "prefill_result": {"disaggregated_params": {}},
        },
        "request-9",
        None,
        DisaggregationMode.DECODE,
    )

    assert prepared.prompt["multi_modal_data"] == {"video": video}
    processor.video_loader.load_video_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_transferred_kwargs_injects_vllm_cache(monkeypatch):
    input_processor = SimpleNamespace(inject_into_mm_cache=MagicMock())
    processor = _processor()
    processor.engine_client = SimpleNamespace(input_processor=input_processor)
    item = MagicMock(spec=mod.MultiModalKwargsItem)
    monkeypatch.setattr(mod.pickle, "loads", lambda payload: item)
    receiver = SimpleNamespace(
        receive=AsyncMock(return_value={"__pickled_kwargs_item__": [b"payload"]})
    )
    metadata = SimpleNamespace(modality="image", mm_hashes=[])

    result = await processor._receive_mm_kwargs(
        {
            "mm_hashes": ["hash"],
            "mm_placeholders": [[1, 2]],
            "expanded_token_ids": [10, 11, 12],
        },
        "shm",
        receiver,
        metadata,
    )

    padded_hash = "hash".ljust(64, "0")
    assert result is not None
    assert result["prompt_token_ids"] == [10, 11, 12]
    assert result["mm_hashes"] == {"image": [padded_hash]}
    input_processor.inject_into_mm_cache.assert_called_once_with(
        {"image": [padded_hash]}, {"image": [item]}
    )


@pytest.mark.asyncio
async def test_receive_transferred_kwargs_uses_grouped_metadata_and_vision_chunk(
    monkeypatch,
):
    input_processor = SimpleNamespace(inject_into_mm_cache=MagicMock())
    processor = _processor(unified_vision_chunk=True)
    processor.engine_client = SimpleNamespace(input_processor=input_processor)
    item = MagicMock(spec=mod.MultiModalKwargsItem)
    monkeypatch.setattr(mod.pickle, "loads", lambda payload: item)
    receiver = SimpleNamespace(
        receive=AsyncMock(return_value={"__pickled_kwargs_item__": [b"payload"]})
    )
    metadata = SimpleNamespace(modality="image", mm_hashes=["metadata_hash"])

    result = await processor._receive_mm_kwargs(
        {
            "mm_hashes_by_modality": {"image": ["grouped_hash"]},
            "mm_placeholders_by_modality": {
                "image": [
                    {
                        "offset": 1,
                        "length": 2,
                        "is_embed": [True, False],
                    }
                ]
            },
            "expanded_token_ids": [10, 11, 12],
        },
        "shm",
        receiver,
        metadata,
    )

    padded_hash = "grouped_hash".ljust(64, "0")
    assert result is not None
    assert result["mm_hashes"] == {"vision_chunk": [padded_hash]}
    placeholder = result["mm_placeholders"]["vision_chunk"][0]
    assert placeholder.get_num_embeds() == 1
    input_processor.inject_into_mm_cache.assert_called_once_with(
        {"vision_chunk": [padded_hash]}, {"vision_chunk": [item]}
    )


@pytest.mark.asyncio
async def test_receive_transferred_kwargs_falls_back_to_metadata_hashes(monkeypatch):
    processor = _processor()
    item = MagicMock(spec=mod.MultiModalKwargsItem)
    monkeypatch.setattr(mod.pickle, "loads", lambda payload: item)
    receiver = SimpleNamespace(
        receive=AsyncMock(return_value={"__pickled_kwargs_item__": [b"payload"]})
    )

    result = await processor._receive_mm_kwargs(
        {
            "mm_placeholders_by_modality": {"video": [(1, 2)]},
            "expanded_token_ids": [10, 11, 12],
        },
        "nixl",
        receiver,
        SimpleNamespace(modality="video", mm_hashes=["metadata_hash"]),
    )

    assert result is not None
    assert result["mm_hashes"] == {"video": ["metadata_hash".ljust(64, "0")]}


@pytest.mark.asyncio
async def test_receive_transferred_kwargs_rejects_partial_feature_transfer(monkeypatch):
    input_processor = SimpleNamespace(inject_into_mm_cache=MagicMock())
    processor = _processor()
    processor.engine_client = SimpleNamespace(input_processor=input_processor)
    item = MagicMock(spec=mod.MultiModalKwargsItem)
    monkeypatch.setattr(mod.pickle, "loads", lambda payload: item)
    receiver = SimpleNamespace(
        receive=AsyncMock(return_value={"__pickled_kwargs_item__": [b"payload"]})
    )

    result = await processor._receive_mm_kwargs(
        {
            "mm_hashes": ["cached_hash", "transferred_hash"],
            "mm_placeholders": [(1, 2), (4, 2)],
            "expanded_token_ids": [10, 11, 12, 13, 14, 15],
        },
        "shm",
        receiver,
        SimpleNamespace(modality="image", mm_hashes=[]),
    )

    assert result is None
    input_processor.inject_into_mm_cache.assert_not_called()


def test_build_prefill_handoff_dispatches_by_model_and_forwards_processor_kwargs(
    monkeypatch,
):
    mm_data = {"image": object()}
    qwen_processor = _processor()
    llava_processor = _processor(model="llava-hf/llava-1.5-7b-hf")
    observed = {}

    def fake_build_qwen(data, params, processor_kwargs):
        observed["processor_kwargs"] = processor_kwargs
        return {"image_grid_thw": [[1, 2, 2]]}

    monkeypatch.setattr(
        mod,
        "build_qwen_embedding_params",
        fake_build_qwen,
    )

    assert qwen_processor.build_prefill_handoff(
        multi_modal_data=mm_data,
        prompt_token_ids=[1, 99, 2],
        mm_processor_kwargs={"max_pixels": 1003520},
    ) == {"image_grid_thw": [[1, 2, 2]]}
    assert observed["processor_kwargs"] == {"max_pixels": 1003520}
    assert llava_processor.build_prefill_handoff(
        multi_modal_data=mm_data,
        prompt_token_ids=[1, 99, 2],
    ) == {"expanded_prompt_token_ids": [1, 99, 2]}


def test_qwen_handoff_applies_per_request_pixel_overrides(monkeypatch):
    from PIL import Image

    base_params = qwen_mod.QwenGridParams(
        patch_size=16,
        merge_size=2,
        factor=32,
        min_pixels=65536,
        max_pixels=16777216,
        vision_hidden_dim=2048,
    )
    captured = {}

    def fake_compute(image_data, params):
        captured["params"] = params
        return [[1, 8, 8]], [16, params.vision_hidden_dim]

    monkeypatch.setattr(qwen_mod, "_compute_qwen_grid_thw", fake_compute)

    result = qwen_mod.build_qwen_embedding_params(
        {"image": Image.new("RGB", (64, 64))},
        base_params,
        {"min_pixels": 1024, "max_pixels": 4096},
    )

    assert result == {
        "image_grid_thw": [[1, 8, 8]],
        "embeddings_shape": [16, 2048],
    }
    assert captured["params"].min_pixels == 1024
    assert captured["params"].max_pixels == 4096


def test_qwen_prefill_handoff_fails_fast_without_grid_metadata(monkeypatch):
    processor = _processor()
    monkeypatch.setattr(
        mod, "load_qwen_grid_params", lambda model, trust_remote_code=False: None
    )

    with pytest.raises(RuntimeError, match="cannot initialize decode mRoPE"):
        processor.initialize_prefill_handoff()


def test_qwen_handoff_computes_grid_for_pil_images():
    from PIL import Image

    result = qwen_mod.build_qwen_embedding_params(
        {"image": Image.new("RGB", (640, 480))},
        qwen_mod.QwenGridParams(
            patch_size=16,
            merge_size=2,
            factor=32,
            min_pixels=65536,
            max_pixels=16777216,
            vision_hidden_dim=2048,
        ),
    )

    assert result == {
        "image_grid_thw": [[1, 30, 40]],
        "embeddings_shape": [300, 2048],
    }


def test_qwen_handoff_accepts_encoder_embeddings():
    import torch

    processor = _processor()
    result = processor.build_prefill_handoff(
        multi_modal_data={
            "image": {
                "image_embeds": torch.randn(1, 256, 1024),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            }
        },
        prompt_token_ids=[1, 2, 3],
    )

    assert result == {
        "image_grid_thw": [[1, 16, 16]],
        "embeddings_shape": [1, 256, 1024],
    }
