# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from dynamo.common.constants import DisaggregationMode
from dynamo.common.multimodal.mm_kwargs_transfer import encode_mm_kwargs_item
from dynamo.vllm.handlers import build_sampling_params
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
                            data=torch.ones(8, 16),
                            field=MultiModalFieldConfig.flat_from_sizes(
                                "video", torch.tensor([8])
                            ).field,
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


@pytest.mark.parametrize("ends_at_boundary", [False, True])
def test_metadata_only_input_has_identical_native_mrope(processed, ends_at_boundary):
    processed["prompt_token_ids"] = [5, 1001, 1004, 1002, 6, 1001, 1004, 1002, 7]
    if ends_at_boundary:
        processed["prompt_token_ids"].pop()
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


def test_placeholder_ending_at_prompt_boundary_roundtrips(processed):
    span = processed["mm_placeholders"]["video"][0]
    processed["mm_placeholders"]["video"][0] = replace(
        span, offset=len(processed["prompt_token_ids"]) - span.length
    )
    restored = restore(mod.QwenVideoHandoff().capture(processed))
    actual = restored["mm_placeholders"]["video"][0]
    assert actual.offset + actual.length == len(restored["prompt_token_ids"])
    start, end = actual.get_embeds_indices_in_range(actual.length - 1, actual.length)
    assert start == end


@pytest.mark.parametrize("mask", [None, [True] * 7])
def test_visual_tail_uses_raw_media_fallback(processed, mask):
    state = mod.QwenVideoHandoff().capture(processed)
    state["mm_placeholders"]["video"][0].update(offset=5, is_embed=mask)
    assert restore(state) is None
    state["metadata"]["video"][0]["timestamps"] = []
    with pytest.raises(ValueError):
        restore(state)


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


@pytest.mark.parametrize("identity", ["x" * 150, "界" * 50])
def test_cache_counts_uuid_bytes_even_when_handoff_is_too_large(
    processed, monkeypatch, identity
):
    monkeypatch.setattr(mod, "_MAX_METADATA_CACHE_BYTES", 256)
    monkeypatch.setattr(mod, "_MAX_HANDOFF_BYTES", 1)
    handoff = mod.QwenVideoHandoff()
    for index in range(5):
        processed["mm_hashes"]["video"] = [f"{identity}-{index}"]
        assert handoff.capture(processed) is None
        expected_bytes = sum(
            len(modality.encode()) + len(uuid.encode()) + len(value)
            for (modality, uuid), value in handoff._metadata.items()
        )
        assert handoff._cache_bytes == expected_bytes <= 256
        assert len(handoff._metadata) == 1


def test_oversized_uuid_is_not_cached(processed, monkeypatch):
    monkeypatch.setattr(mod, "_MAX_METADATA_CACHE_BYTES", 256)
    processed["mm_hashes"]["video"] = ["x" * 256]
    handoff = mod.QwenVideoHandoff()
    assert handoff.capture(processed) is not None
    assert not handoff._metadata and handoff._cache_bytes == 0
    processed["mm_kwargs"]["video"] = [None]
    assert handoff.capture(processed) is None


def test_cache_replacement_counts_key_only_once(processed):
    handoff = mod.QwenVideoHandoff()
    state = handoff.capture(processed)
    fields = state["metadata"]["video"][0]
    key = ("video", state["mm_hashes"]["video"][0])
    fields["timestamps"] = [0.12345, 0.34567]
    for _ in range(3):
        handoff._remember(key, fields)
        expected_bytes = sum(len(part.encode()) for part in key) + len(
            json.dumps(fields, allow_nan=False).encode()
        )
        assert handoff._cache_bytes == expected_bytes
        assert len(handoff._metadata) == 1


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
def transfer_transport(monkeypatch):
    payloads = {}

    async def prepare(features, modality):
        specs = []
        for feature in features:
            key = str(len(payloads))
            payloads[key] = encode_mm_kwargs_item(feature.data)
            specs.append(
                {
                    "field_name": "__pickled_kwargs_item__",
                    "shape": [len(payloads[key])],
                    "dtype_str": "uint8",
                    "serialized_request": key,
                }
            )
        return {
            "mm_kwargs_nixl": {
                "modality": modality,
                "tensor_specs": specs,
                "mm_hashes": [feature.mm_hash for feature in features],
            }
        }, [spec["serialized_request"] for spec in specs]

    async def receive(metadata):
        return {
            "__pickled_kwargs_item__": [
                bytearray(payloads[spec.serialized_request])
                for spec in metadata.tensor_specs
            ]
        }

    sender = SimpleNamespace(
        prepare=AsyncMock(side_effect=prepare), cleanup=AsyncMock()
    )
    receiver = SimpleNamespace(receive=AsyncMock(side_effect=receive))
    monkeypatch.setattr(mod, "MmKwargsNixlSender", lambda: sender)
    monkeypatch.setattr(mod, "MmKwargsNixlReceiver", lambda: receiver)
    return SimpleNamespace(sender=sender, receiver=receiver)


@pytest.mark.parametrize(
    "reason,prompt_logprobs",
    [("prompt_logprobs", 0), ("prompt_logprobs", 1), ("preemption", None)],
)
@pytest.mark.asyncio
async def test_native_recomputation_retains_video_encoder_inputs(
    processed, transfer_transport, reason, prompt_logprobs
):
    owner = mod.QwenVideoHandoff()
    handoff = await owner.prepare(processed)
    restored = await owner.restore_for_decode(
        handoff.state,
        max_model_len=128,
        media_counts={"video": 1},
        spatial_merge_size=2,
    )
    feature = MultiModalFeatureSpec(
        data=restored["mm_kwargs"]["video"][0],
        modality="video",
        identifier=restored["mm_hashes"]["video"][0],
        mm_position=restored["mm_placeholders"]["video"][0],
    )
    params = build_sampling_params(
        {
            "token_ids": restored["prompt_token_ids"],
            "stop_conditions": {"max_tokens": 8},
            "output_options": {"prompt_logprobs": prompt_logprobs}
            if reason == "prompt_logprobs"
            else {},
        },
        {},
    )
    request = Request(
        "recompute", restored["prompt_token_ids"], params, None, mm_features=[feature]
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)
    scheduler.ec_connector = None
    scheduler.aux_output_connector = None
    scheduler.encoder_cache_manager = SimpleNamespace(
        free=MagicMock(),
        get_cached_num_encoder_embeds=MagicMock(return_value=None),
        check_and_update_cache=MagicMock(return_value=False),
        can_allocate=MagicMock(return_value=True),
    )
    scheduler._free_request_blocks = MagicMock()
    scheduler._inflight_prefills = set()
    scheduler.log_stats = False
    scheduler.waiting = MagicMock()
    scheduler.reset_preempted_req_ids = set()
    if reason == "preemption":
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(restored["prompt_token_ids"])
        request.kv_transfer_params = {"do_remote_prefill": False}
        scheduler._preempt_request(request, 0.0)
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        scheduler.encoder_cache_manager.free.assert_called_once_with(request)
    else:
        assert request.get_skip_reading_prefix_cache()
    scheduled, _, _, _, _ = scheduler._try_schedule_encoder_inputs(
        request, request.num_computed_tokens, request.num_tokens, 128
    )
    assert scheduled == [0]
    batched = MultiModalKwargsItems({"video": [request.mm_features[0].data]}).get_data(
        device="cpu"
    )
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.is_multimodal_pruning_enabled = False
    model._process_video_input = MagicMock(return_value=(torch.ones(2, 4),))
    embeddings = model.embed_multimodal(**batched)
    assert embeddings is not None and len(embeddings) == 1
    model._process_video_input.assert_called_once()
    legacy = restore(owner.capture(processed))
    legacy_batch = MultiModalKwargsItems(legacy["mm_kwargs"]).get_data(device="cpu")
    assert model.embed_multimodal(**legacy_batch) is None


def test_cleanup_requires_a_sender_only_for_nonempty_items():
    owner = mod.QwenVideoHandoff()
    owner.release([])
    assert not owner._cleanup_tasks
    with pytest.raises(RuntimeError, match="sender is not initialized"):
        owner.release([object()])


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.asyncio
async def test_transfer_cleanup_does_not_block_prefill_dispatch(
    processed, transfer_transport, cancelled
):
    owner = mod.QwenVideoHandoff()
    handoff = await owner.prepare(processed)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            async with handoff:
                raise asyncio.CancelledError
    else:
        async with handoff:
            transfer_transport.sender.cleanup.assert_not_awaited()
    await asyncio.gather(*owner._cleanup_tasks)
    transfer_transport.sender.cleanup.assert_awaited_once_with(handoff.cleanup_items)


@pytest.mark.asyncio
async def test_unavailable_transport_keeps_legacy_loading(
    processed, transfer_transport
):
    transfer_transport.sender.prepare.return_value = (None, [])
    transfer_transport.sender.prepare.side_effect = None
    assert await mod.QwenVideoHandoff().prepare(processed) is None


@pytest.mark.asyncio
async def test_transfer_hash_mismatch_is_rejected_before_read(
    processed, transfer_transport
):
    owner = mod.QwenVideoHandoff()
    handoff = await owner.prepare(processed)
    handoff.state["inputs"]["video"]["mm_hashes"] = ["wrong-video"]
    with pytest.raises(ValueError, match="items do not match"):
        await owner.restore_for_decode(
            handoff.state,
            max_model_len=128,
            media_counts={"video": 1},
            spatial_merge_size=2,
        )
    transfer_transport.receiver.receive.assert_not_awaited()


@pytest.mark.parametrize("explicit_null", [False, True])
@pytest.mark.asyncio
async def test_received_metadata_without_encoder_inputs_is_rejected(
    processed, transfer_transport, explicit_null
):
    owner = mod.QwenVideoHandoff()
    handoff = await owner.prepare(processed)
    metadata_only = restore(owner.capture(processed))["mm_kwargs"]["video"][0]
    if explicit_null:
        metadata_only["pixel_values_videos"] = MultiModalFieldElem(
            data=None,
            field=processed["mm_kwargs"]["video"][0]["pixel_values_videos"].field,
        )
    transfer_transport.receiver.receive.side_effect = None
    transfer_transport.receiver.receive.return_value = {
        "__pickled_kwargs_item__": [bytearray(encode_mm_kwargs_item(metadata_only))]
    }
    with pytest.raises(ValueError, match="missing encoder inputs"):
        await owner.restore_for_decode(
            handoff.state,
            max_model_len=128,
            media_counts={"video": 1},
            spatial_merge_size=2,
        )


@pytest.fixture
def processor(processed, transfer_transport):
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
    assert state.state["layout"]["prompt_token_ids"] == processed["prompt_token_ids"]
    processor.engine_client.input_processor.renderer.render_cmpl_async.assert_awaited_once_with(
        [prompt], skip_mm_cache=True
    )


@pytest.mark.parametrize("model_type", ["cosmos3", "cosmos3_omni"])
@pytest.mark.asyncio
async def test_cosmos3_uses_the_qwen_video_handoff(processor, processed, model_type):
    processor.engine_client.vllm_config.model_config.hf_config.model_type = model_type
    rendered, state = await processor.prepare_prefill_video_prompt(
        {"prompt_token_ids": [1, 2]}, {"video": object()}
    )
    assert rendered is processed and state is not None


@pytest.mark.parametrize("ends_at_boundary", [False, True])
@pytest.mark.asyncio
async def test_decode_skips_every_media_loader_with_valid_handoff(
    processor, processed, ends_at_boundary
):
    if ends_at_boundary:
        span = processed["mm_placeholders"]["video"][0]
        processed["mm_placeholders"]["video"][0] = replace(
            span, offset=len(processed["prompt_token_ids"]) - span.length
        )
    handoff = await processor._qwen_video_handoff.prepare(processed)
    state = handoff.state
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
    assert torch.equal(
        result.pre_rendered_prompt["mm_kwargs"]["video"][0]["pixel_values_videos"].data,
        processed["mm_kwargs"]["video"][0]["pixel_values_videos"].data,
    )
    processor.video_loader.load_video_batch.assert_not_awaited()
    processor.image_loader.load_image_batch.assert_not_awaited()
    processor.audio_loader.load_audio_batch.assert_not_awaited()


@pytest.mark.parametrize(
    "reason", ["unknown_version", "legacy", "visual_tail", "unmasked_tail"]
)
@pytest.mark.asyncio
async def test_unusable_handoff_falls_back_to_real_video(processor, processed, reason):
    state = mod.QwenVideoHandoff().capture(processed)
    if reason == "unknown_version":
        state["version"] = 3
    elif reason != "legacy":
        state["mm_placeholders"]["video"][0].update(offset=5)
        state["mm_placeholders"]["video"][0]["is_embed"] = (
            [True] * 7 if reason == "visual_tail" else None
        )
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {"video_url": [{"Url": "video"}]},
        "prefill_result": {
            "disaggregated_params": {
                "embedding_params": {mod.QWEN_VIDEO_HANDOFF_KEY: state}
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
