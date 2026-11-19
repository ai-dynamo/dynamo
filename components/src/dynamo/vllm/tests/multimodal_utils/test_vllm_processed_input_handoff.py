# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
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
from dynamo.vllm.multimodal_utils import processed_input_handoff as mod
from dynamo.vllm.multimodal_utils.request_processor import (
    MissingMultimodalHandoffError,
    VllmMultimodalRequestProcessor,
)
from tests.serve.launch import vllm_video_pd_worker
from tests.serve.launch.vllm_video_pd_worker import require_video_inputs

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


@pytest.fixture
def transport(monkeypatch):
    payloads = {}

    async def prepare(features, modality):
        specs = []
        for feature in features:
            key = str(len(payloads))
            payloads[key] = encode_mm_kwargs_item(feature.data)
            specs.append(
                dict(
                    field_name="__pickled_kwargs_item__",
                    shape=[len(payloads[key])],
                    dtype_str="uint8",
                    serialized_request=key,
                )
            )
        return {
            "mm_kwargs_nixl": dict(
                modality=modality,
                tensor_specs=specs,
                mm_hashes=[f.mm_hash for f in features],
            )
        }, [s["serialized_request"] for s in specs]

    async def receive(metadata):
        return {
            "__pickled_kwargs_item__": [
                bytearray(payloads[s.serialized_request]) for s in metadata.tensor_specs
            ]
        }

    sender = SimpleNamespace(
        prepare=AsyncMock(side_effect=prepare), cleanup=AsyncMock()
    )
    receiver = SimpleNamespace(receive=AsyncMock(side_effect=receive))
    monkeypatch.setattr(mod, "MmKwargsNixlSender", lambda: sender)
    monkeypatch.setattr(mod, "MmKwargsNixlReceiver", lambda: receiver)
    return SimpleNamespace(sender=sender, receiver=receiver)


async def roundtrip(processed, media_counts=None):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    return await owner.restore_for_decode(
        json.loads(json.dumps(handoff.state)),
        max_model_len=128,
        media_counts=media_counts or {"video": 1},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("custom_fields", [False, True])
async def test_roundtrip_preserves_complete_native_inputs(
    processed, transport, tail, custom_fields
):
    if custom_fields:
        item = processed["mm_kwargs"]["video"][0]
        item["future_model_pixels"] = item.pop("pixel_values_videos")
        item.pop("video_grid_thw")
        item.pop("timestamps")
    if tail:
        processed["mm_placeholders"]["video"][0] = PlaceholderRange(offset=5, length=7)
    restored = await roundtrip(processed)
    assert restored["prompt_token_ids"] == processed["prompt_token_ids"]
    assert restored["mm_hashes"] == processed["mm_hashes"]
    actual, expected = (p["mm_placeholders"]["video"][0] for p in (restored, processed))
    assert (actual.offset, actual.length) == (expected.offset, expected.length)
    if expected.is_embed is None:
        assert actual.is_embed is None
    else:
        assert torch.equal(actual.is_embed, expected.is_embed)
    actual, expected = (p["mm_kwargs"]["video"][0] for p in (restored, processed))
    assert set(actual) == set(expected)
    for name, field in expected.items():
        assert type(actual[name].field) is type(field.field)
        if isinstance(field.data, torch.Tensor):
            assert actual[name].data.dtype == field.data.dtype
            assert torch.equal(actual[name].data, field.data)
        else:
            assert actual[name].data == field.data


@pytest.mark.asyncio
async def test_multiple_videos_and_mixed_image_preserve_every_item(
    processed, transport
):
    processed["prompt_token_ids"] = list(range(30))
    processed["mm_hashes"]["video"].append("video-two")
    processed["mm_kwargs"]["video"].append(
        copy.deepcopy(processed["mm_kwargs"]["video"][0])
    )
    processed["mm_kwargs"]["video"][1]["pixel_values_videos"].data.mul_(2)
    processed["mm_kwargs"]["video"][1]["timestamps"].data = [0.5, 0.7]
    processed["mm_placeholders"]["video"].append(PlaceholderRange(offset=12, length=7))
    processed["mm_hashes"]["image"] = ["image-one"]
    processed["mm_placeholders"]["image"] = [PlaceholderRange(offset=22, length=1)]
    processed["mm_kwargs"]["image"] = [
        MultiModalKwargsItem(
            {
                "image_pixels": MultiModalFieldElem(
                    data=torch.ones(1, 4),
                    field=MultiModalFieldConfig.batched("image").field,
                )
            }
        )
    ]
    restored = await roundtrip(processed, {"video": 2, "image": 1})
    assert restored["mm_hashes"] == processed["mm_hashes"]
    assert [p.offset for p in restored["mm_placeholders"]["video"]] == [2, 12]
    assert torch.equal(
        restored["mm_kwargs"]["video"][0]["pixel_values_videos"].data,
        torch.ones(8, 16),
    )
    assert torch.equal(
        restored["mm_kwargs"]["video"][1]["pixel_values_videos"].data,
        torch.full((8, 16), 2.0),
    )
    assert restored["mm_kwargs"]["video"][1]["timestamps"].data == [0.5, 0.7]
    assert torch.equal(
        restored["mm_kwargs"]["image"][0]["image_pixels"].data, torch.ones(1, 4)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 1, 2, 4, "3"])
async def test_unknown_or_metadata_only_version_uses_raw_media(version, transport):
    assert (
        await mod.ProcessedInputHandoff().restore_for_decode(
            {"version": version}, max_model_len=128, media_counts={"video": 1}
        )
        is None
    )
    transport.receiver.receive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt_token_ids", []),
        ("prompt_token_ids", [True]),
        ("prompt_token_ids", [-1]),
        ("prompt_token_ids", list(range(129))),
        ("mm_hashes", {"video": [""]}),
        ("mm_hashes", {"video": [None]}),
        ("field_names", {"video": [[]]}),
        ("field_names", {"video": [["x", "x"]]}),
    ],
)
async def test_invalid_layout_is_rejected_before_read(processed, transport, key, value):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    handoff.state["layout"][key] = value
    with pytest.raises(ValueError):
        await owner.restore_for_decode(
            handoff.state, max_model_len=128, media_counts={"video": 1}
        )
    transport.receiver.receive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        dict(offset=-1),
        dict(length=100),
        dict(is_embed=[True]),
        dict(offset=True),
        dict(is_embed=[1] * 7),
    ],
)
async def test_invalid_placeholder_is_rejected(processed, transport, changes):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    handoff.state["layout"]["mm_placeholders"]["video"][0].update(changes)
    with pytest.raises(ValueError):
        await owner.restore_for_decode(
            handoff.state, max_model_len=128, media_counts={"video": 1}
        )
    transport.receiver.receive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["count", "hash", "modality", "overlap"])
async def test_mismatched_transfer_is_rejected_before_read(processed, transport, case):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    counts = {"video": 1}
    if case == "count":
        counts["video"] = 2
    elif case == "hash":
        handoff.state["inputs"]["video"]["mm_hashes"] = ["wrong"]
    elif case == "modality":
        handoff.state["inputs"]["video"]["modality"] = "audio"
    else:
        layout = handoff.state["layout"]
        for name in ("mm_hashes", "mm_placeholders", "field_names"):
            layout[name]["video"] *= 2
        handoff.state["inputs"]["video"]["mm_hashes"] *= 2
        handoff.state["inputs"]["video"]["tensor_specs"] *= 2
        counts["video"] = 2
    with pytest.raises(ValueError):
        await owner.restore_for_decode(
            handoff.state, max_model_len=128, media_counts=counts
        )
    transport.receiver.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_dropped_native_field_cannot_be_submitted(processed, transport):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    item = copy.deepcopy(processed["mm_kwargs"]["video"][0])
    del item["pixel_values_videos"]
    transport.receiver.receive.side_effect = None
    transport.receiver.receive.return_value = {
        "__pickled_kwargs_item__": [bytearray(encode_mm_kwargs_item(item))]
    }
    with pytest.raises(ValueError, match="missing native fields"):
        await owner.restore_for_decode(
            handoff.state, max_model_len=128, media_counts={"video": 1}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("item", [None, {}])
async def test_incomplete_cached_items_fall_back(processed, transport, item):
    processed["mm_kwargs"]["video"] = [item]
    assert await mod.ProcessedInputHandoff().prepare(processed) is None
    transport.sender.prepare.assert_not_awaited()


@pytest.mark.asyncio
async def test_size_limit_and_transport_unavailable_keep_raw_path(
    processed, transport, monkeypatch
):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    monkeypatch.setattr(mod, "_MAX_HANDOFF_BYTES", 1)
    assert await owner.prepare(processed) is None
    with pytest.raises(ValueError, match="metadata limit"):
        await owner.restore_for_decode(
            handoff.state, max_model_len=128, media_counts={"video": 1}
        )
    monkeypatch.setattr(mod, "_MAX_HANDOFF_BYTES", 2 * 1024 * 1024)
    transport.sender.prepare.side_effect = None
    transport.sender.prepare.return_value = (None, [])
    assert await owner.prepare(processed) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_cleanup_does_not_block_decode_dispatch(processed, transport, cancelled):
    owner = mod.ProcessedInputHandoff()
    owner.release([])
    handoff = await owner.prepare(processed)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            async with handoff:
                raise asyncio.CancelledError
    else:
        async with handoff:
            transport.sender.cleanup.assert_not_awaited()
    await asyncio.gather(*owner._cleanup_tasks)
    transport.sender.cleanup.assert_awaited_once_with(handoff.cleanup_items)


@pytest.fixture
def processor(processed, transport):
    engine = SimpleNamespace(
        input_processor=SimpleNamespace(
            renderer=SimpleNamespace(
                render_cmpl_async=AsyncMock(return_value=[processed])
            )
        ),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
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
@pytest.mark.parametrize(
    "model_type",
    [
        "qwen3_vl",
        "qwen3_vl_moe",
        "cosmos3",
        "cosmos3_omni",
        "qwen3_5",
        "qwen3_5_moe",
        "future_video_model",
    ],
)
async def test_native_handoff_does_not_depend_on_model_name(
    processor, processed, model_type
):
    processor.engine_client.vllm_config.model_config.hf_config = SimpleNamespace(
        model_type=model_type
    )
    rendered, handoff = await processor.prepare_prefill_video_prompt(
        {"prompt_token_ids": [1, 2]}, {"video": object()}
    )
    assert rendered is processed and handoff is not None
    processor.engine_client.input_processor.renderer.render_cmpl_async.assert_awaited_once_with(
        [{"prompt_token_ids": [1, 2]}], skip_mm_cache=True
    )
    result = await processor.prepare_input(
        {
            "token_ids": [1, 2],
            "multi_modal_data": {"video_url": [{"Url": "must-not-fetch"}]},
            "prefill_result": {
                "disaggregated_params": {
                    "embedding_params": {mod.VIDEO_HANDOFF_KEY: handoff.state}
                }
            },
        },
        "decode",
        None,
        DisaggregationMode.DECODE,
    )
    assert (
        result.pre_rendered_prompt["prompt_token_ids"] == processed["prompt_token_ids"]
    )
    processor.video_loader.load_video_batch.assert_not_awaited()
    processor.image_loader.load_image_batch.assert_not_awaited()
    processor.audio_loader.load_audio_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_known_state_fails_closed(processor, processed):
    handoff = await processor._video_handoff.prepare(processed)
    handoff.state["layout"]["prompt_token_ids"] = []
    with pytest.raises(MissingMultimodalHandoffError):
        await processor.prepare_input(
            {
                "token_ids": [1, 2],
                "multi_modal_data": {"video_url": [{"Url": "must-not-fetch"}]},
                "prefill_result": {
                    "disaggregated_params": {
                        "embedding_params": {mod.VIDEO_HANDOFF_KEY: handoff.state}
                    }
                },
            },
            "decode",
            None,
            DisaggregationMode.DECODE,
        )
    processor.video_loader.load_video_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_and_image_only_keep_existing_path(processor):
    prompt = {"prompt_token_ids": [1, 2]}
    for data in ({"video": object(), "audio": object()}, {"image": object()}):
        rendered, handoff = await processor.prepare_prefill_video_prompt(prompt, data)
        assert rendered is prompt and handoff is None
    processor.engine_client.input_processor.renderer.render_cmpl_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,prompt_logprobs",
    [("prompt_logprobs", 0), ("prompt_logprobs", 1), ("preemption", None)],
)
async def test_native_recomputation_retains_encoder_inputs(
    processed, transport, reason, prompt_logprobs
):
    restored = await roundtrip(processed)
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
        assert (
            request.status == RequestStatus.PREEMPTED
            and request.num_computed_tokens == 0
        )
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
    assert len(model.embed_multimodal(**batched)) == 1
    model._process_video_input.assert_called_once()


@pytest.mark.asyncio
async def test_receive_timeout_fails_instead_of_reloading(
    processed, transport, monkeypatch
):
    owner = mod.ProcessedInputHandoff()
    handoff = await owner.prepare(processed)
    monkeypatch.setattr(mod, "MM_NIXL_CLEANUP_TIMEOUT_S", 0)
    with pytest.raises(TimeoutError):
        await owner.restore_for_decode(
            handoff.state, max_model_len=128, media_counts={"video": 1}
        )


@pytest.mark.asyncio
async def test_partial_registration_is_cleaned_without_blocking(processed, transport):
    owner = mod.ProcessedInputHandoff()
    transport.sender.prepare.side_effect = None
    transport.sender.prepare.return_value = (None, ["partial-registration"])
    assert await owner.prepare(processed) is None
    await asyncio.gather(*owner._cleanup_tasks)
    transport.sender.cleanup.assert_awaited_once_with(["partial-registration"])


@pytest.mark.asyncio
async def test_native_mrope_is_unchanged(processed, transport):
    processed["prompt_token_ids"] = [5, 1001, 1004, 1002, 6, 1001, 1004, 1002, 7]
    processed["mm_placeholders"]["video"] = [
        PlaceholderRange(
            offset=1,
            length=7,
            is_embed=torch.tensor([False, True, False, False, False, True, False]),
        )
    ]
    restored = await roundtrip(processed)
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


@pytest.mark.asyncio
async def test_unknown_handoff_version_loads_real_media(processor):
    request = {
        "token_ids": [1, 2],
        "multi_modal_data": {"video_url": [{"Url": "real-video"}]},
        "prefill_result": {
            "disaggregated_params": {
                "embedding_params": {mod.VIDEO_HANDOFF_KEY: {"version": 4}}
            }
        },
    }
    result = await processor.prepare_input(
        request, "decode", None, DisaggregationMode.DECODE
    )
    assert result.pre_rendered_prompt is None and "video" in result.multi_modal_data
    processor.video_loader.load_video_batch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("salt", ["", "tenant-specific-prefix-cache"])
async def test_preserves_native_cache_salt_and_prompt(processed, transport, salt):
    processed["cache_salt"] = salt
    processed["prompt"] = "original prompt"
    restored = await roundtrip(processed)
    assert restored["cache_salt"] == salt
    assert restored["prompt"] == "original prompt"


@pytest.mark.asyncio
async def test_encoder_decoder_input_keeps_native_path(processed, transport):
    processed["encoder_prompt_token_ids"] = [1, 2]
    assert await mod.ProcessedInputHandoff().prepare(processed) is None
    transport.sender.prepare.assert_not_awaited()


def test_ci_guard_accepts_complete_native_video(processed):
    require_video_inputs(SimpleNamespace(pre_rendered_prompt=processed))


@pytest.mark.parametrize(
    "missing", ["prompt", "item", "pixels", "null_pixels", "empty_pixels"]
)
def test_ci_guard_rejects_missing_or_metadata_only_video(processed, missing):
    if missing == "prompt":
        processed = None
    elif missing == "item":
        processed["mm_kwargs"]["video"] = []
    elif missing == "pixels":
        processed["mm_kwargs"]["video"][0].pop("pixel_values_videos")
    else:
        processed["mm_kwargs"]["video"][0]["pixel_values_videos"].data = (
            None if missing == "null_pixels" else torch.empty(0)
        )
    with pytest.raises(RuntimeError):
        require_video_inputs(SimpleNamespace(pre_rendered_prompt=processed))


@pytest.mark.parametrize("stage", ["load", "process"])
def test_ci_worker_fails_if_decode_reloads_or_reprocesses(monkeypatch, stage):
    monkeypatch.setattr("sys.argv", ["worker", "--disaggregation-mode", "decode"])

    def run_worker(*args, **kwargs):
        if stage == "load":
            asyncio.run(vllm_video_pd_worker.VideoLoader.load_video_batch(object(), []))
        else:
            vllm_video_pd_worker.BaseRenderer._process_multimodal(object(), {})

    monkeypatch.setattr(vllm_video_pd_worker.runpy, "run_module", run_worker)
    with pytest.raises(RuntimeError, match="Decode must not"):
        vllm_video_pd_worker.main()
