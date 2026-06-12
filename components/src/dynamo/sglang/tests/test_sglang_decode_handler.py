# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dynamo.common.metadata_upload import MetadataUploader
from dynamo.sglang.request_handlers.llm.decode_handler import (
    DecodeWorkerHandler,
    _extract_sglang_stop_reason,
    _nvext_extra_field_requested,
    _openai_stop_sampling_params,
    _user_stop_token_ids,
)
from dynamo.sglang.request_handlers.llm.mm_disagg_utils import (
    extract_media_urls,
    raise_if_unextracted_multimodal,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import StreamProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _read_zstd_payload(path):
    import zstandard as zstd

    raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
    import msgspec

    return msgspec.msgpack.decode(raw)


def test_extract_media_urls_supports_string_and_wire_items():
    mm_data = {
        "video_url": [
            "file:///tmp/test.mp4",
            {"Url": "https://example.com/test.mp4"},
        ]
    }

    assert extract_media_urls(mm_data, "video_url") == [
        "file:///tmp/test.mp4",
        "https://example.com/test.mp4",
    ]


def test_extract_media_urls_returns_none_for_missing_modality():
    assert extract_media_urls({}, "image_url") is None
    assert extract_media_urls(None, "image_url") is None
    assert extract_media_urls({"image_url": []}, "image_url") is None


def test_extract_media_urls_rejects_malformed_payloads():
    # A bare string (not a list) would otherwise split into per-character items.
    with pytest.raises(ValueError, match="must be a list"):
        extract_media_urls({"image_url": "https://example.com/a.png"}, "image_url")

    # Frontend-decoded media is URL-passthrough only in the disaggregated path.
    with pytest.raises(ValueError, match="Frontend-decoded"):
        extract_media_urls({"image_url": [{"Decoded": "..."}]}, "image_url")

    # Unsupported dict variant fails clearly instead of degrading to text.
    with pytest.raises(ValueError, match="Unsupported"):
        extract_media_urls({"image_url": [{"ignored": "value"}]}, "image_url")


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ({"type": "stop", "matched": "END"}, "END"),
        ({"type": "stop", "matched": 128001}, 128001),
        ({"type": "stop", "matched": [128001, 128009]}, [128001, 128009]),
        ({"type": "stop", "matched": True}, None),
        ({"type": "stop", "matched": ["END"]}, None),
        ({"type": "length"}, None),
        (None, None),
    ],
)
def test_extract_sglang_stop_reason(finish_reason, expected):
    assert _extract_sglang_stop_reason(finish_reason) == expected


def test_extract_sglang_stop_reason_filters_hidden_token_ids():
    finish_reason = {"type": "stop", "matched": 128001}

    assert _extract_sglang_stop_reason(finish_reason, {576}) is None
    assert _extract_sglang_stop_reason(finish_reason, {128001}) == 128001


def test_extract_sglang_stop_reason_filters_hidden_token_id_arrays():
    finish_reason = {"type": "stop", "matched": [128001, 128009]}

    assert _extract_sglang_stop_reason(finish_reason, {128001}) is None
    assert _extract_sglang_stop_reason(finish_reason, {128001, 128009}) == [
        128001,
        128009,
    ]


def test_user_stop_token_ids_ignores_hidden_ids():
    assert _user_stop_token_ids(
        {
            "stop_conditions": {
                "stop_token_ids": [576],
                "stop_token_ids_hidden": [128001],
            }
        }
    ) == {576}


def test_user_stop_token_ids_handles_null_fields():
    assert _user_stop_token_ids({"stop_conditions": {"stop_token_ids": None}}) == set()
    assert _user_stop_token_ids({"stop_token_ids": None}) == set()


def test_user_stop_token_ids_accepts_stop_token_id_array():
    assert _user_stop_token_ids({"stop": [32, 34]}) == {32, 34}


def test_user_stop_token_ids_treats_token_id_display_as_string_stop():
    assert _user_stop_token_ids({"stop": ["token_id:576"]}) == set()


def test_openai_stop_sampling_params_preserves_string_stops():
    assert _openai_stop_sampling_params({"stop": "END"}) == {"stop": "END"}
    assert _openai_stop_sampling_params({"stop": ["END"]}) == {"stop": ["END"]}
    assert _openai_stop_sampling_params({"stop": ["token_id:576"]}) == {
        "stop": ["token_id:576"]
    }


def test_openai_stop_sampling_params_maps_token_id_stop_array():
    assert _openai_stop_sampling_params({"stop": [32, 34]}) == {
        "stop_token_ids": [32, 34]
    }
    assert _openai_stop_sampling_params({"stop_token_ids": [32, 34]}) == {
        "stop_token_ids": [32, 34]
    }


def _new_decode_handler(*, use_sglang_tokenizer: bool = False, enable_rl: bool = False):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.use_sglang_tokenizer = use_sglang_tokenizer
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model"),
        dynamo_args=SimpleNamespace(enable_rl=enable_rl),
    )

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor
    return handler


async def _stream(items):
    for item in items:
        yield item


class _Context:
    def is_stopped(self):
        return False


def test_build_sampling_params_passes_n_for_token_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=False)

    sampling_params = handler._build_sampling_params(
        {
            "sampling_options": {"temperature": 0.2, "top_p": 0.9, "n": 3},
            "stop_conditions": {"max_tokens": 8},
        }
    )

    assert sampling_params["n"] == 3
    assert sampling_params["temperature"] == 0.2
    assert sampling_params["max_new_tokens"] == 8


def test_build_sampling_params_forwards_repetition_controls_for_token_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=False)

    sampling_params = handler._build_sampling_params(
        {
            "sampling_options": {
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "repetition_penalty": 1.05,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
                "min_p": 0.01,
                "seed": 1234,
            },
            "stop_conditions": {"max_tokens": 8},
        }
    )

    assert sampling_params["presence_penalty"] == 0.1
    assert sampling_params["frequency_penalty"] == 0.2
    assert sampling_params["repetition_penalty"] == 1.05
    assert sampling_params["temperature"] == 1.0
    assert sampling_params["top_p"] == 0.95
    assert sampling_params["top_k"] == 40
    assert sampling_params["min_p"] == 0.01
    assert sampling_params["sampling_seed"] == 1234
    assert "seed" not in sampling_params


def test_build_sampling_params_passes_n_for_sglang_tokenizer_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=True)

    sampling_params = handler._build_sampling_params(
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "n": 2,
            "max_tokens": 8,
            "stop": [32, 34],
        }
    )

    assert sampling_params["n"] == 2
    assert sampling_params["temperature"] == 0.2
    assert sampling_params["max_new_tokens"] == 8
    assert sampling_params["stop_token_ids"] == [32, 34]


def test_build_sampling_params_forwards_repetition_controls_for_openai_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=True)

    sampling_params = handler._build_sampling_params(
        {
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.05,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.01,
            "seed": 1234,
            "max_tokens": 8,
        }
    )

    assert sampling_params["presence_penalty"] == 0.1
    assert sampling_params["frequency_penalty"] == 0.2
    assert sampling_params["repetition_penalty"] == 1.05
    assert sampling_params["temperature"] == 1.0
    assert sampling_params["top_p"] == 0.95
    assert sampling_params["top_k"] == 40
    assert sampling_params["min_p"] == 0.01
    assert sampling_params["sampling_seed"] == 1234
    assert "seed" not in sampling_params


class TestMultimodalGuard:
    """Tests for multimodal guard when frontend extraction is missing."""

    IMAGE_MESSAGE = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
            {"type": "text", "text": "describe image"},
        ],
    }

    @pytest.mark.parametrize(
        "request_factory",
        [
            lambda msg: {"token_ids": [1, 2, 3], "messages": [msg]},
            lambda msg: {"token_ids": [1, 2, 3], "extra_args": {"messages": [msg]}},
        ],
        ids=["top_level_messages", "extra_args_messages"],
    )
    def test_raises_for_image_url(self, request_factory):
        with pytest.raises(RuntimeError, match="multi_modal_data"):
            raise_if_unextracted_multimodal(request_factory(self.IMAGE_MESSAGE))

    def test_text_only_request_bypasses_guard(self):
        raise_if_unextracted_multimodal({"token_ids": [10, 20, 30]})


def test_build_logprob_kwargs_allows_chosen_token_logprobs(monkeypatch):
    monkeypatch.delenv("DYN_SGL_ALLOW_TOP_LOGPROBS", raising=False)

    kwargs = DecodeWorkerHandler._build_logprob_kwargs(
        {"output_options": {"logprobs": 0}}
    )

    assert kwargs == {"return_logprob": True, "top_logprobs_num": 0}


def test_build_logprob_kwargs_rejects_top_logprobs_by_default(monkeypatch):
    monkeypatch.delenv("DYN_SGL_ALLOW_TOP_LOGPROBS", raising=False)

    with pytest.raises(ValueError, match="does not currently support logprobs >= 1"):
        DecodeWorkerHandler._build_logprob_kwargs({"output_options": {"logprobs": 1}})


def test_build_logprob_kwargs_allows_top_logprobs_with_escape_hatch(monkeypatch):
    monkeypatch.setenv("DYN_SGL_ALLOW_TOP_LOGPROBS", "1")

    kwargs = DecodeWorkerHandler._build_logprob_kwargs(
        {"output_options": {"logprobs": 2}}
    )

    assert kwargs == {"return_logprob": True, "top_logprobs_num": 2}


def test_extract_logprobs_formats_top_tokens_as_token_ids():
    log_probs, top_logprobs, total = DecodeWorkerHandler._extract_logprobs(
        {
            "output_token_logprobs": [(-0.1, 101, "a")],
            "output_top_logprobs": [[(-0.1, 101, "a"), (-0.2, 102, "b")]],
        },
        0,
        return_tokens_as_token_ids=True,
    )

    assert log_probs == [-0.1]
    assert top_logprobs == [
        [
            {"rank": 1, "token_id": 101, "token": "token_id:101", "logprob": -0.1},
            {"rank": 2, "token_id": 102, "token": "token_id:102", "logprob": -0.2},
        ]
    ]
    assert total == 1


def test_metadata_uploader_parses_extra_args_nvext():
    uploader = MetadataUploader.from_backend_request(
        {
            "extra_args": {
                "nvext": {
                    "metadata_upload": {
                        "url": "s3://bucket/root/rollouts",
                    }
                }
            }
        }
    )

    assert uploader is not None
    assert uploader.url == "s3://bucket/root/rollouts"

    uploader = MetadataUploader.from_backend_request(
        {"nvext": {"metadata_upload": {"url": "s3://bucket/root/rollouts"}}}
    )
    assert uploader is not None


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({}, "metadata_upload.url is required"),
        ({"url": ""}, "metadata_upload.url must not be empty"),
        ({"url": 1}, "metadata_upload.url must be a string"),
        ("invalid", "metadata_upload must be an object"),
        (
            {"url": "s3://bucket/root/rollouts", "format": "json"},
            "metadata_upload.format is not supported",
        ),
    ],
)
def test_metadata_uploader_rejects_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        MetadataUploader.from_backend_request({"nvext": {"metadata_upload": settings}})


def test_nvext_extra_field_requested_supports_raw_and_preprocessed_shapes():
    assert _nvext_extra_field_requested(
        {"nvext": {"extra_fields": ["stop_reason"]}}, "stop_reason"
    )
    assert _nvext_extra_field_requested(
        {"extra_args": {"nvext": {"extra_fields": ["stop_reason"]}}},
        "stop_reason",
    )
    assert not _nvext_extra_field_requested(
        {"extra_args": {"nvext": {"extra_fields": ["engine_data"]}}},
        "stop_reason",
    )


def test_metadata_upload_requires_enable_rl():
    request = {
        "extra_args": {
            "nvext": {"metadata_upload": {"url": "s3://bucket/root/rollouts/run-1"}}
        }
    }

    assert _new_decode_handler()._metadata_uploader_from_request(request) is None

    uploader = _new_decode_handler(enable_rl=True)._metadata_uploader_from_request(
        request
    )
    assert uploader is not None
    assert uploader.url == "s3://bucket/root/rollouts/run-1"


@pytest.mark.asyncio
async def test_metadata_upload_normalizes_tensor_values(tmp_path):
    torch = pytest.importorskip("torch")
    uploader = MetadataUploader(url=(tmp_path / "metadata/tensors").as_uri())
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    await uploader.upload_choice(0, {"router_tensor": tensor})

    payload = _read_zstd_payload(tmp_path / "metadata/tensors/choice_0.msgpack.zst")
    uploaded_tensor = payload["metadata"]["router_tensor"]
    assert uploaded_tensor["type"] == "tensor"
    assert uploaded_tensor["dtype"] == "int32"
    assert uploaded_tensor["shape"] == [2, 2]
    assert uploaded_tensor["data"] == tensor.numpy().tobytes()

    bfloat_tensor = torch.tensor([1, 2], dtype=torch.bfloat16)
    await uploader.upload_choice(1, {"router_tensor": bfloat_tensor})

    payload = _read_zstd_payload(tmp_path / "metadata/tensors/choice_1.msgpack.zst")
    uploaded_tensor = payload["metadata"]["router_tensor"]
    assert uploaded_tensor["dtype"] == "bfloat16"
    assert uploaded_tensor["shape"] == [2]
    assert uploaded_tensor["data"] == bfloat_tensor.view(torch.uint8).numpy().tobytes()


@pytest.mark.asyncio
async def test_metadata_upload_normalizes_numpy_values(tmp_path):
    np = pytest.importorskip("numpy")
    uploader = MetadataUploader(url=(tmp_path / "metadata/arrays").as_uri())
    array = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    await uploader.upload_choice(0, {"scores": array[:, ::-1]})

    payload = _read_zstd_payload(tmp_path / "metadata/arrays/choice_0.msgpack.zst")
    uploaded_array = payload["metadata"]["scores"]
    expected = np.ascontiguousarray(array[:, ::-1])
    assert uploaded_array["type"] == "ndarray"
    assert uploaded_array["dtype"] == "float32"
    assert uploaded_array["shape"] == [2, 2]
    assert uploaded_array["data"] == expected.tobytes()


@pytest.mark.asyncio
async def test_process_token_stream_tracks_logprobs_per_choice_index():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [(-0.1, 101, "a")],
                        },
                    },
                    {
                        "index": 1,
                        "output_ids": [201],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [(-0.2, 201, "b")],
                        },
                    },
                    {
                        "index": 0,
                        "output_ids": [102],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [
                                (-0.1, 101, "a"),
                                (-0.3, 102, "c"),
                            ],
                        },
                    },
                ]
            ),
            _Context(),
        )
    )

    assert [chunk["index"] for chunk in chunks] == [0, 1, 0]
    assert [chunk["token_ids"] for chunk in chunks] == [[101], [201], [102]]
    assert [chunk["log_probs"] for chunk in chunks] == [[-0.1], [-0.2], [-0.3]]


@pytest.mark.asyncio
async def test_process_token_stream_uploads_large_metadata(tmp_path):
    handler = _new_decode_handler()
    uploader = MetadataUploader(
        url=(tmp_path / "metadata/rollout-7").as_uri(),
    )
    meta_info = {
        "id": "sglang-1",
        "finish_reason": {"type": "stop"},
        "output_token_logprobs": [(-0.1, 101, "a")],
        "output_top_logprobs": [[(-0.1, 101, "a"), (-0.2, 102, "b")]],
        "routed_experts": b"expert-bytes",
        "custom_router_state": {"step": 1, "scores": [0.25, 0.75]},
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "cached_tokens": 0,
    }

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "meta_info": meta_info,
                    }
                ]
            ),
            _Context(),
            metadata_uploader=uploader,
        )
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "log_probs" not in chunk
    assert "top_logprobs" not in chunk
    assert "disaggregated_params" not in chunk
    assert "engine_data" not in chunk
    uploaded_path = tmp_path / "metadata/rollout-7/choice_0.msgpack.zst"

    payload = _read_zstd_payload(uploaded_path)
    assert payload["metadata"]["id"] == "sglang-1"
    assert payload["metadata"]["finish_reason"] == {"type": "stop"}
    assert payload["metadata"]["output_token_logprobs"] == [[-0.1, 101, "a"]]
    assert payload["metadata"]["output_top_logprobs"][0][1][1] == 102
    assert payload["metadata"]["routed_experts"] == b"expert-bytes"
    assert payload["metadata"]["custom_router_state"] == {
        "step": 1,
        "scores": [0.25, 0.75],
    }
    assert payload["metadata"]["prompt_tokens"] == 2
    assert meta_info == {}


@pytest.mark.asyncio
async def test_process_token_stream_uploads_only_final_meta_info(tmp_path):
    handler = _new_decode_handler()
    uploader = MetadataUploader(url=(tmp_path / "metadata/rollout-final").as_uri())
    first_meta_info = {
        "id": "sglang-1",
        "finish_reason": None,
        "output_token_logprobs": [(-0.1, 101, "a")],
        "custom_router_state": {"step": 1},
    }
    final_meta_info = {
        "id": "sglang-1",
        "finish_reason": {"type": "stop"},
        "output_token_logprobs": [(-0.1, 101, "a"), (-0.2, 102, "b")],
        "custom_router_state": {"step": 2},
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "cached_tokens": 0,
    }

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {"index": 0, "output_ids": [101], "meta_info": first_meta_info},
                    {"index": 0, "output_ids": [102], "meta_info": final_meta_info},
                ]
            ),
            _Context(),
            metadata_uploader=uploader,
        )
    )

    assert len(chunks) == 2
    assert "log_probs" not in chunks[0]
    assert "log_probs" not in chunks[1]
    payload = _read_zstd_payload(
        tmp_path / "metadata/rollout-final/choice_0.msgpack.zst"
    )
    assert payload["metadata"]["output_token_logprobs"] == [
        [-0.1, 101, "a"],
        [-0.2, 102, "b"],
    ]
    assert payload["metadata"]["custom_router_state"] == {"step": 2}
    assert first_meta_info == {}
    assert final_meta_info == {}


@pytest.mark.asyncio
async def test_process_token_stream_upload_failure_blocks_final_chunk(tmp_path):
    handler = _new_decode_handler()

    class FailingUploader(MetadataUploader):
        async def upload_choice(self, choice_index, metadata):
            raise RuntimeError("metadata upload failed")

    with pytest.raises(RuntimeError, match="metadata upload failed"):
        await _collect(
            handler._process_token_stream(
                _stream(
                    [
                        {
                            "index": 0,
                            "output_ids": [101],
                            "meta_info": {
                                "id": "sglang-1",
                                "finish_reason": {"type": "stop"},
                                "output_token_logprobs": [(-0.1, 101, "a")],
                                "prompt_tokens": 2,
                                "completion_tokens": 1,
                                "cached_tokens": 0,
                            },
                        }
                    ]
                ),
                _Context(),
                metadata_uploader=FailingUploader(
                    url=(tmp_path / "metadata/rollout-fail").as_uri(),
                ),
            )
        )


@pytest.mark.asyncio
async def test_process_text_stream_tracks_delta_per_choice_index():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "He",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 1,
                        "text": "Go",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 1,
                        "text": "Good",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                ]
            ),
            _Context(),
        )
    )

    choices = [chunk["choices"][0] for chunk in chunks]
    assert [choice["index"] for choice in choices] == [0, 1, 0, 1]
    assert [choice["delta"]["content"] for choice in choices] == [
        "He",
        "Go",
        "llo",
        "od",
    ]


@pytest.mark.asyncio
async def test_process_text_stream_stop_reason_uses_response_nvext():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": {"type": "stop", "matched": "END"},
                        },
                    }
                ]
            ),
            _Context(),
            request={"nvext": {"extra_fields": ["stop_reason"]}},
        )
    )

    assert "stop_reason" not in chunks[0]["choices"][0]
    assert chunks[0]["nvext"]["stop_reason"] == "END"


@pytest.mark.asyncio
async def test_process_text_stream_stop_reason_requires_nvext_extra_field():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": {"type": "stop", "matched": "END"},
                        },
                    }
                ]
            ),
            _Context(),
        )
    )

    assert "stop_reason" not in chunks[0]["choices"][0]
    assert "nvext" not in chunks[0]


@pytest.mark.asyncio
async def test_process_text_stream_uploads_routed_experts(tmp_path):
    handler = _new_decode_handler(use_sglang_tokenizer=True)
    uploader = MetadataUploader(url=(tmp_path / "metadata/rollout-8").as_uri())
    meta_info = {
        "id": "sglang-2",
        "finish_reason": {"type": "stop"},
        "routed_experts": "base64-experts",
    }

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": meta_info,
                    }
                ]
            ),
            _Context(),
            metadata_uploader=uploader,
        )
    )

    assert "nvext" not in chunks[0]
    payload = _read_zstd_payload(tmp_path / "metadata/rollout-8/choice_0.msgpack.zst")
    assert payload["metadata"]["id"] == "sglang-2"
    assert payload["metadata"]["finish_reason"] == {"type": "stop"}
    assert payload["metadata"]["routed_experts"] == "base64-experts"
    assert meta_info == {}


@pytest.mark.asyncio
async def test_process_token_stream_suppresses_hidden_stop_token_reason():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [128001],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": {"type": "stop", "matched": 128001},
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "cached_tokens": None,
                        },
                    }
                ]
            ),
            _Context(),
            user_stop_token_ids={576},
        )
    )

    assert "stop_reason" not in chunks[0]


@pytest.mark.asyncio
async def test_multimodal_stream_keeps_reading_after_one_choice_finishes():
    chunks = await _collect(
        StreamProcessor.process_sglang_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "text": "a",
                        "meta_info": {"finish_reason": None},
                    },
                    {
                        "index": 1,
                        "output_ids": [201],
                        "text": "b",
                        "meta_info": {"finish_reason": None},
                    },
                    {
                        "index": 0,
                        "output_ids": [],
                        "text": "a",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    },
                    {
                        "index": 1,
                        "output_ids": [],
                        "text": "b",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    },
                ]
            )
        )
    )

    outputs = [json.loads(chunk) for chunk in chunks]

    assert [output["index"] for output in outputs] == [0, 1, 0, 1]
    assert [output["finished"] for output in outputs] == [False, False, True, True]
    assert [output.get("finish_reason") for output in outputs] == [
        None,
        None,
        "stop",
        "stop",
    ]


async def _collect(stream):
    return [item async for item in stream]
