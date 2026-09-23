# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.skipif(
        importlib.util.find_spec("vllm") is None,
        reason="vllm not installed in this container",
    ),
]


def _request(*, sampling_params=None, features=None, token_ids=None, **envelope):
    payload = {
        "request_id": "request-1",
        "sampling_params": sampling_params or {},
        **envelope,
    }
    if features is not None:
        payload["features"] = features
    return {
        "model": "test-model",
        "token_ids": token_ids or [11, 22, 33],
        "extra_args": {"vllm_tito": payload},
    }


def _vllm_config(*, max_num_seqs=8, max_model_len=128):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            generation_config="vllm",
            override_generation_config={},
        ),
    )


def test_tito_adapter_uses_outer_tokens_and_rl_sampling_defaults():
    from vllm.sampling_params import RequestOutputKind

    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    request = _request(
        token_ids=[41, 42],
        sampling_params={
            "temperature": 0.7,
            "top_k": 8,
            "max_tokens": 5,
            "routed_experts_prompt_start": 1,
        },
    )
    request["extra_args"]["vllm_tito"]["token_ids"] = [999]

    adapted = adapt_engine_generate_request(
        request,
        enable_multimodal=False,
        decode_capable=True,
        vllm_config=_vllm_config(),
        default_sampling_params={},
    )

    assert adapted is not None
    assert adapted.prompt["prompt_token_ids"] == [41, 42]
    assert adapted.sampling_params.temperature == pytest.approx(0.7)
    assert adapted.sampling_params.top_k == 8
    assert adapted.sampling_params.max_tokens == 5
    assert adapted.sampling_params.routed_experts_prompt_start == 1
    assert adapted.sampling_params.detokenize is False
    assert adapted.sampling_params.output_kind is RequestOutputKind.DELTA


def test_tito_adapter_preserves_kv_transfer_params_in_sampling_extra_args():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    request = _request(
        sampling_params={"max_tokens": 5, "extra_args": {"existing": "value"}},
        kv_transfer_params={"connector_data": {"block_ids": [1, 2]}},
    )
    request["kv_hint"] = {"source": "worker-a"}
    adapted = adapt_engine_generate_request(
        request,
        enable_multimodal=False,
        decode_capable=True,
        vllm_config=_vllm_config(),
        default_sampling_params={},
    )

    assert adapted is not None
    assert adapted.sampling_params.extra_args == {
        "existing": "value",
        "kv_transfer_params": {
            "connector_data": {"block_ids": [1, 2]},
            "kv_hint": {"source": "worker-a"},
        },
    }


@pytest.mark.parametrize("prompt_start", [True, 1.0, -1])
def test_tito_adapter_rejects_invalid_routed_experts_prompt_start(prompt_start):
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    with pytest.raises(ValueError, match="routed_experts_prompt_start"):
        adapt_engine_generate_request(
            _request(
                sampling_params={
                    "max_tokens": 1,
                    "routed_experts_prompt_start": prompt_start,
                }
            ),
            enable_multimodal=False,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


@pytest.mark.parametrize("prompt_start", [3, 99])
def test_tito_adapter_accepts_prompt_length_or_larger_routed_experts_start(
    prompt_start,
):
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    adapted = adapt_engine_generate_request(
        _request(
            sampling_params={
                "max_tokens": 1,
                "routed_experts_prompt_start": prompt_start,
            }
        ),
        enable_multimodal=False,
        decode_capable=True,
        vllm_config=_vllm_config(),
        default_sampling_params={},
    )

    assert adapted is not None
    assert adapted.sampling_params.routed_experts_prompt_start == prompt_start


def test_tito_adapter_rejects_nonprogressing_guided_json_cycle():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    schema = {
        "$defs": {"A": {"allOf": [{"$ref": "#/$defs/A"}]}},
        "$ref": "#/$defs/A",
    }
    with pytest.raises(ValueError, match=r"non-progressing local \$ref cycle"):
        adapt_engine_generate_request(
            _request(
                sampling_params={
                    "max_tokens": 1,
                    "structured_outputs": {"json": schema},
                }
            ),
            enable_multimodal=False,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


def test_tito_adapter_rejects_stop_strings_but_preserves_stop_token_ids():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    with pytest.raises(ValueError, match="stop strings"):
        adapt_engine_generate_request(
            _request(sampling_params={"max_tokens": 1, "stop": ["END"]}),
            enable_multimodal=False,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )

    adapted = adapt_engine_generate_request(
        _request(sampling_params={"max_tokens": 1, "stop_token_ids": [42]}),
        enable_multimodal=False,
        decode_capable=True,
        vllm_config=_vllm_config(),
        default_sampling_params={},
    )
    assert adapted is not None
    assert adapted.sampling_params.stop_token_ids == [42]


def test_tito_adapter_builds_preprocessed_image_input_without_reprocessing():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    features = {
        "mm_hashes": {"image": ["renderer-hash"]},
        "mm_placeholders": {
            "image": [
                {
                    "offset": 1,
                    "length": 2,
                    "is_embed": [False, True],
                }
            ]
        },
        "kwargs_data": None,
    }
    request = _request(features=features)
    request["extra_args"]["dynamo_mm_routing_hashes"] = ["a" * 16 + "0" * 48]

    adapted = adapt_engine_generate_request(
        request,
        enable_multimodal=True,
        decode_capable=True,
        vllm_config=_vllm_config(),
        default_sampling_params={},
    )

    assert adapted is not None
    assert adapted.prompt["type"] == "multimodal"
    assert adapted.prompt["prompt_token_ids"] == [11, 22, 33]
    assert adapted.prompt["mm_hashes"] == {"image": ["a" * 16 + "0" * 48]}
    assert adapted.prompt["mm_kwargs"]["image"] == [None]
    placeholder = adapted.prompt["mm_placeholders"]["image"][0]
    assert (placeholder.offset, placeholder.length) == (1, 2)
    assert placeholder.is_embed.tolist() == [False, True]


def test_tito_adapter_rejects_preprocessed_features_on_disaggregated_decode():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    with pytest.raises(ValueError, match="aggregated vLLM worker"):
        adapt_engine_generate_request(
            _request(
                features={
                    "mm_hashes": {"image": ["renderer-hash"]},
                    "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
                    "kwargs_data": None,
                }
            ),
            enable_multimodal=True,
            decode_capable=True,
            allow_multimodal_features=False,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


@pytest.mark.parametrize(
    ("enable_multimodal", "decode_capable", "features", "match"),
    [
        (
            False,
            True,
            {
                "mm_hashes": {"image": ["x"]},
                "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
            },
            "multimodal",
        ),
        (True, False, None, "aggregated or decode"),
        (
            True,
            True,
            {
                "mm_hashes": {"audio": ["x"]},
                "mm_placeholders": {"audio": [{"offset": 0, "length": 1}]},
            },
            "image",
        ),
    ],
)
def test_tito_adapter_rejects_unsupported_execution_paths(
    enable_multimodal, decode_capable, features, match
):
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    with pytest.raises(ValueError, match=match):
        adapt_engine_generate_request(
            _request(features=features),
            enable_multimodal=enable_multimodal,
            decode_capable=decode_capable,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


@pytest.mark.parametrize(
    "features",
    [
        {"mm_hashes": {}, "mm_placeholders": {"image": []}},
        {"mm_hashes": {"image": []}, "mm_placeholders": {}},
    ],
)
def test_tito_adapter_rejects_asymmetric_image_feature_objects(features):
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    with pytest.raises(TypeError, match="hashes and placeholders must be lists"):
        adapt_engine_generate_request(
            _request(features=features, sampling_params={"max_tokens": 1}),
            enable_multimodal=True,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


def test_tito_adapter_rejects_routing_hash_count_mismatch():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    request = _request(
        features={
            "mm_hashes": {"image": ["one"]},
            "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
            "kwargs_data": None,
        }
    )
    request["extra_args"]["dynamo_mm_routing_hashes"] = ["one", "two"]

    with pytest.raises(ValueError, match="routing hash"):
        adapt_engine_generate_request(
            request,
            enable_multimodal=True,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


def test_tito_adapter_rejects_non_object_kwargs_data():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    request = _request(
        features={
            "mm_hashes": {"image": ["one"]},
            "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
            "kwargs_data": [],
        }
    )

    with pytest.raises(TypeError, match="kwargs_data must be an object or null"):
        adapt_engine_generate_request(
            request,
            enable_multimodal=True,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


@pytest.mark.parametrize(
    ("is_embed", "error", "match"),
    [
        ({"unexpected": True}, TypeError, "sequence"),
        ([False, True], ValueError, "placeholder length"),
        ([1], ValueError, "booleans"),
    ],
)
def test_tito_adapter_rejects_invalid_placeholder_mask_before_tensor_conversion(
    is_embed, error, match
):
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    request = _request(
        token_ids=[11],
        features={
            "mm_hashes": {"image": ["one"]},
            "mm_placeholders": {
                "image": [{"offset": 0, "length": 1, "is_embed": is_embed}]
            },
            "kwargs_data": None,
        },
    )

    with pytest.raises(error, match=match):
        adapt_engine_generate_request(
            request,
            enable_multimodal=True,
            decode_capable=True,
            vllm_config=_vllm_config(),
            default_sampling_params={},
        )


def test_tito_adapter_rejects_sampling_choices_above_scheduler_capacity():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    with pytest.raises(ValueError, match="max_num_seqs"):
        adapt_engine_generate_request(
            _request(sampling_params={"n": 5, "max_tokens": 4}),
            enable_multimodal=False,
            decode_capable=True,
            vllm_config=_vllm_config(max_num_seqs=4),
            default_sampling_params={},
        )


def test_tito_adapter_resolves_omitted_max_tokens_from_server_limits():
    from dynamo.vllm.engine_generate import adapt_engine_generate_request

    adapted = adapt_engine_generate_request(
        _request(token_ids=[1, 2, 3], sampling_params={}),
        enable_multimodal=False,
        decode_capable=True,
        vllm_config=_vllm_config(max_model_len=20),
        default_sampling_params={},
    )

    assert adapted is not None
    assert adapted.sampling_params.max_tokens == 17
