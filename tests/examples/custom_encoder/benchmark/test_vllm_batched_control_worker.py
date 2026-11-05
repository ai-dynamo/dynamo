# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import (
    Preprocessed,
    VisionEncoderBackend,
)
from examples.custom_encoder.benchmark import batched_control_worker as control

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeBackend(VisionEncoderBackend[str, str, str]):
    image_token_id = 151655
    max_batch_cost = 4
    max_batch_items = 2

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def build(self, model_id: str) -> None:
        self.events.append(("build", model_id))

    def preprocess(self, raw: str) -> Preprocessed[str]:
        self.events.append(("preprocess", raw))
        return Preprocessed(item=raw, cost=1, bucket_key=raw[0])

    def forward_batch(
        self, items: list[str], target_bucket: int | None = None
    ) -> list[str]:
        self.events.append(("forward", list(items)))
        return [f"artifact:{item}" for item in items]


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[str]]] = []

    def prepare_prompt(self, token_ids: list[int], artifacts: list[str]) -> object:
        self.calls.append((token_ids, artifacts))
        return {"token_ids": token_ids, "artifacts": artifacts}


class _FakeLlm:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.calls: list[tuple[list[object], list[object], bool]] = []

    def generate(
        self,
        prompts: list[object],
        sampling_params: list[object],
        *,
        use_tqdm: bool,
    ) -> list[object]:
        self.backend.events.append(("generate", len(prompts)))
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    SimpleNamespace(
                        token_ids=[index + 10, index + 20],
                        index=0,
                        finish_reason="length",
                    )
                ],
            )
            for index in range(len(prompts))
        ]


def _request(token_id: int) -> dict[str, Any]:
    return {
        "token_ids": [token_id],
        "stop_conditions": {"max_tokens": 2},
        "multi_modal_data": {"image_url": [{"Url": f"a-{token_id}"}]},
    }


def _engine() -> control.BatchedControlEngine:
    return control.BatchedControlEngine(
        model_name="decoder",
        served_model_name="decoder",
        engine_args=SimpleNamespace(enable_prompt_embeds=True),
        encoder_backend_type=_FakeBackend,
        control_max_batch_items=8,
        control_max_queue_delay_us=1_000,
    )


def test_pipeline_preprocesses_then_batches_vision_then_generates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine()
    backend = _FakeBackend()
    adapter = _FakeAdapter()
    llm = _FakeLlm(backend)
    engine._backend = backend
    engine._adapter = adapter
    engine._llm = llm
    engine._model_max_len = 2048
    engine._default_sampling_params = {}
    monkeypatch.setattr(
        control,
        "build_sampling_params",
        lambda request, defaults, model_max_len: SimpleNamespace(),
    )
    work_items = [
        control._ControlRequest(_request(1), "a-one"),
        control._ControlRequest(_request(2), "b-one"),
        control._ControlRequest(_request(3), "a-two"),
    ]

    chunks = engine._run_pipeline_batch(work_items)

    assert backend.events == [
        ("preprocess", "a-one"),
        ("preprocess", "b-one"),
        ("preprocess", "a-two"),
        ("forward", ["a-one", "a-two"]),
        ("forward", ["b-one"]),
        ("generate", 3),
    ]
    assert adapter.calls == [
        ([1], ["artifact:a-one"]),
        ([2], ["artifact:b-one"]),
        ([3], ["artifact:a-two"]),
    ]
    assert len(llm.calls) == 1
    assert llm.calls[0][2] is False
    assert [batch[0]["token_ids"] for batch in chunks] == [
        [10, 20],
        [11, 21],
        [12, 22],
    ]
    assert all(
        params.output_kind is control.RequestOutputKind.FINAL_ONLY
        for params in llm.calls[0][1]
    )


def test_output_translation_uses_expanded_prompt_length() -> None:
    output = SimpleNamespace(
        prompt_token_ids=list(range(9)),
        outputs=[
            SimpleNamespace(
                token_ids=(5, 6, 7),
                index=0,
                finish_reason="length",
            )
        ],
    )

    chunks = control.BatchedControlEngine._request_output_to_chunks(
        {"token_ids": [1]}, output
    )

    assert chunks == [
        {
            "token_ids": [5, 6, 7],
            "index": 0,
            "finish_reason": "length",
            "completion_usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12,
            },
        }
    ]


def test_single_image_validation_rejects_missing_or_extra_modalities() -> None:
    valid = {
        "token_ids": [1],
        "multi_modal_data": {"image_url": [{"Url": "data:image/jpeg;base64,x"}]},
    }
    assert control.BatchedControlEngine._single_image_url(valid).endswith(",x")

    with pytest.raises(InvalidArgument, match="exactly one image"):
        control.BatchedControlEngine._single_image_url({"token_ids": [1]})
    with pytest.raises(InvalidArgument, match="unsupported multimodal"):
        control.BatchedControlEngine._single_image_url(
            {
                "token_ids": [1],
                "multi_modal_data": {"audio_url": [{"Url": "audio"}]},
            }
        )
