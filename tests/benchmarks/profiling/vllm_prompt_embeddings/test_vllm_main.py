# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from benchmarks.profiling.vllm_prompt_embeddings.main import (
    ExperimentConfig,
    create_prompt_embeddings,
    run_experiment,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs


class FakeLLM:
    latest: FakeLLM
    wrong_output = False

    def __init__(self, **kwargs: Any) -> None:
        type(self).latest = self
        self.kwargs = kwargs
        self.events: list[str] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.model_config = SimpleNamespace(
            dtype=torch.float32,
            get_hidden_size=lambda: 8,
            get_inputs_embeds_size=lambda: 8,
        )
        compilation = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
            cudagraph_capture_sizes=kwargs["compilation_config"][
                "cudagraph_capture_sizes"
            ],
        )
        model = SimpleNamespace(
            enforce_eager=kwargs["enforce_eager"],
            hf_config=SimpleNamespace(_commit_hash="revision"),
        )
        cache = SimpleNamespace(
            enable_prefix_caching=kwargs["enable_prefix_caching"],
            block_size=kwargs["block_size"],
        )
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                compilation_config=compilation,
                model_config=model,
                cache_config=cache,
            )
        )

    def generate(
        self, prompts: list[dict[str, torch.Tensor]], params: Any, *, use_tqdm: bool
    ) -> list[Any]:
        self.events.append("generate")
        self.generate_calls.append(
            {"prompts": prompts, "params": params, "use_tqdm": use_tqdm}
        )
        output_tokens = params.values["max_tokens"] - int(self.wrong_output)
        completion = SimpleNamespace(
            token_ids=list(range(output_tokens)), finish_reason="length"
        )
        return [
            SimpleNamespace(
                request_id=f"request-{len(self.generate_calls)}",
                outputs=[completion],
            )
        ]

    def start_profile(self) -> None:
        self.events.append("start_profile")

    def stop_profile(self) -> None:
        self.events.append("stop_profile")


def test_prompt_embeddings_are_deterministic_and_contiguous() -> None:
    first = create_prompt_embeddings(
        prompt_tokens=515,
        hidden_dimension=8,
        dtype=torch.bfloat16,
        seed=7,
    )
    second = create_prompt_embeddings(
        prompt_tokens=515,
        hidden_dimension=8,
        dtype=torch.bfloat16,
        seed=7,
    )
    assert first.shape == (515, 8)
    assert first.dtype == torch.bfloat16
    assert first.device.type == "cpu"
    assert first.is_contiguous()
    assert torch.equal(first, second)


def test_run_experiment_uses_sequential_single_request_calls(tmp_path: Path) -> None:
    config = ExperimentConfig(
        requests=3,
        warmup_requests=2,
        prompt_tokens=35,
        output_tokens=4,
        max_model_len=64,
    )
    summary = run_experiment(
        config,
        tmp_path,
        llm_factory=FakeLLM,
        sampling_params_factory=FakeSamplingParams,
    )
    engine = FakeLLM.latest
    assert engine.events == [
        "generate",
        "generate",
        "start_profile",
        "generate",
        "generate",
        "generate",
        "stop_profile",
    ]
    assert len(engine.generate_calls) == 5
    prompt_tensor = engine.generate_calls[0]["prompts"][0]["prompt_embeds"]
    assert prompt_tensor.shape == (35, 8)
    assert all(len(call["prompts"]) == 1 for call in engine.generate_calls)
    assert all(
        call["prompts"][0]["prompt_embeds"] is prompt_tensor
        for call in engine.generate_calls
    )
    assert all(call["use_tqdm"] is False for call in engine.generate_calls)
    assert engine.generate_calls[0]["params"].values == {
        "temperature": 0.0,
        "max_tokens": 4,
        "min_tokens": 4,
        "ignore_eos": True,
        "seed": 0,
        "detokenize": False,
    }
    assert engine.kwargs["compilation_config"] == {
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1, 3, 35],
    }
    assert engine.kwargs["enforce_eager"] is False
    assert summary["accepted"] is True
    assert summary["results"]["requests"] == 3
    assert (tmp_path / "requests.jsonl").read_text().count("\n") == 3


def test_profile_stops_when_exact_osl_validation_fails(tmp_path: Path) -> None:
    FakeLLM.wrong_output = True
    try:
        with pytest.raises(ValueError, match="OSL mismatch"):
            run_experiment(
                ExperimentConfig(
                    requests=1,
                    warmup_requests=0,
                    prompt_tokens=35,
                    output_tokens=4,
                    max_model_len=64,
                ),
                tmp_path,
                llm_factory=FakeLLM,
                sampling_params_factory=FakeSamplingParams,
            )
        assert FakeLLM.latest.events == [
            "start_profile",
            "generate",
            "stop_profile",
        ]
    finally:
        FakeLLM.wrong_output = False


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ExperimentConfig(requests=0), "requests must be positive"),
        (
            ExperimentConfig(prompt_tokens=1000, output_tokens=75),
            "exceed max_model_len",
        ),
        (
            ExperimentConfig(gpu_memory_utilization=1.0),
            "must be between zero and one",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    config: ExperimentConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()
