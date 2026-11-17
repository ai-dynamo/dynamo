# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.common.guided_output import explicit_gate_capability, has_named_parameter

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return {"<think>": [4], "</think>": [5]}[text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return {(4,): "<think>", (5,): "</think>"}.get(tuple(ids), "")


def test_capability_matches_exact_gate_ids_and_serializes_a_versioned_contract() -> (
    None
):
    capability = explicit_gate_capability(
        "sglang", "qwen3", Tokenizer(), "<think>", "</think>", [5], True
    )
    value = json.loads(capability.to_json())
    assert value["version"] == 1
    assert value["mode"] == "explicit_gate"
    assert value["end_token_ids"] == [5]
    assert value["end_text"] == "</think>"
    assert value["request_controls"] is True


@pytest.mark.parametrize("ids", [[], [163588, 39964, 163589], [6]])
def test_unverified_or_non_lossless_gate_is_not_advertised(ids: list[int]) -> None:
    capability = explicit_gate_capability(
        "sglang", "kimi_k3", Tokenizer(), "<think>", "</think>", ids, True
    )
    assert capability.mode == "unsupported"


def test_kwargs_is_not_evidence_for_request_local_reasoning_controls() -> None:
    def named(*, reasoning_ended: bool | None = None) -> None:
        pass

    def catch_all(**kwargs: object) -> None:
        pass

    assert has_named_parameter(named, "reasoning_ended")
    assert not has_named_parameter(catch_all, "reasoning_ended")
    capability = explicit_gate_capability(
        "vllm", "step3", Tokenizer(), "<think>", "</think>", [5], False
    )
    assert capability.mode == "unsupported"
