# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned native-head preparation fixtures, distinct from mock adapter contracts."""

from types import SimpleNamespace

import pytest

from dynamo.vllm import decision_heads as heads
from dynamo.vllm import decision_native_registry as native_registry
from dynamo.vllm.tests.decision_head_test_support import Engine, adapter, question, wire

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.mark.asyncio
async def test_missing_native_d3_codes_becomes_optional_runtime_error(
    pinned_preparation,
):
    engine = Engine()
    engine.model_config.hf_config = SimpleNamespace(decision_config={})
    with pytest.raises(heads.NativeRuntimeError, match="initialize"):
        await adapter(engine)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_exact_pinned_d3_prompt_and_order(pinned_preparation):
    engine = Engine()
    engine.model_config.hf_config = SimpleNamespace(
        decision_config={"codes": ["A", "B", "C"], "temperature": 2.0}
    )
    instance = await adapter(engine)
    prepared = await instance._prepare(
        wire(questions=[question(kind="Predicate", values=(True, False))])
    )
    prompt = bytes(prepared.reads[0].prompt_ids).decode()
    assert (
        "State:\nstate\n\nQuestion:\nChoose.\n\nOptions:\nA: No / false\nB: Yes / true"
        in prompt
    )
    assert prompt.endswith("Reply with only the code of the best option.")
    assert prepared.reads[0].orders == ((1, 0),)
    assert "client-name" not in prompt


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "profile,architecture",
    [
        (native_registry.CLEF_FLASH_PROFILE, "ClefFlashForJointSchema"),
        (native_registry.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema"),
    ],
)
async def test_exact_pinned_clef_joint_schema_and_state_truncation(
    pinned_preparation, profile, architecture
):
    engine = Engine(architecture)
    engine.model_config.max_model_len = 1200
    instance = await adapter(engine, profile)
    questions = [question(), question(1, "Predicate", (True, False))]
    request = wire(profile, questions)
    request["decision_head"]["input"] = "x" * 3000
    prepared = await instance._prepare(request)
    read = prepared.reads[0]
    schema = read.pooling_params.extra_kwargs["joint_schema"]
    assert len(read.prompt_ids) == 1200
    assert schema["question_types"] == [1, 0]
    assert schema["option_counts"] == [2, 2]
    assert read.orders == ((1, 0), (0, 1))
    prompt = bytes(read.prompt_ids).decode()
    assert "JOINT SCHEMA DECISIONS:" in prompt
    assert "ID: question-0" in prompt and "ID: question-1" in prompt
    for start, end in [*schema["question_spans"], *schema["option_spans"]]:
        assert -len(read.prompt_ids) <= start < end <= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,architecture",
    [
        (native_registry.CLEF_FLASH_PROFILE, "ClefFlashForJointSchema"),
        (native_registry.CLEF_OMNI_PROFILE, "ClefOmniForJointSchema"),
    ],
)
async def test_pinned_clef_blank_instructions_fall_back_to_canonical_question_id(
    pinned_preparation, profile, architecture
):
    instance = await adapter(Engine(architecture), profile)
    item = {**question(), "id": "caller-supplied-id", "instructions": ""}
    prepared = await instance._prepare(wire(profile, [item]))
    read = prepared.reads[0]
    prompt = bytes(read.prompt_ids).decode()
    assert "ID: question-0\n" in prompt
    start, end = read.pooling_params.extra_kwargs["joint_schema"]["question_spans"][0]
    assert bytes(read.prompt_ids[start:end]).decode() == "question-0"
    assert "caller-supplied-id" not in prompt
