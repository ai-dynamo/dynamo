# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned prompt mechanics distinguish canonical questions from upstream adapters."""

import ast
import json
from types import SimpleNamespace

import pytest

from dynamo.vllm.decision_native_wire import _question
from dynamo.vllm.tests.decision_head_test_support import Engine, adapter, question

pytest_plugins = ["dynamo.vllm.tests.decision_pinned_fixtures"]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def _upstream_adapter(pinned_source, questions):
    tree = ast.parse(pinned_source("vllm/entrypoints/openai/decisions/adapters.py"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_read_question"
    ]
    # Shells select the actual pinned adapter branches, not upstream request validation.
    predicate = type("PredicateQuestion", (SimpleNamespace,), {})
    choice = type("ChoiceQuestion", (SimpleNamespace,), {})
    namespace = {
        **vars(questions),
        "json": json,
        "PredicateQuestion": predicate,
        "ChoiceQuestion": choice,
        "LabelQuestion": questions.ChoiceQuestion,
        "DecisionQuestion": object,
    }
    exec(compile(tree, "pinned-openai-question-adapter", "exec"), namespace)
    return namespace["make_read_question"], predicate, choice


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["Predicate", "Choice", "Score"])
async def test_canonical_question_prompts_intentionally_differ_from_upstream_openai(
    pinned_source, pinned_preparation, kind
):
    engine = Engine()
    engine.model_config.hf_config = SimpleNamespace(
        decision_config={"codes": ["A", "B", "C"], "temperature": 2.0}
    )
    instance = await adapter(engine)
    make_question, predicate, choice = _upstream_adapter(
        pinned_source, instance.native.questions
    )
    if kind == "Predicate":
        reference = predicate(instructions="Decide.")
        raw = question(kind=kind, values=(True, False))
    elif kind == "Choice":
        reference = choice(
            instructions="Decide.",
            choices=[
                SimpleNamespace(value=value, description="")
                for value in ("red", "blue")
            ],
        )
        raw = question(kind=kind, values=("red", "blue"))
    else:
        reference = SimpleNamespace(
            instructions="Decide.",
            levels=[
                SimpleNamespace(label="low", description="poor"),
                SimpleNamespace(label="high", description="good"),
            ],
        )
        raw = question(kind=kind, values=("0", "1"))
        raw["candidates"] = [
            {"value": str(i), "label": level.label, "description": level.description}
            for i, level in enumerate(reference.levels)
        ]
    raw["instructions"] = "Decide."
    upstream = make_question(0, reference)
    canonical = _question(raw, 0, instance.native, 26)
    upstream_prompt = instance.strategy._prompt("state", upstream, [])
    canonical_prompt = instance.strategy._prompt("state", canonical, [])
    assert upstream_prompt != canonical_prompt
    if kind == "Predicate":
        assert upstream.type.name == "choice" and canonical.type.name == "noul"
        assert [option.name for option in upstream.options] == ["false", "true"]
        assert [option.name for option in canonical.options] == ["true", "false"]
        assert "A: false\nB: true" in upstream_prompt
        assert "A: No / false\nB: Yes / true" in canonical_prompt
        assert instance.strategy._options(canonical)[1] == [1, 0]
    elif kind == "Choice":
        assert [option.name for option in upstream.options] == ['"red"', '"blue"']
        assert [option.name for option in canonical.options] == ["red", "blue"]
    else:
        assert upstream.type.name == "choice" and canonical.type.name == "score"
        assert [(option.name, option.description) for option in upstream.options] == [
            ("low", "poor"),
            ("high", "good"),
        ]
        assert [(option.name, option.description) for option in canonical.options] == [
            ("low - poor", None),
            ("high - good", None),
        ]
        assert "A: low: poor" in upstream_prompt and "A: low - poor" in canonical_prompt
