# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest
from dynamo_decision_perf.rendering import (
    Candidate,
    Question,
    prepare_question,
    render_question,
)
from dynamo_decision_perf.workloads import (
    Shape,
    generate_workloads,
    ofat_matrix,
    questions_for,
    wire_payload,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class FakeTokenizer:
    """Character IDs preserve exact prefix semantics; AA is intentionally two tokens."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return messages[0]["content"] + "\nassistant:"


def test_matrix_is_frozen_and_changes_one_axis():
    shapes = ofat_matrix()
    base = shapes[0]
    assert (base.target_tokens, base.questions, base.choices) == (256, 1, 8)
    for shape in shapes[1:7]:
        assert (
            sum(
                getattr(base, key) != getattr(shape, key)
                for key in ("target_tokens", "questions", "choices")
            )
            == 1
        )
    with pytest.raises(FrozenInstanceError):
        base.questions = 2


def test_renderer_preserves_typed_values_and_structured_evidence():
    question = Question(
        "route",
        "choice",
        "Pick.",
        (Candidate(True, "Boolean"), Candidate("true", {"note": "String"})),
    )
    text, labels = render_question([{"role": "user", "content": "evidence"}], question)
    assert (
        text
        == '[{"role":"user","content":"evidence"}]\n\nQuestion: Pick.\nA: true - Boolean\nB: "true" - {"note":"String"}\nAnswer with the letter of one option only.'
    )
    assert labels == ("A", "B")
    with pytest.raises(ValueError, match="description"):
        wire_payload("model", [], (question,), "oai")
    typed = Question(
        "route",
        "choice",
        "Pick.",
        (Candidate(True, "Boolean"), Candidate("true", "String")),
    )
    body = wire_payload("model", "fact", (typed,), "oai")
    assert body["questions"][0]["choices"][0]["value"] is True
    for dialect in ("sglang_native", "systemone"):
        with pytest.raises(ValueError, match="typed"):
            wire_payload("model", [], (question,), dialect)


def test_predicate_score_and_ordered_wire_shapes():
    questions = (
        Question("urgent", "predicate", "Urgent?"),
        Question(
            "impact",
            "score",
            "Impact?",
            (Candidate("0", label="Low"), Candidate("1", label="High")),
        ),
    )
    assert render_question("fact", questions[0]) == (
        "fact\n\nIs the following true? Urgent?\nAnswer with yes or no only.",
        ("yes", "no"),
    )
    assert "0: Low\n1: High" in render_question("fact", questions[1])[0]
    for dialect in ("oai", "sglang_native", "systemone"):
        body = wire_payload("model", "fact", questions, dialect)
        assert len(body["questions"]) == 2
        assert "answers" not in body
    native = wire_payload("model", "fact", questions, "sglang_native")
    assert native["nvext"] == {"format": "sglang_native"}
    assert native["questions"][0]["type"] == "yes_no"
    assert list(wire_payload("model", "fact", questions, "systemone")["questions"]) == [
        "urgent",
        "impact",
    ]


def test_wire_does_not_drop_descriptions():
    predicate = Question(
        "p",
        "predicate",
        "Check",
        (Candidate(True, "Positive"), Candidate(False, "Negative")),
    )
    with pytest.raises(ValueError, match="predicate descriptions"):
        wire_payload("model", "fact", (predicate,), "oai")
    assert (
        wire_payload("model", "fact", (predicate,), "sglang_native")["questions"][0][
            "yes"
        ]
        == "Positive"
    )
    assert wire_payload("model", "fact", (predicate,), "systemone")["questions"]["p"][
        "criteria"
    ] == {"true": "Positive", "false": "Negative"}
    score = Question(
        "s",
        "score",
        "Rate",
        (Candidate("0", "Minor", "Low"), Candidate("1", "Major", "High")),
    )
    assert wire_payload("model", "fact", (score,), "sglang_native")["questions"][0][
        "levels"
    ] == ["Low - Minor", "High - Major"]
    assert wire_payload("model", "fact", (score,), "systemone")["questions"]["s"][
        "criteria"
    ] == ["Low - Minor", "High - Major"]


def test_blank_instructions_match_core():
    question = Question("c", "choice", {}, (Candidate("a"), Candidate("b")))
    assert "Question:" not in render_question("fact", question)[0]
    with pytest.raises(ValueError, match="instructions"):
        wire_payload("model", "fact", (question,), "oai")


def test_question_fanout_has_distinct_prompts_not_only_distinct_ids():
    questions = questions_for(Shape("fanout", questions=4, choices=2))
    prepared = [
        prepare_question(FakeTokenizer(), "shared evidence", q) for q in questions
    ]
    assert len({p.input_ids for p in prepared}) == 4
    assert all(p.content.startswith("shared evidence\n") for p in prepared)


def test_candidate_suffix_and_context_checks():
    question = Question(
        "q", "choice", "Pick.", tuple(Candidate(str(i)) for i in range(32))
    )
    with pytest.raises(ValueError, match="one distinct token"):
        prepare_question(FakeTokenizer(), "x", question)
    with pytest.raises(ValueError, match="context"):
        prepare_question(
            FakeTokenizer(),
            "x" * 2048,
            Question("q", "choice", "Pick.", (Candidate("a"), Candidate("b"))),
        )


def test_generation_is_deterministic_manifested_and_never_overwrites(tmp_path):
    shape = Shape("tiny", 256, 1, 2)
    for name in ("a", "b"):
        generate_workloads(
            tmp_path / name, FakeTokenizer(), "model", seed=17, count=2, shapes=(shape,)
        )
    a, b = tmp_path / "a", tmp_path / "b"
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }
    manifest = json.loads((a / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == digest
    rows = [
        entry["payloads"][0]
        for entry in json.loads((a / "tiny.native_score.json").read_text())["data"]
    ]
    assert all("cache_salt" not in row for row in rows)
    assert rows[0]["sampling_params"]["max_new_tokens"] == 0
    metadata = json.loads((a / "tiny.metadata.json").read_text())
    assert all(row["expanded_tokens"] == sum(row["prompt_tokens"]) for row in metadata)
    assert all(abs(row["prompt_tokens"][0] - 256) <= 32 for row in metadata)
    with pytest.raises(FileExistsError):
        generate_workloads(a, FakeTokenizer(), "model", shapes=(shape,))


def test_unsupported_shape_is_explicit_and_has_no_wire_rows(tmp_path):
    output = tmp_path / "out"
    generate_workloads(
        output, FakeTokenizer(), "model", count=1, shapes=(Shape("wide", 1024, 1, 32),)
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["unsupported"][0]["shape"] == "wide"
    assert "one distinct token" in manifest["unsupported"][0]["reason"]
    assert not list(output.glob("*.oai.json"))


@pytest.mark.parametrize(
    "kwargs",
    [{"target_tokens": 2049}, {"questions": 0}, {"choices": 1}, {"name": "../escape"}],
)
def test_invalid_shapes_rejected(kwargs):
    with pytest.raises(ValueError):
        Shape(**{"name": "base", **kwargs})
