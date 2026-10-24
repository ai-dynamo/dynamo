# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-format 1 mirror of dynamo-decisions at the manifest's pinned revision."""

import json
from dataclasses import dataclass
from typing import Any


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def text(value: Any) -> str:
    return "" if value is None else value if isinstance(value, str) else compact(value)


def is_blank(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, (dict, list)) and not value)
    )


def score_level(candidate: "Candidate") -> str:
    description = text(candidate.description)
    if candidate.label is None:
        return description
    return candidate.label + (f" - {description}" if description else "")


@dataclass(frozen=True)
class Candidate:
    value: str | bool
    description: Any = None
    label: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    instructions: Any = None
    candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class PreparedQuestion:
    content: str
    prompt: str
    labels: tuple[str, ...]
    input_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]


def render_question(evidence: Any, question: Question) -> tuple[str, tuple[str, ...]]:
    instruction = "" if is_blank(question.instructions) else text(question.instructions)
    candidates = question.candidates
    if question.kind == "predicate":
        labels = ("yes", "no")
        lines = [
            f"Is the following true? {instruction}"
            if instruction
            else "Is the following true?"
        ]
        for label, candidate in zip(labels, candidates):
            if not is_blank(candidate.description):
                lines.append(f"{label}: {text(candidate.description)}")
        lines.append("Answer with yes or no only.")
    elif question.kind in ("choice", "score"):
        count = len(candidates)
        if not 1 <= count <= (10 if question.kind == "score" else 255):
            raise ValueError("question exceeds prompt label capacity")
        labels = tuple(
            str(i)
            if question.kind == "score"
            else chr(65 + i)
            if count <= 26
            else chr(65 + i // 26) + chr(65 + i % 26)
            for i in range(count)
        )
        lines = [f"Question: {instruction}"] if instruction else []
        typed = any(isinstance(candidate.value, bool) for candidate in candidates)
        for label, candidate in zip(labels, candidates):
            description = text(candidate.description)
            if question.kind == "score":
                lines.append(f"{label}: {score_level(candidate)}")
            else:
                name = compact(candidate.value) if typed else text(candidate.value)
                lines.append(
                    f"{label}: {name}" + (f" - {description}" if description else "")
                )
        lines.append(
            "Answer with the number of one level only."
            if question.kind == "score"
            else "Answer with the letter of one option only."
        )
    else:
        raise ValueError(f"unsupported question kind: {question.kind}")
    return "\n".join([text(evidence), "", *lines]), labels


def prepare_question(
    tokenizer: Any, evidence: Any, question: Question, context_limit: int = 2048
) -> PreparedQuestion:
    content, labels = render_question(evidence, question)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    if not ids or len(ids) > context_limit:
        raise ValueError(f"rendered context {len(ids)} exceeds limit {context_limit}")
    candidate_ids = []
    for label in labels:
        encoded = tuple(tokenizer.encode(prompt + label, add_special_tokens=False))
        if (
            len(encoded) != len(ids) + 1
            or encoded[:-1] != ids
            or encoded[-1] in candidate_ids
        ):
            raise ValueError(
                f"answer label {label!r} is not one distinct token at the answer position"
            )
        candidate_ids.append(encoded[-1])
    return PreparedQuestion(content, prompt, labels, ids, tuple(candidate_ids))


def native_payload(prepared: PreparedQuestion, model: str) -> dict:
    """Salt is injected per actual send, never baked into a reusable workload."""
    return {
        "model": model,
        "input_ids": list(prepared.input_ids),
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1,
        },
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "token_ids_logprob": list(prepared.candidate_ids),
        "stream": True,
    }
