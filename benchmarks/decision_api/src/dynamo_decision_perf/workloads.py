# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic text-only decision workloads; generation never loads model weights."""

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .rendering import (
    Candidate,
    Question,
    compact,
    native_payload,
    prepare_question,
    score_level,
)

MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CORE_REVISION = "6ce6582b25e78d8dd85b5f4e9dc107838217e1fa"
DIALECTS = ("oai", "sglang_native", "systemone")


@dataclass(frozen=True)
class Shape:
    name: str
    target_tokens: int = 256
    questions: int = 1
    choices: int = 8
    kind: str = "choice"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise ValueError("shape name must be a safe lowercase identifier")
        if (
            not 1 <= self.target_tokens <= 2048
            or not 1 <= self.questions <= 128
            or not 2 <= self.choices <= 255
        ):
            raise ValueError("shape dimensions exceed the supported profile")
        if self.kind not in ("choice", "predicate", "score", "mixed"):
            raise ValueError("unsupported question kind")


def ofat_matrix() -> tuple[Shape, ...]:
    return (
        Shape("base"),
        Shape("tokens1024", 1024),
        Shape("tokens1536", 1536),
        Shape("questions4", questions=4),
        Shape("questions16", questions=16),
        Shape("choices2", choices=2),
        Shape("choices32", choices=32),
        Shape("predicate", kind="predicate"),
        Shape("score", kind="score"),
        Shape("mixed", questions=3, kind="mixed"),
    )


def questions_for(shape: Shape) -> tuple[Question, ...]:
    kinds = ("choice", "predicate", "score") if shape.kind == "mixed" else (shape.kind,)
    questions = []
    for index in range(shape.questions):
        kind = kinds[index % len(kinds)]
        candidates = (
            tuple(
                Candidate(f"team{i}", f"Department {i}") for i in range(shape.choices)
            )
            if kind == "choice"
            else tuple(
                Candidate(str(i), label=name)
                for i, name in enumerate(("Low", "Medium", "High"))
            )
            if kind == "score"
            else ()
        )
        instruction = {
            "choice": "Choose the responsible department.",
            "predicate": "The request needs action today.",
            "score": "Rate the impact.",
        }[kind]
        instruction = f"Assessment {index:03d}: {instruction}"
        questions.append(Question(f"q{index:03d}", kind, instruction, candidates))
    return tuple(questions)


def wire_payload(
    model: str, evidence: Any, questions: tuple[Question, ...], dialect: str
) -> dict:
    if dialect not in DIALECTS:
        raise ValueError(f"unsupported dialect: {dialect}")
    if dialect != "oai" and any(
        isinstance(c.value, bool)
        for q in questions
        for c in q.candidates
        if q.kind == "choice"
    ):
        raise ValueError(
            "typed boolean choice values are supported only by the OpenAI wire contract"
        )
    items = []
    for q in questions:
        if dialect == "oai":
            if not isinstance(q.instructions, str):
                raise ValueError("OpenAI instructions must be strings")
            if q.kind == "predicate" and any(
                c.description is not None for c in q.candidates
            ):
                raise ValueError("OpenAI does not support predicate descriptions")
            if any(
                c.description is not None and not isinstance(c.description, str)
                for c in q.candidates
            ):
                raise ValueError("OpenAI candidate descriptions must be strings")
            item = {"name": q.id, "type": q.kind, "instructions": q.instructions}
            if q.kind == "choice":
                item["choices"] = [
                    {
                        "value": c.value,
                        **(
                            {"description": c.description}
                            if c.description is not None
                            else {}
                        ),
                    }
                    for c in q.candidates
                ]
            elif q.kind == "score":
                item["levels"] = [
                    {
                        "label": c.label,
                        **(
                            {"description": c.description}
                            if c.description is not None
                            else {}
                        ),
                    }
                    for c in q.candidates
                ]
        elif dialect == "sglang_native":
            item = {
                "id": q.id,
                "type": "yes_no" if q.kind == "predicate" else q.kind,
                "question": q.instructions,
            }
            if q.kind == "choice":
                item["options"] = [
                    {
                        "name": c.value,
                        **(
                            {"description": c.description}
                            if c.description is not None
                            else {}
                        ),
                    }
                    for c in q.candidates
                ]
            elif q.kind == "score":
                item["levels"] = [score_level(c) for c in q.candidates]
            elif q.kind == "predicate":
                item.update(
                    {
                        key: c.description
                        for key, c in zip(("yes", "no"), q.candidates)
                        if c.description is not None
                    }
                )
        else:
            item = {
                "type": "noul" if q.kind == "predicate" else q.kind,
                "instructions": q.instructions,
            }
            if q.kind == "choice":
                item["criteria"] = {
                    c.value: "" if c.description is None else c.description
                    for c in q.candidates
                }
            elif q.kind == "score":
                item["criteria"] = [score_level(c) for c in q.candidates]
            elif q.kind == "predicate" and q.candidates:
                item["criteria"] = {
                    key: c.description
                    for key, c in zip(("true", "false"), q.candidates)
                    if c.description is not None
                }
        items.append(item)
    body = {
        "model": model,
        "state" if dialect == "systemone" else "input": evidence,
        "questions": dict(zip((q.id for q in questions), items))
        if dialect == "systemone"
        else items,
    }
    if dialect != "oai":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if dialect == "sglang_native":
        body["nvext"] = {"format": "sglang_native"}
    return body


def _evidence(words: list[str], count: int, logical_id: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Record {logical_id}. A customer reports a duplicate payment. "
                    + " ".join(words[:count]),
                }
            ],
        }
    ]


def _prepare_row(tokenizer, shape, questions, rng, logical_id):
    words = [
        rng.choice(
            (
                "payment",
                "review",
                "customer",
                "department",
                "request",
                "invoice",
                "today",
                "support",
            )
        )
        for _ in range(4096)
    ]
    low, high, best = 0, len(words), None
    # Tokenization is bounded; no truncation of an already-rendered prompt.
    for _ in range(14):
        if low > high:
            break
        middle = (low + high) // 2
        evidence = _evidence(words, middle, logical_id)
        try:
            prepared = tuple(
                prepare_question(tokenizer, evidence, q) for q in questions
            )
        except ValueError as error:
            if "context" not in str(error):
                raise
            high = middle - 1
            continue
        distance = max(abs(len(p.input_ids) - shape.target_tokens) for p in prepared)
        if best is None or distance < best[0]:
            best = (distance, evidence, prepared)
        if max(len(p.input_ids) for p in prepared) > shape.target_tokens:
            high = middle - 1
        else:
            low = middle + 1
    if best is None or best[0] > max(32, shape.target_tokens // 8):
        raise ValueError(
            "rendered prompt cannot meet the approximate token target without truncation"
        )
    return best[1], best[2]


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(compact(payload).encode()).hexdigest()


def generate_workloads(
    output: Path,
    tokenizer: Any,
    model: str,
    *,
    seed: int = 17,
    count: int = 8,
    shapes: tuple[Shape, ...] | None = None,
    tokenizer_details: dict | None = None,
) -> dict:
    if not 1 <= count <= 10000 or not model.strip():
        raise ValueError("count must be 1..10000 and model must not be blank")
    shapes = ofat_matrix() if shapes is None else shapes
    if len({s.name for s in shapes}) != len(shapes):
        raise ValueError("duplicate shape names")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files, unsupported = {}, []
    for shape in shapes:
        rng = random.Random(f"{seed}:{shape.name}")
        questions = questions_for(shape)
        rows = {dialect: [] for dialect in (*DIALECTS, "native_score")}
        metadata = []
        try:
            for index in range(count):
                logical_id = f"{shape.name}-{seed}-{index:06d}"
                evidence, prepared = _prepare_row(
                    tokenizer, shape, questions, rng, logical_id
                )
                payloads = {
                    d: wire_payload(model, evidence, questions, d) for d in DIALECTS
                }
                if len(questions) == 1:
                    payloads["native_score"] = native_payload(prepared[0], model)
                for dialect, payload in payloads.items():
                    rows[dialect].append(
                        {"session_id": logical_id, "payloads": [payload]}
                    )
                metadata.append(
                    {
                        "logical_id": logical_id,
                        "row_index": index,
                        "shape": shape.name,
                        "expected_questions": len(questions),
                        "prompt_tokens": [len(p.input_ids) for p in prepared],
                        "expanded_tokens": sum(len(p.input_ids) for p in prepared),
                        "candidate_token_ids": [
                            list(p.candidate_ids) for p in prepared
                        ],
                        "prompt_sha256": [
                            hashlib.sha256(p.prompt.encode()).hexdigest()
                            for p in prepared
                        ],
                        "payload_sha256": {
                            d: payload_sha256(p) for d, p in payloads.items()
                        },
                    }
                )
        except ValueError as error:
            unsupported.append({"shape": shape.name, "reason": str(error)})
            continue
        for dialect, entries in rows.items():
            if entries:
                files[f"{shape.name}.{dialect}.json"] = {"data": entries}
        files[f"{shape.name}.metadata.json"] = metadata
        if len(questions) != 1:
            unsupported.append(
                {
                    "shape": shape.name,
                    "dialect": "native_score",
                    "reason": "native single-question reference cannot represent logical multi-question latency",
                }
            )
    digests = {}
    for name, value in sorted(files.items()):
        data = (compact(value) + "\n").encode()
        (output / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "schema_version": 1,
        "model": model,
        "model_revision": MODEL_REVISION,
        "frontend_crates_revision": CORE_REVISION,
        "prompt_format_version": 1,
        "seed": seed,
        "count": count,
        "context_limit": 2048,
        "enable_thinking": False,
        "question_semantics": "shared evidence; ordinal-specific instructions prevent identical full-prompt reuse across sibling questions",
        "custom_dataset_type": "inputs_json",
        "cache_salt_policy": "fresh_per_actual_native_send",
        "shapes": [asdict(s) for s in shapes],
        "tokenizer": tokenizer_details
        or {"kind": type(tokenizer).__name__, "unqualified": True},
        "sha256": digests,
        "unsupported": unsupported,
    }
    (output / "manifest.json").write_text(compact(manifest) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    path = args.tokenizer.resolve(strict=True)
    if path.name != MODEL_REVISION:
        parser.error(f"tokenizer must be the cached snapshot revision {MODEL_REVISION}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    artifact_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file()
        and (
            p.name.startswith("tokenizer")
            or p.name
            in (
                "chat_template.jinja",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            )
        )
    }
    manifest = generate_workloads(
        args.output,
        tokenizer,
        args.model,
        seed=args.seed,
        count=args.count,
        tokenizer_details={
            "snapshot_revision": MODEL_REVISION,
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
            "artifact_sha256": artifact_hashes,
            "serving_tokenizer_parity": "not_established_by_generation",
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(manifest["sha256"]),
                "unsupported": manifest["unsupported"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
