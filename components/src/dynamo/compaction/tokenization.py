# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned local text-only rendering profile; no remote tokenizer code or downloads."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .protocol import Invalid, canonical, require, strict_json

PROFILE_MODEL = "Qwen/Qwen3-0.6B"
PROFILE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROFILE_ARTIFACT_SHA256 = (
    "0d96ecc916745b3a6eb234433e6ddeeab051736a6336ef0db95b3022c425c3eb"
)
RENDERER_REVISION = "dynamo-decisions-0.1.0-prompt-v1"


def text_messages(messages):
    require(type(messages) is list and bool(messages), "invalid_messages")
    result = []
    for message in messages:
        require(
            type(message) is dict and set(message) == {"role", "content"},
            "unsupported_message_fields",
        )
        require(
            message["role"] in ("system", "user", "assistant"),
            "unsupported_message_role",
        )
        content = message["content"]
        if type(content) is list:
            require(bool(content), "empty_content")
            for part in content:
                require(
                    type(part) is dict
                    and set(part) == {"type", "text"}
                    and part["type"] == "text"
                    and type(part["text"]) is str,
                    "unsupported_content",
                )
            content = "\n".join(part["text"] for part in content)
        require(type(content) is str, "unsupported_content")
        result.append({"role": message["role"], "content": content})
    return result


class TextTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        /,
        *,
        tokenize: Literal[False],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> object:
        ...

    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]:
        ...


@dataclass(frozen=True)
class RenderedCounter:
    tokenizer: TextTokenizer
    name: str

    def render(self, messages) -> str:
        rendered = self.tokenizer.apply_chat_template(
            text_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise Invalid("invalid_rendered_prompt")
        return rendered

    def __call__(self, serialized):
        return max(self.counts(serialized))

    def usage(self, serialized):
        return sum(self.counts(serialized))

    def counts(self, serialized):
        payload = strict_json(serialized)
        if "questions" in payload:
            require(
                type(payload["input"]) is str
                and type(payload["questions"]) is list
                and 0 < len(payload["questions"]) <= 64,
                "unsupported_decision_shape",
            )
            counts = []
            for question in payload["questions"]:
                require(
                    question["type"] == "choice" and len(question["choices"]) == 2,
                    "unsupported_decision_shape",
                )
                content = payload["input"] + "\n\nQuestion: " + question["instructions"]
                for label, choice in zip(("A", "B"), question["choices"]):
                    require(
                        type(choice["value"]) is str
                        and type(choice["description"]) is str,
                        "unsupported_choice",
                    )
                    content += f"\n{label}: {choice['value']} - {choice['description']}"
                content += "\nAnswer with the letter of one option only."
                prompt = self.render([{"role": "user", "content": content}])
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                candidate_ids = []
                for label in ("A", "B"):
                    extended = self.tokenizer.encode(
                        prompt + label, add_special_tokens=False
                    )
                    require(
                        len(extended) == len(ids) + 1 and extended[:-1] == ids,
                        "unsupported_candidate_tokens",
                    )
                    candidate_ids.append(extended[-1])
                require(len(set(candidate_ids)) == 2, "unsupported_candidate_tokens")
                counts.append(len(ids))
            return counts
        if "messages" in payload:
            return [
                len(
                    self.tokenizer.encode(
                        self.render(payload["messages"]), add_special_tokens=False
                    )
                )
            ]
        require(set(payload) == {"summary", "retained"}, "unsupported_counter_input")
        # This is only the candidate projection; Pi separately counts the actor request.
        return [len(self.tokenizer.encode(serialized, add_special_tokens=False))]


def load_counter(tokenizer_path):
    from transformers import AutoTokenizer

    path = Path(tokenizer_path)
    require(path.is_absolute() and path.is_dir(), "invalid_tokenizer_path")
    require(not (path / "chat_templates").exists(), "unqualified_tokenizer_profile")
    artifacts = (path / "tokenizer.json", path / "tokenizer_config.json")
    require(all(item.is_file() for item in artifacts), "missing_tokenizer_artifacts")
    digest = hashlib.sha256()
    for artifact in artifacts:
        digest.update(artifact.read_bytes())
    template = path / "chat_template.jinja"
    if template.is_file():
        digest.update(template.read_bytes())
    require(
        path.name == PROFILE_REVISION and digest.hexdigest() == PROFILE_ARTIFACT_SHA256,
        "unqualified_tokenizer_profile",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )
    require(
        type(tokenizer.chat_template) is str and bool(tokenizer.chat_template),
        "missing_chat_template",
    )
    return RenderedCounter(
        tokenizer, "qwen3-0.6b:" + RENDERER_REVISION + ":" + digest.hexdigest()
    )


def count_request(payload, counter):
    require(
        type(payload) is dict and set(payload) == {"messages", "max_output_tokens"},
        "unsupported_count_request",
    )
    output = payload["max_output_tokens"]
    require(type(output) is int and 0 < output <= 65536, "invalid_output_reserve")
    count = counter(canonical({"messages": payload["messages"]}))
    rendered = counter.tokenizer.encode(
        counter.render(payload["messages"]), add_special_tokens=False
    )
    return {
        "version": 1,
        "status": "counted",
        "input_tokens": count,
        "output_reserve": output,
        "total_tokens": count + output,
        "counter_name": counter.name,
        "qualified_counter": True,
        "request_sha256": hashlib.sha256(canonical(payload).encode()).hexdigest(),
        "rendered_sha256": hashlib.sha256(canonical(rendered).encode()).hexdigest(),
    }
