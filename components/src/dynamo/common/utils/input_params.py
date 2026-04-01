#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional


def _inject_reasoning_content(messages: list) -> None:
    """Inject reasoning_content as <think> blocks into content.

    Chat templates only reference message["content"] — they don't see
    reasoning_content. This converts it back to <think> blocks so the
    model sees its own prior chain-of-thought across turns.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        reasoning = msg.get("reasoning_content")
        if not reasoning:
            continue

        # Build <think> wrapped text
        if isinstance(reasoning, str):
            think_text = f"<think>{reasoning}</think>" if reasoning else ""
        elif isinstance(reasoning, list):
            # Segments variant: wrap each non-empty segment
            parts = [f"<think>{seg}</think>" for seg in reasoning if seg]
            think_text = "".join(parts)
        else:
            continue

        if not think_text:
            continue

        # Prepend to content
        existing = msg.get("content")
        if isinstance(existing, str):
            msg["content"] = think_text + existing
        elif isinstance(existing, list):
            # Multimodal content array — prepend as text part
            msg["content"] = [{"type": "text", "text": think_text}] + existing
        else:
            # null or absent
            msg["content"] = think_text

        # Remove so template doesn't see both
        msg.pop("reasoning_content", None)


class InputParamManager:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def get_input_param(self, request: dict, use_tokenizer: bool) -> Optional[Any]:
        """
        Get the input parameter for the request.
        """

        if use_tokenizer:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not available")

            if "messages" in request:
                # Forward chat_template_args / chat_template_kwargs to the
                # template so model-specific variables (e.g. enable_thinking)
                # are available during rendering.
                extra_kwargs = {}
                if "chat_template_kwargs" in request:
                    extra_kwargs.update(request["chat_template_kwargs"])
                if "chat_template_args" in request:
                    extra_kwargs.update(request["chat_template_args"])
                # Strip keys that are already set explicitly to avoid
                # TypeError: got multiple values for keyword argument.
                for reserved in ("tokenize", "add_generation_prompt"):
                    extra_kwargs.pop(reserved, None)

                # Inject reasoning_content as <think> blocks into content,
                # but only if the template doesn't handle it natively.
                # Templates like Nemotron and Qwen3 reference reasoning_content
                # directly — injecting would produce duplicate <think> blocks.
                chat_template_src = getattr(self.tokenizer, "chat_template", "") or ""
                if "reasoning_content" not in chat_template_src:
                    _inject_reasoning_content(request["messages"])

                return self.tokenizer.apply_chat_template(
                    request["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
            elif "prompt" in request:
                return self.tokenizer.encode(request["prompt"])
            elif "text" in request:
                return self.tokenizer.encode(request["text"])
            else:
                raise ValueError("No input parameter found in request")
        return request.get("token_ids")
