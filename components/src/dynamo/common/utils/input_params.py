#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from dynamo.common.utils.token_ids import token_ids_to_list


def resolve_thinking_token_budget(request: dict) -> Optional[Any]:
    """Resolve the thinking-token budget from an OpenAI-compatible request.

    Supports both the OpenAI-compatible root-level field
    ``thinking_token_budget`` and the legacy Dynamo extension
    ``nvext.max_thinking_tokens``. The root-level field takes precedence.
    """
    root = request.get("thinking_token_budget")
    legacy = (request.get("nvext") or {}).get("max_thinking_tokens")
    return root if root is not None else legacy


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


def _chat_template_token_ids(rendered: Any) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` output to ``list[int]``.

    Hugging Face tokenizers return a list of ids, or a ``BatchEncoding`` when a
    caller-supplied template kwarg forces ``return_dict``; mistral-common's
    backend returns a list of ids. Anything else is a tokenizer we do not
    understand, and is reported rather than forwarded.
    """
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if (
        isinstance(rendered, list)
        and rendered
        and isinstance(rendered[0], list)
        and len(rendered) == 1
    ):
        rendered = rendered[0]
    if not isinstance(rendered, list) or not all(
        isinstance(t, int) and not isinstance(t, bool) for t in rendered
    ):
        raise TypeError(
            "apply_chat_template(tokenize=True) returned "
            f"{type(rendered).__name__}, expected a list of token ids"
        )
    return rendered


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
                for reserved in ("tokenize", "return_dict", "add_generation_prompt"):
                    extra_kwargs.pop(reserved, None)

                # Inject reasoning_content as <think> blocks into content,
                # but only if the template doesn't handle it natively.
                # Templates like Nemotron and Qwen3 reference reasoning_content
                # directly — injecting would produce duplicate <think> blocks.
                chat_template_src = getattr(self.tokenizer, "chat_template", "") or ""
                if "reasoning_content" not in chat_template_src:
                    _inject_reasoning_content(request["messages"])

                # Tokenize inside the template so the engine receives ids and
                # never re-encodes rendered text: encoding the rendered string
                # again would add a second BOS wherever the template already
                # emitted one, and it loses Mistral's [INST] control tokens,
                # which mistral-common renders as SentencePiece pieces.
                # return_dict=False: transformers 5 returns a BatchEncoding
                # from tokenize=True by default.
                return _chat_template_token_ids(
                    self.tokenizer.apply_chat_template(
                        request["messages"],
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        **extra_kwargs,
                    )
                )
            elif "prompt" in request:
                return self.tokenizer.encode(request["prompt"])
            elif "text" in request:
                return self.tokenizer.encode(request["text"])
            else:
                raise ValueError("No input parameter found in request")
        return token_ids_to_list(request.get("token_ids"))
