#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the live Dynamo OpenAI path accepts a prompt beyond Qwen's native 32K window."""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

NATIVE_CONTEXT = 32768
TARGET_PROMPT_TOKENS = 120000
MAX_CONTEXT = 131072


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--server-info-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_server_info(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        status = response.status
        payload = json.loads(response.read())
    assert status == 200, status
    assert payload["status"] == "ready", payload.get("status")
    assert int(payload["context_length"]) == MAX_CONTEXT, payload["context_length"]
    assert int(payload["tp_size"]) == 2, payload["tp_size"]
    assert int(payload["ep_size"]) == 2, payload["ep_size"]
    assert payload["quantization"] == "modelopt_fp4", payload["quantization"]
    assert payload["enable_return_routed_experts"] is True
    override = payload["json_model_override_args"]
    if isinstance(override, str):
        override = json.loads(override)
    rope = override["rope_scaling"]
    assert rope["rope_type"] == "yarn", rope
    assert float(rope["factor"]) == 4.0, rope
    assert int(rope["original_max_position_embeddings"]) == NATIVE_CONTEXT, rope
    assert float(rope["rope_theta"]) == 1000000.0, rope
    return payload


def token_count(tokenized: Any) -> int:
    """Return the sequence length from list or BatchEncoding tokenizer output."""
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if tokenized and isinstance(tokenized[0], list):
        assert len(tokenized) == 1, len(tokenized)
        tokenized = tokenized[0]
    return len(tokenized)


def render_prompt(tokenizer: Any, repeats: int) -> tuple[str, int]:
    content = "Preserve this repository context while answering briefly. " * repeats
    messages = [{"role": "user", "content": content}]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return content, token_count(tokens)


def make_prompt(tokenizer: Any) -> tuple[str, int]:
    _, one_repeat_tokens = render_prompt(tokenizer, 1)
    _, two_repeat_tokens = render_prompt(tokenizer, 2)
    tokens_per_repeat = two_repeat_tokens - one_repeat_tokens
    assert tokens_per_repeat > 0, tokens_per_repeat
    required_tokens = TARGET_PROMPT_TOKENS - one_repeat_tokens
    repeats = 1 + max(0, -(-required_tokens // tokens_per_repeat))
    content, prompt_tokens = render_prompt(tokenizer, repeats)
    assert NATIVE_CONTEXT < prompt_tokens <= TARGET_PROMPT_TOKENS + 64, prompt_tokens
    assert prompt_tokens < MAX_CONTEXT, prompt_tokens
    return content, prompt_tokens


def main() -> None:
    cli = parse_args()
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    server_info = load_server_info(cli.server_info_url)
    (output.parent / "server-info.json").write_text(
        json.dumps(server_info, indent=2, sort_keys=True) + "\n"
    )
    tokenizer = AutoTokenizer.from_pretrained(cli.tokenizer, local_files_only=True)
    content, local_prompt_tokens = make_prompt(tokenizer)
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-30B-A3B",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    (output.parent / "long-context-request.json").write_bytes(body + b"\n")
    request = urllib.request.Request(
        cli.url, data=body, headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        status = response.status
        payload = json.loads(response.read())
    (output.parent / "long-context-response.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    assert status == 200, status
    assert payload.get("choices"), payload
    server_prompt_tokens = int(payload.get("usage", {}).get("prompt_tokens", -1))
    assert server_prompt_tokens == local_prompt_tokens, (
        server_prompt_tokens,
        local_prompt_tokens,
    )
    summary = {
        "result": "passed",
        "boundary": "Dynamo /v1/chat/completions -> SGLang NVFP4",
        "native_context": NATIVE_CONTEXT,
        "configured_context": MAX_CONTEXT,
        "local_prompt_tokens": local_prompt_tokens,
        "server_prompt_tokens": server_prompt_tokens,
        "http_status": status,
        "model": payload.get("model"),
        "finish_reason": payload["choices"][0].get("finish_reason"),
        "server_info": {
            "context_length": server_info["context_length"],
            "json_model_override_args": server_info["json_model_override_args"],
            "quantization": server_info["quantization"],
            "tp_size": server_info["tp_size"],
            "ep_size": server_info["ep_size"],
            "enable_return_routed_experts": server_info["enable_return_routed_experts"],
        },
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
