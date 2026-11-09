# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU parity with explicitly staged tokenizer-only artifacts.

Set DYNAMO_DELTA_TOKENIZER_MODEL_DIR to a local Qwen3 tokenizer snapshot. These
tests never fetch model weights or tokenizer artifacts during collection.
"""
import random
from collections.abc import Mapping

import pytest

from dynamo.frontend.delta_tokenizer import DeltaConfig
from dynamo.frontend.toktier_session import create_delta_tokenizer

pytestmark = [pytest.mark.post_merge, pytest.mark.integration, pytest.mark.gpu_0]


def test_real_random_append_parity(delta_staged_tokenizer):
    reference, directory = delta_staged_tokenizer
    manager = create_delta_tokenizer(
        reference, directory, "qwen", DeltaConfig(verify_rate=1)
    )
    assert (
        manager is not None
    ), "staged artifacts and installed oracle must be certified"
    rng = random.Random(42)
    text = "A test conversation with punctuation and words.\n" * 4096
    with manager:
        for _ in range(100):
            text += rng.choice(
                ["a", "\u0301", "  ", "\n", "🙂漢字", "<|im_start|>", "tool result: {}\n"]
            )
            actual = manager.tokenize(
                "session", text, scope="tenant", template_identity="qwen"
            )
            assert actual == reference.encode(text, add_special_tokens=False)
            manager.wait_for_idle()
        assert manager.stats().get("incremental", 0) > 0
        assert manager.stats()["mismatches"] == 0


def test_chat_template_and_tool_transcripts(delta_staged_tokenizer):
    reference, directory = delta_staged_tokenizer
    manager = create_delta_tokenizer(reference, directory, "qwen", DeltaConfig())
    assert manager is not None
    messages = [{"role": "user", "content": "Describe the test."}]
    with manager:
        for turn in range(4):
            text = reference.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            actual = manager.tokenize(
                "session", text, scope="tenant", template_identity="qwen"
            )
            expected = reference.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if isinstance(expected, Mapping):
                expected = expected["input_ids"]
            assert actual == list(expected)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": {"turn": turn},
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"result":"ok"}'},
                    {"role": "user", "content": "Continue."},
                ]
            )


def test_unknown_artifact_uses_original_path(delta_staged_tokenizer, tmp_path):
    reference, _directory = delta_staged_tokenizer
    (tmp_path / "tokenizer.json").write_text("{}")
    assert (
        create_delta_tokenizer(reference, str(tmp_path), "unknown", DeltaConfig())
        is None
    )


def test_mutated_added_tokens_are_not_admitted(delta_staged_tokenizer):
    reference, directory = delta_staged_tokenizer
    reference.add_special_tokens({"additional_special_tokens": ["<delta-new-token>"]})
    assert create_delta_tokenizer(reference, directory, "qwen", DeltaConfig()) is None
