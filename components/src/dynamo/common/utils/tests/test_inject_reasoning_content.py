# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for _inject_reasoning_content in input_params.py.

Verifies that reasoning_content from prior assistant turns is converted
to <think> blocks in the content field before chat template rendering.
"""

import copy

import pytest

from dynamo.common.utils.input_params import _inject_reasoning_content

# Total runtime ~0.04s — no need for parallel marker.
pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


class TestInjectReasoningContent:
    """Test suite for _inject_reasoning_content"""

    def test_text_variant_prepends_to_content(self):
        """Text reasoning_content is wrapped in <think> and prepended."""
        messages = [
            {
                "role": "assistant",
                "content": "The answer is 12.",
                "reasoning_content": "sqrt(144) = 12",
            },
        ]
        _inject_reasoning_content(messages)

        assert (
            messages[0]["content"] == "<think>sqrt(144) = 12</think>The answer is 12."
        )
        assert "reasoning_content" not in messages[0]

    def test_segments_variant_wraps_each_segment(self):
        """Segments are individually wrapped in <think> blocks."""
        messages = [
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning_content": ["first thought", "second thought", ""],
            },
        ]
        _inject_reasoning_content(messages)

        content = messages[0]["content"]
        assert content.startswith("<think>first thought</think>")
        assert "<think>second thought</think>" in content
        assert "<think></think>" not in content  # empty segment skipped
        assert content.endswith("Done.")
        assert "reasoning_content" not in messages[0]

    def test_null_content_creates_from_reasoning(self):
        """When content is null/None, reasoning becomes the content."""
        messages = [
            {"role": "assistant", "content": None, "reasoning_content": "Thinking..."},
        ]
        _inject_reasoning_content(messages)

        assert messages[0]["content"] == "<think>Thinking...</think>"

    def test_absent_content_creates_from_reasoning(self):
        """When content key is absent, reasoning becomes the content."""
        messages = [
            {"role": "assistant", "reasoning_content": "Thinking..."},
        ]
        _inject_reasoning_content(messages)

        assert messages[0]["content"] == "<think>Thinking...</think>"

    def test_multimodal_content_prepends_text_part(self):
        """Array content gets a text part prepended, not replaced."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is the image."}],
                "reasoning_content": "Analyzing the image...",
            },
        ]
        _inject_reasoning_content(messages)

        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {
            "type": "text",
            "text": "<think>Analyzing the image...</think>",
        }
        assert content[1] == {"type": "text", "text": "Here is the image."}

    def test_skips_non_assistant_messages(self):
        """User and tool messages are not modified."""
        messages = [
            {
                "role": "user",
                "content": "hello",
                "reasoning_content": "should not touch",
            },
            {
                "role": "tool",
                "content": "result",
                "reasoning_content": "should not touch",
            },
        ]
        original = copy.deepcopy(messages)
        _inject_reasoning_content(messages)

        assert messages == original

    def test_skips_empty_reasoning(self):
        """Empty string reasoning_content is skipped."""
        messages = [
            {"role": "assistant", "content": "Answer.", "reasoning_content": ""},
        ]
        _inject_reasoning_content(messages)

        assert messages[0]["content"] == "Answer."
        # reasoning_content not removed since we skipped (falsy check)

    def test_agentic_multi_turn_tool_call_flow(self):
        """Full agentic flow: reason → tool_call → tool_result → reason → answer."""
        messages = [
            {"role": "user", "content": "What is sqrt(144) + sqrt(256)?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need to compute sqrt(144) first.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "calc",
                            "arguments": '{"expr": "sqrt(144)"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "12"},
            {
                "role": "assistant",
                "content": "The answer is 28.",
                "reasoning_content": "Got 12. sqrt(256) = 16. Sum = 28.",
            },
            {"role": "user", "content": "Thanks!"},
        ]
        _inject_reasoning_content(messages)

        # First assistant turn: reasoning injected, null content → reasoning only
        assert (
            messages[1]["content"]
            == "<think>I need to compute sqrt(144) first.</think>"
        )
        assert "reasoning_content" not in messages[1]
        assert "tool_calls" in messages[1]  # tool_calls untouched

        # Tool message untouched
        assert messages[2]["content"] == "12"

        # Second assistant turn: reasoning prepended to content
        assert (
            messages[3]["content"]
            == "<think>Got 12. sqrt(256) = 16. Sum = 28.</think>The answer is 28."
        )
        assert "reasoning_content" not in messages[3]

        # User messages untouched
        assert messages[0]["content"] == "What is sqrt(144) + sqrt(256)?"
        assert messages[4]["content"] == "Thanks!"


class TestInputParamManagerReasoningInjection:
    """Test that InputParamManager respects template introspection."""

    def test_injects_when_template_ignores_reasoning(self):
        """Templates without reasoning_content get injection."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
        )
        tokenizer.apply_chat_template = MagicMock(return_value="rendered")

        from dynamo.common.utils.input_params import InputParamManager

        mgr = InputParamManager(tokenizer)
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hi.",
                    "reasoning_content": "thinking...",
                },
                {"role": "user", "content": "Bye"},
            ]
        }
        mgr.get_input_param(request, use_tokenizer=True)

        # Verify injection happened: reasoning_content removed, content has <think>
        called_messages = tokenizer.apply_chat_template.call_args[0][0]
        assert "reasoning_content" not in called_messages[0]
        assert called_messages[0]["content"].startswith("<think>thinking...</think>")

    def test_skips_injection_when_template_handles_reasoning(self):
        """Templates with reasoning_content are left alone."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}"
            "{{ m.role }}: {{ m.content }}{% endfor %}"
        )
        tokenizer.apply_chat_template = MagicMock(return_value="rendered")

        from dynamo.common.utils.input_params import InputParamManager

        mgr = InputParamManager(tokenizer)
        request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hi.",
                    "reasoning_content": "thinking...",
                },
                {"role": "user", "content": "Bye"},
            ]
        }
        mgr.get_input_param(request, use_tokenizer=True)

        # Verify injection was skipped: reasoning_content still present, content unchanged
        called_messages = tokenizer.apply_chat_template.call_args[0][0]
        assert called_messages[0]["reasoning_content"] == "thinking..."
        assert called_messages[0]["content"] == "Hi."
