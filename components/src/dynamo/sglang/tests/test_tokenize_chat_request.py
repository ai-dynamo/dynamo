# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tokenize_chat_request helper function."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Direct import to avoid complex import chain from dynamo.common
def _load_input_params_module():
    """Load InputParamManager directly from the module file."""
    # Path: tests/ -> sglang/ -> dynamo/ -> common/utils/input_params.py
    # parents[0] = tests/, parents[1] = sglang/, parents[2] = dynamo/
    module_path = (
        Path(__file__).resolve().parents[2]  # -> dynamo/
        / "common"
        / "utils"
        / "input_params.py"
    )
    spec = importlib.util.spec_from_file_location("input_params", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InputParamManager


InputParamManager = _load_input_params_module()


pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.pre_merge,
]


class TestTokenizeChatRequest:
    """Tests for InputParamManager.tokenize_chat_request()."""

    def test_tokenize_chat_request_with_dict(self):
        """Test tokenizing a chat completion request passed as a dict."""
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        expected_token_ids = [1, 2, 3, 100, 200, 300, 4]
        mock_tokenizer.apply_chat_template.return_value = expected_token_ids

        # Create the manager
        manager = InputParamManager(mock_tokenizer)

        # Create a chat completion request as a dict
        request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        # Call the function
        token_ids = manager.tokenize_chat_request(request)

        # Verify the result
        assert token_ids == expected_token_ids
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            request["messages"], tokenize=True, add_generation_prompt=True
        )

    def test_tokenize_chat_request_with_object(self):
        """Test tokenizing a chat completion request passed as an object."""
        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        expected_token_ids = [1, 5, 10, 15, 20, 2]
        mock_tokenizer.apply_chat_template.return_value = expected_token_ids

        # Create the manager
        manager = InputParamManager(mock_tokenizer)

        # Create a mock request object with messages attribute
        mock_request = MagicMock()
        mock_request.messages = [
            {"role": "user", "content": "What is 2+2?"},
        ]

        # Call the function
        token_ids = manager.tokenize_chat_request(mock_request)

        # Verify the result
        assert token_ids == expected_token_ids
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            mock_request.messages, tokenize=True, add_generation_prompt=True
        )

    def test_tokenize_chat_request_no_tokenizer(self):
        """Test that ValueError is raised when tokenizer is not available."""
        # Create manager without tokenizer
        manager = InputParamManager(None)

        request = {
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        with pytest.raises(ValueError, match="Tokenizer is not available"):
            manager.tokenize_chat_request(request)

    def test_tokenize_chat_request_no_messages(self):
        """Test that ValueError is raised when request has no messages."""
        mock_tokenizer = MagicMock()
        manager = InputParamManager(mock_tokenizer)

        # Request without messages field
        request = {
            "model": "test-model",
            "prompt": "This is a prompt, not messages",
        }

        with pytest.raises(ValueError, match="No 'messages' field found"):
            manager.tokenize_chat_request(request)

    def test_tokenize_chat_request_multi_turn_conversation(self):
        """Test tokenizing a multi-turn conversation."""
        mock_tokenizer = MagicMock()
        expected_token_ids = [1, 100, 200, 300, 400, 500, 600, 700, 2]
        mock_tokenizer.apply_chat_template.return_value = expected_token_ids

        manager = InputParamManager(mock_tokenizer)

        # Multi-turn conversation
        request = {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "What about 3+3?"},
            ],
        }

        token_ids = manager.tokenize_chat_request(request)

        assert token_ids == expected_token_ids
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            request["messages"], tokenize=True, add_generation_prompt=True
        )

    def test_tokenize_chat_request_returns_list_of_ints(self):
        """Test that the function returns a list of integers."""
        mock_tokenizer = MagicMock()
        # Simulate realistic token IDs
        mock_tokenizer.apply_chat_template.return_value = [
            128000,
            128006,
            9125,
            128007,
            271,
            2675,
            527,
            264,
            11190,
            18328,
            13,
            128009,
            128006,
            882,
            128007,
            271,
            9906,
            0,
            128009,
            128006,
            78191,
            128007,
            271,
        ]

        manager = InputParamManager(mock_tokenizer)

        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        token_ids = manager.tokenize_chat_request(request)

        # Verify it's a list
        assert isinstance(token_ids, list)
        # Verify all elements are integers
        assert all(isinstance(tid, int) for tid in token_ids)
        # Verify it's not empty
        assert len(token_ids) > 0
