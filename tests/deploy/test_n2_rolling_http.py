# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

import pytest

from tests.deploy.n2.rolling_http import (
    ContractError,
    validate_chat,
    validate_embedding,
    validate_stream,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
]


class RollingHttpContracts(unittest.TestCase):
    def test_float_contract_rejects_base64_and_nonfinite(self):
        body = {
            "object": "list",
            "data": [{"index": 0, "object": "embedding", "embedding": [1.0, 2.0]}],
        }
        validate_embedding(body, 1, 2)
        for value in ("AACAPwAAAEA=", [float("nan"), 2], [True, 2], [1.0]):
            body["data"][0]["embedding"] = value
            with self.assertRaises(ContractError):
                validate_embedding(body, 1, 2)

    def test_chat_contract(self):
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 2},
        }
        validate_chat(body, 32)
        with self.assertRaises(ContractError):
            validate_chat(body, 1)
        body["choices"][0]["finish_reason"] = "error"
        with self.assertRaises(ContractError):
            validate_chat(body, 32)

    def test_stream_must_finish_and_not_hide_errors(self):
        def chunk(delta, finish):
            return "data: " + json.dumps(
                {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            )

        valid = [chunk({"content": "Hello"}, None), chunk({}, "stop"), "data: [DONE]"]
        validate_stream(valid)
        for invalid in [
            valid[:-1],
            valid[:1] + ["data: [DONE]"],
            ['data: {"error":"worker failed"}'],
            ["event: error"],
            valid + [valid[0]],
        ]:
            with self.assertRaises(ContractError):
                validate_stream(invalid)
