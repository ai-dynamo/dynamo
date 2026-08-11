# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import drive_harness


class EmptyClient:
    def start_turn(self):
        pass

    def response(self):
        return ""


class EmptyConnection:
    async def prompt(self, **kwargs):
        return None


class PromptTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_response_emits_error_without_raising(self):
        with patch.object(drive_harness, "emit") as emit:
            await drive_harness.prompt(
                EmptyConnection(), EmptyClient(), "session-1", "hello"
            )

        emit.assert_called_once_with(
            {
                "type": "error",
                "session_id": "session-1",
                "ok": False,
                "error": "agent returned no text response",
            }
        )


if __name__ == "__main__":
    unittest.main()
