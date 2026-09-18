# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

if importlib.util.find_spec("acp") is None:
    acp = ModuleType("acp")
    acp.PROTOCOL_VERSION = 1
    acp.spawn_agent_process = None
    acp.text_block = lambda text: text
    sys.modules["acp"] = acp

drive_harness = importlib.import_module("drive_harness")


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
