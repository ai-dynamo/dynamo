# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
