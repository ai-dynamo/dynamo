# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tool_calling_probe as probe  # noqa: E402


class KimiK3ProfileTests(unittest.TestCase):
    def test_profile_contains_152_two_mode_records(self) -> None:
        cases = probe.build_cases("kimi_k3")
        self.assertEqual(len(cases), 76)
        self.assertEqual(len(cases) * 2, 152)
        self.assertEqual(sum(case.case_id.startswith("k3_tool_") for case in cases), 42)
        self.assertEqual(
            sum(case.case_id.startswith("k3_reasoning_") for case in cases), 34
        )
        self.assertEqual(sum(case.scripted_followup is not None for case in cases), 5)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_exact_arguments_and_reserved_markers_are_validated(self) -> None:
        case = next(
            case
            for case in probe.build_cases("kimi_k3")
            if case.case_id == "k3_tool_core_required_single_disabled"
        )
        result = probe.ChatResult(
            tool_calls=[
                {
                    "id": "functions.add_numbers:0",
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "arguments": json.dumps({"a": 17, "b": 19}),
                    },
                }
            ],
            finish_reason="tool_calls",
        )
        errors, _warnings = probe.validate_result(case, result)
        self.assertEqual(errors, [])

        result.content = "42<|close|>response<|sep|>"
        errors, _warnings = probe.validate_result(case, result)
        self.assertIn(
            "reserved_marker_leaked_to_content",
            {item["kind"] for item in errors},
        )

    def test_scripted_followup_preserves_tool_call_ids(self) -> None:
        case = next(
            case
            for case in probe.build_cases("kimi_k3")
            if case.case_id == "k3_tool_lifecycle_single_result"
        )
        first = probe.ChatResult(
            reasoning_content="Use the weather tool.",
            tool_calls=[
                {
                    "id": "functions.get_weather:0",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris", "unit": "celsius"}),
                    },
                }
            ],
            finish_reason="tool_calls",
        )
        second = probe.ChatResult(
            content="PARIS_SUNNY",
            reasoning_content="The result says sunny.",
            finish_reason="stop",
        )
        captured_messages: list[list[dict]] = []

        def fake_request_chat(_case, _mode, *, messages, **_kwargs):
            captured_messages.append(messages)
            result = first if len(captured_messages) == 1 else second
            return result, {"messages": messages}

        with mock.patch.object(probe, "request_chat", side_effect=fake_request_chat):
            record = probe.run_scripted_followup_case(
                case,
                "nonstream",
                iteration=1,
                url="http://example.test/v1/chat/completions",
                api_key=None,
                model="moonshotai/Kimi-K3",
                temperature=0,
                max_tokens=4096,
                timeout=10,
                extra_headers={},
                raw_chars=20000,
                record_success_raw=False,
            )

        self.assertTrue(record["pass"], record["errors"])
        self.assertEqual(len(captured_messages), 2)
        tool_message = captured_messages[1][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "functions.get_weather:0")


if __name__ == "__main__":
    unittest.main()
