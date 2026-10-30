# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loss, duplication, interruption, and provenance checks for Pi ATIF export."""

import json

import pytest
from harbor.utils.trajectory_validator import TrajectoryValidator
from pi_trajectory import convert_events, write_trajectory

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]


def encode(events):
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


@pytest.fixture
def events():
    assistant = {
        "role": "assistant",
        "model": "test-model",
        "timestamp": 1000,
        "stopReason": "toolUse",
        "content": [
            {"type": "thinking", "thinking": "Inspect both files."},
            {"type": "text", "text": "Running two checks."},
            {
                "type": "toolCall",
                "id": "a",
                "name": "bash",
                "arguments": {"command": "ls"},
            },
            {
                "type": "toolCall",
                "id": "b",
                "name": "read",
                "arguments": {"path": "missing"},
            },
        ],
        "usage": {
            "input": 10,
            "cacheRead": 20,
            "cacheWrite": 5,
            "output": 7,
            "cost": {"total": 0},
        },
    }
    return [
        {"type": "session", "version": 3, "id": "pi-session"},
        {
            "type": "message_end",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Fix this."}],
            },
        },
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Running"},
        },
        {"type": "message_end", "message": assistant},
        {
            "type": "tool_execution_end",
            "toolCallId": "b",
            "result": {"content": [{"type": "text", "text": "missing file"}]},
            "isError": True,
        },
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolCallId": "b",
                "isError": True,
                "content": [{"type": "text", "text": "missing file"}],
            },
        },
        {
            "type": "message_end",
            "message": {
                "role": "toolResult",
                "toolCallId": "a",
                "isError": False,
                "content": [{"type": "text", "text": "README.md"}],
            },
        },
        {"type": "turn_end", "message": assistant},
        {"type": "agent_end", "messages": [assistant]},
    ]


def test_one_inference_not_counted_three_times_and_results_match_calls(events):
    result = convert_events(encode(events), agent_version="0.72.1")
    validator = TrajectoryValidator()
    assert validator.validate(result), validator.errors
    assert len(result["steps"]) == 2
    step = result["steps"][1]
    assert step["reasoning_content"] == "Inspect both files."
    assert step["timestamp"] == "1970-01-01T00:00:01+00:00"
    assert step["message"] == "Running two checks."
    assert len(step["tool_calls"]) == 2
    observations = {r["source_call_id"]: r for r in step["observation"]["results"]}
    assert observations["a"]["content"] == "README.md"
    assert observations["b"]["content"] == "missing file"
    assert observations["b"]["extra"]["isError"] is True
    assert result["final_metrics"]["total_prompt_tokens"] == 35
    assert result["final_metrics"]["total_completion_tokens"] == 7
    assert result["final_metrics"]["total_cached_tokens"] == 20
    assert "total_cost_usd" not in result["final_metrics"]
    assert result["extra"]["agent_end_seen"] is True


def test_timeout_retains_unfinished_call_and_execution_result_without_message(events):
    raw = encode(events[:6]) + b'{"type":"tool_execution_update"'
    result = convert_events(raw)
    assert result["extra"]["agent_end_seen"] is False
    assert result["extra"]["unfinished_tool_call_ids"] == ["a"]
    assert result["extra"]["skipped_non_json_lines"] == [7]
    observations = result["steps"][1]["observation"]["results"]
    assert len(observations) == 1
    assert observations[0]["source_call_id"] == "b"
    assert observations[0]["extra"]["isError"] is True
    assert "partial" in result["notes"]


@pytest.mark.parametrize("stop", ["length", "error", "aborted"])
def test_abnormal_stop_and_error_message_are_preserved(events, stop):
    events[4]["message"]["stopReason"] = stop
    events[4]["message"]["errorMessage"] = "recorded failure"
    result = convert_events(encode(events))
    assert result["extra"]["last_stop_reason"] == stop
    assert result["steps"][1]["extra"]["errorMessage"] == "recorded failure"


def test_unmatched_observation_fails_instead_of_associating_wrong_tool(events):
    events[6]["message"]["toolCallId"] = "unknown"
    with pytest.raises(ValueError, match="no recorded call"):
        convert_events(encode(events))


def test_unknown_content_is_not_silently_dropped(events):
    events[1]["message"]["content"].append({"type": "image", "data": "abc"})
    with pytest.raises(ValueError, match="Unsupported Pi content"):
        convert_events(encode(events))


def test_write_preserves_inputs_rewards_and_existing_atif(tmp_path, events):
    agent = tmp_path / "agent"
    agent.mkdir()
    raw = encode(events)
    log = agent / "pi.txt"
    log.write_bytes(raw)
    (tmp_path / "config.json").write_text(
        json.dumps({"agent": {"kwargs": {"version": "0.72.1"}}})
    )
    reward = tmp_path / "result.json"
    reward.write_text('{"reward": 0}')
    output = write_trajectory(log)
    assert output == agent / "trajectory.json"
    assert json.loads(output.read_text())["agent"]["version"] == "0.72.1"
    original = output.read_bytes()
    assert write_trajectory(log) is None
    assert output.read_bytes() == original
    assert log.read_bytes() == raw
    assert reward.read_text() == '{"reward": 0}'
    assert not list(agent.glob("*.tmp"))


def test_truncated_utf8_preserves_completed_events(events):
    raw = encode(events[:5]) + b'{"type":"message_update","delta":"\xe2\x82'
    result = convert_events(raw)
    assert len(result["steps"]) == 2
    assert result["extra"]["skipped_non_json_lines"] == [6]


def test_empty_interrupted_log_does_not_block_verification(tmp_path):
    raw = b'{"type":"session","id":"started"}\n' + b"\xe2\x82"
    log = tmp_path / "pi.txt"
    log.write_bytes(raw)
    assert write_trajectory(log) is None
    assert not (tmp_path / "trajectory.json").exists()
    assert log.read_bytes() == raw
