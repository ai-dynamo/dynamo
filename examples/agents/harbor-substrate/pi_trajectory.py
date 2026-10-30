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

"""Convert Pi 0.72 JSON events to Harbor ATIF without changing raw evidence."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.models.trajectories import Trajectory


class NoCompleteMessages(ValueError):
    """The agent stopped before recording a message that ATIF can represent."""


def _text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    unsupported = {p["type"] for p in content} - {"text", "thinking", "toolCall"}
    if unsupported:
        raise ValueError(f"Unsupported Pi content types: {sorted(unsupported)}")
    return "\n".join(p["text"] for p in content if p["type"] == "text")


def convert_events(
    raw: bytes,
    *,
    agent_version: str = "unknown",
    model_name: str | None = None,
    harbor_session_id: str | None = None,
) -> dict[str, Any]:
    """Use message_end once per turn; summaries contain duplicate messages."""
    steps: list[dict[str, Any]] = []
    calls: dict[str, dict[str, Any]] = {}
    observations: dict[str, dict[str, Any]] = {}
    execution_results: dict[str, dict[str, Any]] = {}
    session: dict[str, Any] = {}
    skipped_lines: list[int] = []
    agent_end_seen = False
    last_stop = None

    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Pi stderr shares stdout; a killed tool can also leave a partial line.
            skipped_lines.append(line_number)
            continue
        if not isinstance(event, dict):
            raise ValueError(f"Pi event at line {line_number} is not an object")
        kind = event.get("type")
        if kind == "session":
            if session:
                raise ValueError("Expected one Pi session per log")
            session = event
        elif kind == "agent_end":
            agent_end_seen = True
        elif kind == "tool_execution_end":
            execution_results[event["toolCallId"]] = {
                **event["result"],
                "isError": event.get("isError", False),
            }
        elif kind == "message_end":
            message = event["message"]
            role = message["role"]
            if role == "toolResult":
                call_id = message["toolCallId"]
                if call_id in observations:
                    raise ValueError(f"Duplicate Pi tool result: {call_id}")
                observations[call_id] = message
                continue
            if role not in {"assistant", "user", "system"}:
                raise ValueError(f"Unsupported Pi message role: {role}")
            content = message["content"]
            step: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent" if role == "assistant" else role,
                "message": _text(content),
            }
            if "timestamp" in message:
                step["timestamp"] = datetime.fromtimestamp(
                    message["timestamp"] / 1000, tz=timezone.utc
                ).isoformat()
            if role == "assistant":
                step["llm_call_count"] = 1
                step["model_name"] = message.get("model") or model_name
                step["extra"] = {
                    k: message[k]
                    for k in ("stopReason", "errorMessage", "provider", "responseId")
                    if k in message
                }
                last_stop = message.get("stopReason")
                blocks = content if isinstance(content, list) else []
                reasoning = "\n".join(
                    p["thinking"] for p in blocks if p["type"] == "thinking"
                )
                if reasoning:
                    step["reasoning_content"] = reasoning
                for part in blocks:
                    if part["type"] != "toolCall":
                        continue
                    call_id = part["id"]
                    if call_id in calls:
                        raise ValueError(f"Duplicate Pi tool call: {call_id}")
                    step.setdefault("tool_calls", []).append(
                        {
                            "tool_call_id": call_id,
                            "function_name": part["name"],
                            "arguments": part["arguments"],
                        }
                    )
                    calls[call_id] = step
                usage = message.get("usage")
                if usage is not None:
                    step["metrics"] = {
                        # Pi's input excludes both cache reads and cache writes.
                        "prompt_tokens": sum(
                            usage.get(k, 0)
                            for k in ("input", "cacheRead", "cacheWrite")
                        ),
                        "completion_tokens": usage.get("output", 0),
                        "cached_tokens": usage.get("cacheRead", 0),
                        "extra": {"pi_usage": usage},
                    }
                    cost = usage.get("cost", {}).get("total", 0)
                    # Self-hosted providers report zero without a price schedule.
                    if cost > 0:
                        step["metrics"]["cost_usd"] = cost
            steps.append(step)

    for call_id, result in execution_results.items():
        # A killed process may emit execution_end without the toolResult message.
        observations.setdefault(call_id, result)
    for call_id, result in observations.items():
        if call_id not in calls:
            raise ValueError(f"Tool result has no recorded call: {call_id}")
        calls[call_id].setdefault("observation", {"results": []})["results"].append(
            {
                "source_call_id": call_id,
                "content": _text(result.get("content", [])),
                "extra": {
                    k: result[k]
                    for k in ("isError", "details", "timestamp")
                    if k in result
                },
            }
        )
    metrics = [s["metrics"] for s in steps if "metrics" in s]
    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    if metrics:
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            final_metrics[f"total_{key}"] = sum(m[key] for m in metrics)
        if all("cost_usd" in m for m in metrics):
            final_metrics["total_cost_usd"] = sum(m["cost_usd"] for m in metrics)
    pending = sorted(calls.keys() - observations.keys())
    notes = (
        "Converted from Pi JSON events. Only recorded messages are included; "
        "the system prompt, tool definitions, and suppressed stream deltas are "
        "not reconstructed. Timestamps are Pi message timestamps. "
        "Stop reasons do not imply verifier success."
    )
    if not agent_end_seen or skipped_lines or pending:
        notes += (
            " This recording is partial or contains non-JSON lines; "
            "see extra for skipped lines and unfinished calls."
        )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": session.get("id") or harbor_session_id,
        "agent": {
            "name": "pi",
            "version": agent_version,
            "model_name": next(
                (s["model_name"] for s in steps if s.get("model_name")), model_name
            ),
        },
        "steps": steps,
        "final_metrics": final_metrics,
        "notes": notes,
        "extra": {
            "converter": "pi_trajectory.py/v1",
            "source_file": "pi.txt",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "pi_session": session,
            "harbor_session_id": harbor_session_id,
            "agent_end_seen": agent_end_seen,
            "last_stop_reason": last_stop,
            "skipped_non_json_lines": skipped_lines,
            "unfinished_tool_call_ids": pending,
        },
    }
    if not steps:
        raise NoCompleteMessages("No complete Pi messages recorded")
    return Trajectory.model_validate(trajectory).to_json_dict()


def write_trajectory(
    log_path: Path,
    *,
    overwrite: bool = False,
    agent_version: str | None = None,
    model_name: str | None = None,
    harbor_session_id: str | None = None,
) -> Path | None:
    """Write a validated sibling trajectory.json; never modify pi.txt or rewards."""
    output = log_path.with_name("trajectory.json")
    if output.exists() and not overwrite:
        return None
    config_path = log_path.parent.parent / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    agent = config.get("agent", {})
    try:
        trajectory = convert_events(
            log_path.read_bytes(),
            agent_version=agent_version
            or agent.get("kwargs", {}).get("version")
            or "unknown",
            model_name=model_name or agent.get("model_name"),
            harbor_session_id=harbor_session_id,
        )
    except NoCompleteMessages:
        return None
    # The viewer may read while backfill runs. Publish a complete file atomically.
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Pi log, trial, job, or jobs directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing ATIF"
    )
    args = parser.parse_args()
    logs = (
        [args.path] if args.path.is_file() else sorted(args.path.rglob("agent/pi.txt"))
    )
    if not logs:
        parser.error("No agent/pi.txt logs found")
    written = 0
    for log in logs:
        output = write_trajectory(log, overwrite=args.overwrite)
        if output:
            written += 1
            print(output)
    print(
        f"Wrote {written} trajectories; skipped {len(logs) - written} existing files."
    )


if __name__ == "__main__":
    main()
