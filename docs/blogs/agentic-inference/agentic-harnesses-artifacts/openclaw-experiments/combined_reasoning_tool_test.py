# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Combined reasoning + tool calling experiment through Dynamo's Anthropic API.

Tests that a single response correctly contains both a `thinking` block and
a `tool_use` block, surfaced through the Anthropic Messages API with:
  - reasoning parser: nemotron_deci
  - tool call parser: qwen3_coder

Runs 5 rounds each of non-streaming and streaming requests, capturing full
response structures and timing data.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000"
MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
OUTPUT_FILE = "/home/scratch.mkosec_hw/openclaw-exp/combined-reasoning-tool.jsonl"
NUM_ROUNDS = 5

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '15 * 23'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for",
                }
            },
            "required": ["city"],
        },
    },
]

SYSTEM_PROMPT = "You are a helpful assistant with access to tools. Think step by step before using tools."

USER_MESSAGE = (
    "Think carefully about what 15 * 23 equals, then use the calculator to verify."
)


def build_request_body(stream=False):
    return {
        "model": MODEL,
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": USER_MESSAGE}],
        "stream": stream,
    }


def send_non_streaming_request():
    """Send a non-streaming request and return (response_json, elapsed_ms)."""
    body = build_request_body(stream=False)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/messages",
        data=data,
        headers={"Content-Type": "application/json", "x-api-key": "dummy"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            elapsed_ms = (time.monotonic() - t0) * 1000
            return json.loads(raw), elapsed_ms
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        error_body = e.read().decode("utf-8")
        return {"error": error_body, "status": e.code}, elapsed_ms


def send_streaming_request():
    """Send a streaming request and return (events_list, elapsed_ms, raw_lines)."""
    body = build_request_body(stream=True)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/messages",
        data=data,
        headers={"Content-Type": "application/json", "x-api-key": "dummy"},
        method="POST",
    )
    events = []
    raw_lines = []
    t0 = time.monotonic()
    first_event_time = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            current_event_type = None
            current_data = ""
            for line_bytes in resp:
                line = line_bytes.decode("utf-8").rstrip("\n").rstrip("\r")
                raw_lines.append(line)
                if line.startswith("event: "):
                    current_event_type = line[7:].strip()
                elif line.startswith("data: "):
                    current_data = line[6:]
                elif line == "" and current_event_type:
                    now = time.monotonic()
                    if first_event_time is None:
                        first_event_time = now
                    event_record = {
                        "event": current_event_type,
                        "time_ms": (now - t0) * 1000,
                    }
                    try:
                        event_record["data"] = json.loads(current_data)
                    except json.JSONDecodeError:
                        event_record["data_raw"] = current_data
                    events.append(event_record)
                    current_event_type = None
                    current_data = ""
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        error_body = e.read().decode("utf-8")
        return [{"error": error_body, "status": e.code}], elapsed_ms, raw_lines

    elapsed_ms = (time.monotonic() - t0) * 1000
    return events, elapsed_ms, raw_lines


def analyze_non_streaming(response):
    """Analyze a non-streaming response for correctness."""
    result = {
        "has_thinking": False,
        "has_tool_use": False,
        "correct_order": False,
        "stop_reason_is_tool_use": False,
        "tool_name": None,
        "tool_input": None,
        "thinking_length": 0,
        "content_block_types": [],
    }

    if "error" in response:
        result["error"] = response["error"]
        return result

    content = response.get("content", [])
    result["stop_reason_is_tool_use"] = response.get("stop_reason") == "tool_use"

    thinking_idx = -1
    tool_use_idx = -1

    for i, block in enumerate(content):
        block_type = block.get("type")
        result["content_block_types"].append(block_type)
        if block_type == "thinking":
            result["has_thinking"] = True
            thinking_idx = i
            result["thinking_length"] = len(block.get("thinking", ""))
        elif block_type == "tool_use":
            result["has_tool_use"] = True
            tool_use_idx = i
            result["tool_name"] = block.get("name")
            result["tool_input"] = block.get("input")

    if thinking_idx >= 0 and tool_use_idx >= 0:
        result["correct_order"] = thinking_idx < tool_use_idx

    return result


def analyze_streaming(events):
    """Analyze streaming events for correctness and timing."""
    result = {
        "total_events": len(events),
        "thinking_block_start_ms": None,
        "tool_use_block_start_ms": None,
        "tool_call_dispatch_events": [],
        "content_block_types_in_order": [],
        "message_stop_ms": None,
        "has_thinking": False,
        "has_tool_use": False,
        "has_dispatch": False,
        "correct_order": False,
        "event_type_sequence": [],
    }

    for ev in events:
        if "error" in ev:
            result["error"] = ev["error"]
            return result

        etype = ev.get("event", "")
        result["event_type_sequence"].append(etype)

        if etype == "content_block_start":
            data = ev.get("data", {})
            cb = data.get("content_block", {})
            cb_type = cb.get("type")
            result["content_block_types_in_order"].append(cb_type)
            if cb_type == "thinking":
                result["has_thinking"] = True
                result["thinking_block_start_ms"] = ev["time_ms"]
            elif cb_type == "tool_use":
                result["has_tool_use"] = True
                result["tool_use_block_start_ms"] = ev["time_ms"]

        elif etype == "tool_call_dispatch":
            result["has_dispatch"] = True
            result["tool_call_dispatch_events"].append(
                {"time_ms": ev["time_ms"], "data": ev.get("data")}
            )

        elif etype == "message_stop":
            result["message_stop_ms"] = ev["time_ms"]

    if (
        result["thinking_block_start_ms"] is not None
        and result["tool_use_block_start_ms"] is not None
    ):
        result["correct_order"] = (
            result["thinking_block_start_ms"] < result["tool_use_block_start_ms"]
        )

    return result


def main():
    import os

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    all_results = []

    # Quick health check
    print("Checking health...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            print(f"Models: {json.dumps(models, indent=2)}")
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

    # Non-streaming rounds
    print(f"\n{'='*60}")
    print(f"NON-STREAMING ROUNDS ({NUM_ROUNDS})")
    print(f"{'='*60}")

    for i in range(NUM_ROUNDS):
        print(f"\n--- Round {i+1}/{NUM_ROUNDS} (non-streaming) ---")
        response, elapsed_ms = send_non_streaming_request()
        analysis = analyze_non_streaming(response)

        record = {
            "type": "non_streaming",
            "round": i + 1,
            "elapsed_ms": round(elapsed_ms, 2),
            "analysis": analysis,
            "response": response,
        }
        all_results.append(record)

        print(f"  Elapsed: {elapsed_ms:.0f}ms")
        print(
            f"  Has thinking: {analysis['has_thinking']} ({analysis['thinking_length']} chars)"
        )
        print(f"  Has tool_use: {analysis['has_tool_use']}")
        print(f"  Correct order: {analysis['correct_order']}")
        print(f"  stop_reason==tool_use: {analysis['stop_reason_is_tool_use']}")
        print(f"  Tool: {analysis['tool_name']}({json.dumps(analysis['tool_input'])})")
        print(f"  Content types: {analysis['content_block_types']}")

        if i == 0:
            # Print full response for first round
            print("\n  Full response (round 1):")
            print(json.dumps(response, indent=2))

    # Streaming rounds
    print(f"\n{'='*60}")
    print(f"STREAMING ROUNDS ({NUM_ROUNDS})")
    print(f"{'='*60}")

    for i in range(NUM_ROUNDS):
        print(f"\n--- Round {i+1}/{NUM_ROUNDS} (streaming) ---")
        events, elapsed_ms, raw_lines = send_streaming_request()
        analysis = analyze_streaming(events)

        record = {
            "type": "streaming",
            "round": i + 1,
            "elapsed_ms": round(elapsed_ms, 2),
            "analysis": analysis,
            "events_summary": [
                {"event": ev.get("event"), "time_ms": round(ev.get("time_ms", 0), 2)}
                for ev in events
            ],
        }
        # Include full events for first round only
        if i == 0:
            record["events_full"] = events
            record["raw_lines_sample"] = raw_lines[:100]

        all_results.append(record)

        print(f"  Elapsed: {elapsed_ms:.0f}ms")
        print(f"  Total events: {analysis['total_events']}")
        print(f"  Has thinking: {analysis['has_thinking']}")
        print(f"  Has tool_use: {analysis['has_tool_use']}")
        print(f"  Has dispatch: {analysis['has_dispatch']}")
        print(f"  Correct order: {analysis['correct_order']}")
        print(f"  Content types: {analysis['content_block_types_in_order']}")
        if analysis["thinking_block_start_ms"] is not None:
            print(
                f"  Thinking block start: {analysis['thinking_block_start_ms']:.0f}ms"
            )
        if analysis["tool_use_block_start_ms"] is not None:
            print(
                f"  Tool_use block start: {analysis['tool_use_block_start_ms']:.0f}ms"
            )
        if analysis["message_stop_ms"] is not None:
            print(f"  Message stop: {analysis['message_stop_ms']:.0f}ms")
        if analysis["tool_call_dispatch_events"]:
            for d in analysis["tool_call_dispatch_events"]:
                print(f"  Dispatch event at: {d['time_ms']:.0f}ms")

        if i == 0:
            # Print event type sequence for first round
            print("\n  Event type sequence (round 1):")
            for ev in events:
                etype = ev.get("event", "")
                t = ev.get("time_ms", 0)
                extra = ""
                if etype == "content_block_start":
                    cb = ev.get("data", {}).get("content_block", {})
                    extra = f" type={cb.get('type')}"
                elif etype == "tool_call_dispatch":
                    td = ev.get("data", {})
                    tc = td.get("tool_call", {})
                    fn = tc.get("function", {})
                    extra = f" name={fn.get('name')}"
                print(f"    {t:8.1f}ms  {etype}{extra}")

    # Write results
    with open(OUTPUT_FILE, "w") as f:
        for record in all_results:
            f.write(json.dumps(record) + "\n")
    print(f"\nResults written to {OUTPUT_FILE}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    ns_results = [r for r in all_results if r["type"] == "non_streaming"]
    s_results = [r for r in all_results if r["type"] == "streaming"]

    ns_thinking = sum(1 for r in ns_results if r["analysis"]["has_thinking"])
    ns_tooluse = sum(1 for r in ns_results if r["analysis"]["has_tool_use"])
    ns_order = sum(1 for r in ns_results if r["analysis"]["correct_order"])
    ns_stop = sum(1 for r in ns_results if r["analysis"]["stop_reason_is_tool_use"])
    ns_times = [r["elapsed_ms"] for r in ns_results]

    print(f"\nNon-streaming ({NUM_ROUNDS} rounds):")
    print(f"  Has thinking: {ns_thinking}/{NUM_ROUNDS}")
    print(f"  Has tool_use: {ns_tooluse}/{NUM_ROUNDS}")
    print(f"  Correct order (thinking before tool_use): {ns_order}/{NUM_ROUNDS}")
    print(f"  stop_reason==tool_use: {ns_stop}/{NUM_ROUNDS}")
    print(
        f"  Latency: min={min(ns_times):.0f}ms  max={max(ns_times):.0f}ms  mean={sum(ns_times)/len(ns_times):.0f}ms"
    )

    s_thinking = sum(1 for r in s_results if r["analysis"]["has_thinking"])
    s_tooluse = sum(1 for r in s_results if r["analysis"]["has_tool_use"])
    s_order = sum(1 for r in s_results if r["analysis"]["correct_order"])
    s_dispatch = sum(1 for r in s_results if r["analysis"]["has_dispatch"])
    s_times = [r["elapsed_ms"] for r in s_results]

    print(f"\nStreaming ({NUM_ROUNDS} rounds):")
    print(f"  Has thinking: {s_thinking}/{NUM_ROUNDS}")
    print(f"  Has tool_use: {s_tooluse}/{NUM_ROUNDS}")
    print(f"  Correct order (thinking before tool_use): {s_order}/{NUM_ROUNDS}")
    print(f"  Has dispatch event: {s_dispatch}/{NUM_ROUNDS}")
    print(
        f"  Latency: min={min(s_times):.0f}ms  max={max(s_times):.0f}ms  mean={sum(s_times)/len(s_times):.0f}ms"
    )

    # Timing details for streaming
    thinking_starts = [
        r["analysis"]["thinking_block_start_ms"]
        for r in s_results
        if r["analysis"]["thinking_block_start_ms"] is not None
    ]
    tool_starts = [
        r["analysis"]["tool_use_block_start_ms"]
        for r in s_results
        if r["analysis"]["tool_use_block_start_ms"] is not None
    ]
    msg_stops = [
        r["analysis"]["message_stop_ms"]
        for r in s_results
        if r["analysis"]["message_stop_ms"] is not None
    ]

    if thinking_starts:
        print(
            f"\n  Thinking block start: mean={sum(thinking_starts)/len(thinking_starts):.0f}ms"
        )
    if tool_starts:
        print(f"  Tool_use block start: mean={sum(tool_starts)/len(tool_starts):.0f}ms")
    if msg_stops:
        print(f"  Message stop: mean={sum(msg_stops)/len(msg_stops):.0f}ms")
    if thinking_starts and tool_starts:
        gaps = [t - th for th, t in zip(thinking_starts, tool_starts)]
        print(f"  Gap (thinking -> tool_use): mean={sum(gaps)/len(gaps):.0f}ms")

    dispatch_times = []
    for r in s_results:
        for d in r["analysis"].get("tool_call_dispatch_events", []):
            dispatch_times.append(d["time_ms"])
    if dispatch_times and msg_stops:
        dispatch_before_stop = [s - d for d, s in zip(dispatch_times, msg_stops)]
        print(
            f"  Dispatch before message_stop: mean={sum(dispatch_before_stop)/len(dispatch_before_stop):.0f}ms"
        )


if __name__ == "__main__":
    main()
