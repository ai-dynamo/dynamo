"""
Standalone smoke test for Dynamo-compatible streaming sessions.

Flow:
1. Send a few normal multi-turn chat completion requests.
2. Start a streaming session with `nvext.session_control.action="open"`.
3. Send a few more session-backed chat completion requests.
4. End the session with `nvext.session_control.action="close"`.
5. Send a few more normal chat completion requests.

Example:
    uv run python ~/memory/agentic-cache-control/benchmarks/scripts/streaming_session_smoke.py \
        --base-url http://localhost:8000 \
        --model Qwen/Qwen3-0.6B
"""

import argparse
import json
import time
import uuid
from typing import Any, Optional

import requests


PRE_SESSION_MESSAGES = [
    "Reply with a short greeting.",
    "Reply with a short sentence about Paris.",
]

SESSION_MESSAGES = [
    "My name is Bob. Reply with exactly one short acknowledgment.",
    "What is my name? Reply with one word if you know it.",
    "Now reply with a short sentence about what we just established.",
    "Repeat my name in one word.",
    "What city did we not discuss in this session? Reply with unknown.",
    "Reply with exactly two words acknowledging the session.",
    "What is my name? Reply with one word.",
    "Reply with a short sentence proving you still remember my name.",
]

CLOSE_SESSION_MESSAGE = "Reply with exactly the word closing."

POST_SESSION_MESSAGES = [
    "Reply with a short sentence about Rome.",
    "Reply with a short sentence about Berlin.",
]


def _get_model(base_url: str) -> str:
    response = requests.get(f"{base_url}/v1/models", timeout=30)
    response.raise_for_status()
    body = response.json()
    return body["data"][0]["id"]


def _chat_completion(
    base_url: str,
    model: str,
    message: str,
    session_id: Optional[str] = None,
    session_action: Optional[str] = None,
    max_tokens: int = 32,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if session_id is not None:
        session_control = {"session_id": session_id}
        if session_action is not None:
            session_control["action"] = session_action
            if session_action == "open":
                session_control["timeout"] = 60
        payload["nvext"] = {"session_control": session_control}

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _run_plain_requests(base_url: str, model: str, messages: list[str], label: str) -> None:
    print(f"{label}:")
    for idx, message in enumerate(messages, start=1):
        body = _chat_completion(base_url, model, message)
        print(f"  req {idx}:")
        print(json.dumps(body, indent=2, sort_keys=True))


def run_smoke_test(
    base_url: str,
    model: str,
    max_tokens: int,
    session_delay_seconds: float,
    pre_close_delay_seconds: float,
) -> None:
    _run_plain_requests(base_url, model, PRE_SESSION_MESSAGES, "pre-session requests")

    session_id = f"smoke-{uuid.uuid4().hex[:12]}"
    print(f"session id: {session_id}")

    for idx, message in enumerate(SESSION_MESSAGES, start=1):
        action = "open" if idx == 1 else None
        body = _chat_completion(
            base_url=base_url,
            model=model,
            message=message,
            session_id=session_id,
            session_action=action,
            max_tokens=max_tokens,
        )
        print(f"  session req {idx}:")
        print(json.dumps(body, indent=2, sort_keys=True))
        if idx != len(SESSION_MESSAGES):
            print(f"  sleeping {session_delay_seconds:.1f}s before next session request")
            time.sleep(session_delay_seconds)

    print(f"holding session open for {pre_close_delay_seconds:.1f}s before close")
    time.sleep(pre_close_delay_seconds)

    body = _chat_completion(
        base_url=base_url,
        model=model,
        message=CLOSE_SESSION_MESSAGE,
        session_id=session_id,
        session_action="close",
        max_tokens=max_tokens,
    )
    print("  session close req:")
    print(json.dumps(body, indent=2, sort_keys=True))
    print("closed session")

    _run_plain_requests(base_url, model, POST_SESSION_MESSAGES, "post-session requests")
    print("streaming session smoke test passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone smoke test for streaming sessions via /v1/chat/completions."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", help="Model ID. If omitted, auto-detect from /v1/models.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--session-delay-seconds", type=float, default=1.0)
    parser.add_argument("--pre-close-delay-seconds", type=float, default=3.0)
    args = parser.parse_args()

    model = args.model or _get_model(args.base_url)
    print(f"using model: {model}")
    run_smoke_test(
        base_url=args.base_url,
        model=model,
        max_tokens=args.max_tokens,
        session_delay_seconds=args.session_delay_seconds,
        pre_close_delay_seconds=args.pre_close_delay_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
