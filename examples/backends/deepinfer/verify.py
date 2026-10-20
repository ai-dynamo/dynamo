# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real HTTP → Dynamo sidecar → OpenEngine → mocked DeepInfer output."""

import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def wait_ready(url, model, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with HTTP.open(url + "/v1/models", timeout=2) as response:
                models = json.load(response)["data"]
            selected = [
                item["id"] for item in models if model is None or item["id"] == model
            ]
            if selected:
                return selected[0]
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"Model not ready at {url}: {last_error}")


def payload(model, scenario, thinking=False, stream=False, max_tokens=256):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "mock:" + scenario}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "chat_template_kwargs": {"thinking": thinking},
    }
    if scenario == "tool":
        body.update(tools=TOOLS, tool_choice="auto")
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def open_request(url, body, endpoint="/v1/chat/completions"):
    try:
        return HTTP.open(
            urllib.request.Request(
                url + endpoint,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=30,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def request(url, body):
    with open_request(url, body) as response:
        if not body["stream"]:
            result = json.load(response)
            choice = result["choices"][0]
            return {
                "message": choice["message"],
                "finish_reason": choice["finish_reason"],
                "usage": result["usage"],
                **({"logprobs": choice["logprobs"]} if choice.get("logprobs") else {}),
            }
        content, reasoning, calls, terminals, usage = [], [], {}, [], None
        done = False
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice["delta"]
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                for call in delta.get("tool_calls") or []:
                    target = calls.setdefault(
                        call["index"], {"name": "", "arguments": ""}
                    )
                    function = call.get("function") or {}
                    target["name"] += function.get("name") or ""
                    target["arguments"] += function.get("arguments") or ""
                if choice.get("finish_reason"):
                    terminals.append(choice["finish_reason"])
        assert done and len(terminals) == 1, (done, terminals)
        assert sorted(calls) == list(range(len(calls))), calls
        return {
            "message": {
                "content": "".join(content),
                "reasoning_content": "".join(reasoning),
                "tool_calls": [{"function": calls[index]} for index in sorted(calls)],
            },
            "finish_reason": terminals[0],
            "usage": usage,
        }


def check(result, scenario, thinking):
    message = result["message"]
    reasoning = message.get("reasoning_content") or ""
    assert reasoning == (
        "I will check the request carefully." if thinking else ""
    ), result
    content = message.get("content") or ""
    assert (
        "DSML" not in content and "<think>" not in content and "</think>" not in content
    ), result
    if scenario == "tool":
        calls = message["tool_calls"]
        assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", result
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "city": "Tokyo"
        }, result
        assert result["finish_reason"] == "tool_calls", result
    else:
        expected = {
            "plain": "Hello from DeepInfer mock.",
            "reasoning": "42",
            "unicode": "東京 café 𐍈",
        }[scenario]
        assert content == expected and not message.get("tool_calls"), result
        assert result["finish_reason"] == "stop", result
    usage = result["usage"]
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, result
    assert (
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    ), result


def probe(args):
    return json.loads(
        subprocess.check_output(
            [str(args.server_bin), "probe", "--endpoint", args.engine_endpoint],
            text=True,
            timeout=5,
        )
    )


def verify(args, model):
    results = []
    for stream in (False, True):
        for scenario, thinking in (
            ("plain", False),
            ("reasoning", True),
            ("tool", False),
            ("tool", True),
            ("unicode", False),
        ):
            result = request(args.url, payload(model, scenario, thinking, stream))
            check(result, scenario, thinking)
            results.append(
                {
                    "scenario": scenario,
                    "thinking": thinking,
                    "stream": stream,
                    "result": result,
                }
            )
    limited = request(args.url, payload(model, "plain", max_tokens=1))
    assert (
        limited["finish_reason"] == "length"
        and limited["usage"]["completion_tokens"] == 1
    ), limited
    results.append({"scenario": "token-budget", "result": limited})
    body = payload(model, "plain")
    body.update(logprobs=True, top_logprobs=0)
    logged = request(args.url, body)
    check(logged, "plain", False)
    # Dynamo's unified chat parser drops logprobs after rewriting token text.
    assert "logprobs" not in logged, logged
    results.append({"scenario": "parsed-chat-logprobs-suppressed", "result": logged})
    with open_request(
        args.url,
        {"model": model, "prompt": "mock:plain", "max_tokens": 64, "logprobs": 0},
        endpoint="/v1/completions",
    ) as response:
        completion = json.load(response)
    choice = completion["choices"][0]
    assert choice["text"] == "Hello from DeepInfer mock.", completion
    assert choice["logprobs"]["token_logprobs"], completion
    assert all(
        value == 0.0 for value in choice["logprobs"]["token_logprobs"]
    ), completion
    results.append(
        {"scenario": "completion-selected-token-logprobs", "result": completion}
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(
                request,
                args.url,
                payload(model, "reasoning" if i % 2 else "plain", bool(i % 2), True),
            )
            for i in range(16)
        ]
        for i, future in enumerate(futures):
            result = future.result()
            check(result, "reasoning" if i % 2 else "plain", bool(i % 2))
            results.append({"scenario": "concurrent", "result": result})
    if args.server_bin and args.engine_endpoint:
        response = open_request(
            args.url, payload(model, "slow", stream=True, max_tokens=4096)
        )
        try:
            for line in response:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    event = json.loads(line[6:])
                    if any(c["delta"].get("content") for c in event.get("choices", [])):
                        break
            load_before = probe(args)["running_requests"]
            assert load_before > 0
        finally:
            response.close()
        disconnected = time.monotonic()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            load_after = probe(args)["running_requests"]
            if load_after == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                "client disconnect did not release the mock engine request"
            )
        cancellation_seconds = time.monotonic() - disconnected
        recovered = request(args.url, payload(model, "plain"))
        check(recovered, "plain", False)
        results.append(
            {
                "scenario": "disconnect-cancellation-and-recovery",
                "load_before_disconnect": load_before,
                "load_after_disconnect": load_after,
                "cancellation_seconds": cancellation_seconds,
                "result": recovered,
            }
        )
    summary = {
        "passed": len(results),
        "model": model,
        "chain": "HTTP frontend → Dynamo sidecar → Python-hosted Rust OpenEngine → PyO3 → DeepInfer mock",
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "results"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--wait-only", action="store_true")
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--engine-endpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = wait_ready(args.url, args.model, args.timeout)
    if args.wait_only:
        print(f"Model ready: {model}")
    else:
        verify(args, model)
