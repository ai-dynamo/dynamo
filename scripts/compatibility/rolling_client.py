# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-cluster HTTP client; never retries or decodes an incompatible response.

Only this client imports the checkout's contracts. Frontend/worker Pods run
unmodified images. A stable client Pod uses the Kubernetes Service directly,
so replacing frontend Pods cannot break an artificial port-forward tunnel.
"""

import argparse
import json
import time
from pathlib import Path

import requests
from runner import ContractError, validate_chat, validate_embedding, validate_stream


def payloads(scenario, model):
    if scenario == "embedding":
        return [
            ("default", {"model": model, "input": "hello"}),
            ("float", {"model": model, "input": "hello", "encoding_format": "float"}),
            (
                "batch",
                {
                    "model": model,
                    "input": ["hello", "world"],
                    "encoding_format": "float",
                },
            ),
        ]
    base = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with a short greeting."}],
        "temperature": 0,
    }
    # A stop prefix derived on one worker is not necessarily reproduced on a
    # different engine version. Keep that same-worker test in the static suite.
    return [
        ("unary", {**base, "max_tokens": 32}),
        ("stream", {**base, "max_tokens": 32, "stream": True}),
        ("limited", {**base, "max_tokens": 1}),
    ]


def request_once(base, scenario, body, dimensions):
    record = {"request": body, "started": time.time(), "status": "failed"}
    path = "/v1/embeddings" if scenario == "embedding" else "/v1/chat/completions"
    try:
        with requests.post(
            base + path,
            json=body,
            stream=body.get("stream", False),
            headers={"Connection": "close"},
            timeout=(5, 90),
        ) as response:
            record["http_status"] = response.status_code
            if body.get("stream"):
                lines = []
                record["response"] = lines

                def capture():
                    for line in response.iter_lines():
                        value = line.decode("utf-8")
                        lines.append(value)
                        yield value

                response.raise_for_status()
                validate_stream(capture())
            else:
                record["response"] = response.text
                response.raise_for_status()
                value = response.json()
                if scenario == "embedding":
                    count = len(body["input"]) if isinstance(body["input"], list) else 1
                    validate_embedding(value, count, dimensions)
                else:
                    validate_chat(value, body["max_tokens"])
            record["status"] = "passed"
    except (
        ContractError,
        KeyError,
        TypeError,
        ValueError,
        requests.RequestException,
    ) as error:
        record["error"] = str(error)
    record["ended"] = time.time()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--scenario", choices=("embedding", "chat"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    control_path = args.state / "control.json"
    if not control_path.exists():
        control_path.write_text(json.dumps({"phase": "before"}))
    summary = {"requests": 0, "failures": 0, "phases": {}}
    requests_to_send = payloads(args.scenario, args.model)
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        control = json.loads((args.state / "control.json").read_text())
        if control["phase"] == "stop":
            print(json.dumps({"summary": summary}), flush=True)
            return
        name, body = requests_to_send[summary["requests"] % len(requests_to_send)]
        record = request_once(args.base, args.scenario, body, args.dimensions)
        record.update(case=name, phase=control["phase"])
        print(json.dumps(record), flush=True)
        summary["requests"] += 1
        summary["failures"] += record["status"] != "passed"
        phase = summary["phases"].setdefault(control["phase"], {})
        phase[name] = phase.get(name, 0) + 1
        summary["last_completed"] = time.time()
        temporary = args.state / "summary.tmp"
        temporary.write_text(json.dumps(summary))
        temporary.replace(args.state / "summary.json")
        time.sleep(1)  # Deliberately below single-worker capacity; no catch-up burst.
    raise SystemExit("Rollout client exceeded its lifetime budget")


if __name__ == "__main__":
    main()
