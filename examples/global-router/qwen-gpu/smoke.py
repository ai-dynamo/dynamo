# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
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

"""Check real model responses through a supplied OpenAI-compatible endpoint."""

import json
import os
import time
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE = os.environ.get("TARGET_URL", "http://qwen-gpu-frontend:8000").rstrip("/")
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with HTTP.open(req, timeout=90) as response:
        assert response.status == 200
        return response.read().decode()


def run():
    deadline = time.monotonic() + 900
    error = None
    while time.monotonic() < deadline:
        try:
            models = json.loads(request("/v1/models"))
            assert {m["id"] for m in models["data"]} == {MODEL}
            break
        except Exception as failure:
            error = repr(failure)
            time.sleep(5)
    else:
        raise TimeoutError("Model did not become ready: " + str(error))
    samples = []
    for prompt, expected in [
        ("What is 7 times 8? Reply with only the number.", "56"),
        ("What is the capital city of France? Reply with only the city name.", "Paris"),
    ]:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
            "stream": False,
        }
        result = json.loads(request("/v1/chat/completions", body))
        content = result["choices"][0]["message"]["content"]
        assert expected.lower() in content.lower(), content
        assert result["usage"]["completion_tokens"] > 0
        samples.append({"prompt": prompt, "content": content, "usage": result["usage"]})
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5, separated by commas."}
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
    }
    data = request("/v1/chat/completions", body)
    assert "data: [DONE]" in data
    chunks = [
        json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: {")
    ]
    assert chunks and not any("error" in chunk for chunk in chunks)
    content = "".join(
        choice.get("delta", {}).get("content", "") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    assert "1" in content and "5" in content, content
    samples.append({"stream": True, "content": content, "chunks": len(chunks)})
    result = json.loads(
        request(
            "/v1/completions",
            {
                "model": MODEL,
                "prompt": "The capital of France is",
                "max_tokens": 32,
                "temperature": 0,
            },
        )
    )
    assert result["choices"] and result["usage"]["completion_tokens"] > 0
    samples.append(
        {"completion": result["choices"][0]["text"], "usage": result["usage"]}
    )
    return {
        "success": True,
        "target": BASE,
        "model": MODEL,
        "requests": 4,
        "samples": samples,
    }


if __name__ == "__main__":
    try:
        result = run()
    except Exception as failure:
        result = {"success": False, "target": BASE, "error": repr(failure)}
        if Path("/dev/termination-log").exists():
            Path("/dev/termination-log").write_text(json.dumps(result))
        print(json.dumps(result), flush=True)
        raise
    if Path("/dev/termination-log").exists():
        Path("/dev/termination-log").write_text(json.dumps(result))
    print(json.dumps(result), flush=True)
