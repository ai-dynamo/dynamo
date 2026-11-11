# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport fixtures validate the executor, never certify a model or parser."""

import copy
import json
import os
import subprocess
import sys
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dynamo.compatibility.runner import main

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@pytest.fixture
def run_pair(tmp_path):
    manifest = {
        "schema_version": 1,
        "profile": "runner-transport-fixture",
        "model": "test-model",
        "provenance": {
            "model_revision": "fixture-only",
            "dynamo_build": "fixture-only",
            "sglang_build": "fixture-only",
            "processor": "fixture-only",
        },
        "required_cases": ["text"],
        "cases": [
            {
                "id": "text",
                "request": {"messages": [{"role": "user", "content": "你好"}]},
            }
        ],
    }
    states = []
    servers = []
    with ExitStack() as cleanup:
        for kind in ("dynamo", "sglang"):
            state = {
                "kind": kind,
                "tokens": [1, 2, 3],
                "requests": [],
                "model": "test-model",
                "status": 200,
                "post_status": 200,
            }

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_GET(self):
                    self.reply({"data": [{"id": self.server.state["model"]}]})

                def do_POST(self):
                    current = self.server.state
                    payload = json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    current["requests"].append(payload)
                    result = {
                        "model": current["model"],
                        "choices": [{"index": 0, "message": {"content": "ignored"}}],
                        "usage": {"prompt_tokens": 3},
                    }
                    if current["kind"] == "dynamo":
                        result["nvext"] = {"prompt_token_ids": current["tokens"]}
                    else:
                        result["choices"][0]["prompt_token_ids"] = current["tokens"]
                    self.reply(result, current["post_status"])

                def reply(self, value, status=None):
                    encoded = json.dumps(value).encode()
                    self.send_response(status or self.server.state["status"])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.state = state
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            thread.start()
            cleanup.callback(server.server_close)
            cleanup.callback(thread.join, 2)
            cleanup.callback(server.shutdown)
            servers.append(server)
            states.append(state)

        def run(*, module=False, raw=None):
            path = tmp_path / "manifest.json"
            path.write_text(
                raw if raw is not None else json.dumps(manifest), encoding="utf-8"
            )
            output = tmp_path / "evidence"
            args = ["--manifest", str(path), "--output", str(output), "--timeout", "2"]
            for kind, server in zip(("dynamo", "sglang"), servers):
                args.extend([f"--{kind}-url", f"http://127.0.0.1:{server.server_port}"])
            if module:
                result = subprocess.run(
                    [sys.executable, "-m", "dynamo.compatibility", *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=os.environ.copy(),
                )
                code = result.returncode
            else:
                code = main(args)
            report = json.loads((output / "results.json").read_text())
            return code, report, output

        yield manifest, states, run


def test_module_runs_and_preserves_request(run_pair):
    manifest, states, run = run_pair
    request = manifest["cases"][0]["request"]
    request.update(
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="none",
        chat_template_kwargs={"enable_thinking": False},
    )
    original = copy.deepcopy(request)
    code, report, output = run(module=True)
    assert code == 0 and report["status"] == "PASS"
    assert len(report["manifest_sha256"]) == 64
    for state in states:
        assert {key: state["requests"][0][key] for key in original} == original
    assert states[0]["requests"][0]["nvext"] == {"extra_fields": ["prompt_token_ids"]}
    assert states[1]["requests"][0]["return_prompt_token_ids"] is True
    assert "NOT TESTED" in (output / "compatibility-matrix.md").read_text()
    assert "not verified" in (output / "compatibility-matrix.html").read_text()


@pytest.mark.parametrize(
    "tokens,index", [([1, 9, 3], 1), ([1, 2], 2), ([1, 2, 3, 4], 3)]
)
def test_token_differences_fail_with_evidence(run_pair, tokens, index):
    _, states, run = run_pair
    states[1]["tokens"] = tokens
    code, report, _ = run()
    assert code == 1
    assert report["cases"][0]["status"] == "FAIL"
    assert report["cases"][0]["first_difference"]["index"] == index


@pytest.mark.parametrize("tokens", [None, [], [True], [2**32]])
def test_missing_or_invalid_ids_never_use_usage_counts(run_pair, tokens):
    _, states, run = run_pair
    states[1]["tokens"] = tokens
    code, report, output = run()
    assert code == 1
    assert report["cases"][0]["status"] == "BLOCKED"
    assert "not a substitute" in report["cases"][0]["errors"]["sglang"]
    assert "BLOCKED" in (output / "compatibility-matrix.md").read_text()


def test_missing_required_case_blocks_before_requests(run_pair):
    manifest, states, run = run_pair
    manifest["required_cases"].append("tool-history")
    code, report, _ = run()
    assert code == 1
    assert [row["status"] for row in report["cases"]] == ["BLOCKED", "MISSING"]
    assert not any(state["requests"] for state in states)


@pytest.mark.parametrize(
    "change", ["empty", "duplicate", "unknown", "media", "role", "placeholder"]
)
def test_invalid_scope_cannot_pass(run_pair, change):
    manifest, _, run = run_pair
    if change == "empty":
        manifest["required_cases"] = []
    elif change == "duplicate":
        manifest["cases"].append(copy.deepcopy(manifest["cases"][0]))
    elif change == "unknown":
        manifest["cases"][0]["request"]["input_ids"] = [1, 2, 3]
    elif change == "media":
        manifest["cases"][0]["request"]["messages"][0]["content"] = [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
    elif change == "role":
        manifest["cases"][0]["request"]["messages"][0]["role"] = []
    else:
        manifest["provenance"]["model_revision"] = "SET_TO_DEPLOYED_REVISION"
    code, report, output = run()
    assert code == 1 and report["status"] == "BLOCKED" and report["error"]
    assert (output / "compatibility-matrix.html").is_file()


@pytest.mark.parametrize("failure", ["model", "http", "post"])
def test_endpoint_failures_remain_visible(run_pair, failure):
    _, states, run = run_pair
    if failure == "model":
        states[1]["model"] = "wrong-model"
    elif failure == "http":
        states[1]["status"] = 503
    else:
        states[1]["post_status"] = 400
    code, report, _ = run()
    assert code == 1
    assert report["cases"][0]["errors"]["sglang"]


def test_duplicate_manifest_fields_fail_and_html_is_escaped(run_pair):
    manifest, _, run = run_pair
    code, report, _ = run(raw='{"schema_version":1,"schema_version":1}')
    assert code == 1 and "duplicate JSON" in report["error"]
    manifest["profile"] = '<script>alert("x")</script>|profile'
    code, _, output = run()
    assert code == 0
    assert "<script>" not in (output / "compatibility-matrix.html").read_text()
    assert "\\|profile" in (output / "compatibility-matrix.md").read_text()


def test_deep_json_fails_with_reports(run_pair):
    _, _, run = run_pair
    code, report, _ = run(raw="[" * 30000 + "0" + "]" * 30000)
    assert code == 1 and report["status"] == "BLOCKED"
    assert "nesting" in report["error"]


def test_invalid_api_key_is_never_written_to_reports(run_pair, monkeypatch):
    _, _, run = run_pair
    secret = "synthetic-secret\n"
    monkeypatch.setenv("DYNAMO_COMPAT_DYNAMO_API_KEY", secret)
    code, report, output = run()
    assert code == 1
    assert "invalid header" in report["cases"][0]["errors"]["dynamo"]
    for path in output.iterdir():
        assert "synthetic-secret" not in path.read_text()
