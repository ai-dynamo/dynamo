# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.timeout(30),
]

REPO_ROOT = Path(__file__).parents[2]
RUNNER = REPO_ROOT / "benchmarks/agent_harness/nightly/run_harbor.sh"
TASK_IDS = REPO_ROOT / "benchmarks/agent_harness/nightly/task_ids.txt"


class _ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps({"data": [{"id": "agent-nightly"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _fake_harbor(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

if sys.argv[1:] == ["--version"]:
    print("0.21.0")
    raise SystemExit(0)

args = sys.argv[1:]
with open(os.environ["HARBOR_CALLS"], "a") as output:
    output.write(json.dumps({"args": args}) + "\\n")

harness = args[args.index("-a") + 1]
task_ids = [args[index + 1] for index, value in enumerate(args) if value == "-i"]
with open(os.environ["DYN_REQUEST_TRACE_OUTPUT_PATH"], "a") as trace:
    for task_id in task_ids:
        session_id = f"{harness}-{task_id}"
        for trigger in ("user_message", "tool_result", "tool_result", "tool_result"):
            record = {
                "event_type": "request_end",
                "agent_context": {
                    "session_id": session_id,
                    "input_trigger": trigger,
                },
                "request": {
                    "model": "agent-nightly",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "finish_reason_metadata": {
                        "finish_reason": "tool_calls" if trigger == "user_message" else "stop",
                        "tool_calls": [{}] if trigger == "user_message" else [],
                    },
                },
            }
            trace.write(json.dumps(record) + "\\n")
"""
    )
    path.chmod(0o755)


def test_runner_pins_harbor_contract_and_validates_each_harness(tmp_path: Path) -> None:
    fake_harbor = tmp_path / "harbor"
    calls_path = tmp_path / "calls.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    results_dir = tmp_path / "results"
    _fake_harbor(fake_harbor)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        environment = os.environ.copy()
        environment.update(
            {
                "DYNAMO_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "DYNAMO_MODEL_ALIAS": "agent-nightly",
                "DYN_REQUEST_TRACE_OUTPUT_PATH": str(trace_path),
                "TASK_IDS_FILE": str(TASK_IDS),
                "RESULTS_DIR": str(results_dir),
                "HARBOR_COMMAND": str(fake_harbor),
                "HARBOR_CALLS": str(calls_path),
                "RUN_NAME_SUFFIX": "test-run",
                "TRACE_VALIDATION_TIMEOUT_SECONDS": "1",
            }
        )
        subprocess.run(["bash", str(RUNNER)], env=environment, check=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    calls = [json.loads(line)["args"] for line in calls_path.read_text().splitlines()]
    assert len(calls) == 2
    assert [args[args.index("-a") + 1] for args in calls] == [
        "claude-code",
        "codex",
    ]
    for args in calls:
        assert args[args.index("-d") + 1] == "swebenchpro@1.0"
        assert args[args.index("--n-concurrent") + 1] == "1"
        assert args[args.index("--allow-agent-host") + 1] == "127.0.0.1"
        assert args[args.index("--jobs-dir") + 1] == str(results_dir / "harbor-jobs")
        assert args.count("-i") == 5
        assert "--no-delete" in args

    claude_args = calls[0]
    assert "ANTHROPIC_AUTH_TOKEN=dummy" in claude_args
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096" in claude_args
    assert (results_dir / "claude-code-request-trace-summary.json").is_file()
    assert (results_dir / "codex-request-trace-summary.json").is_file()
