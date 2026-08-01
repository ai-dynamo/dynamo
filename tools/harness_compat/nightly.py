#!/usr/bin/env python3
"""Run the stable native-harness compatibility subset against one live Dynamo endpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


CORE_CASES = (
    ("codex", "steer_after_tool", ("--turn-timeout-s", "180")),
    ("codex", "compact", ("--turn-timeout-s", "180")),
    ("codex", "structured_output", ("--turn-timeout-s", "180")),
    ("codex", "tool_failure", ("--turn-timeout-s", "180")),
    ("claude", "compact", ("--result-timeout-s", "420")),
    ("claude", "structured_output", ("--result-timeout-s", "420")),
    ("claude", "resume", ("--result-timeout-s", "420")),
    ("claude", "tool_failure", ("--result-timeout-s", "420")),
)
FAULT_STATUSES = (400, 401, 403, 404, 409, 429, 500, 502, 503, 529)
SSE_TRUNCATION_EVENTS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-M2")
    parser.add_argument("--artifacts-root", type=Path, default=Path("/tmp/dynamo-harness-compat/nightly"))
    parser.add_argument("--remote-host", default="72.25.69.152")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--endpoint-url", help="Direct Dynamo base URL, for example http://127.0.0.1:8000")
    target.add_argument("--remote-http-port", type=int, help="Dynamo loopback port on the remote host")
    parser.add_argument("--remote-run-root", help="Remote run directory to copy logs from; requires --remote-http-port")
    parser.add_argument("--fault-status", type=int, choices=FAULT_STATUSES)
    parser.add_argument("--skip-fault", action="store_true")
    parser.add_argument("--protocol-baseline", type=Path, default=Path(__file__).with_name("protocol_baseline.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.remote_run_root and args.remote_http_port is None:
        parser.error("--remote-run-root requires --remote-http-port")
    baseline = json.loads(args.protocol_baseline.read_text(encoding="utf-8"))

    status = args.fault_status
    if status is None:
        status = FAULT_STATUSES[dt.datetime.now(dt.timezone.utc).toordinal() % len(FAULT_STATUSES)]
    script = Path(__file__).with_name("live_scenario.py")
    run_prefix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cases = list(CORE_CASES)
    if not args.skip_fault:
        cases.extend(
            (
                ("codex", "expected_error", ("--inject-status", str(status), "--turn-timeout-s", "180")),
                ("claude", "baseline", ("--inject-status", str(status), "--result-timeout-s", "420")),
                (
                    "codex",
                    "baseline",
                    (
                        "--truncate-sse-after-events",
                        str(SSE_TRUNCATION_EVENTS),
                        "--truncate-sse-at-request",
                        str(1 + dt.datetime.now(dt.timezone.utc).toordinal() % 2),
                        "--turn-timeout-s",
                        "180",
                    ),
                ),
                (
                    "claude",
                    "baseline",
                    (
                        "--truncate-sse-after-events",
                        str(SSE_TRUNCATION_EVENTS),
                        "--truncate-sse-at-request",
                        str(1 + dt.datetime.now(dt.timezone.utc).toordinal() % 2),
                        "--result-timeout-s",
                        "420",
                    ),
                ),
            )
        )

    failures = 0
    for index, (harness, scenario, extras) in enumerate(cases, start=1):
        name = f"{run_prefix}-{index:02d}-{harness}-{scenario}"
        command = [
            sys.executable,
            str(script),
            "--harness",
            harness,
            "--scenario",
            scenario,
            "--model",
            args.model,
            "--artifacts",
            str(args.artifacts_root / name),
            *extras,
        ]
        if args.endpoint_url is not None:
            command.extend(["--endpoint-url", args.endpoint_url])
        else:
            command.extend(["--remote-host", args.remote_host, "--remote-http-port", str(args.remote_http_port)])
            if args.remote_run_root is not None:
                command.extend(["--remote-run-root", args.remote_run_root])
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        if subprocess.run(command, check=False).returncode:
            failures += 1
            continue
        summary = subprocess.check_output([sys.executable, str(Path(__file__).with_name("summarize_run.py")), str(args.artifacts_root / name)], text=True)
        actual = json.loads(summary)["http"]["protocol_discriminators"]
        expected = baseline[harness]
        additions = {key: sorted(set(actual.get(key, [])) - set(expected.get(key, []))) for key in expected}
        additions = {key: value for key, value in additions.items() if value}
        if additions:
            drift = {"case": name, "harness": harness, "additions": additions}
            (args.artifacts_root / name / "protocol_drift.json").write_text(
                json.dumps(drift, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps({"protocol_drift": drift}, sort_keys=True), flush=True)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
