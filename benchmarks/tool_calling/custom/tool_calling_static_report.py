#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Custom matrix and publish portable JSON, JSONL, and HTML artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tool_calling_probe as probe  # noqa: E402

DEFAULT_SITE_DIR = Path(probe.DEFAULT_OUTPUT_ROOT) / "static-site"
DEFAULT_RUNS_ROOT = Path(probe.DEFAULT_OUTPUT_ROOT) / "static-runs"
DEFAULT_TITLE = "Tool Calling Qualification Report"
ALLOWED_BASE_URL_HOSTS = {
    "inference-api.nvidia.com",
    "integrate.api.nvidia.com",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def local_timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in value)
    return slug.strip("-")[:96] or "run"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def validate_base_url(base_url: str, *, allow_other_base_url: bool) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must include an http(s) scheme and host")
    host = (parsed.hostname or "").lower()
    if not allow_other_base_url and host not in ALLOWED_BASE_URL_HOSTS:
        allowed = ", ".join(sorted(ALLOWED_BASE_URL_HOSTS))
        raise ValueError(f"base URL host {host!r} is not allowed; expected {allowed}")
    return base_url.rstrip("/")


def bounded(value: Any, max_chars: int) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= max_chars:
        return value
    return {
        "truncated": True,
        "original_chars": len(encoded),
        "json_prefix": encoded[:max_chars],
    }


def public_record(record: dict[str, Any], *, detail_chars: int) -> dict[str, Any]:
    result = {
        key: record.get(key)
        for key in (
            "timestamp",
            "iteration",
            "case_id",
            "description",
            "mode",
            "pass",
            "errors",
            "warnings",
            "agent_loop",
        )
    }
    for key in (
        "request",
        "response",
        "raw_response",
        "content",
        "reasoning_content",
        "final_messages",
        "turns",
        "executed_tool_calls",
    ):
        if key in record:
            result[key] = bounded(record[key], detail_chars)
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    failed = sum(not bool(record.get("pass")) for record in records)
    return {
        "total": total,
        "passed": total - failed,
        "failed": failed,
        "pass_rate": None if total == 0 else round((total - failed) / total, 4),
    }


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    load_env_file(Path(args.env_file).expanduser() if args.env_file else None)
    base_url = validate_base_url(
        args.base_url, allow_other_base_url=args.allow_other_base_url
    )
    case_profile = (
        probe.model_case_profile(args.model)
        if args.case_profile == "auto"
        else args.case_profile
    )
    cases = probe.select_cases(
        probe.build_cases(case_profile), args.cases, args.exclude_cases
    )
    modes = probe.parse_modes(args.modes)
    extra_headers = probe.parse_headers(args.header)
    url = probe.endpoint_url(base_url)
    run_id = f"{local_timestamp()}-{safe_slug(args.model.split('/')[-1])}"
    output_dir = Path(args.output_root).expanduser() / run_id
    config = {
        "base_url": base_url,
        "url": url,
        "model": args.model,
        "auth": "disabled" if args.no_auth else "bearer",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "modes": modes,
        "case_ids": [case.case_id for case in cases],
        "case_profile": case_profile,
        "exclude_cases": args.exclude_cases,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at": utc_now(),
            "config": config,
            "summary": {"total": 0, "passed": 0, "failed": 0},
            "records": [],
        }, []

    api_key = None if args.no_auth else os.environ.get(args.api_key_env)
    if not args.no_auth and not api_key:
        raise RuntimeError(f"missing API key: set ${args.api_key_env}")

    random.seed(args.seed)
    started_at = utc_now()
    started_mono = time.monotonic()
    writer = probe.ReportWriter(output_dir, config=config, cases=cases)
    try:
        for iteration in range(1, args.iterations + 1):
            work = [(case, mode) for case in cases for mode in modes]
            if args.shuffle:
                random.shuffle(work)

            def run_one(item: tuple[probe.Case, str]) -> dict[str, Any]:
                case, mode = item
                if args.case_delay_seconds > 0:
                    time.sleep(args.case_delay_seconds)
                return probe.run_case(
                    case,
                    mode,
                    iteration=iteration,
                    url=url,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout_seconds,
                    extra_headers=extra_headers,
                    raw_chars=args.raw_chars,
                    record_success_raw=args.record_success_raw,
                )

            if args.concurrency == 1:
                for item in work:
                    writer.record(run_one(item))
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as executor:
                    futures = [executor.submit(run_one, item) for item in work]
                    for future in concurrent.futures.as_completed(futures):
                        writer.record(future.result())
            writer.write_summary()
    finally:
        writer.write_summary()
        writer.close()

    records = [
        public_record(record, detail_chars=args.detail_chars)
        for record in writer.records
    ]
    return {
        "schema_version": 1,
        "title": args.title,
        "run_id": run_id,
        "generated_at": utc_now(),
        "started_at": started_at,
        "duration_seconds": round(time.monotonic() - started_mono, 3),
        "model": args.model,
        "model_label": args.model_label or args.model,
        "config": config,
        "summary": summarize(records),
        "failures": [record for record in records if not record.get("pass")],
        "records": records,
    }, records


def json_details(title: str, value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    encoded = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    return (
        f"<details><summary>{html.escape(title)}</summary>"
        f"<pre>{html.escape(encoded)}</pre></details>"
    )


def render_html(report: dict[str, Any], records: list[dict[str, Any]]) -> str:
    summary = report["summary"]
    rows = []
    for record in records:
        passed = bool(record.get("pass"))
        rows.append(
            '<article class="record">'
            f"<h2><code>{html.escape(str(record.get('case_id')))}</code> "
            f"<span class=\"{'pass' if passed else 'fail'}\">"
            f"{'PASS' if passed else 'FAIL'}</span></h2>"
            f"<p>{html.escape(str(record.get('mode')))}</p>"
            + json_details("Errors", record.get("errors"))
            + json_details("Request", record.get("request"))
            + json_details("Response", record.get("response"))
            + json_details("Raw response", record.get("raw_response"))
            + json_details("Turns", record.get("turns"))
            + "</article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(report['title']))}</title>
<style>body{{font:14px/1.5 system-ui;margin:32px;max-width:1200px;color:#172033}}code,pre{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:12px;border-radius:6px;max-height:640px;overflow:auto}}details{{margin:8px 0}}summary{{cursor:pointer;color:#1358c8;font-weight:600}}.record{{border-top:1px solid #d9dee8;padding:16px 0}}.pass{{color:#166534}}.fail{{color:#b91c1c}}</style></head>
<body><h1>{html.escape(str(report['title']))}</h1>
<p><strong>{summary['passed']}/{summary['total']}</strong> passed · <strong>{summary['failed']}</strong> failed</p>
<p><a href="artifacts/latest.json">latest.json</a> · <a href="artifacts/results.public.jsonl">results.public.jsonl</a></p>
{''.join(rows)}</body></html>"""


def write_static_site(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    site_dir: Path,
    model_slug: str,
    root_alias: bool,
) -> None:
    failures = [record for record in records if not record.get("pass")]

    def publish(page_dir: Path) -> None:
        artifacts_dir = page_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        write_json(artifacts_dir / "latest.json", report)
        write_jsonl(artifacts_dir / "results.public.jsonl", records)
        write_jsonl(artifacts_dir / "failures.public.jsonl", failures)
        (page_dir / "index.html").write_text(
            render_html(report, records), encoding="utf-8"
        )

    publish(site_dir / "models" / model_slug)
    if root_alias:
        publish(site_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Custom tool-calling tests and publish static artifacts."
    )
    parser.add_argument("--site-dir", default=str(DEFAULT_SITE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--base-url", default=probe.DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--no-root-alias", action="store_true")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--allow-other-base-url", action="store_true")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--cases", default="all")
    parser.add_argument("--exclude-cases", default="")
    parser.add_argument(
        "--case-profile",
        default="auto",
        choices=("auto", *probe.INLINE_CASE_PROFILES, "all"),
    )
    parser.add_argument("--modes", default="nonstream,stream")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--case-delay-seconds", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--raw-chars", type=int, default=20000)
    parser.add_argument("--detail-chars", type=int, default=6000)
    parser.add_argument("--record-success-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-test-failure", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    report, records = run_probe(args)
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    model_slug = args.model_slug or safe_slug(args.model.split("/")[-1].lower())
    write_static_site(
        report,
        records,
        site_dir=Path(args.site_dir).expanduser(),
        model_slug=model_slug,
        root_alias=not args.no_root_alias,
    )
    latest = Path(args.site_dir) / "models" / model_slug / "artifacts" / "latest.json"
    print(f"Latest JSON: {latest}", flush=True)
    if args.fail_on_test_failure and report["summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
