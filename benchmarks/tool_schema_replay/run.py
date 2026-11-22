# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay tool-schema cases against an OpenAI-compatible endpoint.

Run one arm:
    python3 run.py --base-url http://localhost:8000 --suite A --suite B \
        --repeats 5 --out /tmp/baseline.json

Compare two arms (same cases, seeds and stream modes):
    python3 run.py --compare /tmp/baseline.json /tmp/treatment.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import SUITES, SYSTEM_PROMPT, Case, tool_schema  # noqa: E402


def _chat_body(
    case: Case, model: str, seed: int, stream: bool, max_tokens: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case.prompt},
        ],
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": stream,
        # Generated token IDs, decoded below to compare the engine's raw
        # Harmony text with the parser's tool calls.
        "nvext": {"extra_fields": ["completion_token_ids"]},
    }
    if case.tools:
        body["tools"] = case.tools
    if case.tool_choice is not None:
        body["tool_choice"] = case.tool_choice
    if case.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = case.parallel_tool_calls
    return body


def _responses_body(
    case: Case, model: str, seed: int, max_tokens: int
) -> dict[str, Any]:
    tools = [{"type": "function", **t["function"]} for t in case.tools]
    body: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": case.prompt,
        "tools": tools,
        "max_output_tokens": max_tokens,
        # Dynamo's Responses path ignores `seed`; other servers may honor it.
        "seed": seed,
    }
    if case.tool_choice is not None:
        body["tool_choice"] = case.tool_choice
    return body


def _post_chat(client: httpx.Client, url: str, body: dict[str, Any]) -> dict[str, Any]:
    """Return status, error, content, tool calls and finish reason."""
    if not body["stream"]:
        resp = client.post(url, json=body)
        if resp.status_code != 200:
            return {"status": resp.status_code, "error": resp.text[:2000]}
        choice = resp.json()["choices"][0]
        message = choice.get("message") or {}
        calls = [
            {"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
            for c in message.get("tool_calls") or []
        ]
        return {
            "status": 200,
            "content": message.get("content"),
            "tool_calls": calls,
            "finish_reason": choice.get("finish_reason"),
            "token_ids": (resp.json().get("nvext") or {}).get("completion_token_ids")
            or [],
        }

    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    token_ids: list[int] = []
    finish_reason = None
    with client.stream("POST", url, json=body) as resp:
        if resp.status_code != 200:
            return {
                "status": resp.status_code,
                "error": resp.read().decode(errors="replace")[:2000],
            }
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                return {
                    "status": 200,
                    "error": json.dumps(chunk["error"])[:2000],
                    "stream_error": True,
                }
            token_ids += (chunk.get("nvext") or {}).get("completion_token_ids") or []
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        tc.get("index", 0), {"name": "", "arguments": ""}
                    )
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish_reason = choice.get("finish_reason") or finish_reason
    return {
        "status": 200,
        "content": "".join(content) or None,
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish_reason,
        "token_ids": token_ids,
    }


def _post_responses(
    client: httpx.Client, url: str, body: dict[str, Any]
) -> dict[str, Any]:
    resp = client.post(url, json=body)
    if resp.status_code != 200:
        return {"status": resp.status_code, "error": resp.text[:2000]}
    out = resp.json()
    calls, text = [], []
    for item in out.get("output") or []:
        if item.get("type") == "function_call":
            calls.append({"name": item.get("name"), "arguments": item.get("arguments")})
        elif item.get("type") == "message":
            text += [
                p.get("text", "")
                for p in item.get("content") or []
                if p.get("type") == "output_text"
            ]
    return {
        "status": 200,
        "content": "".join(text) or None,
        "tool_calls": calls,
        "finish_reason": out.get("status"),
    }


# A Harmony tool call: the recipient `to=functions.NAME` sits in the message
# header, either before or after `<|channel|>`; arguments end at `<|call|>`.
_RAW_CALL = re.compile(
    r"to=functions\.([\w.-]+).*?<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", re.S
)
_RAW_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|call\|>|<\|end\|>|$)", re.S
)


def raw_vs_parsed(raw_text: str, calls: list[dict[str, Any]]) -> str:
    """Compare argument JSON in the raw generation with the parsed tool calls.

    ``match``: same tool names and JSON-equal arguments. ``mismatch``: the
    parser returned something other than what the model wrote, which points at
    the parser rather than the generation.
    """
    raw_calls = _RAW_CALL.findall(raw_text)
    if not raw_calls and (final := _RAW_FINAL.search(raw_text)):
        # GPT-OSS with tool_choice required/named: the model writes a JSON
        # array of {"name", "parameters"} in the final channel.
        try:
            raw_calls = [
                (c["name"], json.dumps(c["parameters"]))
                for c in json.loads(final.group(1))
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    if len(raw_calls) != len(calls):
        return f"mismatch: {len(raw_calls)} raw calls, {len(calls)} parsed"
    for (raw_name, raw_args), call in zip(raw_calls, calls):
        if raw_name != call["name"]:
            return f"mismatch: raw name {raw_name}, parsed {call['name']}"
        try:
            equal = json.loads(raw_args) == json.loads(call["arguments"] or "")
        except json.JSONDecodeError:
            equal = raw_args.strip() == (call["arguments"] or "").strip()
        if not equal:
            return f"mismatch: raw {raw_args.strip()[:120]!r}, parsed {call['arguments']!r:.120}"
    return "match"


def classify(case: Case, result: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (outcome, details). Outcomes: pass, schema_violation,
    invalid_json, wrong_tool, no_tool_call, unexpected_tool_call, empty_text,
    rejected, error."""
    if "exception" in result:
        return "error", [result["exception"]]
    if result["status"] != 200 or result.get("error"):
        return "rejected", [f"{result['status']}: {result.get('error', '')[:300]}"]
    calls = result.get("tool_calls") or []
    if case.expect_tool is None:
        if calls:
            return "unexpected_tool_call", [c["name"] for c in calls]
        return (
            ("pass", [])
            if (result.get("content") or "").strip()
            else ("empty_text", [])
        )
    if not calls:
        return "no_tool_call", [(result.get("content") or "")[:200]]
    details: list[str] = []
    outcome = "pass"
    for call in calls:
        schema = tool_schema(case, call["name"])
        if schema is None or call["name"] != case.expect_tool:
            return "wrong_tool", [call["name"]]
        try:
            args = json.loads(call["arguments"] or "")
        except json.JSONDecodeError as exc:
            return "invalid_json", [f"{exc}: {call['arguments']!r:.200}"]
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(args),
            key=lambda e: list(e.absolute_path),
        )
        if errors:
            outcome = "schema_violation"
            details += [
                f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                for e in errors
            ]
    return outcome, details


def run_arm(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    records: list[dict[str, Any]] = []
    tokenizer = None
    with httpx.Client(timeout=args.timeout) as client:
        model = args.model or client.get(f"{base}/v1/models").json()["data"][0]["id"]
        if not args.no_raw:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model)
        for suite in args.suite:
            for case in SUITES[suite]:
                if args.case and case.name not in args.case:
                    continue
                modes = [False] if case.api == "responses" else [False, True]
                for stream in modes:
                    for seed in range(args.repeats):
                        start = time.monotonic()
                        try:
                            if case.api == "responses":
                                result = _post_responses(
                                    client,
                                    f"{base}/v1/responses",
                                    _responses_body(case, model, seed, args.max_tokens),
                                )
                            else:
                                result = _post_chat(
                                    client,
                                    f"{base}/v1/chat/completions",
                                    _chat_body(
                                        case, model, seed, stream, args.max_tokens
                                    ),
                                )
                        # ValueError covers a non-JSON body or SSE line, and
                        # KeyError/IndexError a 200 body without `choices`, so
                        # one bad response does not lose the arm's records.
                        except (
                            httpx.HTTPError,
                            ValueError,
                            KeyError,
                            IndexError,
                        ) as exc:
                            result = {"exception": f"{type(exc).__name__}: {exc}"}
                        latency = time.monotonic() - start
                        outcome, details = classify(case, result)
                        token_ids = result.pop("token_ids", None)
                        if tokenizer is not None and token_ids:
                            result["raw_text"] = tokenizer.decode(
                                token_ids, skip_special_tokens=False
                            )
                            if case.tool_choice == "none":
                                # The frontend is meant to drop any call the
                                # model writes here, so a raw call is no
                                # parser change.
                                result["raw_vs_parsed"] = "skipped: tool_choice none"
                            elif result.get("finish_reason") == "length":
                                # Cut off by max_tokens, possibly inside the
                                # arguments.
                                result[
                                    "raw_vs_parsed"
                                ] = "skipped: finish_reason length"
                            else:
                                result["raw_vs_parsed"] = raw_vs_parsed(
                                    result["raw_text"], result.get("tool_calls") or []
                                )
                        records.append(
                            {
                                "suite": suite,
                                "case": case.name,
                                "stream": stream,
                                "seed": seed,
                                "outcome": outcome,
                                "details": details,
                                "latency_s": round(latency, 3),
                                **result,
                            }
                        )
                        print(
                            f"{suite} {case.name:<30} stream={int(stream)} seed={seed} "
                            f"{outcome:<20} {latency:6.2f}s {result.get('raw_vs_parsed', '-')[:60]} "
                            f"{'; '.join(details)[:160]}",
                            flush=True,
                        )
    payload = {
        "base_url": base,
        "model": model,
        "label": args.label,
        "records": records,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print_summary(records)


def _case_table(
    records: list[dict[str, Any]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    table: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        table.setdefault((r["suite"], r["case"]), []).append(r)
    return table


def print_summary(records: list[dict[str, Any]]) -> None:
    print("\n| suite | case | runs passed | outcomes |")
    print("|---|---|---|---|")
    for (suite, case), runs in _case_table(records).items():
        passed = sum(r["outcome"] == "pass" for r in runs)
        outcomes = ", ".join(
            f"{k}={v}" for k, v in Counter(r["outcome"] for r in runs).most_common()
        )
        print(f"| {suite} | {case} | {passed}/{len(runs)} | {outcomes} |")
    print(
        "\n| suite | cases all-pass | runs passed | rejections | raw/parsed mismatches | finish_reason "
        "| median latency s |"
    )
    print("|---|---|---|---|---|---|---|")
    for suite in sorted({r["suite"] for r in records}):
        rs = [r for r in records if r["suite"] == suite]
        cases = [runs for (s, _), runs in _case_table(rs).items() if s == suite]
        all_pass = sum(all(r["outcome"] == "pass" for r in runs) for runs in cases)
        finish = ", ".join(
            f"{k}={v}"
            for k, v in Counter(r.get("finish_reason") for r in rs).most_common()
        )
        print(
            f"| {suite} | {all_pass}/{len(cases)} | {sum(r['outcome'] == 'pass' for r in rs)}/{len(rs)} | "
            f"{sum(r['outcome'] == 'rejected' for r in rs)} | "
            f"{sum(r.get('raw_vs_parsed', '').startswith('mismatch') for r in rs)} | {finish} | "
            f"{statistics.median(r['latency_s'] for r in rs):.2f} |"
        )


def compare(baseline_path: str, treatment_path: str) -> None:
    """A case passes in an arm when every one of its runs passes."""
    base = _case_table(json.loads(Path(baseline_path).read_text())["records"])
    treat = _case_table(json.loads(Path(treatment_path).read_text())["records"])
    print(
        "| suite | case | baseline runs passed | treatment runs passed | change | new rejections |"
    )
    print("|---|---|---|---|---|---|")
    totals: Counter[str] = Counter()
    for key in sorted(base.keys() & treat.keys()):
        b, t = base[key], treat[key]
        b_ok = all(r["outcome"] == "pass" for r in b)
        t_ok = all(r["outcome"] == "pass" for r in t)
        change = {(False, True): "recovered", (True, False): "regressed"}.get(
            (b_ok, t_ok), "same"
        )
        b_rej = {(r["stream"], r["seed"]) for r in b if r["outcome"] == "rejected"}
        new_rej = [
            r
            for r in t
            if r["outcome"] == "rejected" and (r["stream"], r["seed"]) not in b_rej
        ]
        totals[f"{key[0]}:{change}"] += 1
        totals[f"{key[0]}:baseline_failed"] += not b_ok
        totals[f"{key[0]}:new_rejection_runs"] += len(new_rej)
        b_pass = sum(r["outcome"] == "pass" for r in b)
        t_pass = sum(r["outcome"] == "pass" for r in t)
        print(
            f"| {key[0]} | {key[1]} | {b_pass}/{len(b)} | {t_pass}/{len(t)} | {change} | {len(new_rej)} |"
        )
    print(
        "\n| suite | baseline failed cases | recovered | regressed | new rejection runs |"
    )
    print("|---|---|---|---|---|")
    for suite in sorted({k[0] for k in base}):
        print(
            f"| {suite} | {totals[f'{suite}:baseline_failed']} | {totals[f'{suite}:recovered']} | "
            f"{totals[f'{suite}:regressed']} | {totals[f'{suite}:new_rejection_runs']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--model", help="Served model name; defaults to the first entry of /v1/models."
    )
    parser.add_argument(
        "--suite", action="append", choices=sorted(SUITES), help="Repeatable."
    )
    parser.add_argument(
        "--case", action="append", help="Run only these case names. Repeatable."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Runs per case and stream mode; seeds 0..N-1.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer for decoding raw output; defaults to the served model.",
    )
    parser.add_argument(
        "--no-raw", action="store_true", help="Skip raw-versus-parsed comparison."
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON file for per-run records.")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "TREATMENT"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.suite or not args.out:
        parser.error("--suite and --out are required unless --compare is given")
    run_arm(args)


if __name__ == "__main__":
    main()
