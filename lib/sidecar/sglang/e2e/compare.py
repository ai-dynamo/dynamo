# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare Dynamo /generate against stock SGLang /generate.

For each prompt it sends the same native request to:

- the SGLang reference server with stream=false (the expected answer),
- each Dynamo frontend with stream=false (joined by the frontend), and
- each Dynamo frontend with stream=true (joined here, independently).

Greedy sampling keeps the answers comparable. Prompt-side outputs in P/D come
from the prefill server, so they must match the aggregated reference too.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from typing import Any

from transformers import AutoTokenizer

# Fields SGLang's incremental streaming sends as deltas.
APPEND_META = (
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
    "output_token_sampling_mask",
    "output_token_sampling_logprobs",
)
PROMPTS = (
    "The capital of France is",
    "Write one sentence about the ocean, then list three fish species:",
)

failures: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        failures.append(message)


def post(url: str, body: dict[str, Any], timeout: float = 120) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def unary(url: str, body: dict[str, Any]) -> dict[str, Any]:
    status, data = post(url, {**body, "stream": False})
    if status != 200:
        raise RuntimeError(f"{url} returned {status}: {data[:500]!r}")
    return json.loads(data)


def streamed(url: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    status, data = post(url, {**body, "stream": True})
    if status != 200:
        raise RuntimeError(f"{url} returned {status}: {data[:500]!r}")
    chunks = []
    for line in data.decode().splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            chunks.append(json.loads(line[5:]))
    return chunks


def join(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """SGLang's _coalesce_streaming_chunks: append deltas, keep the last rest."""
    joined = dict(chunks[-1])
    joined["output_ids"] = [i for chunk in chunks for i in chunk.get("output_ids", [])]
    if "text" in joined:
        joined["text"] = "".join(chunk.get("text") or "" for chunk in chunks)
    meta = dict(joined["meta_info"])
    for key in APPEND_META:
        if any(key in chunk["meta_info"] for chunk in chunks):
            meta[key] = [x for chunk in chunks for x in chunk["meta_info"].get(key, [])]
    joined["meta_info"] = meta
    return joined


def logprob_diff(left: list, right: list) -> tuple[bool, float]:
    """Same tokens in the same order; returns the largest logprob difference."""
    if len(left) != len(right):
        return False, float("inf")
    worst = 0.0
    for a, b in zip(left, right):
        if a[1] != b[1]:
            return False, float("inf")
        if a[0] is None or b[0] is None:
            if a[0] is not b[0]:
                return False, float("inf")
            continue
        worst = max(worst, abs(a[0] - b[0]))
    return True, worst


def compare(name: str, got: dict[str, Any], want: dict[str, Any], tol: float) -> None:
    check(got.get("output_ids") == want.get("output_ids"), f"{name}: output_ids")
    check(got.get("text") == want.get("text"), f"{name}: text")
    gm, wm = got["meta_info"], want["meta_info"]
    for key in ("finish_reason", "prompt_tokens", "completion_tokens"):
        check(gm.get(key) == wm.get(key), f"{name}: meta_info.{key}")
    for key in ("input_token_logprobs", "output_token_logprobs"):
        if key in wm:
            same, worst = logprob_diff(gm.get(key) or [], wm[key])
            check(
                same and worst <= tol,
                f"{name}: {key} ({len(gm.get(key) or [])} vs {len(wm[key])} entries, "
                f"max diff {worst:.2e})",
            )
    for key in ("input_top_logprobs", "output_top_logprobs"):
        if key in wm:
            check(
                len(gm.get(key) or []) == len(wm[key]),
                f"{name}: {key} has {len(gm.get(key) or [])} of {len(wm[key])} rows",
            )
    # SGLang sets response_sent_to_client_ts on the first streamed chunk only,
    # so a join of streamed chunks cannot carry it.
    missing = sorted(
        set(wm) - set(gm) - {"routed_experts", "response_sent_to_client_ts"}
    )
    check(not missing, f"{name}: no meta_info keys missing (missing: {missing})")


def wait_ready(url: str, body: dict[str, Any], deadline: float) -> None:
    while True:
        try:
            status, data = post(url, {**body, "stream": False}, timeout=30)
            if status == 200:
                return
            last = f"{status} {data[:200]!r}"
        except Exception as error:  # noqa: BLE001 - retry until the deadline
            last = repr(error)
        if time.time() > deadline:
            raise RuntimeError(f"{url} never became ready: {last}")
        time.sleep(3)


def requests(tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    bodies = []
    for prompt in PROMPTS:
        plain = {
            "input_ids": tokenizer.encode(prompt),
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0,
            },
        }
        bodies.append(plain)
        bodies.append(
            {
                **plain,
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 2,
                "return_routed_experts": True,
            }
        )
    return bodies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True, help="label for this Dynamo stack")
    parser.add_argument("--dynamo", required=True, help="Dynamo frontend URL")
    parser.add_argument(
        "--reference", help="SGLang URL; saves its answers to --reference-file"
    )
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--tolerance", type=float, default=5e-2)
    parser.add_argument(
        "--require-routed-experts",
        action="store_true",
        help="fail when P/D removes routed_experts (prefill returns prompt rows, sgl-project/sglang#44002)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    bodies = requests(tokenizer, args)
    if args.reference:
        references = [unary(args.reference, body) for body in bodies]
        with open(args.reference_file, "w") as file:
            json.dump(references, file)
    else:
        with open(args.reference_file) as file:
            references = json.load(file)

    wait_ready(args.dynamo, bodies[0], time.time() + 300)
    name = args.name
    for body, reference in zip(bodies, references):
        logprobs = body.get("return_logprob", False)
        print(
            f"\n{name}: {len(body['input_ids'])} prompt tokens, "
            f"{'with' if logprobs else 'without'} logprobs and routed experts"
        )
        dynamo_unary = unary(args.dynamo, body)
        compare(f"{name} unary vs SGLang", dynamo_unary, reference, args.tolerance)
        chunks = streamed(args.dynamo, body)
        joined = join(chunks)
        compare(f"{name} stream vs SGLang", joined, reference, args.tolerance)
        check(
            joined["output_ids"] == dynamo_unary["output_ids"],
            f"{name} stream vs unary: output_ids ({len(chunks)} chunks)",
        )
        if not logprobs:
            continue
        prompt_rows = len(reference["meta_info"]["input_token_logprobs"])
        check(
            all(
                len(chunk["meta_info"].get("input_token_logprobs") or []) == prompt_rows
                for chunk in chunks
            ),
            f"{name} stream: every chunk has all {prompt_rows} prompt logprobs",
        )
        want = reference["meta_info"].get("routed_experts")
        for label, response in (("unary", dynamo_unary), ("stream", joined)):
            got = response["meta_info"].get("routed_experts")
            if got is None and name != "agg" and not args.require_routed_experts:
                print(
                    f"  info  {name} {label}: routed_experts removed (prefill sent none)"
                )
                continue
            check(
                got == want,
                f"{name} {label}: routed_experts match "
                f"({len(base64.b64decode(got or ''))} vs {len(base64.b64decode(want or ''))} bytes)",
            )

    print(f"\n{name}: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
