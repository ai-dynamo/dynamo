#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-container worker for `capture_toolcalling_outputs.py --container`.

Runs INSIDE a container where the target impl (vllm / sglang / dynamo) is
installed. Imports the chosen adapter ONCE, then streams results so the heavy
engine import is paid a single time for the whole run.

Protocol (line-delimited JSON over stdin -> stdout):
  in:  {"key","family","mode","model_text"?,"chunks"?,"tools"?}  (one per line)
  out: {"key","calls","normal_text","error"}                     (one per line)

The host (`capture_toolcalling_outputs.py`) cps a minimal bundle
(common.py + the adapter + __init__ files + this worker) into the container and
invokes `python3 -m tests.parity.toolcalling.capture_worker --impl <impl>` with
the requests piped to stdin. This mirrors `run_parser()` on the host side so
in-process and in-container results are identical.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys


def _serialize(call) -> dict:
    if hasattr(call, "to_dict"):
        return call.to_dict()
    return {"name": call.get("name"), "arguments": call.get("arguments")}


def _run_one(wrapper, impl: str, req: dict) -> dict:
    family = req["family"]
    mode = req["mode"]
    try:
        if mode == "stream":
            fn = getattr(wrapper, "parse_tool_calls_stream", None)
            if fn is None:
                return {
                    "calls": None,
                    "normal_text": None,
                    "error": f"UNAVAILABLE: {impl} wrapper has no parse_tool_calls_stream",
                }
            got = fn(family, req["chunks"], req.get("tools"))
        elif mode == "batch":
            got = wrapper.parse_tool_calls_batch(
                family, req["model_text"], req.get("tools")
            )
        else:
            return {
                "calls": None,
                "normal_text": None,
                "error": f"PYTHON_EXC: unsupported parser mode {mode!r}",
            }
        return {
            "calls": [_serialize(c) for c in (got.calls or [])],
            "normal_text": got.normal_text,
            "error": got.error,
        }
    except Exception as e:  # noqa: BLE001 — report per-case, never abort the batch
        return {
            "calls": None,
            "normal_text": None,
            "error": f"PYTHON_EXC: {type(e).__name__}: {e}",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", required=True, choices=["dynamo", "vllm", "sglang"])
    ap.add_argument("--out", required=True, help="write JSONL results to this path")
    args = ap.parse_args()

    # Engine imports (vllm/sglang) often print banners/warnings to stdout, so
    # results go to --out (a file) where that noise can't corrupt them.
    wrapper = importlib.import_module(f"tests.parity.toolcalling.{args.impl}")

    with open(args.out, "w", encoding="utf-8") as outf:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            out = _run_one(wrapper, args.impl, req)
            out["key"] = req["key"]
            outf.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
