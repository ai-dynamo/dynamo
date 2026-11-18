#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo twin of frontend-crates ``capture_sglang_unified.py``.

It takes the same job input (one family per job, JSON on stdin), uses the same
family-to-SGLang-parser mapping and support gate, and writes the same per-chunk
rows as YAML. Only the component under test differs: Dynamo's SGLang chat
processor (see ``dynamo_sglang_rows``) instead of bare SGLang detector calls.

Requires a frontend-crates checkout in ``FRONTEND_CRATES_DIR``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

import yaml

FRONTEND_CRATES_DIR = Path(os.environ["FRONTEND_CRATES_DIR"]).resolve()
sys.path.insert(0, str(FRONTEND_CRATES_DIR / "conformance/utils/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_stimulus import (  # noqa: E402
    capture_peer_results,
    validate_family_scoped_cases,
)
from dynamo_sglang_rows import stream_rows  # noqa: E402
from impls import SGLANG_UNIFIED_PIPELINES as FAMILY_PARSERS  # noqa: E402
from peer_modules import parser_error_detail  # noqa: E402
from sglang.srt.function_call.function_call_parser import (  # noqa: E402
    FunctionCallParser,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser  # noqa: E402
from unified_tools import unified_tools  # noqa: E402

IMPLEMENTATION = "dynamo_sglang_python"
TOOL_SCHEMAS = unified_tools()


def _registration_unavailable(case):
    # Same gate as the native harness, so both sides execute the same cases.
    if case.get("finish_reason", "stop") not in {"stop", "length"}:
        return (
            "peer_termination_unsupported",
            "SGLang harness supports serving stop/length termination only",
        )
    rname, tname = FAMILY_PARSERS[case["family"]]
    missing = []
    if rname not in ReasoningParser.DetectorMap:
        missing.append(f"ReasoningParser {rname}")
    if tname not in FunctionCallParser.ToolCallParserEnum:
        missing.append(f"FunctionCallParser {tname}")
    if missing:
        return (
            "peer_detector_unavailable",
            "SGLang " + version("sglang") + " does not register " + ", ".join(missing),
        )
    return None


def _capture_cases(cases):
    results = {}
    for case in cases:
        family = case["family"]
        if family not in FAMILY_PARSERS:
            continue
        rname, tname = FAMILY_PARSERS[family]
        entry = {
            "parser": f"Dynamo SGLang chat processor (sglang {version('sglang')}, text injection)"
        }
        partial_rows: list = []
        try:
            entry["chunks"] = stream_rows(
                reasoning_parser_name=rname,
                tool_call_parser_name=tname,
                chunks=case.get("chunks") or [],
                tools=case.get("tools", TOOL_SCHEMAS),
                finish_reason=case.get("finish_reason", "stop"),
                terminal_step=case.get("terminal_step", False),
                rows=partial_rows,
            )
        except Exception as exc:  # noqa: BLE001 - record per case, keep the batch going
            entry["partial_chunks"] = partial_rows
            entry["error"] = parser_error_detail(exc)
            entry["error_traceback"] = traceback.format_exc()
        results[case["id"]] = entry
    return results


def main() -> None:
    logging.getLogger("dynamo").setLevel(logging.CRITICAL)
    job = json.load(sys.stdin)
    cases = job.get("cases", [])
    validate_family_scoped_cases(cases)
    results = {}
    for case in cases:
        results.update(
            capture_peer_results(
                [case],
                FAMILY_PARSERS,
                _capture_cases,
                tools=case["tools"],
                unsupported_reason=_registration_unavailable,
                supports_finish=True,
            )
        )
    yaml.dump(
        {
            "sglang_version": version("sglang"),
            "implementation": IMPLEMENTATION,
            "results": results,
        },
        sys.stdout,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=4096,
    )


if __name__ == "__main__":
    main()
