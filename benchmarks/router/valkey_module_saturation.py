#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Measure the saturation knee of one local dynkv Valkey server."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import asyncio
import json

from benchmarks.router.valkey_saturation.artifacts import emit_campaign_artifacts
from benchmarks.router.valkey_saturation.campaign import (
    build_sweep_schedule,
    campaign_summary,
)
from benchmarks.router.valkey_saturation.cli import build_parser, validate_args
from benchmarks.router.valkey_saturation.protocol import (
    ADMISSION_RESERVED,
    ADMISSION_VERSION,
    DYNKV_PROTOCOL,
    EVENT_REMOVE,
    WIRE_VERSION,
    XOR_MASK,
    Counters,
    LatencySeries,
    Resp,
    RespCommandError,
    campaign_provenance,
    event_hash_start,
    file_provenance,
    match_request,
    parse_match_response,
    parse_release_response,
    parse_reservation_response,
    parse_select_response,
    remove_event,
    reserve_request,
    select_request,
    store_event,
)
from benchmarks.router.valkey_saturation.runner import run_campaign
from benchmarks.router.valkey_saturation.server import Telemetry
from benchmarks.router.valkey_saturation.validation import validate_measured_state
from benchmarks.router.valkey_saturation.workload import (
    Topology,
    WorkloadSetup,
    build_topology,
    churn_event_ids,
    churn_hash_start,
    churn_owned_commands,
    register_leased_worker,
)

__all__ = [
    "ADMISSION_RESERVED",
    "ADMISSION_VERSION",
    "DYNKV_PROTOCOL",
    "EVENT_REMOVE",
    "WIRE_VERSION",
    "XOR_MASK",
    "Counters",
    "LatencySeries",
    "Resp",
    "RespCommandError",
    "Telemetry",
    "Topology",
    "WorkloadSetup",
    "build_parser",
    "build_sweep_schedule",
    "build_topology",
    "campaign_summary",
    "churn_event_ids",
    "churn_hash_start",
    "churn_owned_commands",
    "emit_campaign_artifacts",
    "event_hash_start",
    "file_provenance",
    "match_request",
    "parse_match_response",
    "parse_release_response",
    "parse_reservation_response",
    "parse_select_response",
    "register_leased_worker",
    "remove_event",
    "reserve_request",
    "select_request",
    "store_event",
    "validate_measured_state",
]



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    provenance_before = campaign_provenance(args)
    result = asyncio.run(run_campaign(args))
    provenance_after = campaign_provenance(args)
    result["provenance"] = {
        "before": provenance_before,
        "after": provenance_after,
        "consistent": provenance_before == provenance_after,
    }
    result["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if provenance_before != provenance_after:
        result["status"] = "invalid"
        result.setdefault("validation_errors", []).append(
            "server, module, harness, interpreter, git revision, or CPU affinity changed during campaign"
        )
    result = emit_campaign_artifacts(result, args)
    rendered = json.dumps(result, sort_keys=True, indent=args.json_indent)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    if result.get("status") != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
