# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce independent offline qualification of all four native restores.

Run from any directory. Only validation-all-restores.json is written; the
original first-run validation.json and all raw case evidence remain unchanged.
"""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
CASES = (
    "pagebroker-native16-1",
    "pagebroker-native16-2",
    "pagebroker-native64-1",
    "pagebroker-native64-2",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-write", action="store_true")
args = parser.parse_args()
validator = STUDY / "validate_evidence.py"
original_argv = sys.argv
reports = []
try:
    for case in CASES:
        sys.argv = [str(validator), "--case", case, "--no-write"]
        result = runpy.run_path(str(validator), run_name="__main__")
        reports.append(result["report"])
finally:
    sys.argv = original_argv

if any(report["status"] != "passed" for report in reports):
    raise AssertionError("a case did not pass qualification")
if len({report["capture_manifest_sha256"] for report in reports}) != 1:
    raise AssertionError("capture metadata changed across cases")
if any(report["inherited_nccl_environment"] != reports[0]["inherited_nccl_environment"] for report in reports):
    raise AssertionError("inherited communication settings changed across cases")

combined = {
    "schema_version": 1,
    "status": "passed",
    "method": "Replayed the independent assertion suite for all four cases, with CPU quota selected by cohort and native records fenced by the fresh DGD UID.",
    "reproduce": "python3 benchmarks/gms_restore/results/default-config/pagebroker-study/validate_all_restores.py",
    "validator_sha256": hashlib.sha256(validator.read_bytes()).hexdigest(),
    "case_count": len(reports),
    "same_capture_manifest_and_inherited_communication_settings": True,
    "cases": reports,
    "limits": [
        "Two sequential runs per CPU cohort establish functional qualification; they do not isolate a CPU-limit effect or establish a speedup over preselected Python-loader trials.",
        "All metadata selection/discovery follows creation of each DGD; generic service readiness and PageBroker initialization precede the timer.",
        "Default configuration refers to retained captured engine communication paths, including mnnvl allreduce fusion; inherited single-node NCCL overrides remain enabled.",
        "The original single-case validation.json is preserved. Per-case limitations are retained in full below.",
    ],
}
if not args.no_write:
    (STUDY / "validation-all-restores.json").write_text(json.dumps(combined, indent=2) + "\n")
print(json.dumps({"status": "passed", "cases": len(reports), "output_written": not args.no_write}))
