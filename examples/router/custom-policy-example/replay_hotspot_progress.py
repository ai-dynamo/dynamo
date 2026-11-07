#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay the fixed hotspot statistics from all retained group-level records, without a GPU."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

PAIRS = 32
BUDGETS = (16, 128)
DATA_SHA256 = "78affe08ec150fc6459fd09406291d7d33d98a9eb08927bedbf6f5494512b275"
CSV_SHA256 = "953072825f30bc6b50a2058d99a2bc7b88153233594f6e702e5c638e8bd7f38a"
STATISTICS_SHA256 = "add7452683b41f130b9f9bacf59825733d23000f82c5bf7915dd53239ec22ead"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, kind):
    values = sorted(values)
    return (
        values[math.ceil(0.95 * len(values)) - 1]
        if kind == "p95"
        else statistics.median(values)
    )


def assess(records, *, replicates=20000, seed=2026103001, evidence_role="REAL_GPU"):
    need(
        evidence_role in ("REAL_GPU", "CPU_FIXTURE"),
        "unsupported analysis evidence role",
    )
    need(
        type(replicates) is int and replicates >= 1000,
        "insufficient bootstrap resolution",
    )
    if evidence_role == "REAL_GPU":
        need(
            replicates == 20000 and seed == 2026103001,
            "real GPU inference parameters are frozen",
        )
    need(
        len(records) == len({r["identity"] for r in records}) == 512,
        "complete original512-formal cohort required",
    )
    blocks = {}
    for row in records:
        pair = row["pair_index"]
        need(
            type(pair) is int
            and 0 <= pair < PAIRS
            and row["arm"] in ("B", "P")
            and row["budget"] in BUDGETS
            and row["phase"] == "formal"
            and row["wave_index"] == pair // 16
            and row["source_worker_index"] in (0, 1)
            and type(row["group_max_raw_ttft_ns"]) is int
            and row["group_max_raw_ttft_ns"] > 0
            and row["mapping"] == ([0, 1] if pair % 2 == 0 else [1, 0])
            and row["physical_source_index"]
            == row["mapping"][row["source_worker_index"]],
            "formal record/index/physical mapping differs",
        )
        blocks.setdefault(pair, []).append(row)
    quota = Counter({(a, b, s): 2 for a in ("B", "P") for b in BUDGETS for s in (0, 1)})
    need(
        set(blocks) == set(range(PAIRS))
        and all(
            Counter((r["arm"], r["budget"], r["physical_source_index"]) for r in rows)
            == quota
            for rows in blocks.values()
        ),
        "independent pair quotas differ",
    )
    channels = {
        b: {
            a: [
                [
                    r["group_max_raw_ttft_ns"]
                    for r in blocks[p]
                    if r["budget"] == b and r["arm"] == a
                ]
                for p in range(PAIRS)
            ]
            for a in ("B", "P")
        }
        for b in BUDGETS
    }

    def ratios(indices):
        return {
            b: quantile([x for i in indices for x in data["P"][i]], "p95")
            / quantile([x for i in indices for x in data["B"][i]], "p95")
            for b, data in channels.items()
        }

    point = ratios(range(PAIRS))
    draws = {b: [] for b in BUDGETS}
    rng = random.Random(seed)
    strata = [
        [p for p in range(PAIRS) if p // 16 == wave and p % 2 == parity]
        for wave in (0, 1)
        for parity in (0, 1)
    ]
    for _ in range(replicates):
        indices = [rng.choice(stratum) for stratum in strata for _ in stratum]
        for budget, value in ratios(indices).items():
            draws[budget].append(value)
    metrics = {}
    alpha = 0.05 / 4
    for budget, values in draws.items():
        values.sort()
        metrics[f"budget{budget}:pooled:p95"] = {
            "point_ratio": point[budget],
            "lower_ratio": values[max(0, math.ceil(alpha * replicates) - 1)],
            "upper_ratio": values[
                min(replicates - 1, math.ceil((1 - alpha) * replicates) - 1)
            ],
        }
    short, long = (metrics[f"budget{b}:pooled:p95"] for b in BUDGETS)
    short_status = (
        "REGRESSION"
        if short["lower_ratio"] > 1.10
        else "APPROXIMATE_SAFETY_SUPPORTED"
        if short["upper_ratio"] <= 1.10
        else "INCONCLUSIVE"
    )
    long_status = (
        "REGRESSION"
        if long["lower_ratio"] > 1
        else "APPROXIMATE_BENEFIT_SUPPORTED"
        if long["upper_ratio"] < 1
        else "INCONCLUSIVE"
    )
    failures = [r["identity"] for r in records if not r["expected_behavior"]]
    decision = (
        "STOP"
        if failures or "REGRESSION" in (short_status, long_status)
        else "KEEP"
        if short_status == "APPROXIMATE_SAFETY_SUPPORTED"
        and long_status == "APPROXIMATE_BENEFIT_SUPPORTED"
        else "INCONCLUSIVE"
    )
    descriptive = {}
    for budget in BUDGETS:
        descriptive[str(budget)] = {}
        for arm in ("B", "P"):
            rows = [r for r in records if r["budget"] == budget and r["arm"] == arm]
            values = [r["group_max_raw_ttft_ns"] / 1e6 for r in rows]
            descriptive[str(budget)][arm] = {
                "groups": len(rows),
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "p95_ms": quantile(values, "p95"),
                "physical_sources": {
                    f"wave{wave}:source{source}": {
                        "groups": sum(
                            r["wave_index"] == wave
                            and r["physical_source_index"] == source
                            for r in rows
                        ),
                        "median_ms": quantile(
                            [
                                r["group_max_raw_ttft_ns"] / 1e6
                                for r in rows
                                if r["wave_index"] == wave
                                and r["physical_source_index"] == source
                            ],
                            "median",
                        ),
                        "p95_ms": quantile(
                            [
                                r["group_max_raw_ttft_ns"] / 1e6
                                for r in rows
                                if r["wave_index"] == wave
                                and r["physical_source_index"] == source
                            ],
                            "p95",
                        ),
                    }
                    for wave in (0, 1)
                    for source in (0, 1)
                },
            }
    return {
        "status": "COMPLETE_MATCHED_BP_FORMAL_COHORT",
        "evidence_role": evidence_role,
        "decision": decision,
        "short_status": short_status,
        "long_status": long_status,
        "metrics": metrics,
        "descriptive": descriptive,
        "formal_groups": 512,
        "service_pairs": 32,
        "behavior_failures": failures,
        "replicates": replicates,
        "seed": seed,
        "marginal_tail_probability": alpha,
        "metric_family_size": 2,
        "interval_role": "APPROXIMATE_WAVE_MAPPING_PAIRED_CLUSTER_BOOTSTRAP_NO_FINITE_SAMPLE_GUARANTEE",
        "legacy_formal_verdict_unchanged": "INCONCLUSIVE",
        "scope": "Pooled short16 and long128 P95 only; no24/32 or per-source safety pass.",
        "limits": [
            "32pair-cluster approximation and exchangeability/no-carryover assumptions remain unverified.",
            "No guaranteed precision, power, conditional coverage, production or throughput claim.",
            "Entire P costs retained; no posttreatment adjustment or extra sampling.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check",
        action="store_true",
        help="Verify the complete data and exact statistics digest",
    )
    modes.add_argument(
        "--emit-csv",
        action="store_true",
        help="Print all 640 retained rows, including warmups",
    )
    args = parser.parse_args()
    packed = Path(__file__).with_name("hotspot-progress-groups.csv.gz").read_bytes()
    need(
        hashlib.sha256(packed).hexdigest() == DATA_SHA256,
        "Compressed data digest differs",
    )
    raw = gzip.decompress(packed)
    need(hashlib.sha256(raw).hexdigest() == CSV_SHA256, "Group data digest differs")
    if args.emit_csv:
        sys.stdout.buffer.write(raw)
        return
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    need(
        len(rows) == len({r["identity"] for r in rows}) == 640,
        "All 640 unique groups are required",
    )
    need(
        Counter(r["phase"] for r in rows) == {"formal": 512, "warmup": 128},
        "Phase counts differ",
    )
    for row in rows:
        for key in (
            "wave_index",
            "pair_index",
            "budget",
            "physical_source_index",
            "group_max_raw_ttft_ns",
        ):
            row[key] = int(row[key])
        need(row["expected_behavior"] in ("True", "False"), "Invalid behavior value")
        row["expected_behavior"] = row["expected_behavior"] == "True"
        row["mapping"] = [0, 1] if row["pair_index"] % 2 == 0 else [1, 0]
        row["source_worker_index"] = row["mapping"].index(row["physical_source_index"])
    # The fixed protocol excludes warmups from inference, not from the retained export.
    value = assess([r for r in rows if r["phase"] == "formal"])
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    need(
        hashlib.sha256(encoded).hexdigest() == STATISTICS_SHA256,
        "Statistics differ from the frozen result",
    )
    if args.check:
        print(
            json.dumps(
                {
                    "status": "PASS_FIXED_STATISTICS_REPLAY",
                    "formal_groups": 512,
                    "retained_warmups": 128,
                    "decision": value["decision"],
                    "statistics_sha256": STATISTICS_SHA256,
                },
                indent=2,
            )
        )
    else:
        sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
