# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import LOGICAL_MOCKER_WORKERS, SUMMARY_VERSION
from .summary import sample_csv_rows, summarize_samples, summary_csv_rows, write_csv
from .validation import write_json

def configuration_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "frontend_counts": list(args.frontend_counts),
        "repetitions": args.repetitions,
        "mocker_processes": args.mocker_processes,
        "logical_mocker_workers": LOGICAL_MOCKER_WORKERS,
        "frontend_cpus": args.frontend_cpus,
        "mocker_cpus": args.mocker_cpus,
        "valkey_cpus": args.valkey_cpus,
        "aiperf_cpus": args.aiperf_cpus,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "requests": args.requests,
        "warmup_requests": args.warmup_requests,
        "concurrency": args.concurrency,
        "isl": args.isl,
        "osl": args.osl,
        "valkey_admission_lease_ms": args.valkey_admission_lease_ms,
        "valkey_gc_interval_ms": args.valkey_gc_interval_ms,
        "valkey_gc_inspection_budget": args.valkey_gc_inspection_budget,
        "event_plane": args.event_plane,
        "etcd_endpoints": args.etcd_endpoints,
        "nats_server": args.nats_server,
        "aiperf_timeout_seconds": args.aiperf_timeout_seconds,
        "aiperf_request_timeout_seconds": args.aiperf_request_timeout_seconds,
        "tcp_request_timeout_seconds": args.tcp_request_timeout_seconds,
        "ready_timeout": args.ready_timeout,
        "replica_ready_timeout": args.replica_ready_timeout,
        "settle_seconds": args.settle_seconds,
        "harness": str(args.harness),
        "python": str(args.python),
        "aiperf": str(args.aiperf) if args.aiperf else None,
        "aiperf_workers_max": args.aiperf_workers_max,
        "record_processors": args.record_processors,
        "harness_extra_arg": list(args.harness_extra_arg),
    }


def summarize_child_provenance(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [
        sample.get("child_provenance")
        for sample in samples
        if isinstance(sample.get("child_provenance"), Mapping)
    ]
    complete = len(records) == len(samples) and bool(samples)
    consistent = complete and all(record == records[0] for record in records[1:])
    return {
        "samples_with_provenance": len(records),
        "consistent_across_samples": consistent,
        "record": dict(records[0]) if consistent else None,
    }


def build_summary_document(
    args: argparse.Namespace,
    schedule: Sequence[tuple[int, int]],
    samples: Sequence[Mapping[str, Any]],
    *,
    plot: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = summarize_samples(samples, args.frontend_counts, args.repetitions)
    invalid_samples = [sample for sample in samples if sample.get("valid") is not True]
    input_hashes = sorted(
        {
            str(sample["aiperf_input_sha256"])
            for sample in samples
            if isinstance(sample.get("aiperf_input_sha256"), str)
        }
    )
    return {
        "schema_version": SUMMARY_VERSION,
        "updated_at": datetime.now().astimezone().isoformat(),
        "benchmark_provenance": summarize_child_provenance(samples),
        "input_dataset_sha256": input_hashes[0] if len(input_hashes) == 1 else None,
        "input_dataset_consistent": (
            len(input_hashes) == 1 and len(samples) == len(schedule)
        ),
        "configuration": configuration_record(args),
        "schedule": [
            {"sample_index": index, "repetition": repetition, "frontend_count": count}
            for index, (repetition, count) in enumerate(schedule, start=1)
        ],
        "valid": not invalid_samples,
        "planned_samples": len(schedule),
        "started_samples": len(samples),
        "not_started_samples": len(schedule) - len(samples),
        "valid_samples": len(samples) - len(invalid_samples),
        "invalid_samples": len(invalid_samples),
        "samples": list(samples),
        "frontend_summaries": summaries,
        "plot": dict(plot),
        "methodology_caveats": [
            f"Each sample is a fresh child topology: four logical mock workers split across {args.mocker_processes} OS process(es), two module-loaded Valkey instances, and the requested frontend count.",
            "The aiperf client uses one global closed-loop concurrency value and round-robins across all frontends; this measures aggregate host-level throughput, not an open-loop server-capacity limit.",
            "The child harness rejects incomplete or errored aiperf runs. This driver suppresses the RPS plot unless every planned sample is valid.",
            "Roles without an explicit CPU affinity still share the runner CPU set. A plateau can therefore be client, worker, Valkey, or host-CPU limited rather than frontend limited.",
            "The first sweep follows the requested count order; later sweeps cyclically rotate that order to counterbalance host drift.",
        ],
    }


def write_artifacts(output_dir: Path, document: Mapping[str, Any]) -> None:
    write_json(output_dir / "summary.json", document)
    samples = document.get("samples")
    summaries = document.get("frontend_summaries")
    if isinstance(samples, Sequence):
        write_csv(output_dir / "samples.csv", sample_csv_rows(samples))
    if isinstance(summaries, Sequence):
        write_csv(output_dir / "summary.csv", summary_csv_rows(summaries))
