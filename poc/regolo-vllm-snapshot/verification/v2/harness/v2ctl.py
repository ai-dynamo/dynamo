#!/usr/bin/env python3
"""Offline control utility for the V2-A verification harness."""

import argparse
import json
import os
import pathlib
import sys

from v2_harness import (
    ResultsLedger,
    cpu_utilization,
    directory_sizes,
    evaluate_diagnosis_gate,
    evaluate_gate,
    evict_candidate_files,
    make_paired_blinded_plan,
    parse_diskstats,
    parse_io_stat,
    parse_gpu_memory_mib,
    parse_meminfo,
    parse_psi,
    seal_plan,
    validate_lane,
)


def _read_json(path):
    try:
        value = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validated_frozen_lane(path):
    lane = _read_json(path)
    validate_lane(lane)
    frozen = _read_json(pathlib.Path(__file__).resolve().parents[1] / "lane.json")
    if lane != frozen:
        raise ValueError("lane does not match the sealed V2-A lane")
    return lane


def _init_run(args):
    lane = _validated_frozen_lane(args.lane)
    authorization = _read_json(args.authorization)
    validate_lane(lane, authorization=authorization)
    if authorization != {"execution_authorized": True}:
        raise ValueError("V2-A execution requires separate explicit authorization")
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    seal_plan(make_paired_blinded_plan(lane), output)
    fd = os.open(output / "results.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    return {"output": str(output), "lane_digest": lane["digest"]}


def _verify_ledger(args):
    rows = ResultsLedger(args.path).read()
    return {"records": len(rows), "verified": True}


def _append_record(args):
    return ResultsLedger(args.ledger).append(_read_json(args.record))


def _gate(args):
    key = _read_json(args.key)
    schedule = json.loads(pathlib.Path(args.schedule).read_text())
    lane = _validated_frozen_lane(args.lane)
    expected = make_paired_blinded_plan(lane)
    if schedule != expected["schedule"] or key != expected["unblinding_key"]:
        raise ValueError("schedule or key does not match the frozen deterministic plan")
    function = evaluate_gate if args.optimized else evaluate_diagnosis_gate
    return function(
        ResultsLedger(args.ledger).read(), schedule=schedule,
        lane_digest=lane["digest"], unblinding_key=key, artifact_dir=args.artifact_dir,
    )


def _cold_advise(args):
    evict_candidate_files([pathlib.Path(path) for path in args.files], args.allow_root)
    return {"advised_files": len(args.files), "advice": "POSIX_FADV_DONTNEED"}


def _sizes(values):
    paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--size must be LABEL=PATH")
        label, path = value.split("=", 1)
        if not label or label in paths or not path:
            raise ValueError("invalid or duplicate --size label")
        paths[label] = pathlib.Path(path)
    return directory_sizes(paths)


def _collect_host(args):
    diskstats = parse_diskstats(pathlib.Path(args.diskstats).read_text())
    required_devices = {"dm-0", "loop6", "sda"}
    if not required_devices.issubset(diskstats):
        raise ValueError("diskstats must include dm-0, loop6, and sda")
    return {
        "meminfo": parse_meminfo(pathlib.Path(args.meminfo).read_text()),
        "psi": {
            "cpu": parse_psi(pathlib.Path(args.psi_cpu).read_text()),
            "io": parse_psi(pathlib.Path(args.psi_io).read_text()),
            "memory": parse_psi(pathlib.Path(args.psi_memory).read_text()),
        },
        "cgroup_io_stat": parse_io_stat(pathlib.Path(args.io_stat).read_text()),
        "diskstats": {device: diskstats[device] for device in sorted(required_devices)},
        "node_cpu_utilization": cpu_utilization(
            pathlib.Path(args.proc_stat_before).read_text(),
            pathlib.Path(args.proc_stat_after).read_text(),
        ),
        "gpu_memory_mib": parse_gpu_memory_mib(pathlib.Path(args.gpu_memory).read_text()),
        "sizes": _sizes(args.size),
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-run")
    init.add_argument("--lane", required=True)
    init.add_argument("--authorization", required=True)
    init.add_argument("--output", required=True)
    init.set_defaults(handler=_init_run)

    verify = subparsers.add_parser("verify-ledger")
    verify.add_argument("path")
    verify.set_defaults(handler=_verify_ledger)

    append = subparsers.add_parser("append-record")
    append.add_argument("--ledger", required=True)
    append.add_argument("--record", required=True)
    append.set_defaults(handler=_append_record)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--ledger", required=True)
    gate.add_argument("--key", required=True)
    gate.add_argument("--schedule", required=True)
    gate.add_argument("--lane", required=True)
    gate.add_argument("--artifact-dir", required=True)
    gate.add_argument("--optimized", action="store_true")
    gate.set_defaults(handler=_gate)

    cold = subparsers.add_parser("cold-advise")
    cold.add_argument("--allow-root", required=True)
    cold.add_argument("--file", dest="files", action="append", required=True)
    cold.set_defaults(handler=_cold_advise)

    collect = subparsers.add_parser("collect-host")
    collect.add_argument("--meminfo", required=True)
    collect.add_argument("--psi-cpu", required=True)
    collect.add_argument("--psi-io", required=True)
    collect.add_argument("--psi-memory", required=True)
    collect.add_argument("--io-stat", required=True)
    collect.add_argument("--diskstats", required=True)
    collect.add_argument("--proc-stat-before", required=True)
    collect.add_argument("--proc-stat-after", required=True)
    collect.add_argument("--gpu-memory", required=True)
    collect.add_argument("--size", action="append", default=[])
    collect.set_defaults(handler=_collect_host)
    return parser


def main():
    args = _parser().parse_args()
    try:
        result = args.handler(args)
    except (OSError, ValueError) as exc:
        print(f"v2ctl: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
