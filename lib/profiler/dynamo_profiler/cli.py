# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for dynamo-profiler.

Subcommands:
  report       Generate unified HTML report from a sysprofile run directory.
  nsys-convert Convert nsys SQLite export to Perfetto protobuf.
  analyze      Run individual analyzers on trace data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

log = logging.getLogger("dynamo_profiler")


def cmd_report(args):
    from . import report_generator

    run_dir = args.run_dir
    merge_result_path = args.merge_result or os.path.join(run_dir, "merge_result.json")

    merge_result = report_generator._load_json(merge_result_path)
    if merge_result is None:
        log.error("Could not load merge_result.json from %s", merge_result_path)
        return 1

    stage_attr = report_generator._load_json(
        args.stage_attr or os.path.join(run_dir, "stage_attr.json"))
    gpu_util = report_generator._load_json(
        args.gpu_util or os.path.join(run_dir, "gpu_util.json"))
    kernels = report_generator._load_json(
        args.kernels or os.path.join(run_dir, "kernels.json"))
    comm = report_generator._load_json(
        args.comm or os.path.join(run_dir, "comm.json"))

    trace_files = [
        f for f in os.listdir(run_dir)
        if f.endswith(".pftrace.gz") or f.endswith(".pftrace")
    ]

    html = report_generator.generate_report(
        merge_result=merge_result,
        stage_attr=stage_attr,
        gpu_util=gpu_util,
        kernels=kernels,
        comm=comm,
        trace_url=args.trace_url,
        trace_files=trace_files,
        title=args.title or "sysprofile report",
    )

    output = args.output or os.path.join(run_dir, "report.html")
    with open(output, "w") as f:
        f.write(html)
    log.info("Report written to %s (%d bytes)", output, len(html))
    return 0


def cmd_nsys_convert(args):
    from .nsys_to_perfetto import NsysToPerfettoConverter

    converter = NsysToPerfettoConverter(
        sqlite_path=args.sqlite,
        output_path=args.output,
        component=args.component,
        host=args.host,
        ptp_offset_ns=args.ptp_offset_ns,
        gzip_output=not args.no_gzip,
    )
    converter.convert()
    return 0


def cmd_analyze(args):
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir

    pftrace_files = [
        os.path.join(run_dir, f) for f in os.listdir(run_dir)
        if f.endswith(".pftrace.gz") or f.endswith(".pftrace")
    ]

    if not pftrace_files:
        log.warning("No .pftrace[.gz] files found in %s", run_dir)

    if "gpu" in args.analyzers or "all" in args.analyzers:
        from . import gpu_utilization
        for pf in pftrace_files:
            log.info("Running GPU utilization on %s", pf)
            result = gpu_utilization.analyze(pf)
            out = os.path.join(output_dir, "gpu_util.json")
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log.info("Wrote %s", out)
            break

    if "kernels" in args.analyzers or "all" in args.analyzers:
        from . import kernel_hotlist
        for pf in pftrace_files:
            log.info("Running kernel hotlist on %s", pf)
            result = kernel_hotlist.analyze(pf, top_n=args.top_n)
            out = os.path.join(output_dir, "kernels.json")
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log.info("Wrote %s", out)
            break

    if "comm" in args.analyzers or "all" in args.analyzers:
        from . import comm_breakdown
        for pf in pftrace_files:
            log.info("Running comm breakdown on %s", pf)
            result = comm_breakdown.analyze(pf)
            out = os.path.join(output_dir, "comm.json")
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log.info("Wrote %s", out)
            break

    if "stages" in args.analyzers or "all" in args.analyzers:
        from . import stage_attribution
        trace_index_path = os.path.join(run_dir, "trace_index.json")
        if os.path.exists(trace_index_path):
            with open(trace_index_path) as f:
                trace_index = json.load(f)
            report = stage_attribution.compute_attribution(
                trace_index, stage_attribution.DEFAULT_STAGES)
            out = os.path.join(output_dir, "stage_attr.json")
            with open(out, "w") as f:
                json.dump({"report": stage_attribution.to_serializable(report)}, f, indent=2)
            log.info("Wrote %s", out)
        else:
            log.info("No trace_index.json found, skipping stage attribution")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="dynamo-sysprofile-report",
        description="Dynamo distributed profiling tools",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # report
    p_report = sub.add_parser("report", help="Generate unified HTML report")
    p_report.add_argument("run_dir", help="Directory with merge_result.json and traces")
    p_report.add_argument("--merge-result", help="Path to merge_result.json (default: <run_dir>/merge_result.json)")
    p_report.add_argument("--stage-attr", help="Path to stage_attr.json")
    p_report.add_argument("--gpu-util", help="Path to gpu_util.json")
    p_report.add_argument("--kernels", help="Path to kernels.json")
    p_report.add_argument("--comm", help="Path to comm.json")
    p_report.add_argument("--trace-url", help="URL to merged trace for Perfetto deep-links")
    p_report.add_argument("--title", help="Report title")
    p_report.add_argument("--output", "-o", help="Output HTML path (default: <run_dir>/report.html)")

    # nsys-convert
    p_nsys = sub.add_parser("nsys-convert", help="Convert nsys SQLite to Perfetto protobuf")
    p_nsys.add_argument("--sqlite", required=True, help="nsys SQLite export")
    p_nsys.add_argument("--component", required=True, help="Component name")
    p_nsys.add_argument("--host", required=True, help="Host name")
    p_nsys.add_argument("--ptp-offset-ns", type=int, default=0)
    p_nsys.add_argument("--output", required=True, help="Output .pftrace file")
    p_nsys.add_argument("--no-gzip", action="store_true")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run analyzers on trace data")
    p_analyze.add_argument("run_dir", help="Directory with .pftrace[.gz] files")
    p_analyze.add_argument("--analyzers", nargs="+", default=["all"],
                           choices=["all", "gpu", "kernels", "comm", "stages"],
                           help="Which analyzers to run")
    p_analyze.add_argument("--output-dir", help="Output directory (default: run_dir)")
    p_analyze.add_argument("--top-n", type=int, default=20, help="Top N kernels")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "report":
        return cmd_report(args)
    elif args.command == "nsys-convert":
        return cmd_nsys_convert(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
