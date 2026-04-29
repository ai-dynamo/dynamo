# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for dynamo-profiler.

Quick start (one command, end to end):

  python -m dynamo_profiler profile ./out --demo --open

Subcommands:
  profile      Full pipeline: merge + analyze + report + serve (one command)
  report       Generate unified HTML report from a sysprofile run directory.
  analyze      Run individual analyzers on trace data.
  nsys-convert Convert nsys SQLite export to Perfetto protobuf.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import webbrowser

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
        log.info("Running GPU utilization on %d trace files", len(pftrace_files))
        result = gpu_utilization.analyze(pftrace_files)
        out = os.path.join(output_dir, "gpu_util.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Wrote %s", out)

    if "kernels" in args.analyzers or "all" in args.analyzers:
        from . import kernel_hotlist
        log.info("Running kernel hotlist on %d trace files", len(pftrace_files))
        result = kernel_hotlist.analyze(pftrace_files, top_n=args.top_n)
        out = os.path.join(output_dir, "kernels.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Wrote %s", out)

    if "comm" in args.analyzers or "all" in args.analyzers:
        from . import comm_breakdown
        log.info("Running comm breakdown on %d trace files", len(pftrace_files))
        result = comm_breakdown.analyze(pftrace_files)
        out = os.path.join(output_dir, "comm.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Wrote %s", out)

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


def _find_rust_binary(name):
    """Find a Rust binary in PATH or common build directories."""
    found = shutil.which(name)
    if found:
        return found
    for d in ["./target/release", "./target/debug"]:
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    return None


def cmd_profile(args):
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    run_dir = os.path.abspath(args.run_dir)
    port = args.port

    # Step 1: Generate demo traces if requested
    if args.demo:
        demo_bin = _find_rust_binary("dynamo-sysprofile-demo")
        if not demo_bin:
            log.error("dynamo-sysprofile-demo not found. Build with: cargo build --release -p dynamo-sysprofile")
            return 1
        log.info("Generating demo traces in %s ...", run_dir)
        rc = subprocess.run([demo_bin, run_dir]).returncode
        if rc != 0:
            log.error("Demo trace generation failed (exit %d)", rc)
            return rc

    if not os.path.isdir(run_dir):
        log.error("%s is not a directory", run_dir)
        return 1

    # Step 2: Run Rust merge (produces merge_result.json)
    merge_bin = _find_rust_binary(args.merge_bin)
    if merge_bin:
        log.info("Merging traces ...")
        rc = subprocess.run([merge_bin, run_dir]).returncode
        if rc != 0:
            log.error("Merge failed (exit %d)", rc)
            return rc
    else:
        if not os.path.exists(os.path.join(run_dir, "merge_result.json")):
            log.error("dynamo-sysprofile-merge not found and no merge_result.json exists. "
                      "Build with: cargo build --release -p dynamo-sysprofile")
            return 1
        log.info("Merge binary not found, using existing merge_result.json")

    # Step 3: Run Python analyzers
    log.info("Running analyzers ...")
    analyze_ns = argparse.Namespace(
        run_dir=run_dir, output_dir=run_dir,
        analyzers=["all"], top_n=20,
    )
    cmd_analyze(analyze_ns)

    # Step 4: Generate report
    log.info("Generating report ...")
    report_ns = argparse.Namespace(
        run_dir=run_dir, merge_result=None,
        stage_attr=None, gpu_util=None, kernels=None, comm=None,
        trace_url=args.trace_url, title=args.title or "sysprofile report",
        output=os.path.join(run_dir, "report.html"),
    )
    rc = cmd_report(report_ns)
    if rc != 0:
        return rc

    # Step 5: Serve and open
    url = f"http://localhost:{port}/report.html"
    log.info("")
    log.info("Report ready: %s", url)
    log.info("Press Ctrl+C to stop the server")

    if args.open:
        webbrowser.open(url)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=run_dir, **kw)
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()
        def log_message(self, fmt, *a):
            pass

    try:
        HTTPServer(("", port), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("Server stopped.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="dynamo-profiler",
        description="Dynamo distributed profiling tools. "
                    "Use 'profile' for the full pipeline in one command.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # profile (the primary one-command entry point)
    p_prof = sub.add_parser("profile",
        help="Full pipeline: merge + analyze + report + serve")
    p_prof.add_argument("run_dir", help="Directory with .pftrace.gz traces (or target for --demo)")
    p_prof.add_argument("--demo", action="store_true",
        help="Generate synthetic demo traces first")
    p_prof.add_argument("--port", type=int, default=9001,
        help="HTTP server port (default: 9001)")
    p_prof.add_argument("--open", action="store_true",
        help="Auto-open report in browser")
    p_prof.add_argument("--trace-url",
        help="Remote URL for Perfetto deep-links (skip local server)")
    p_prof.add_argument("--merge-bin", default="dynamo-sysprofile-merge",
        help="Path to merge binary (default: search PATH + target/)")
    p_prof.add_argument("--title", help="Report title")

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

    if args.command == "profile":
        return cmd_profile(args)
    elif args.command == "report":
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
