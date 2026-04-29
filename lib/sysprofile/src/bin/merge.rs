// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo-sysprofile-merge` — merge per-component traces into a single
//! time-aligned Perfetto trace with critical-path attribution and HTML report.
//!
//! With `--full-report`, also invokes the Python `dynamo_profiler` pipeline
//! for deep analysis (kernel hotlist, GPU utilization, comm breakdown).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dynamo-sysprofile-merge <run-dir> [--trace-url <url>] [--full-report]");
        eprintln!();
        eprintln!("  <run-dir>       Directory containing .pftrace.gz files from a benchmark run");
        eprintln!("  --trace-url     Optional URL to the merged trace for Perfetto deep-links");
        eprintln!("  --full-report   Also run Python analyzers for deep analysis sections");
        std::process::exit(1);
    }

    let run_dir = PathBuf::from(&args[1]);
    let trace_url = args
        .windows(2)
        .find(|w| w[0] == "--trace-url")
        .map(|w| w[1].as_str());
    let full_report = args.iter().any(|a| a == "--full-report");

    if !run_dir.is_dir() {
        eprintln!("error: {} is not a directory", run_dir.display());
        std::process::exit(1);
    }

    eprintln!("sysprofile-merge: reading traces from {}", run_dir.display());

    let result = match dynamo_sysprofile::merger::merge_run(&run_dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: merge failed: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "  {} requests, {} slices across {} components",
        result.total_requests,
        result.total_slices,
        result.components.len()
    );
    eprintln!("  p99 = {:.1}ms, p50 = {:.1}ms", result.p99_total_ms, result.p50_total_ms);

    // Write merge result as JSON
    let json_path = run_dir.join("merge_result.json");
    match serde_json::to_string_pretty(&result) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&json_path, &json) {
                eprintln!("warning: could not write {}: {e}", json_path.display());
            } else {
                eprintln!("  wrote {}", json_path.display());
            }
        }
        Err(e) => eprintln!("warning: could not serialize merge result: {e}"),
    }

    if full_report {
        // Invoke Python pipeline for deep analysis + unified report
        eprintln!("  running Python analyzers (--full-report)...");
        let python_ok = run_python_pipeline(&run_dir, trace_url);
        if python_ok {
            eprintln!();
            let report_path = run_dir.join("report.html");
            eprintln!(
                "Open in browser: file://{}",
                report_path.canonicalize().unwrap_or(report_path).display()
            );
            return;
        }
        eprintln!("  Python pipeline unavailable, falling back to Rust-only report");
    }

    // Generate Rust-only HTML report (default path, or fallback)
    let html = dynamo_sysprofile::report::generate_report(&result, trace_url);
    let report_path = run_dir.join("report.html");
    match std::fs::write(&report_path, &html) {
        Ok(()) => {
            eprintln!("  wrote {}", report_path.display());
            eprintln!();
            eprintln!("Open in browser: file://{}", report_path.canonicalize().unwrap_or(report_path).display());
        }
        Err(e) => {
            eprintln!("error: could not write report: {e}");
            std::process::exit(1);
        }
    }
}

/// Run the Python dynamo_profiler pipeline: analyze traces, then generate
/// unified report. Returns true if Python was available and succeeded.
fn run_python_pipeline(run_dir: &PathBuf, trace_url: Option<&str>) -> bool {
    // Try python3 first, then python
    let python = find_python();
    let python = match python {
        Some(p) => p,
        None => {
            eprintln!("  warning: python3/python not found in PATH");
            return false;
        }
    };

    // Step 1: Run analyzers on the trace files
    let analyze_status = Command::new(&python)
        .args(["-m", "dynamo_profiler", "analyze"])
        .arg(run_dir)
        .args(["--analyzers", "all"])
        .status();

    match analyze_status {
        Ok(s) if s.success() => {
            eprintln!("  analyzers complete");
        }
        Ok(s) => {
            eprintln!("  warning: analyzers exited with {}, continuing with available data", s);
        }
        Err(e) => {
            eprintln!("  warning: could not run analyzers: {e}");
            // Continue — report can still use merge_result.json alone
        }
    }

    // Step 2: Generate unified report
    let mut report_args = vec![
        "-m".to_string(),
        "dynamo_profiler".to_string(),
        "report".to_string(),
        run_dir.display().to_string(),
    ];
    if let Some(url) = trace_url {
        report_args.push("--trace-url".to_string());
        report_args.push(url.to_string());
    }

    let report_status = Command::new(&python)
        .args(&report_args)
        .status();

    match report_status {
        Ok(s) if s.success() => {
            eprintln!("  report written to {}", run_dir.join("report.html").display());
            true
        }
        Ok(s) => {
            eprintln!("  warning: report generation exited with {}", s);
            false
        }
        Err(e) => {
            eprintln!("  warning: could not generate report: {e}");
            false
        }
    }
}

fn find_python() -> Option<String> {
    for cmd in ["python3", "python"] {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                return Some(cmd.to_string());
            }
        }
    }
    None
}
