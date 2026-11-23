<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rust console test reports

Test steps run `cargo test` and `cargo llvm-cov` directly.
`tee` saves their console output.
Bash `pipefail` preserves test failures.
The test steps do not execute the converter.

```bash
mkdir -p "$GITHUB_WORKSPACE/rust-test-results"
set -o pipefail
cargo test --locked --all-targets 2>&1 | tee "$GITHUB_WORKSPACE/rust-test-results/unit.log"
```

After all test steps, the composite action converts saved logs to JUnit XML.
It then archives logs and reports.
It uploads XML through Datadog's JUnit CLI.
Call it with `if: always()` after the test steps.
Missing credentials skip the upload.
Upload errors fail the reporting step; they cannot hide a failed test step.
The API key is available only during the upload step.

Each invocation needs a distinct log filename within the job.
The `workspace` input identifies the workspace directory in each suite name.
The optional `serial-log` input identifies a log from a single-threaded test command.
The converter needs only Python 3.9 or newer.
It reads existing files without test execution.

## Limits

This adapter parses human-readable output from the standard Rust test harness.
It supports normal test rows, ignored tests, expected panics, and doctest rows.
It compares observed outcomes with each suite's summary.
For explicit single-thread runs, it also joins test prefixes with later outcome lines.
Parallel runs require complete test rows.
Interleaved stderr can cause suite omission during parallel uncaptured runs.
Incomplete or ambiguous suites produce warnings instead of XML results.
Completed suites before a crash remain reportable.
The original logs remain available for omitted suites.
Custom benchmark harnesses remain outside individual test reporting.
Test output that imitates harness lines can defeat text parsing.

Per-test durations are omitted because stable console output does not provide them.
Suite durations come from the harness summary.
Failure elements contain a bounded suite output tail.
Source locations and automatic retries are outside this adapter's scope.

## Local checks

```bash
python3 -m unittest discover -s .github/actions/rust-test-report -p 'test_*.py'
```
