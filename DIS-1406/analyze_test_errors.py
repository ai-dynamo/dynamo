#!/usr/bin/env python3
"""
Analyze pytest logs and categorize errors by type (A1, A2, A3, A4).

Error types from reproduce_flaky_test.plan.md:
- A1: KV Prefix Routing Assertion (test_router_decisions)
      "Timed out waiting for router indexer to apply KV events"
      Events spread across different (worker_id, dp_rank) pairs
- A2: Event Sync Mismatch (test_indexers_sync)
      "Router states have different numbers of events"
- A3: Timeouts (test_indexers_sync)
      Test times out waiting for conditions
- A4: Indefinite Hang (test_indexers_sync)
      Test hangs creating second router (global OnceCell/Tokio)

Usage:
    ./analyze_test_errors.py logs/serial_fix13
    ./analyze_test_errors.py logs/serial_fix12 --verbose
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ErrorSummary:
    """Summary of errors found in a test run."""

    run_name: str
    test_name: str
    error_type: str
    error_details: str
    exit_code: Optional[int] = None


class TestLogAnalyzer:
    """Analyzer for pytest log files."""

    # Error patterns for classification
    PATTERNS = {
        "A1_router_decisions": [
            r"Timed out waiting for router indexer to apply KV events",
            r"expected_worker_id=.* expected_dp_rank=.* min_events=\d+ last_count=\d+ total_events=\d+",
        ],
        "A2_sync_mismatch": [
            r"Router states have different numbers of events",
            r"AssertionError.*router_state.*!=.*worker_router_state",
        ],
        "A3_timeout": [
            r"TimeoutError",
            r"asyncio\.TimeoutError",
            r"pytest\.timeout\.Timeout",
        ],
        "A4_crash": [
            r"Fatal Python error: Aborted",
            r"SIGABRT",
            r"Aborted \(core dumped\)",
        ],
        "A4_hang": [
            r"indefinite.*hang",
            r"stuck.*creating.*router",
            r"OnceCell.*deadlock",
        ],
    }

    def __init__(self, log_dir: Path, verbose: bool = False):
        self.log_dir = log_dir
        self.verbose = verbose
        self.errors: List[ErrorSummary] = []

    def analyze_run(self, pytest_log: Path) -> Optional[ErrorSummary]:
        """Analyze a single pytest log file."""
        run_name = pytest_log.parent.name

        try:
            content = pytest_log.read_text()
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not read {pytest_log}: {e}", file=sys.stderr)
            return None

        # Check if test passed - look for pytest summary line
        if re.search(r"=+ \d+ passed.*in \d+", content):
            # Check if there were any failures
            if "FAILED" not in content and not re.search(r"\d+ failed", content):
                return None

        # Extract test name
        test_name = self._extract_test_name(content)

        # Classify error type
        error_type, error_details = self._classify_error(content)

        # Extract exit code if available
        exit_code = self._extract_exit_code(pytest_log.parent)

        return ErrorSummary(
            run_name=run_name,
            test_name=test_name,
            error_type=error_type,
            error_details=error_details,
            exit_code=exit_code,
        )

    def _extract_test_name(self, content: str) -> str:
        """Extract test name from pytest log."""
        # Look for "FAILED tests/..." pattern
        match = re.search(r"FAILED (tests/[^\s]+)", content)
        if match:
            return match.group(1)

        # Look for test collection pattern
        match = re.search(r"collected \d+ item.*\n\n(tests/[^\s]+)", content)
        if match:
            return match.group(1)

        return "unknown"

    def _classify_error(self, content: str) -> tuple[str, str]:
        """Classify error type and extract details."""
        # Check A1: Router decisions (KV event timeout)
        if any(
            re.search(p, content, re.IGNORECASE)
            for p in self.PATTERNS["A1_router_decisions"]
        ):
            # Extract detailed error line
            match = re.search(
                r"AssertionError: Timed out waiting for router indexer.*\n.*top_keys=\[.*?\]",
                content,
                re.DOTALL,
            )
            if match:
                detail = match.group(0).replace("\n", " ").strip()
                # Shorten for readability
                if len(detail) > 150:
                    detail = detail[:150] + "..."
                return "A1_router_decisions", detail
            return "A1_router_decisions", "Timed out waiting for KV events"

        # Check A2: Sync mismatch
        if any(
            re.search(p, content, re.IGNORECASE)
            for p in self.PATTERNS["A2_sync_mismatch"]
        ):
            match = re.search(
                r"AssertionError:.*different numbers of events.*", content
            )
            if match:
                return "A2_sync_mismatch", match.group(0)[:100]
            return "A2_sync_mismatch", "Router states have different numbers of events"

        # Check A3: Timeout
        if any(
            re.search(p, content, re.IGNORECASE) for p in self.PATTERNS["A3_timeout"]
        ):
            match = re.search(r"(TimeoutError|asyncio\.TimeoutError).*", content)
            if match:
                return "A3_timeout", match.group(0)[:100]
            return "A3_timeout", "Test timed out"

        # Check A4_crash: Python crash (SIGABRT)
        if any(re.search(p, content, re.IGNORECASE) for p in self.PATTERNS["A4_crash"]):
            match = re.search(r"Fatal Python error: ([^\n]+)", content)
            if match:
                return "A4_crash", f"Python crash: {match.group(1)}"
            return "A4_crash", "Python crashed (SIGABRT)"

        # Check A4_hang: Hang
        if any(re.search(p, content, re.IGNORECASE) for p in self.PATTERNS["A4_hang"]):
            return "A4_hang", "Test hung (OnceCell/Tokio deadlock)"

        # Unknown error - try to extract assertion
        match = re.search(r"AssertionError: ([^\n]+)", content)
        if match:
            return "UNKNOWN", match.group(1)[:100]

        # Generic failure
        if "FAILED" in content:
            return "UNKNOWN", "Test failed (see log for details)"

        return "UNKNOWN", "No clear error pattern found"

    def _extract_exit_code(self, run_dir: Path) -> Optional[int]:
        """Extract exit code from .exit_code file."""
        exit_code_file = run_dir / ".exit_code"
        if exit_code_file.exists():
            try:
                return int(exit_code_file.read_text().strip())
            except Exception:
                pass
        return None

    def analyze_all(self) -> Dict[str, List[ErrorSummary]]:
        """Analyze all pytest logs in the directory."""
        pytest_logs = sorted(self.log_dir.glob("run*/pytest.log"))

        if not pytest_logs:
            print(f"Error: No pytest logs found in {self.log_dir}", file=sys.stderr)
            sys.exit(1)

        errors_by_type = defaultdict(list)

        for log in pytest_logs:
            error = self.analyze_run(log)
            if error:
                errors_by_type[error.error_type].append(error)
                self.errors.append(error)

        return errors_by_type

    def print_summary(self, errors_by_type: Dict[str, List[ErrorSummary]]):
        """Print summary of errors."""
        total_runs = len(list(self.log_dir.glob("run*")))
        total_failures = len(self.errors)
        total_passes = total_runs - total_failures

        print("=" * 80)
        print(f"Test Error Analysis: {self.log_dir}")
        print("=" * 80)
        print(f"Total runs: {total_runs}")
        print(f"Passed: {total_passes} ({100 * total_passes / total_runs:.1f}%)")
        print(f"Failed: {total_failures} ({100 * total_failures / total_runs:.1f}%)")
        print()

        if not self.errors:
            print("✓ All tests passed!")
            return

        print("Error Breakdown by Type:")
        print("-" * 80)

        # Sort by error type
        for error_type in sorted(errors_by_type.keys()):
            errors = errors_by_type[error_type]
            count = len(errors)
            pct = 100 * count / total_runs

            print(f"\n{error_type}: {count}/{total_runs} ({pct:.1f}%)")

            # Group by test name
            by_test = defaultdict(list)
            for err in errors:
                by_test[err.test_name].append(err)

            for test_name, test_errors in sorted(by_test.items()):
                print(f"  {test_name}: {len(test_errors)} failures")

                if self.verbose:
                    for err in test_errors[:3]:  # Show first 3
                        print(f"    - {err.run_name}: {err.error_details}")
                    if len(test_errors) > 3:
                        print(f"    ... and {len(test_errors) - 3} more")

        print()
        print("=" * 80)

        # List all failed runs
        print("\nFailed runs:")
        for err in sorted(self.errors, key=lambda e: e.run_name):
            print(f"  {err.run_name}: {err.error_type} - {err.test_name}")

        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze pytest logs and categorize errors by type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing run* subdirectories with pytest.log files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed error messages",
    )

    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: Directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    analyzer = TestLogAnalyzer(args.log_dir, verbose=args.verbose)
    errors_by_type = analyzer.analyze_all()
    analyzer.print_summary(errors_by_type)


if __name__ == "__main__":
    main()
