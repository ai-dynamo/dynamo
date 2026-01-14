#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Coverage Analysis Script for Nightly Builds

This script analyzes coverage data from different test types (unit, integration, e2e)
and generates incremental coverage reports showing what each test type uniquely covers.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict


def parse_coverage_report(report_path: Path) -> Dict[str, float]:
    """
    Parse a coverage report text file and extract coverage percentage.

    Expected format from 'coverage report':
    Name                      Stmts   Miss  Cover
    ---------------------------------------------
    module1.py                  100     10    90%
    module2.py                   50      5    90%
    ---------------------------------------------
    TOTAL                       150     15    90%
    """
    if not report_path.exists():
        print(f"Warning: Coverage report not found: {report_path}")
        return {"total": 0.0, "statements": 0, "missing": 0}

    with open(report_path, "r") as f:
        content = f.read()

    # Find the TOTAL line
    total_match = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", content)
    if total_match:
        statements = int(total_match.group(1))
        missing = int(total_match.group(2))
        coverage = int(total_match.group(3))
        return {
            "total": coverage,
            "statements": statements,
            "missing": missing,
            "covered": statements - missing,
        }

    return {"total": 0.0, "statements": 0, "missing": 0, "covered": 0}


def calculate_incremental_coverage(
    unit_data: Dict, integration_data: Dict, e2e_data: Dict
) -> str:
    """
    Calculate incremental coverage showing what each test type adds.

    Returns a formatted string showing the incremental analysis.
    """
    unit_cov = unit_data.get("total", 0)
    unit_covered = unit_data.get("covered", 0)

    integration_cov = integration_data.get("total", 0)
    integration_covered = integration_data.get("covered", 0)

    e2e_cov = e2e_data.get("total", 0)
    e2e_covered = e2e_data.get("covered", 0)

    # Calculate incremental increases
    integration_increase = integration_cov - unit_cov
    e2e_increase = e2e_cov - integration_cov

    # Calculate lines uniquely covered by each type
    integration_unique_lines = integration_covered - unit_covered
    e2e_unique_lines = e2e_covered - integration_covered

    report = f"""
## Python Test Coverage - Incremental Analysis

### Coverage by Test Type
```
Unit tests only:              {unit_cov}%
+ Integration tests:          {integration_cov}% (+{integration_increase}%)
+ End-to-end tests:           {e2e_cov}% (+{e2e_increase}%)
```

### Lines Covered by Test Type
```
Unit tests:                   {unit_covered:,} lines
+ Integration tests:          {integration_covered:,} lines (+{integration_unique_lines:,})
+ End-to-end tests:           {e2e_covered:,} lines (+{e2e_unique_lines:,})
```

### Lines Only Covered By
```
- Integration tests:  {integration_unique_lines:,} lines
- E2E tests:          {e2e_unique_lines:,} lines
```

### Total Statements
```
Total statements:     {e2e_data.get('statements', 0):,}
Covered:              {e2e_covered:,}
Missing:              {e2e_data.get('missing', 0):,}
Final coverage:       {e2e_cov}%
```
"""

    return report


def generate_json_summary(
    unit_data: Dict, integration_data: Dict, e2e_data: Dict, output_path: Path
) -> None:
    """Generate a JSON summary for historical tracking."""
    from datetime import datetime

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "unit": unit_data,
        "integration": integration_data,
        "e2e": e2e_data,
        "incremental": {
            "unit_only": unit_data.get("total", 0),
            "integration_increase": integration_data.get("total", 0)
            - unit_data.get("total", 0),
            "e2e_increase": e2e_data.get("total", 0) - integration_data.get("total", 0),
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ JSON summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze incremental test coverage")
    parser.add_argument(
        "--unit", type=Path, required=True, help="Path to unit test coverage report"
    )
    parser.add_argument(
        "--integration",
        type=Path,
        required=True,
        help="Path to unit+integration coverage report",
    )
    parser.add_argument(
        "--e2e",
        type=Path,
        required=True,
        help="Path to unit+integration+e2e coverage report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coverage-incremental-summary.txt"),
        help="Output file for incremental summary",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("coverage-data.json"),
        help="Output file for JSON data",
    )

    args = parser.parse_args()

    print("🔍 Parsing coverage reports...")

    # Parse coverage reports
    unit_data = parse_coverage_report(args.unit)
    integration_data = parse_coverage_report(args.integration)
    e2e_data = parse_coverage_report(args.e2e)

    print(f"  Unit coverage:        {unit_data.get('total', 0)}%")
    print(f"  Integration coverage: {integration_data.get('total', 0)}%")
    print(f"  E2E coverage:         {e2e_data.get('total', 0)}%")

    # Generate incremental report
    print("\n📊 Generating incremental coverage report...")
    report = calculate_incremental_coverage(unit_data, integration_data, e2e_data)

    # Save report
    with open(args.output, "w") as f:
        f.write(report)

    print(f"✅ Incremental summary saved to: {args.output}")

    # Generate JSON summary
    generate_json_summary(unit_data, integration_data, e2e_data, args.json_output)

    # Print report to stdout
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
