#!/usr/bin/env python3
"""
Phase 0: Error Pattern Analysis Script

Analyzes recent workflow failures to validate the 10 proposed error categories
against real failure patterns. Generates a report with error examples and
recommendations.

Usage:
    export GITHUB_TOKEN=<your-token>
    export REPO=ai-dynamo/dynamo
    python3 analyze_recent_errors.py --hours 48 --output error_analysis_report.md
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import requests


class ErrorPatternAnalyzer:
    """Analyze error patterns from recent workflow failures."""

    def __init__(self):
        """Initialize analyzer with GitHub API credentials."""
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("REPO", "ai-dynamo/dynamo")

        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        self.github_headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "error-pattern-analyzer/1.0"
        }

        # Proposed categories from plan
        self.categories = [
            "dependency_error",
            "timeout",
            "resource_exhaustion",
            "network_error",
            "assertion_failure",
            "compilation_error",
            "runtime_error",
            "infrastructure_error",
            "configuration_error",
            "flaky_test",
        ]

        # Pattern matching for rough classification
        self.category_patterns = {
            "dependency_error": [
                r"could not find a version",
                r"no matching distribution",
                r"importerror",
                r"modulenotfounderror",
                r"version conflict",
                r"cuda.*version",
            ],
            "timeout": [
                r"timeout",
                r"timed out",
                r"exceeded.*time",
                r"deadlock",
            ],
            "resource_exhaustion": [
                r"out of memory",
                r"oom",
                r"no space left",
                r"disk.*full",
                r"cuda out of memory",
            ],
            "network_error": [
                r"connection.*refused",
                r"connection.*reset",
                r"dns.*error",
                r"could not resolve",
                r"network.*error",
            ],
            "assertion_failure": [
                r"assertionerror",
                r"assert.*==",
                r"expected.*but.*got",
            ],
            "compilation_error": [
                r"compilation.*failed",
                r"build.*failed",
                r"undefined reference",
                r"cargo.*error",
            ],
            "runtime_error": [
                r"segmentation fault",
                r"segfault",
                r"nullpointerexception",
                r"indexerror",
                r"keyerror",
            ],
            "infrastructure_error": [
                r"runner.*failed",
                r"docker.*daemon",
                r"artifact.*failed",
            ],
            "configuration_error": [
                r"filenotfounderror",
                r"permission denied",
                r"invalid.*config",
            ],
            "flaky_test": [
                r"race condition",
                r"non-deterministic",
            ],
        }

    def fetch_recent_failures(self, hours_back: int = 48) -> List[Dict[str, Any]]:
        """
        Fetch failed workflows from the past N hours.

        Args:
            hours_back: How many hours back to look

        Returns:
            List of workflow run data
        """
        print(f"🔍 Fetching failed workflows from past {hours_back} hours...")

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        # Fetch workflow runs
        url = f"https://api.github.com/repos/{self.repo}/actions/runs"
        params = {
            "status": "completed",
            "per_page": 100,
            "created": f">={cutoff_time.isoformat()}"
        }

        all_runs = []
        page = 1

        while True:
            params["page"] = page
            response = requests.get(url, headers=self.github_headers, params=params)

            if response.status_code != 200:
                print(f"✗ Error fetching workflows: {response.status_code}")
                break

            data = response.json()
            runs = data.get("workflow_runs", [])

            if not runs:
                break

            # Filter for failures
            failed_runs = [run for run in runs if run.get("conclusion") == "failure"]
            all_runs.extend(failed_runs)

            print(f"  Page {page}: Found {len(failed_runs)} failed runs")

            page += 1
            if page > 10:  # Safety limit
                break

        print(f"✓ Found {len(all_runs)} failed workflow runs")
        return all_runs

    def extract_error_messages(self, workflow_runs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract error messages from workflow runs.

        Args:
            workflow_runs: List of workflow run data

        Returns:
            List of error samples with metadata
        """
        print(f"📋 Extracting error messages from {len(workflow_runs)} workflows...")

        error_samples = []

        for run in workflow_runs:
            workflow_id = run.get("id")
            workflow_name = run.get("name")

            # Fetch jobs for this workflow
            jobs_url = run.get("jobs_url")
            if not jobs_url:
                continue

            response = requests.get(jobs_url, headers=self.github_headers)
            if response.status_code != 200:
                continue

            jobs_data = response.json()
            jobs = jobs_data.get("jobs", [])

            # Extract errors from failed jobs
            for job in jobs:
                if job.get("conclusion") != "failure":
                    continue

                job_name = job.get("name")

                # Get annotations (errors/failures)
                annotations_url = f"https://api.github.com/repos/{self.repo}/check-runs/{job['id']}/annotations"
                annotations_response = requests.get(annotations_url, headers=self.github_headers)

                if annotations_response.status_code == 200:
                    annotations = annotations_response.json()

                    for annotation in annotations:
                        level = annotation.get("annotation_level", "")
                        if level in ["failure", "error"]:
                            message = annotation.get("message", "")
                            if message:
                                error_samples.append({
                                    "workflow_id": workflow_id,
                                    "workflow_name": workflow_name,
                                    "job_name": job_name,
                                    "error_text": message,
                                    "source": "github_annotation",
                                })

        print(f"✓ Extracted {len(error_samples)} error messages")
        return error_samples

    def classify_error(self, error_text: str) -> Optional[str]:
        """
        Roughly classify error using pattern matching.

        Args:
            error_text: Error message

        Returns:
            Category name or None
        """
        error_lower = error_text.lower()

        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return category

        return None

    def analyze_patterns(self, error_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze error patterns and generate statistics.

        Args:
            error_samples: List of error samples

        Returns:
            Analysis results
        """
        print(f"📊 Analyzing error patterns...")

        # Classify errors
        category_counts = Counter()
        category_examples = defaultdict(list)
        unclassified = []

        for sample in error_samples:
            error_text = sample["error_text"]
            category = self.classify_error(error_text)

            if category:
                category_counts[category] += 1

                # Keep first 5 examples per category
                if len(category_examples[category]) < 5:
                    category_examples[category].append({
                        "workflow_name": sample["workflow_name"],
                        "job_name": sample["job_name"],
                        "error_text": error_text[:500],  # Truncate
                    })
            else:
                unclassified.append(sample)

        # Calculate coverage
        classified_count = sum(category_counts.values())
        total_count = len(error_samples)
        coverage = (classified_count / total_count * 100) if total_count > 0 else 0

        return {
            "total_errors": total_count,
            "classified_count": classified_count,
            "unclassified_count": len(unclassified),
            "coverage_percent": coverage,
            "category_counts": dict(category_counts),
            "category_examples": dict(category_examples),
            "unclassified_samples": unclassified[:10],  # First 10
        }

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """
        Generate markdown report.

        Args:
            analysis: Analysis results

        Returns:
            Markdown report string
        """
        report = []
        report.append("# Error Pattern Analysis Report")
        report.append("")
        report.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}")
        report.append("")

        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total Errors**: {analysis['total_errors']}")
        report.append(f"- **Classified**: {analysis['classified_count']} ({analysis['coverage_percent']:.1f}%)")
        report.append(f"- **Unclassified**: {analysis['unclassified_count']}")
        report.append("")

        # Category distribution
        report.append("## Category Distribution")
        report.append("")
        report.append("| Category | Count | Percentage |")
        report.append("|----------|-------|------------|")

        category_counts = analysis["category_counts"]
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        for category, count in sorted_categories:
            pct = (count / analysis["total_errors"] * 100) if analysis["total_errors"] > 0 else 0
            report.append(f"| {category} | {count} | {pct:.1f}% |")

        report.append("")

        # Examples for each category
        report.append("## Error Examples by Category")
        report.append("")

        category_examples = analysis["category_examples"]

        for category in self.categories:
            if category in category_examples:
                examples = category_examples[category]
                report.append(f"### {category}")
                report.append("")
                report.append(f"**Count**: {category_counts.get(category, 0)}")
                report.append("")

                for i, example in enumerate(examples, 1):
                    report.append(f"**Example {i}**:")
                    report.append(f"- Workflow: {example['workflow_name']}")
                    report.append(f"- Job: {example['job_name']}")
                    report.append(f"- Error:")
                    report.append("```")
                    report.append(example['error_text'])
                    report.append("```")
                    report.append("")

        # Unclassified samples
        if analysis["unclassified_samples"]:
            report.append("## Unclassified Errors")
            report.append("")
            report.append("These errors did not match any category patterns:")
            report.append("")

            for i, sample in enumerate(analysis["unclassified_samples"], 1):
                report.append(f"**Sample {i}**:")
                report.append(f"- Workflow: {sample['workflow_name']}")
                report.append(f"- Job: {sample['job_name']}")
                report.append(f"- Error:")
                report.append("```")
                report.append(sample['error_text'][:500])
                report.append("```")
                report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        if analysis["coverage_percent"] >= 90:
            report.append("✅ **Good coverage**: The 10 proposed categories cover most errors.")
        elif analysis["coverage_percent"] >= 70:
            report.append("⚠️ **Moderate coverage**: Consider refining category definitions or adding patterns.")
        else:
            report.append("❌ **Low coverage**: Categories may need significant refinement.")

        report.append("")

        # Suggest missing categories
        if analysis["unclassified_count"] > analysis["total_errors"] * 0.2:
            report.append("Consider analyzing unclassified errors to identify missing categories.")

        return "\n".join(report)

    def run_analysis(self, hours_back: int, output_file: str):
        """
        Run full analysis pipeline.

        Args:
            hours_back: Hours to look back
            output_file: Output markdown file path
        """
        print("=" * 60)
        print("ERROR PATTERN ANALYSIS")
        print("=" * 60)

        # Fetch failures
        workflow_runs = self.fetch_recent_failures(hours_back)

        if not workflow_runs:
            print("✗ No failed workflows found")
            return

        # Extract errors
        error_samples = self.extract_error_messages(workflow_runs)

        if not error_samples:
            print("✗ No error messages extracted")
            return

        # Analyze patterns
        analysis = self.analyze_patterns(error_samples)

        # Generate report
        report = self.generate_report(analysis)

        # Write to file
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"✅ Report written to {output_file}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze error patterns from recent workflow failures"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="Hours to look back (default: 48)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="error_analysis_report.md",
        help="Output file path (default: error_analysis_report.md)"
    )

    args = parser.parse_args()

    try:
        analyzer = ErrorPatternAnalyzer()
        analyzer.run_analysis(args.hours, args.output)

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
