#!/usr/bin/env python3
"""
Test error classification on real workflow failures from the last 6 hours.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensearch_upload.error_classification import (
    Config,
    ErrorClassifier,
    ErrorContext,
    ERROR_CATEGORIES,
)


class RealErrorTester:
    """Test error classification on real workflow failures."""

    def __init__(self):
        """Initialize with GitHub and NVIDIA credentials."""
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("REPO", "ai-dynamo/dynamo")

        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        self.github_headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "error-classification-tester/1.0"
        }

        # Read NVIDIA API key
        api_key_file = os.path.expanduser("~/.claude2")
        if os.path.exists(api_key_file):
            with open(api_key_file, "r") as f:
                api_key = f.read().strip()
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("API key not found in ~/.claude2 or ANTHROPIC_API_KEY")

        # Create classifier with NVIDIA API
        self.config = Config(
            anthropic_api_key=api_key,
            anthropic_model="aws/anthropic/claude-opus-4-5",
            api_format="openai",
            api_base_url="https://inference-api.nvidia.com/v1",
            max_error_length=10000,
        )

        self.classifier = ErrorClassifier(self.config, opensearch_client=None)

    def fetch_recent_failures(self, hours_back: int = 6) -> List[Dict[str, Any]]:
        """Fetch failed workflows from the past N hours."""
        print(f"🔍 Fetching failed workflows from past {hours_back} hours...")

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        url = f"https://api.github.com/repos/{self.repo}/actions/runs"
        params = {
            "status": "completed",
            "per_page": 100,
            "created": f">={cutoff_time.isoformat()}"
        }

        all_runs = []
        page = 1

        while page <= 3:  # Limit to 3 pages to avoid rate limits
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

        print(f"✓ Found {len(all_runs)} failed workflow runs")
        return all_runs

    def extract_error_from_job(self, job: Dict) -> List[Dict[str, Any]]:
        """Extract errors from a failed job."""
        errors = []
        job_name = job.get("name", "unknown")

        # Get annotations (errors/failures)
        annotations_url = f"https://api.github.com/repos/{self.repo}/check-runs/{job['id']}/annotations"
        response = requests.get(annotations_url, headers=self.github_headers)

        if response.status_code == 200:
            annotations = response.json()

            for annotation in annotations:
                level = annotation.get("annotation_level", "")
                if level in ["failure", "error"]:
                    message = annotation.get("message", "")
                    if message and len(message) > 50:  # Skip very short messages
                        errors.append({
                            "job_name": job_name,
                            "error_text": message,
                            "source": "github_annotation",
                        })

        return errors

    def extract_errors_from_workflows(
        self,
        workflow_runs: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract error messages from workflow runs."""
        print(f"\n📋 Extracting errors from workflows...")

        all_errors = []

        for i, run in enumerate(workflow_runs[:10], 1):  # Limit to 10 workflows
            workflow_id = run.get("id")
            workflow_name = run.get("name")

            print(f"  [{i}/10] Processing: {workflow_name}...")

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
                if job.get("conclusion") == "failure":
                    job_errors = self.extract_error_from_job(job)

                    for error in job_errors:
                        error["workflow_id"] = workflow_id
                        error["workflow_name"] = workflow_name
                        all_errors.append(error)

        print(f"✓ Extracted {len(all_errors)} error messages")
        return all_errors

    def classify_errors(
        self,
        errors: List[Dict[str, Any]],
        max_errors: int = 20
    ) -> List[Dict[str, Any]]:
        """Classify errors using the error classification system."""
        print(f"\n🤖 Classifying errors with Claude (NVIDIA API)...")
        print(f"   Model: {self.config.anthropic_model}")
        print(f"   Max errors to classify: {max_errors}")

        # Deduplicate errors by hash
        from opensearch_upload.error_classification import ErrorDeduplicator
        deduplicator = ErrorDeduplicator()

        unique_errors = {}
        for error in errors:
            error_hash = deduplicator.compute_error_hash(error["error_text"])
            if error_hash not in unique_errors:
                unique_errors[error_hash] = error
                error["error_hash"] = error_hash
                error["occurrence_count"] = 1
            else:
                unique_errors[error_hash]["occurrence_count"] += 1

        errors_to_classify = list(unique_errors.values())[:max_errors]

        print(f"   Unique errors: {len(unique_errors)} (from {len(errors)} total)")
        print(f"   Classifying: {len(errors_to_classify)} errors")
        print()

        classifications = []

        for i, error in enumerate(errors_to_classify, 1):
            print(f"  [{i}/{len(errors_to_classify)}] {error['job_name'][:50]}...")

            # Create ErrorContext
            error_context = ErrorContext(
                error_text=error["error_text"],
                source_type=error["source"],
                workflow_id=str(error.get("workflow_id", "")),
                workflow_name=error.get("workflow_name", ""),
                job_name=error["job_name"],
            )

            try:
                # Classify
                classification = self.classifier.classify_error(
                    error_context,
                    use_cache=False,
                    classification_method="test"
                )

                result = {
                    "workflow_name": error["workflow_name"],
                    "job_name": error["job_name"],
                    "error_snippet": error["error_text"][:200],
                    "category": classification.primary_category,
                    "confidence": classification.confidence_score,
                    "root_cause": classification.root_cause_summary,
                    "occurrence_count": error.get("occurrence_count", 1),
                    "error_hash": error.get("error_hash", ""),
                    "prompt_tokens": classification.prompt_tokens,
                    "completion_tokens": classification.completion_tokens,
                }

                classifications.append(result)

                print(f"      → {classification.primary_category} ({classification.confidence_score:.1%})")

            except Exception as e:
                print(f"      ✗ Failed: {e}")
                continue

        print(f"\n✓ Classified {len(classifications)} errors")
        return classifications

    def generate_report(
        self,
        classifications: List[Dict[str, Any]],
        hours_back: int
    ) -> str:
        """Generate summary report."""
        report = []
        report.append("=" * 80)
        report.append("ERROR CLASSIFICATION REPORT")
        report.append("=" * 80)
        report.append(f"\n**Time Range**: Last {hours_back} hours")
        report.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}")
        report.append(f"**Repository**: {self.repo}")
        report.append(f"**Model**: {self.config.anthropic_model}")
        report.append("")

        # Summary statistics
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total Errors Classified**: {len(classifications)}")

        total_occurrences = sum(c.get("occurrence_count", 1) for c in classifications)
        report.append(f"- **Total Error Occurrences**: {total_occurrences}")

        avg_confidence = sum(c["confidence"] for c in classifications) / len(classifications) if classifications else 0
        report.append(f"- **Average Confidence**: {avg_confidence:.1%}")

        # Token usage
        total_prompt = sum(c.get("prompt_tokens", 0) for c in classifications)
        total_completion = sum(c.get("completion_tokens", 0) for c in classifications)
        report.append(f"- **Total Tokens Used**: {total_prompt + total_completion:,}")
        report.append(f"  - Prompt: {total_prompt:,}")
        report.append(f"  - Completion: {total_completion:,}")
        report.append("")

        # Category distribution
        report.append("## Category Distribution")
        report.append("")

        category_counts = Counter(c["category"] for c in classifications)
        category_occurrences = defaultdict(int)
        for c in classifications:
            category_occurrences[c["category"]] += c.get("occurrence_count", 1)

        report.append("| Category | Unique Errors | Total Occurrences | Percentage |")
        report.append("|----------|---------------|-------------------|------------|")

        for category, count in category_counts.most_common():
            occurrences = category_occurrences[category]
            pct = (count / len(classifications) * 100) if classifications else 0
            report.append(f"| {category} | {count} | {occurrences} | {pct:.1f}% |")

        report.append("")

        # High confidence classifications
        report.append("## Sample Classifications")
        report.append("")

        # Group by category
        by_category = defaultdict(list)
        for c in classifications:
            by_category[c["category"]].append(c)

        for category in sorted(by_category.keys()):
            examples = by_category[category][:2]  # Top 2 per category

            report.append(f"### {category}")
            report.append("")

            for example in examples:
                report.append(f"**Workflow**: {example['workflow_name']}")
                report.append(f"**Job**: {example['job_name']}")
                report.append(f"**Confidence**: {example['confidence']:.1%}")
                report.append(f"**Occurrences**: {example.get('occurrence_count', 1)}")
                report.append("")
                report.append("**Root Cause**:")
                report.append(example['root_cause'])
                report.append("")
                report.append("**Error Snippet**:")
                report.append("```")
                report.append(example['error_snippet'])
                report.append("```")
                report.append("")

        # Cost estimate
        report.append("## Cost Analysis")
        report.append("")
        # Rough estimate: $3/million input tokens, $15/million output tokens for Opus
        input_cost = (total_prompt / 1_000_000) * 3
        output_cost = (total_completion / 1_000_000) * 15
        total_cost = input_cost + output_cost
        report.append(f"- **Estimated Cost**: ${total_cost:.4f}")
        report.append(f"  - Input tokens: ${input_cost:.4f}")
        report.append(f"  - Output tokens: ${output_cost:.4f}")
        report.append("")
        report.append(f"- **Cost per error**: ${total_cost / len(classifications):.4f}" if classifications else "")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def run_test(self, hours_back: int, max_errors: int, output_file: str):
        """Run full test pipeline."""
        print("=" * 80)
        print("REAL ERROR CLASSIFICATION TEST")
        print("=" * 80)
        print()

        # Fetch failures
        workflow_runs = self.fetch_recent_failures(hours_back)

        if not workflow_runs:
            print("\n✗ No failed workflows found in the time range")
            return

        # Extract errors
        errors = self.extract_errors_from_workflows(workflow_runs)

        if not errors:
            print("\n✗ No error messages extracted")
            return

        # Classify errors
        classifications = self.classify_errors(errors, max_errors)

        if not classifications:
            print("\n✗ No errors were classified")
            return

        # Generate report
        report = self.generate_report(classifications, hours_back)

        # Print to console
        print("\n")
        print(report)

        # Write to file
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\n✅ Report written to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test error classification on real workflow failures"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=6,
        help="Hours to look back (default: 6)"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum errors to classify (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="real_error_classification_report.md",
        help="Output file path (default: real_error_classification_report.md)"
    )

    args = parser.parse_args()

    try:
        tester = RealErrorTester()
        tester.run_test(args.hours, args.max_errors, args.output)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
