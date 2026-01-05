# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory module for auto-detecting and parsing test results."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def detect_result_type(log_dir: str) -> Optional[str]:
    """Auto-detect the type of test results in a directory.

    Checks for characteristic files to determine if results are from:
    - AI-Perf client: client_N/attempt_M/profile_export_aiperf.json
    - Legacy client: client_N.log.txt with JSONL format

    Args:
        log_dir: Directory containing test results

    Returns:
        "aiperf" if AI-Perf results detected
        "legacy" if legacy client results detected
        None if unable to detect or directory doesn't exist
    """
    if not os.path.exists(log_dir):
        logging.warning(f"Directory does not exist: {log_dir}")
        return None

    if not os.path.isdir(log_dir):
        logging.warning(f"Not a directory: {log_dir}")
        return None

    log_path = Path(log_dir)

    # Check for AI-Perf results
    # Pattern: client_N/attempt_M/profile_export_aiperf.json
    aiperf_indicators = 0
    legacy_indicators = 0

    for item in log_path.iterdir():
        if item.is_dir() and item.name.startswith("client_"):
            # Check for AI-Perf structure
            for attempt_dir in item.iterdir():
                if attempt_dir.is_dir() and attempt_dir.name.startswith("attempt_"):
                    # Look for AI-Perf result files
                    if (attempt_dir / "profile_export_aiperf.json").exists():
                        aiperf_indicators += 1
                        break
                    if (attempt_dir / "profile_export_aiperf.csv").exists():
                        aiperf_indicators += 1
                        break
                    if (attempt_dir / "genai_perf.log").exists():
                        aiperf_indicators += 1
                        break

        # Check for legacy client results
        # Pattern: client_N.log.txt with JSONL content
        if (
            item.is_file()
            and item.name.startswith("client_")
            and item.name.endswith(".log.txt")
        ):
            legacy_indicators += 1

    # Determine result type based on indicators
    if aiperf_indicators > 0 and legacy_indicators == 0:
        return "aiperf"
    elif legacy_indicators > 0 and aiperf_indicators == 0:
        return "legacy"
    elif aiperf_indicators > 0 and legacy_indicators > 0:
        # Mixed results - prioritize AI-Perf as it's newer
        logging.warning(
            f"Mixed result types detected in {log_dir}. "
            f"Found {aiperf_indicators} AI-Perf indicators and {legacy_indicators} legacy indicators. "
            f"Using AI-Perf parser."
        )
        return "aiperf"
    else:
        # No clear indicators
        logging.warning(
            f"Unable to detect result type in {log_dir}. "
            f"No client result files found."
        )
        return None


def parse_test_results(
    log_dir: Optional[str] = None,
    log_paths: Optional[List[str]] = None,
    tablefmt: str = "grid",
    sla: Optional[float] = None,
    success_threshold: float = 90.0,
    force_parser: Optional[str] = None,
    print_output: bool = True,
) -> Any:
    """Auto-detect and parse test results using the appropriate parser.

    This function automatically detects whether results are from the legacy
    client (JSONL format) or AI-Perf client (JSON format) and routes to the
    correct parser.

    Args:
        log_dir: Base directory for logs (for single directory processing)
        log_paths: List of log directories to process (for multiple directories)
        tablefmt: Table format for output (e.g., "fancy_grid", "pipe")
        sla: Optional SLA threshold for latency violations
        success_threshold: Success rate threshold for pass/fail (default: 90.0)
        force_parser: Optional override to force using a specific parser
                     ("aiperf" or "legacy"). If not provided, auto-detection is used.
        print_output: If True, print tables and summaries. If False, only return results.

    Returns:
        Results from the appropriate parser

    Raises:
        ValueError: If force_parser is invalid or unable to detect result type

    Example:
        >>> # Auto-detect and parse single directory
        >>> parse_test_results(log_dir="test_fault_scenario[...]")

        >>> # Auto-detect and parse multiple directories
        >>> parse_test_results(log_paths=["test1", "test2"])

        >>> # Force use of legacy parser
        >>> parse_test_results(log_dir="test_dir", force_parser="legacy")
    """
    # Validate force_parser if provided
    if force_parser is not None:
        if force_parser not in ["aiperf", "legacy"]:
            raise ValueError(
                f"Invalid force_parser value: '{force_parser}'. "
                f"Valid options are: 'aiperf', 'legacy'"
            )

    # Determine which parser to use
    parser_type = None

    if force_parser:
        # Use forced parser without detection
        parser_type = force_parser
        logging.info(f"Using forced parser: {parser_type}")
    else:
        # Auto-detect parser type
        if log_paths:
            # Detect from first log path
            if log_paths:
                parser_type = detect_result_type(log_paths[0])

                # Validate all paths use same type
                for log_path in log_paths[1:]:
                    detected = detect_result_type(log_path)
                    if detected != parser_type:
                        logging.warning(
                            f"Inconsistent result types detected. "
                            f"Using {parser_type} for all paths."
                        )
        elif log_dir:
            # Detect from single directory
            parser_type = detect_result_type(log_dir)
        else:
            raise ValueError("Must provide either log_dir or log_paths")

    if parser_type is None:
        raise ValueError(
            "Unable to auto-detect result type. "
            "Use force_parser='aiperf' or force_parser='legacy' to specify explicitly."
        )

    # Route to appropriate parser
    logging.info(f"Using {parser_type} parser for results")

    if parser_type == "aiperf":
        from tests.fault_tolerance.deploy.parse_results import main as parse_aiperf

        if log_paths:
            return parse_aiperf(
                logs_dir=None,
                log_paths=log_paths,
                tablefmt=tablefmt,
                sla=sla,
                success_threshold=success_threshold,
                print_output=print_output,
            )
        else:
            return parse_aiperf(
                logs_dir=log_dir,
                log_paths=None,
                tablefmt=tablefmt,
                sla=sla,
                success_threshold=success_threshold,
                print_output=print_output,
            )

    elif parser_type == "legacy":
        from tests.fault_tolerance.deploy.legacy_parse_results import (
            main as parse_legacy,
        )

        if log_paths:
            return parse_legacy(
                logs_dir=None,
                log_paths=log_paths,
                tablefmt=tablefmt,
                sla=sla,
                print_output=print_output,
            )
        else:
            return parse_legacy(
                logs_dir=log_dir,
                log_paths=None,
                tablefmt=tablefmt,
                sla=sla,
                print_output=print_output,
            )

    else:
        raise ValueError(f"Unknown parser type: {parser_type}")


def get_result_info(log_dir: str) -> Dict[str, Any]:
    """Get information about test results in a directory.

    Args:
        log_dir: Directory containing test results

    Returns:
        Dictionary with result information:
        {
            "type": "aiperf" or "legacy" or None,
            "client_count": number of clients detected,
            "has_test_log": whether test.log.txt exists,
            "details": additional format-specific details
        }
    """
    info: Dict[str, Any] = {
        "type": None,
        "client_count": 0,
        "has_test_log": False,
        "details": {},
    }

    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        return info

    log_path = Path(log_dir)

    # Check for test.log.txt
    info["has_test_log"] = (log_path / "test.log.txt").exists()

    # Detect result type
    info["type"] = detect_result_type(log_dir)

    # Count clients and gather details
    if info["type"] == "aiperf":
        attempt_counts = []
        for item in log_path.iterdir():
            if item.is_dir() and item.name.startswith("client_"):
                info["client_count"] += 1
                # Count attempts for this client
                attempts = len(
                    [
                        d
                        for d in item.iterdir()
                        if d.is_dir() and d.name.startswith("attempt_")
                    ]
                )
                attempt_counts.append(attempts)

        info["details"]["attempt_counts"] = attempt_counts
        info["details"]["total_attempts"] = sum(attempt_counts)

    elif info["type"] == "legacy":
        for item in log_path.iterdir():
            if (
                item.is_file()
                and item.name.startswith("client_")
                and item.name.endswith(".log.txt")
            ):
                info["client_count"] += 1

    return info


def print_result_info(log_dir: str) -> None:
    """Print human-readable information about test results.

    Args:
        log_dir: Directory containing test results
    """
    info = get_result_info(log_dir)

    logging.info(f"\nTest Results Information: {log_dir}")
    logging.info("=" * 60)
    logging.info(f"Result Type: {info['type'] or 'Unknown'}")
    logging.info(f"Client Count: {info['client_count']}")
    logging.info(f"Has Test Log: {info['has_test_log']}")

    if info["details"]:
        logging.info("\nDetails:")
        for key, value in info["details"].items():
            logging.info(f"  {key}: {value}")

    logging.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-detect and parse fault tolerance test results"
    )
    parser.add_argument(
        "log_dir", nargs="?", default=None, help="Directory containing test results"
    )
    parser.add_argument(
        "--log-paths", nargs="+", help="Multiple log directories to process"
    )
    parser.add_argument(
        "--format", choices=["fancy", "markdown"], default="fancy", help="Table format"
    )
    parser.add_argument(
        "--sla", type=float, default=None, help="SLA threshold for latency"
    )
    parser.add_argument(
        "--force-parser",
        choices=["aiperf", "legacy"],
        default=None,
        help="Force use of specific parser (skip auto-detection)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print information about results without parsing",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Map format choices to tabulate formats
    tablefmt = "fancy_grid" if args.format == "fancy" else "pipe"

    # Info mode
    if args.info:
        if args.log_dir:
            print_result_info(args.log_dir)
        elif args.log_paths:
            for log_path in args.log_paths:
                print_result_info(log_path)
        else:
            logging.error("Must provide log_dir or --log-paths")
    else:
        # Parse mode
        try:
            parse_test_results(
                log_dir=args.log_dir,
                log_paths=args.log_paths,
                tablefmt=tablefmt,
                sla=args.sla,
                force_parser=args.force_parser,
            )
        except Exception as e:
            logging.error(f"Failed to parse results: {e}")
            exit(1)
