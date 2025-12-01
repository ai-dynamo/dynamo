#!/usr/bin/env python3
"""
Extract errors from logs using Salesforce LogAI.
This script analyzes log files and extracts the most relevant error messages.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from logai.information_extraction.log_parser import LogParser  # noqa: F401
    from logai.preprocess.preprocessor import Preprocessor  # noqa: F401

    LOGAI_AVAILABLE = True
    LOGAI_IMPORT_ERROR = None

except ImportError as e:
    LOGAI_AVAILABLE = False
    LOGAI_IMPORT_ERROR = str(e)
    print(
        f"Warning: LogAI not available ({e}), using fallback error extraction",
        file=sys.stderr,
    )


class LogErrorExtractor:
    """Extract errors from log files using LogAI or fallback methods."""

    # Common error patterns for fallback - just extract the error lines
    ERROR_PATTERNS = [
        # Kubernetes specific errors
        r"error:\s+(no matching resources found)(?:\s+.*)?",
        r"error:\s+(timed out waiting for the condition)(?:\s+.*)?",
        r"kubectl\s+.*\s+error:\s+(.+?)(?:\n|$)",
        # General errors
        r"Error:?\s+(.+?)(?:\n|$)",
        r"ERROR[:\s]+(.+?)(?:\n|$)",
        r"Failed\s+(.+?)(?:\n|$)",
        r"failed\s+(.+?)(?:\n|$)",
        r"FAILED[:\s]+(.+?)(?:\n|$)",
        r"Exception[:\s]+(.+?)(?:\n|$)",
        r"Traceback \(most recent call last\):(.+?)(?:\n\n|$)",
        r"fatal[:\s]+(.+?)(?:\n|$)",
        r"FATAL[:\s]+(.+?)(?:\n|$)",
        r"panic[:\s]+(.+?)(?:\n|$)",
        r"timed out\s+(.+?)(?:\n|$)",
        r"timeout\s+(.+?)(?:\n|$)",
    ]

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_content = ""

        if log_file.exists():
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                self.log_content = f.read()

    def extract_with_logai(self) -> List[Dict[str, Any]]:
        """Extract errors using LogAI library."""
        if not LOGAI_AVAILABLE or not self.log_content:
            return []

        try:
            # Check if required LogAI components are available
            try:
                from logai.dataloader.data_loader import FileDataLoader
                from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig
                from logai.information_extraction.log_parser import (
                    LogParser,
                    LogParserConfig,
                )
                from logai.preprocess.openset_preprocessor import PreprocessorConfig
                from logai.preprocess.preprocessor import Preprocessor
            except (ImportError, AttributeError) as e:
                print(
                    f"LogAI components not fully available: {e}, using fallback",
                    file=sys.stderr,
                )
                return []

            # Write log content to a temporary file for LogAI processing
            temp_log = Path("/tmp/analysis.log")
            temp_log.write_text(self.log_content)

            # Configure LogAI components with default configs
            dataloader_config = OpenSetDataLoaderConfig()
            preprocessor_config = PreprocessorConfig()
            parser_config = LogParserConfig()

            # Load data
            dataloader = FileDataLoader(dataloader_config)
            logrecord = dataloader.load_data(str(temp_log))

            # Preprocess
            preprocessor = Preprocessor(preprocessor_config)
            logrecord = preprocessor.clean_log(logrecord)

            # Parse log patterns
            log_parser = LogParser(parser_config)
            parsed_result = log_parser.parse(logrecord)

            # Extract anomalies (errors)
            errors = []
            if hasattr(parsed_result, "body") and parsed_result.body is not None:
                for idx, log_line in enumerate(parsed_result.body.get("logline", [])):
                    log_lower = str(log_line).lower()
                    if any(
                        keyword in log_lower
                        for keyword in [
                            "error",
                            "failed",
                            "exception",
                            "fatal",
                            "panic",
                        ]
                    ):
                        errors.append(
                            {
                                "line_number": idx + 1,
                                "message": str(log_line).strip(),
                                "source": "logai",
                            }
                        )

            return errors[:10]  # Return top 10 errors

        except Exception as e:
            # Log to stderr but don't let it pollute the main output
            print(
                f"LogAI extraction failed: {e}, falling back to regex extraction",
                file=sys.stderr,
            )
            return []

    def extract_with_fallback(self) -> List[Dict[str, Any]]:
        """Fallback error extraction using regex patterns."""
        errors = []
        lines = self.log_content.split("\n")

        for pattern in self.ERROR_PATTERNS:
            matches = re.finditer(
                pattern, self.log_content, re.MULTILINE | re.IGNORECASE
            )
            for match in matches:
                # Find line number
                match_pos = match.start()
                line_num = self.log_content[:match_pos].count("\n") + 1

                # Extract error message
                error_msg = match.group(1) if match.groups() else match.group(0)
                error_msg = error_msg.strip()

                # Get context (surrounding lines)
                context_start = max(0, line_num - 2)
                context_end = min(len(lines), line_num + 3)
                context = "\n".join(lines[context_start:context_end])

                if error_msg and len(error_msg) > 10:  # Filter out very short matches
                    error_dict = {
                        "line_number": line_num,
                        "message": error_msg[:500],  # Limit message length
                        "context": context[:1000],  # Limit context length
                        "source": "fallback",
                    }
                    errors.append(error_dict)

        # Deduplicate and sort by line number
        seen = set()
        unique_errors = []
        for error in sorted(errors, key=lambda x: x["line_number"]):
            # Simple deduplication based on first 100 chars
            key = error["message"][:100]
            if key not in seen:
                seen.add(key)
                unique_errors.append(error)

        return unique_errors[:10]  # Return top 10 errors

    def extract_errors(self) -> List[Dict[str, Any]]:
        """Extract errors using LogAI first, then fallback."""
        if not self.log_content:
            return []

        # Try LogAI first
        if LOGAI_AVAILABLE:
            errors = self.extract_with_logai()
            if errors:
                return errors

        # Fallback to regex-based extraction
        return self.extract_with_fallback()

    def get_summary(self) -> str:
        """Get a summary of extracted errors."""
        errors = self.extract_errors()

        if not errors:
            return "No specific errors detected in logs"

        summary_parts = []
        for i, error in enumerate(errors[:5], 1):  # Top 5 errors
            summary_parts.append(
                f"{i}. [Line {error['line_number']}] {error['message']}"
            )
            if "context" in error:
                summary_parts.append(f"   Context: {error['context'][:200]}...")

        return "\n".join(summary_parts)

    def get_primary_error(self) -> str:
        """Get the most relevant error message."""
        errors = self.extract_errors()

        if not errors:
            return "Unknown error occurred"

        # Return the first (most relevant) error
        primary = errors[0]
        message = primary["message"]

        # Add context if available
        if "context" in primary:
            message += f"\n\nContext:\n{primary['context']}"

        return message


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: extract_log_errors.py <log_file> [--json]", file=sys.stderr)
        sys.exit(1)

    log_file = Path(sys.argv[1])
    output_json = "--json" in sys.argv

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)

    extractor = LogErrorExtractor(log_file)

    if output_json:
        errors = extractor.extract_errors()
        print(
            json.dumps(
                {
                    "errors": errors,
                    "count": len(errors),
                    "primary_error": extractor.get_primary_error(),
                },
                indent=2,
            )
        )
    else:
        # Human-readable output
        summary = extractor.get_summary()
        print(summary)
        print("\n" + "=" * 80)
        print("\nPrimary Error:")
        print(extractor.get_primary_error())


if __name__ == "__main__":
    main()
