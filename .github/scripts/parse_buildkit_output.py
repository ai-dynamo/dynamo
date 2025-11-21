#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse BuildKit output to extract detailed step-by-step metadata.
BuildKit provides rich information about each build step including timing,
cache status, sizes, and layer IDs.
"""

import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class BuildKitParser:
    """Parser for BuildKit output logs"""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.current_step = None
        self.step_counter = 0

    def parse_log(self, log_content: str) -> Dict[str, Any]:
        """
        Parse BuildKit log output and extract step metadata.

        BuildKit output format (with --progress=plain):
        #1 [internal] load build definition from Dockerfile
        #1 transferring dockerfile: 2.34kB done
        #1 DONE 0.1s

        #2 [internal] load metadata for nvcr.io/nvidia/cuda:12.8...
        #2 DONE 2.3s

        #3 [1/5] FROM nvcr.io/nvidia/cuda:12.8...
        #3 resolve nvcr.io/nvidia/cuda:12.8... done
        #3 CACHED

        #4 [2/5] RUN apt-get update && apt-get install...
        #4 0.234 Reading package lists...
        #4 DONE 45.2s
        """
        lines = log_content.split("\n")
        step_data = {}
        current_step_num = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match step headers: #N [...]
            step_match = re.match(r"^#(\d+)\s+\[(.*?)\](.*)$", line)
            if step_match:
                step_num = step_match.group(1)
                step_name = step_match.group(2).strip()
                step_command = step_match.group(3).strip()

                if step_num not in step_data:
                    step_data[step_num] = {
                        "step_number": int(step_num),
                        "step_name": step_name,
                        "command": step_command,
                        "status": "unknown",
                        "cached": False,
                        "duration_sec": 0.0,
                        "size_transferred": 0,
                        "logs": [],
                        "substeps": [],
                    }
                current_step_num = step_num
                continue

            # Match step status lines: #N DONE 1.2s, #N CACHED, #N ERROR
            if current_step_num:
                # DONE with timing
                done_match = re.match(
                    rf"^#{current_step_num}\s+DONE\s+([\d.]+)s?", line
                )
                if done_match:
                    step_data[current_step_num]["status"] = "done"
                    step_data[current_step_num]["duration_sec"] = float(
                        done_match.group(1)
                    )
                    continue

                # CACHED
                if re.match(rf"^#{current_step_num}\s+CACHED", line):
                    step_data[current_step_num]["status"] = "cached"
                    step_data[current_step_num]["cached"] = True
                    continue

                # ERROR
                if re.match(rf"^#{current_step_num}\s+ERROR", line):
                    step_data[current_step_num]["status"] = "error"
                    continue

                # Substep information (timing and progress)
                substep_match = re.match(
                    rf"^#{current_step_num}\s+([\d.]+)\s+(.*)", line
                )
                if substep_match:
                    timestamp = substep_match.group(1)
                    message = substep_match.group(2)
                    step_data[current_step_num]["substeps"].append(
                        {"timestamp": float(timestamp), "message": message}
                    )

                    # Extract size information
                    size_match = re.search(r"([\d.]+)\s*([KMGT]?i?B)", message)
                    if size_match:
                        size_bytes = self._parse_size(
                            size_match.group(1), size_match.group(2)
                        )
                        step_data[current_step_num]["size_transferred"] += size_bytes
                    continue

                # Other step-related information
                if re.match(rf"^#{current_step_num}\s+", line):
                    step_data[current_step_num]["logs"].append(
                        line.replace(f"#{current_step_num} ", "")
                    )

        # Convert to sorted list
        steps = [step_data[num] for num in sorted(step_data.keys(), key=int)]

        # Calculate aggregate statistics
        total_duration = sum(s["duration_sec"] for s in steps)
        cached_steps = sum(1 for s in steps if s["cached"])
        total_steps = len(steps)
        cache_hit_rate = (
            (cached_steps / total_steps * 100) if total_steps > 0 else 0.0
        )
        total_size = sum(s["size_transferred"] for s in steps)

        return {
            "steps": steps,
            "summary": {
                "total_steps": total_steps,
                "cached_steps": cached_steps,
                "built_steps": total_steps - cached_steps,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "total_duration_sec": round(total_duration, 2),
                "total_size_transferred_bytes": total_size,
            },
            "parsed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_size(self, value: str, unit: str) -> int:
        """Convert size string to bytes"""
        try:
            val = float(value)
        except ValueError:
            return 0

        # Normalize unit
        unit = unit.upper().replace("I", "")  # Remove 'i' from KiB, MiB, etc.

        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
        }

        return int(val * multipliers.get(unit, 1))

    def generate_report(self, build_data: Dict[str, Any]) -> str:
        """Generate a human-readable report from parsed build data"""
        report = []
        report.append("=" * 80)
        report.append("📊 BUILDKIT DETAILED BUILD REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary section
        summary = build_data["summary"]
        report.append("📈 BUILD SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Steps:        {summary['total_steps']}")
        report.append(f"Cached Steps:       {summary['cached_steps']}")
        report.append(f"Built Steps:        {summary['built_steps']}")
        report.append(f"Cache Hit Rate:     {summary['cache_hit_rate_percent']:.1f}%")
        report.append(f"Total Duration:     {summary['total_duration_sec']:.2f}s")

        total_size = summary["total_size_transferred_bytes"]
        if total_size > 0:
            report.append(
                f"Data Transferred:   {self._format_size(total_size)}"
            )
        report.append("")

        # Steps section
        report.append("🔨 DETAILED STEPS")
        report.append("-" * 80)
        report.append("")

        for step in build_data["steps"]:
            step_num = step["step_number"]
            status_icon = self._get_status_icon(step)
            cached_indicator = " [CACHED]" if step["cached"] else ""

            report.append(
                f"{status_icon} Step #{step_num}: {step['step_name']}{cached_indicator}"
            )

            if step["command"]:
                # Truncate very long commands
                cmd = step["command"]
                if len(cmd) > 100:
                    cmd = cmd[:97] + "..."
                report.append(f"   Command: {cmd}")

            report.append(f"   Status:   {step['status'].upper()}")

            if step["duration_sec"] > 0:
                report.append(f"   Duration: {step['duration_sec']:.2f}s")

            if step["size_transferred"] > 0:
                report.append(
                    f"   Size:     {self._format_size(step['size_transferred'])}"
                )

            # Show interesting substeps
            if step["substeps"]:
                interesting = [
                    s
                    for s in step["substeps"]
                    if any(
                        keyword in s["message"].lower()
                        for keyword in ["done", "complete", "extracting", "pulling"]
                    )
                ]
                if interesting and len(interesting) <= 5:
                    for substep in interesting[:3]:  # Show max 3
                        msg = substep["message"]
                        if len(msg) > 70:
                            msg = msg[:67] + "..."
                        report.append(f"   └─ {msg}")

            report.append("")

        # Top 5 slowest steps
        slowest = sorted(
            build_data["steps"], key=lambda s: s["duration_sec"], reverse=True
        )[:5]
        slowest = [s for s in slowest if s["duration_sec"] > 0]

        if slowest:
            report.append("⏱️  TOP 5 SLOWEST STEPS")
            report.append("-" * 80)
            for i, step in enumerate(slowest, 1):
                report.append(
                    f"{i}. Step #{step['step_number']}: {step['step_name']} "
                    f"({step['duration_sec']:.2f}s)"
                )
            report.append("")

        # Top 5 largest transfers
        largest = sorted(
            build_data["steps"], key=lambda s: s["size_transferred"], reverse=True
        )[:5]
        largest = [s for s in largest if s["size_transferred"] > 0]

        if largest:
            report.append("📦 TOP 5 LARGEST DATA TRANSFERS")
            report.append("-" * 80)
            for i, step in enumerate(largest, 1):
                report.append(
                    f"{i}. Step #{step['step_number']}: {step['step_name']} "
                    f"({self._format_size(step['size_transferred'])})"
                )
            report.append("")

        report.append("=" * 80)
        report.append(f"Report generated at: {build_data['parsed_at']}")
        report.append("=" * 80)

        return "\n".join(report)

    def _get_status_icon(self, step: Dict[str, Any]) -> str:
        """Get emoji icon for step status"""
        if step["cached"]:
            return "⚡"
        elif step["status"] == "done":
            return "✅"
        elif step["status"] == "error":
            return "❌"
        else:
            return "⏳"

    def _format_size(self, bytes_size: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(
            "Usage: parse_buildkit_output.py <build_log_file> [output_json] [output_report]",
            file=sys.stderr,
        )
        sys.exit(1)

    log_file = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None
    output_report = sys.argv[3] if len(sys.argv) > 3 else None

    # Read build log
    try:
        with open(log_file, "r") as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse BuildKit output
    parser = BuildKitParser()
    build_data = parser.parse_log(log_content)

    # Output JSON
    if output_json:
        with open(output_json, "w") as f:
            json.dump(build_data, f, indent=2)
        print(f"✅ Build data written to: {output_json}", file=sys.stderr)

    # Output report
    report_text = parser.generate_report(build_data)
    if output_report:
        with open(output_report, "w") as f:
            f.write(report_text)
        print(f"✅ Build report written to: {output_report}", file=sys.stderr)
    else:
        # Print to stdout if no output file specified
        print(report_text)

    # Also print summary to stderr for immediate feedback
    print("", file=sys.stderr)
    print("📊 Build Summary:", file=sys.stderr)
    print(
        f"   Steps: {build_data['summary']['total_steps']} total, "
        f"{build_data['summary']['cached_steps']} cached, "
        f"{build_data['summary']['built_steps']} built",
        file=sys.stderr,
    )
    print(
        f"   Cache Hit Rate: {build_data['summary']['cache_hit_rate_percent']:.1f}%",
        file=sys.stderr,
    )
    print(
        f"   Total Duration: {build_data['summary']['total_duration_sec']:.2f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

