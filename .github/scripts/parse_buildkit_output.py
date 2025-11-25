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


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print(
            "Usage: parse_buildkit_output.py <build_log_file> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    log_file = sys.argv[1]
    output_json = sys.argv[2]

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
    try:
        with open(output_json, "w") as f:
            json.dump(build_data, f, indent=2)
        print(f"✅ Build data written to: {output_json}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary to stderr for immediate feedback
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

