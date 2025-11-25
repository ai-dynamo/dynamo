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

        # Enrich steps with stage information
        steps = self._enrich_steps_with_stage_info(steps)

        # Calculate aggregate statistics
        total_duration = sum(s["duration_sec"] for s in steps)
        cached_steps = sum(1 for s in steps if s["cached"])
        total_steps = len(steps)
        cache_hit_rate = (
            (cached_steps / total_steps * 100) if total_steps > 0 else 0.0
        )
        total_size = sum(s["size_transferred"] for s in steps)

        # Calculate per-stage metrics
        stage_metrics = self._calculate_stage_metrics(steps)

        return {
            "container": {
                "total_steps": total_steps,
                "cached_steps": cached_steps,
                "built_steps": total_steps - cached_steps,
                "overall_cache_hit_rate": round(cache_hit_rate, 2),
                "total_duration_sec": round(total_duration, 2),
                "total_size_transferred_bytes": total_size,
            },
            "stages": stage_metrics,
            "layers": steps,
            "metadata": {
                "parsed_at": datetime.now(timezone.utc).isoformat(),
                "parser_version": "1.0",
            },
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

    def _extract_stage_name(self, step_name: str) -> str:
        """Extract stage name from BuildKit step name"""
        # Match patterns like "[base 1/5]" or "[vllm-builder 2/3]"
        match = re.match(r"^\[?([a-zA-Z0-9_-]+)\s+\d+/\d+", step_name)
        if match:
            return match.group(1)
        
        # Handle internal steps
        if "internal" in step_name.lower():
            return "internal"
        
        # Handle unnamed stages (stage-0, stage-1, etc)
        stage_match = re.search(r"stage-(\d+)", step_name)
        if stage_match:
            return f"stage-{stage_match.group(1)}"
        
        return "unknown"

    def _calculate_stage_metrics(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate cache hit rate and timing per stage"""
        stage_data = {}
        stage_order = {}
        order_counter = 0
        
        for step in steps:
            stage = self._extract_stage_name(step["step_name"])
            
            if stage not in stage_data:
                order_counter += 1
                stage_order[stage] = order_counter
                stage_data[stage] = {
                    "stage_name": stage,
                    "stage_order": order_counter,
                    "total_steps": 0,
                    "cached_steps": 0,
                    "built_steps": 0,
                    "total_duration_sec": 0.0,
                    "build_duration_sec": 0.0,
                    "cache_hit_rate": 0.0,
                }
            
            stage_data[stage]["total_steps"] += 1
            stage_data[stage]["total_duration_sec"] += step["duration_sec"]
            
            if step["cached"]:
                stage_data[stage]["cached_steps"] += 1
            else:
                stage_data[stage]["built_steps"] += 1
                stage_data[stage]["build_duration_sec"] += step["duration_sec"]
        
        # Calculate percentages and round
        for stage in stage_data.values():
            if stage["total_steps"] > 0:
                stage["cache_hit_rate"] = round(
                    (stage["cached_steps"] / stage["total_steps"]) * 100, 2
                )
            stage["total_duration_sec"] = round(stage["total_duration_sec"], 2)
            stage["build_duration_sec"] = round(stage["build_duration_sec"], 2)
        
        # Convert to sorted list by stage order
        return sorted(stage_data.values(), key=lambda x: x["stage_order"])

    def _enrich_steps_with_stage_info(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add stage information to each step"""
        stage_order = {}
        order_counter = 0
        
        for step in steps:
            stage = self._extract_stage_name(step["step_name"])
            
            if stage not in stage_order:
                order_counter += 1
                stage_order[stage] = order_counter
            
            # Add stage info to step
            step["stage_name"] = stage
            step["stage_order"] = stage_order[stage]
        
        return steps


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print(
            "Usage: parse_buildkit_output.py <build_log_file> <output_json> [container_metadata_json]",
            file=sys.stderr,
        )
        sys.exit(1)

    log_file = sys.argv[1]
    output_json = sys.argv[2]
    container_metadata_file = sys.argv[3] if len(sys.argv) > 3 else None

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
    
    # Merge container metadata if provided
    if container_metadata_file:
        try:
            with open(container_metadata_file, "r") as f:
                container_metadata = json.load(f)
                # Merge into container section (overwrites BuildKit fields with action.yml values)
                build_data["container"].update(container_metadata)
        except Exception as e:
            print(f"Warning: Could not read container metadata: {e}", file=sys.stderr)

    # Output JSON
    try:
        with open(output_json, "w") as f:
            json.dump(build_data, f, indent=2)
        print(f"✅ Build data written to: {output_json}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary to stderr for immediate feedback
    container = build_data["container"]
    print("", file=sys.stderr)
    print("📊 Build Summary:", file=sys.stderr)
    print(
        f"   Steps: {container['total_steps']} total, "
        f"{container['cached_steps']} cached, "
        f"{container['built_steps']} built",
        file=sys.stderr,
    )
    print(
        f"   Overall Cache Hit Rate: {container['overall_cache_hit_rate']:.1f}%",
        file=sys.stderr,
    )
    print(
        f"   Total Duration: {container['total_duration_sec']:.2f}s",
        file=sys.stderr,
    )
    
    # Print per-stage summary
    if build_data.get("stages"):
        print("", file=sys.stderr)
        print("📦 Per-Stage Breakdown:", file=sys.stderr)
        for stage in build_data["stages"]:
            print(
                f"   [{stage['stage_name']}] "
                f"{stage['cached_steps']}/{stage['total_steps']} cached "
                f"({stage['cache_hit_rate']:.1f}%), "
                f"{stage['total_duration_sec']:.1f}s",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()


