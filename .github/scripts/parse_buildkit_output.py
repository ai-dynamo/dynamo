#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse BuildKit output to extract detailed step-by-step metadata.
BuildKit provides rich information about each build step including timing,
cache status, sizes, and layer IDs.

Also parses sccache statistics from the build log output.
"""

import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List


def parse_sccache_from_log(log_content: str, debug: bool = False) -> Dict[str, Any]:
    """
    Parse sccache statistics from build log output.

    In BuildKit logs, lines are prefixed with step numbers like:
    #43 103.6 === sccache statistics AFTER Dynamo ===
    #43 103.6 Compile requests                    2097
    #43 103.6 Cache hits                          1670
    #43 103.6 Cache hits rate                   100.00 %
    """
    sccache_data = {}

    # Find sccache statistics section(s) - get the last one
    # The section ends at the next DONE/CACHED marker or end of content
    sections = re.findall(
        r"=== sccache statistics AFTER ([^=]+) ===(.*?)(?=#\d+\s+DONE|#\d+\s+CACHED|#\d+\s+\[|$)",
        log_content,
        re.DOTALL,
    )

    if not sections:
        if debug:
            print("DEBUG: No sccache sections found in log", file=sys.stderr)
        return {}

    # Use the last sccache section (final stats)
    build_name, stats_block = sections[-1]
    sccache_data["build_name"] = build_name.strip()

    if debug:
        print(
            f"DEBUG: Found sccache section for '{build_name.strip()}'", file=sys.stderr
        )
        print(
            f"DEBUG: Stats block (first 1000 chars):\n{stats_block[:1000]}",
            file=sys.stderr,
        )

    # Parse each statistic line
    for line in stats_block.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Remove BuildKit prefix if present: "#43 103.6 " or "#43 "
        line = re.sub(r"^#\d+\s+[\d.]+\s+", "", line)
        line = re.sub(r"^#\d+\s+", "", line)
        line = line.strip()

        if not line:
            continue

        # Skip section headers like "Non-cacheable reasons:"
        if line.endswith(":"):
            if debug:
                print(f"DEBUG: Skipping section header: '{line}'", file=sys.stderr)
            continue

        # Match pattern: "Key Name    Value" or "Key Name    Value unit"
        # Examples:
        #   Compile requests                    2097
        #   Cache hits (C/C++)                   890
        #   Non-cacheable calls                  411
        #   Cache hits rate                   100.00 %
        #   Average cache read hit             0.050 s
        #
        # Note: there may be a space before the unit: "100.00 %" or "0.050 s"
        # Key can contain: letters, numbers, (), /, space, hyphen, plus
        match = re.match(
            r"^([A-Za-z][A-Za-z0-9() /+\-]+?)\s{2,}([\d.]+)\s*(%|s)?\s*$", line
        )
        if match:
            key_raw = match.group(1).strip()
            value_str = match.group(2)
            unit = match.group(3)

            # Convert key to snake_case
            key = key_raw.lower()
            key = re.sub(r"[+()]", "", key)  # Remove plus and parentheses
            key = re.sub(r"[/\-\s]+", "_", key)  # Replace /, -, spaces with _
            key = re.sub(r"_+", "_", key)  # Collapse multiple underscores
            key = key.strip("_")  # Remove leading/trailing underscores

            # Add unit suffix
            if unit == "%":
                key = key + "_percent"
            elif unit == "s":
                key = key + "_seconds"

            # Convert value to number
            try:
                if "." in value_str:
                    sccache_data[key] = float(value_str)
                else:
                    sccache_data[key] = int(value_str)
            except ValueError:
                sccache_data[key] = value_str

            if debug:
                print(f"DEBUG: Parsed: {key} = {sccache_data[key]}", file=sys.stderr)
        elif debug and line and any(c.isalpha() for c in line):
            # Print non-matching lines in debug mode for troubleshooting
            print(f"DEBUG: No match for line: '{line}'", file=sys.stderr)

    return sccache_data


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
                    message = substep_match.group(2)
                    # Extract size information
                    size_match = re.search(r"([\d.]+)\s*([KMGT]?i?B)", message)
                    if size_match:
                        size_bytes = self._parse_size(
                            size_match.group(1), size_match.group(2)
                        )
                        step_data[current_step_num]["size_transferred"] += size_bytes
                    continue

        # Convert to sorted list
        steps = [step_data[num] for num in sorted(step_data.keys(), key=int)]

        # Calculate aggregate statistics
        cached_steps = sum(1 for s in steps if s["cached"])
        total_steps = len(steps)
        cache_hit_rate = (cached_steps / total_steps * 100) if total_steps > 0 else 0.0
        total_size = sum(s["size_transferred"] for s in steps)

        # Create single stage for this Docker build (stage name will be updated from metadata)
        build_duration_sec = sum(s["duration_sec"] for s in steps if not s["cached"])
        stage_metrics = [
            {
                "stage_name": "unknown",  # Will be set from container metadata
                "total_steps": total_steps,
                "cached_steps": cached_steps,
                "built_steps": total_steps - cached_steps,
                "build_duration_sec": round(build_duration_sec, 2),
                "cache_hit_rate": round(cache_hit_rate, 2),
            }
        ]

        return {
            "container": {
                "total_steps": total_steps,
                "cached_steps": cached_steps,
                "built_steps": total_steps - cached_steps,
                "overall_cache_hit_rate": round(cache_hit_rate, 2),
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


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print(
            "Usage: parse_buildkit_output.py <output_json> <stage1_name:log_file> [stage2_name:log_file] ... [--metadata=<container_metadata_json>]",
            file=sys.stderr,
        )
        print(
            "Example: parse_buildkit_output.py output.json base:base.log runtime:framework.log --metadata=meta.json",
            file=sys.stderr,
        )
        sys.exit(1)

    output_json = sys.argv[1]

    # Parse arguments to find stage logs and metadata
    stage_logs = []  # List of (stage_name, log_file) tuples
    container_metadata_file = None
    debug_mode = "--debug" in sys.argv

    for arg in sys.argv[2:]:
        if arg == "--debug":
            continue
        elif arg.startswith("--metadata="):
            container_metadata_file = arg.split("=", 1)[1]
        elif ":" in arg:
            stage_name, log_file = arg.split(":", 1)
            stage_logs.append((stage_name, log_file))
        else:
            # Backwards compatibility: assume unnamed logs are base, runtime, etc.
            if not stage_logs:
                stage_logs.append(("base", arg))
            elif len(stage_logs) == 1:
                stage_logs.append(("runtime", arg))
            else:
                stage_logs.append((f"stage{len(stage_logs)}", arg))

    # Initialize combined structure
    combined_data = {"container": {}, "stages": [], "layers": []}

    total_steps = 0
    total_cached = 0
    total_size = 0
    sccache_data = {}

    # Parse each stage log
    for stage_name, log_file in stage_logs:
        try:
            with open(log_file, "r") as f:
                log_content = f.read()

            parser = BuildKitParser()
            stage_data = parser.parse_log(log_content)

            # Add stage with custom name
            if stage_data.get("stages"):
                stage_info = stage_data["stages"][0].copy()
                stage_info["stage_name"] = stage_name
                combined_data["stages"].append(stage_info)

            # Add layers with stage identifier
            for layer in stage_data.get("layers", []):
                layer["stage"] = stage_name
                combined_data["layers"].append(layer)

            # Accumulate metrics
            total_steps += stage_data["container"]["total_steps"]
            total_cached += stage_data["container"]["cached_steps"]
            total_size += stage_data["container"]["total_size_transferred_bytes"]

            # Extract sccache stats from the log (last one wins)
            log_sccache = parse_sccache_from_log(log_content, debug=debug_mode)
            if log_sccache:
                sccache_data = log_sccache
                print(
                    f"✅ Found sccache stats in {stage_name} log: {len(log_sccache)} metrics",
                    file=sys.stderr,
                )

            print(
                f"✅ Parsed {stage_name} stage: {stage_data['container']['total_steps']} steps",
                file=sys.stderr,
            )
        except FileNotFoundError:
            print(
                f"⚠️  Log file not found for {stage_name} stage: {log_file}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Warning: Could not parse {stage_name} log: {e}", file=sys.stderr)

    # Calculate rolled-up container metrics
    total_built = total_steps - total_cached
    overall_cache_hit_rate = (
        (total_cached / total_steps * 100) if total_steps > 0 else 0.0
    )

    combined_data["container"] = {
        "total_steps": total_steps,
        "cached_steps": total_cached,
        "built_steps": total_built,
        "overall_cache_hit_rate": round(overall_cache_hit_rate, 2),
        "total_size_transferred_bytes": total_size,
    }

    build_data = combined_data

    # Merge container metadata if provided
    if container_metadata_file:
        try:
            with open(container_metadata_file, "r") as f:
                container_metadata = json.load(f)
                # Merge into container section (overwrites BuildKit fields with action.yml values)
                # Note: We don't overwrite stage names since they're explicitly set from log file names
                build_data["container"].update(container_metadata)
        except Exception as e:
            print(f"Warning: Could not read container metadata: {e}", file=sys.stderr)

    # Add sccache statistics (parsed from build logs)
    if sccache_data:
        build_data["container"]["sccache"] = sccache_data
        print(
            f"✅ sccache metrics added to container: {len(sccache_data)} metrics",
            file=sys.stderr,
        )
    else:
        print("ℹ️  No sccache stats found in build logs", file=sys.stderr)

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
        f"   Cache Hit Rate: {container['overall_cache_hit_rate']:.1f}%",
        file=sys.stderr,
    )

    # Print sccache summary if available
    if "sccache" in container:
        sccache = container["sccache"]
        print("", file=sys.stderr)
        print("🔨 sccache Summary:", file=sys.stderr)
        if "compile_requests" in sccache:
            print(
                f"   Compile Requests: {sccache['compile_requests']}", file=sys.stderr
            )
        if "cache_hits" in sccache:
            print(f"   Cache Hits: {sccache['cache_hits']}", file=sys.stderr)
        if "cache_misses" in sccache:
            print(f"   Cache Misses: {sccache['cache_misses']}", file=sys.stderr)
        if "cache_hits_rate_percent" in sccache:
            print(
                f"   Cache Hit Rate: {sccache['cache_hits_rate_percent']:.2f}%",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
