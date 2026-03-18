#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def parse_sccache_json_from_log(
    log_content: str, debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Parse multiple sccache JSON statistics blocks from build log output.

    Returns a list of sccache stats dictionaries, one for each build section.
    """
    sccache_sections = []

    # Find all JSON blocks between markers
    pattern = r"=== SCCACHE_JSON_BEGIN ===\s*\n" r"(.*?)" r"=== SCCACHE_JSON_END ==="

    matches = re.findall(pattern, log_content, re.DOTALL)

    if not matches:
        if debug:
            print("DEBUG: No sccache JSON sections found in log", file=sys.stderr)
        return []

    for i, json_block in enumerate(matches):
        try:
            # Remove BuildKit prefixes from each line: "#43 103.6 " or "#43 "
            cleaned_lines = []
            for line in json_block.split("\n"):
                cleaned_line = re.sub(r"^#\d+\s+[\d.]+\s+", "", line)
                cleaned_line = re.sub(r"^#\d+\s+", "", cleaned_line)
                cleaned_lines.append(cleaned_line)

            cleaned_json = "\n".join(cleaned_lines)

            if debug:
                print(f"DEBUG: Parsing JSON block {i+1}", file=sys.stderr)

            # Parse JSON
            section_data = json.loads(cleaned_json)

            # Extract key metrics into flat structure
            section_name = section_data.get("section", f"section_{i+1}")
            timestamp = section_data.get("timestamp", "")
            native_stats = section_data.get("sccache_stats", {})

            stats_dict = {
                "section_name": section_name,
                "timestamp": timestamp,
            }

            # Extract compile statistics from native JSON
            if "stats" in native_stats:
                stats = native_stats["stats"]
                stats_dict["compile_requests"] = stats.get("compile_requests", 0)
                stats_dict["requests_executed"] = stats.get("requests_executed", 0)

                # Handle cache_hits (sum all language counts)
                cache_hits = stats.get("cache_hits", {})
                if isinstance(cache_hits, dict) and "counts" in cache_hits:
                    stats_dict["cache_hits"] = sum(cache_hits["counts"].values())
                else:
                    stats_dict["cache_hits"] = 0

                # Handle cache_misses
                cache_misses = stats.get("cache_misses", {})
                if isinstance(cache_misses, dict) and "counts" in cache_misses:
                    stats_dict["cache_misses"] = sum(cache_misses["counts"].values())
                else:
                    stats_dict["cache_misses"] = 0

                # Calculate hit rate
                total = stats_dict["cache_hits"] + stats_dict["cache_misses"]
                if total > 0:
                    stats_dict["cache_hits_rate_percent"] = (
                        stats_dict["cache_hits"] / total * 100
                    )
                else:
                    stats_dict["cache_hits_rate_percent"] = 0.0

                # Additional metrics
                stats_dict["cache_timeouts"] = stats.get("cache_timeouts", 0)
                stats_dict["cache_read_errors"] = stats.get("cache_read_errors", 0)
                stats_dict["cache_write_errors"] = stats.get("cache_write_errors", 0)
                stats_dict["cache_writes"] = stats.get("cache_writes", 0)
                stats_dict["compile_fails"] = stats.get("compile_fails", 0)
                stats_dict["non_cacheable_compilations"] = stats.get(
                    "non_cacheable_compilations", 0
                )

                # Duration metrics (convert secs + nanos to total seconds)
                for dur_key in [
                    "cache_write_duration",
                    "cache_read_hit_duration",
                    "compiler_write_duration",
                ]:
                    dur = stats.get(dur_key, {})
                    if dur:
                        stats_dict[f"{dur_key}_seconds"] = (
                            dur.get("secs", 0) + dur.get("nanos", 0) / 1_000_000_000
                        )

            # Metadata
            stats_dict["cache_location"] = native_stats.get("cache_location", "")
            stats_dict["sccache_version"] = native_stats.get("version", "")
            stats_dict["_raw_stats"] = native_stats

            sccache_sections.append(stats_dict)

            if debug:
                print(
                    f"DEBUG: Parsed section '{section_name}': {len(stats_dict)} metrics",
                    file=sys.stderr,
                )

        except json.JSONDecodeError as e:
            if debug:
                print(f"DEBUG: JSON parse error in block {i+1}: {e}", file=sys.stderr)
        except Exception as e:
            if debug:
                print(f"DEBUG: Error parsing block {i+1}: {e}", file=sys.stderr)

    return sccache_sections


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
    all_sccache_sections = []

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

            # Extract all sccache JSON sections from the log
            log_sccache_sections = parse_sccache_json_from_log(
                log_content, debug=debug_mode
            )
            if log_sccache_sections:
                # Associate sections with this stage
                for section_stats in log_sccache_sections:
                    section_stats["stage"] = stage_name
                    all_sccache_sections.append(section_stats)

                print(
                    f"✅ Found {len(log_sccache_sections)} sccache section(s) in {stage_name} log",
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

    # Add all sccache statistics
    if all_sccache_sections:
        build_data["container"]["sccache_sections"] = all_sccache_sections
        build_data["container"]["sccache_available"] = True

        # Calculate aggregate stats
        total_compile_requests = sum(
            s.get("compile_requests", 0) for s in all_sccache_sections
        )
        total_cache_hits = sum(s.get("cache_hits", 0) for s in all_sccache_sections)
        total_cache_misses = sum(s.get("cache_misses", 0) for s in all_sccache_sections)

        aggregate_hit_rate = 0.0
        if total_cache_hits + total_cache_misses > 0:
            aggregate_hit_rate = (
                total_cache_hits / (total_cache_hits + total_cache_misses)
            ) * 100

        build_data["container"]["sccache_aggregate"] = {
            "total_sections": len(all_sccache_sections),
            "section_names": [s["section_name"] for s in all_sccache_sections],
            "total_compile_requests": total_compile_requests,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "aggregate_hit_rate_percent": round(aggregate_hit_rate, 2),
        }

        print(
            f"✅ sccache metrics added: {len(all_sccache_sections)} sections, "
            f"{total_compile_requests} total requests",
            file=sys.stderr,
        )
    else:
        build_data["container"]["sccache_available"] = False
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

    # Print sccache summary
    print("", file=sys.stderr)
    print("🔨 sccache Summary:", file=sys.stderr)
    print(
        f"   Available: {container.get('sccache_available', False)}",
        file=sys.stderr,
    )

    if "sccache_aggregate" in container:
        agg = container["sccache_aggregate"]
        print(
            f"   Sections: {agg['total_sections']} ({', '.join(agg['section_names'])})",
            file=sys.stderr,
        )
        print(
            f"   Total Compile Requests: {agg['total_compile_requests']}",
            file=sys.stderr,
        )
        print(f"   Total Cache Hits: {agg['total_cache_hits']}", file=sys.stderr)
        print(f"   Total Cache Misses: {agg['total_cache_misses']}", file=sys.stderr)
        print(
            f"   Aggregate Hit Rate: {agg['aggregate_hit_rate_percent']:.2f}%",
            file=sys.stderr,
        )

    if "sccache_sections" in container:
        print("", file=sys.stderr)
        print("   Per-section breakdown:", file=sys.stderr)
        for section in container["sccache_sections"]:
            section_name = section.get("section_name", "unknown")
            requests = section.get("compile_requests", 0)
            hits = section.get("cache_hits", 0)
            hit_rate = section.get("cache_hits_rate_percent", 0.0)
            print(
                f"     • {section_name}: {requests} requests, {hits} hits ({hit_rate:.1f}%)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
