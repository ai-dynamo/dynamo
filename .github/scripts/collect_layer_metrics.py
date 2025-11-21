#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Script to collect layer-level metrics from a Docker image.
This extracts information about each layer including size, creation time, and commands.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def run_command(cmd: List[str]) -> str:
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return ""
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {' '.join(cmd)}", file=sys.stderr)
        return ""


def get_image_layers(image_tag: str) -> List[Dict[str, Any]]:
    """
    Get layer information from docker history.
    Returns a list of layer dictionaries with size, created time, and command.
    """
    layers = []
    
    # Get detailed layer information using docker history
    cmd = ["docker", "history", "--no-trunc", "--format", "{{json .}}", image_tag]
    output = run_command(cmd)
    
    if not output:
        print(f"Warning: Could not get layer history for {image_tag}", file=sys.stderr)
        return layers
    
    # Parse each line as JSON
    layer_index = 0
    for line in output.split("\n"):
        if not line.strip():
            continue
        
        try:
            layer_data = json.loads(line)
            
            # Extract relevant information
            layer = {
                "layer_index": layer_index,
                "layer_id": layer_data.get("ID", ""),
                "created_by": layer_data.get("CreatedBy", ""),
                "size_human": layer_data.get("Size", "0B"),
                "size_bytes": parse_size_to_bytes(layer_data.get("Size", "0B")),
                "created_at": layer_data.get("CreatedAt", ""),
                "comment": layer_data.get("Comment", ""),
            }
            
            layers.append(layer)
            layer_index += 1
            
        except json.JSONDecodeError as e:
            print(f"Error parsing layer JSON: {e}", file=sys.stderr)
            print(f"Line: {line}", file=sys.stderr)
            continue
    
    return layers


def parse_size_to_bytes(size_str: str) -> int:
    """
    Convert Docker size string (e.g., "1.2GB", "500MB", "10kB") to bytes.
    """
    size_str = size_str.strip().upper()
    
    if size_str == "0B" or size_str == "0":
        return 0
    
    # Extract number and unit
    import re
    match = re.match(r"^([\d.]+)\s*([KMGT]?B)$", size_str)
    if not match:
        print(f"Warning: Could not parse size string: {size_str}", file=sys.stderr)
        return 0
    
    value = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    
    return int(value * multipliers.get(unit, 1))


def get_layer_cache_info(build_log: Optional[str] = None) -> Dict[str, str]:
    """
    Parse build log to determine which layers were cached.
    Returns a dict mapping layer index to cache status.
    """
    cache_info = {}
    
    if not build_log:
        return cache_info
    
    # Parse build log for CACHED indicators
    # This is best-effort as docker build output format can vary
    layer_idx = 0
    for line in build_log.split("\n"):
        if "CACHED" in line:
            cache_info[str(layer_idx)] = "cached"
        elif line.strip().startswith("Step "):
            layer_idx += 1
    
    return cache_info


def collect_layer_metrics(
    image_tag: str, framework: str, platform: str, build_log: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect comprehensive layer metrics for a Docker image.
    
    Args:
        image_tag: Docker image tag
        framework: Framework name (vllm, sglang, trtllm)
        platform: Platform architecture (amd64, arm64)
        build_log: Optional build log for cache information
        
    Returns:
        Dictionary containing layer metrics
    """
    print(f"Collecting layer metrics for {image_tag}...", file=sys.stderr)
    
    # Get layer information
    layers = get_image_layers(image_tag)
    
    if not layers:
        print(f"Warning: No layers found for {image_tag}", file=sys.stderr)
    
    # Get cache information if build log is provided
    cache_info = get_layer_cache_info(build_log) if build_log else {}
    
    # Add cache status to layers
    for layer in layers:
        layer_idx_str = str(layer["layer_index"])
        layer["cache_status"] = cache_info.get(layer_idx_str, "unknown")
    
    # Calculate total sizes
    total_size_bytes = sum(layer["size_bytes"] for layer in layers)
    
    metrics = {
        "image_tag": image_tag,
        "framework": framework,
        "platform": platform,
        "total_layers": len(layers),
        "total_size_bytes": total_size_bytes,
        "layers": layers,
        "collection_time": datetime.now(timezone.utc).isoformat(),
    }
    
    print(
        f"Collected metrics for {len(layers)} layers (total size: {total_size_bytes} bytes)",
        file=sys.stderr,
    )
    
    return metrics


def main():
    """Main entry point"""
    if len(sys.argv) < 4:
        print(
            "Usage: collect_layer_metrics.py <image_tag> <framework> <platform> [output_file]",
            file=sys.stderr,
        )
        sys.exit(1)
    
    image_tag = sys.argv[1]
    framework = sys.argv[2]
    platform = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Collect metrics
    metrics = collect_layer_metrics(image_tag, framework, platform)
    
    # Output results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Layer metrics written to {output_file}", file=sys.stderr)
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

