#!/usr/bin/env python3
"""
Analyze worker selection from dynamo frontend logs.

Supports two log formats:
1. KV Router format: "Selected worker:" entries with detailed metrics
2. Round Robin format: "round robin router selected" entries

Provides statistics on job allocation across workers.
"""

import argparse
import re
import sys
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LogFormat(Enum):
    KV_ROUTER = auto()
    ROUND_ROBIN = auto()
    UNKNOWN = auto()


def detect_log_format(log_path: Path) -> LogFormat:
    """
    Detect the log format by scanning for characteristic patterns.
    
    Returns LogFormat.KV_ROUTER if "Selected worker:" lines are found,
    LogFormat.ROUND_ROBIN if "round robin router selected" lines are found,
    or LogFormat.UNKNOWN if neither pattern is found.
    """
    kv_router_count = 0
    round_robin_count = 0
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        # Read and check only the last 1000 lines using a plain list
        lines = f.readlines()
        check_lines = lines[-1000:] if len(lines) > 1000 else lines
        for line in check_lines:
            if "Selected worker:" in line:
                kv_router_count += 1
            if "round robin router selected" in line:
                round_robin_count += 1
    
    return LogFormat.KV_ROUTER if kv_router_count > 0 and kv_router_count >= round_robin_count else LogFormat.ROUND_ROBIN


def parse_selected_worker_line(line: str) -> Optional[Dict]:
    """
    Parse a KV router log line containing "Selected worker:" and extract fields.
    
    Returns a dict with worker_id, dp_rank, logit, cached_blocks, tree_size, total_blocks
    or None if the line doesn't match.
    """
    # Remove ANSI escape codes for cleaner parsing
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[([0-9;]*)m')
    clean_line = ansi_escape.sub('', line)
    
    if "Selected worker:" not in clean_line:
        return None
    
    # Extract timestamp
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)', clean_line)
    timestamp = timestamp_match.group(1) if timestamp_match else None
    
    # Extract worker selection fields
    pattern = r'Selected worker:\s*worker_id=(\d+)\s+dp_rank=(\d+),\s*logit:\s*([\d.]+),\s*cached blocks:\s*(\d+),\s*tree size:\s*(\d+),\s*total blocks:\s*(\d+)'
    match = re.search(pattern, clean_line)
    
    if not match:
        return None
    
    return {
        'timestamp': timestamp,
        'worker_id': match.group(1),
        'dp_rank': int(match.group(2)),
        'logit': float(match.group(3)),
        'cached_blocks': int(match.group(4)),
        'tree_size': int(match.group(5)),
        'total_blocks': int(match.group(6)),
        'raw_line': line.rstrip('\n'),
    }


def parse_round_robin_line(line: str) -> Optional[Dict]:
    """
    Parse a round robin router log line and extract fields.
    
    Example line format:
    2026-01-13T22:12:13.101965Z ... round robin router selected 15112443643421252000 ... x_request_id="..."
    
    Returns a dict with worker_id, timestamp, request_id or None if the line doesn't match.
    """
    # Remove ANSI escape codes for cleaner parsing
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[([0-9;]*)m')
    clean_line = ansi_escape.sub('', line)
    
    if "round robin router selected" not in clean_line:
        return None
    
    # Extract timestamp
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)', clean_line)
    timestamp = timestamp_match.group(1) if timestamp_match else None
    
    # Extract worker ID (the number after "round robin router selected")
    worker_match = re.search(r'round robin router selected\s+(\d+)', clean_line)
    if not worker_match:
        return None
    
    worker_id = worker_match.group(1)
    
    # Extract request ID if present
    request_id_match = re.search(r'x_request_id="([^"]+)"', clean_line)
    request_id = request_id_match.group(1) if request_id_match else None
    
    return {
        'timestamp': timestamp,
        'worker_id': worker_id,
        'request_id': request_id,
        'raw_line': line.rstrip('\n'),
    }


def get_dedup_key_kv_router(line: str) -> str:
    """
    Extract a key from a Selected worker line for deduplication.
    
    Removes the timestamp portion so lines with identical worker selection
    data but different timestamps are considered duplicates.
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[([0-9;]*)m')
    clean_line = ansi_escape.sub('', line)
    
    # Extract just the "Selected worker:" portion and everything after
    match = re.search(r'(Selected worker:.*)$', clean_line)
    if match:
        return match.group(1).strip()
    return clean_line.strip()


def get_dedup_key_round_robin(line: str) -> str:
    """
    Extract a key from a round robin selection line for deduplication.
    
    Uses worker_id + request_id for deduplication.
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[([0-9;]*)m')
    clean_line = ansi_escape.sub('', line)
    
    # Extract worker ID
    worker_match = re.search(r'round robin router selected\s+(\d+)', clean_line)
    worker_id = worker_match.group(1) if worker_match else ""
    
    # Extract request ID
    request_id_match = re.search(r'x_request_id="([^"]+)"', clean_line)
    request_id = request_id_match.group(1) if request_id_match else ""
    
    return f"{worker_id}:{request_id}"


def extract_worker_id_from_formula_line(line: str) -> Optional[str]:
    """
    Extract worker_id from a "Formula for worker_id=X" log line.
    
    Returns the worker_id string or None if not found.
    """
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[([0-9;]*)m')
    clean_line = ansi_escape.sub('', line)
    
    match = re.search(r'Formula for worker_id=(\d+)', clean_line)
    if match:
        return match.group(1)
    return None


def analyze_kv_router_log(log_path: Path, dedup: bool = False) -> Tuple[List[Dict], List[str], set]:
    """
    Read a log file with KV router format and extract all Selected worker entries.
    
    Args:
        log_path: Path to the log file
        dedup: If True, deduplicate lines based on worker selection content
    
    Returns:
        - List of parsed worker selection records
        - List of raw lines containing "Selected worker:"
        - Set of all unique worker IDs observed in the log
    """
    records = []
    raw_lines = []
    seen_keys = set()
    all_worker_ids = set()
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Extract worker IDs from "Formula for worker_id=" lines
            if "Formula for worker_id=" in line:
                worker_id = extract_worker_id_from_formula_line(line)
                if worker_id:
                    all_worker_ids.add(worker_id)
            
            if "Selected worker:" in line:
                if dedup:
                    key = get_dedup_key_kv_router(line)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                
                raw_lines.append(line)
                parsed = parse_selected_worker_line(line)
                if parsed:
                    records.append(parsed)
                    all_worker_ids.add(parsed['worker_id'])
    
    return records, raw_lines, all_worker_ids


def analyze_round_robin_log(log_path: Path, dedup: bool = False) -> Tuple[List[Dict], List[str], set]:
    """
    Read a log file with round robin format and extract all selection entries.
    
    Args:
        log_path: Path to the log file
        dedup: If True, deduplicate lines based on worker_id + request_id
    
    Returns:
        - List of parsed worker selection records
        - List of raw lines containing round robin selections
        - Set of all unique worker IDs observed in the log
    """
    records = []
    raw_lines = []
    seen_keys = set()
    all_worker_ids = set()
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if "round robin router selected" in line:
                if dedup:
                    key = get_dedup_key_round_robin(line)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                
                raw_lines.append(line)
                parsed = parse_round_robin_line(line)
                if parsed:
                    records.append(parsed)
                    all_worker_ids.add(parsed['worker_id'])
    
    return records, raw_lines, all_worker_ids


def analyze_log_file(log_path: Path, dedup: bool = False, log_format: Optional[LogFormat] = None) -> Tuple[List[Dict], List[str], set, LogFormat]:
    """
    Read a log file and extract all worker selection entries.
    
    Automatically detects log format if not specified.
    
    Args:
        log_path: Path to the log file
        dedup: If True, deduplicate lines based on worker selection content
        log_format: Optional format override. If None, auto-detects.
    
    Returns:
        - List of parsed worker selection records
        - List of raw lines
        - Set of all unique worker IDs observed in the log
        - Detected LogFormat
    """
    if log_format is None:
        log_format = detect_log_format(log_path)
    
    if log_format == LogFormat.KV_ROUTER:
        records, raw_lines, all_worker_ids = analyze_kv_router_log(log_path, dedup)
    elif log_format == LogFormat.ROUND_ROBIN:
        records, raw_lines, all_worker_ids = analyze_round_robin_log(log_path, dedup)
    else:
        records, raw_lines, all_worker_ids = [], [], set()
    
    return records, raw_lines, all_worker_ids, log_format


def print_kv_router_summary(records: List[Dict], all_worker_ids: set) -> None:
    """Print summary statistics for KV router worker selection data."""
    if not records:
        print("No 'Selected worker:' entries found in log file.")
        if all_worker_ids:
            print(f"Total unique worker IDs observed: {len(all_worker_ids)}")
        return
    
    print("=" * 80)
    print("WORKER SELECTION SUMMARY (KV Router)")
    print("=" * 80)
    print(f"\nTotal selections: {len(records)}")
    
    # Group by worker_id
    by_worker = defaultdict(list)
    for r in records:
        by_worker[r['worker_id']].append(r)
    
    print(f"Unique workers selected: {len(by_worker)}")
    print(f"Total unique worker IDs observed: {len(all_worker_ids)}")
    
    # Job allocation summary
    print("\n" + "-" * 80)
    print("JOB ALLOCATION BY WORKER")
    print("-" * 80)
    print(f"{'Worker ID':<25} {'Count':>8} {'Pct':>8} {'Avg Logit':>12} {'Avg Cached':>12} {'Avg Tree':>12}")
    print("-" * 80)
    
    # Sort by count descending
    sorted_workers = sorted(by_worker.items(), key=lambda x: len(x[1]), reverse=True)
    
    for worker_id, worker_records in sorted_workers:
        count = len(worker_records)
        pct = (count / len(records)) * 100
        avg_logit = sum(r['logit'] for r in worker_records) / count
        avg_cached = sum(r['cached_blocks'] for r in worker_records) / count
        avg_tree = sum(r['tree_size'] for r in worker_records) / count
        
        print(f"{worker_id:<25} {count:>8} {pct:>7.2f}% {avg_logit:>12.3f} {avg_cached:>12.1f} {avg_tree:>12.1f}")
    
    # Overall statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)
    
    all_logits = [r['logit'] for r in records]
    all_cached = [r['cached_blocks'] for r in records]
    all_tree = [r['tree_size'] for r in records]
    all_total = [r['total_blocks'] for r in records]
    
    print(f"{'Metric':<20} {'Min':>12} {'Max':>12} {'Avg':>12}")
    print("-" * 60)
    print(f"{'Logit':<20} {min(all_logits):>12.3f} {max(all_logits):>12.3f} {sum(all_logits)/len(all_logits):>12.3f}")
    print(f"{'Cached Blocks':<20} {min(all_cached):>12} {max(all_cached):>12} {sum(all_cached)/len(all_cached):>12.1f}")
    print(f"{'Tree Size':<20} {min(all_tree):>12} {max(all_tree):>12} {sum(all_tree)/len(all_tree):>12.1f}")
    print(f"{'Total Blocks':<20} {min(all_total):>12} {max(all_total):>12} {sum(all_total)/len(all_total):>12.1f}")
    
    # DP rank distribution
    dp_ranks = defaultdict(int)
    for r in records:
        dp_ranks[r['dp_rank']] += 1
    
    if len(dp_ranks) > 1:
        print("\n" + "-" * 80)
        print("DP RANK DISTRIBUTION")
        print("-" * 80)
        for rank in sorted(dp_ranks.keys()):
            count = dp_ranks[rank]
            pct = (count / len(records)) * 100
            print(f"dp_rank={rank}: {count} ({pct:.2f}%)")
    
    print("\n" + "=" * 80)


def print_round_robin_summary(records: List[Dict], all_worker_ids: set) -> None:
    """Print summary statistics for round robin worker selection data."""
    if not records:
        print("No 'round robin router selected' entries found in log file.")
        if all_worker_ids:
            print(f"Total unique worker IDs observed: {len(all_worker_ids)}")
        return
    
    print("=" * 80)
    print("WORKER SELECTION SUMMARY (Round Robin)")
    print("=" * 80)
    print(f"\nTotal selections: {len(records)}")
    
    # Group by worker_id
    by_worker = defaultdict(list)
    for r in records:
        by_worker[r['worker_id']].append(r)
    
    print(f"Unique workers selected: {len(by_worker)}")
    print(f"Total unique worker IDs observed: {len(all_worker_ids)}")
    
    # Calculate expected distribution for round robin
    expected_pct = 100.0 / len(by_worker) if by_worker else 0
    expected_count = len(records) / len(by_worker) if by_worker else 0
    
    # Job allocation summary
    print("\n" + "-" * 80)
    print("JOB ALLOCATION BY WORKER")
    print("-" * 80)
    print(f"{'Worker ID':<25} {'Count':>10} {'Pct':>10} {'Deviation':>12}")
    print("-" * 80)
    
    # Sort by count descending
    sorted_workers = sorted(by_worker.items(), key=lambda x: len(x[1]), reverse=True)
    
    for worker_id, worker_records in sorted_workers:
        count = len(worker_records)
        pct = (count / len(records)) * 100
        deviation = pct - expected_pct
        deviation_str = f"{deviation:+.2f}%"
        
        print(f"{worker_id:<25} {count:>10} {pct:>9.2f}% {deviation_str:>12}")
    
    # Distribution statistics
    print("\n" + "-" * 80)
    print("DISTRIBUTION STATISTICS")
    print("-" * 80)
    
    counts = [len(w) for w in by_worker.values()]
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    # Standard deviation
    variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
    std_dev = variance ** 0.5
    
    # Coefficient of variation (lower = more uniform)
    cv = (std_dev / avg_count * 100) if avg_count > 0 else 0
    
    print(f"Expected count per worker (ideal): {expected_count:.1f}")
    print(f"Min count:                         {min_count}")
    print(f"Max count:                         {max_count}")
    print(f"Avg count:                         {avg_count:.1f}")
    print(f"Std deviation:                     {std_dev:.2f}")
    print(f"Coefficient of variation:          {cv:.2f}% (lower = more uniform)")
    
    # Check for perfectly balanced distribution
    if min_count == max_count:
        print("\nDistribution is perfectly balanced.")
    else:
        imbalance = ((max_count - min_count) / expected_count) * 100 if expected_count > 0 else 0
        print(f"\nMax imbalance:                     {imbalance:.2f}% of expected")
    
    # Time range if timestamps available
    timestamps = [r['timestamp'] for r in records if r.get('timestamp')]
    if timestamps:
        print("\n" + "-" * 80)
        print("TIME RANGE")
        print("-" * 80)
        print(f"First selection: {min(timestamps)}")
        print(f"Last selection:  {max(timestamps)}")
    
    print("\n" + "=" * 80)


def print_summary(records: List[Dict], all_worker_ids: set, log_format: LogFormat) -> None:
    """Print summary statistics based on log format."""
    if log_format == LogFormat.KV_ROUTER:
        print_kv_router_summary(records, all_worker_ids)
    elif log_format == LogFormat.ROUND_ROBIN:
        print_round_robin_summary(records, all_worker_ids)
    else:
        print("Unknown log format. No worker selection entries found.")
        if all_worker_ids:
            print(f"Total unique worker IDs observed: {len(all_worker_ids)}")


def save_filtered_lines(raw_lines: List[str], output_path: Path) -> None:
    """Save the filtered lines containing worker selections to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(raw_lines)
    print(f"Saved {len(raw_lines)} filtered lines to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze worker selection from dynamo frontend logs"
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to the log file to analyze"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file for filtered worker selection lines (default: <logfile>_selected_workers.log)"
    )
    parser.add_argument(
        "--no-filter-output",
        action="store_true",
        help="Skip saving filtered lines to a file"
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate lines based on worker selection content (ignoring timestamps)"
    )
    parser.add_argument(
        "--format",
        choices=["auto", "kv-router", "round-robin"],
        default="auto",
        help="Log format to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    if not args.logfile.exists():
        print(f"Error: Log file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)
    
    # Determine log format
    if args.format == "auto":
        log_format = None
    elif args.format == "kv-router":
        log_format = LogFormat.KV_ROUTER
    else:
        log_format = LogFormat.ROUND_ROBIN
    
    print(f"Analyzing: {args.logfile}")
    if args.dedup:
        print("Deduplication enabled: removing duplicate worker selections")
    
    records, raw_lines, all_worker_ids, detected_format = analyze_log_file(
        args.logfile, dedup=args.dedup, log_format=log_format
    )
    
    format_name = "KV Router" if detected_format == LogFormat.KV_ROUTER else (
        "Round Robin" if detected_format == LogFormat.ROUND_ROBIN else "Unknown"
    )
    print(f"Detected format: {format_name}")
    
    print_summary(records, all_worker_ids, detected_format)
    
    if not args.no_filter_output:
        if args.output:
            output_path = args.output
        else:
            output_path = args.logfile.parent / f"{args.logfile.stem}_selected_workers.log"
        save_filtered_lines(raw_lines, output_path)


if __name__ == "__main__":
    main()
