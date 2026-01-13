#!/usr/bin/env python3
"""
Analyze worker selection from dynamo frontend logs.

Parses log files for "Selected worker:" entries and provides statistics
on job allocation across workers.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_selected_worker_line(line: str) -> Optional[Dict]:
    """
    Parse a log line containing "Selected worker:" and extract fields.
    
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


def get_dedup_key(line: str) -> str:
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


def analyze_log_file(log_path: Path, dedup: bool = False) -> Tuple[List[Dict], List[str]]:
    """
    Read a log file and extract all Selected worker entries.
    
    Args:
        log_path: Path to the log file
        dedup: If True, deduplicate lines based on worker selection content
               (ignoring timestamps)
    
    Returns:
        - List of parsed worker selection records
        - List of raw lines containing "Selected worker:"
    """
    records = []
    raw_lines = []
    seen_keys = set()
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if "Selected worker:" in line:
                if dedup:
                    key = get_dedup_key(line)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                
                raw_lines.append(line)
                parsed = parse_selected_worker_line(line)
                if parsed:
                    records.append(parsed)
    
    return records, raw_lines


def print_summary(records: List[Dict]) -> None:
    """Print summary statistics for worker selection data."""
    if not records:
        print("No 'Selected worker:' entries found in log file.")
        return
    
    print("=" * 80)
    print("WORKER SELECTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal selections: {len(records)}")
    
    # Group by worker_id
    by_worker = defaultdict(list)
    for r in records:
        by_worker[r['worker_id']].append(r)
    
    print(f"Unique workers: {len(by_worker)}")
    
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


def save_filtered_lines(raw_lines: List[str], output_path: Path) -> None:
    """Save the filtered lines containing 'Selected worker:' to a file."""
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
        help="Output file for filtered 'Selected worker:' lines (default: <logfile>_selected_workers.log)"
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
    
    args = parser.parse_args()
    
    if not args.logfile.exists():
        print(f"Error: Log file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Analyzing: {args.logfile}")
    if args.dedup:
        print("Deduplication enabled: removing duplicate worker selections")
    records, raw_lines = analyze_log_file(args.logfile, dedup=args.dedup)
    
    print_summary(records)
    
    if not args.no_filter_output:
        if args.output:
            output_path = args.output
        else:
            output_path = args.logfile.parent / f"{args.logfile.stem}_selected_workers.log"
        save_filtered_lines(raw_lines, output_path)


if __name__ == "__main__":
    main()

