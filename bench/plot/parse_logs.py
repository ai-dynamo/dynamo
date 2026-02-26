# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fallback parser: extract per-concurrency aiperf metrics from client.log.

Used for runs that predate per-concurrency artifact directories.  The aiperf
rich-text summary table is printed once per concurrency level in client.log.
We split on the Unicode box-drawing ``│`` character to extract values.
"""

from __future__ import annotations

import re
from pathlib import Path

# Maps the human-readable metric name (as it appears in the aiperf table)
# to the key we use in the returned dict.  Matching is done via substring,
# checked longest-first to avoid partial matches.
_METRIC_KEY_MAP: dict[str, str] = {
    "Time to First Token": "time_to_first_token",
    "Time to First Output Token": "time_to_first_token",
    "Inter Token Latency": "inter_token_latency",
    "Request Latency": "request_latency",
    "Output Token Throughput Per User": "output_token_throughput_per_user",
    "Output Token Throughput": "output_token_throughput",
    "Request Throughput": "request_throughput",
    "Request Count": "request_count",
}

# Sorted longest-first for matching priority
_SORTED_DISPLAY_NAMES = sorted(_METRIC_KEY_MAP.keys(), key=len, reverse=True)

# Column order in the aiperf table
_COLUMNS = ("avg", "min", "max", "p99", "p90", "p50", "std")

# Unicode box-drawing vertical bar used by Rich tables
_BOX_VERT = "\u2502"  # │

# Regex to detect a concurrency header emitted by client.sh
_CONCURRENCY_RE = re.compile(r"\[client\.sh\]\s+Concurrency:\s+(\d+)")


def _parse_num(s: str) -> float | None:
    """Parse a number string, stripping commas.  Return None for N/A / empty."""
    s = s.strip()
    if not s or s == "N/A":
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _split_row(line: str) -> list[str] | None:
    """Split a Rich table row on │ and return the inner cells, or None."""
    if _BOX_VERT not in line:
        return None
    parts = line.split(_BOX_VERT)
    # A valid data row: │ label │ v1 │ v2 │ ... │ v7 │
    # After split: ["", " label ", " v1 ", ..., " v7 ", ""]
    # Minimum 9 segments (empty + label + 7 values + empty)
    if len(parts) < 9:
        return None
    return [p.strip() for p in parts[1:-1]]  # drop boundary empties


def _try_match(label: str) -> str | None:
    """Try to match a (possibly accumulated) label to a metric key."""
    for display_name in _SORTED_DISPLAY_NAMES:
        if display_name in label:
            return _METRIC_KEY_MAP[display_name]
    return None


def _emit(
    results: dict[int, dict[str, dict[str, float | None]]],
    concurrency: int,
    label: str,
    values: list[float | None],
) -> None:
    """Try to match *label* and store *values* in results if matched."""
    key = _try_match(label)
    if key is None:
        return
    row_data = dict(zip(_COLUMNS, values))
    results.setdefault(concurrency, {})[key] = row_data


def parse_client_log(log_path: Path) -> dict[int, dict[str, dict[str, float | None]]]:
    """Parse a client.log and return per-concurrency aiperf metrics.

    Returns
    -------
    dict mapping concurrency (int) -> metric_key -> {avg, min, max, p99, p90, p50, std}
    """
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    results: dict[int, dict[str, dict[str, float | None]]] = {}
    current_concurrency: int | None = None

    # Pending: a data row whose label may be extended by continuation rows.
    pending_label: str = ""
    pending_values: list[float | None] | None = None

    def flush_pending() -> None:
        nonlocal pending_label, pending_values
        if current_concurrency is not None and pending_values is not None:
            _emit(results, current_concurrency, pending_label, pending_values)
        pending_label = ""
        pending_values = None

    for line in lines:
        # Check for concurrency header
        m = _CONCURRENCY_RE.search(line)
        if m:
            flush_pending()
            current_concurrency = int(m.group(1))
            continue

        if current_concurrency is None:
            continue

        cells = _split_row(line)
        if cells is None or len(cells) < 8:
            continue

        label_cell = cells[0]
        value_cells = cells[1:8]
        parsed_values = [_parse_num(v) for v in value_cells]
        has_data = any(v is not None for v in parsed_values)

        if has_data:
            # This is a new data row.
            # First, flush any pending (previous metric).
            flush_pending()
            # Start accumulating for this new metric.
            pending_label = label_cell
            pending_values = parsed_values
        else:
            # Continuation row (no data) -- extend the label.
            if label_cell and not label_cell.startswith("("):
                if pending_label:
                    pending_label = pending_label + " " + label_cell
                else:
                    pending_label = label_cell
            # Unit rows like "(ms)" or "(tokens/sec…" are ignored.

    # Flush the last pending metric
    flush_pending()

    return results
