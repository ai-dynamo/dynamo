#!/usr/bin/env python3
import json
import math
import sys
from collections import defaultdict


INTERESTING_ATTRS = {
    "http-request": [
        "input_tokens",
        "output_tokens",
        "ttft_ms",
        "avg_itl_ms",
        "busy_ns",
        "idle_ns",
    ],
    "kv_router.compute_block_hashes": ["isl_tokens", "block_count"],
    "kv_router.compute_seq_hashes": ["seq_hash_count"],
    "kv_router.find_matches": ["block_count"],
    "kv_router.schedule": [
        "block_count",
        "overlap_blocks",
        "cached_tokens",
        "pending_count",
        "pending_isl_tokens",
    ],
    "kv_router.route_request": [
        "overlap_blocks",
        "effective_overlap_blocks",
        "cached_tokens",
    ],
    "kv_router.backend_stream": [
        "first_item_ms",
        "chunks",
        "data_chunks",
        "error_chunks",
        "elapsed_ms",
    ],
    "openai.chat_completions.sse_stream": [
        "first_backend_chunk_ms",
        "first_sse_event_ms",
        "backend_chunks",
        "empty_chunks",
        "sse_events",
        "tool_dispatch_events",
        "reasoning_dispatch_events",
        "error_events",
        "elapsed_ms",
    ],
    "request_plane.frontend.build_envelope": ["buffer_bytes"],
    "request_plane.frontend.send_request": ["buffer_bytes", "ack_bytes"],
    "request_plane.frontend.decode_response_stream": [
        "first_response_ms",
        "frames",
        "data_frames",
        "error_frames",
        "decode_errors",
        "response_bytes",
        "elapsed_ms",
    ],
    "request_plane.worker.handle_payload": ["payload_bytes"],
    "request_plane.worker.pump_response_stream": [
        "first_response_ms",
        "chunks",
        "error_chunks",
        "response_bytes",
        "elapsed_ms",
    ],
}


def usage() -> None:
    print(f"usage: {sys.argv[0]} OTEL_TRACES_JSON [SPAN_NAME...]", file=sys.stderr)


def attr_value(attrs, key):
    for attr in attrs or []:
        if attr.get("key") != key:
            continue
        value = attr.get("value", {})
        if "intValue" in value:
            return int(value["intValue"])
        if "doubleValue" in value:
            return float(value["doubleValue"])
        if "stringValue" in value:
            raw = value["stringValue"]
            try:
                return int(raw)
            except ValueError:
                try:
                    return float(raw)
                except ValueError:
                    return raw
        if "boolValue" in value:
            return bool(value["boolValue"])
    return None


def numeric_attr(attrs, key):
    value = attr_value(attrs, key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def iter_spans(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch = json.loads(line)
            for resource_span in batch.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    for span in scope_span.get("spans", []):
                        yield span


def percentile(values, pct):
    if not values:
        return 0.0
    idx = math.ceil((pct / 100.0) * len(values)) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def duration_ms(span):
    start = int(span.get("startTimeUnixNano", 0))
    end = int(span.get("endTimeUnixNano", 0))
    if end <= start:
        return None
    return (end - start) / 1_000_000.0


def print_span_duration_summary(by_name):
    print("| span | count | total_ms | avg_ms | p50_ms | p95_ms | p99_ms | max_ms | busy_total_ms | idle_total_ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, data in sorted(by_name.items(), key=lambda item: sum(item[1]["dur_ms"]), reverse=True):
        dur = sorted(data["dur_ms"])
        busy = data["busy_ms"]
        idle = data["idle_ms"]
        total = sum(dur)
        count = len(dur)
        avg = total / count if count else 0.0
        print(
            f"| `{name}` | {count} | {total:.2f} | {avg:.4f} | "
            f"{percentile(dur, 50):.4f} | {percentile(dur, 95):.4f} | "
            f"{percentile(dur, 99):.4f} | {max(dur):.4f} | "
            f"{sum(busy):.2f} | {sum(idle):.2f} |"
        )


def print_attr_summary(attr_by_name):
    rows = []
    for name in sorted(attr_by_name):
        for key in sorted(attr_by_name[name]):
            values = sorted(attr_by_name[name][key])
            if not values:
                continue
            rows.append((name, key, values))

    if not rows:
        return

    print()
    print("## Selected Span Attributes")
    print()
    print("| span | attr | count | avg | p50 | p95 | p99 | max |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, key, values in rows:
        total = sum(values)
        count = len(values)
        avg = total / count if count else 0.0
        print(
            f"| `{name}` | `{key}` | {count} | {avg:.4f} | "
            f"{percentile(values, 50):.4f} | {percentile(values, 95):.4f} | "
            f"{percentile(values, 99):.4f} | {max(values):.4f} |"
        )


def print_http_outliers(http_roots, limit=20):
    if not http_roots:
        return

    print()
    print("## Slowest HTTP Requests")
    print()
    print("| rank | duration_ms | request_id | model | input_tokens | output_tokens | ttft_ms | avg_itl_ms | busy_ms | idle_ms |")
    print("| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, (dur, attrs) in enumerate(sorted(http_roots, reverse=True)[:limit], start=1):
        request_id = attr_value(attrs, "request_id") or ""
        model = attr_value(attrs, "model") or ""
        input_tokens = attr_value(attrs, "input_tokens") or 0
        output_tokens = attr_value(attrs, "output_tokens") or 0
        ttft_ms = attr_value(attrs, "ttft_ms") or 0
        avg_itl_ms = attr_value(attrs, "avg_itl_ms") or 0
        busy_ns = attr_value(attrs, "busy_ns") or 0
        idle_ns = attr_value(attrs, "idle_ns") or 0
        busy_ms = float(busy_ns) / 1_000_000.0 if isinstance(busy_ns, (int, float)) else 0.0
        idle_ms = float(idle_ns) / 1_000_000.0 if isinstance(idle_ns, (int, float)) else 0.0
        print(
            f"| {rank} | {dur:.2f} | `{request_id}` | `{model}` | "
            f"{input_tokens} | {output_tokens} | {float(ttft_ms):.2f} | "
            f"{float(avg_itl_ms):.2f} | {busy_ms:.2f} | {idle_ms:.2f} |"
        )


def main() -> int:
    if len(sys.argv) < 2:
        usage()
        return 2

    path = sys.argv[1]
    names_filter = set(sys.argv[2:])
    by_name = defaultdict(lambda: {"dur_ms": [], "busy_ms": [], "idle_ms": []})
    attr_by_name = defaultdict(lambda: defaultdict(list))
    http_roots = []

    for span in iter_spans(path):
        name = span.get("name", "")
        if names_filter and name not in names_filter:
            continue
        dur = duration_ms(span)
        if dur is None:
            continue
        attrs = span.get("attributes", [])
        bucket = by_name[name]
        bucket["dur_ms"].append(dur)
        busy_ns = attr_value(attrs, "busy_ns")
        idle_ns = attr_value(attrs, "idle_ns")
        if isinstance(busy_ns, (int, float)):
            bucket["busy_ms"].append(float(busy_ns) / 1_000_000.0)
        if isinstance(idle_ns, (int, float)):
            bucket["idle_ms"].append(float(idle_ns) / 1_000_000.0)

        if name == "http-request":
            http_roots.append((dur, attrs))

        for key in INTERESTING_ATTRS.get(name, []):
            value = numeric_attr(attrs, key)
            if value is not None:
                attr_by_name[name][key].append(value)

    print_span_duration_summary(by_name)
    print_attr_summary(attr_by_name)
    if not names_filter:
        print_http_outliers(http_roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
