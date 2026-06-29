# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny metrics registry + Prometheus text exporter.

Why not just use `prometheus_client`: we want zero non-stdlib
dependencies on the daemon's hot path. The counters here are pure
Python (atomic int via Lock, fine for the consumer-thread cadence).

Usage:

    from gms_kv_ring.common import metrics
    metrics.evict_records.inc(engine_id="eng-A", n=1)
    metrics.evict_d2h_bytes.inc(engine_id="eng-A", n=4096)

    # Anywhere:
    text = metrics.render_prometheus()
    # or expose over HTTP:
    server = metrics.serve_http("0.0.0.0", 9090)
    ...
    server.shutdown()
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from typing import Optional


class Counter:
    """Monotonic counter with one optional string label dimension.

    We deliberately keep this at *one* label (typically engine_id).
    A real metrics library handles arbitrary label sets; we don't
    need that here, and limiting it keeps the code dumb-simple."""

    def __init__(
        self, name: str, help_text: str, label_name: Optional[str] = None
    ) -> None:
        self.name = name
        self.help_text = help_text
        self.label_name = label_name
        self._values: dict[str, int] = {}
        self._lock = threading.Lock()

    def inc(self, **labels) -> None:
        n = int(labels.pop("n", 1))
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + n

    def set(self, value: int, **labels) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = int(value)

    def get(self, **labels) -> int:
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0)

    def _key(self, labels: dict) -> str:
        if not self.label_name:
            if labels:
                raise ValueError(f"{self.name}: counter has no label, got {labels}")
            return ""
        if list(labels.keys()) != [self.label_name]:
            raise ValueError(
                f"{self.name}: expected single label {self.label_name!r}, "
                f"got {list(labels.keys())}"
            )
        return str(labels[self.label_name])

    def render(self) -> str:
        out = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            items = list(self._values.items())
        for key, val in items:
            if self.label_name:
                out.append(f'{self.name}{{{self.label_name}="{key}"}} {val}')
            else:
                out.append(f"{self.name} {val}")
        return "\n".join(out) + "\n"


# ----- registry -----

_ALL_COUNTERS: list[Counter] = []


def _register(c: Counter) -> Counter:
    _ALL_COUNTERS.append(c)
    return c


evict_records = _register(
    Counter(
        "gms_kvr_evict_records_total",
        "Evict ring records consumed by the daemon",
        label_name="engine_id",
    )
)
evict_d2h_bytes = _register(
    Counter(
        "gms_kvr_evict_d2h_bytes_total",
        "Bytes copied D2H by the evict consumer",
        label_name="engine_id",
    )
)
evict_errors = _register(
    Counter(
        "gms_kvr_evict_errors_total",
        "cuMemcpyAsync D2H errors during evict consumption",
        label_name="engine_id",
    )
)
restore_records = _register(
    Counter(
        "gms_kvr_restore_records_total",
        "Restore ring records consumed by the daemon",
        label_name="engine_id",
    )
)
restore_failures = _register(
    Counter(
        "gms_kvr_restore_failures_total",
        "Restores that signalled failure (target+1) to the engine — "
        "missing host_tier slot, H2D error, or sync error",
        label_name="engine_id",
    )
)
restore_h2d_bytes = _register(
    Counter(
        "gms_kvr_restore_h2d_bytes_total",
        "Bytes copied H2D by the restore consumer",
        label_name="engine_id",
    )
)
ring_drops = _register(
    Counter(
        "gms_kvr_ring_drops_total",
        "Records dropped because the ring was full (engine producer)",
        label_name="ring_type",
    )
)
host_tier_leaked_slots = _register(
    Counter(
        "gms_kvr_host_tier_leaked_slots_total",
        "Host-tier slots intentionally leaked (index dropped, buffer NOT "
        "freed) because stream sync failed at detach — prevents "
        "cudaFreeHost-during-DMA undefined behavior. Process restart "
        "reclaims.",
        label_name="engine_id",
    )
)
host_tier_slots = _register(
    Counter(
        "gms_kvr_host_tier_slots",
        "Current host-tier slot count (gauge — set, not incremented)",
        label_name="engine_id",
    )
)
checksum_mismatches = _register(
    Counter(
        "gms_kvr_checksum_mismatches_total",
        "Host-tier slot CRC32 mismatches detected on restore — indicates "
        "the offloaded bytes were corrupted between evict and restore "
        "(DMA tear, bitrot, or stale slot read). Restore signals failure "
        "to the engine on mismatch.",
        label_name="engine_id",
    )
)
storage_tier_slots = _register(
    Counter(
        "gms_kvr_storage_tier_slots",
        "Current storage-tier (filesystem) slot count (gauge — set, not "
        "incremented). One slot per demoted (engine_id, layer, offset).",
        label_name="engine_id",
    )
)
storage_tier_promote_failures = _register(
    Counter(
        "gms_kvr_storage_tier_promote_failures_total",
        "Storage-tier promote failures — file missing, header malformed, "
        "or CRC mismatch (at-rest corruption).",
        label_name="engine_id",
    )
)
storage_tier_evictions = _register(
    Counter(
        "gms_kvr_storage_tier_evictions_total",
        "Storage-tier slots evicted by operator-driven cleanup — sum of "
        "TTL prune + byte-quota enforcement + explicit release.",
        label_name="reason",
    )
)
storage_tier_bytes = _register(
    Counter(
        "gms_kvr_storage_tier_bytes",
        "Total payload bytes resident in the storage tier (gauge).",
    )
)
backend_call_failures = _register(
    Counter(
        "gms_kvr_backend_call_failures_total",
        "Python exceptions raised by a supervised storage backend method. "
        "Labeled by backend name; method name is in the log line.",
        label_name="backend",
    )
)
backend_restarts = _register(
    Counter(
        "gms_kvr_backend_restarts_total",
        "Supervised storage backend re-instantiations after persistent "
        "Python-level failures. Process-level (C SIGSEGV) crashes are NOT "
        "counted here — they kill the daemon.",
        label_name="backend",
    )
)
mooncake_manifest_compactions = _register(
    Counter(
        "gms_kvr_mooncake_manifest_compactions_total",
        "MooncakeBackend manifest compactions (sidecar JSONL rewrites).",
    )
)
demote_hbm_overlapped = _register(
    Counter(
        "gms_kvr_demote_hbm_overlapped_total",
        "demote_hbm_to_storage calls that used the overlapped fast "
        "path (GDS write concurrent with async D2H+CPU CRC).",
        label_name="engine_id",
    )
)
demote_hbm_sync = _register(
    Counter(
        "gms_kvr_demote_hbm_sync_total",
        "demote_hbm_to_storage calls that used the synchronous "
        "fallback (CRC pre-computed before GDS write).",
        label_name="engine_id",
    )
)
promote_hbm_ok = _register(
    Counter(
        "gms_kvr_promote_hbm_ok_total",
        "promote_storage_to_hbm calls that completed with verified "
        "CRC. Engine consumes the HBM bytes.",
        label_name="engine_id",
    )
)
promote_hbm_failures = _register(
    Counter(
        "gms_kvr_promote_hbm_failures_total",
        "promote_storage_to_hbm calls that failed verification "
        "(slot missing, header mismatch, CRC mismatch). Engine falls "
        "to cold compute.",
        label_name="engine_id",
    )
)
daemon_engine_attaches = _register(
    Counter(
        "gms_kvr_daemon_engine_attaches_total",
        "Engine pool attach RPCs accepted by the daemon. A rising rate "
        "for the same engine_id usually means process restart, failover, "
        "or a supervisor crash loop.",
        label_name="engine_id",
    )
)
daemon_engine_detaches = _register(
    Counter(
        "gms_kvr_daemon_engine_detaches_total",
        "Engine pool detach operations completed by the daemon. Pair "
        "with attach counts to detect unbalanced lifecycle churn.",
        label_name="engine_id",
    )
)
daemon_engine_reattaches = _register(
    Counter(
        "gms_kvr_daemon_engine_reattaches_total",
        "Attach RPCs that replaced an already-attached pool for the "
        "same engine_id. This should be rare outside restart/failover "
        "tests; sustained increases indicate duplicate owners or a "
        "flapping engine supervisor.",
        label_name="engine_id",
    )
)
daemon_storage_releases = _register(
    Counter(
        "gms_kvr_daemon_storage_releases_total",
        "Explicit release_engine_storage RPCs. Each increment means "
        "persistent storage for the engine_id was intentionally retired.",
        label_name="engine_id",
    )
)
daemon_generation_mismatches = _register(
    Counter(
        "gms_kvr_daemon_generation_mismatches_total",
        "promote_storage_to_hbm calls rejected because expected_generation "
        "did not match the daemon's persisted block generation. These are "
        "Race #3 stale-read preventions; the engine should recompute.",
        label_name="engine_id",
    )
)
daemon_stale_demotes = _register(
    Counter(
        "gms_kvr_daemon_stale_demotes_total",
        "demote_hbm_to_storage calls rejected because the caller supplied "
        "a generation older than the daemon's persisted block generation. "
        "This indicates a stale writer or duplicate owner attempted to "
        "overwrite newer durable KV bytes.",
        label_name="engine_id",
    )
)
daemon_persisted_generations = _register(
    Counter(
        "gms_kvr_daemon_persisted_generations",
        "Current count of block generation records retained by the daemon "
        "for an engine_id. This is a gauge despite the simple Counter "
        "implementation; it is updated with set().",
        label_name="engine_id",
    )
)

# Connector-level metrics (V1 KVCacheConnector wiring). These
# fire from the worker side of GMSKVCacheConnectorV1 in
# `bind_connector_metadata` to surface restore/evict outcomes
# the daemon-level counters don't capture (e.g., ring full).
connector_evict_failures = _register(
    Counter(
        "gms_kvr_connector_evict_failures_total",
        "V1 connector saw `evict_blocks_to_storage` raise or return "
        "partial failure. Bytes did NOT reach durable storage; vLLM "
        "still receives a 'finished save' ack to avoid block-pin "
        "deadlock, so the storage tier is the source of truth for "
        "whether the spill succeeded.",
        label_name="engine_id",
    )
)
connector_restore_failures = _register(
    Counter(
        "gms_kvr_connector_restore_failures_total",
        "V1 connector restore failed verification at the daemon "
        "(restore_succeeded() returned False after stream sync) or "
        "the connector's sync remap raised. Engine reads stale HBM "
        "for these blocks — output for the affected requests will be "
        "wrong. Alert on this counter.",
        label_name="engine_id",
    )
)
connector_restore_ring_full = _register(
    Counter(
        "gms_kvr_connector_restore_ring_full_total",
        "V1 connector's `record_restore_gds` returned None (ring "
        "full). Connector fell back to synchronous restore in bind; "
        "forward pass blocks for the cuFile read. Sustained increases "
        "mean the restore ring capacity is undersized for cache-hit "
        "rate × per-step pair count.",
        label_name="engine_id",
    )
)
connector_restore_conflict_sync = _register(
    Counter(
        "gms_kvr_connector_restore_conflict_sync_total",
        "V1 connector routed a restore through the SYNCHRONOUS lane "
        "because its src_block_id was also in the same step's evict "
        "set (race-#2 conflict). Steady-state should be near zero in "
        "well-behaved workloads; spikes indicate frequent prefix-cache "
        "thrash where hash-indexed blocks are being immediately "
        "re-evicted.",
        label_name="engine_id",
    )
)
connector_daemon_epoch_changes = _register(
    Counter(
        "gms_kvr_connector_daemon_epoch_changes_total",
        "V1 connector observed a daemon-epoch change (daemon process "
        "restarted) and dropped its in-memory _PrefixIndex. After this "
        "the connector behaves like a cold cache until evictions "
        "repopulate the index. Sustained increases mean the daemon "
        "is crash-looping.",
        label_name="engine_id",
    )
)
connector_prefix_invalidated_on_failure = _register(
    Counter(
        "gms_kvr_connector_prefix_invalidated_on_failure_total",
        "V1 connector dropped a (engine_id, block_id) entry from its "
        "_PrefixIndex after `restore_succeeded()=False` indicated the "
        "daemon could not deliver clean bytes for that slot. Bounds "
        "the blast radius of an unrecoverable restore failure to ONE "
        "request — subsequent requests can't re-claim the same broken "
        "slot. Pair with `connector_restore_failures` (one-to-one in "
        "the common case).",
        label_name="engine_id",
    )
)
daemon_scrub_corruptions = _register(
    Counter(
        "gms_kvr_daemon_scrub_corruptions_total",
        "Daemon scrub thread detected a host_tier slot whose in-RAM "
        "CRC didn't match the stored CRC. Slot was dropped. Each "
        "increment is an at-rest corruption event — typically rare "
        "on ECC RAM; a non-zero rate may indicate failing hardware.",
        label_name="engine_id",
    )
)
daemon_scrub_scanned = _register(
    Counter(
        "gms_kvr_daemon_scrub_scanned_total",
        "Number of host_tier slots scrubbed (CRC re-verified) by the "
        "daemon's background scrub thread. Useful as a denominator "
        "for `daemon_scrub_corruptions`.",
        label_name="engine_id",
    )
)
daemon_backend_scrub_corruptions = _register(
    Counter(
        "gms_kvr_daemon_backend_scrub_corruptions_total",
        "Daemon backend scrub thread detected a durable-storage slot "
        "(NIXL file, mooncake blob, etc.) whose payload CRC didn't "
        "match the stored CRC. Slot was dropped. Critical for the GDS "
        "production path — GDS reads bypass host_tier verification, "
        "so backend scrub is the only at-rest corruption detector for "
        "those deployments.",
        label_name="engine_id",
    )
)
daemon_backend_scrub_scanned = _register(
    Counter(
        "gms_kvr_daemon_backend_scrub_scanned_total",
        "Number of backend slots verified by the daemon's background "
        "backend-scrub thread. Denominator for "
        "`daemon_backend_scrub_corruptions`.",
        label_name="engine_id",
    )
)


def render_prometheus() -> str:
    """Render every registered counter in Prometheus text format."""
    return "".join(c.render() + "\n" for c in _ALL_COUNTERS)


# ----- HTTP exporter -----


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a) -> None:
        return  # silence per-request stderr noise


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_http(host: str = "127.0.0.1", port: int = 0) -> _ThreadedHTTPServer:
    """Start a /metrics HTTP server in a background thread.

    Returns the server object — call `.shutdown()` to stop. If port=0
    the OS assigns one; read it from `server.server_address[1]`."""
    server = _ThreadedHTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
