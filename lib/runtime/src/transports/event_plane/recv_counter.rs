// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DIS-2172 receive-side event-plane THROUGHPUT + LOSS counter.
//!
//! This instruments the REAL event-plane consumers (KV Router kv-events ingest,
//! Planner/FPM receiver, peer-router active-sequences sync) to measure how well
//! they actually CONSUME events under request-driven load. It is deliberately
//! **clock-free**: we count received events and per-publisher sequence gaps
//! (transport-level `EventEnvelope::sequence`), NOT per-event latency.
//!
//! Design (mirrors the synthetic `event_plane_bench_sub` counting, on the recv
//! side of the real consumers):
//!   - `received`: total envelopes handed to this consumer.
//!   - `gaps`: sum of per-publisher sequence-number gaps (silent drops).
//!   - `n_publishers`: distinct publisher_ids seen.
//!   - window: anchored to the FIRST received event (like the existing bench
//!     fix) so a slow-to-subscribe consumer is not measured against an empty
//!     wall-clock window. Each `window_secs` we emit one JSON line to stderr
//!     and reset the per-window counters (cumulative totals also reported).
//!
//! Gated entirely by the `DYN_BENCH_COUNT` env var (off by default -> the
//! counter is a no-op and adds a single relaxed atomic load on the hot path).
//! A per-site label comes from `DYN_BENCH_COUNT_SITE` (or the explicit `site`
//! arg) so the bench harness can attribute a stderr line to kv-events / fpm /
//! active_sequences.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Returns true if receive-side counting is enabled (`DYN_BENCH_COUNT` set to a
/// truthy value). Cached after first read.
pub fn counting_enabled() -> bool {
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let on = std::env::var("DYN_BENCH_COUNT")
            .map(|v| !matches!(v.as_str(), "" | "0" | "false" | "False"))
            .unwrap_or(false);
        ENABLED.store(on, Ordering::Relaxed);
    });
    ENABLED.load(Ordering::Relaxed)
}

fn window_secs() -> u64 {
    std::env::var("DYN_BENCH_WINDOW_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10)
}

struct CounterState {
    last_seq: HashMap<u64, u64>,
    // cumulative (whole-run) totals
    total_received: u64,
    total_gaps: u64,
    // per-window deltas (reset on each emitted window)
    window_received: u64,
    window_gaps: u64,
    // window anchoring
    start: Option<Instant>,
    window_anchor: Option<Instant>,
    window_idx: u64,
}

/// Per-consumer receive-side counter. One instance lives in each real consumer
/// (kv-events subscriber task, FPM tracker, replica-sync subscriber).
pub struct RecvCounter {
    enabled: bool,
    site: String,
    window: Duration,
    state: Mutex<CounterState>,
}

impl RecvCounter {
    /// Create a counter for a given site label. If `DYN_BENCH_COUNT` is unset,
    /// the returned counter is a cheap no-op.
    ///
    /// `site` falls back to `DYN_BENCH_COUNT_SITE` then to the provided default.
    pub fn new(default_site: &str) -> Self {
        let enabled = counting_enabled();
        let site =
            std::env::var("DYN_BENCH_COUNT_SITE").unwrap_or_else(|_| default_site.to_string());
        Self {
            enabled,
            site,
            window: Duration::from_secs(window_secs()),
            state: Mutex::new(CounterState {
                last_seq: HashMap::new(),
                total_received: 0,
                total_gaps: 0,
                window_received: 0,
                window_gaps: 0,
                start: None,
                window_anchor: None,
                window_idx: 0,
            }),
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Record one received envelope by `(publisher_id, sequence)`. Performs
    /// per-publisher seq-gap accounting and emits a per-window JSON line to
    /// stderr when the window elapses. No-op if counting is disabled.
    pub fn record(&self, publisher_id: u64, sequence: u64) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        let mut st = self.state.lock().unwrap();

        // Anchor the measurement window to the first event this consumer sees.
        if st.start.is_none() {
            st.start = Some(now);
            st.window_anchor = Some(now);
            eprintln!(
                "{{\"dis2172_recv\":\"window_start\",\"site\":\"{}\",\"window_secs\":{}}}",
                self.site,
                self.window.as_secs()
            );
        }

        st.total_received += 1;
        st.window_received += 1;

        // Per-publisher transport-level sequence gap (silent drops).
        match st.last_seq.get(&publisher_id).copied() {
            Some(prev) if sequence > prev + 1 => {
                let g = sequence - prev - 1;
                st.total_gaps += g;
                st.window_gaps += g;
            }
            _ => {}
        }
        st.last_seq.insert(publisher_id, sequence);

        // Emit a per-window snapshot when the window elapses.
        let anchor = st.window_anchor.unwrap_or(now);
        if now.duration_since(anchor) >= self.window {
            self.emit_window(&mut st, now);
        }
    }

    fn emit_window(&self, st: &mut CounterState, now: Instant) {
        let elapsed = st
            .window_anchor
            .map(|a| now.duration_since(a).as_secs_f64())
            .unwrap_or(0.0);
        st.window_idx += 1;
        let rate = if elapsed > 0.0 {
            st.window_received as f64 / elapsed
        } else {
            0.0
        };
        let sent_est = st.window_received + st.window_gaps;
        let drop_rate = if sent_est > 0 {
            st.window_gaps as f64 / sent_est as f64
        } else {
            0.0
        };
        // Single-line JSON to stderr; the harness scrapes the last lines.
        eprintln!(
            "{{\"dis2172_recv\":\"window\",\"site\":\"{}\",\"window_idx\":{},\
\"window_secs\":{:.3},\"received\":{},\"gaps\":{},\"sent_est\":{},\
\"events_per_sec\":{:.2},\"drop_rate\":{:.6},\"n_publishers\":{},\
\"total_received\":{},\"total_gaps\":{}}}",
            self.site,
            st.window_idx,
            elapsed,
            st.window_received,
            st.window_gaps,
            sent_est,
            rate,
            drop_rate,
            st.last_seq.len(),
            st.total_received,
            st.total_gaps,
        );
        st.window_received = 0;
        st.window_gaps = 0;
        st.window_anchor = Some(now);
    }

    /// Emit a final cumulative summary line (e.g. on shutdown). No-op if
    /// disabled or if nothing was ever received.
    pub fn emit_final(&self) {
        if !self.enabled {
            return;
        }
        let st = self.state.lock().unwrap();
        if st.total_received == 0 {
            eprintln!(
                "{{\"dis2172_recv\":\"final\",\"site\":\"{}\",\"received\":0,\"gaps\":0,\"n_publishers\":0}}",
                self.site
            );
            return;
        }
        let sent_est = st.total_received + st.total_gaps;
        let drop_rate = if sent_est > 0 {
            st.total_gaps as f64 / sent_est as f64
        } else {
            0.0
        };
        eprintln!(
            "{{\"dis2172_recv\":\"final\",\"site\":\"{}\",\"received\":{},\"gaps\":{},\
\"sent_est\":{},\"drop_rate\":{:.6},\"n_publishers\":{}}}",
            self.site,
            st.total_received,
            st.total_gaps,
            sent_est,
            drop_rate,
            st.last_seq.len(),
        );
    }

    /// Cumulative `(received, gaps, n_publishers)` snapshot. Used by the FPM
    /// Python binding (`get_throughput_stats`) so the standalone receiver can
    /// print the counter itself.
    pub fn snapshot(&self) -> (u64, u64, u64) {
        let st = self.state.lock().unwrap();
        (st.total_received, st.total_gaps, st.last_seq.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_accounting() {
        // Force-enabled counter regardless of env.
        let c = RecvCounter {
            enabled: true,
            site: "test".to_string(),
            window: Duration::from_secs(3600),
            state: Mutex::new(CounterState {
                last_seq: HashMap::new(),
                total_received: 0,
                total_gaps: 0,
                window_received: 0,
                window_gaps: 0,
                start: None,
                window_anchor: None,
                window_idx: 0,
            }),
        };
        // publisher 1: 0,1,3 -> one gap (2 missing)
        c.record(1, 0);
        c.record(1, 1);
        c.record(1, 3);
        // publisher 2: 0,1 -> no gap
        c.record(2, 0);
        c.record(2, 1);
        let (received, gaps, n_pub) = c.snapshot();
        assert_eq!(received, 5);
        assert_eq!(gaps, 1);
        assert_eq!(n_pub, 2);
    }

    #[test]
    fn test_disabled_is_noop() {
        let c = RecvCounter {
            enabled: false,
            site: "test".to_string(),
            window: Duration::from_secs(1),
            state: Mutex::new(CounterState {
                last_seq: HashMap::new(),
                total_received: 0,
                total_gaps: 0,
                window_received: 0,
                window_gaps: 0,
                start: None,
                window_anchor: None,
                window_idx: 0,
            }),
        };
        c.record(1, 0);
        c.record(1, 5);
        let (received, gaps, _) = c.snapshot();
        assert_eq!(received, 0);
        assert_eq!(gaps, 0);
    }
}
