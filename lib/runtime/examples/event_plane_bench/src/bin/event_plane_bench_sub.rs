// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DIS-2172 event-plane latency benchmark subscriber.
//!
//! Subscribes to a component-scoped event-plane topic and measures one-way
//! publish->deliver latency from `EventEnvelope::published_at_ns` (nanosecond
//! wall-clock stamped by the publisher at `publish_bytes`). SAME-HOST ONLY:
//! publisher and this subscriber share `CLOCK_REALTIME`, so the recv-minus-
//! published delta is a valid one-way latency. Across hosts NTP/PTP skew would
//! dominate.
//!
//! Also tracks per-publisher sequence gaps. ZMQ uses `SNDTIMEOUT=0` (fail-fast)
//! and silently drops on a full HWM; without gap accounting ZMQ would look
//! artificially fast because dropped events are never measured.
//!
//! Config via env vars (set by the bench harness):
//!   DYN_BENCH_NAMESPACE   (default "dynamo")
//!   DYN_BENCH_COMPONENT   (default "backend")
//!   DYN_BENCH_TOPIC       (default "forward-pass-metrics")
//!   DYN_BENCH_DURATION    measurement window seconds (default 30)
//!   DYN_BENCH_WARMUP      warmup seconds, excluded from latency (default 3)
//!   DYN_BENCH_TRANSPORT   "nats"|"zmq" to override; default = runtime default
//!   DYN_BENCH_OUT         output JSON path (default "/out/result.json")
//!
//! Transport honors `DYN_EVENT_PLANE` via the runtime default unless
//! `DYN_BENCH_TRANSPORT` overrides it.

use dynamo_runtime::discovery::EventTransportKind;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use dynamo_runtime::{DistributedRuntime, Runtime, Worker, logging};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Monotonic ns (`CLOCK_MONOTONIC`) — must match the publisher's clock so the
/// recv-minus-published delta is valid (system-wide on Linux, same host).
fn now_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid timespec; CLOCK_MONOTONIC is always available on Linux.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    (ts.tv_sec as u64)
        .wrapping_mul(1_000_000_000)
        .wrapping_add(ts.tv_nsec as u64)
}

fn main() -> anyhow::Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> anyhow::Result<()> {
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    let namespace = env_or("DYN_BENCH_NAMESPACE", "dynamo");
    let component_name = env_or("DYN_BENCH_COMPONENT", "backend");
    let topic = env_or("DYN_BENCH_TOPIC", "forward-pass-metrics");
    let duration_secs = env_u64("DYN_BENCH_DURATION", 30);
    let warmup_secs = env_u64("DYN_BENCH_WARMUP", 3);
    let out_path = env_or("DYN_BENCH_OUT", "/out/result.json");

    let transport = match std::env::var("DYN_BENCH_TRANSPORT").ok().as_deref() {
        Some("nats") => EventTransportKind::Nats,
        Some("zmq") => EventTransportKind::Zmq,
        _ => drt.default_event_transport_kind(),
    };

    // Scope must match the publisher: kv-events / forward-pass-metrics are
    // component-scoped; kv_metrics is namespace-scoped (EventPublisher::for_namespace).
    let scope = env_or("DYN_BENCH_SCOPE", "component");
    let ns_obj = drt.namespace(namespace.clone())?;
    let mut sub = if scope == "namespace" {
        EventSubscriber::for_namespace_with_transport(&ns_obj, topic.clone(), transport).await?
    } else {
        let component = ns_obj.component(component_name.clone())?;
        EventSubscriber::for_component_with_transport(&component, topic.clone(), transport).await?
    };

    eprintln!(
        "[bench-sub] subscribed ns={namespace} comp={component_name} topic={topic} \
         scope={scope} transport={transport:?} warmup={warmup_secs}s window={duration_secs}s"
    );

    let mut latencies_ns: Vec<u64> = Vec::new();
    let mut last_seq: HashMap<u64, u64> = HashMap::new();
    let mut received: u64 = 0; // total across warmup+window (for gap accounting)
    let mut measured: u64 = 0; // post-warmup events with a recorded latency
    let mut gaps: u64 = 0;
    let mut zero_ts: u64 = 0;

    let warmup = Duration::from_secs(warmup_secs);
    let total = Duration::from_secs(warmup_secs + duration_secs);
    let start = Instant::now();

    loop {
        let Some(remaining) = total.checked_sub(start.elapsed()) else {
            break;
        };
        // Cap each wait slice so we re-check the deadline even on an idle stream.
        let slice = remaining.min(Duration::from_millis(500));
        match tokio::time::timeout(slice, sub.next()).await {
            Ok(Some(Ok(envelope))) => {
                received += 1;
                // Per-publisher sequence gap detection (catches silent drops).
                if let Some(&prev) = last_seq.get(&envelope.publisher_id) {
                    if envelope.sequence > prev + 1 {
                        gaps += envelope.sequence - prev - 1;
                    }
                }
                last_seq.insert(envelope.publisher_id, envelope.sequence);

                if start.elapsed() >= warmup {
                    if envelope.published_at_ns == 0 {
                        zero_ts += 1;
                    } else {
                        latencies_ns.push(now_ns().saturating_sub(envelope.published_at_ns));
                        measured += 1;
                    }
                }
            }
            Ok(Some(Err(e))) => eprintln!("[bench-sub] event error: {e}"),
            Ok(None) => {
                eprintln!("[bench-sub] stream ended early");
                break;
            }
            Err(_) => { /* timeout slice; re-check deadline */ }
        }
    }

    latencies_ns.sort_unstable();
    let pct = |p: f64| -> u64 {
        if latencies_ns.is_empty() {
            return 0;
        }
        let rank = (p / 100.0 * (latencies_ns.len() as f64 - 1.0)).round() as usize;
        latencies_ns[rank.min(latencies_ns.len() - 1)]
    };
    let mean: u64 = if latencies_ns.is_empty() {
        0
    } else {
        (latencies_ns.iter().map(|&v| v as u128).sum::<u128>() / latencies_ns.len() as u128) as u64
    };
    let min = latencies_ns.first().copied().unwrap_or(0);
    let max = latencies_ns.last().copied().unwrap_or(0);
    let p50 = pct(50.0);
    let p90 = pct(90.0);
    let p95 = pct(95.0);
    let p99 = pct(99.0);
    let n_publishers = last_seq.len();
    let drop_rate = if received + gaps == 0 {
        0.0
    } else {
        gaps as f64 / (received + gaps) as f64
    };
    // sent_est = window-span estimate: received + intra-span seq gaps. Under our
    // continuous-load model there is no "tail loss" (mocker never stops sending),
    // so this is the seq-span loss within the subscribe window. drop_rate below
    // == gaps/sent_est is therefore the exact intra-window loss rate.
    let sent_est = received + gaps;
    let transport_str = format!("{transport:?}");

    let json = format!(
        concat!(
            "{{\n",
            "  \"transport\": \"{transport_str}\",\n",
            "  \"namespace\": \"{namespace}\",\n",
            "  \"component\": \"{component_name}\",\n",
            "  \"topic\": \"{topic}\",\n",
            "  \"warmup_secs\": {warmup_secs},\n",
            "  \"window_secs\": {duration_secs},\n",
            "  \"n_publishers\": {n_publishers},\n",
            "  \"sent_est\": {sent_est},\n",
            "  \"received\": {received},\n",
            "  \"measured\": {measured},\n",
            "  \"gaps\": {gaps},\n",
            "  \"drop_rate\": {drop_rate:.6},\n",
            "  \"zero_ts_events\": {zero_ts},\n",
            "  \"latency_ns\": {{\n",
            "    \"min\": {min},\n",
            "    \"mean\": {mean},\n",
            "    \"p50\": {p50},\n",
            "    \"p90\": {p90},\n",
            "    \"p95\": {p95},\n",
            "    \"p99\": {p99},\n",
            "    \"max\": {max}\n",
            "  }}\n",
            "}}\n",
        ),
        transport_str = transport_str,
        namespace = namespace,
        component_name = component_name,
        topic = topic,
        warmup_secs = warmup_secs,
        duration_secs = duration_secs,
        n_publishers = n_publishers,
        sent_est = sent_est,
        received = received,
        measured = measured,
        gaps = gaps,
        drop_rate = drop_rate,
        zero_ts = zero_ts,
        min = min,
        mean = mean,
        p50 = p50,
        p90 = p90,
        p95 = p95,
        p99 = p99,
        max = max,
    );

    std::fs::write(&out_path, &json)?;
    eprintln!("[bench-sub] wrote {out_path}");
    print!("{json}");
    Ok(())
}
