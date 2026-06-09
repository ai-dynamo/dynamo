// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DIS-2172 event-plane TRANSPORT-LAYER benchmark subscriber (counting only).
//!
//! Subscribes to a component- or namespace-scoped event-plane topic and counts
//! delivered events for the transport throughput + loss comparison (ZMQ vs
//! NATS). This is the STOCK-TRANSPORT variant: it carries NO latency
//! instrumentation. It builds against stock `main` (the `EventEnvelope` has NO
//! `published_at_ns` field) so the publish hot path is completely un-perturbed
//! by this benchmark — we are measuring the transport, not our own timestamps.
//!
//! What it measures (all clock-free, transport-only):
//!   - received        : total events delivered inside the measurement window.
//!   - per-publisher sequence gaps : silent-drop detection. ZMQ uses
//!     `SNDTIMEOUT=0` (fail-fast) and silently drops on a full HWM; without gap
//!     accounting ZMQ would look artificially good because dropped events are
//!     never counted. NATS over TCP does not silently drop, so its gaps≈0.
//!   - drop_rate       : gaps / (received + gaps), the intra-window loss rate.
//!
//! WINDOW ANCHORING (DIS-2172 "s=16" fix): the warmup+measure window is anchored
//! to the FIRST event this subscriber actually receives, NOT to a fixed
//! wall-clock taken at process start. With many subscribers (e.g. s=16) a sub
//! can finish its subscription handshake well after the orchestrator started the
//! load, so a fixed-clock window would open (and even close) before this sub saw
//! a single event — reporting a spurious received=0 ("s=16 zero events" bug). By
//! waiting up to `DYN_BENCH_FIRST_EVENT_TIMEOUT` for event #1 and only then
//! starting warmup+measure, every subscriber measures a full, comparable window.
//!
//! Config via env vars (set by the bench harness):
//!   DYN_BENCH_NAMESPACE            (default "dynamo")
//!   DYN_BENCH_COMPONENT           (default "backend")
//!   DYN_BENCH_TOPIC              (default "forward-pass-metrics")
//!   DYN_BENCH_SCOPE             "component"|"namespace" (default "component")
//!   DYN_BENCH_DURATION         measurement window seconds (default 20)
//!   DYN_BENCH_WARMUP          warmup seconds, excluded from counts (default 4)
//!   DYN_BENCH_FIRST_EVENT_TIMEOUT  seconds to wait for event #1 (default 120)
//!   DYN_BENCH_TRANSPORT      "nats"|"zmq" to override; default = runtime default
//!   DYN_BENCH_OUT           output JSON path (default "/out/result.json")
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
    let duration_secs = env_u64("DYN_BENCH_DURATION", 20);
    let warmup_secs = env_u64("DYN_BENCH_WARMUP", 4);
    let first_event_timeout_secs = env_u64("DYN_BENCH_FIRST_EVENT_TIMEOUT", 120);
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
         scope={scope} transport={transport:?} warmup={warmup_secs}s window={duration_secs}s \
         first_event_timeout={first_event_timeout_secs}s"
    );

    let mut last_seq: HashMap<u64, u64> = HashMap::new();
    let mut received: u64 = 0; // events counted INSIDE the post-warmup measure window
    let mut gaps: u64 = 0; // intra-window per-publisher sequence gaps (silent drops)

    let warmup = Duration::from_secs(warmup_secs);
    let measure = Duration::from_secs(duration_secs);
    let first_event_timeout = Duration::from_secs(first_event_timeout_secs);

    // --- Phase 1: wait for the FIRST event (window anchor, DIS-2172 s=16 fix). ---
    // Block (bounded) until event #1 arrives, then anchor warmup+measure to NOW.
    // Events seen here (the handshake-warmup tail) are intentionally not counted.
    let wait_start = Instant::now();
    let anchor: Instant = loop {
        let Some(remaining) = first_event_timeout.checked_sub(wait_start.elapsed()) else {
            // No event ever arrived within the budget -> report received=0 (the
            // honest outcome: this sub never got traffic on this topic).
            eprintln!("[bench-sub] no first event within {first_event_timeout_secs}s; received=0");
            break Instant::now();
        };
        let slice = remaining.min(Duration::from_millis(500));
        match tokio::time::timeout(slice, sub.next()).await {
            Ok(Some(Ok(envelope))) => {
                // Seed the per-publisher sequence tracker so the first counted
                // event in-window does not register a spurious gap.
                last_seq.insert(envelope.publisher_id, envelope.sequence);
                break Instant::now();
            }
            Ok(Some(Err(e))) => eprintln!("[bench-sub] event error: {e}"),
            Ok(None) => {
                eprintln!("[bench-sub] stream ended before first event");
                break Instant::now();
            }
            Err(_) => { /* timeout slice; re-check budget */ }
        }
    };

    // --- Phase 2: warmup + measure window, anchored to the first event. ---
    let total = warmup + measure;
    loop {
        let Some(remaining) = total.checked_sub(anchor.elapsed()) else {
            break;
        };
        let slice = remaining.min(Duration::from_millis(500));
        match tokio::time::timeout(slice, sub.next()).await {
            Ok(Some(Ok(envelope))) => {
                // Per-publisher sequence gap detection (catches silent drops),
                // tracked across the whole post-anchor span so a drop straddling
                // the warmup boundary is still seen.
                if let Some(&prev) = last_seq.get(&envelope.publisher_id) {
                    if envelope.sequence > prev + 1 {
                        if anchor.elapsed() >= warmup {
                            gaps += envelope.sequence - prev - 1;
                        }
                    }
                }
                last_seq.insert(envelope.publisher_id, envelope.sequence);

                if anchor.elapsed() >= warmup {
                    received += 1;
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

    let n_publishers = last_seq.len();
    let drop_rate = if received + gaps == 0 {
        0.0
    } else {
        gaps as f64 / (received + gaps) as f64
    };
    // sent_est = window-span estimate: received + intra-span seq gaps. Under our
    // continuous-load model there is no "tail loss" (mocker never stops sending
    // mid-window), so this is the seq-span the transport SHOULD have delivered in
    // the window. drop_rate == gaps/sent_est is therefore the intra-window loss.
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
            "  \"gaps\": {gaps},\n",
            "  \"drop_rate\": {drop_rate:.6}\n",
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
        gaps = gaps,
        drop_rate = drop_rate,
    );

    std::fs::write(&out_path, &json)?;
    eprintln!("[bench-sub] wrote {out_path}");
    print!("{json}");
    Ok(())
}
