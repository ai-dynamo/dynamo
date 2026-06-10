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
//! DYNAMIC-BEHAVIOR / RECOVERY MODE (DIS-2172 dynamic test): in addition to the
//! single scalar window summary above, this subscriber emits a TIME-RESOLVED
//! timeline of per-bucket (received, gaps) counts on the node0 CLOCK_MONOTONIC
//! axis (bucket width `DYN_BENCH_BUCKET_MS`, default 100ms). The orchestrator
//! injects a disturbance (worker leave/join, sub restart) at a known node0
//! monotonic instant t0; recovery.py aligns the bucket timeline to t0 (via the
//! monotonic-epoch line below) and computes per-sub recovery time + loss. The
//! timeline runs for the WHOLE post-anchor span (warmup + measure +
//! `DYN_BENCH_OBSERVE_TAIL`) so the dip and recovery after t0 are captured.
//!
//! SLOW-CONSUMER MODE: when `DYN_BENCH_SLOW_SLEEP_MS` > 0 this sub sleeps that
//! long after counting each event, deliberately consuming slowly so its
//! RCVHWM / merge-channel / broadcast-channel fill and it starts dropping. This
//! drives the slow-subscriber-backpressure scenario (is the back-pressure
//! isolated to the slow sub, or does it bleed onto the fast subs?).
//!
//! MONOTONIC-EPOCH LINE: at startup the sub prints one stderr JSON line
//! `{"dis2172_clock":"epoch",...}` carrying both a CLOCK_MONOTONIC reading
//! (ns since process-internal ref) and a CLOCK_REALTIME reading (unix ns) taken
//! back-to-back. The orchestrator prints its own realtime↔monotonic pair; since
//! orchestrator and subs are co-located on node0 (same kernel clock), recovery.py
//! maps both t0 and the bucket timestamps onto one node0-monotonic axis. The
//! recovery time is therefore a difference of two node0-monotonic instants —
//! PTP-immune (no cross-node clock comparison).
//!
//! Config via env vars (set by the bench harness):
//!   DYN_BENCH_NAMESPACE            (default "dynamo")
//!   DYN_BENCH_COMPONENT           (default "backend")
//!   DYN_BENCH_TOPIC              (default "forward-pass-metrics")
//!   DYN_BENCH_SCOPE             "component"|"namespace" (default "component")
//!   DYN_BENCH_DURATION         measurement window seconds (default 20)
//!   DYN_BENCH_WARMUP          warmup seconds, excluded from counts (default 4)
//!   DYN_BENCH_OBSERVE_TAIL   extra post-window seconds the timeline keeps
//!                            recording (for recovery after a mid-window inject;
//!                            default 0 = steady-state behaviour, no tail)
//!   DYN_BENCH_BUCKET_MS      timeline bucket width ms (default 100; 0 = off)
//!   DYN_BENCH_SLOW_SLEEP_MS  per-event sleep ms for slow-consumer mode (def 0)
//!   DYN_BENCH_FIRST_EVENT_TIMEOUT  seconds to wait for event #1 (default 120)
//!   DYN_BENCH_TRANSPORT      "nats"|"zmq" to override; default = runtime default
//!   DYN_BENCH_OUT           output JSON path (default "/out/result.json")
//!   DYN_BENCH_BUCKETS_OUT   timeline JSONL path (default "<DYN_BENCH_OUT>.buckets.jsonl")
//!
//! Transport honors `DYN_EVENT_PLANE` via the runtime default unless
//! `DYN_BENCH_TRANSPORT` overrides it.

use dynamo_runtime::discovery::EventTransportKind;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use dynamo_runtime::{DistributedRuntime, Runtime, Worker, logging};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// One timeline bucket: counts in `[bucket_start, bucket_start + bucket_ms)` on
/// the node0 CLOCK_MONOTONIC axis. `t_mono_ns` is the bucket START, expressed as
/// ns since the same process-internal monotonic reference used by the epoch line
/// (see `mono_epoch_ns`), so recovery.py can align it to the orchestrator's t0.
struct Bucket {
    received: u64,
    gaps: u64,
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
    let observe_tail_secs = env_u64("DYN_BENCH_OBSERVE_TAIL", 0);
    let bucket_ms = env_u64("DYN_BENCH_BUCKET_MS", 100);
    let slow_sleep_ms = env_u64("DYN_BENCH_SLOW_SLEEP_MS", 0);
    let first_event_timeout_secs = env_u64("DYN_BENCH_FIRST_EVENT_TIMEOUT", 120);
    let out_path = env_or("DYN_BENCH_OUT", "/out/result.json");
    let buckets_out_path = env_or(
        "DYN_BENCH_BUCKETS_OUT",
        &format!("{out_path}.buckets.jsonl"),
    );

    let transport = match std::env::var("DYN_BENCH_TRANSPORT").ok().as_deref() {
        Some("nats") => EventTransportKind::Nats,
        Some("zmq") => EventTransportKind::Zmq,
        _ => drt.default_event_transport_kind(),
    };

    // --- Monotonic-epoch line: a back-to-back (monotonic, realtime) pair so the
    // orchestrator (also on node0) can map bucket monotonic ns <-> its own t0.
    // `mono_epoch` is the process-internal monotonic reference: all bucket
    // `t_mono_ns` below are ns since this instant.
    let mono_epoch = Instant::now();
    let unix_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id();
    eprintln!(
        "{{\"dis2172_clock\":\"epoch\",\"pid\":{pid},\"mono_epoch_ns\":0,\"unix_ns\":{unix_ns},\
         \"out\":\"{out_path}\",\"buckets_out\":\"{buckets_out_path}\"}}"
    );

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
         observe_tail={observe_tail_secs}s bucket_ms={bucket_ms} slow_sleep_ms={slow_sleep_ms} \
         first_event_timeout={first_event_timeout_secs}s"
    );

    let mut last_seq: HashMap<u64, u64> = HashMap::new();
    let mut received: u64 = 0; // events counted INSIDE the post-warmup measure window
    let mut gaps: u64 = 0; // intra-window per-publisher sequence gaps (silent drops)

    // Time-resolved timeline buckets (node0 monotonic). `bucket_anchor_ns` is the
    // ns-since-mono_epoch of bucket index 0 (set when the anchor is established).
    let bucketing = bucket_ms > 0;
    let bucket_dur = Duration::from_millis(bucket_ms.max(1));
    let mut buckets: Vec<Bucket> = Vec::new();
    let slow_sleep = Duration::from_millis(slow_sleep_ms);

    let warmup = Duration::from_secs(warmup_secs);
    let measure = Duration::from_secs(duration_secs);
    let observe_tail = Duration::from_secs(observe_tail_secs);
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
    // Anchor the bucket axis to the same instant the window is anchored to.
    let bucket_anchor_ns: u64 = anchor.duration_since(mono_epoch).as_nanos() as u64;

    // Helper: bucket index for `now` since the bucket anchor. Lazily grows the
    // bucket vec so empty (dip) intervals are recorded as zero-count buckets.
    let bucket_idx_for = |now: Instant| -> usize {
        (now.duration_since(anchor).as_nanos() / bucket_dur.as_nanos()) as usize
    };

    // --- Phase 2: warmup + measure + observe_tail. The scalar received/gaps only
    // count the original [warmup, warmup+measure] window (report.py parity); the
    // timeline buckets cover the whole post-anchor span (recovery after inject). ---
    let total = warmup + measure + observe_tail;
    loop {
        let Some(remaining) = total.checked_sub(anchor.elapsed()) else {
            break;
        };
        // Cap the wait slice so we wake at least every bucket (and at least every
        // 500ms) even with no traffic -> empty buckets during a dip are recorded.
        let slice = remaining.min(bucket_dur).min(Duration::from_millis(500));
        match tokio::time::timeout(slice, sub.next()).await {
            Ok(Some(Ok(envelope))) => {
                let now = Instant::now();

                // Per-publisher sequence gap detection (catches silent drops).
                let mut this_gap: u64 = 0;
                if let Some(&prev) = last_seq.get(&envelope.publisher_id)
                    && envelope.sequence > prev + 1
                {
                    this_gap = envelope.sequence - prev - 1;
                }
                last_seq.insert(envelope.publisher_id, envelope.sequence);

                // Scalar window counters (report.py parity): post-warmup only.
                if now.duration_since(anchor) >= warmup {
                    received += 1;
                    gaps += this_gap;
                }

                // Timeline bucket (whole post-anchor span). Gaps land in the
                // bucket of the event that revealed them (the recovery curve uses
                // received for throughput and gaps for loss attribution).
                if bucketing {
                    let bi = bucket_idx_for(now);
                    if bi >= buckets.len() {
                        buckets.resize_with(bi + 1, || Bucket {
                            received: 0,
                            gaps: 0,
                        });
                    }
                    buckets[bi].received += 1;
                    buckets[bi].gaps += this_gap;
                }

                // Slow-consumer mode: deliberately consume slowly so HWM fills.
                if slow_sleep_ms > 0 {
                    tokio::time::sleep(slow_sleep).await;
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

    // Timeline: one JSONL line per bucket. t_mono_ns = bucket START as ns since
    // the process-internal monotonic reference (mono_epoch). The FIRST line is a
    // self-describing header carrying the realtime anchor (the unix-ns reading
    // taken back-to-back with mono_epoch at startup), so recovery.py can put the
    // timeline onto the node0 realtime axis (bucket_realtime_ns = anchor_unix_ns +
    // t_mono_ns) and subtract t0_real_ns from the inject file — a node0-only delta.
    if bucketing && !buckets.is_empty() {
        use std::io::Write;
        let mut s = String::with_capacity(buckets.len() * 48 + 96);
        s.push_str(&format!(
            "{{\"dis2172_buckets\":\"header\",\"anchor_unix_ns\":{},\"anchor_mono_ns\":0,\
             \"bucket_ms\":{},\"n\":{}}}\n",
            unix_ns,
            bucket_ms,
            buckets.len()
        ));
        for (i, b) in buckets.iter().enumerate() {
            let t_mono_ns = bucket_anchor_ns + (i as u64) * bucket_dur.as_nanos() as u64;
            s.push_str(&format!(
                "{{\"t_mono_ns\":{},\"received\":{},\"gaps\":{}}}\n",
                t_mono_ns, b.received, b.gaps
            ));
        }
        match std::fs::File::create(&buckets_out_path).and_then(|mut f| f.write_all(s.as_bytes())) {
            Ok(()) => eprintln!(
                "[bench-sub] wrote {} buckets -> {buckets_out_path}",
                buckets.len()
            ),
            Err(e) => eprintln!("[bench-sub] bucket write failed: {e}"),
        }
        // Live stderr echo of the timeline (slurm streams stderr to node0 LOGDIR,
        // so the orchestrator can recover buckets even if the --out file is on a
        // remote node's local fs). One compact line carrying the whole series.
        eprintln!(
            "{{\"dis2172_buckets\":\"final\",\"bucket_ms\":{},\"anchor_mono_ns\":{},\"n\":{}}}",
            bucket_ms,
            bucket_anchor_ns,
            buckets.len()
        );
    }

    print!("{json}");
    Ok(())
}
