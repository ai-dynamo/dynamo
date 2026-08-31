// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best-effort producer-side persistence for forward-pass metrics.
//!
//! This trace is an analysis aid, not a durable replacement for the FPM event
//! plane. Each producer publishes into its own bounded in-process queue and
//! never waits for disk I/O. A slow sink can therefore drop trace records
//! without delaying inference or normal FPM delivery.

pub mod config;
mod sink;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tokio::sync::{Mutex, Notify, broadcast};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::utils::GracefulTaskGuard;

pub use config::{FpmTraceMode, FpmTracePolicy, is_enabled, policy};

#[derive(Clone, Debug)]
pub(crate) struct FpmTraceEvent {
    observed_at_unix_ms: u64,
    payload: Bytes,
    source: Arc<sink::FpmTraceSource>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FpmTraceKey {
    runtime_id: String,
    producer_id: String,
}

struct FpmTraceInner {
    producer_id: String,
    sender: broadcast::Sender<FpmTraceEvent>,
    shutdown: CancellationToken,
    accepting: AtomicBool,
    publishers_in_flight: AtomicUsize,
    publishers_idle: Notify,
    leases: AtomicUsize,
    dropped: AtomicU64,
    closed: AtomicBool,
    closed_notify: Notify,
}

impl FpmTraceInner {
    fn begin_publish(&self) -> bool {
        if !self.accepting.load(Ordering::SeqCst) {
            return false;
        }
        self.publishers_in_flight.fetch_add(1, Ordering::SeqCst);
        if self.accepting.load(Ordering::SeqCst) {
            true
        } else {
            self.finish_publish();
            false
        }
    }

    fn finish_publish(&self) {
        if self.publishers_in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.publishers_idle.notify_waiters();
        }
    }

    async fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::SeqCst);
        loop {
            let idle = self.publishers_idle.notified();
            if self.publishers_in_flight.load(Ordering::SeqCst) == 0 {
                return;
            }
            idle.await;
        }
    }

    fn record_dropped(&self, count: u64) -> u64 {
        self.dropped.fetch_add(count, Ordering::Relaxed) + count
    }

    fn mark_closed(&self) {
        self.accepting.store(false, Ordering::SeqCst);
        self.closed.store(true, Ordering::Release);
        self.publishers_idle.notify_waiters();
        self.closed_notify.notify_waiters();
    }

    async fn wait_closed(&self) {
        loop {
            let notified = self.closed_notify.notified();
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
}

/// Producer-owned, nonblocking handle to one FPM trace sink.
///
/// Handles with the same runtime and producer share one queue and one writer.
/// Each handle stamps its own namespace/component source on every event, so
/// components sharing a producer cannot collide on the output file or inherit
/// another component's attribution. Different producers have independent
/// queues, files, and retention budgets.
pub(crate) struct FpmTrace {
    inner: Arc<FpmTraceInner>,
    source: Arc<sink::FpmTraceSource>,
}

impl FpmTrace {
    fn try_new_lease(inner: Arc<FpmTraceInner>, source: Arc<sink::FpmTraceSource>) -> Option<Self> {
        // Increment only while at least one producer handle is still alive.
        // This makes lease acquisition atomic with the last handle's 1 -> 0
        // transition, so a concurrent initializer cannot resurrect a sink
        // after its shutdown has begun.
        inner
            .leases
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |leases| {
                if leases == 0 {
                    None
                } else {
                    leases.checked_add(1)
                }
            })
            .ok()?;
        Some(Self { inner, source })
    }

    /// Try to enqueue one finalized FPM msgpack payload for persistence.
    ///
    /// This never waits. A bounded broadcast queue intentionally retains the
    /// newest records under load; the sink reports overwritten-record counts.
    /// `false` means shutdown had already stopped accepting trace records.
    pub(crate) fn publish_payload(&self, payload: Bytes) -> bool {
        if !self.inner.begin_publish() {
            return false;
        }
        let observed_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or_default();
        let sent = self
            .inner
            .sender
            .send(FpmTraceEvent {
                observed_at_unix_ms,
                payload,
                source: self.source.clone(),
            })
            .is_ok();
        self.inner.finish_publish();
        if !sent {
            self.inner.record_dropped(1);
        }
        sent
    }

    #[cfg(test)]
    async fn wait_closed(&self) {
        self.inner.wait_closed().await;
    }
}

impl Clone for FpmTrace {
    fn clone(&self) -> Self {
        Self::try_new_lease(self.inner.clone(), self.source.clone())
            .expect("a live FPM trace handle must own a producer lease")
    }
}

impl Drop for FpmTrace {
    fn drop(&mut self) {
        if self.inner.leases.fetch_sub(1, Ordering::AcqRel) == 1 {
            // The last producer stopped. Close its sink even when the runtime
            // itself remains alive; other producers have independent sinks.
            self.inner.shutdown.cancel();
        }
    }
}

enum TraceRegistryEntry {
    Empty,
    Active(Weak<FpmTraceInner>),
    /// Initialization failures are terminal for this producer. A multi-DP
    /// producer otherwise retries the same bad path and emits one warning per
    /// relay, while failures for unrelated producers must remain isolated.
    Failed,
}

type TraceSlot = Arc<Mutex<TraceRegistryEntry>>;
type TraceRegistry = HashMap<FpmTraceKey, TraceSlot>;
static REGISTRY: OnceLock<Mutex<TraceRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<TraceRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) async fn init_from_env_with_shutdown(
    runtime_id: &str,
    namespace: &str,
    component: &str,
    producer_id: &str,
    shutdown: CancellationToken,
    graceful_guard: Option<GracefulTaskGuard>,
) -> anyhow::Result<Option<FpmTrace>> {
    let policy = policy();
    if !policy.enabled {
        return Ok(None);
    }

    init_with_policy(
        runtime_id,
        namespace,
        component,
        producer_id,
        policy.clone(),
        shutdown,
        graceful_guard,
    )
    .await
}

async fn init_with_policy(
    runtime_id: &str,
    namespace: &str,
    component: &str,
    producer_id: &str,
    policy: FpmTracePolicy,
    shutdown: CancellationToken,
    graceful_guard: Option<GracefulTaskGuard>,
) -> anyhow::Result<Option<FpmTrace>> {
    let source = sink::FpmTraceSource {
        namespace: namespace.to_string(),
        component: component.to_string(),
        producer_id: producer_id.to_string(),
    };
    let source = Arc::new(source);
    let key = FpmTraceKey {
        runtime_id: runtime_id.to_string(),
        producer_id: producer_id.to_string(),
    };

    // Serialize initialization only within one producer. Unrelated producers
    // can preflight and open their writers concurrently.
    let slot = {
        let mut registry = registry().lock().await;
        registry
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(TraceRegistryEntry::Empty)))
            .clone()
    };
    let mut entry = slot.lock().await;
    match &*entry {
        TraceRegistryEntry::Failed => return Ok(None),
        TraceRegistryEntry::Active(inner) => {
            if let Some(inner) = inner.upgrade() {
                if let Some(trace) = FpmTrace::try_new_lease(inner.clone(), source.clone()) {
                    if !inner.shutdown.is_cancelled() && !inner.closed.load(Ordering::Acquire) {
                        return Ok(Some(trace));
                    }
                    drop(trace);
                }
                // Do not let a rapid producer restart open the same segment
                // while the previous gzip writer is still finalizing it.
                inner.wait_closed().await;
            }
        }
        TraceRegistryEntry::Empty => {}
    }

    let (sender, receiver) = broadcast::channel(config::DEFAULT_CAPACITY);
    let inner = Arc::new(FpmTraceInner {
        producer_id: producer_id.to_string(),
        sender,
        // A child token makes producer-local close idempotent while also
        // inheriting runtime phase-1 cancellation.
        shutdown: shutdown.child_token(),
        accepting: AtomicBool::new(true),
        publishers_in_flight: AtomicUsize::new(0),
        publishers_idle: Notify::new(),
        // The handle returned below is the initial producer lease.
        leases: AtomicUsize::new(1),
        dropped: AtomicU64::new(0),
        closed: AtomicBool::new(false),
        closed_notify: Notify::new(),
    });

    if let Err(error) =
        sink::spawn_worker(policy.clone(), receiver, inner.clone(), graceful_guard).await
    {
        *entry = TraceRegistryEntry::Failed;
        return Err(error);
    }
    *entry = TraceRegistryEntry::Active(Arc::downgrade(&inner));

    tracing::info!(
        namespace,
        component,
        producer_id,
        mode = ?policy.mode,
        sample_interval_ms = policy.sample_interval_ms,
        output_path = %policy.output_path,
        max_segments = policy.max_segments,
        "FPM trace initialized"
    );
    Ok(Some(FpmTrace { inner, source }))
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::path::Path;

    use flate2::read::MultiGzDecoder;
    use serde_json::Value;

    use super::*;
    use crate::telemetry::jsonl_gz::segment_path;

    fn full_policy(path: &Path) -> FpmTracePolicy {
        FpmTracePolicy {
            enabled: true,
            output_path: path.display().to_string(),
            mode: FpmTraceMode::Full,
            sample_interval_ms: 100,
            jsonl_gz_roll_bytes: 1024 * 1024,
            max_segments: 4,
        }
    }

    fn payload(worker_id: &str, counter_id: i64) -> Bytes {
        Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({
                "version": 1,
                "worker_id": worker_id,
                "dp_rank": 0,
                "counter_id": counter_id,
                "wall_time": 0.01,
                "scheduled_requests": {
                    "num_prefill_requests": 1,
                    "sum_prefill_tokens": 16,
                    "var_prefill_length": 0.0,
                    "sum_prefill_kv_tokens": 0,
                    "num_decode_requests": 2,
                    "sum_decode_kv_tokens": 64,
                    "var_decode_kv_tokens": 0.0,
                },
                "queued_requests": {
                    "num_prefill_requests": 0,
                    "sum_prefill_tokens": 0,
                    "var_prefill_length": 0.0,
                    "num_decode_requests": 0,
                    "sum_decode_kv_tokens": 0,
                    "var_decode_kv_tokens": 0.0,
                },
            }))
            .unwrap(),
        )
    }

    fn records(base_path: &Path, producer_id: &str) -> Vec<Value> {
        let producer_path =
            sink::producer_output_path(&base_path.display().to_string(), producer_id);
        let bytes = std::fs::read(segment_path(Path::new(&producer_path), 0)).unwrap();
        let mut decoder = MultiGzDecoder::new(bytes.as_slice());
        let mut content = String::new();
        decoder.read_to_string(&mut content).unwrap();
        content
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn trace_without_worker() -> FpmTrace {
        let (sender, _receiver) = broadcast::channel(1);
        let inner = Arc::new(FpmTraceInner {
            producer_id: "producer-lease-test".to_string(),
            sender,
            shutdown: CancellationToken::new(),
            accepting: AtomicBool::new(true),
            publishers_in_flight: AtomicUsize::new(0),
            publishers_idle: Notify::new(),
            leases: AtomicUsize::new(1),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            closed_notify: Notify::new(),
        });
        let source = Arc::new(sink::FpmTraceSource {
            namespace: "namespace".to_string(),
            component: "backend".to_string(),
            producer_id: "producer-lease-test".to_string(),
        });
        FpmTrace { inner, source }
    }

    #[test]
    fn producer_lease_cannot_be_resurrected_after_last_handle_drops() {
        let trace = trace_without_worker();
        let inner = trace.inner.clone();
        let source = trace.source.clone();

        drop(trace);

        assert!(inner.shutdown.is_cancelled());
        assert_eq!(inner.leases.load(Ordering::Acquire), 0);
        assert!(FpmTrace::try_new_lease(inner.clone(), source).is_none());
        assert_eq!(inner.leases.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn producer_registry_shares_matching_producers_and_isolates_others() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("trace");
        let policy = full_policy(&base_path);
        let runtime_a_shutdown = CancellationToken::new();
        let runtime_b_shutdown = CancellationToken::new();

        let producer_a = init_with_policy(
            "runtime-a",
            "namespace",
            "backend",
            "producer-a",
            policy.clone(),
            runtime_a_shutdown.clone(),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let producer_a_second_relay = init_with_policy(
            "runtime-a",
            "namespace",
            "backend",
            "producer-a",
            policy.clone(),
            runtime_a_shutdown.clone(),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let producer_a_other_component = init_with_policy(
            "runtime-a",
            "namespace",
            "sidecar",
            "producer-a",
            policy.clone(),
            runtime_a_shutdown.clone(),
            None,
        )
        .await
        .unwrap()
        .unwrap();
        let producer_b = init_with_policy(
            "runtime-b",
            "namespace",
            "backend",
            "producer-b",
            policy,
            runtime_b_shutdown.clone(),
            None,
        )
        .await
        .unwrap()
        .unwrap();

        assert!(Arc::ptr_eq(
            &producer_a.inner,
            &producer_a_second_relay.inner
        ));
        assert!(Arc::ptr_eq(
            &producer_a.inner,
            &producer_a_other_component.inner
        ));
        assert!(!Arc::ptr_eq(&producer_a.inner, &producer_b.inner));

        assert!(producer_a.publish_payload(payload("worker-a", 1)));
        assert!(producer_a_second_relay.publish_payload(payload("worker-a", 2)));
        assert!(producer_a_other_component.publish_payload(payload("worker-a", 3)));
        assert!(producer_b.publish_payload(payload("worker-b", 7)));

        // Runtime shutdown is idempotent. Each worker first closes the
        // publish acceptance boundary, then drains the stable queue.
        runtime_a_shutdown.cancel();
        runtime_a_shutdown.cancel();
        runtime_b_shutdown.cancel();
        runtime_b_shutdown.cancel();
        producer_a.wait_closed().await;
        producer_b.wait_closed().await;

        let a_records = records(&base_path, "producer-a");
        let b_records = records(&base_path, "producer-b");
        let a_counters: Vec<_> = a_records
            .iter()
            .map(|record| record["event"]["fpm"]["counter_id"].as_i64().unwrap())
            .collect();
        assert_eq!(a_counters, [1, 2, 3]);
        assert_eq!(b_records.len(), 1);
        assert_eq!(b_records[0]["event"]["fpm"]["counter_id"], 7);
        assert!(a_records[..2].iter().all(|record| {
            record["event"]["source"]["producer_id"] == "producer-a"
                && record["event"]["fpm"]["worker_id"] == "worker-a"
                && record["event"]["source"]["component"] == "backend"
        }));
        assert_eq!(a_records[2]["event"]["source"]["component"], "sidecar");
        assert_eq!(b_records[0]["event"]["source"]["producer_id"], "producer-b");
        assert!(!producer_a.publish_payload(payload("worker-a", 3)));
    }

    #[tokio::test]
    async fn failed_initialization_is_terminal_only_for_its_producer() {
        let dir = tempfile::tempdir().unwrap();
        let blocked_parent = dir.path().join("blocked-parent");
        std::fs::write(&blocked_parent, b"not a directory").unwrap();
        let failed_policy = full_policy(&blocked_parent.join("trace"));
        let valid_base = dir.path().join("valid-trace");
        let valid_policy = full_policy(&valid_base);

        let first = init_with_policy(
            "failure-runtime",
            "namespace",
            "backend",
            "failed-producer",
            failed_policy,
            CancellationToken::new(),
            None,
        )
        .await;
        assert!(first.is_err());

        // A valid policy would succeed if retried. The cached failure instead
        // returns disabled without repeating preflight or another warning.
        let repeated = init_with_policy(
            "failure-runtime",
            "namespace",
            "backend",
            "failed-producer",
            valid_policy.clone(),
            CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
        assert!(repeated.is_none());

        let other_shutdown = CancellationToken::new();
        let other = init_with_policy(
            "failure-runtime",
            "namespace",
            "backend",
            "healthy-producer",
            valid_policy,
            other_shutdown.clone(),
            None,
        )
        .await
        .unwrap()
        .expect("one producer's failure must not disable another producer");
        assert!(other.publish_payload(payload("healthy-worker", 1)));
        other_shutdown.cancel();
        other.wait_closed().await;
        assert_eq!(records(&valid_base, "healthy-producer").len(), 1);
    }
}
