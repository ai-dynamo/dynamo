// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward Pass Metrics (FPM = ForwardPassMetrics) relay.
//!
//! Subscribes to the raw ZMQ PUB from `InstrumentedScheduler` (running in
//! a vLLM EngineCore child process) and re-publishes the payloads to the
//! Dynamo event plane with automatic discovery registration.
//!
//! This follows the same two-layer architecture as
//! [`crate::kv_router::publisher::KvEventPublisher`], but is much simpler:
//! no event transformation, no batching, no local indexer — just raw byte relay.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_mocker::common::protocols::{ForwardPassSnapshot, FpmPublisher, FpmSink};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventPublisher;

use crate::utils::zmq::{connect_sub_socket, multipart_message};

const FPM_TOPIC: &str = "forward-pass-metrics";
const FPM_VERSION: i32 = 1;
/// Matches Python `_FpmPublisherThread.HEARTBEAT_INTERVAL`.
const IDLE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);

/// A relay that bridges ForwardPassMetrics from a local raw ZMQ PUB socket
/// to the Dynamo event plane.
pub struct FpmEventRelay {
    cancel: CancellationToken,
}

impl FpmEventRelay {
    /// Create and start a new relay.
    ///
    /// - `component`: Dynamo component (provides runtime + discovery scope).
    /// - `zmq_endpoint`: Local ZMQ PUB address to subscribe to
    ///   (e.g., `tcp://127.0.0.1:20380`).
    pub fn new(component: Component, zmq_endpoint: String) -> Result<Self> {
        let rt = component.drt().runtime().secondary();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let publisher =
            rt.block_on(async { EventPublisher::for_component(&component, FPM_TOPIC).await })?;

        rt.spawn(async move {
            Self::relay_loop(zmq_endpoint, publisher, cancel_clone).await;
        });

        Ok(Self { cancel })
    }

    /// Shut down the relay task.
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    async fn relay_loop(
        zmq_endpoint: String,
        publisher: EventPublisher,
        cancel: CancellationToken,
    ) {
        let socket = match connect_sub_socket(&zmq_endpoint, None).await {
            Ok(socket) => socket,
            Err(error) => {
                tracing::error!(endpoint = %zmq_endpoint, error = %error, "FPM relay: failed to connect");
                return;
            }
        };
        let mut socket = socket;
        tracing::info!("FPM relay: connected to {zmq_endpoint}");

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::info!("FPM relay: shutting down");
                    break;
                }
                result = socket.next() => {
                    match result {
                        Some(Ok(frames)) => {
                            let mut frames = multipart_message(frames);
                            // ZMQ multipart: [topic, seq, payload]
                            if frames.len() == 3 {
                                let payload = frames.swap_remove(2);
                                if let Err(e) = publisher.publish_bytes(payload).await {
                                    tracing::warn!("FPM relay: event plane publish failed: {e}");
                                }
                            } else {
                                tracing::warn!(
                                    "FPM relay: unexpected ZMQ frame count: expected 3, got {}",
                                    frames.len()
                            );
                            }
                        }
                        Some(Err(e)) => {
                            tracing::error!("FPM relay: ZMQ recv failed: {e}");
                            break;
                        }
                        None => {
                            tracing::error!("FPM relay: ZMQ stream ended");
                            break;
                        }
                    }
                }
            }
        }
    }
}

impl Drop for FpmEventRelay {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

// ---------------------------------------------------------------------------
// Direct publisher: mocker scheduler -> event plane (no ZMQ hop)
// ---------------------------------------------------------------------------

/// Serialization struct matching Python `ScheduledRequestMetrics` from
/// `forward_pass_metrics.py`. Field names must match exactly since msgspec
/// (without `array_like=True`) encodes structs as msgpack **maps** with
/// string keys.
#[derive(Serialize)]
struct ScheduledRequestMetricsSer {
    num_prefill_requests: i32,
    sum_prefill_tokens: i64,
    var_prefill_length: f64,
    sum_prefill_kv_tokens: i64,
    num_decode_requests: i32,
    sum_decode_kv_tokens: i64,
    var_decode_kv_tokens: f64,
}

/// Serialization struct matching Python `QueuedRequestMetrics`.
#[derive(Serialize)]
struct QueuedRequestMetricsSer {
    num_prefill_requests: i32,
    sum_prefill_tokens: i64,
    var_prefill_length: f64,
    num_decode_requests: i32,
    sum_decode_kv_tokens: i64,
    var_decode_kv_tokens: f64,
}

/// Top-level serialization struct matching Python `ForwardPassMetrics`.
#[derive(Serialize)]
struct ForwardPassMetricsSer {
    version: i32,
    worker_id: String,
    dp_rank: i64,
    counter_id: i64,
    wall_time: f64,
    scheduled_requests: ScheduledRequestMetricsSer,
    queued_requests: QueuedRequestMetricsSer,
}

fn serialize_fpm(
    snapshot: &ForwardPassSnapshot,
    worker_id: &str,
    dp_rank: u32,
    counter_id: i64,
) -> Result<Vec<u8>> {
    let metrics = ForwardPassMetricsSer {
        version: FPM_VERSION,
        worker_id: worker_id.to_owned(),
        dp_rank: dp_rank as i64,
        counter_id,
        wall_time: snapshot.wall_time_secs,
        scheduled_requests: ScheduledRequestMetricsSer {
            num_prefill_requests: snapshot.num_prefill_requests as i32,
            sum_prefill_tokens: snapshot.sum_prefill_tokens as i64,
            var_prefill_length: snapshot.var_prefill_length,
            sum_prefill_kv_tokens: snapshot.sum_prefill_kv_tokens as i64,
            num_decode_requests: snapshot.num_decode_requests as i32,
            sum_decode_kv_tokens: snapshot.sum_decode_kv_tokens as i64,
            var_decode_kv_tokens: snapshot.var_decode_kv_tokens,
        },
        queued_requests: QueuedRequestMetricsSer {
            num_prefill_requests: snapshot.num_queued_prefill as i32,
            sum_prefill_tokens: snapshot.sum_queued_prefill_tokens as i64,
            var_prefill_length: snapshot.var_queued_prefill_length,
            num_decode_requests: snapshot.num_queued_decode as i32,
            sum_decode_kv_tokens: snapshot.sum_queued_decode_kv_tokens as i64,
            var_decode_kv_tokens: snapshot.var_queued_decode_kv_tokens,
        },
    };
    rmp_serde::to_vec_named(&metrics).map_err(|e| anyhow::anyhow!("FPM serialization failed: {e}"))
}

/// Live FPM sink that forwards snapshots to the `FpmDirectPublisher`'s
/// internal serialization pipeline via an mpsc channel.
struct LiveFpmSink {
    tx: mpsc::UnboundedSender<ForwardPassSnapshot>,
}

impl FpmSink for LiveFpmSink {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> Result<()> {
        self.tx
            .send(snapshot)
            .map_err(|_| anyhow::anyhow!("FPM publisher channel closed"))
    }
}

/// Direct FPM publisher for the mocker engine.
///
/// Unlike [`FpmEventRelay`] (which bridges raw ZMQ from a forked vLLM child
/// process), this publishes [`ForwardPassSnapshot`] data directly to the
/// event plane from in-process mocker schedulers.
pub struct FpmDirectPublisher {
    cancel: CancellationToken,
}

impl FpmDirectPublisher {
    /// Create and start a new direct publisher, returning per-dp-rank sink handles.
    ///
    /// Each returned [`FpmPublisher`] wraps a sink that feeds the shared
    /// serialization + event-plane publish pipeline. The scheduler passes
    /// one to each engine via the deferred-sink model.
    ///
    /// - `component`: Dynamo component (provides runtime + discovery scope).
    /// - `worker_id`: Unique worker identifier (typically `connection_id().to_string()`).
    /// - `dp_size`: Number of data-parallel ranks.
    pub async fn new(
        component: Component,
        worker_id: String,
        dp_size: u32,
    ) -> Result<(Self, Vec<FpmPublisher>)> {
        let rt = component.drt().runtime().secondary();
        let cancel = CancellationToken::new();

        let publisher = EventPublisher::for_component(&component, FPM_TOPIC).await?;

        // Shared channel: per-dp_rank serialization tasks send bytes here,
        // a single publisher task writes them to the event plane.
        let (pub_tx, mut pub_rx) = mpsc::unbounded_channel::<Vec<u8>>();

        // Publisher task
        let cancel_pub = cancel.clone();
        rt.spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = cancel_pub.cancelled() => break,
                    result = pub_rx.recv() => {
                        match result {
                            Some(payload) => {
                                if let Err(e) = publisher.publish_bytes(payload).await {
                                    tracing::warn!("FPM direct publisher: event plane publish failed: {e}");
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
            tracing::info!("FPM direct publisher: shutting down");
        });

        // Per-dp_rank: create internal channels, return sink handles.
        //
        // Each task forwards active-pass snapshots and emits periodic idle
        // heartbeats (zeroed snapshot, wall_time=0.0) when the scheduler is
        // idle, matching the Python `_FpmPublisherThread` contract.
        let mut fpm_publishers = Vec::with_capacity(dp_size as usize);
        for dp_rank in 0..dp_size {
            let (fpm_tx, mut fpm_rx) = mpsc::unbounded_channel();
            let sink = Arc::new(LiveFpmSink { tx: fpm_tx }) as Arc<dyn FpmSink>;
            fpm_publishers.push(FpmPublisher::new(Some(sink)));

            let pub_tx = pub_tx.clone();
            let worker_id = worker_id.clone();
            let cancel_ser = cancel.clone();

            rt.spawn(async move {
                let mut counter: i64 = 0;
                let heartbeat_sleep = tokio::time::sleep(IDLE_HEARTBEAT_INTERVAL);
                tokio::pin!(heartbeat_sleep);

                loop {
                    let snapshot = tokio::select! {
                        biased;
                        _ = cancel_ser.cancelled() => break,
                        result = fpm_rx.recv() => {
                            match result {
                                Some(snapshot) => {
                                    // Active pass — reset the heartbeat timer.
                                    heartbeat_sleep
                                        .as_mut()
                                        .reset(tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL);
                                    snapshot
                                }
                                None => break,
                            }
                        }
                        _ = &mut heartbeat_sleep => {
                            // No snapshot for IDLE_HEARTBEAT_INTERVAL — emit
                            // zeroed idle heartbeat, then reset for the next
                            // interval.
                            heartbeat_sleep
                                .as_mut()
                                .reset(tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL);
                            ForwardPassSnapshot::default()
                        }
                    };

                    counter += 1;
                    match serialize_fpm(&snapshot, &worker_id, dp_rank, counter) {
                        Ok(bytes) => {
                            let _ = pub_tx.send(bytes);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "FPM serialization failed for dp_rank {dp_rank}: {e}"
                            );
                        }
                    }
                }
            });
        }

        tracing::info!(
            worker_id = %worker_id,
            "FPM direct publisher started"
        );

        Ok((Self { cancel }, fpm_publishers))
    }

    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

impl Drop for FpmDirectPublisher {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::HashMap;

    /// Verify that serialize_fpm produces valid msgpack that round-trips
    /// through deserialization with the exact field names and values
    /// expected by the Python `ForwardPassMetrics` schema.
    #[test]
    fn test_serialize_fpm_round_trip() {
        let snapshot = ForwardPassSnapshot {
            num_prefill_requests: 2,
            sum_prefill_tokens: 256,
            var_prefill_length: 100.0,
            sum_prefill_kv_tokens: 64,
            num_decode_requests: 3,
            sum_decode_kv_tokens: 1024,
            var_decode_kv_tokens: 50.0,
            num_queued_prefill: 1,
            sum_queued_prefill_tokens: 128,
            var_queued_prefill_length: 0.0,
            num_queued_decode: 0,
            sum_queued_decode_kv_tokens: 0,
            var_queued_decode_kv_tokens: 0.0,
            wall_time_secs: 0.025,
        };

        let bytes = serialize_fpm(&snapshot, "worker-abc", 2, 42).unwrap();

        // Deserialize with matching struct (Deserialize derived) to verify
        // the wire format round-trips correctly.
        #[derive(Deserialize, Debug)]
        #[allow(dead_code)]
        struct ScheduledDe {
            num_prefill_requests: i32,
            sum_prefill_tokens: i64,
            var_prefill_length: f64,
            sum_prefill_kv_tokens: i64,
            num_decode_requests: i32,
            sum_decode_kv_tokens: i64,
            var_decode_kv_tokens: f64,
        }
        #[derive(Deserialize, Debug)]
        #[allow(dead_code)]
        struct QueuedDe {
            num_prefill_requests: i32,
            sum_prefill_tokens: i64,
            var_prefill_length: f64,
            num_decode_requests: i32,
            sum_decode_kv_tokens: i64,
            var_decode_kv_tokens: f64,
        }
        #[derive(Deserialize, Debug)]
        #[allow(dead_code)]
        struct FpmDe {
            version: i32,
            worker_id: String,
            dp_rank: i64,
            counter_id: i64,
            wall_time: f64,
            scheduled_requests: ScheduledDe,
            queued_requests: QueuedDe,
        }

        let decoded: FpmDe = rmp_serde::from_slice(&bytes).expect("round-trip decode failed");

        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.worker_id, "worker-abc");
        assert_eq!(decoded.dp_rank, 2);
        assert_eq!(decoded.counter_id, 42);
        assert!((decoded.wall_time - 0.025).abs() < 1e-10);

        assert_eq!(decoded.scheduled_requests.num_prefill_requests, 2);
        assert_eq!(decoded.scheduled_requests.sum_prefill_tokens, 256);
        assert!((decoded.scheduled_requests.var_prefill_length - 100.0).abs() < 1e-10);
        assert_eq!(decoded.scheduled_requests.sum_prefill_kv_tokens, 64);
        assert_eq!(decoded.scheduled_requests.num_decode_requests, 3);
        assert_eq!(decoded.scheduled_requests.sum_decode_kv_tokens, 1024);

        assert_eq!(decoded.queued_requests.num_prefill_requests, 1);
        assert_eq!(decoded.queued_requests.sum_prefill_tokens, 128);
        assert_eq!(decoded.queued_requests.num_decode_requests, 0);
    }

    /// Verify that worker_id and dp_rank can be extracted from the serialized
    /// bytes by deserializing into a flat HashMap, simulating the subscriber's
    /// `extract_fpm_key` approach of scanning the msgpack map for specific keys.
    #[test]
    fn test_serialize_fpm_extractable_key() {
        let snapshot = ForwardPassSnapshot {
            num_prefill_requests: 1,
            sum_prefill_tokens: 100,
            wall_time_secs: 0.01,
            ..Default::default()
        };

        let bytes = serialize_fpm(&snapshot, "my-worker-id", 7, 99).unwrap();

        // Deserialize only the top-level flat fields (nested maps become
        // opaque), matching the subscriber's partial-decode approach.
        #[derive(Deserialize)]
        struct PartialFpm {
            worker_id: String,
            dp_rank: i64,
        }
        let partial: PartialFpm = rmp_serde::from_slice(&bytes).expect("partial decode failed");
        assert_eq!(partial.worker_id, "my-worker-id");
        assert_eq!(partial.dp_rank, 7);
    }

    /// Verify that the idle heartbeat fires when no FPM arrives within
    /// IDLE_HEARTBEAT_INTERVAL. We replicate the per-dp_rank serialization
    /// task logic with real channels to test the timeout behavior.
    #[tokio::test]
    async fn test_idle_heartbeat_emits_zeroed_snapshot() {
        let (fpm_tx, mut fpm_rx) = mpsc::unbounded_channel::<ForwardPassSnapshot>();
        let (pub_tx, mut pub_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let worker_id = "test-worker".to_string();
        let dp_rank: u32 = 0;

        // Spawn the same task logic as FpmDirectPublisher
        tokio::spawn(async move {
            let mut counter: i64 = 0;
            let heartbeat_sleep = tokio::time::sleep(IDLE_HEARTBEAT_INTERVAL);
            tokio::pin!(heartbeat_sleep);

            loop {
                let snapshot = tokio::select! {
                    biased;
                    _ = cancel_clone.cancelled() => break,
                    result = fpm_rx.recv() => {
                        match result {
                            Some(snapshot) => {
                                heartbeat_sleep
                                    .as_mut()
                                    .reset(tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL);
                                snapshot
                            }
                            None => break,
                        }
                    }
                    _ = &mut heartbeat_sleep => {
                        heartbeat_sleep
                            .as_mut()
                            .reset(tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL);
                        ForwardPassSnapshot::default()
                    }
                };

                counter += 1;
                if let Ok(bytes) = serialize_fpm(&snapshot, &worker_id, dp_rank, counter) {
                    let _ = pub_tx.send(bytes);
                }
            }
        });

        // 1) Send an active snapshot first
        let active = ForwardPassSnapshot {
            num_prefill_requests: 2,
            sum_prefill_tokens: 100,
            wall_time_secs: 0.05,
            ..Default::default()
        };
        fpm_tx.send(active).unwrap();

        // Receive the active snapshot
        let bytes = tokio::time::timeout(Duration::from_secs(2), pub_rx.recv())
            .await
            .expect("timed out waiting for active FPM")
            .expect("channel closed");

        #[derive(Deserialize)]
        struct FpmWallTime {
            wall_time: f64,
        }
        let decoded: FpmWallTime = rmp_serde::from_slice(&bytes).expect("active FPM decode failed");
        assert!(
            decoded.wall_time > 0.0,
            "active snapshot should have wall_time > 0"
        );

        // 2) Now wait for the idle heartbeat (should arrive within ~1s)
        let heartbeat_bytes = tokio::time::timeout(Duration::from_secs(3), pub_rx.recv())
            .await
            .expect("timed out waiting for idle heartbeat")
            .expect("channel closed");

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct HeartbeatDe {
            wall_time: f64,
            counter_id: i64,
            worker_id: String,
        }
        let heartbeat: HeartbeatDe =
            rmp_serde::from_slice(&heartbeat_bytes).expect("heartbeat decode failed");
        assert_eq!(
            heartbeat.wall_time, 0.0,
            "idle heartbeat should have wall_time=0.0"
        );
        assert_eq!(heartbeat.counter_id, 2, "heartbeat is the second message");
        assert_eq!(heartbeat.worker_id, "test-worker");

        cancel.cancel();
    }

    /// Verify that active snapshots reset the heartbeat timer so heartbeats
    /// only fire after a period of true inactivity.
    #[tokio::test]
    async fn test_active_snapshots_suppress_heartbeat() {
        let (fpm_tx, mut fpm_rx) = mpsc::unbounded_channel::<ForwardPassSnapshot>();
        let (pub_tx, mut pub_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        tokio::spawn(async move {
            let mut counter: i64 = 0;
            let heartbeat_sleep = tokio::time::sleep(IDLE_HEARTBEAT_INTERVAL);
            tokio::pin!(heartbeat_sleep);

            loop {
                let snapshot = tokio::select! {
                    biased;
                    _ = cancel_clone.cancelled() => break,
                    result = fpm_rx.recv() => {
                        match result {
                            Some(snapshot) => {
                                heartbeat_sleep
                                    .as_mut()
                                    .reset(tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL);
                                snapshot
                            }
                            None => break,
                        }
                    }
                    _ = &mut heartbeat_sleep => {
                        heartbeat_sleep
                            .as_mut()
                            .reset(tokio::time::Instant::now() + IDLE_HEARTBEAT_INTERVAL);
                        ForwardPassSnapshot::default()
                    }
                };

                counter += 1;
                if let Ok(bytes) = serialize_fpm(&snapshot, "w", 0, counter) {
                    let _ = pub_tx.send(bytes);
                }
            }
        });

        // Send active snapshots every 500ms for 2 seconds — heartbeat should
        // NOT fire during this time since each send resets the timer.
        for _ in 0..4 {
            let active = ForwardPassSnapshot {
                num_decode_requests: 1,
                wall_time_secs: 0.01,
                ..Default::default()
            };
            fpm_tx.send(active).unwrap();
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Drain all active snapshots
        let mut active_count = 0;
        while let Ok(Some(bytes)) =
            tokio::time::timeout(Duration::from_millis(100), pub_rx.recv()).await
        {
            #[derive(Deserialize)]
            struct Wt {
                wall_time: f64,
            }
            let wt: Wt = rmp_serde::from_slice(&bytes).unwrap();
            assert!(
                wt.wall_time > 0.0,
                "all messages during active period should have wall_time > 0"
            );
            active_count += 1;
        }
        assert_eq!(
            active_count, 4,
            "should have received exactly 4 active snapshots"
        );

        // Now wait for the heartbeat (should fire ~1s after last active send)
        let heartbeat_bytes = tokio::time::timeout(Duration::from_secs(3), pub_rx.recv())
            .await
            .expect("timed out waiting for heartbeat after active period")
            .expect("channel closed");

        #[derive(Deserialize)]
        struct Wt2 {
            wall_time: f64,
        }
        let hb: Wt2 = rmp_serde::from_slice(&heartbeat_bytes).unwrap();
        assert_eq!(hb.wall_time, 0.0, "heartbeat should have wall_time=0.0");

        cancel.cancel();
    }

    /// Verify all 7 expected field names appear in scheduled_requests and
    /// 6 in queued_requests — matching the Python schema exactly.
    #[test]
    fn test_serialize_fpm_field_names() {
        let snapshot = ForwardPassSnapshot::default();
        let bytes = serialize_fpm(&snapshot, "", 0, 0).unwrap();

        // Deserialize the whole thing as nested HashMaps to inspect field names
        #[derive(Deserialize)]
        struct Wrapper {
            scheduled_requests: HashMap<String, serde_json::Value>,
            queued_requests: HashMap<String, serde_json::Value>,
        }
        let w: Wrapper = rmp_serde::from_slice(&bytes).expect("decode failed");

        let expected_sched = [
            "num_prefill_requests",
            "sum_prefill_tokens",
            "var_prefill_length",
            "sum_prefill_kv_tokens",
            "num_decode_requests",
            "sum_decode_kv_tokens",
            "var_decode_kv_tokens",
        ];
        for key in &expected_sched {
            assert!(
                w.scheduled_requests.contains_key(*key),
                "scheduled_requests missing field: {key}"
            );
        }
        assert_eq!(
            w.scheduled_requests.len(),
            expected_sched.len(),
            "scheduled_requests has unexpected extra fields"
        );

        let expected_queued = [
            "num_prefill_requests",
            "sum_prefill_tokens",
            "var_prefill_length",
            "num_decode_requests",
            "sum_decode_kv_tokens",
            "var_decode_kv_tokens",
        ];
        for key in &expected_queued {
            assert!(
                w.queued_requests.contains_key(*key),
                "queued_requests missing field: {key}"
            );
        }
        assert_eq!(
            w.queued_requests.len(),
            expected_queued.len(),
            "queued_requests has unexpected extra fields"
        );
    }
}
