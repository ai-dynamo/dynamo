// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Best-effort producer-side persistence for forward-pass metrics.
//!
//! This trace is an analysis aid, not a durable replacement for the FPM event
//! plane. Producers publish into a bounded in-process bus and never wait for
//! disk I/O. A slow sink can therefore drop trace records without delaying
//! inference or normal FPM delivery.

pub mod config;
mod sink;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use tokio_util::sync::CancellationToken;

use crate::telemetry::bus::TelemetryBus;

pub use config::{FpmTraceMode, FpmTracePolicy, is_enabled, policy};

#[derive(Clone, Debug)]
pub(crate) struct FpmTraceEvent {
    observed_at_unix_ms: u64,
    payload: Bytes,
}

static BUS: TelemetryBus<FpmTraceEvent> = TelemetryBus::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);

pub(crate) async fn init_from_env_with_shutdown(
    namespace: &str,
    component: &str,
    producer_id: &str,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    BUS.init(config::DEFAULT_CAPACITY);
    let started = sink::spawn_worker_from_env(namespace, component, producer_id, shutdown).await?;
    if started {
        ACTIVE.store(true, Ordering::Release);
    } else if !is_active() {
        // A previous initialization attempt failed. Keep tracing terminally
        // disabled for this process instead of retrying once per DP relay.
        return Ok(());
    }
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
    Ok(())
}

/// Enqueue one finalized FPM msgpack payload for best-effort persistence.
///
/// `TelemetryBus::publish` is synchronous and bounded; sink lag never applies
/// backpressure to the FPM producer or event-plane publisher.
pub(crate) fn publish_payload(payload: Bytes) {
    if !is_active() {
        return;
    }
    let observed_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default();
    BUS.publish(FpmTraceEvent {
        observed_at_unix_ms,
        payload,
    });
}

pub(crate) fn is_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

fn subscribe() -> tokio::sync::broadcast::Receiver<FpmTraceEvent> {
    BUS.subscribe()
}
