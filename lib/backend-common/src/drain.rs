// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use crate::DisaggregationMode;
use crate::worker::EngineKind;

const POLL_INTERVAL: Duration = Duration::from_millis(500);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const CLEANUP_RESERVE: Duration = Duration::from_secs(5);

pub(crate) async fn until_idle(
    engine: &EngineKind,
    mode: DisaggregationMode,
    configured_poll_budget: Duration,
    shutdown_budget: Duration,
) {
    let start = Instant::now();
    let cleanup_reserve = CLEANUP_RESERVE.min(shutdown_budget / 2);
    let drain_deadline = start + shutdown_budget.saturating_sub(cleanup_reserve);
    let mut last_heartbeat = start;
    let mut announced = false;

    let remaining = drain_deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        tracing::warn!("drain: no budget available to stop engine admission");
        return;
    }
    let initial = match tokio::time::timeout(remaining, engine.begin_drain()).await {
        Ok(Ok(state)) => state,
        Ok(Err(error)) => {
            tracing::warn!(%error, "drain: failed to stop engine admission");
            // The engine implements drain but failed this attempt. Do not
            // confuse that with the legacy `None` (unsupported) result and
            // skip quiescence for aggregated/decode workers.
            Some(false)
        }
        Err(_) => {
            tracing::warn!("drain: timed out while stopping engine admission");
            return;
        }
    };

    if initial == Some(true) || (initial.is_none() && !mode.is_prefill()) {
        return;
    }

    let deadline = drain_deadline.min(Instant::now() + configured_poll_budget);
    let budget = deadline.saturating_duration_since(start).as_secs_f64();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!(
                "drain: timed out at {:.1}s; proceeding with cleanup",
                start.elapsed().as_secs_f64()
            );
            return;
        }
        match tokio::time::timeout(remaining, engine.is_quiescent()).await {
            Ok(Ok(Some(true))) => {
                if announced {
                    tracing::info!(
                        "drain: exited (quiescent, elapsed={:.1}s)",
                        start.elapsed().as_secs_f64()
                    );
                }
                return;
            }
            Ok(Ok(Some(false))) | Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                tracing::debug!(%error, "is_quiescent raised; treating as not quiescent")
            }
            Err(_) => {
                tracing::warn!("drain: quiescence poll exhausted the remaining budget");
                return;
            }
        }
        if !announced {
            tracing::info!(
                "drain: waiting for engine to quiesce (timeout={:.1}s)",
                budget
            );
            announced = true;
        }
        if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            tracing::info!(
                "drain: heartbeat (elapsed={:.1}s)",
                start.elapsed().as_secs_f64()
            );
            last_heartbeat = Instant::now();
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        tokio::time::sleep(remaining.min(POLL_INTERVAL)).await;
    }
}
