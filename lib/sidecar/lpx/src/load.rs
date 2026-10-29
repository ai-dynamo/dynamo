// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine's load from `Control.GetLoad`, published as the router's load
//! signal and the worker's KV cache gauges.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dynamo_backend_common::{ComponentSnapshot, DynamoError, SnapshotPublisher};
use tokio::time::{Instant, MissedTickBehavior, sleep_until};
use tokio_util::sync::CancellationToken;

use crate::client::{OpenEngineClient, protocol_error};
use crate::pb;

/// The one data-parallel rank the sidecar registers.
pub(crate) const RANK: u32 = 0;
/// How long the load may go without a valid answer before the worker
/// withdraws.
pub(crate) const STALENESS_LIMIT: Duration = Duration::from_secs(30);
pub(crate) const GET_LOAD_TIMEOUT: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_AGE: Duration = Duration::from_secs(5);
const MAX_CLOCK_SKEW: Duration = Duration::from_secs(1);

/// Rank 0's load from a `GetLoad` answer. It must be fresh and carry exactly
/// that rank, with no more blocks used than there are.
pub(crate) fn snapshot(peer: &str, load: pb::LoadInfo) -> Result<ComponentSnapshot, DynamoError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let fresh = load.timestamp_unix_nanos.map(u128::from).is_some_and(|at| {
        now.saturating_sub(at) <= MAX_AGE.as_nanos()
            && at.saturating_sub(now) <= MAX_CLOCK_SKEW.as_nanos()
    });
    if !fresh {
        return Err(protocol_error(peer, "GetLoad returned no fresh load"));
    }
    match load.ranks.as_slice() {
        [rank] => match (
            rank.data_parallel_rank,
            rank.used_kv_blocks,
            rank.total_kv_blocks,
        ) {
            (Some(RANK), Some(used), Some(total)) if total > 0 && used <= total => {
                Ok(ComponentSnapshot {
                    kv_used_blocks: used,
                    kv_total_blocks: total,
                    gpu_cache_usage: used as f32 / total as f32,
                    kv_cache_hit_rate: None,
                    dp_rank: RANK,
                })
            }
            _ => Err(protocol_error(
                peer,
                "GetLoad returned invalid rank 0 counters",
            )),
        },
        ranks => Err(protocol_error(
            peer,
            format!("GetLoad returned {} ranks for one", ranks.len()),
        )),
    }
}

/// Publishes the engine's load every [`POLL_INTERVAL`] until `shutdown`. Once
/// no poll has succeeded for [`STALENESS_LIMIT`], it cancels `lost` and stops.
pub(crate) async fn poll(
    client: Arc<OpenEngineClient>,
    publisher: Arc<SnapshotPublisher>,
    shutdown: CancellationToken,
    lost: CancellationToken,
) {
    let mut interval = tokio::time::interval(POLL_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut stale_at = Instant::now() + STALENESS_LIMIT;
    let mut warned = false;
    loop {
        let load = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return,
            _ = sleep_until(stale_at) => {
                lost.cancel();
                return;
            }
            load = async {
                interval.tick().await;
                client.load(Instant::now() + GET_LOAD_TIMEOUT).await
            } => load,
        };
        let polled = load.and_then(|load| {
            let load = load
                .ok_or_else(|| protocol_error(client.peer(), "GetLoad answered UNIMPLEMENTED"))?;
            snapshot(client.peer(), load)
        });
        match polled {
            Ok(snapshot) => {
                warned = false;
                stale_at = Instant::now() + STALENESS_LIMIT;
                publisher.publish(RANK, snapshot);
            }
            // Escaped: the message can carry the engine's status text.
            Err(error) if !warned => {
                warned = true;
                tracing::warn!(
                    error = %error.to_string().escape_debug(),
                    "GetLoad poll failed; suppressing repeated warnings"
                );
            }
            Err(error) => {
                tracing::debug!(error = %error.to_string().escape_debug(), "GetLoad poll failed")
            }
        }
    }
}
