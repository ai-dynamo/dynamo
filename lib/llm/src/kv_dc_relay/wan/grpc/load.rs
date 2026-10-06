// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use super::identity::{producer_to_wire, relay_identity_to_wire, unix_timestamp};
use super::protocol as proto;
use crate::kv_dc_relay::identity::DcRelayIdentity;
use crate::kv_dc_relay::load::PoolLoadSnapshot;

/// A complete load window published by [`run_load_publisher`].
pub(super) trait LoadWindow: Clone + Send + Sync + 'static {
    fn window_sequence(&self) -> u64;
}

impl LoadWindow for proto::KvPoolLoadUpdate {
    fn window_sequence(&self) -> u64 {
        self.window_sequence
    }
}

impl LoadWindow for proto::ServingLoadUpdate {
    fn window_sequence(&self) -> u64 {
        self.window_sequence
    }
}

/// Latest window plus bounded fanout of the following ones.
#[derive(Clone)]
pub(super) struct LoadUpdateHub<T> {
    updates: broadcast::Sender<T>,
    current: Arc<RwLock<T>>,
}

impl<T: LoadWindow> LoadUpdateHub<T> {
    pub(super) fn new(initial: T, capacity: usize) -> Self {
        let (updates, _) = broadcast::channel(capacity);
        Self {
            updates,
            current: Arc::new(RwLock::new(initial)),
        }
    }

    pub(super) fn subscribe(&self) -> broadcast::Receiver<T> {
        self.updates.subscribe()
    }

    pub(super) fn current(&self) -> T {
        self.current.read().clone()
    }

    fn publish(&self, update: T) {
        *self.current.write() = update.clone();
        let _ = self.updates.send(update);
    }
}

/// Publishes `window_update(sequence)` every `window`, starting at sequence 1.
/// A `window_update` error stops the publisher unless it is already cancelled.
pub(super) async fn run_load_publisher<T: LoadWindow>(
    window: Duration,
    updates: LoadUpdateHub<T>,
    cancel: CancellationToken,
    mut window_update: impl FnMut(u64) -> anyhow::Result<T>,
) -> anyhow::Result<()> {
    let first_tick = tokio::time::Instant::now() + window;
    let mut tick = tokio::time::interval_at(first_tick, window);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut sequence = 0u64;
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            _ = tick.tick() => {}
        }
        sequence = sequence
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("KV Relay load window sequence exhausted"))?;
        match window_update(sequence) {
            Ok(update) => updates.publish(update),
            Err(_) if cancel.is_cancelled() => return Ok(()),
            Err(error) => return Err(error),
        }
    }
}

pub(super) fn window_ms(window: Duration) -> u64 {
    u64::try_from(window.as_millis()).unwrap_or(u64::MAX)
}

pub(super) fn load_update(
    relay: DcRelayIdentity,
    snapshots: Vec<PoolLoadSnapshot>,
    window: Duration,
    sequence: u64,
) -> proto::KvPoolLoadUpdate {
    proto::KvPoolLoadUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        window_sequence: sequence,
        observed_ms: unix_timestamp::<1_000>(),
        window_ms: window_ms(window),
        pools: snapshots.into_iter().map(load_entry_to_wire).collect(),
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

fn load_entry_to_wire(snapshot: PoolLoadSnapshot) -> proto::KvPoolLoadEntry {
    proto::KvPoolLoadEntry {
        producer: Some(producer_to_wire(snapshot.producer)),
        kv_used_blocks: snapshot.kv_used_blocks.unwrap_or_default(),
        total_kv_blocks: snapshot.total_kv_blocks.unwrap_or_default(),
        kv_observed_ranks: saturating_u32(snapshot.kv_observed_ranks),
        kv_expected_ranks: saturating_u32(snapshot.kv_expected_ranks),
    }
}

fn saturating_u32(value: usize) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState, ProducerIdentity};

    use super::*;

    fn producer() -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    #[tokio::test(start_paused = true)]
    async fn publisher_stops_on_window_errors_unless_cancelled() {
        let window = Duration::from_millis(10);
        let initial = load_update(DcRelayIdentity::new(1, 2), Vec::new(), window, 0);
        let hub = LoadUpdateHub::new(initial.clone(), 4);
        let cancel = CancellationToken::new();
        let result = run_load_publisher(window, hub.clone(), cancel.clone(), |sequence| {
            anyhow::ensure!(sequence < 3, "source closed");
            Ok(proto::KvPoolLoadUpdate {
                window_sequence: sequence,
                ..initial.clone()
            })
        })
        .await;
        assert!(result.unwrap_err().to_string().contains("source closed"));
        assert_eq!(hub.current().window_sequence, 2);

        cancel.cancel();
        run_load_publisher(window, hub, cancel, |_| anyhow::bail!("source closed"))
            .await
            .unwrap();
    }

    #[test]
    fn saturated_main_aggregates_are_forwarded_without_reinterpretation() {
        let entry = load_entry_to_wire(PoolLoadSnapshot {
            producer: producer(),
            kv_used_blocks: Some(u64::MAX),
            total_kv_blocks: Some(u64::MAX),
            kv_observed_ranks: 2,
            kv_capacity_ranks: 2,
            kv_expected_ranks: 2,
        });

        assert_eq!(entry.kv_used_blocks, u64::MAX);
        assert_eq!(entry.total_kv_blocks, u64::MAX);
        assert_eq!(entry.kv_observed_ranks, 2);
        assert_eq!(entry.kv_expected_ranks, 2);
    }
}
