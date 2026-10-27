// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::{broadcast, watch};
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
    /// `None` until the first window, so no subscriber sees a placeholder
    /// that would read as every pool gone.
    current: Arc<RwLock<Option<T>>>,
}

impl<T: LoadWindow> LoadUpdateHub<T> {
    pub(super) fn new(capacity: usize) -> Self {
        let (updates, _) = broadcast::channel(capacity);
        Self {
            updates,
            current: Arc::new(RwLock::new(None)),
        }
    }

    pub(super) fn subscribe(&self) -> broadcast::Receiver<T> {
        self.updates.subscribe()
    }

    pub(super) fn current(&self) -> Option<T> {
        self.current.read().clone()
    }

    fn publish(&self, update: T) {
        *self.current.write() = Some(update.clone());
        let _ = self.updates.send(update);
    }
}

/// Publishes `window_update(sequence)` every `window`, starting at sequence 1.
pub(super) async fn run_load_publisher<T: LoadWindow>(
    window: Duration,
    updates: LoadUpdateHub<T>,
    cancel: CancellationToken,
    mut window_update: impl FnMut(u64) -> T,
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
        sequence = next_sequence(sequence)?;
        updates.publish(window_update(sequence));
    }
}

/// Publishes `window_update(snapshot, sequence)` for every snapshot `source`
/// produces, starting at sequence 1, so windows follow the source's cadence.
/// Fails once the source's sender is dropped, unless already cancelled.
pub(super) async fn run_snapshot_publisher<S: Clone, T: LoadWindow>(
    mut source: watch::Receiver<Option<S>>,
    updates: LoadUpdateHub<T>,
    cancel: CancellationToken,
    mut window_update: impl FnMut(&S, u64) -> T,
) -> anyhow::Result<()> {
    let mut sequence = 0u64;
    loop {
        let changed = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            changed = source.changed() => changed,
        };
        if changed.is_err() {
            if cancel.is_cancelled() {
                return Ok(());
            }
            anyhow::bail!("KV Relay load source stopped");
        }
        // Clone out of the borrow so the source's sender is not blocked while
        // the window is built.
        let Some(snapshot) = source.borrow_and_update().clone() else {
            continue;
        };
        sequence = next_sequence(sequence)?;
        updates.publish(window_update(&snapshot, sequence));
    }
}

fn next_sequence(sequence: u64) -> anyhow::Result<u64> {
    sequence
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("KV Relay load window sequence exhausted"))
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

    /// One window per snapshot, none before the first; a stopped source
    /// fails the publisher unless it was cancelled.
    #[tokio::test]
    async fn snapshot_publisher_follows_its_source() {
        let window = Duration::from_millis(10);
        let relay = DcRelayIdentity::new(1, 2);
        let hub = LoadUpdateHub::new(4);
        let (source, receiver) = watch::channel(None::<Vec<PoolLoadSnapshot>>);
        let publisher = tokio::spawn(run_snapshot_publisher(
            receiver,
            hub.clone(),
            CancellationToken::new(),
            move |snapshots: &Vec<PoolLoadSnapshot>, sequence| {
                load_update(relay, snapshots.clone(), window, sequence)
            },
        ));
        let mut updates = hub.subscribe();

        source.send_replace(Some(Vec::new()));
        assert_eq!(updates.recv().await.unwrap().window_sequence, 1);
        source.send_replace(Some(Vec::new()));
        assert_eq!(updates.recv().await.unwrap().window_sequence, 2);

        drop(source);
        let error = publisher.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("stopped"));
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
