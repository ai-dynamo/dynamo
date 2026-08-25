// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use super::super::identity::DcRelayIdentity;
use super::super::load::PoolLoadSnapshot;
use super::super::protocol as proto;
use super::identity::{producer_to_wire, relay_identity_to_wire, unix_timestamp};
use super::source::WanPublicationSource;

#[derive(Clone)]
pub(crate) struct LoadUpdateHub {
    updates: broadcast::Sender<proto::KvPoolLoadUpdate>,
    current: Arc<RwLock<proto::KvPoolLoadUpdate>>,
}

impl LoadUpdateHub {
    pub(crate) fn new(source: &WanPublicationSource, window: Duration, capacity: usize) -> Self {
        let (updates, _) = broadcast::channel(capacity);
        let current = load_update(source.relay_identity(), source.load_snapshots(), window, 0);
        Self {
            updates,
            current: Arc::new(RwLock::new(current)),
        }
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<proto::KvPoolLoadUpdate> {
        self.updates.subscribe()
    }

    pub(crate) fn current(&self) -> proto::KvPoolLoadUpdate {
        self.current.read().clone()
    }

    fn publish(&self, update: proto::KvPoolLoadUpdate) {
        *self.current.write() = update.clone();
        let _ = self.updates.send(update);
    }
}

pub(crate) async fn run_load_publisher(
    source: WanPublicationSource,
    window: Duration,
    updates: LoadUpdateHub,
    cancel: CancellationToken,
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
        let snapshots = source.load_snapshots();
        let update = load_update(source.relay_identity(), snapshots, window, sequence);
        updates.publish(update);
    }
}

fn load_update(
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
        window_ms: u64::try_from(window.as_millis()).unwrap_or(u64::MAX),
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
