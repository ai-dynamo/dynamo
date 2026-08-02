// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Correctness-only end-to-end CKF façade for `lib/bench` parity replays.
//!
//! Every configured lane is an actor-owned [`DcCkfState`] and serialized [`DcCkfPublisher`]. All
//! pool lanes feed one real indexer-domain-scoped global consumer. This keeps replay parity coverage after
//! retirement of the router-local hybrid without reintroducing a combined performance benchmark
//! or pretending that the façade is a production transport.

use std::collections::VecDeque;
use std::convert::Infallible;

use anyhow::{Context, Result, anyhow, bail};
use dynamo_kv_router::identity::{
    CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
};
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, PrefixSearchConfig};
use dynamo_kv_router::indexer::cuckoo::{
    ConsumerInstanceId, DcCkfDelta, DcCkfDeltaSink, DcCkfPublicationBatch, DcCkfPublishError,
    DcCkfPublisher, DcCkfState, GlobalCkfIndexer, GlobalCkfIngestOutcome, GlobalCkfLaneIngestor,
    GlobalCkfManifest, LaneLease, ProducerIdentity,
};
use dynamo_kv_router::protocols::{OverlapScores, RouterEvent, WorkerWithDpRank};
use rustc_hash::FxHashMap;

const MAX_LANES: usize = 16;
const PARITY_CONSUMER: ConsumerInstanceId = ConsumerInstanceId::new(1);
const PARITY_LAYOUT_GENERATION: u64 = 1;
const PARITY_ASSIGNMENT_EPOCH: u64 = 1;

#[derive(Debug, Clone, Copy)]
pub struct DirectCkfParityConfig {
    pub expected_blocks_per_pool: usize,
    pub publish_every_n_events: usize,
    pub kv_block_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectCkfParityDrain {
    pub terminal_sequences: Vec<u64>,
    pub ready_lanes: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectCkfParityMatchMode {
    FullMap,
    MaxDepthMatches,
}

#[derive(Debug, Default)]
struct ParityDeltaQueue {
    deltas: VecDeque<DcCkfDelta>,
}

impl DcCkfDeltaSink for ParityDeltaQueue {
    type Error = Infallible;

    fn enqueue(&mut self, delta: DcCkfDelta) -> std::result::Result<(), Self::Error> {
        self.deltas.push_back(delta);
        Ok(())
    }
}

#[derive(Debug)]
struct ProducerLane {
    owner: WorkerWithDpRank,
    state: DcCkfState,
    publisher: DcCkfPublisher<ParityDeltaQueue>,
    ingestor: GlobalCkfLaneIngestor,
    consumer_shadow: Box<[u64]>,
}

impl ProducerLane {
    fn submit(&mut self, event: RouterEvent) -> Result<()> {
        let outcome = self.state.apply_event(event);
        if let Some(error) = outcome.first_error() {
            bail!(
                "CKF parity producer {:?} rejected a replay block: {error}",
                self.owner
            );
        }
        if let Some(batch) = outcome.into_publication() {
            self.publish(batch)?;
        }
        Ok(())
    }

    fn exact_drain(&mut self) -> Result<u64> {
        // This correctness-only shim deliberately takes the actor barrier snapshot on every
        // query/drain. Production queries retain the documented weak-read contract.
        let (tail, producer_buckets) = self
            .state
            .barrier_snapshot()
            .context("failed to copy a CKF parity producer barrier snapshot")?;
        if let Some(batch) = tail {
            self.publish(batch)?;
        }

        while let Some(delta) = self.publisher.sink_mut().deltas.pop_front() {
            let outcome = self.ingestor.apply_delta(&delta);
            if !matches!(outcome, GlobalCkfIngestOutcome::DeltaApplied { .. }) {
                bail!("CKF parity consumer rejected an in-order delta: {outcome:?}");
            }
            for image in delta.images() {
                let bucket = image.bucket();
                let Some(shadow) = self.consumer_shadow.get_mut(bucket) else {
                    bail!("CKF parity delta references invalid bucket {bucket}");
                };
                *shadow = image.value();
            }
        }

        let terminal_sequence = self.publisher.last_sequence();
        let marker = self
            .publisher
            .exact_drain_marker()
            .ok_or_else(|| anyhow!("CKF parity publisher lost its lease before exact drain"))?;
        let outcome = self.ingestor.complete_drain(marker);
        if outcome
            != (GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: terminal_sequence,
            })
        {
            bail!("CKF parity exact drain did not converge: {outcome:?}");
        }
        if self.consumer_shadow.as_ref() != producer_buckets.as_ref() {
            bail!(
                "CKF parity producer and consumer reconstruction differ at terminal sequence \
                 {terminal_sequence}"
            );
        }
        Ok(terminal_sequence)
    }

    fn publish(&mut self, batch: DcCkfPublicationBatch) -> Result<()> {
        map_publish_result(self.publisher.publish(batch))
    }
}

/// A small synchronous façade used only by deterministic replay and parity tests.
#[derive(Debug)]
pub struct DirectCkfParityIndexer {
    indexer: GlobalCkfIndexer,
    lanes: Vec<ProducerLane>,
    configured_workers: Vec<WorkerWithDpRank>,
    worker_lanes: FxHashMap<WorkerWithDpRank, usize>,
}

impl DirectCkfParityIndexer {
    pub fn new(
        configured_workers: Vec<WorkerWithDpRank>,
        config: DirectCkfParityConfig,
    ) -> Result<Self> {
        if configured_workers.is_empty() || configured_workers.len() > MAX_LANES {
            bail!("CKF parity lanes must be within 1..={MAX_LANES}");
        }
        if config.publish_every_n_events == 0 {
            bail!("CKF parity publication cadence must be nonzero");
        }
        if config.kv_block_size == 0 {
            bail!("CKF parity KV block size must be nonzero");
        }
        let mut worker_lanes = FxHashMap::default();
        worker_lanes.try_reserve(configured_workers.len())?;
        for (lane, worker) in configured_workers.iter().copied().enumerate() {
            if worker_lanes.insert(worker, lane).is_some() {
                bail!("duplicate CKF parity worker identity {worker:?}");
            }
        }

        let mut ckf_config = CkfConfig::new(config.expected_blocks_per_pool);
        ckf_config.publish_every_n_events = config.publish_every_n_events;

        let mut states = Vec::with_capacity(configured_workers.len());
        for _ in &configured_workers {
            states.push(DcCkfState::new(ckf_config)?);
        }
        let format = states[0].format();
        let domain = IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([2; 16], IdentitySource::Explicit),
        );
        let mut pools = [None; MAX_LANES];
        for (lane, pool) in pools.iter_mut().enumerate().take(configured_workers.len()) {
            *pool = Some(PoolId::new(domain, DcId::new(lane as u64)));
        }
        let manifest = GlobalCkfManifest::new(PARITY_CONSUMER, domain, format, pools)?;
        let indexer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default())?;

        let mut lanes = Vec::with_capacity(configured_workers.len());
        for (lane, (owner, mut state)) in configured_workers.iter().copied().zip(states).enumerate()
        {
            let identity = ProducerIdentity::new(
                PoolId::new(domain, DcId::new(lane as u64)),
                lane as u64 + 1,
                PARITY_LAYOUT_GENERATION,
                format,
            );
            let lease = LaneLease::new(PARITY_CONSUMER, lane as u8, PARITY_ASSIGNMENT_EPOCH);
            let mut ingestor = indexer.claim_lane(lane)?;
            ingestor.assign(identity, lease)?;
            let mut publisher = DcCkfPublisher::new(identity, 0, ParityDeltaQueue::default());
            let snapshot = publisher
                .snapshot_after_barrier(&mut state, lease)
                .map_err(|error| anyhow!("initial CKF parity snapshot failed: {error:?}"))?;
            let outcome = ingestor.install_snapshot(&snapshot);
            if outcome != (GlobalCkfIngestOutcome::SnapshotInstalled { sequence: 0 }) {
                bail!("initial CKF parity snapshot was rejected: {outcome:?}");
            }

            lanes.push(ProducerLane {
                owner,
                state,
                publisher,
                ingestor,
                consumer_shadow: snapshot.buckets().to_vec().into_boxed_slice(),
            });
        }

        Ok(Self {
            indexer,
            lanes,
            configured_workers,
            worker_lanes,
        })
    }

    pub fn submit_event(&mut self, event: RouterEvent) -> Result<()> {
        let member = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let lane =
            self.worker_lanes.get(&member).copied().ok_or_else(|| {
                anyhow!("CKF parity event references unconfigured source {member:?}")
            })?;
        self.lanes[lane].submit(event)
    }

    pub fn exact_drain(&mut self) -> Result<DirectCkfParityDrain> {
        let terminal_sequences = self
            .lanes
            .iter_mut()
            .map(ProducerLane::exact_drain)
            .collect::<Result<Vec<_>>>()?;
        Ok(DirectCkfParityDrain {
            terminal_sequences,
            ready_lanes: self.indexer.ready_lanes(),
        })
    }

    pub fn find_matches(
        &mut self,
        sequence: &[dynamo_kv_router::LocalBlockHash],
    ) -> Result<OverlapScores> {
        self.find_matches_with_mode(sequence, DirectCkfParityMatchMode::FullMap)
    }

    pub fn find_matches_with_mode(
        &mut self,
        sequence: &[dynamo_kv_router::LocalBlockHash],
        mode: DirectCkfParityMatchMode,
    ) -> Result<OverlapScores> {
        self.exact_drain()?;
        let result = self.indexer.find_prefix_matches(sequence)?;
        let best_depth = result
            .lanes()
            .iter()
            .flatten()
            .map(|lane_match| lane_match.prefix_depth())
            .max()
            .unwrap_or(0);
        let mut overlap = OverlapScores::new();
        for lane_match in result.lanes().iter().flatten() {
            let depth = lane_match.prefix_depth();
            let selected = match mode {
                DirectCkfParityMatchMode::FullMap => depth > 0,
                DirectCkfParityMatchMode::MaxDepthMatches => depth > 0 && depth == best_depth,
            };
            if selected {
                overlap.scores.insert(
                    self.configured_workers[usize::from(lane_match.physical_lane())],
                    depth,
                );
            }
        }
        Ok(overlap)
    }

    pub fn ready_lanes(&self) -> u16 {
        self.indexer.ready_lanes()
    }
}

fn map_publish_result(
    result: std::result::Result<
        dynamo_kv_router::indexer::cuckoo::PublisherEmitOutcome,
        DcCkfPublishError<Infallible>,
    >,
) -> Result<()> {
    match result {
        Ok(_) => Ok(()),
        Err(DcCkfPublishError::Fenced(reason)) => {
            bail!("CKF parity publisher fenced: {reason:?}")
        }
        Err(error) => Err(anyhow!("CKF parity publication failed: {error:?}")),
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::LocalBlockHash;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData,
    };

    use super::*;

    fn stored(worker: u64, event_id: u64, hash: u64) -> RouterEvent {
        RouterEvent::new(
            worker,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        )
    }

    fn removed(worker: u64, event_id: u64, hash: u64) -> RouterEvent {
        RouterEvent::new(
            worker,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(hash)],
                }),
                dp_rank: 0,
            },
        )
    }

    #[test]
    fn submission_exact_drain_and_query_converge_across_lanes() -> Result<()> {
        let workers = vec![WorkerWithDpRank::new(10, 0), WorkerWithDpRank::new(20, 0)];
        let mut indexer = DirectCkfParityIndexer::new(
            workers.clone(),
            DirectCkfParityConfig {
                expected_blocks_per_pool: 32,
                publish_every_n_events: 16,
                kv_block_size: 512,
            },
        )?;

        indexer.submit_event(stored(10, 1, 7))?;
        indexer.submit_event(stored(20, 2, 7))?;
        let matches = indexer.find_matches(&[LocalBlockHash(7)])?;
        assert_eq!(matches.scores.get(&workers[0]), Some(&1));
        assert_eq!(matches.scores.get(&workers[1]), Some(&1));
        let drain = indexer.exact_drain()?;
        assert_eq!(drain.ready_lanes, 0b11);
        assert_eq!(drain.terminal_sequences, vec![1, 1]);

        indexer.submit_event(removed(10, 3, 7))?;
        let matches = indexer.find_matches(&[LocalBlockHash(7)])?;
        assert_eq!(matches.scores.get(&workers[0]), None);
        assert_eq!(matches.scores.get(&workers[1]), Some(&1));
        assert_eq!(indexer.exact_drain()?.terminal_sequences, vec![2, 1]);
        Ok(())
    }
}
