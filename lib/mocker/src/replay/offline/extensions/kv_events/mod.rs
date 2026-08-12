// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::RouterEvent;
pub(in crate::replay) use dynamo_kv_router::protocols::{KvCacheEventData, StorageTier};

use super::super::components::{
    AdmissionQueue, NoReplayMetadata, ObservedWorkerEvents, ReplayEngineObservation, ReplayMode,
    ReplayWorkerCore,
};
use super::super::core::EngineEventBatch;
use super::super::core::round_robin::PoolRoundRobinPlacement;
use super::super::disagg::DisaggRuntimeImpl;
use super::super::evidence::{
    KvIngestBoundary, KvIngestEventEncoder, WorkerPool, record_kv_ingest,
};
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::{Trace, WorkloadTerminalStatus};
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest,
    ReplayWorkerArtifacts, TraceCollector,
};
use crate::scheduler::RouterEventVisibility;
use std::collections::VecDeque;

#[derive(Debug, Default)]
pub(in crate::replay) struct RouterEventBatch(pub Vec<RouterEvent>);

impl EngineEventBatch for RouterEventBatch {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
pub(in crate::replay) struct RouterEventObservation;

impl ReplayEngineObservation for RouterEventObservation {
    type Batch = RouterEventBatch;

    const CAPTURE_RAW: bool = true;

    #[inline]
    fn take_pass_events(pass: &mut crate::scheduler::EnginePassResult) -> Self::Batch {
        Self::take(&mut pass.kv_events)
    }

    #[inline]
    fn take_command_events(effects: &mut crate::scheduler::SchedulerCommandEffects) -> Self::Batch {
        Self::take(&mut effects.kv_events)
    }

    #[inline]
    fn drain_worker_events(
        worker: &super::super::state::OfflineWorkerState,
    ) -> ObservedWorkerEvents<Self::Batch> {
        let mut events = worker.engine_core().drain_kv_events();
        ObservedWorkerEvents::from_events(Self::take(&mut events))
    }

    #[cfg(feature = "kvbm-offload")]
    #[inline]
    fn take_offload_events(effects: &mut crate::scheduler::OffloadTickEffects) -> Self::Batch {
        Self::take(&mut effects.kv_events)
    }

    fn stored_hashes(batch: &Self::Batch) -> Vec<u64> {
        batch
            .0
            .iter()
            .flat_map(|event| match &event.event.data {
                dynamo_kv_router::protocols::KvCacheEventData::Stored(store) => {
                    store.blocks.as_slice()
                }
                dynamo_kv_router::protocols::KvCacheEventData::Removed(_)
                | dynamo_kv_router::protocols::KvCacheEventData::Cleared => &[],
            })
            .map(|block| block.tokens_hash.0)
            .collect()
    }

    fn record_ingestion(
        batch: &Self::Batch,
        pool: WorkerPool,
        boundary: KvIngestBoundary,
        at_ms: f64,
    ) -> anyhow::Result<()> {
        record_kv_ingest(pool, boundary, at_ms, batch.0.len(), |encoder| {
            encode_events(encoder, &batch.0)
        })
    }
}

fn encode_events(
    encoder: &mut KvIngestEventEncoder<'_>,
    events: &[RouterEvent],
) -> anyhow::Result<()> {
    for event in events {
        let (tier_tag, tier_name) = storage_tier_identity(event.storage_tier);
        encoder.begin_event(
            event.worker_id,
            event.event.dp_rank,
            tier_tag,
            tier_name,
            event.event.event_id,
        );
        match &event.event.data {
            KvCacheEventData::Stored(stored) => {
                encoder.begin_kind(0, "stored");
                encoder.put_optional_u64(stored.parent_hash.map(|hash| hash.0));
                encoder.put_optional_u32(stored.start_position);
                encoder.put_len(stored.blocks.len(), "stored KV block count")?;
                encoder.add_blocks(stored.blocks.len(), "stored KV block count")?;
                for block in &stored.blocks {
                    encoder.put_u64(block.block_hash.0);
                    encoder.put_u64(block.tokens_hash.0);
                    match &block.mm_extra_info {
                        Some(extra) => {
                            encoder.put_u8(1);
                            encoder.put_len(extra.mm_objects.len(), "multimodal object count")?;
                            for object in &extra.mm_objects {
                                encoder.put_u64(object.mm_hash);
                                encoder.put_len(object.offsets.len(), "multimodal offset count")?;
                                for &(start, end) in &object.offsets {
                                    encoder.put_len(start, "multimodal start offset")?;
                                    encoder.put_len(end, "multimodal end offset")?;
                                }
                            }
                        }
                        None => encoder.put_u8(0),
                    }
                }
            }
            KvCacheEventData::Removed(removed) => {
                encoder.begin_kind(1, "removed");
                encoder.put_len(removed.block_hashes.len(), "removed KV block count")?;
                encoder.add_blocks(removed.block_hashes.len(), "removed KV block count")?;
                for hash in &removed.block_hashes {
                    encoder.put_u64(hash.0);
                }
            }
            KvCacheEventData::Cleared => encoder.begin_kind(2, "cleared"),
        }
    }
    Ok(())
}

fn storage_tier_identity(tier: StorageTier) -> (u8, &'static str) {
    match tier {
        StorageTier::Device => (0, "device"),
        StorageTier::HostPinned => (1, "host_pinned"),
        StorageTier::Disk => (2, "disk"),
        StorageTier::External => (3, "external"),
    }
}

impl RouterEventObservation {
    #[inline]
    fn take(events: &mut Vec<RouterEvent>) -> RouterEventBatch {
        RouterEventBatch(std::mem::take(events))
    }
}

pub(in crate::replay) type HandoffDisaggRuntime = DisaggRuntimeImpl<
    PoolRoundRobinPlacement<RouterEventBatch>,
    RouterEventObservation,
    NoReplayMetadata,
>;

impl
    DisaggRuntimeImpl<
        PoolRoundRobinPlacement<RouterEventBatch>,
        RouterEventObservation,
        NoReplayMetadata,
    >
{
    pub(in crate::replay) fn new_handoff_conformance(
        config: &OfflineDisaggReplayConfig,
        pending: VecDeque<DirectRequest>,
    ) -> anyhow::Result<Self> {
        Self::new_composed(
            config,
            AdmissionQueue::new_requests(pending, ReplayMode::Trace),
            false,
            true,
            true,
            |args, topology| {
                Ok(PoolRoundRobinPlacement::with_taints(
                    topology,
                    &args.worker_taints,
                ))
            },
            |args, topology| {
                Ok(PoolRoundRobinPlacement::with_taints(
                    topology,
                    &args.worker_taints,
                ))
            },
        )
    }
}

fn timestamp_us_from_ms(timestamp_ms: f64) -> u64 {
    if !timestamp_ms.is_finite() || timestamp_ms <= 0.0 {
        return 0;
    }

    (timestamp_ms * 1000.0) as u64
}

pub(in crate::replay) fn generate_trace_worker_artifacts_with_visibility(
    args: MockEngineArgs,
    trace: Trace,
    router_event_visibility_override: Option<RouterEventVisibility>,
) -> anyhow::Result<ReplayWorkerArtifacts> {
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let mut worker = ReplayWorkerCore::new_with_kv_capture(args, u64::default());
    let mut driver = trace.into_trace_driver_with_block_size(engine_block_size)?;
    let mut collector = TraceCollector::default();
    let mut artifacts = ReplayWorkerArtifacts::default();
    let mut current_time_ms = 0.0;

    while !driver.is_drained() || !worker.is_empty() {
        for ready_turn in driver.pop_ready(current_time_ms, usize::MAX) {
            let replay_hashes = ready_turn
                .replay_hashes
                .ok_or_else(|| anyhow::anyhow!("offline artifacts require synthesized hashes"))?;
            let output_length = ready_turn.request.effective_max_output_tokens();
            collector.on_arrival(
                ready_turn.request_uuid,
                ready_turn.scheduled_ready_at_ms,
                ready_turn.request.tokens.len(),
                output_length,
            );
            artifacts.requests.push(ReplayTimedRequest {
                uuid: ready_turn.request_uuid,
                timestamp_us: timestamp_us_from_ms(current_time_ms),
                scheduled_ready_at_ms: ready_turn.scheduled_ready_at_ms,
                input_length: ready_turn.request.tokens.len(),
                output_length,
                replay_hashes,
            });
            worker.receive(ready_turn.request);
        }

        if worker.is_empty() {
            let Some(next_ready_ms) = driver.next_ready_time_ms() else {
                break;
            };
            current_time_ms = next_ready_ms;
            continue;
        }

        let pass_start_ms = current_time_ms;
        let pass = worker.execute_pass(&mut collector, current_time_ms)?;
        current_time_ms = pass.end_ms;

        let router_event_visibility =
            router_event_visibility_override.unwrap_or(pass.router_event_visibility);
        let kv_event_timestamp_us = match router_event_visibility {
            RouterEventVisibility::PassStart => timestamp_us_from_ms(pass_start_ms),
            RouterEventVisibility::PassEnd => timestamp_us_from_ms(current_time_ms),
        };
        artifacts
            .kv_events
            .extend(pass.kv_events.into_iter().map(|event| ReplayTimedKvEvent {
                storage_tier: event.storage_tier,
                event: event.event,
                timestamp_us: kv_event_timestamp_us,
            }));

        let output_timestamp_us = timestamp_us_from_ms(current_time_ms);
        for signal in pass.output_signals {
            if let Some(token_id) = signal.token_id {
                driver.on_output_token(signal.uuid, token_id)?;
            }
            if signal.completed {
                let status = if signal.rejected {
                    WorkloadTerminalStatus::Rejected
                } else {
                    WorkloadTerminalStatus::Completed
                };
                let cascaded = driver.on_terminal(signal.uuid, current_time_ms, status)?;
                anyhow::ensure!(
                    cascaded.is_empty(),
                    "single-worker KV artifact capture does not accept agentic dependency cascades"
                );
            }
            artifacts.output_signals.push(ReplayTimedOutputSignal {
                signal,
                timestamp_us: output_timestamp_us,
            });
        }
    }

    Ok(artifacts)
}

#[cfg(all(test, feature = "replay-bench"))]
mod canonical_digest_tests {
    use dynamo_kv_router::protocols::{
        BlockExtraInfo, BlockMmObjectInfo, ExternalSequenceBlockHash, KvCacheEvent,
        KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };

    use super::*;
    use crate::replay::{ReplayCaptureOptions, ReplayDeterminism, with_runtime_evidence};

    fn batch() -> RouterEventBatch {
        RouterEventBatch(vec![
            RouterEvent {
                worker_id: 7,
                storage_tier: StorageTier::Device,
                residency_domain: Default::default(),
                event: KvCacheEvent {
                    event_id: 11,
                    dp_rank: 2,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: Some(ExternalSequenceBlockHash(101)),
                        start_position: Some(4),
                        blocks: vec![KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(202),
                            tokens_hash: LocalBlockHash(303),
                            mm_extra_info: Some(BlockExtraInfo {
                                mm_objects: vec![BlockMmObjectInfo {
                                    mm_hash: 404,
                                    offsets: vec![(1, 3), (5, 8)],
                                }],
                            }),
                        }],
                    }),
                },
            },
            RouterEvent {
                worker_id: 7,
                storage_tier: StorageTier::HostPinned,
                residency_domain: Default::default(),
                event: KvCacheEvent {
                    event_id: 12,
                    dp_rank: 2,
                    data: KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: vec![
                            ExternalSequenceBlockHash(202),
                            ExternalSequenceBlockHash(505),
                        ],
                    }),
                },
            },
            RouterEvent {
                worker_id: 8,
                storage_tier: StorageTier::Device,
                residency_domain: Default::default(),
                event: KvCacheEvent {
                    event_id: 13,
                    dp_rank: 0,
                    data: KvCacheEventData::Cleared,
                },
            },
        ])
    }

    fn capture(
        batch: &RouterEventBatch,
        pool: WorkerPool,
        boundary: KvIngestBoundary,
        at_ms: f64,
    ) -> crate::replay::KvIngestEvidence {
        let options = ReplayCaptureOptions {
            capture_canonical_evidence: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..ReplayCaptureOptions::default()
        };
        let (result, evidence) = with_runtime_evidence(options, || {
            RouterEventObservation::record_ingestion(batch, pool, boundary, at_ms)
        });
        result.unwrap();
        evidence.kv_ingest.unwrap()
    }

    #[test]
    fn hand_authored_batch_has_pinned_digest_and_exact_counters() {
        let evidence = capture(
            &batch(),
            WorkerPool::Decode,
            KvIngestBoundary::PassEnd,
            12.5,
        );

        assert_eq!(
            evidence.blake3_256,
            "3f185621bf6ef47c9cc7cc5fba7a123bdfe750efedde79df1915b331a801238b"
        );
        assert_eq!(evidence.batches, 1);
        assert_eq!(evidence.events, 3);
        assert_eq!(evidence.blocks, 3);
        assert_eq!(evidence.kind_counts["stored"], 1);
        assert_eq!(evidence.kind_counts["removed"], 1);
        assert_eq!(evidence.kind_counts["cleared"], 1);
        assert_eq!(evidence.tier_counts["device"], 2);
        assert_eq!(evidence.tier_counts["host_pinned"], 1);
        assert_eq!(evidence.pool_counts["decode"], 3);
        assert_eq!(evidence.boundaries["pass_end"].first_at_ms, 12.5);
        assert_eq!(evidence.boundaries["pass_end"].last_at_ms, 12.5);
    }

    #[test]
    fn digest_covers_content_time_order_boundary_pool_tier_and_multimodal_data() {
        let original = batch();
        let digest = |batch: &RouterEventBatch, pool, boundary, at_ms| {
            capture(batch, pool, boundary, at_ms).blake3_256
        };
        let baseline = digest(
            &original,
            WorkerPool::Decode,
            KvIngestBoundary::PassEnd,
            12.5,
        );

        let mut content = batch();
        let KvCacheEventData::Stored(stored) = &mut content.0[0].event.data else {
            unreachable!()
        };
        stored.blocks[0].tokens_hash = LocalBlockHash(999);

        let mut order = batch();
        order.0.swap(0, 1);

        let mut tier = batch();
        tier.0[0].storage_tier = StorageTier::Disk;

        let mut multimodal = batch();
        let KvCacheEventData::Stored(stored) = &mut multimodal.0[0].event.data else {
            unreachable!()
        };
        stored.blocks[0].mm_extra_info.as_mut().unwrap().mm_objects[0].offsets[0] = (2, 3);

        for changed in [
            digest(
                &content,
                WorkerPool::Decode,
                KvIngestBoundary::PassEnd,
                12.5,
            ),
            digest(
                &original,
                WorkerPool::Decode,
                KvIngestBoundary::PassEnd,
                12.75,
            ),
            digest(&order, WorkerPool::Decode, KvIngestBoundary::PassEnd, 12.5),
            digest(
                &original,
                WorkerPool::Decode,
                KvIngestBoundary::PassStart,
                12.5,
            ),
            digest(
                &original,
                WorkerPool::Prefill,
                KvIngestBoundary::PassEnd,
                12.5,
            ),
            digest(&tier, WorkerPool::Decode, KvIngestBoundary::PassEnd, 12.5),
            digest(
                &multimodal,
                WorkerPool::Decode,
                KvIngestBoundary::PassEnd,
                12.5,
            ),
        ] {
            assert_ne!(changed, baseline);
        }
    }
}
