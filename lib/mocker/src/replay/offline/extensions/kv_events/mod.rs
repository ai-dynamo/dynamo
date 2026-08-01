// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_engine::KvEvent;
use anyhow::Context;
use dynamo_kv_router::protocols::{RouterEvent, StorageTier};

use crate::common::protocols::{MockEngineArgs, OutputSignal};
use crate::engine_observations::dynamo_kv_event;
use crate::loadgen::Trace;
use crate::replay::{
    ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest, ReplayWorkerArtifacts,
};
use crate::scheduler::RouterEventVisibility;
use aisimulate_replay::{
    CURRENT_REPLAY_SPEC_VERSION, EngineEventBatch, KvIngestEventEncoder, ProviderSpec,
    ReplayAdapters, ReplayArtifactKvEventVisibility, ReplayEngineObservation, ReplayRuntimeInput,
    ReplaySpec, ReplayTopology, Replayer, WorkerPoolSpec, WorkerStage,
};

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

    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn capture_engine_kv_events(stage: WorkerStage) -> bool {
        !matches!(stage, WorkerStage::Decode)
    }

    fn observe_engine_events(
        stage: WorkerStage,
        worker_id: usize,
        _dp_rank: u32,
        events: Vec<KvEvent>,
    ) -> Self::Batch {
        // Disaggregated decode placement is load-only: the established Dynamo
        // composition does not feed decode-pool KV mutations back into its
        // Router indexer. Aggregated and prefill placement still consume every
        // native event, now at the shared pass-completion boundary.
        if matches!(stage, WorkerStage::Decode) {
            return RouterEventBatch::default();
        }
        let worker_id = u64::try_from(worker_id)
            .expect("logical replay worker id must fit the Dynamo Router wire type");
        RouterEventBatch(
            events
                .into_iter()
                .map(|event| {
                    let (event, _) = dynamo_kv_event(event);
                    RouterEvent::with_storage_tier(worker_id, event, StorageTier::Device)
                })
                .collect(),
        )
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

    fn kv_ingest_event_count(batch: &Self::Batch) -> Option<usize> {
        Some(batch.0.len())
    }

    fn encode_kv_ingest(
        batch: &Self::Batch,
        encoder: &mut KvIngestEventEncoder<'_>,
    ) -> anyhow::Result<()> {
        encode_events(encoder, &batch.0)
    }
}

fn encode_events(
    encoder: &mut KvIngestEventEncoder<'_>,
    events: &[RouterEvent],
) -> anyhow::Result<()> {
    use dynamo_kv_router::protocols::KvCacheEventData;

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
    let (engine, factory) = crate::engine_adapter::aggregated_replay_setup(&args)?;
    let driver = trace.into_trace_driver_with_block_size(engine_block_size)?;
    let spec = ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec::default(),
        },
        engine: serde_json::to_value(engine)?,
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: false,
        sla: Default::default(),
        requests: Vec::new(),
    };
    let visibility = match router_event_visibility_override {
        None => ReplayArtifactKvEventVisibility::Native,
        Some(RouterEventVisibility::PassStart) => ReplayArtifactKvEventVisibility::PassStart,
        Some(RouterEventVisibility::PassEnd) => ReplayArtifactKvEventVisibility::PassEnd,
    };
    let (_, artifacts) = Replayer::new(spec, factory)?
        .with_runtime_input(ReplayRuntimeInput::Workload(driver))
        .run_with_artifacts(visibility)?;

    Ok(ReplayWorkerArtifacts {
        requests: artifacts
            .requests
            .into_iter()
            .map(|request| {
                Ok(ReplayTimedRequest {
                    uuid: request.request_id,
                    timestamp_us: timestamp_us_from_ms(request.observed_at_ms),
                    scheduled_ready_at_ms: request.scheduled_ready_at_ms,
                    input_length: request.input_length,
                    output_length: request.output_length,
                    replay_hashes: request
                        .replay_hashes
                        .context("offline artifacts require synthesized replay hashes")?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        output_signals: artifacts
            .outputs
            .into_iter()
            .map(|output| ReplayTimedOutputSignal {
                signal: OutputSignal {
                    uuid: output.request_id,
                    token_id: output.token_id,
                    completed: output.completed,
                    rejected: output.rejected,
                    handoff_delay_ms: None,
                },
                timestamp_us: timestamp_us_from_ms(output.observed_at_ms),
            })
            .collect(),
        kv_events: artifacts
            .kv_events
            .into_iter()
            .map(|event| ReplayTimedKvEvent {
                storage_tier: StorageTier::Device,
                event: dynamo_kv_event(event.event).0,
                timestamp_us: timestamp_us_from_ms(event.observed_at_ms),
            })
            .collect(),
    })
}
