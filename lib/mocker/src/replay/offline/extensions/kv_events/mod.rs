// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
pub(in crate::replay) use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_kv_router::protocols::RouterEvent;
#[cfg(all(test, feature = "kvbm-offload"))]
pub(in crate::replay) use dynamo_kv_router::protocols::StorageTier;

use super::super::components::{
    AdmissionQueue, NoReplayMetadata, ReplayEngineObservation, ReplayMode, ReplayWorkerCore,
};
use super::super::core::EngineEventBatch;
use super::super::core::round_robin::PoolRoundRobinPlacement;
use super::super::disagg::DisaggRuntimeImpl;
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::Trace;
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
    fn drain_worker_events(worker: &super::super::state::OfflineWorkerState) -> Self::Batch {
        let mut events = worker.engine_core().drain_kv_events();
        Self::take(&mut events)
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
            |_, topology| Ok(PoolRoundRobinPlacement::new(topology)),
            |_, topology| Ok(PoolRoundRobinPlacement::new(topology)),
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
            collector.on_arrival(
                ready_turn.request_uuid,
                ready_turn.scheduled_ready_at_ms,
                ready_turn.request.tokens.len(),
                ready_turn.request.max_output_tokens,
            );
            artifacts.requests.push(ReplayTimedRequest {
                uuid: ready_turn.request_uuid,
                timestamp_us: timestamp_us_from_ms(current_time_ms),
                scheduled_ready_at_ms: ready_turn.scheduled_ready_at_ms,
                input_length: ready_turn.request.tokens.len(),
                output_length: ready_turn.request.max_output_tokens,
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
        let pass = worker.execute_pass(&mut collector, current_time_ms);
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
                driver.on_terminal(signal.uuid, current_time_ms, signal.rejected)?;
            }
            artifacts.output_signals.push(ReplayTimedOutputSignal {
                signal,
                timestamp_us: output_timestamp_us,
            });
        }
    }

    Ok(artifacts)
}
