// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-neutral observations captured from a shared Replay execution.

use std::sync::{Arc, Mutex};

use crate::engine::KvEvent;
use uuid::Uuid;

use crate::replay::loadgen::ReplayRequestHashes;
use crate::replay::{ReplayError, ReplayResult};

/// Timestamp policy used when rendering native KV observations as replay
/// artifacts.
///
/// `Native` preserves each backend's publication boundary. The two explicit
/// variants exist for parity fixtures that intentionally normalize all events
/// to one side of a pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReplayArtifactKvEventVisibility {
    #[default]
    Native,
    PassStart,
    PassEnd,
}

/// One request released by Replay's workload source.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayArtifactRequest {
    pub request_id: Uuid,
    /// Virtual time at which Replay made the request visible to the engine.
    pub observed_at_ms: f64,
    /// Workload-authored ready time, which can precede `observed_at_ms` while
    /// admission or workload scheduling is gated.
    pub scheduled_ready_at_ms: f64,
    pub input_length: usize,
    pub output_length: usize,
    pub replay_hashes: Option<ReplayRequestHashes>,
}

/// One client-visible output released at a pass-completion boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayArtifactOutput {
    pub request_id: Uuid,
    pub token_id: Option<u32>,
    pub completed: bool,
    pub rejected: bool,
    pub observed_at_ms: f64,
}

/// One native G1 observation at the timestamp selected for the artifact run.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayArtifactKvEvent {
    pub event: KvEvent,
    pub observed_at_ms: f64,
}

/// Optional detailed observations produced by the same virtual-clock/pass
/// loop that generates the normal replay report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayArtifacts {
    pub requests: Vec<ReplayArtifactRequest>,
    pub outputs: Vec<ReplayArtifactOutput>,
    pub kv_events: Vec<ReplayArtifactKvEvent>,
}

/// Shared sink retained by the caller while [`crate::replay::Replayer`] owns the run.
///
/// This is intentionally a concrete Replay-owned sink instead of a plugin ABI:
/// it carries only neutral request/output/native-KV values and introduces no
/// dependency on an adapter runtime.
#[derive(Debug, Clone)]
pub(crate) struct ReplayArtifactSink {
    visibility: ReplayArtifactKvEventVisibility,
    state: Arc<Mutex<ReplayArtifactState>>,
}

#[derive(Debug, Default)]
struct ReplayArtifactState {
    artifacts: ReplayArtifacts,
    deferred_pass_start_kv_events: Vec<KvEvent>,
}

impl ReplayArtifactSink {
    pub(crate) fn new(visibility: ReplayArtifactKvEventVisibility) -> Self {
        Self {
            visibility,
            state: Arc::new(Mutex::new(ReplayArtifactState::default())),
        }
    }

    pub(crate) fn record_request(&self, request: ReplayArtifactRequest) -> ReplayResult<()> {
        self.lock()?.artifacts.requests.push(request);
        Ok(())
    }

    pub(crate) fn record_outputs(
        &self,
        observed_at_ms: f64,
        outputs: &[crate::replay::protocol::OutputSignal],
    ) -> ReplayResult<()> {
        self.lock()?
            .artifacts
            .outputs
            .extend(outputs.iter().map(|output| ReplayArtifactOutput {
                request_id: output.uuid,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
                observed_at_ms,
            }));
        Ok(())
    }

    pub(crate) fn record_pass_start_kv_events(
        &self,
        pass_start_ms: f64,
        pass_start_events: &[KvEvent],
    ) -> ReplayResult<()> {
        let mut state = self.lock()?;
        match self.visibility {
            ReplayArtifactKvEventVisibility::Native
            | ReplayArtifactKvEventVisibility::PassStart => {
                state
                    .artifacts
                    .kv_events
                    .extend(
                        pass_start_events
                            .iter()
                            .cloned()
                            .map(|event| ReplayArtifactKvEvent {
                                event,
                                observed_at_ms: pass_start_ms,
                            }),
                    )
            }
            ReplayArtifactKvEventVisibility::PassEnd => {
                if !state.deferred_pass_start_kv_events.is_empty() {
                    return Err(ReplayError::Invariant(
                        "artifact sink observed overlapping passes".to_string(),
                    ));
                }
                state
                    .deferred_pass_start_kv_events
                    .extend_from_slice(pass_start_events);
            }
        }
        Ok(())
    }

    pub(crate) fn record_pass_completion_kv_events(
        &self,
        pass_start_ms: f64,
        pass_end_ms: f64,
        pass_end_events: &[KvEvent],
    ) -> ReplayResult<()> {
        let mut state = self.lock()?;
        let timestamp_ms = match self.visibility {
            ReplayArtifactKvEventVisibility::Native | ReplayArtifactKvEventVisibility::PassEnd => {
                pass_end_ms
            }
            ReplayArtifactKvEventVisibility::PassStart => pass_start_ms,
        };
        if self.visibility == ReplayArtifactKvEventVisibility::PassEnd {
            let pass_start_events = std::mem::take(&mut state.deferred_pass_start_kv_events);
            state
                .artifacts
                .kv_events
                .extend(
                    pass_start_events
                        .into_iter()
                        .map(|event| ReplayArtifactKvEvent {
                            event,
                            observed_at_ms: timestamp_ms,
                        }),
                );
        }
        state
            .artifacts
            .kv_events
            .extend(
                pass_end_events
                    .iter()
                    .cloned()
                    .map(|event| ReplayArtifactKvEvent {
                        event,
                        observed_at_ms: timestamp_ms,
                    }),
            );
        Ok(())
    }

    pub(crate) fn take(&self) -> ReplayResult<ReplayArtifacts> {
        Ok(std::mem::take(&mut self.lock()?.artifacts))
    }

    fn lock(&self) -> ReplayResult<std::sync::MutexGuard<'_, ReplayArtifactState>> {
        self.state.lock().map_err(|_| {
            ReplayError::Invariant("replay artifact sink lock was poisoned".to_string())
        })
    }
}
