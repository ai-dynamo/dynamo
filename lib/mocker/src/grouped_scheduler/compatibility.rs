// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo request and handoff compatibility state for the grouped live engine.

use std::collections::HashMap;

use aisimulate_core::engine::{
    HandoffId as EngineHandoffId, LifecycleEvent, Output, Request,
    TransferTimingMode as EngineTransferTimingMode,
};
use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::common::handoff::{HandoffId as DynamoHandoffId, HandoffTransferTiming};
use crate::common::protocols::{DirectRequest, KvTransferTimingMode, MockEngineArgs, OutputSignal};
use crate::common::utils::compute_prefill_handoff_delay_ms;
use crate::scheduler::SchedulerLifecycleEvent;

#[derive(Clone, Copy)]
pub(super) enum Cleanup {
    Request(Uuid),
    SourceHandoff(DynamoHandoffId),
    DestinationHandoff(DynamoHandoffId),
}

pub(super) struct CompatibilityState {
    args: MockEngineArgs,
    request_prompt_lengths: Mutex<HashMap<Uuid, usize>>,
    handoffs: Mutex<HandoffMap>,
}

impl CompatibilityState {
    pub(super) fn new(args: MockEngineArgs) -> Self {
        Self {
            args,
            request_prompt_lengths: Mutex::new(HashMap::new()),
            handoffs: Mutex::new(HandoffMap::default()),
        }
    }

    pub(super) fn native_request(&self, request: DirectRequest) -> Request {
        let request_id = request.uuid.unwrap_or_else(Uuid::new_v4);
        self.request_prompt_lengths
            .lock()
            .insert(request_id, request.tokens.len());
        Request {
            request_id,
            tokens: request.tokens,
            max_output_tokens: request.max_output_tokens,
            output_token_ids: request.output_token_ids,
        }
    }

    pub(super) fn engine_handoff(&self, handoff_id: DynamoHandoffId) -> Result<EngineHandoffId> {
        self.handoffs.lock().native(handoff_id)
    }

    pub(super) fn mark_source(&self, handoff_id: DynamoHandoffId, request_id: Uuid) {
        self.handoffs.lock().mark_source(handoff_id, request_id);
    }

    pub(super) fn mark_destination(&self, handoff_id: DynamoHandoffId, request_id: Uuid) {
        self.handoffs
            .lock()
            .mark_destination(handoff_id, request_id);
    }

    fn dynamo_handoff(&self, handoff_id: EngineHandoffId) -> Result<DynamoHandoffId> {
        self.handoffs.lock().dynamo(handoff_id)
    }

    pub(super) fn apply_cleanup(&self, cleanup: Cleanup) {
        match cleanup {
            Cleanup::Request(request_id) => {
                self.request_prompt_lengths.lock().remove(&request_id);
                self.handoffs.lock().cancel_request(request_id);
            }
            Cleanup::SourceHandoff(handoff_id) => self.handoffs.lock().finish_source(handoff_id),
            Cleanup::DestinationHandoff(handoff_id) => {
                self.handoffs.lock().finish_destination(handoff_id)
            }
        }
    }

    pub(super) fn output_signal(&self, output: Output) -> OutputSignal {
        let prompt_len = output.completed.then(|| {
            self.request_prompt_lengths
                .lock()
                .remove(&output.request_id)
        });
        if output.completed {
            self.handoffs
                .lock()
                .finish_destination_request(output.request_id);
        }
        OutputSignal {
            uuid: output.request_id,
            token_id: output.token_id,
            completed: output.completed,
            rejected: output.rejected,
            handoff_delay_ms: prompt_len.flatten().and_then(|prompt_len| {
                compute_prefill_handoff_delay_ms(
                    self.args.worker_type,
                    output.completed,
                    prompt_len,
                    self.args.kv_transfer_bandwidth,
                    self.args.kv_bytes_per_token,
                )
            }),
        }
    }

    pub(super) fn lifecycle_event(&self, event: LifecycleEvent) -> Result<SchedulerLifecycleEvent> {
        Ok(match event {
            LifecycleEvent::SourceHeld {
                handoff_id,
                request_id,
                transfer_timing,
            } => SchedulerLifecycleEvent::SourceHeld {
                handoff_id: self.dynamo_handoff(handoff_id)?,
                request_id,
                transfer_timing: HandoffTransferTiming {
                    mode: match transfer_timing.mode {
                        EngineTransferTimingMode::FullPrompt => KvTransferTimingMode::FullPrompt,
                        EngineTransferTimingMode::DestinationMissing => {
                            KvTransferTimingMode::DestinationMissing
                        }
                    },
                    full_prompt_tokens: transfer_timing.full_prompt_tokens,
                    kv_bytes_per_token: transfer_timing.kv_bytes_per_token,
                    bandwidth_gb_s: transfer_timing.bandwidth_gb_s,
                },
            },
            LifecycleEvent::DestinationReserved {
                handoff_id,
                request_id,
                transferable_prompt_tokens,
            } => SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: self.dynamo_handoff(handoff_id)?,
                request_id,
                transferable_prompt_tokens,
            },
        })
    }
}

#[derive(Default)]
struct HandoffMap {
    by_dynamo: HashMap<DynamoHandoffId, HandoffEntry>,
}

struct HandoffEntry {
    native: EngineHandoffId,
    source_request: Option<Uuid>,
    destination_request: Option<Uuid>,
}

impl HandoffMap {
    fn native(&mut self, dynamo: DynamoHandoffId) -> Result<EngineHandoffId> {
        if let Some(entry) = self.by_dynamo.get(&dynamo) {
            return Ok(entry.native);
        }
        // Both boundaries carry the caller-owned UUID. Retain a reverse map
        // only for lifecycle cleanup and request correlation; do not allocate
        // a second process-local identity.
        let native = EngineHandoffId::from(Uuid::from(dynamo));
        self.by_dynamo.insert(
            dynamo,
            HandoffEntry {
                native,
                source_request: None,
                destination_request: None,
            },
        );
        Ok(native)
    }

    fn dynamo(&self, native: EngineHandoffId) -> Result<DynamoHandoffId> {
        let dynamo = DynamoHandoffId::from(native.get());
        self.by_dynamo
            .contains_key(&dynamo)
            .then_some(dynamo)
            .ok_or_else(|| anyhow!("native handoff {} has no Dynamo UUID mapping", native.get()))
    }

    fn mark_source(&mut self, dynamo: DynamoHandoffId, request_id: Uuid) {
        if let Some(entry) = self.by_dynamo.get_mut(&dynamo) {
            entry.source_request = Some(request_id);
        }
    }

    fn mark_destination(&mut self, dynamo: DynamoHandoffId, request_id: Uuid) {
        if let Some(entry) = self.by_dynamo.get_mut(&dynamo) {
            entry.destination_request = Some(request_id);
        }
    }

    fn cancel_request(&mut self, request_id: Uuid) {
        let ids = self
            .by_dynamo
            .iter_mut()
            .filter_map(|(id, entry)| {
                let mut matched = false;
                if entry.source_request == Some(request_id) {
                    entry.source_request = None;
                    matched = true;
                }
                if entry.destination_request == Some(request_id) {
                    entry.destination_request = None;
                    matched = true;
                }
                (matched && entry.source_request.is_none() && entry.destination_request.is_none())
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in ids {
            self.remove_if_finished(id);
        }
    }

    fn finish_source(&mut self, id: DynamoHandoffId) {
        if let Some(entry) = self.by_dynamo.get_mut(&id) {
            entry.source_request = None;
        }
        self.remove_if_finished(id);
    }

    fn finish_destination(&mut self, id: DynamoHandoffId) {
        if let Some(entry) = self.by_dynamo.get_mut(&id) {
            entry.destination_request = None;
        }
        self.remove_if_finished(id);
    }

    fn finish_destination_request(&mut self, request_id: Uuid) {
        let ids = self
            .by_dynamo
            .iter_mut()
            .filter_map(|(id, entry)| {
                if entry.destination_request != Some(request_id) {
                    return None;
                }
                entry.destination_request = None;
                (entry.source_request.is_none() && entry.destination_request.is_none())
                    .then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in ids {
            self.remove_if_finished(id);
        }
    }

    fn remove_if_finished(&mut self, id: DynamoHandoffId) {
        let should_remove = self.by_dynamo.get(&id).is_some_and(|entry| {
            entry.source_request.is_none() && entry.destination_request.is_none()
        });
        if !should_remove {
            return;
        }
        self.by_dynamo.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handoff(value: u128) -> DynamoHandoffId {
        DynamoHandoffId::from(Uuid::from_u128(value))
    }

    #[test]
    fn cancelling_request_does_not_remove_unrelated_empty_handoff() {
        let mut handoffs = HandoffMap::default();
        let target = handoff(1);
        let unrelated = handoff(2);
        let target_native = handoffs.native(target).unwrap();
        let unrelated_native = handoffs.native(unrelated).unwrap();
        let request_id = Uuid::from_u128(10);
        handoffs.mark_source(target, request_id);

        handoffs.cancel_request(request_id);

        assert!(handoffs.dynamo(target_native).is_err());
        assert_eq!(handoffs.dynamo(unrelated_native).unwrap(), unrelated);
    }

    #[test]
    fn finishing_destination_request_does_not_remove_unrelated_empty_handoff() {
        let mut handoffs = HandoffMap::default();
        let target = handoff(3);
        let unrelated = handoff(4);
        let target_native = handoffs.native(target).unwrap();
        let unrelated_native = handoffs.native(unrelated).unwrap();
        let request_id = Uuid::from_u128(11);
        handoffs.mark_destination(target, request_id);

        handoffs.finish_destination_request(request_id);

        assert!(handoffs.dynamo(target_native).is_err());
        assert_eq!(handoffs.dynamo(unrelated_native).unwrap(), unrelated);
    }
}
