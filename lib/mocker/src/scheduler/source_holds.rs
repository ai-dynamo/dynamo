// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::common::handoff::{HandoffId, HandoffTransferTiming};
use crate::common::protocols::DirectRequest;

#[allow(dead_code)]
pub enum SchedulerCommand {
    Submit(DirectRequest),
    /// Cancel an ordinary aggregate request by its request identifier.
    CancelRequest {
        request_id: Uuid,
    },
    SubmitHandoffPrefill {
        handoff_id: HandoffId,
        request: DirectRequest,
    },
    ReleaseSource {
        handoff_id: HandoffId,
    },
    CancelSource {
        handoff_id: HandoffId,
    },
    ReserveDestination {
        handoff_id: HandoffId,
        request: DirectRequest,
    },
    ActivateDestination {
        handoff_id: HandoffId,
    },
    CancelDestination {
        handoff_id: HandoffId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerCommandResult {
    Submitted(Uuid),
    DestinationAccepted { request_id: Uuid },
    Applied,
    Noop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SchedulerLifecycleEvent {
    SourceHeld {
        handoff_id: HandoffId,
        request_id: Uuid,
        transfer_timing: HandoffTransferTiming,
    },
    DestinationReserved {
        handoff_id: HandoffId,
        request_id: Uuid,
        transferable_prompt_tokens: usize,
    },
}

impl SchedulerLifecycleEvent {
    pub fn handoff_id(&self) -> HandoffId {
        match *self {
            Self::SourceHeld { handoff_id, .. } | Self::DestinationReserved { handoff_id, .. } => {
                handoff_id
            }
        }
    }
}

#[derive(Debug)]
pub struct SchedulerCommandEffects {
    pub result: SchedulerCommandResult,
    pub lifecycle_events: Vec<SchedulerLifecycleEvent>,
    pub kv_events: Vec<dynamo_kv_router::protocols::RouterEvent>,
}

impl SchedulerCommandEffects {
    pub(crate) fn new(result: SchedulerCommandResult) -> Self {
        Self {
            result,
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }
    }
}

pub(crate) enum SourceCompletion<T> {
    Release(T),
    Held { handoff_id: HandoffId },
}

pub(crate) enum RemovedSource<T> {
    Held(T),
    Pending { request_id: Uuid },
    Missing,
}

pub(crate) struct SourceHolds<T> {
    pending_by_request: FxHashMap<Uuid, HandoffId>,
    pending_by_handoff: FxHashMap<HandoffId, Uuid>,
    held_prefills: FxHashMap<HandoffId, (Uuid, T)>,
    held_by_request: FxHashMap<Uuid, HandoffId>,
}

impl<T> Default for SourceHolds<T> {
    fn default() -> Self {
        Self {
            pending_by_request: FxHashMap::default(),
            pending_by_handoff: FxHashMap::default(),
            held_prefills: FxHashMap::default(),
            held_by_request: FxHashMap::default(),
        }
    }
}

impl<T> SourceHolds<T> {
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.pending_by_request.len(), self.pending_by_handoff.len());
        debug_assert_eq!(self.held_prefills.len(), self.held_by_request.len());
        self.pending_by_request.is_empty()
            && self.held_prefills.is_empty()
            && self.held_by_request.is_empty()
    }

    pub(crate) fn register(&mut self, request_id: Uuid, handoff_id: HandoffId) -> Result<()> {
        if self.contains_request(request_id) {
            bail!("source hold already active for request {request_id}");
        }
        if self.pending_by_handoff.contains_key(&handoff_id)
            || self.held_prefills.contains_key(&handoff_id)
        {
            bail!("handoff {handoff_id:?} is already active");
        }

        self.pending_by_request.insert(request_id, handoff_id);
        self.pending_by_handoff.insert(handoff_id, request_id);
        Ok(())
    }

    pub(crate) fn contains_request(&self, request_id: Uuid) -> bool {
        self.pending_by_request.contains_key(&request_id)
            || self.held_by_request.contains_key(&request_id)
    }

    pub(crate) fn complete_source(&mut self, request_id: Uuid, payload: T) -> SourceCompletion<T> {
        let Some(handoff_id) = self.pending_by_request.remove(&request_id) else {
            return SourceCompletion::Release(payload);
        };

        let registered_request = self
            .pending_by_handoff
            .remove(&handoff_id)
            .expect("validated source-hold registration must be bidirectional");
        debug_assert_eq!(registered_request, request_id);

        let previous = self.held_prefills.insert(handoff_id, (request_id, payload));
        debug_assert!(previous.is_none());
        let previous = self.held_by_request.insert(request_id, handoff_id);
        debug_assert!(previous.is_none());
        SourceCompletion::Held { handoff_id }
    }

    pub(crate) fn remove(&mut self, handoff_id: HandoffId) -> RemovedSource<T> {
        if let Some((request_id, payload)) = self.held_prefills.remove(&handoff_id) {
            let removed = self.held_by_request.remove(&request_id);
            debug_assert_eq!(removed, Some(handoff_id));
            return RemovedSource::Held(payload);
        }

        let Some(request_id) = self.pending_by_handoff.remove(&handoff_id) else {
            return RemovedSource::Missing;
        };
        let removed = self.pending_by_request.remove(&request_id);
        debug_assert_eq!(removed, Some(handoff_id));
        RemovedSource::Pending { request_id }
    }

    pub(crate) fn remove_request(&mut self, request_id: Uuid) {
        let Some(handoff_id) = self.pending_by_request.remove(&request_id) else {
            return;
        };
        let removed = self.pending_by_handoff.remove(&handoff_id);
        debug_assert_eq!(removed, Some(request_id));
    }

    #[cfg(test)]
    pub(crate) fn is_held(&self, handoff_id: HandoffId) -> bool {
        self.held_prefills.contains_key(&handoff_id)
    }

    #[cfg(test)]
    pub(crate) fn is_registered(&self, handoff_id: HandoffId) -> bool {
        self.pending_by_handoff.contains_key(&handoff_id)
    }
}

pub(crate) struct DestinationHolds<T> {
    by_handoff: FxHashMap<HandoffId, (Uuid, T)>,
    by_request: FxHashMap<Uuid, HandoffId>,
}

#[derive(Default)]
pub(crate) struct ActiveHandoffRequests {
    by_handoff: FxHashMap<HandoffId, Uuid>,
    by_request: FxHashMap<Uuid, HandoffId>,
}

impl ActiveHandoffRequests {
    pub(crate) fn insert(&mut self, handoff_id: HandoffId, request_id: Uuid) {
        let previous = self.by_handoff.insert(handoff_id, request_id);
        debug_assert!(previous.is_none());
        let previous = self.by_request.insert(request_id, handoff_id);
        debug_assert!(previous.is_none());
    }

    pub(crate) fn remove_handoff(&mut self, handoff_id: HandoffId) -> Option<Uuid> {
        let request_id = self.by_handoff.remove(&handoff_id)?;
        let removed = self.by_request.remove(&request_id);
        debug_assert_eq!(removed, Some(handoff_id));
        Some(request_id)
    }

    pub(crate) fn remove_request(&mut self, request_id: Uuid) -> Option<HandoffId> {
        let handoff_id = self.by_request.remove(&request_id)?;
        let removed = self.by_handoff.remove(&handoff_id);
        debug_assert_eq!(removed, Some(request_id));
        Some(handoff_id)
    }

    pub(crate) fn contains_request(&self, request_id: Uuid) -> bool {
        self.by_request.contains_key(&request_id)
    }

    pub(crate) fn contains_handoff(&self, handoff_id: HandoffId) -> bool {
        self.by_handoff.contains_key(&handoff_id)
    }

    pub(crate) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.by_handoff.len(), self.by_request.len());
        self.by_handoff.is_empty()
    }
}

pub(crate) struct PendingDestinations<T> {
    fifo: VecDeque<HandoffId>,
    by_handoff: FxHashMap<HandoffId, PendingDestination<T>>,
    by_request: FxHashMap<Uuid, HandoffId>,
}

struct PendingDestination<T> {
    request_id: Uuid,
    payload: T,
    last_attempt_generation: Option<u64>,
}

impl<T> Default for PendingDestinations<T> {
    fn default() -> Self {
        Self {
            fifo: VecDeque::new(),
            by_handoff: FxHashMap::default(),
            by_request: FxHashMap::default(),
        }
    }
}

impl<T> PendingDestinations<T> {
    pub(crate) fn validate(&self, request_id: Uuid, handoff_id: HandoffId) -> Result<()> {
        if self.by_request.contains_key(&request_id) {
            bail!("destination request {request_id} is already pending");
        }
        if self.by_handoff.contains_key(&handoff_id) {
            bail!("destination handoff {handoff_id:?} is already pending");
        }
        Ok(())
    }

    pub(crate) fn insert(&mut self, request_id: Uuid, handoff_id: HandoffId, payload: T) {
        debug_assert!(self.validate(request_id, handoff_id).is_ok());
        self.fifo.push_back(handoff_id);
        let previous = self.by_handoff.insert(
            handoff_id,
            PendingDestination {
                request_id,
                payload,
                last_attempt_generation: None,
            },
        );
        debug_assert!(previous.is_none());
        let previous = self.by_request.insert(request_id, handoff_id);
        debug_assert!(previous.is_none());
    }

    pub(crate) fn front_due(&mut self, generation: u64) -> Option<(HandoffId, Uuid, &T)> {
        self.normalize_front();
        let handoff_id = *self.fifo.front()?;
        let pending = self
            .by_handoff
            .get(&handoff_id)
            .expect("normalized pending destination must exist");
        if pending.last_attempt_generation == Some(generation) {
            return None;
        }
        Some((handoff_id, pending.request_id, &pending.payload))
    }

    pub(crate) fn front_due_mut(&mut self, generation: u64) -> Option<(HandoffId, Uuid, &mut T)> {
        self.normalize_front();
        let handoff_id = *self.fifo.front()?;
        let pending = self
            .by_handoff
            .get_mut(&handoff_id)
            .expect("normalized pending destination must exist");
        if pending.last_attempt_generation == Some(generation) {
            return None;
        }
        Some((handoff_id, pending.request_id, &mut pending.payload))
    }

    pub(crate) fn mark_front_attempted(&mut self, generation: u64) {
        self.normalize_front();
        let Some(handoff_id) = self.fifo.front() else {
            return;
        };
        let pending = self
            .by_handoff
            .get_mut(handoff_id)
            .expect("normalized pending destination must exist");
        pending.last_attempt_generation = Some(generation);
    }

    pub(crate) fn pop_front(&mut self) -> Option<(HandoffId, Uuid, T)> {
        self.normalize_front();
        let handoff_id = self.fifo.pop_front()?;
        let pending = self
            .by_handoff
            .remove(&handoff_id)
            .expect("normalized pending destination must exist");
        let removed = self.by_request.remove(&pending.request_id);
        debug_assert_eq!(removed, Some(handoff_id));
        self.compact_if_sparse();
        Some((handoff_id, pending.request_id, pending.payload))
    }

    pub(crate) fn remove(&mut self, handoff_id: HandoffId) -> Option<(Uuid, T)> {
        let pending = self.by_handoff.remove(&handoff_id)?;
        let removed = self.by_request.remove(&pending.request_id);
        debug_assert_eq!(removed, Some(handoff_id));
        self.compact_if_sparse();
        Some((pending.request_id, pending.payload))
    }

    pub(crate) fn contains_request(&self, request_id: Uuid) -> bool {
        self.by_request.contains_key(&request_id)
    }

    #[cfg(test)]
    pub(crate) fn contains_handoff(&self, handoff_id: HandoffId) -> bool {
        self.by_handoff.contains_key(&handoff_id)
    }

    pub(crate) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.by_handoff.len(), self.by_request.len());
        self.by_handoff.is_empty()
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.by_handoff.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.by_handoff.len()
    }

    pub(crate) fn payloads(&self) -> impl Iterator<Item = &T> {
        self.by_handoff.values().map(|pending| &pending.payload)
    }

    fn normalize_front(&mut self) {
        while self
            .fifo
            .front()
            .is_some_and(|handoff_id| !self.by_handoff.contains_key(handoff_id))
        {
            self.fifo.pop_front();
        }
    }

    fn compact_if_sparse(&mut self) {
        if self.fifo.len() <= self.by_handoff.len().saturating_mul(2) {
            return;
        }
        self.fifo
            .retain(|handoff_id| self.by_handoff.contains_key(handoff_id));
    }
}

impl<T> Default for DestinationHolds<T> {
    fn default() -> Self {
        Self {
            by_handoff: FxHashMap::default(),
            by_request: FxHashMap::default(),
        }
    }
}

impl<T> DestinationHolds<T> {
    pub(crate) fn validate(&self, request_id: Uuid, handoff_id: HandoffId) -> Result<()> {
        if self.by_request.contains_key(&request_id) {
            bail!("destination reservation already exists for request {request_id}");
        }
        if self.by_handoff.contains_key(&handoff_id) {
            bail!("destination handoff {handoff_id:?} is already active");
        }
        Ok(())
    }

    pub(crate) fn contains_request(&self, request_id: Uuid) -> bool {
        self.by_request.contains_key(&request_id)
    }

    pub(crate) fn insert(&mut self, request_id: Uuid, handoff_id: HandoffId, payload: T) {
        debug_assert!(self.validate(request_id, handoff_id).is_ok());
        let previous = self.by_handoff.insert(handoff_id, (request_id, payload));
        debug_assert!(previous.is_none());
        let previous = self.by_request.insert(request_id, handoff_id);
        debug_assert!(previous.is_none());
    }

    pub(crate) fn remove(&mut self, handoff_id: HandoffId) -> Option<(Uuid, T)> {
        let (request_id, payload) = self.by_handoff.remove(&handoff_id)?;
        let removed = self.by_request.remove(&request_id);
        debug_assert_eq!(removed, Some(handoff_id));
        Some((request_id, payload))
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.by_handoff.len(), self.by_request.len());
        self.by_handoff.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.by_handoff.len()
    }

    pub(crate) fn payloads(&self) -> impl Iterator<Item = &T> {
        self.by_handoff.values().map(|(_, payload)| payload)
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, handoff_id: HandoffId) -> bool {
        self.by_handoff.contains_key(&handoff_id)
    }

    #[cfg(test)]
    pub(crate) fn get(&self, handoff_id: HandoffId) -> Option<&T> {
        self.by_handoff.get(&handoff_id).map(|(_, payload)| payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_atomically_moves_registered_payload_to_held() {
        let request_id = Uuid::from_u128(1);
        let handoff_id = HandoffId::from(Uuid::from_u128(2));
        let mut holds = SourceHolds::default();

        holds.register(request_id, handoff_id).unwrap();
        assert!(holds.is_registered(handoff_id));

        let completion = holds.complete_source(request_id, "payload");
        assert!(matches!(
            completion,
            SourceCompletion::Held { handoff_id: held } if held == handoff_id
        ));
        assert!(!holds.is_registered(handoff_id));
        assert!(holds.is_held(handoff_id));
        assert!(
            holds
                .register(request_id, HandoffId::from(Uuid::from_u128(7)))
                .is_err()
        );
        assert!(matches!(
            holds.remove(handoff_id),
            RemovedSource::Held("payload")
        ));
        assert!(matches!(holds.remove(handoff_id), RemovedSource::Missing));
    }

    #[test]
    fn active_ids_are_rejected_but_released_ids_can_be_reused() {
        let request_id = Uuid::from_u128(3);
        let handoff_id = HandoffId::from(Uuid::from_u128(4));
        let mut holds = SourceHolds::<()>::default();

        holds.register(request_id, handoff_id).unwrap();
        assert!(holds.register(Uuid::from_u128(5), handoff_id).is_err());
        assert!(holds.register(request_id, HandoffId::new()).is_err());
        assert!(matches!(
            holds.remove(handoff_id),
            RemovedSource::Pending { request_id: pending } if pending == request_id
        ));

        holds.register(Uuid::from_u128(5), handoff_id).unwrap();
    }

    #[test]
    fn unregistered_completion_releases_payload() {
        let mut holds = SourceHolds::default();
        let payload = String::from("payload");

        assert!(matches!(
            holds.complete_source(Uuid::from_u128(6), payload),
            SourceCompletion::Release(value) if value == "payload"
        ));
    }

    #[test]
    fn pending_destination_preserves_fifo_after_head_cancellation() {
        let first = HandoffId::from(Uuid::from_u128(10));
        let second = HandoffId::from(Uuid::from_u128(11));
        let mut pending = PendingDestinations::default();
        pending.insert(Uuid::from_u128(20), first, "first");
        pending.insert(Uuid::from_u128(21), second, "second");

        assert_eq!(pending.remove(first), Some((Uuid::from_u128(20), "first")));
        let (handoff_id, request_id, payload) = pending.front_due(0).unwrap();
        assert_eq!(handoff_id, second);
        assert_eq!(request_id, Uuid::from_u128(21));
        assert_eq!(*payload, "second");

        pending.mark_front_attempted(0);
        assert!(pending.front_due(0).is_none());
        assert!(pending.front_due(1).is_some());
    }

    #[test]
    fn pending_destination_indexes_reject_conflicts_and_compact_tombstones() {
        let mut pending = PendingDestinations::default();
        let handoffs = (0..8)
            .map(|index| HandoffId::from(Uuid::from_u128(100 + index)))
            .collect::<Vec<_>>();
        for (index, handoff_id) in handoffs.iter().copied().enumerate() {
            pending.insert(Uuid::from_u128(200 + index as u128), handoff_id, index);
        }
        assert!(
            pending
                .validate(Uuid::from_u128(200), HandoffId::new())
                .is_err()
        );
        assert!(pending.validate(Uuid::new_v4(), handoffs[0]).is_err());

        for handoff_id in handoffs.iter().take(6) {
            assert!(pending.remove(*handoff_id).is_some());
        }
        assert!(pending.fifo.len() <= pending.by_handoff.len() * 2);
        let (handoff_id, request_id, payload) = pending.front_due(0).unwrap();
        assert_eq!(handoff_id, handoffs[6]);
        assert_eq!(request_id, Uuid::from_u128(206));
        assert_eq!(*payload, 6);
        assert_eq!(pending.pop_front(), Some((handoffs[6], request_id, 6)));
        assert_eq!(pending.pop_front().map(|entry| entry.0), Some(handoffs[7]));
        assert!(pending.is_empty());
    }
}
