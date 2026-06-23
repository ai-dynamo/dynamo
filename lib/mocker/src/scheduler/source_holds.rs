// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::common::handoff::HandoffId;
use crate::common::protocols::DirectRequest;

#[allow(dead_code)]
pub(crate) enum SchedulerCommand {
    Submit(DirectRequest),
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
pub(crate) enum SchedulerCommandResult {
    Submitted(Uuid),
    DestinationReserved { request_id: Uuid },
    DestinationUnavailable,
    Applied,
    Noop,
}

pub(crate) enum SourceCompletion<T> {
    Release(T),
    Held,
}

pub(crate) enum RemovedSource<T> {
    Held(T),
    Pending,
    Missing,
}

pub(crate) struct SourceHolds<T> {
    pending_by_request: FxHashMap<Uuid, HandoffId>,
    pending_by_handoff: FxHashMap<HandoffId, Uuid>,
    held_prefills: FxHashMap<HandoffId, (Uuid, T)>,
}

impl<T> Default for SourceHolds<T> {
    fn default() -> Self {
        Self {
            pending_by_request: FxHashMap::default(),
            pending_by_handoff: FxHashMap::default(),
            held_prefills: FxHashMap::default(),
        }
    }
}

impl<T> SourceHolds<T> {
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.pending_by_request.len(), self.pending_by_handoff.len());
        self.pending_by_request.is_empty() && self.held_prefills.is_empty()
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
            || self
                .held_prefills
                .values()
                .any(|(held_request_id, _)| *held_request_id == request_id)
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
        SourceCompletion::Held
    }

    pub(crate) fn remove(&mut self, handoff_id: HandoffId) -> RemovedSource<T> {
        if let Some((_, payload)) = self.held_prefills.remove(&handoff_id) {
            return RemovedSource::Held(payload);
        }

        let Some(request_id) = self.pending_by_handoff.remove(&handoff_id) else {
            return RemovedSource::Missing;
        };
        let removed = self.pending_by_request.remove(&request_id);
        debug_assert_eq!(removed, Some(handoff_id));
        RemovedSource::Pending
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
        assert!(matches!(completion, SourceCompletion::Held));
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
        assert!(matches!(holds.remove(handoff_id), RemovedSource::Pending));

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
}
