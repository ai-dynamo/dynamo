// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use dynamo_kv_router::{
    protocols::{DpRank, KvCacheEventData, RouterEvent, WorkerId},
    recovery::{CursorObservation, CursorState},
};

pub(super) type RecoveryKey = (WorkerId, DpRank);

const RECOVERY_PENDING_LIVE_EVENT_LIMIT: usize = 1024;
const RECOVERY_PENDING_FAST_PRUNE_MARGIN: usize = 10;

pub(super) enum LiveEventAction {
    Ignore,
    Apply(RouterEvent),
    Clear {
        event_id: u64,
    },
    Recover {
        start_event_id: Option<u64>,
    },
    ResetDegraded {
        event: RouterEvent,
        recover_from: Option<u64>,
    },
}

pub(super) enum PendingDrainAction {
    Apply(RouterEvent),
    RecoverFrom(u64),
    Complete,
}

#[derive(Debug, Default)]
pub(super) struct RankState {
    pub(super) cursor: CursorState,
    pub(super) recovery_inflight: bool,
    pending_live_events: VecDeque<RouterEvent>,
    max_seen_live_id: Option<u64>,
}

impl RankState {
    pub(super) fn activate(&mut self, recoverable: bool) {
        *self = Self {
            recovery_inflight: recoverable,
            ..Self::default()
        };
    }

    pub(super) fn last_applied_id(&self) -> Option<u64> {
        self.cursor.last_applied_id()
    }

    pub(super) fn observe_live_event(
        &mut self,
        event: RouterEvent,
        recoverable: bool,
    ) -> LiveEventAction {
        let event_id = event.event.event_id;

        if matches!(&event.event.data, KvCacheEventData::Cleared) {
            if self
                .last_applied_id()
                .is_some_and(|last_applied_id| event_id <= last_applied_id)
            {
                return LiveEventAction::Ignore;
            }
            self.cursor = self.cursor.apply_barrier(event_id);
            self.recovery_inflight = false;
            self.pending_live_events.clear();
            self.max_seen_live_id = None;
            return LiveEventAction::Clear { event_id };
        }

        match self.cursor.observe(event_id) {
            CursorObservation::Stale { .. } => LiveEventAction::Ignore,
            observation if self.recovery_inflight => {
                if matches!(
                    observation,
                    CursorObservation::Initial { .. }
                        | CursorObservation::Contiguous { .. }
                        | CursorObservation::Gap { .. }
                        | CursorObservation::FreshAfterBarrier { .. }
                ) {
                    self.observe_and_buffer(event);
                }
                LiveEventAction::Ignore
            }
            CursorObservation::Initial { .. } if recoverable => {
                self.observe_and_buffer(event.clone());
                self.recovery_inflight = true;
                LiveEventAction::Recover {
                    start_event_id: None,
                }
            }
            CursorObservation::Gap { expected, got } if recoverable => {
                self.cursor = self.cursor.advance_to(got);
                self.recovery_inflight = true;
                LiveEventAction::ResetDegraded {
                    event,
                    recover_from: Some(expected),
                }
            }
            CursorObservation::Gap { .. } => {
                self.cursor = self.cursor.advance_to(event_id);
                LiveEventAction::ResetDegraded {
                    event,
                    recover_from: None,
                }
            }
            CursorObservation::Initial { got }
            | CursorObservation::Contiguous { got }
            | CursorObservation::FreshAfterBarrier { got, .. } => {
                self.cursor = self.cursor.advance_to(got);
                self.clear_max_seen_if_caught_up(got);
                LiveEventAction::Apply(event)
            }
        }
    }

    pub(super) fn begin_successful_recovery_drain(&mut self, cursor: CursorState) {
        self.cursor = cursor;
        self.recovery_inflight = true;
    }

    pub(super) fn apply_worker_clear_barrier(&mut self, event_id: u64, emitter: bool) {
        self.cursor = if emitter {
            self.cursor.apply_barrier(event_id)
        } else {
            self.cursor.invalidate_by_barrier()
        };
        self.recovery_inflight = false;
        self.pending_live_events.clear();
        self.max_seen_live_id = None;
    }

    pub(super) fn next_pending_drain_action(&mut self) -> PendingDrainAction {
        let mut last_applied_id = self.last_applied_id().unwrap_or(0);
        self.fast_prune_stale_pending_prefix(last_applied_id);

        loop {
            let Some(front_event_id) = self
                .pending_live_events
                .front()
                .map(|event| event.event.event_id)
            else {
                self.clear_max_seen_if_caught_up(last_applied_id);
                if self
                    .max_seen_live_id
                    .is_some_and(|max_seen| max_seen > last_applied_id)
                {
                    return PendingDrainAction::RecoverFrom(last_applied_id.saturating_add(1));
                }
                self.recovery_inflight = false;
                return PendingDrainAction::Complete;
            };

            if front_event_id <= last_applied_id {
                self.pending_live_events.pop_front();
                continue;
            }

            let expected = last_applied_id.saturating_add(1);
            if front_event_id != expected {
                return PendingDrainAction::RecoverFrom(expected);
            }

            let event = self
                .pending_live_events
                .pop_front()
                .expect("front event exists while draining pending live events");
            self.cursor = self.cursor.advance_to(front_event_id);
            last_applied_id = front_event_id;
            self.clear_max_seen_if_caught_up(last_applied_id);
            return PendingDrainAction::Apply(event);
        }
    }

    pub(super) fn finish_failed_recovery(&mut self) {
        self.recovery_inflight = false;
        self.pending_live_events.clear();
        self.max_seen_live_id = None;
    }

    pub(super) fn finish_failed_recovery_degraded(&mut self) -> Vec<RouterEvent> {
        self.recovery_inflight = false;
        self.max_seen_live_id = None;
        let mut events = Vec::with_capacity(self.pending_live_events.len());
        while let Some(event) = self.pending_live_events.pop_front() {
            let event_id = event.event.event_id;
            if self
                .last_applied_id()
                .is_some_and(|last_applied_id| event_id <= last_applied_id)
            {
                continue;
            }
            self.cursor = self.cursor.advance_to(event_id);
            events.push(event);
        }
        events
    }

    fn observe_and_buffer(&mut self, event: RouterEvent) {
        let event_id = event.event.event_id;
        self.max_seen_live_id = Some(self.max_seen_live_id.unwrap_or(0).max(event_id));
        self.pending_live_events.push_back(event);
        while self.pending_live_events.len() > RECOVERY_PENDING_LIVE_EVENT_LIMIT {
            self.pending_live_events.pop_front();
        }
    }

    fn clear_max_seen_if_caught_up(&mut self, last_applied_id: u64) {
        if self
            .max_seen_live_id
            .is_some_and(|max_seen| max_seen <= last_applied_id)
        {
            self.max_seen_live_id = None;
        }
    }

    fn fast_prune_stale_pending_prefix(&mut self, last_applied_id: u64) {
        if self.pending_live_events.len() <= RECOVERY_PENDING_FAST_PRUNE_MARGIN {
            return;
        }
        let split_at = self.pending_live_events.len() - RECOVERY_PENDING_FAST_PRUNE_MARGIN;
        if self
            .pending_live_events
            .get(split_at)
            .is_some_and(|event| event.event.event_id <= last_applied_id)
        {
            self.pending_live_events.drain(..split_at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };

    fn store(event_id: u64) -> RouterEvent {
        RouterEvent::new(
            1,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(event_id),
                        tokens_hash: LocalBlockHash(event_id),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        )
    }

    #[test]
    fn live_only_source_accepts_first_event_without_recovery() {
        let mut state = RankState::default();
        assert!(matches!(
            state.observe_live_event(store(9), false),
            LiveEventAction::Apply(_)
        ));
        assert_eq!(state.last_applied_id(), Some(9));
    }

    #[test]
    fn recoverable_source_buffers_until_restore() {
        let mut state = RankState::default();
        assert!(matches!(
            state.observe_live_event(store(9), true),
            LiveEventAction::Recover {
                start_event_id: None,
            }
        ));
        assert!(state.recovery_inflight);
    }
}
