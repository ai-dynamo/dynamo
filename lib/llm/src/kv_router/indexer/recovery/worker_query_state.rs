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
    Apply {
        event_id: u64,
        event: RouterEvent,
    },
    Clear {
        event_id: u64,
        event: RouterEvent,
    },
    Recover {
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
        reset: bool,
    },
    ResetDegraded {
        event: RouterEvent,
    },
}

pub(super) struct PendingDrainPlan {
    pub(super) events: Vec<RouterEvent>,
    pub(super) cursor: CursorState,
    pub(super) next_recovery_start: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct RankState {
    /// NOTE: This coordinator tracks the last event successfully admitted to the indexer's
    /// per-worker FIFO, not confirmed backend application. A trailing `Flush` proves queue
    /// progress, not event success. Stronger applied-state semantics require an explicit
    /// batch-result acknowledgement. Plan without mutating this cursor, advance only after every
    /// event in the commit group is admitted, and fence/reset on partial admission failure.
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

    pub(super) fn last_admitted_id(&self) -> Option<u64> {
        self.cursor.last_applied_id()
    }

    #[cfg(feature = "ckf-diagnostics")]
    pub(super) fn pending_live_event_count(&self) -> usize {
        self.pending_live_events.len()
    }

    pub(super) fn observe_live_event(
        &mut self,
        event: RouterEvent,
        recoverable: bool,
    ) -> LiveEventAction {
        let event_id = event.event.event_id;

        if matches!(&event.event.data, KvCacheEventData::Cleared) {
            if self
                .last_admitted_id()
                .is_some_and(|last_admitted_id| event_id <= last_admitted_id)
            {
                return LiveEventAction::Ignore;
            }
            return LiveEventAction::Clear { event_id, event };
        }

        match self.cursor.observe(event_id) {
            CursorObservation::Stale { .. } => LiveEventAction::Ignore,
            observation if self.recovery_inflight => {
                if matches!(
                    observation,
                    CursorObservation::Initial { .. }
                        | CursorObservation::Contiguous { .. }
                        | CursorObservation::Gap { .. }
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
                    end_event_id: None,
                    reset: false,
                }
            }
            CursorObservation::Gap { .. } if recoverable => {
                self.observe_and_buffer(event);
                self.recovery_inflight = true;
                LiveEventAction::Recover {
                    start_event_id: None,
                    end_event_id: None,
                    reset: true,
                }
            }
            CursorObservation::Gap { .. } => LiveEventAction::ResetDegraded { event },
            CursorObservation::Initial { got } | CursorObservation::Contiguous { got } => {
                LiveEventAction::Apply {
                    event_id: got,
                    event,
                }
            }
        }
    }

    pub(super) fn commit_live_admission(&mut self, event_id: u64) {
        self.cursor = self.cursor.advance_to(event_id);
        self.clear_max_seen_if_caught_up(event_id);
    }

    pub(super) fn begin_successful_recovery_drain(&mut self, cursor: CursorState) {
        self.cursor = cursor;
        self.recovery_inflight = true;
    }

    pub(super) fn discard_recovery_before_clear(&mut self) {
        self.recovery_inflight = false;
        self.pending_live_events.clear();
        self.max_seen_live_id = None;
    }

    pub(super) fn plan_pending_drain(&mut self) -> PendingDrainPlan {
        let mut last_admitted_id = self.last_admitted_id().unwrap_or(0);
        let mut cursor = self.cursor;
        self.pending_live_events
            .make_contiguous()
            .sort_unstable_by_key(|event| event.event.event_id);
        self.fast_prune_stale_pending_prefix(last_admitted_id);
        let mut events = Vec::new();

        loop {
            let Some(front_event_id) = self
                .pending_live_events
                .front()
                .map(|event| event.event.event_id)
            else {
                self.clear_max_seen_if_caught_up(last_admitted_id);
                if self
                    .max_seen_live_id
                    .is_some_and(|max_seen| max_seen > last_admitted_id)
                {
                    return PendingDrainPlan {
                        events,
                        cursor,
                        next_recovery_start: Some(last_admitted_id.saturating_add(1)),
                    };
                }
                return PendingDrainPlan {
                    events,
                    cursor,
                    next_recovery_start: None,
                };
            };

            if front_event_id <= last_admitted_id {
                self.pending_live_events.pop_front();
                continue;
            }

            let expected = last_admitted_id.saturating_add(1);
            if front_event_id != expected {
                return PendingDrainPlan {
                    events,
                    cursor,
                    next_recovery_start: Some(expected),
                };
            }

            let event = self
                .pending_live_events
                .pop_front()
                .expect("front event exists while draining pending live events");
            last_admitted_id = front_event_id;
            cursor = cursor.advance_to(front_event_id);
            events.push(event);
        }
    }

    pub(super) fn commit_pending_drain(
        &mut self,
        cursor: CursorState,
        next_recovery_start: Option<u64>,
    ) {
        self.cursor = cursor;
        self.clear_max_seen_if_caught_up(self.last_admitted_id().unwrap_or(0));
        self.recovery_inflight = next_recovery_start.is_some();
    }

    pub(super) fn finish_failed_recovery(&mut self) {
        self.recovery_inflight = false;
        self.pending_live_events.clear();
        self.max_seen_live_id = None;
    }

    pub(super) fn take_failed_recovery_degraded(&mut self) -> Vec<RouterEvent> {
        let last_admitted_id = self.last_admitted_id().unwrap_or(0);
        let mut events: Vec<_> = self.pending_live_events.drain(..).collect();
        events.sort_unstable_by_key(|event| event.event.event_id);
        events.dedup_by_key(|event| event.event.event_id);
        events.retain(|event| event.event.event_id > last_admitted_id);
        events
    }

    pub(super) fn commit_failed_recovery_degraded(&mut self, last_event_id: Option<u64>) {
        if let Some(last_event_id) = last_event_id {
            self.cursor = self.cursor.advance_to(last_event_id);
        }
        self.recovery_inflight = false;
        self.max_seen_live_id = None;
    }

    fn observe_and_buffer(&mut self, event: RouterEvent) {
        let event_id = event.event.event_id;
        self.max_seen_live_id = Some(self.max_seen_live_id.unwrap_or(0).max(event_id));
        self.pending_live_events.push_back(event);
        while self.pending_live_events.len() > RECOVERY_PENDING_LIVE_EVENT_LIMIT {
            self.pending_live_events.pop_front();
        }
    }

    fn clear_max_seen_if_caught_up(&mut self, last_admitted_id: u64) {
        if self
            .max_seen_live_id
            .is_some_and(|max_seen| max_seen <= last_admitted_id)
        {
            self.max_seen_live_id = None;
        }
    }

    fn fast_prune_stale_pending_prefix(&mut self, last_admitted_id: u64) {
        if self.pending_live_events.len() <= RECOVERY_PENDING_FAST_PRUNE_MARGIN {
            return;
        }
        let split_at = self.pending_live_events.len() - RECOVERY_PENDING_FAST_PRUNE_MARGIN;
        if self
            .pending_live_events
            .get(split_at)
            .is_some_and(|event| event.event.event_id <= last_admitted_id)
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
        let action = state.observe_live_event(store(9), false);
        assert!(matches!(action, LiveEventAction::Apply { event_id: 9, .. }));
        assert_eq!(state.last_admitted_id(), None);
        state.commit_live_admission(9);
        assert_eq!(state.last_admitted_id(), Some(9));
    }

    #[test]
    fn recoverable_source_buffers_until_restore() {
        let mut state = RankState::default();
        assert!(matches!(
            state.observe_live_event(store(9), true),
            LiveEventAction::Recover {
                start_event_id: None,
                end_event_id: None,
                reset: false,
            }
        ));
        assert!(state.recovery_inflight);
    }

    #[test]
    fn gap_recovery_buffers_and_drains_live_events_in_event_id_order() {
        let mut state = RankState::default();
        assert!(matches!(
            state.observe_live_event(store(1), false),
            LiveEventAction::Apply { event_id: 1, .. }
        ));
        state.commit_live_admission(1);

        assert!(matches!(
            state.observe_live_event(store(4), true),
            LiveEventAction::Recover {
                start_event_id: None,
                end_event_id: None,
                reset: true,
            }
        ));
        assert!(matches!(
            state.observe_live_event(store(3), true),
            LiveEventAction::Ignore
        ));
        assert_eq!(state.last_admitted_id(), Some(1));

        state.begin_successful_recovery_drain(CursorState::Initial.advance_to(2));
        let plan = state.plan_pending_drain();
        assert_eq!(
            plan.events
                .iter()
                .map(|event| event.event.event_id)
                .collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert_eq!(plan.cursor.last_applied_id(), Some(4));
        assert_eq!(plan.next_recovery_start, None);
        assert_eq!(state.last_admitted_id(), Some(2));
        state.commit_pending_drain(plan.cursor, plan.next_recovery_start);
        assert_eq!(state.last_admitted_id(), Some(4));
        assert!(!state.recovery_inflight);
    }

    #[test]
    fn clear_supersedes_same_rank_gap_recovery() {
        let mut state = RankState::default();
        assert!(matches!(
            state.observe_live_event(store(1), false),
            LiveEventAction::Apply { event_id: 1, .. }
        ));
        state.commit_live_admission(1);
        assert!(matches!(
            state.observe_live_event(store(4), true),
            LiveEventAction::Recover { reset: true, .. }
        ));

        let mut clear = store(5);
        clear.event.data = KvCacheEventData::Cleared;
        assert!(matches!(
            state.observe_live_event(clear, true),
            LiveEventAction::Clear { event_id: 5, .. }
        ));
        state.discard_recovery_before_clear();
        state.commit_live_admission(5);

        assert!(!state.recovery_inflight);
        assert!(matches!(
            state.observe_live_event(store(6), true),
            LiveEventAction::Apply { event_id: 6, .. }
        ));
    }
}
