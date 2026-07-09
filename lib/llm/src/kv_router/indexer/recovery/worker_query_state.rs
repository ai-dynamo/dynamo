// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};

use dynamo_kv_router::{
    protocols::{DpRank, KvCacheEventData, RouterEvent, WorkerId},
    recovery::{CursorObservation, CursorState},
};

pub(super) type RecoveryKey = (WorkerId, DpRank);

pub(super) const RECOVERY_PENDING_LIVE_EVENT_LIMIT: usize = 1024;
pub(super) const RECOVERY_PENDING_FAST_PRUNE_MARGIN: usize = 10;

pub(super) enum LiveEventAction {
    Ignore,
    ApplyDirect(RouterEvent),
    ApplyClear(RouterEvent),
    SpawnFullRestore { epoch: u64 },
    SpawnIncremental { epoch: u64, start_event_id: u64 },
}

pub(super) struct DiscoveredRankAction {
    pub(super) reset_rank: bool,
    pub(super) restore_epoch: Option<u64>,
}

pub(super) enum PendingDrainAction {
    Apply(RouterEvent),
    RecoverFrom(u64),
    Complete,
}

#[derive(Debug, Default)]
pub(super) struct RankState {
    pub(super) cursor: CursorState,
    pub(super) max_seen_live_id: Option<u64>,
    pub(super) recovery_inflight: bool,
    pub(super) pending_live_events: VecDeque<RouterEvent>,
}

impl RankState {
    pub(super) fn last_applied_id(&self) -> Option<u64> {
        self.cursor.last_applied_id()
    }

    pub(super) fn observe_live_id(&mut self, event_id: u64) {
        self.max_seen_live_id = Some(self.max_seen_live_id.unwrap_or(0).max(event_id));
    }

    fn clear_max_seen_if_caught_up(&mut self, last_applied_id: u64) {
        if self
            .max_seen_live_id
            .is_some_and(|max_seen| max_seen <= last_applied_id)
        {
            self.max_seen_live_id = None;
        }
    }

    fn observe_and_buffer_live_event(&mut self, event: RouterEvent) {
        self.observe_live_id(event.event.event_id);
        self.push_pending_live_event(event);
    }

    fn push_pending_live_event(&mut self, event: RouterEvent) {
        self.pending_live_events.push_back(event);
        while self.pending_live_events.len() > RECOVERY_PENDING_LIVE_EVENT_LIMIT {
            self.pending_live_events.pop_front();
        }
    }

    fn clear_pending_live_events(&mut self) {
        self.pending_live_events.clear();
    }

    fn fast_prune_stale_pending_prefix(&mut self, last_applied_id: u64) {
        if self.pending_live_events.len() <= RECOVERY_PENDING_FAST_PRUNE_MARGIN {
            return;
        }

        let split_at = self.pending_live_events.len() - RECOVERY_PENDING_FAST_PRUNE_MARGIN;
        let Some(boundary_event) = self.pending_live_events.get(split_at) else {
            return;
        };
        if boundary_event.event.event_id <= last_applied_id {
            self.pending_live_events.drain(..split_at);
        }
    }

    fn next_pending_drain_action(&mut self) -> PendingDrainAction {
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
}

#[derive(Debug, Default)]
pub(super) struct WorkerState {
    pub(super) epoch: u64,
    pub(super) ranks: HashMap<DpRank, RankState>,
}

impl WorkerState {
    pub(super) fn handle_discovered_rank(
        &mut self,
        dp_rank: DpRank,
        reset_rank: bool,
    ) -> DiscoveredRankAction {
        if reset_rank {
            self.epoch += 1;
            self.ranks.insert(dp_rank, RankState::default());
        }

        let rank_state = self.ranks.entry(dp_rank).or_default();
        let restore_epoch =
            if matches!(rank_state.cursor, CursorState::Initial) && !rank_state.recovery_inflight {
                rank_state.recovery_inflight = true;
                Some(self.epoch)
            } else {
                None
            };

        DiscoveredRankAction {
            reset_rank,
            restore_epoch,
        }
    }

    pub(super) fn remove_rank(&mut self, dp_rank: DpRank) -> bool {
        self.ranks.remove(&dp_rank).is_some()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    pub(super) fn observe_live_event(&mut self, event: RouterEvent) -> LiveEventAction {
        let dp_rank = event.event.dp_rank;
        let event_id = event.event.event_id;

        if matches!(&event.event.data, KvCacheEventData::Cleared) {
            let rank_state = self.ranks.entry(dp_rank).or_default();
            if rank_state
                .last_applied_id()
                .is_some_and(|last_applied_id| event_id <= last_applied_id)
            {
                return LiveEventAction::Ignore;
            }

            self.apply_worker_clear_barrier(dp_rank, event_id);
            return LiveEventAction::ApplyClear(event);
        }

        let rank_state = self.ranks.entry(dp_rank).or_default();
        match rank_state.cursor.observe(event_id) {
            CursorObservation::Stale { .. } => LiveEventAction::Ignore,
            observation if rank_state.recovery_inflight => {
                match observation {
                    CursorObservation::Initial { .. }
                    | CursorObservation::Contiguous { .. }
                    | CursorObservation::Gap { .. }
                    | CursorObservation::FreshAfterBarrier { .. } => {
                        rank_state.observe_and_buffer_live_event(event);
                    }
                    CursorObservation::Stale { .. } => {}
                }
                LiveEventAction::Ignore
            }
            CursorObservation::Initial { .. } => {
                rank_state.observe_and_buffer_live_event(event);
                rank_state.recovery_inflight = true;
                LiveEventAction::SpawnFullRestore { epoch: self.epoch }
            }
            CursorObservation::Gap { expected, .. } => {
                rank_state.observe_and_buffer_live_event(event);
                rank_state.recovery_inflight = true;
                LiveEventAction::SpawnIncremental {
                    epoch: self.epoch,
                    start_event_id: expected,
                }
            }
            CursorObservation::Contiguous { got }
            | CursorObservation::FreshAfterBarrier { got, .. } => {
                rank_state.cursor = rank_state.cursor.advance_to(got);
                rank_state.clear_max_seen_if_caught_up(got);
                LiveEventAction::ApplyDirect(event)
            }
        }
    }

    pub(super) fn apply_worker_clear_barrier(
        &mut self,
        clear_dp_rank: DpRank,
        clear_event_id: u64,
    ) {
        self.epoch += 1;
        for rank_state in self.ranks.values_mut() {
            let post_clear_max_seen = rank_state
                .max_seen_live_id
                .filter(|max_seen| *max_seen > clear_event_id);
            rank_state.cursor = rank_state.cursor.invalidate_by_barrier();
            rank_state.max_seen_live_id = post_clear_max_seen;
            rank_state.recovery_inflight = false;
            rank_state.clear_pending_live_events();
        }

        let rank_state = self.ranks.entry(clear_dp_rank).or_default();
        rank_state.cursor = rank_state.cursor.apply_barrier(clear_event_id);
    }

    pub(super) fn rank_cursor(&self, dp_rank: DpRank) -> Option<CursorState> {
        self.ranks.get(&dp_rank).map(|rank_state| rank_state.cursor)
    }

    pub(super) fn begin_successful_recovery_drain(&mut self, dp_rank: DpRank, cursor: CursorState) {
        let rank_state = self
            .ranks
            .get_mut(&dp_rank)
            .expect("rank state should exist while finishing recovery");
        rank_state.cursor = cursor;
        rank_state.recovery_inflight = true;
    }

    pub(super) fn next_pending_drain_action(&mut self, dp_rank: DpRank) -> PendingDrainAction {
        let Some(rank_state) = self.ranks.get_mut(&dp_rank) else {
            return PendingDrainAction::Complete;
        };
        rank_state.next_pending_drain_action()
    }

    pub(super) fn finish_failed_recovery(&mut self, dp_rank: DpRank) {
        let Some(rank_state) = self.ranks.get_mut(&dp_rank) else {
            return;
        };
        rank_state.recovery_inflight = false;
        rank_state.clear_pending_live_events();
    }

    /// Restart with a full dump after an external generation fence rejects a
    /// stale snapshot. Pending events were observed against the rejected dump;
    /// discard them because the fresh tree dump becomes the new base and new
    /// arrivals will be buffered behind this in-flight marker.
    pub(super) fn begin_full_restore_retry(&mut self, dp_rank: DpRank) -> Option<u64> {
        let rank_state = self.ranks.get_mut(&dp_rank)?;
        rank_state.cursor = CursorState::Initial;
        rank_state.clear_pending_live_events();
        rank_state.recovery_inflight = true;
        Some(self.epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };

    fn make_store_event(event_id: u64) -> RouterEvent {
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

    fn pending_ids(rank_state: &RankState) -> Vec<u64> {
        rank_state
            .pending_live_events
            .iter()
            .map(|event| event.event.event_id)
            .collect()
    }

    #[test]
    fn test_pending_fast_prune_keeps_margin_before_normal_drain() {
        let mut rank_state = RankState {
            cursor: CursorState::Live(35),
            recovery_inflight: true,
            ..Default::default()
        };
        for event_id in 1..=40 {
            rank_state.push_pending_live_event(make_store_event(event_id));
        }

        match rank_state.next_pending_drain_action() {
            PendingDrainAction::Apply(event) => assert_eq!(event.event.event_id, 36),
            _ => panic!("expected first non-stale contiguous pending event to apply"),
        }
        assert_eq!(pending_ids(&rank_state), vec![37, 38, 39, 40]);
    }

    #[test]
    fn test_pending_stale_gappy_queue_after_recovery_completes_without_follow_up() {
        let mut rank_state = RankState {
            cursor: CursorState::Live(20),
            recovery_inflight: true,
            ..Default::default()
        };
        for event_id in [13, 14, 18, 19] {
            rank_state.push_pending_live_event(make_store_event(event_id));
        }

        match rank_state.next_pending_drain_action() {
            PendingDrainAction::Complete => {}
            _ => panic!("expected stale pending queue to complete without follow-up"),
        }
        assert!(rank_state.pending_live_events.is_empty());
        assert!(!rank_state.recovery_inflight);
    }

    #[test]
    fn stale_external_generation_restarts_a_clean_full_restore() {
        let mut worker = WorkerState {
            epoch: 7,
            ..Default::default()
        };
        worker.ranks.insert(
            3,
            RankState {
                cursor: CursorState::Live(10),
                recovery_inflight: false,
                pending_live_events: VecDeque::from([make_store_event(11)]),
                ..Default::default()
            },
        );

        assert_eq!(worker.begin_full_restore_retry(3), Some(7));
        let rank = worker.ranks.get(&3).unwrap();
        assert!(matches!(rank.cursor, CursorState::Initial));
        assert!(rank.recovery_inflight);
        assert!(rank.pending_live_events.is_empty());
    }
}
