// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use rustc_hash::FxHashMap;
use tokio::time::Instant;

use super::single::RequestId;
use dynamo_router_protocol::worker::WorkerWithDpRank;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillLoadState {
    pub(super) initial_effective_prefill_tokens: usize,
    pub(super) expected_prefill_duration: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AnchoredPrefillSnapshot {
    pub(super) initial_effective_prefill_tokens: usize,
    pub(super) expected_prefill_duration: Option<Duration>,
    /// Effective decay anchor time for the oldest active prefill.
    ///
    /// This usually starts as the router time when the request became oldest,
    /// but it may be shifted forward when modeled non-anchor prefills complete
    /// out of order. The elapsed time from this effective anchor is applied to
    /// the aggregate all-modeled backlog.
    pub(super) anchored_since: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillLoadSnapshot {
    /// Includes discarded progress when all prefills have modeled durations.
    pub(super) prefill_full_tokens_sum: usize,
    pub(super) anchored_prefill: Option<AnchoredPrefillSnapshot>,
    /// True when every active prefill carries an expected duration, so the
    /// anchor's decay can be applied to the aggregate backlog.
    pub(super) all_prefills_modeled: bool,
}

// Stored per worker rank in the load table that every projection walks, so growing it
// widens every entry. Measure the projection before changing this size.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const _: () = assert!(std::mem::size_of::<PrefillLoadSnapshot>() == 56);

impl Default for PrefillLoadSnapshot {
    fn default() -> Self {
        Self {
            prefill_full_tokens_sum: 0,
            anchored_prefill: None,
            all_prefills_modeled: true,
        }
    }
}

impl PrefillLoadSnapshot {
    /// A `now` earlier than credit settlement can read above the booked total.
    pub(super) fn active_tokens_at(&self, now: Instant) -> usize {
        let Some(anchored_prefill) = self.anchored_prefill else {
            return 0;
        };

        if !self.all_prefills_modeled {
            return self.anchor_only_active_tokens_at(anchored_prefill, now);
        }

        let completed_tokens = completed_prefill_tokens(
            anchored_prefill.initial_effective_prefill_tokens,
            anchored_prefill.expected_prefill_duration,
            now.saturating_duration_since(anchored_prefill.anchored_since),
        );
        self.prefill_full_tokens_sum
            .saturating_sub(completed_tokens)
    }

    fn anchor_only_active_tokens_at(
        &self,
        anchored_prefill: AnchoredPrefillSnapshot,
        now: Instant,
    ) -> usize {
        let anchored_full = anchored_prefill.initial_effective_prefill_tokens;
        let anchored_remaining = match anchored_prefill.expected_prefill_duration {
            None => anchored_full,
            Some(expected_prefill_duration) if expected_prefill_duration.is_zero() => 0,
            Some(expected_prefill_duration) => {
                let elapsed = now.saturating_duration_since(anchored_prefill.anchored_since);
                let remaining_fraction = (1.0
                    - (elapsed.as_secs_f64() / expected_prefill_duration.as_secs_f64()))
                .clamp(0.0, 1.0);
                ((anchored_full as f64) * remaining_fraction).ceil() as usize
            }
        };

        self.prefill_full_tokens_sum
            .checked_sub(anchored_full)
            .expect("prefill_full_tokens_sum smaller than anchored load")
            + anchored_remaining
    }
}

fn completed_prefill_tokens(
    initial_effective_prefill_tokens: usize,
    expected_prefill_duration: Option<Duration>,
    elapsed: Duration,
) -> usize {
    let Some(expected_prefill_duration) = expected_prefill_duration else {
        return 0;
    };
    if expected_prefill_duration.is_zero() {
        return initial_effective_prefill_tokens;
    }

    let completed = (initial_effective_prefill_tokens as u128).saturating_mul(elapsed.as_nanos())
        / expected_prefill_duration.as_nanos();
    completed.min(usize::MAX as u128) as usize
}

/// Per-worker prefill token deltas already projected by the scheduler.
///
/// The sequence layer only combines this with active prefill load; it does not know how the
/// deltas were derived from ISL, cache hits, overlap scores, or shared-cache policy.
#[derive(Debug, Clone, Default)]
pub struct PrefillTokenDeltas {
    default_tokens: usize,
    by_worker: FxHashMap<WorkerWithDpRank, usize>,
}

impl PrefillTokenDeltas {
    pub fn none() -> Self {
        Self::default()
    }

    /// Test hook: the same delta for every worker.
    #[cfg(any(test, feature = "testing"))]
    pub fn uniform(tokens: usize) -> Self {
        Self {
            default_tokens: tokens,
            by_worker: FxHashMap::default(),
        }
    }

    pub fn new(default_tokens: usize, by_worker: FxHashMap<WorkerWithDpRank, usize>) -> Self {
        Self {
            default_tokens,
            by_worker,
        }
    }

    pub fn tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        self.by_worker
            .get(&worker)
            .copied()
            .unwrap_or(self.default_tokens)
    }

    pub fn default_tokens(&self) -> usize {
        self.default_tokens
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TrackedPrefill {
    load: PrefillLoadState,
    order: u64,
}

#[derive(Debug, Default)]
pub(super) struct PrefillLoadTracker {
    pub(super) prefills: HashMap<RequestId, TrackedPrefill>,
    pub(super) prefill_order: BTreeMap<u64, RequestId>,
    next_order: u64,
    pub(super) prefill_full_tokens_sum: usize,
    pub(super) unmodeled_prefill_count: usize,
    /// The front of `prefill_order` plus its effective decay anchor time.
    ///
    /// This anchors token decay. It starts as the local time the front request
    /// became oldest, may shift forward when a modeled non-front prefill is
    /// removed, and is capped at that removal time.
    pub(super) anchored_prefill: Option<(RequestId, Instant)>,
    /// Load state of the anchored request, cached so snapshots need no lookup.
    anchored_load: Option<PrefillLoadState>,
    /// Modeled token progress not charged to the active prefills: it exceeded available
    /// work before a later arrival, or a removed prefill had absorbed it.
    discarded_completed_tokens: usize,
}

impl PrefillLoadTracker {
    pub(super) fn insert(
        &mut self,
        request_id: &RequestId,
        prefill: PrefillLoadState,
        decay_now: Instant,
    ) {
        self.discard_prearrival_progress(decay_now);
        let order = self.next_order;
        self.next_order += 1;
        self.prefill_order.insert(order, request_id.clone());
        self.prefills.insert(
            request_id.clone(),
            TrackedPrefill {
                load: prefill,
                order,
            },
        );
        self.prefill_full_tokens_sum += prefill.initial_effective_prefill_tokens;
        if prefill.expected_prefill_duration.is_none() {
            self.unmodeled_prefill_count += 1;
        }
        let should_anchor = self.anchored_prefill.is_none();
        if should_anchor {
            self.anchored_prefill = Some((request_id.clone(), decay_now));
            self.anchored_load = Some(prefill);
        }
    }

    fn discard_prearrival_progress(&mut self, now: Instant) {
        let (Some((_, anchored_since)), Some(load)) =
            (self.anchored_prefill.as_ref(), self.anchored_load)
        else {
            return;
        };
        let available_tokens = if self.unmodeled_prefill_count == 0 {
            self.prefill_full_tokens_sum
        } else {
            // The mixed path only decays the anchor, not later requests.
            load.initial_effective_prefill_tokens
        };
        self.discarded_completed_tokens = self.discarded_completed_tokens.max(
            completed_prefill_tokens(
                load.initial_effective_prefill_tokens,
                load.expected_prefill_duration,
                now.saturating_duration_since(*anchored_since),
            )
            .saturating_sub(available_tokens),
        );
    }

    pub(super) fn remove(
        &mut self,
        request_id: &RequestId,
        decay_now: Instant,
    ) -> Option<PrefillLoadState> {
        let tracked = self.prefills.remove(request_id)?;
        let prefill = tracked.load;
        self.prefill_full_tokens_sum = self
            .prefill_full_tokens_sum
            .checked_sub(prefill.initial_effective_prefill_tokens)
            .expect("prefill_full_tokens_sum underflow");
        if prefill.expected_prefill_duration.is_none() {
            self.unmodeled_prefill_count = self
                .unmodeled_prefill_count
                .checked_sub(1)
                .expect("unmodeled_prefill_count underflow");
        }
        let removed = self.prefill_order.remove(&tracked.order);
        debug_assert_eq!(removed.as_ref(), Some(request_id));
        let removed_front = self
            .anchored_prefill
            .as_ref()
            .is_some_and(|(anchored_id, _)| anchored_id == request_id);
        if !removed_front && let Some(expected_prefill_duration) = prefill.expected_prefill_duration
        {
            self.shift_anchor_forward(tracked, expected_prefill_duration, decay_now);
        }
        if removed_front {
            self.set_anchor_to_front(decay_now);
        }
        Some(prefill)
    }

    fn shift_anchor_forward(
        &mut self,
        removed: TrackedPrefill,
        duration: Duration,
        max_anchor_time: Instant,
    ) {
        let (Some((_, anchored_since)), Some(load)) =
            (self.anchored_prefill.as_mut(), self.anchored_load)
        else {
            return;
        };
        let progress_since = |since: Instant| {
            completed_prefill_tokens(
                load.initial_effective_prefill_tokens,
                load.expected_prefill_duration,
                max_anchor_time.saturating_duration_since(since),
            )
        };
        let before = progress_since(*anchored_since);
        let shifted = anchored_since
            .checked_add(duration)
            .unwrap_or(max_anchor_time);
        *anchored_since = shifted.min(max_anchor_time);
        let after = progress_since(*anchored_since);
        // Progress fills the active prefills oldest first. Discard what had reached the
        // removed prefill and the shift did not return, or it is charged to the prefills
        // that remain.
        let mut reached = before.saturating_sub(self.discarded_completed_tokens);
        for (_, request_id) in self.prefill_order.range(..removed.order) {
            if reached == 0 {
                break;
            }
            let ahead = self.prefills[request_id].load;
            reached = reached.saturating_sub(ahead.initial_effective_prefill_tokens);
        }
        let absorbed = reached.min(removed.load.initial_effective_prefill_tokens);
        self.discarded_completed_tokens = self
            .discarded_completed_tokens
            .saturating_add(absorbed.saturating_sub(before.saturating_sub(after)))
            .min(after);
    }

    pub(super) fn set_anchor_to_front(&mut self, now: Instant) {
        self.discarded_completed_tokens = 0;
        self.anchored_prefill = self
            .prefill_order
            .first_key_value()
            .map(|(_, request_id)| (request_id.clone(), now));
        self.anchored_load = self.anchored_prefill.as_ref().map(|(request_id, _)| {
            self.prefills
                .get(request_id)
                .map(|prefill| prefill.load)
                .expect("anchored prefill missing request state")
        });
    }

    pub(super) fn snapshot(&self) -> PrefillLoadSnapshot {
        let all_prefills_modeled = self.unmodeled_prefill_count == 0;
        let prefill_full_tokens_sum = if all_prefills_modeled {
            self.prefill_full_tokens_sum
                .saturating_add(self.discarded_completed_tokens)
        } else {
            self.prefill_full_tokens_sum
        };

        PrefillLoadSnapshot {
            prefill_full_tokens_sum,
            anchored_prefill: self.anchored_prefill.as_ref().map(|(_, anchored_since)| {
                let prefill = self
                    .anchored_load
                    .expect("anchored prefill missing cached load state");
                AnchoredPrefillSnapshot {
                    initial_effective_prefill_tokens: prefill.initial_effective_prefill_tokens,
                    expected_prefill_duration: prefill.expected_prefill_duration,
                    anchored_since: *anchored_since,
                }
            }),
            all_prefills_modeled,
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent(&self) {
        let active_prefills: std::collections::HashSet<RequestId> =
            self.prefills.keys().cloned().collect();
        let ordered_prefills: std::collections::HashSet<RequestId> =
            self.prefill_order.values().cloned().collect();
        let recomputed_prefill_sum: usize = self
            .prefills
            .values()
            .map(|prefill| prefill.load.initial_effective_prefill_tokens)
            .sum();
        assert_eq!(
            self.anchored_load,
            self.anchored_prefill
                .as_ref()
                .and_then(|(request_id, _)| self
                    .prefills
                    .get(request_id)
                    .map(|prefill| prefill.load)),
            "anchored_load drifted from the anchored request state",
        );
        let recomputed_unmodeled_prefill_count = self
            .prefills
            .values()
            .filter(|prefill| prefill.load.expected_prefill_duration.is_none())
            .count();

        assert_eq!(
            ordered_prefills.len(),
            self.prefill_order.len(),
            "prefill_order contains duplicate request ids",
        );
        assert_eq!(
            ordered_prefills, active_prefills,
            "prefill_order must match active prefill requests",
        );
        assert_eq!(
            self.prefill_full_tokens_sum, recomputed_prefill_sum,
            "prefill_full_tokens_sum drifted from tracker state",
        );
        assert_eq!(
            self.unmodeled_prefill_count, recomputed_unmodeled_prefill_count,
            "unmodeled_prefill_count drifted from tracker state",
        );
        if let Some((_, oldest_request_id)) = self.prefill_order.first_key_value() {
            let Some((anchored_request_id, _)) = self.anchored_prefill.as_ref() else {
                panic!("anchored_prefill must exist when prefill_order is non-empty");
            };
            assert!(
                self.prefills.contains_key(oldest_request_id),
                "prefill_order front must point to an active prefill request",
            );
            assert_eq!(
                anchored_request_id, oldest_request_id,
                "anchored_prefill must match the oldest ordered prefill",
            );
        } else {
            assert!(
                self.anchored_prefill.is_none(),
                "anchored_prefill must be absent when no active prefills remain",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill_state(tokens: usize, duration_secs: u64) -> PrefillLoadState {
        PrefillLoadState {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: Some(Duration::from_secs(duration_secs)),
        }
    }

    fn prefill_state_ms(tokens: usize, duration_ms: u64) -> PrefillLoadState {
        PrefillLoadState {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: Some(Duration::from_millis(duration_ms)),
        }
    }

    fn unmodeled_prefill_state(tokens: usize) -> PrefillLoadState {
        PrefillLoadState {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: None,
        }
    }

    #[test]
    fn snapshot_without_anchor_reports_zero_active_tokens() {
        let tracker = PrefillLoadTracker::default();
        let snapshot = tracker.snapshot();

        assert_eq!(snapshot.active_tokens_at(Instant::now()), 0);
    }

    #[test]
    fn snapshot_only_decays_oldest_prefill() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = prefill_state(60, 6);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch + Duration::from_secs(2));

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(2)),
            140
        );
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(5)),
            110
        );
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(12)),
            40
        );
    }

    #[test]
    fn active_tokens_with_unmodeled_prefills_use_anchor_only_decay() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        tracker.insert(&r1, prefill_state(100, 10), epoch);
        tracker.insert(&r2, unmodeled_prefill_state(60), epoch);

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(12)),
            60
        );
    }

    #[test]
    fn active_tokens_zero_duration_anchor_only_completes_anchor() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        tracker.insert(&r1, prefill_state_ms(100, 0), epoch);
        tracker.insert(&r2, prefill_state(60, 6), epoch);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.active_tokens_at(epoch), 60);
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(3)),
            60
        );
    }

    #[test]
    fn nonfront_modeled_removal_preserves_time_and_token_continuity() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = prefill_state(40, 4);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);

        let completion_time = epoch + Duration::from_secs(5);
        let before = tracker.snapshot();
        assert_eq!(before.active_tokens_at(completion_time), 90);

        assert_eq!(tracker.remove(&r2, completion_time), Some(p2));

        assert_eq!(tracker.prefill_order.values().collect::<Vec<_>>(), [&r1]);
        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, anchored_since)| {
                    request_id == &r1 && *anchored_since == epoch + Duration::from_secs(4)
                })
        );

        let after = tracker.snapshot();
        assert_eq!(after.active_tokens_at(completion_time), 90);
    }

    #[test]
    fn nonfront_modeled_removal_caps_anchor_shift_at_removal_time() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = prefill_state(40, 4);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);

        let completion_time = epoch + Duration::from_secs(2);
        assert_eq!(tracker.remove(&r2, completion_time), Some(p2));

        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, anchored_since)| {
                    request_id == &r1 && *anchored_since == completion_time
                })
        );

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.active_tokens_at(completion_time), 100);
        assert_eq!(
            snapshot.active_tokens_at(completion_time + Duration::from_secs(1)),
            90
        );
    }

    #[test]
    fn nonfront_unmodeled_removal_does_not_shift_anchor() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = unmodeled_prefill_state(40);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);

        assert_eq!(
            tracker.remove(&r2, epoch + Duration::from_secs(5)),
            Some(p2)
        );

        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, anchored_since)| {
                    request_id == &r1 && *anchored_since == epoch
                })
        );
        assert_eq!(
            tracker
                .snapshot()
                .active_tokens_at(epoch + Duration::from_secs(5)),
            50
        );
    }

    #[test]
    fn removing_anchored_prefill_reanchors_front_and_resets_decay() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = prefill_state(40, 8);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);

        assert_eq!(
            tracker.remove(&r1, epoch + Duration::from_secs(3)),
            Some(p1)
        );

        assert_eq!(tracker.prefill_order.values().collect::<Vec<_>>(), [&r2]);
        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, _)| request_id == &r2)
        );

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(3)),
            40
        );
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(5)),
            30
        );
    }

    #[test]
    fn removing_nonfront_modeled_prefill_shifts_existing_anchor() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(30, 6);
        let p2 = prefill_state(20, 4);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);

        assert_eq!(
            tracker.remove(&r2, epoch + Duration::from_secs(2)),
            Some(p2)
        );

        assert_eq!(tracker.prefill_order.values().collect::<Vec<_>>(), [&r1]);
        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, anchored_since)| {
                    request_id == &r1 && *anchored_since == epoch + Duration::from_secs(2)
                })
        );

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(2)),
            30
        );
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(4)),
            20
        );
    }

    #[test]
    fn duplicate_cleanup_is_idempotent() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();
        let p1 = prefill_state(50, 10);
        let p2 = prefill_state(30, 10);

        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);
        tracker.assert_consistent();

        assert_eq!(tracker.remove(&r1, epoch), Some(p1));
        assert_eq!(tracker.remove(&r1, epoch), None);
        assert_eq!(tracker.prefill_full_tokens_sum, 30);
        assert_eq!(tracker.prefill_order.values().collect::<Vec<_>>(), [&r2]);

        assert_eq!(tracker.remove(&r2, epoch), Some(p2));
        assert_eq!(tracker.remove(&r2, epoch), None);
        tracker.assert_consistent();
        assert_eq!(tracker.prefill_full_tokens_sum, 0);
        assert!(tracker.prefill_order.is_empty());
        assert!(tracker.prefills.is_empty());
    }

    #[test]
    fn interleaved_removal_and_id_reuse_preserve_oldest_prefill() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let anchor = "anchor".to_string();
        let state = unmodeled_prefill_state(64);
        tracker.insert(&anchor, state, epoch);
        for _ in 0..2 {
            let ids: Vec<_> = (0..32).map(|i| format!("request-{i}")).collect();
            for id in &ids {
                tracker.insert(id, state, epoch);
            }
            for index in (0..32).step_by(2).chain((1..32).step_by(2).rev()) {
                assert_eq!(tracker.remove(&ids[index], epoch), Some(state));
                assert_eq!(tracker.remove(&ids[index], epoch), None);
                tracker.assert_consistent();
            }
            assert_eq!(tracker.prefill_order.values().next(), Some(&anchor));
            assert_eq!(tracker.prefill_order.len(), 1);
            assert_eq!(tracker.snapshot().active_tokens_at(epoch), 64);
        }
        assert_eq!(tracker.remove(&anchor, epoch), Some(state));
        assert!(tracker.prefill_order.is_empty());
        assert!(tracker.prefills.is_empty());
        tracker.insert(&anchor, state, epoch);
        tracker.assert_consistent();
        assert_eq!(tracker.prefill_order.values().next(), Some(&anchor));
        assert_eq!(tracker.prefill_order.len(), 1);
    }

    #[test]
    fn new_arrival_preserves_its_full_token_work() {
        for (case, first, second, arrival_ms) in [
            ("long overrun", prefill_state_ms(100, 10), None, 30),
            (
                "aggregate backlog drained",
                prefill_state_ms(100, 10),
                Some(prefill_state_ms(10, 100)),
                30,
            ),
            (
                "aggregate backlog remains",
                prefill_state_ms(10, 10),
                Some(prefill_state_ms(100, 1)),
                30,
            ),
            (
                "zero-duration anchor",
                prefill_state_ms(100, 0),
                Some(prefill_state_ms(60, 6)),
                30,
            ),
        ] {
            let epoch = Instant::now();
            let now = epoch + Duration::from_millis(arrival_ms);
            let mut tracker = PrefillLoadTracker::default();
            tracker.insert(&"A".into(), first, epoch);
            if let Some(second) = second {
                tracker.insert(&"older".into(), second, epoch);
            }
            let before = tracker.snapshot();
            tracker.insert(&"new".into(), prefill_state_ms(100, 10), now);
            tracker.assert_consistent();
            let after = tracker.snapshot();
            assert_eq!(
                after.active_tokens_at(now),
                before.active_tokens_at(now) + 100,
                "{case}",
            );
        }
    }

    #[test]
    fn arrival_progress_survives_nonfront_completion_and_resets_with_new_front() {
        // Removing C shifts the anchor and bounds discarded progress. The capped
        // case leaves no elapsed progress; the partial shift leaves 20 tokens.
        for (case, duration, remaining_tokens) in [
            ("partial anchor shift", 10, 180),
            ("anchor shift capped at completion", 100, 200),
        ] {
            let epoch = Instant::now();
            let mut tracker = PrefillLoadTracker::default();
            tracker.insert(&"A".into(), prefill_state_ms(100, 10), epoch);
            tracker.insert(
                &"B".into(),
                prefill_state_ms(100, 10),
                epoch + Duration::from_millis(30),
            );
            let next = epoch + Duration::from_millis(31);
            assert_eq!(tracker.snapshot().active_tokens_at(next), 90, "{case}");
            tracker.insert(&"C".into(), prefill_state_ms(100, duration), next);
            let completion = epoch + Duration::from_millis(32);
            tracker.remove(&"C".into(), completion);
            tracker.assert_consistent();
            let after = tracker.snapshot();
            assert_eq!(
                after.active_tokens_at(completion),
                remaining_tokens,
                "{case}"
            );
            let later = completion + Duration::from_millis(1);
            assert_eq!(
                after.active_tokens_at(later),
                remaining_tokens - 10,
                "{case}"
            );
            tracker.remove(&"A".into(), later);
            assert_eq!(tracker.snapshot().active_tokens_at(later), 100, "{case}");
            let after_front_change = later + Duration::from_millis(1);
            assert_eq!(
                tracker.snapshot().active_tokens_at(after_front_change),
                90,
                "{case}",
            );
            tracker.remove(&"B".into(), after_front_change);
            let reuse = epoch + Duration::from_millis(100);
            tracker.insert(&"A".into(), prefill_state_ms(100, 10), reuse);
            tracker.assert_consistent();
            assert_eq!(tracker.snapshot().active_tokens_at(reuse), 100, "{case}");
        }
    }

    #[test]
    fn modeled_arrival_is_not_erased_when_unmodeled_work_is_removed() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        tracker.insert(&"A".into(), prefill_state_ms(100, 10), epoch);
        tracker.insert(&"unknown".into(), unmodeled_prefill_state(100), epoch);
        let now = epoch + Duration::from_millis(30);
        tracker.insert(&"B".into(), prefill_state_ms(100, 10), now);
        assert_eq!(tracker.snapshot().active_tokens_at(now), 200);
        tracker.remove(&"unknown".into(), now);
        tracker.assert_consistent();
        assert_eq!(tracker.snapshot().active_tokens_at(now), 100);
        let later = now + Duration::from_millis(1);
        assert_eq!(tracker.snapshot().active_tokens_at(later), 90);
    }

    #[test]
    fn stale_read_instant_remains_conservative_after_later_arrival() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        tracker.insert(&"A".into(), prefill_state_ms(100, 10), epoch);
        let arrival = epoch + Duration::from_millis(30);
        tracker.insert(&"B".into(), prefill_state_ms(100, 10), arrival);
        let snapshot = tracker.snapshot();
        // A reader can capture its instant before a concurrent writer publishes.
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_millis(10)),
            300
        );
        assert_eq!(snapshot.active_tokens_at(arrival), 100);
    }

    #[test]
    fn nonfront_removal_keeps_later_arrival_booked() {
        // A is 100 tokens / 10 ms and stays live past its estimate. Z books at `z_ms`,
        // B books 100 tokens at `b_ms`, and Z completes 1 ms after B books.
        for (case, z_ms, z_tokens, b_ms, expected) in [
            ("Z at the anchor's rate", 1, 100, 50, 90),
            ("Z at 10x the anchor's rate", 1, 1000, 50, 100),
            ("Z at 10x, backlog drained before B", 1, 1000, 200, 90),
            ("Z at 10x, booked after a discard", 200, 1000, 230, 100),
        ] {
            let epoch = Instant::now();
            let ms = |n: u64| epoch + Duration::from_millis(n);
            let mut tracker = PrefillLoadTracker::default();
            tracker.insert(&"A".into(), prefill_state_ms(100, 10), ms(0));
            tracker.insert(&"Z".into(), prefill_state_ms(z_tokens, 10), ms(z_ms));
            tracker.insert(&"B".into(), prefill_state_ms(100, 10), ms(b_ms));
            let later = ms(b_ms + 1);
            tracker.remove(&"Z".into(), later);
            tracker.assert_consistent();
            let active = tracker.snapshot().active_tokens_at(later);
            assert_eq!(active, expected, "{case}");
        }
    }

    #[test]
    fn nonfront_removal_inside_anchor_estimate_returns_own_duration() {
        // 80 of A's 100 tokens have decayed, so X absorbed none: only its own 5 ms return.
        let epoch = Instant::now();
        let now = epoch + Duration::from_millis(8);
        let mut tracker = PrefillLoadTracker::default();
        tracker.insert(&"A".into(), prefill_state_ms(100, 10), epoch);
        tracker.insert(&"X".into(), prefill_state_ms(100, 5), now);
        tracker.remove(&"X".into(), now);
        assert_eq!(tracker.snapshot().active_tokens_at(now), 70);
    }

    #[test]
    fn nonfront_removal_behind_older_prefill_returns_own_duration() {
        // At 51 ms the 410 tokens of progress beyond A are inside Z1, so Z2 absorbed none
        // and only its own 10 ms return. Z1 then takes what it absorbed with it.
        let epoch = Instant::now();
        let ms = |n: u64| epoch + Duration::from_millis(n);
        let mut tracker = PrefillLoadTracker::default();
        tracker.insert(&"A".into(), prefill_state_ms(100, 10), ms(0));
        tracker.insert(&"Z1".into(), prefill_state_ms(1000, 10), ms(1));
        tracker.insert(&"Z2".into(), prefill_state_ms(1000, 10), ms(2));
        tracker.insert(&"B".into(), prefill_state_ms(100, 10), ms(50));
        tracker.remove(&"Z2".into(), ms(51));
        assert_eq!(tracker.snapshot().active_tokens_at(ms(51)), 790);
        tracker.remove(&"Z1".into(), ms(52));
        assert_eq!(tracker.snapshot().active_tokens_at(ms(52)), 100);
    }

    #[test]
    fn nonfront_removal_with_unmodeled_prefill_returns_own_duration() {
        // Only A decays while an unmodeled prefill is booked, and A stays past its
        // estimate after Z's own 10 ms return.
        let epoch = Instant::now();
        let ms = |n: u64| epoch + Duration::from_millis(n);
        let mut tracker = PrefillLoadTracker::default();
        tracker.insert(&"A".into(), prefill_state_ms(100, 10), ms(0));
        tracker.insert(&"unknown".into(), unmodeled_prefill_state(1), ms(0));
        tracker.insert(&"Z".into(), prefill_state_ms(1000, 10), ms(1));
        tracker.remove(&"Z".into(), ms(51));
        assert_eq!(tracker.snapshot().active_tokens_at(ms(51)), 1);
    }
}
