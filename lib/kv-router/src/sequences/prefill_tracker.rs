// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use rustc_hash::FxHashMap;
use tokio::time::Instant;

use super::single::RequestId;
use crate::protocols::WorkerWithDpRank;

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
pub(crate) enum PrefillTimeLoadError {
    MissingExpectedDuration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillTimeLoad {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) modeled_remaining_prefill_time_ms: Result<u64, PrefillTimeLoadError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillLoadSnapshot {
    pub(super) prefill_full_tokens_sum: usize,
    pub(super) anchored_prefill: Option<AnchoredPrefillSnapshot>,
    pub(super) total_modeled_prefill_time_ms: Option<u64>,
}

impl Default for PrefillLoadSnapshot {
    fn default() -> Self {
        Self {
            prefill_full_tokens_sum: 0,
            anchored_prefill: None,
            total_modeled_prefill_time_ms: Some(0),
        }
    }
}

impl PrefillLoadSnapshot {
    pub(super) fn active_tokens_at(&self, now: Instant) -> usize {
        let Some(anchored_prefill) = self.anchored_prefill else {
            return 0;
        };

        if self.total_modeled_prefill_time_ms.is_none() {
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

    /// Return modeled remaining prefill time in non-negative milliseconds.
    ///
    /// Formula:
    /// `total_modeled_prefill_ms.saturating_sub(elapsed_since_oldest_anchor_ms)`.
    ///
    /// Elapsed time from the oldest active prefill is intentionally applied to the
    /// aggregate modeled backlog, so work can spill past the front request and
    /// reduce later modeled prefills. The final worker backlog is clipped at zero.
    ///
    /// `Err(MissingExpectedDuration)` means at least one active prefill was added
    /// without a modeled duration, which is expected when AIC is absent or
    /// prediction failed. In that case the snapshot was created via the fast path
    /// and did not scan active prefills to sum modeled time.
    pub(super) fn modeled_remaining_prefill_time_ms_at(
        &self,
        now: Instant,
    ) -> Result<u64, PrefillTimeLoadError> {
        let total_modeled_ms = self
            .total_modeled_prefill_time_ms
            .ok_or(PrefillTimeLoadError::MissingExpectedDuration)?;

        let Some(anchored_prefill) = self.anchored_prefill else {
            return Ok(total_modeled_ms);
        };

        anchored_prefill
            .expected_prefill_duration
            .ok_or(PrefillTimeLoadError::MissingExpectedDuration)?;
        let elapsed_ms =
            duration_millis_u64(now.saturating_duration_since(anchored_prefill.anchored_since));
        Ok(total_modeled_ms.saturating_sub(elapsed_ms))
    }
}

fn duration_millis_u64(duration: Duration) -> u64 {
    let millis = duration.as_millis();
    millis.min(u64::MAX as u128) as u64
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

#[derive(Debug, Default)]
pub(super) struct PrefillLoadTracker {
    pub(super) prefills: HashMap<RequestId, PrefillLoadState>,
    pub(super) prefill_order: VecDeque<RequestId>,
    pub(super) prefill_full_tokens_sum: usize,
    pub(super) unmodeled_prefill_count: usize,
    /// The front of `prefill_order` plus its effective decay anchor time.
    ///
    /// This anchors both token decay and modeled-time decay. It starts as the
    /// local time the front request became oldest, may shift forward when a
    /// modeled non-front prefill is removed, and is capped at that removal time.
    pub(super) anchored_prefill: Option<(RequestId, Instant)>,
}

impl PrefillLoadTracker {
    pub(super) fn insert(
        &mut self,
        request_id: &RequestId,
        prefill: PrefillLoadState,
        decay_now: Instant,
    ) {
        self.prefills.insert(request_id.clone(), prefill);
        self.prefill_full_tokens_sum += prefill.initial_effective_prefill_tokens;
        if prefill.expected_prefill_duration.is_none() {
            self.unmodeled_prefill_count += 1;
        }
        let should_anchor = self.anchored_prefill.is_none();
        self.prefill_order.push_back(request_id.clone());
        if should_anchor {
            self.anchored_prefill = Some((request_id.clone(), decay_now));
        }
    }

    pub(super) fn remove(
        &mut self,
        request_id: &RequestId,
        decay_now: Instant,
    ) -> Option<PrefillLoadState> {
        let prefill = self.prefills.remove(request_id)?;
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
        let removed_front = self.prefill_order.front() == Some(request_id);
        if removed_front {
            let removed = self.prefill_order.pop_front();
            debug_assert_eq!(removed.as_ref(), Some(request_id));
        } else {
            self.prefill_order
                .retain(|queued_request_id| queued_request_id != request_id);
            if let Some(expected_prefill_duration) = prefill.expected_prefill_duration {
                self.shift_anchor_forward(expected_prefill_duration, decay_now);
            }
        }
        if self
            .anchored_prefill
            .as_ref()
            .is_some_and(|(anchored_request_id, _)| anchored_request_id == request_id)
        {
            self.set_anchor_to_front(decay_now);
        }
        Some(prefill)
    }

    fn shift_anchor_forward(&mut self, duration: Duration, max_anchor_time: Instant) {
        let Some((_, anchored_since)) = self.anchored_prefill.as_mut() else {
            return;
        };
        let shifted = anchored_since
            .checked_add(duration)
            .unwrap_or(max_anchor_time);
        *anchored_since = shifted.min(max_anchor_time);
    }

    pub(super) fn set_anchor_to_front(&mut self, now: Instant) {
        self.anchored_prefill = self
            .prefill_order
            .front()
            .cloned()
            .map(|request_id| (request_id, now));
    }

    pub(super) fn snapshot(&self) -> PrefillLoadSnapshot {
        let total_modeled_prefill_time_ms = if self.unmodeled_prefill_count > 0 {
            None
        } else {
            // TODO: This all-modeled path walks active prefills on snapshot refresh even if the
            // modeled-time diagnostic read is never consumed. Consider moving this to an on-demand
            // worker-slot read if snapshot-time O(active_prefills) work becomes undesirable.
            let sum = self
                .prefill_order
                .iter()
                .map(|request_id| {
                    let prefill = self
                        .prefills
                        .get(request_id)
                        .expect("prefill_order references missing request state");
                    duration_millis_u64(
                        prefill
                            .expected_prefill_duration
                            .expect("modeled snapshot saw unmodeled prefill after zero count"),
                    )
                })
                .fold(0_u64, |acc, millis| acc.saturating_add(millis));
            Some(sum)
        };

        PrefillLoadSnapshot {
            prefill_full_tokens_sum: self.prefill_full_tokens_sum,
            anchored_prefill: self
                .anchored_prefill
                .as_ref()
                .map(|(request_id, anchored_since)| {
                    let prefill = self
                        .prefills
                        .get(request_id)
                        .copied()
                        .expect("anchored prefill missing request state");
                    AnchoredPrefillSnapshot {
                        initial_effective_prefill_tokens: prefill.initial_effective_prefill_tokens,
                        expected_prefill_duration: prefill.expected_prefill_duration,
                        anchored_since: *anchored_since,
                    }
                }),
            total_modeled_prefill_time_ms,
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub(super) fn assert_consistent(&self) {
        let active_prefills: std::collections::HashSet<RequestId> =
            self.prefills.keys().cloned().collect();
        let ordered_prefills: std::collections::HashSet<RequestId> =
            self.prefill_order.iter().cloned().collect();
        let recomputed_prefill_sum: usize = self
            .prefills
            .values()
            .map(|prefill| prefill.initial_effective_prefill_tokens)
            .sum();
        let recomputed_unmodeled_prefill_count = self
            .prefills
            .values()
            .filter(|prefill| prefill.expected_prefill_duration.is_none())
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
        if let Some(oldest_request_id) = self.prefill_order.front() {
            let Some((anchored_request_id, _)) = self.anchored_prefill.as_ref() else {
                panic!("anchored_prefill must exist when prefill_order is non-empty");
            };
            assert!(
                self.prefills.contains_key(oldest_request_id),
                "prefill_order front must point to an active prefill request",
            );
            assert_eq!(
                anchored_request_id, oldest_request_id,
                "anchored_prefill must match prefill_order.front()",
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
    fn modeled_remaining_snapshot_without_anchor_reports_zero() {
        let tracker = PrefillLoadTracker::default();
        let snapshot = tracker.snapshot();

        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(Instant::now()),
            Ok(0)
        );
    }

    #[test]
    fn modeled_remaining_no_aic_prefill_returns_missing_duration_error() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();

        tracker.insert(&r1, unmodeled_prefill_state(100), epoch);
        tracker.assert_consistent();

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch),
            Err(PrefillTimeLoadError::MissingExpectedDuration)
        );
    }

    #[test]
    fn modeled_remaining_mixed_modeled_and_unmodeled_prefills_return_error() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        tracker.insert(&r1, prefill_state(100, 10), epoch);
        tracker.insert(&r2, unmodeled_prefill_state(60), epoch);
        tracker.assert_consistent();

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(2)),
            Err(PrefillTimeLoadError::MissingExpectedDuration)
        );
    }

    #[test]
    fn modeled_remaining_single_prefill_clips_at_zero() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();

        tracker.insert(&r1, prefill_state(100, 10), epoch);

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(15)),
            Ok(0)
        );
    }

    #[test]
    fn modeled_remaining_subtracts_elapsed_from_total_modeled_duration() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        tracker.insert(&r1, prefill_state(100, 10), epoch);
        tracker.insert(&r2, prefill_state(60, 6), epoch + Duration::from_secs(2));

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(2)),
            Ok(14_000)
        );
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(12)),
            Ok(4_000)
        );
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(17)),
            Ok(0)
        );
    }

    #[test]
    fn modeled_remaining_zero_duration_prefill_contributes_no_time() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        tracker.insert(&r1, prefill_state_ms(100, 0), epoch);
        tracker.insert(&r2, prefill_state(60, 6), epoch);

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch),
            Ok(6_000)
        );
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(3)),
            Ok(3_000)
        );
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(7)),
            Ok(0)
        );
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
        assert_eq!(
            before.modeled_remaining_prefill_time_ms_at(completion_time),
            Ok(9_000)
        );

        assert_eq!(tracker.remove(&r2, completion_time), Some(p2));

        assert_eq!(tracker.prefill_order, VecDeque::from([r1.clone()]));
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
        assert_eq!(
            after.modeled_remaining_prefill_time_ms_at(completion_time),
            Ok(9_000)
        );
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
            snapshot.modeled_remaining_prefill_time_ms_at(completion_time),
            Ok(10_000)
        );
        assert_eq!(
            snapshot.active_tokens_at(completion_time + Duration::from_secs(1)),
            90
        );
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(completion_time + Duration::from_secs(1)),
            Ok(9_000)
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

        assert_eq!(tracker.prefill_order, VecDeque::from([r2.clone()]));
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
        assert_eq!(
            snapshot.modeled_remaining_prefill_time_ms_at(epoch + Duration::from_secs(5)),
            Ok(6_000)
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

        assert_eq!(tracker.prefill_order, VecDeque::from([r1.clone()]));
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
        assert_eq!(tracker.prefill_order, VecDeque::from([r2.clone()]));

        assert_eq!(tracker.remove(&r2, epoch), Some(p2));
        assert_eq!(tracker.remove(&r2, epoch), None);
        tracker.assert_consistent();
        assert_eq!(tracker.prefill_full_tokens_sum, 0);
        assert!(tracker.prefill_order.is_empty());
        assert!(tracker.prefills.is_empty());
    }
}
