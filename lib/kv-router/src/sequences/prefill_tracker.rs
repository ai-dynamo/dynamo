// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use tokio::time::Instant;

use super::single::RequestId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillLoadState {
    pub(super) initial_effective_prefill_tokens: usize,
    pub(super) expected_prefill_duration: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AnchoredPrefillSnapshot {
    pub(super) initial_effective_prefill_tokens: usize,
    pub(super) expected_prefill_duration: Option<Duration>,
    pub(super) anchored_since: Instant,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillLoadSnapshot {
    pub(super) prefill_full_tokens_sum: usize,
    pub(super) anchored_prefill: Option<AnchoredPrefillSnapshot>,
}

impl PrefillLoadSnapshot {
    pub(super) fn active_tokens_at(&self, now: Instant) -> usize {
        let Some(anchored_prefill) = self.anchored_prefill else {
            return 0;
        };
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

pub(super) fn added_prefill_tokens(block_size: usize, isl: usize, overlap: u32) -> usize {
    let cached_tokens = (overlap as usize) * block_size;
    isl.checked_sub(cached_tokens).unwrap_or_else(|| {
        tracing::error!(
            "prefill_tokens < 0 with ISL {isl} < cached_tokens {cached_tokens} (overlap {overlap} * block_size {block_size}), returning 0",
        );
        0
    })
}

#[derive(Debug, Default)]
pub(super) struct PrefillLoadTracker {
    pub(super) prefills: HashMap<RequestId, PrefillLoadState>,
    pub(super) prefill_order: VecDeque<RequestId>,
    pub(super) prefill_full_tokens_sum: usize,
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
        let removed_front = self.prefill_order.front() == Some(request_id);
        if removed_front {
            let removed = self.prefill_order.pop_front();
            debug_assert_eq!(removed.as_ref(), Some(request_id));
        } else {
            self.prefill_order
                .retain(|queued_request_id| queued_request_id != request_id);
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

    pub(super) fn set_anchor_to_front(&mut self, now: Instant) {
        self.anchored_prefill = self
            .prefill_order
            .front()
            .cloned()
            .map(|request_id| (request_id, now));
    }

    pub(super) fn snapshot(&self) -> PrefillLoadSnapshot {
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
    }

    #[test]
    fn removing_nonfront_prefill_preserves_existing_anchor() {
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
                    request_id == &r1 && *anchored_since == epoch
                })
        );

        let snapshot = tracker.snapshot();
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(2)),
            21
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
