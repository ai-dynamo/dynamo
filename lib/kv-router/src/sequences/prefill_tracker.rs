// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::time::Duration;
use tokio::time::Instant;

use super::single::RequestId;

#[derive(Debug, Clone, Copy)]
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
        prefill: PrefillLoadState,
        decay_now: Instant,
    ) {
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
    }

    pub(super) fn set_anchor_to_front(&mut self, now: Instant) {
        self.anchored_prefill = self
            .prefill_order
            .front()
            .cloned()
            .map(|request_id| (request_id, now));
    }

    pub(super) fn snapshot(
        &self,
        anchored_prefill: Option<(PrefillLoadState, Instant)>,
    ) -> PrefillLoadSnapshot {
        PrefillLoadSnapshot {
            prefill_full_tokens_sum: self.prefill_full_tokens_sum,
            anchored_prefill: anchored_prefill.map(|(prefill, anchored_since)| {
                AnchoredPrefillSnapshot {
                    initial_effective_prefill_tokens: prefill.initial_effective_prefill_tokens,
                    expected_prefill_duration: prefill.expected_prefill_duration,
                    anchored_since,
                }
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn prefill_state(tokens: usize, duration_secs: u64) -> PrefillLoadState {
        PrefillLoadState {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: Some(Duration::from_secs(duration_secs)),
        }
    }

    fn snapshot_for(
        tracker: &PrefillLoadTracker,
        prefills: &HashMap<RequestId, PrefillLoadState>,
    ) -> PrefillLoadSnapshot {
        let anchored_prefill =
            tracker
                .anchored_prefill
                .as_ref()
                .map(|(request_id, anchored_since)| {
                    (
                        *prefills
                            .get(request_id)
                            .expect("anchored request must have prefill state"),
                        *anchored_since,
                    )
                });
        tracker.snapshot(anchored_prefill)
    }

    #[test]
    fn snapshot_without_anchor_reports_zero_active_tokens() {
        let tracker = PrefillLoadTracker::default();
        let snapshot = tracker.snapshot(None);

        assert_eq!(snapshot.active_tokens_at(Instant::now()), 0);
    }

    #[test]
    fn snapshot_only_decays_oldest_prefill() {
        let epoch = Instant::now();
        let mut tracker = PrefillLoadTracker::default();
        let mut prefills = HashMap::new();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = prefill_state(60, 6);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch + Duration::from_secs(2));
        prefills.insert(r1.clone(), p1);
        prefills.insert(r2, p2);

        let snapshot = snapshot_for(&tracker, &prefills);
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
        let mut prefills = HashMap::new();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(100, 10);
        let p2 = prefill_state(40, 8);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);
        prefills.insert(r1.clone(), p1);
        prefills.insert(r2.clone(), p2);

        tracker.remove(&r1, p1, epoch + Duration::from_secs(3));
        prefills.remove(&r1);

        assert_eq!(tracker.prefill_order, VecDeque::from([r2.clone()]));
        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, _)| request_id == &r2)
        );

        let snapshot = snapshot_for(&tracker, &prefills);
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
        let mut prefills = HashMap::new();
        let r1 = "r1".to_string();
        let r2 = "r2".to_string();

        let p1 = prefill_state(30, 6);
        let p2 = prefill_state(20, 4);
        tracker.insert(&r1, p1, epoch);
        tracker.insert(&r2, p2, epoch);
        prefills.insert(r1.clone(), p1);
        prefills.insert(r2.clone(), p2);

        tracker.remove(&r2, p2, epoch + Duration::from_secs(2));
        prefills.remove(&r2);

        assert_eq!(tracker.prefill_order, VecDeque::from([r1.clone()]));
        assert!(
            tracker
                .anchored_prefill
                .as_ref()
                .is_some_and(|(request_id, anchored_since)| {
                    request_id == &r1 && *anchored_since == epoch
                })
        );

        let snapshot = snapshot_for(&tracker, &prefills);
        assert_eq!(
            snapshot.active_tokens_at(epoch + Duration::from_secs(2)),
            21
        );
    }
}
