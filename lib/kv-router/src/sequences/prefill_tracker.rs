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
}
