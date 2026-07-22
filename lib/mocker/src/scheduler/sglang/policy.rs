// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use crate::kv_manager::SglangKvManager;
use rustc_hash::FxHashSet;

use super::config::{
    IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD, IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD,
    LPM_FALLBACK_THRESHOLD, SchedulePolicy, SglangConfig,
};
use super::request::SglangRequest;

pub(super) fn apply_schedule_policy(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
) {
    match config.schedule_policy {
        SchedulePolicy::Fifo => {}
        SchedulePolicy::Lpm => {
            if waiting.len() > LPM_FALLBACK_THRESHOLD {
                return;
            }

            let page_size = config.block_size.max(1);
            let duplicate_prefix_len =
                IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD.div_ceil(page_size) * page_size;
            let mut waiting_prefixes = FxHashSet::default();
            let mut scored = Vec::with_capacity(waiting.len());

            for req in waiting.drain(..) {
                let sequence = req.sequence_tokens();
                let prefix_len = kv_manager.cache().prefix_match_len(&sequence);
                let deprioritized = prefix_len <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD
                    && sequence.len() >= duplicate_prefix_len
                    && !waiting_prefixes.insert(sequence[..duplicate_prefix_len].to_vec());

                scored.push((prefix_len, deprioritized, req));
            }

            scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));

            for (_, _, req) in scored {
                waiting.push_back(req);
            }
        }
    }
}
