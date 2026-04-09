// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashSet, VecDeque};

use crate::cache::radix_cache::RadixCache;
use crate::kv_manager::SglangKvManager;

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
            let total_tokens = waiting
                .iter()
                .map(SglangRequest::current_sequence_len)
                .sum::<usize>()
                .max(page_size);
            let mut waiting_queue_cache = RadixCache::new(total_tokens, page_size);
            let mut temporary_deprioritized = HashSet::new();
            let mut scored = Vec::with_capacity(waiting.len());

            for req in waiting.drain(..) {
                let sequence = req.sequence_tokens();
                let prefix_len = kv_manager.cache().prefix_match_len(&sequence);

                if prefix_len <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD {
                    let in_batch_prefix = waiting_queue_cache.prefix_match_len(&sequence);
                    if in_batch_prefix >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD {
                        temporary_deprioritized.insert(req.uuid);
                    } else if !sequence.is_empty() {
                        let values: Vec<usize> = (0..sequence.len()).collect();
                        waiting_queue_cache.insert(&sequence, &values);
                    }
                }

                scored.push((prefix_len, temporary_deprioritized.contains(&req.uuid), req));
            }

            scored.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));

            for (_, _, req) in scored {
                waiting.push_back(req);
            }
        }
    }
}
