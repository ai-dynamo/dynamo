// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::common::perf_model::PerfModel;
use crate::common::protocols::{MockEngineArgs, WorkerType};

const DEFAULT_MAX_PREFILL_TOKENS: usize = 16384;
const DEFAULT_CHUNKED_PREFILL_SIZE: usize = 8192;
const DEFAULT_CLIP_MAX_NEW_TOKENS: usize = 4096;
const DEFAULT_INIT_NEW_TOKEN_RATIO: f64 = 0.7;
const DEFAULT_MIN_NEW_TOKEN_RATIO_FACTOR: f64 = 0.14;
const DEFAULT_NEW_TOKEN_RATIO_DECAY_STEPS: f64 = 600.0;
pub(super) const LPM_FALLBACK_THRESHOLD: usize = 128;
pub(super) const IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD: usize = 32;
pub(super) const IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchedulePolicy {
    #[default]
    Fifo,
    Lpm,
}

pub(super) struct SglangConfig {
    pub(super) schedule_policy: SchedulePolicy,
    pub(super) max_prefill_tokens: usize,
    pub(super) chunked_prefill_size: usize,
    pub(super) clip_max_new_tokens: usize,
    pub(super) init_new_token_ratio: f64,
    pub(super) min_new_token_ratio: f64,
    pub(super) new_token_ratio_decay_step: f64,
    pub(super) perf_model: Arc<PerfModel>,
    pub(super) speedup_ratio: f64,
    pub(super) decode_speedup_ratio: f64,
    pub(super) worker_type: WorkerType,
    pub(super) block_size: usize,
}

impl SglangConfig {
    pub(super) fn from_args(args: &MockEngineArgs) -> Self {
        let sglang = args.sglang.as_ref();
        let schedule_conservativeness = sglang
            .and_then(|s| s.schedule_conservativeness)
            .unwrap_or(1.0);
        let init_new_token_ratio = DEFAULT_INIT_NEW_TOKEN_RATIO * schedule_conservativeness;
        let min_new_token_ratio = init_new_token_ratio * DEFAULT_MIN_NEW_TOKEN_RATIO_FACTOR;
        let decay_steps = DEFAULT_NEW_TOKEN_RATIO_DECAY_STEPS;
        let decay_step = (init_new_token_ratio - min_new_token_ratio) / decay_steps;

        let policy_str = sglang.and_then(|s| s.schedule_policy.as_deref());
        let schedule_policy = match policy_str {
            Some("lpm") => SchedulePolicy::Lpm,
            Some("fifo") | Some("fcfs") | None => SchedulePolicy::Fifo,
            Some(other) => {
                tracing::warn!(
                    "Unknown sglang schedule_policy '{}', falling back to FIFO",
                    other
                );
                SchedulePolicy::Fifo
            }
        };

        Self {
            schedule_policy,
            max_prefill_tokens: sglang
                .and_then(|s| s.max_prefill_tokens)
                .unwrap_or(DEFAULT_MAX_PREFILL_TOKENS),
            chunked_prefill_size: sglang
                .and_then(|s| s.chunked_prefill_size)
                .unwrap_or(DEFAULT_CHUNKED_PREFILL_SIZE),
            clip_max_new_tokens: sglang
                .and_then(|s| s.clip_max_new_tokens)
                .unwrap_or(DEFAULT_CLIP_MAX_NEW_TOKENS),
            init_new_token_ratio,
            min_new_token_ratio,
            new_token_ratio_decay_step: decay_step,
            perf_model: args.perf_model.clone(),
            speedup_ratio: args.speedup_ratio,
            decode_speedup_ratio: args.decode_speedup_ratio,
            worker_type: args.worker_type,
            block_size: args.block_size,
        }
    }
}

pub(super) fn ceil_to_block(tokens: usize, block_size: usize) -> usize {
    if tokens == 0 {
        return 0;
    }

    tokens.div_ceil(block_size) * block_size
}

pub(super) fn floor_to_block(tokens: usize, block_size: usize) -> usize {
    tokens / block_size * block_size
}
