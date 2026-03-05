// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang scheduler simulation with FIFO policy and adaptive admission control.
//!
//! Reference: sglang/python/sglang/srt/managers/scheduler.py

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use validator::Validate;

use crate::cache::radix_cache::NodeId;
use crate::common::perf_model::PerfModel;
use crate::common::protocols::{
    DirectRequest, KvCacheEventSink, MockEngineArgs, OutputSignal, WorkerType,
};
use crate::common::utils::sleep_until_precise;
use crate::kv_manager::SglangKvManager;

use super::MockerMetrics;

/// Tracks a single request inside the SGLang scheduler.
struct SglangRequest {
    uuid: Uuid,
    token_ids: Vec<u64>,
    max_output_tokens: usize,
    output_len: usize,
    /// Deepest matched node in radix tree (for lock management).
    last_node: Option<NodeId>,
    /// Pool page indices for the full sequence.
    kv_indices: Vec<usize>,
    /// Number of input tokens already prefilled (for chunked prefill).
    prefilled_tokens: usize,
}

impl SglangRequest {
    fn total_tokens_needed(&self, clip_max_new_tokens: usize) -> usize {
        let remaining_input = self.token_ids.len() - self.prefilled_tokens;
        let clipped_output = self.max_output_tokens.min(clip_max_new_tokens);
        remaining_input + clipped_output
    }

    fn extend_input_len(&self) -> usize {
        self.token_ids.len() - self.prefilled_tokens
    }
}

/// SGLang scheduler with FIFO policy and adaptive admission control.
///
/// The scheduling loop mirrors SGLang's `Scheduler.event_loop_normal`:
/// `receive_requests → apply_schedule_policy → get_new_batch_prefill →
///  simulate_prefill → simulate_decode → decay_new_token_ratio`
pub struct SglangScheduler {
    request_tx: mpsc::UnboundedSender<DirectRequest>,
    metrics_rx: tokio::sync::watch::Receiver<MockerMetrics>,
    _cancel_guard: Arc<CancelGuard>,
}

struct CancelGuard(CancellationToken);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Scheduling policy for reordering the waiting queue.
#[derive(Clone, Copy, Debug, Default)]
pub enum SchedulePolicy {
    /// Process in arrival order.
    #[default]
    Fifo,
    /// Longest prefix match — prioritise requests with the most cached tokens.
    /// Falls back to FIFO when `waiting.len() > 128` (prefix matching is expensive).
    Lpm,
}

/// Configuration extracted from MockEngineArgs for SGLang-specific params.
struct SglangConfig {
    schedule_policy: SchedulePolicy,
    max_prefill_tokens: usize,
    chunked_prefill_size: usize,
    clip_max_new_tokens: usize,
    init_new_token_ratio: f64,
    min_new_token_ratio: f64,
    new_token_ratio_decay_step: f64,
    perf_model: Arc<PerfModel>,
    speedup_ratio: f64,
    worker_type: WorkerType,
    page_size: usize,
}

impl SglangConfig {
    fn from_args(args: &MockEngineArgs) -> Self {
        let schedule_conservativeness = args.sglang_schedule_conservativeness.unwrap_or(1.0);
        let init_new_token_ratio = 0.7 * schedule_conservativeness;
        let min_new_token_ratio = init_new_token_ratio * 0.14;
        let decay_steps = 600.0;
        let decay_step = (init_new_token_ratio - min_new_token_ratio) / decay_steps;

        let schedule_policy = match args.sglang_schedule_policy.as_deref() {
            Some("lpm") => SchedulePolicy::Lpm,
            Some("fifo") | Some("fcfs") | None => SchedulePolicy::Fifo,
            Some(other) => {
                tracing::warn!(
                    "Unknown sglang_schedule_policy '{}', falling back to FIFO",
                    other
                );
                SchedulePolicy::Fifo
            }
        };

        Self {
            schedule_policy,
            max_prefill_tokens: args.sglang_max_prefill_tokens.unwrap_or(16384),
            chunked_prefill_size: args.sglang_chunked_prefill_size.unwrap_or(8192),
            clip_max_new_tokens: args.sglang_clip_max_new_tokens.unwrap_or(4096),
            init_new_token_ratio,
            min_new_token_ratio,
            new_token_ratio_decay_step: decay_step,
            perf_model: args.perf_model.clone(),
            speedup_ratio: args.speedup_ratio,
            worker_type: args.worker_type,
            page_size: args.sglang_page_size.unwrap_or(1),
        }
    }
}

impl SglangScheduler {
    pub fn new(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        cancellation_token: Option<CancellationToken>,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<DirectRequest>();
        let initial_metrics = MockerMetrics {
            dp_rank,
            active_decode_blocks: 0,
        };
        let (metrics_tx, metrics_rx) =
            tokio::sync::watch::channel::<MockerMetrics>(initial_metrics);

        let cancel_token = cancellation_token.unwrap_or_default();
        let cancel_token_clone = cancel_token.clone();
        let cancel_guard = Arc::new(CancelGuard(cancel_token));

        args.validate().expect("invalid MockEngineArgs");
        let config = SglangConfig::from_args(&args);
        let total_tokens = args.num_gpu_blocks * args.block_size;

        tokio::spawn(async move {
            let mut kv_manager =
                SglangKvManager::new(total_tokens, config.page_size, kv_event_sink, dp_rank);
            let mut waiting: VecDeque<SglangRequest> = VecDeque::new();
            let mut running: Vec<SglangRequest> = Vec::new();
            let mut new_token_ratio = config.init_new_token_ratio;

            loop {
                // 1. Receive requests
                if receive_requests(&mut waiting, &mut request_rx, &cancel_token_clone, &running)
                    .await
                    .is_none()
                {
                    break;
                }

                // 2. Apply scheduling policy
                apply_schedule_policy(&mut waiting, &kv_manager, &config);

                // 3. Admit new requests for prefill
                let admit = get_new_batch_prefill(
                    &mut waiting,
                    &mut kv_manager,
                    &config,
                    new_token_ratio,
                    &running,
                );

                if admit.oom {
                    new_token_ratio = config.init_new_token_ratio;
                }

                // 4. Simulate prefill
                simulate_prefill(&admit.can_run, &config).await;

                // Separate fully-prefilled from chunked requests
                for mut req in admit.can_run {
                    if req.prefilled_tokens < req.token_ids.len() {
                        // Chunked prefill: cache partial sequence, put back in waiting
                        if let Some(last_node) = req.last_node {
                            let new_last = kv_manager.cache_unfinished_req(
                                &req.token_ids[..req.prefilled_tokens],
                                &req.kv_indices,
                                last_node,
                            );
                            req.last_node = Some(new_last);
                        }
                        waiting.push_front(req);
                    } else {
                        running.push(req);
                    }
                }

                // 5. Simulate decode
                simulate_decode(
                    &mut running,
                    &mut kv_manager,
                    &output_tx,
                    &config,
                    dp_rank,
                    &metrics_tx,
                )
                .await;

                // 6. Decay new_token_ratio
                new_token_ratio = (new_token_ratio - config.new_token_ratio_decay_step)
                    .max(config.min_new_token_ratio);
            }
        });

        Self {
            request_tx,
            metrics_rx,
            _cancel_guard: cancel_guard,
        }
    }
}

impl super::SchedulerHandle for SglangScheduler {
    fn receive(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request);
    }

    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        self.request_tx.clone()
    }

    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.metrics_rx.clone()
    }
}

async fn receive_requests(
    waiting: &mut VecDeque<SglangRequest>,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    cancel_token: &CancellationToken,
    running: &[SglangRequest],
) -> Option<()> {
    if cancel_token.is_cancelled() {
        return None;
    }

    if waiting.is_empty() && running.is_empty() {
        // Fully idle — block until request or shutdown
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return None,
            result = request_rx.recv() => {
                let request = result?;
                waiting.push_back(direct_to_sglang(request));
            }
        }
    }

    // Drain any pending requests without blocking
    while let Ok(request) = request_rx.try_recv() {
        waiting.push_back(direct_to_sglang(request));
    }
    Some(())
}

fn direct_to_sglang(req: DirectRequest) -> SglangRequest {
    SglangRequest {
        uuid: req.uuid.unwrap_or_else(Uuid::new_v4),
        token_ids: req.tokens.iter().map(|&t| t as u64).collect(),
        max_output_tokens: req.max_output_tokens,
        output_len: 0,
        last_node: None,
        kv_indices: Vec::new(),
        prefilled_tokens: 0,
    }
}

/// Reorder waiting queue based on scheduling policy.
fn apply_schedule_policy(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &SglangKvManager,
    config: &SglangConfig,
) {
    match config.schedule_policy {
        SchedulePolicy::Fifo => {} // already in arrival order
        SchedulePolicy::Lpm => {
            const LPM_FALLBACK_THRESHOLD: usize = 128;
            if waiting.len() > LPM_FALLBACK_THRESHOLD {
                return; // too expensive, fall back to FIFO
            }
            // Score each request by prefix match length (read-only, no mutation)
            let mut scored: Vec<(usize, SglangRequest)> = waiting
                .drain(..)
                .map(|req| {
                    let prefix_len = kv_manager.cache().prefix_match_len(&req.token_ids);
                    (prefix_len, req)
                })
                .collect();
            // Sort descending by prefix match length (stable sort preserves FIFO for ties)
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            for (_, req) in scored {
                waiting.push_back(req);
            }
        }
    }
}

struct AdmitResult {
    can_run: Vec<SglangRequest>,
    oom: bool,
}

/// Admit requests from waiting queue within budget constraints.
fn get_new_batch_prefill(
    waiting: &mut VecDeque<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    config: &SglangConfig,
    new_token_ratio: f64,
    running: &[SglangRequest],
) -> AdmitResult {
    let cache = kv_manager.cache();
    let reserved: f64 = running
        .iter()
        .map(|req| {
            let remaining_output =
                (req.max_output_tokens - req.output_len).min(config.clip_max_new_tokens);
            remaining_output as f64 * new_token_ratio
        })
        .sum();

    let mut rem_total_tokens = (cache.available_tokens() + cache.evictable_size) as f64 - reserved;
    let mut rem_input_tokens = config.max_prefill_tokens as f64;
    let mut rem_chunk_tokens = config.chunked_prefill_size as f64;

    let mut can_run = Vec::new();
    let mut rejected = VecDeque::new();
    let mut oom = false;

    while let Some(mut req) = waiting.pop_front() {
        let extend_input = req.extend_input_len() as f64;
        let total_needed = req.total_tokens_needed(config.clip_max_new_tokens) as f64;

        // For chunked prefill: check against the chunk size, not the full input.
        // A request with extend_input > max_prefill_tokens can still be admitted
        // if we only prefill a chunk of it.
        let effective_input = extend_input.min(config.chunked_prefill_size as f64);

        if total_needed > rem_total_tokens || effective_input > rem_input_tokens {
            rejected.push_back(req);
            break;
        }

        // Release lock from previous chunk before re-allocating (chunked continuation)
        if let Some(prev_node) = req.last_node.take() {
            kv_manager.free_request(prev_node);
        }

        // Determine chunk boundary before allocation
        let chunk_end = if extend_input > rem_chunk_tokens && rem_chunk_tokens > 0.0 {
            let chunk = (rem_chunk_tokens as usize) / config.page_size * config.page_size;
            if chunk > 0 {
                req.prefilled_tokens + chunk
            } else {
                req.token_ids.len()
            }
        } else {
            req.token_ids.len()
        };

        // Only allocate for the current chunk, not the entire sequence
        let alloc_tokens = &req.token_ids[..chunk_end];
        let needed_pages = alloc_tokens.len().div_ceil(config.page_size);
        let available_pages = kv_manager.cache().token_pool.available();
        if available_pages < needed_pages {
            let deficit = (needed_pages - available_pages) * config.page_size;
            kv_manager.evict(deficit);
        }

        let alloc = kv_manager.allocate_for_request(alloc_tokens);
        let Some(alloc) = alloc else {
            rejected.push_back(req);
            oom = true;
            break;
        };

        req.last_node = Some(alloc.last_node);
        req.kv_indices = alloc.kv_indices;
        req.prefilled_tokens = chunk_end;

        let actual_prefilled = (chunk_end - (req.token_ids.len() - extend_input as usize)) as f64;
        rem_total_tokens -= total_needed;
        rem_input_tokens -= actual_prefilled;
        rem_chunk_tokens -= actual_prefilled;

        can_run.push(req);

        if rem_chunk_tokens <= 0.0 {
            break;
        }
    }

    while let Some(req) = rejected.pop_back() {
        waiting.push_front(req);
    }

    AdmitResult { can_run, oom }
}

async fn simulate_prefill(can_run: &[SglangRequest], config: &SglangConfig) {
    if can_run.is_empty() {
        return;
    }

    if config.worker_type == WorkerType::Decode {
        return;
    }

    let start = Instant::now();
    let total_new_tokens: usize = can_run.iter().map(|r| r.extend_input_len()).sum();
    let prefill_time = config.perf_model.predict_prefill_time(total_new_tokens);
    let total_time = Duration::from_secs_f64(prefill_time / 1000.0);

    if config.speedup_ratio > 0.0 && total_time > Duration::ZERO {
        let sleep_duration =
            Duration::from_secs_f64(total_time.as_secs_f64() / config.speedup_ratio);
        sleep_until_precise(start + sleep_duration).await;
    }
}

async fn simulate_decode(
    running: &mut Vec<SglangRequest>,
    kv_manager: &mut SglangKvManager,
    output_tx: &Option<mpsc::UnboundedSender<OutputSignal>>,
    config: &SglangConfig,
    dp_rank: u32,
    metrics_tx: &tokio::sync::watch::Sender<MockerMetrics>,
) {
    if running.is_empty() {
        return;
    }

    let start = Instant::now();

    let total_tokens: usize = running
        .iter()
        .map(|r| r.token_ids.len() + r.output_len)
        .sum();
    let avg_context = total_tokens / running.len();
    let decode_time = config
        .perf_model
        .predict_decode_time(total_tokens, avg_context);
    let total_time = Duration::from_secs_f64(decode_time / 1000.0);

    // Each decode step: allocate a new pool page when crossing a page boundary.
    // kv_indices stores 1 index per page (matching RadixCache value semantics).
    for req in running.iter_mut() {
        if req.output_len % config.page_size == 0 {
            if kv_manager.cache().token_pool.available() == 0 {
                kv_manager.evict(config.page_size);
            }
            if let Some(new_idx) = kv_manager.allocate_decode_page() {
                req.kv_indices.push(new_idx);
            }
            // If allocate fails, kv_indices will be short. On completion,
            // all_tokens is truncated to match available pages.
            // TODO: SGLang retracts youngest decode request on OOM (retract_decode).
            // Current impl relies on eviction + truncation instead.
        }
        req.output_len += 1;
    }

    // Send output signals and handle completions (always runs, even after OOM)
    let mut completed_indices = Vec::new();
    for (i, req) in running.iter_mut().enumerate() {
        let is_complete = req.output_len >= req.max_output_tokens;

        if let Some(tx) = output_tx {
            let _ = tx.send(OutputSignal {
                uuid: req.uuid,
                completed: is_complete,
            });
        }

        if is_complete {
            let mut all_tokens = req.token_ids.clone();
            for j in 0..req.output_len {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                req.uuid.hash(&mut hasher);
                j.hash(&mut hasher);
                all_tokens.push(hasher.finish());
            }

            // Cap by alignment AND available indices (decode OOM may cause shortages).
            let page_size = config.page_size;
            let aligned_pages = all_tokens.len() / page_size;
            let pages_used = aligned_pages.min(req.kv_indices.len());
            let tokens_to_cache = pages_used * page_size;
            all_tokens.truncate(tokens_to_cache);

            // Free excess page indices not covered by the cached sequence.
            if req.kv_indices.len() > pages_used {
                let excess = req.kv_indices[pages_used..].to_vec();
                kv_manager.cache_mut().token_pool.free(&excess);
            }

            if let Some(last_node) = req.last_node {
                if tokens_to_cache > 0 {
                    kv_manager.cache_finished_req(
                        &all_tokens,
                        &req.kv_indices[..pages_used],
                        last_node,
                    );
                } else {
                    kv_manager.free_request(last_node);
                }
            }
            completed_indices.push(i);
        }
    }

    // Remove completed requests in reverse order so swap_remove doesn't
    // invalidate pending indices (completed_indices is built in ascending order).
    for &i in completed_indices.iter().rev() {
        running.swap_remove(i);
    }

    // Publish metrics
    let cache = kv_manager.cache();
    let active_blocks = (cache.total_tokens() - cache.available_tokens()) / config.page_size.max(1);
    let _ = metrics_tx.send(MockerMetrics {
        dp_rank,
        active_decode_blocks: active_blocks as u64,
    });

    if config.speedup_ratio > 0.0 && total_time > Duration::ZERO {
        let sleep_duration =
            Duration::from_secs_f64(total_time.as_secs_f64() / config.speedup_ratio);
        sleep_until_precise(start + sleep_duration).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::SchedulerHandle;

    #[tokio::test]
    async fn test_sglang_scheduler_fifo_ordering() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(100)
            .block_size(64)
            .speedup_ratio(100.0)
            .build()
            .unwrap();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();
        let scheduler = SglangScheduler::new(args, 0, Some(output_tx), None, None);

        let num_requests = 5;
        let max_output = 3;

        for i in 0..num_requests {
            scheduler.receive(DirectRequest {
                tokens: vec![i as u32; 10],
                max_output_tokens: max_output,
                uuid: None,
                dp_rank: 0,
            });
        }

        // Collect all output signals
        let expected_signals = num_requests * max_output;
        let mut received = 0;
        let timeout = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some(_) = output_rx.recv() => {
                    received += 1;
                    if received >= expected_signals {
                        break;
                    }
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }
                _ = &mut timeout => break,
            }
        }

        assert_eq!(
            received, expected_signals,
            "Expected {expected_signals} signals, got {received}"
        );
    }

    #[tokio::test]
    async fn test_sglang_scheduler_admission_budget() {
        // Small pool — only enough for a few requests
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(2) // 2 * 64 = 128 tokens
            .block_size(64)
            .speedup_ratio(100.0)
            .build()
            .unwrap();

        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();
        let scheduler = SglangScheduler::new(args, 0, Some(output_tx), None, None);

        // Send requests that collectively exceed budget
        for _ in 0..10 {
            scheduler.receive(DirectRequest {
                tokens: vec![1; 20],
                max_output_tokens: 5,
                uuid: None,
                dp_rank: 0,
            });
        }

        // Should still complete all eventually (as earlier ones finish, budget frees up)
        let mut received = 0;
        let timeout = tokio::time::sleep(Duration::from_secs(10));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some(_) = output_rx.recv() => {
                    received += 1;
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }
                _ = &mut timeout => break,
            }
        }

        let expected = 10 * 5;
        assert_eq!(
            received, expected,
            "Expected {expected} signals, got {received}"
        );
    }

    #[test]
    fn test_lpm_ordering() {
        use crate::cache::radix_cache::RadixCache;

        let mut cache = RadixCache::new(1000, 1);
        // Insert a cached prefix [1,2,3,4,5]
        cache.insert(&[1, 2, 3, 4, 5], &[0, 1, 2, 3, 4]);

        // Request A: shares 5 tokens with cache
        let req_a = SglangRequest {
            uuid: Uuid::new_v4(),
            token_ids: vec![1, 2, 3, 4, 5, 6, 7],
            max_output_tokens: 3,
            output_len: 0,
            last_node: None,
            kv_indices: Vec::new(),
            prefilled_tokens: 0,
        };
        // Request B: shares 0 tokens
        let req_b = SglangRequest {
            uuid: Uuid::new_v4(),
            token_ids: vec![9, 8, 7],
            max_output_tokens: 3,
            output_len: 0,
            last_node: None,
            kv_indices: Vec::new(),
            prefilled_tokens: 0,
        };

        // prefix_match_len should score them correctly
        assert_eq!(cache.prefix_match_len(&req_a.token_ids), 5);
        assert_eq!(cache.prefix_match_len(&req_b.token_ids), 0);
    }

    #[test]
    fn test_lpm_fallback_to_fifo() {
        let kv_manager = SglangKvManager::new(1000, 1, None, 0);
        let config = SglangConfig {
            schedule_policy: SchedulePolicy::Lpm,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };

        // Create 129 requests (> 128 threshold)
        let mut waiting: VecDeque<SglangRequest> = (0..129)
            .map(|i| SglangRequest {
                uuid: Uuid::new_v4(),
                token_ids: vec![i as u64; 3],
                max_output_tokens: 1,
                output_len: 0,
                last_node: None,
                kv_indices: Vec::new(),
                prefilled_tokens: 0,
            })
            .collect();

        let first_uuid = waiting[0].uuid;
        apply_schedule_policy(&mut waiting, &kv_manager, &config);
        // Should fall back to FIFO — first request unchanged
        assert_eq!(waiting[0].uuid, first_uuid);
    }

    #[test]
    fn test_chunked_prefill_budget() {
        let config = SglangConfig {
            chunked_prefill_size: 10,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };

        let mut kv_manager = SglangKvManager::new(10000, 1, None, 0);
        let mut waiting: VecDeque<SglangRequest> = VecDeque::new();
        waiting.push_back(SglangRequest {
            uuid: Uuid::new_v4(),
            token_ids: vec![1; 20], // 20 tokens > chunked_prefill_size=10
            max_output_tokens: 3,
            output_len: 0,
            last_node: None,
            kv_indices: Vec::new(),
            prefilled_tokens: 0,
        });

        let admit = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[]);
        assert_eq!(admit.can_run.len(), 1);
        // Should only prefill 10 tokens (chunked_prefill_size), not all 20
        assert_eq!(admit.can_run[0].prefilled_tokens, 10);
        assert!(admit.can_run[0].prefilled_tokens < admit.can_run[0].token_ids.len());
    }

    #[test]
    fn test_new_token_ratio_decay_and_oom_reset() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );

        let mut ratio = config.init_new_token_ratio;
        for _ in 0..600 {
            ratio = (ratio - config.new_token_ratio_decay_step).max(config.min_new_token_ratio);
        }

        // After 600 steps, ratio should be at or near minimum
        assert!(
            (ratio - config.min_new_token_ratio).abs() < 0.01,
            "ratio={ratio}, min={}",
            config.min_new_token_ratio
        );

        // Simulate OOM reset
        ratio = config.init_new_token_ratio;
        assert!((ratio - 0.7).abs() < 0.001);
    }

    use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
    use std::sync::Mutex;

    struct MockSink {
        events: Mutex<Vec<KvCacheEvent>>,
    }

    impl MockSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }
        fn stored_count(&self) -> usize {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
                .count()
        }
    }

    impl crate::common::protocols::KvCacheEventSink for MockSink {
        fn publish(
            &self,
            event: KvCacheEvent,
            _block_token_ids: Option<&[Vec<u32>]>,
        ) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    /// Helper: collect output signals with a timeout.
    async fn collect_signals(
        output_rx: &mut mpsc::UnboundedReceiver<OutputSignal>,
        expected: usize,
        timeout_secs: u64,
    ) -> usize {
        let mut received = 0;
        let timeout = tokio::time::sleep(Duration::from_secs(timeout_secs));
        tokio::pin!(timeout);

        loop {
            tokio::select! {
                Some(_) = output_rx.recv() => {
                    received += 1;
                    if received >= expected {
                        break;
                    }
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }
                _ = &mut timeout => break,
            }
        }
        received
    }

    /// End-to-end: send requests with shared prefixes, verify all output
    /// signals arrive and KV events are published. Tests both FIFO and LPM.
    #[tokio::test]
    async fn test_e2e_shared_prefix_fifo() {
        let sink = Arc::new(MockSink::new());
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

        let args = MockEngineArgs::builder()
            .engine_type(crate::common::protocols::EngineType::Sglang)
            .num_gpu_blocks(100)
            .block_size(64)
            .speedup_ratio(100.0)
            .build()
            .unwrap();

        let engine =
            crate::engine::create_engine(args, 0, Some(output_tx), Some(sink.clone()), None);

        // Request A: tokens [1,2,3,4,5,6,7], osl=3
        engine.receive(DirectRequest {
            tokens: vec![1, 2, 3, 4, 5, 6, 7],
            max_output_tokens: 3,
            uuid: None,
            dp_rank: 0,
        });

        // Request B: shared prefix [1,2,3,4,5] + different suffix, osl=4
        engine.receive(DirectRequest {
            tokens: vec![1, 2, 3, 4, 5, 8, 9],
            max_output_tokens: 4,
            uuid: None,
            dp_rank: 0,
        });

        let expected = 3 + 4; // total output tokens
        let received = collect_signals(&mut output_rx, expected, 10).await;
        assert_eq!(
            received, expected,
            "Expected {expected} signals, got {received}"
        );

        // KV events should have been published for new pages
        assert!(sink.stored_count() > 0, "Expected BlockStored events");
    }

    #[tokio::test]
    async fn test_e2e_shared_prefix_lpm() {
        let sink = Arc::new(MockSink::new());
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

        // max_prefill_tokens=9: only one 7-token request fits per batch,
        // so LPM reordering determines which request gets admitted first.
        let args = MockEngineArgs::builder()
            .engine_type(crate::common::protocols::EngineType::Sglang)
            .num_gpu_blocks(100)
            .block_size(64)
            .speedup_ratio(100.0)
            .sglang_schedule_policy(Some("lpm".to_string()))
            .sglang_max_prefill_tokens(Some(9))
            .build()
            .unwrap();

        let engine =
            crate::engine::create_engine(args, 0, Some(output_tx), Some(sink.clone()), None);

        // Step 1: Seed the cache with prefix [1,2,3,4,5,6,7]
        let seed_uuid = Uuid::new_v4();
        engine.receive(DirectRequest {
            tokens: vec![1, 2, 3, 4, 5, 6, 7],
            max_output_tokens: 1,
            uuid: Some(seed_uuid),
            dp_rank: 0,
        });

        // Wait for seed to complete
        let timeout = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(timeout);
        loop {
            tokio::select! {
                Some(sig) = output_rx.recv() => {
                    if sig.uuid == seed_uuid && sig.completed {
                        break;
                    }
                }
                _ = &mut timeout => panic!("Seed request timed out"),
            }
        }

        // Let the scheduler drain and block on recv_many
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Step 2: Send two requests — no_match first (FIFO), then match.
        // Both sends are synchronous (unbounded channel), so both messages
        // will be in the buffer when recv_many next polls (current_thread rt).
        // LPM reorders match ahead of no_match in the waiting queue.
        // max_prefill_tokens=9 → only one 7-token request per batch →
        // admission order = LPM order → match completes first.
        let no_match_uuid = Uuid::new_v4();
        let match_uuid = Uuid::new_v4();

        engine.receive(DirectRequest {
            tokens: vec![99, 98, 97, 96, 95, 94, 93], // 7 tokens, no cache match
            max_output_tokens: 2,
            uuid: Some(no_match_uuid),
            dp_rank: 0,
        });
        engine.receive(DirectRequest {
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9], // 9 tokens, 7-token prefix match
            max_output_tokens: 2,
            uuid: Some(match_uuid),
            dp_rank: 0,
        });

        // Collect the first completed UUID — LPM should prioritize match_req
        let mut first_completed: Option<Uuid> = None;
        let timeout = tokio::time::sleep(Duration::from_secs(10));
        tokio::pin!(timeout);
        let mut received = 0;
        let expected = 2 * 2; // 2 requests * 2 output tokens

        loop {
            tokio::select! {
                Some(sig) = output_rx.recv() => {
                    if sig.completed && first_completed.is_none() {
                        first_completed = Some(sig.uuid);
                    }
                    received += 1;
                    if received >= expected { break; }
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }
                _ = &mut timeout => break,
            }
        }

        assert_eq!(
            received, expected,
            "Expected {expected} signals, got {received}"
        );
        assert_eq!(
            first_completed,
            Some(match_uuid),
            "LPM should schedule the request with longest prefix match first"
        );
        assert!(sink.stored_count() > 0, "Expected BlockStored events");
    }

    #[tokio::test]
    async fn test_e2e_chunked_prefill() {
        let sink = Arc::new(MockSink::new());
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

        let args = MockEngineArgs::builder()
            .engine_type(crate::common::protocols::EngineType::Sglang)
            .num_gpu_blocks(100)
            .block_size(64)
            .speedup_ratio(100.0)
            .sglang_chunked_prefill_size(Some(10))
            .build()
            .unwrap();

        let engine =
            crate::engine::create_engine(args, 0, Some(output_tx), Some(sink.clone()), None);

        // Input > chunked_prefill_size → forces chunked prefill
        let tokens: Vec<u32> = (0..30).collect();
        engine.receive(DirectRequest {
            tokens,
            max_output_tokens: 5,
            uuid: None,
            dp_rank: 0,
        });

        let expected = 5;
        let received = collect_signals(&mut output_rx, expected, 10).await;
        assert_eq!(
            received, expected,
            "Expected {expected} signals, got {received}"
        );
        assert!(sink.stored_count() > 0, "Expected BlockStored events");
    }
}
