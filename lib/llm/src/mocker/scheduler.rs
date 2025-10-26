// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous Scheduler for LLM Request Management
//!
//! This module implements an asynchronous scheduler that handles three main functions:
//! 1. Receiving new requests and placing them in the waiting queue
//! 2. Scheduling waiting requests against available KV cache resources
//! 3. Simulating the execution of running requests with realistic timing
//!
//! ## Scheduling Process
//! The scheduler uses a watermark-based approach to determine if there's sufficient
//! KV cache space for new requests. It also enforces a batched tokens budget to prevent
//! oversubscription of computational resources. Only requests that can be allocated
//! these resources are moved from waiting to running state.
//!
//! ## Request Simulation
//! The simulation models two key phases:
//! - Prefill phase: Uses a quadratic cost function: (cached_tokens + new_tokens) * new_tokens
//! - Decode phase: Uses a cost function proportional to active KV blocks (linear)
//!
//! ## Resource Management
//! The scheduler communicates with the KvManager through MoveBlock signals at each
//! stage of request processing. When resources become constrained, it employs an
//! LRU-based preemption strategy where the oldest running request is evicted and
//! placed at the back of the waiting queue to be rescheduled later.
//!
//! ## NOTE
//! The current prefill and decoding time simulations are not scientific at all and are WIP

use crate::kv_router::protocols::{ForwardPassMetrics, KvStats, WorkerStats};
use crate::mocker::evictor::LRUEvictor;
use crate::mocker::kv_manager::KvManager;
use crate::mocker::protocols::{
    DirectRequest, MockEngineArgs, MoveBlock, OutputSignal, PrefillCost, WorkerType,
};
use crate::mocker::running_mean::RunningMean;
use crate::mocker::sequence::ActiveSequence;
use crate::tokens::blocks::UniqueBlock;
use std::collections::{HashMap, VecDeque};
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Enum representing either a direct request or an active sequence
pub enum Request {
    Direct(DirectRequest),
    Active(ActiveSequence),
}

#[derive(Default)]
struct SchedulerState {
    waiting: VecDeque<Uuid>,
    prefill: VecDeque<Uuid>,
    decode: LRUEvictor<Uuid>,
    requests: HashMap<Uuid, Request>,
    prefill_costs: HashMap<Uuid, PrefillCost>,
    max_num_batched_tokens: Option<usize>,
    active_tokens: usize,
    waiting_tokens: usize,
}

impl SchedulerState {
    fn new(max_num_batched_tokens: Option<usize>) -> Self {
        SchedulerState {
            max_num_batched_tokens,
            ..Default::default()
        }
    }

    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Create a new UUID for a DirectRequest, add it to requests, and push the UUID to waiting.
    fn receive(&mut self, request: DirectRequest) -> Uuid {
        // Use the provided UUID if available, otherwise generate a new one
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        self.requests.insert(uuid, Request::Direct(request));
        self.waiting.push_back(uuid);
        uuid
    }

    /// Get the next UUID from ready or waiting queue and its associated Request.
    fn next(&mut self) -> Option<(Uuid, Request)> {
        let uuid = self.waiting.pop_front()?;
        let request = self
            .requests
            .remove(&uuid)
            .expect("Request does not exist.");
        Some((uuid, request))
    }

    /// Move a UUID and its Request to the waiting queue (front).
    fn first_in_line(&mut self, uuid: Uuid, request: Request) {
        self.requests.insert(uuid, request);
        self.waiting.push_front(uuid);
    }

    /// Move a UUID and its Request to the ready queue.
    fn move_to_prefill(&mut self, uuid: Uuid, active_seq: ActiveSequence, cost: PrefillCost) {
        self.waiting_tokens += cost.new_tokens;
        self.requests.insert(uuid, Request::Active(active_seq));
        self.prefill.push_back(uuid);
        self.prefill_costs.insert(uuid, cost);
    }

    /// Try (chunked) prefill and move to decode queue
    ///
    /// Returns `Some((prefill_compute, creation_signal, is_full_prefill))` where:
    /// - `prefill_compute`: The compute time in milliseconds for this prefill operation
    /// - `creation_signal`: Optional MoveBlock signal for KV cache block creation
    /// - `is_full_prefill`: true if the entire sequence was prefilled, false if chunked
    fn try_prefill(&mut self) -> Option<(f64, Option<MoveBlock>, bool)> {
        let uuid = self.prefill.pop_front()?;

        // Remove and extract prefill_compute from prefill_costs
        let mut prefill_cost = self
            .prefill_costs
            .remove(&uuid)
            .expect("Expects valid prefill cost.");

        let new_tokens = prefill_cost.new_tokens;

        let maybe_prefill_tokens = self.max_num_batched_tokens.and_then(|max_tokens| {
            let remaining_tokens = max_tokens - self.active_tokens;
            if prefill_cost.new_tokens > remaining_tokens {
                Some(remaining_tokens)
            } else {
                None
            }
        });

        let (prefill_compute, is_full_prefill) = if let Some(prefill_tokens) = maybe_prefill_tokens
        {
            let prefill_compute = prefill_cost.predict_prefill_compute(Some(prefill_tokens));
            prefill_cost.new_tokens -= prefill_tokens;
            assert!(
                (prefill_cost.new_tokens > 0) && (prefill_compute > 0.0),
                "Encountered negative prefill tokens or prefill compute cost."
            );

            self.prefill.push_front(uuid);
            self.prefill_costs.insert(uuid, prefill_cost);

            self.active_tokens = self.max_num_batched_tokens.unwrap();
            self.waiting_tokens -= prefill_tokens;

            (prefill_compute, false)
        } else {
            // Assume possible to complete prefilling the sequence, transfer to decode
            self.decode.insert(uuid);

            self.active_tokens += new_tokens;
            self.waiting_tokens -= new_tokens;

            (prefill_cost.predict_prefill_compute(None), true)
        };

        // NOTE: the current behavior allocates the KV blocks for the entire sequence,
        // even if only a chunk is prefilled
        let Some(Request::Active(sequence)) = self.requests.get_mut(&uuid) else {
            panic!("Request does not exist.");
        };

        Some((
            prefill_compute,
            sequence.take_creation_signal(),
            is_full_prefill,
        ))
    }

    // assume (chunked) prefills are completed, then active tokens would be 1 per decoding sequence
    fn reset_active_tokens(&mut self) {
        self.active_tokens = self.decode.len();
    }

    fn run(&mut self, uuid: Uuid) -> Option<&mut ActiveSequence> {
        if !self.decode.contains(&uuid) {
            return None;
        }
        let Some(Request::Active(sequence)) = self.requests.get_mut(&uuid) else {
            panic!("Request does not exist.");
        };
        Some(sequence)
    }

    fn num_active_requests(&self) -> usize {
        self.prefill.len() + self.decode.len()
    }

    /// Remove a UUID and its associated Request from collections.
    fn complete(&mut self, uuid: &Uuid) {
        tracing::trace!("Request {uuid} will complete");
        self.decode.remove(uuid);
        self.requests.remove(uuid);
        self.prefill_costs.remove(uuid);
        self.active_tokens -= 1;
    }

    /// Preempt the oldest running request by evicting it from running, resetting the sequence,
    /// and adding it back to the waiting queue.
    /// Returns the signal from reset_with_signal or None if no requests are running.
    fn preempt(&mut self) -> Vec<MoveBlock> {
        // Evict the oldest UUID from running
        let uuid = self
            .decode
            .evict()
            .expect("Nothing to evict for preemption.");
        let request = self
            .requests
            .remove(&uuid)
            .expect("Request does not exist.");
        self.prefill_costs.remove(&uuid);
        self.active_tokens -= 1;
        tracing::warn!("Request {uuid} will be preempted");

        // Reset the sequence and get the new sequence and signal
        // Insert the new sequence back into the requests map and add to waiting queue
        let Request::Active(mut active_sequence) = request else {
            panic!("Expected ActiveSequence in running queue")
        };
        let signals = active_sequence.reset_with_signal();

        // Note: For preemption, we don't compute hit rate since we don't have access to new_tokens
        // and the sequence is being reset anyway. Hit rate tracking is primarily for new scheduling attempts.

        self.first_in_line(uuid, Request::Active(active_sequence));

        signals
    }
}

/// Manages scheduling of requests using KvManager resources
#[derive(Clone)]
pub struct Scheduler {
    request_tx: mpsc::UnboundedSender<DirectRequest>,
    metrics_rx: tokio::sync::watch::Receiver<ForwardPassMetrics>,
}

impl Scheduler {
    /// Create a new Scheduler with the given parameters
    pub fn new(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        component: Option<dynamo_runtime::component::Component>,
        cancellation_token: Option<CancellationToken>,
    ) -> Self {
        // Assert speedup_ratio is greater than 0
        assert!(
            args.speedup_ratio > 0.0,
            "speedup_ratio must be greater than 0, got: {}",
            args.speedup_ratio
        );

        // Create channel for request handling
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<DirectRequest>();
        let mut initial_metrics = ForwardPassMetrics::default();
        initial_metrics.worker_stats.data_parallel_rank = Some(dp_rank);
        let (metrics_tx, metrics_rx) =
            tokio::sync::watch::channel::<ForwardPassMetrics>(initial_metrics);

        let cancel_token_clone = cancellation_token.unwrap_or_default().clone();

        // Spawn main background task with cancellation token
        tokio::spawn(async move {
            // Create state and kv_manager as local variables owned by this task
            let mut state = SchedulerState::new(args.max_num_batched_tokens);
            let mut kv_manager = KvManager::new_with_publisher(
                args.num_gpu_blocks,
                args.block_size,
                component,
                dp_rank,
            );
            let mut hit_rates = RunningMean::new(1000);

            loop {
                // 1. Receive requests
                if state.is_empty() {
                    // Fully idle - block until new request arrives
                    tokio::select! {
                        biased;
                        Some(request) = request_rx.recv() => {
                            state.receive(request);
                        }
                        _ = cancel_token_clone.cancelled() => {
                            break;
                        }
                    }
                } else {
                    // Has active/waiting work - collect any pending requests without blocking
                    while let Ok(request) = request_rx.try_recv() {
                        state.receive(request);
                    }

                    // Check for cancellation
                    if cancel_token_clone.is_cancelled() {
                        break;
                    }
                }

                // Start timing for this forward pass (schedule + simulate)
                let iteration_start = std::time::Instant::now();

                // 2. Schedule waiting requests (once per iteration)
                try_schedule(&mut state, &kv_manager, &mut hit_rates, &args);

                // 3. Simulate prefill + decode
                let mut total_time = Duration::ZERO;

                // Process prefilling
                while let Some((prefill_compute, maybe_creation_signal, is_full_prefill)) =
                    state.try_prefill()
                {
                    // NOTE: Prefill cost/time is always incremented for new blocks, even if they
                    // could be cached by other requests in the same batch. This matches vLLM behavior.
                    // For decode workers, skip adding prefill compute time
                    if args.worker_type != WorkerType::Decode {
                        total_time += Duration::from_secs_f64(prefill_compute / 1000.0);
                    }

                    if let Some(creation_signal) = maybe_creation_signal
                        && !process_signals(&mut kv_manager, std::slice::from_ref(&creation_signal))
                    {
                        panic!("Block allocation for prefilling cannot fail.");
                    }

                    // Impossible to schedule more prefills if we encounter one incomplete (chunked) prefill
                    if !is_full_prefill {
                        break;
                    }
                }

                let active_perc = kv_manager.get_active_perc();
                let decoding_time = -25.74 * active_perc.powi(2) + 54.01 * active_perc + 5.74;
                total_time += Duration::from_secs_f64(decoding_time / 1000.0);

                state.reset_active_tokens();

                // Process decoding
                let uuids: Vec<Uuid> = state.decode.keys().cloned().collect();
                for uuid in uuids {
                    let Some(sequence) = state.run(uuid) else {
                        continue;
                    };
                    let signals = sequence.generate();

                    // Process all signals with the KvManager
                    // Handling of preemption on failure
                    if !process_signals(&mut kv_manager, &signals) {
                        sequence.pop(); // revert the failed generation op
                        for signal in state.preempt() {
                            kv_manager.process(&signal);
                        }
                        continue;
                    }

                    // Check completion and send notification
                    let is_complete = sequence.generated_tokens() >= sequence.max_output_tokens();
                    let should_output =
                        sequence.generated_tokens() > sequence.already_generated_tokens();

                    let mut send_failed = false;
                    if should_output {
                        send_failed = output_tx.as_ref().is_some_and(|tx| {
                            tx.send(OutputSignal {
                                uuid,
                                completed: is_complete,
                            })
                            .is_err()
                        });
                    }

                    if send_failed {
                        for signal in &sequence.free_signal() {
                            kv_manager.process(signal);
                        }
                    }

                    if send_failed || is_complete {
                        state.complete(&uuid);
                        continue;
                    }
                }

                // Send metrics once per forward pass (after all prefill and decode processing)
                {
                    let metrics = get_fwd_pass_metrics(&state, &kv_manager, &hit_rates, dp_rank);
                    let _ = metrics_tx.send(metrics);
                }

                // 4. Sleep to maintain target iteration timing
                let target_duration =
                    Duration::from_secs_f64(total_time.as_secs_f64() / args.speedup_ratio);
                let elapsed = iteration_start.elapsed();

                if elapsed < target_duration {
                    tokio::time::sleep(target_duration - elapsed).await;
                }
            }
        });

        Self {
            request_tx,
            metrics_rx,
        }
    }

    /// Add a new request to the waiting queue
    pub async fn receive(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request);
    }

    pub fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        self.request_tx.clone()
    }

    /// Get a watch receiver for forward pass metrics
    pub fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<ForwardPassMetrics> {
        self.metrics_rx.clone()
    }
}

/// Calculate forward pass metrics from current state
fn get_fwd_pass_metrics(
    state: &SchedulerState,
    kv_manager: &KvManager,
    hit_rates: &RunningMean<f32>,
    dp_rank: u32,
) -> ForwardPassMetrics {
    // Get state metrics
    let request_active_slots = state.decode.len() as u64;
    let num_requests_waiting = state.waiting.len() as u64;

    // Get KV manager metrics
    let active_blocks_count = kv_manager.num_active_blocks() as u64;
    let total_capacity = kv_manager.max_capacity() as u64;
    let gpu_cache_usage_perc = if total_capacity > 0 {
        active_blocks_count as f32 / total_capacity as f32
    } else {
        0.0
    };

    // Get hit rate metrics - O(1) access
    let gpu_prefix_cache_hit_rate = hit_rates.mean();

    let worker_stats = WorkerStats {
        data_parallel_rank: Some(dp_rank),
        request_active_slots,
        request_total_slots: 1024, // vllm max_num_seqs for gpu >= 70 vram, otherwise 256, fallback is 128
        num_requests_waiting,
    };

    let kv_stats = KvStats {
        kv_active_blocks: active_blocks_count,
        kv_total_blocks: total_capacity,
        gpu_cache_usage_perc,
        gpu_prefix_cache_hit_rate,
    };

    let spec_decode_stats = None;

    ForwardPassMetrics {
        worker_stats,
        kv_stats,
        spec_decode_stats,
    }
}

/// Attempts to schedule waiting requests from the state queue.
/// Returns the number of requests successfully scheduled.
fn try_schedule(
    state: &mut SchedulerState,
    kv_manager: &KvManager,
    hit_rates: &mut RunningMean<f32>,
    args: &MockEngineArgs,
) -> usize {
    let mut scheduled_count = 0;
    let mut current_blocks = kv_manager.num_active_blocks();
    let mut current_tokens = state.active_tokens + state.waiting_tokens;
    let mut current_seqs = state.num_active_requests();

    while let Some((uuid, request)) = state.next() {
        // Convert Request to ActiveSequence
        let active_sequence = match request {
            Request::Active(active_seq) => active_seq,
            Request::Direct(direct_request) => ActiveSequence::new(
                direct_request.tokens,
                direct_request.max_output_tokens,
                Some(args.block_size),
                args.enable_prefix_caching,
            ),
        };

        // Update predictive budgets
        let prefill_cost = kv_manager.get_prefill_cost(&active_sequence);
        let total_tokens = active_sequence.len();
        // this is conservative, assumes no cache hit so never over-schedules
        let new_blocks = (total_tokens as u32).div_ceil(args.block_size as u32) as usize;
        let new_tokens = prefill_cost.new_tokens;

        current_blocks += new_blocks;
        current_tokens += new_tokens;
        current_seqs += 1;

        // Check various budgets to see if possible to schedule
        let under_block_budget =
            current_blocks as f64 <= (1. - args.watermark) * kv_manager.max_capacity() as f64;
        // If chunked prefill is enabled, we can be under token budget when scheduling
        let comparison_tokens = if args.enable_chunked_prefill {
            current_tokens - new_tokens
        } else {
            current_tokens
        };
        let under_token_budget = args
            .max_num_batched_tokens
            .is_none_or(|limit| comparison_tokens <= limit);
        let under_seq_budget = args.max_num_seqs.is_none_or(|limit| current_seqs <= limit);

        // Cannot schedule, put first in line instead
        if !(under_block_budget && under_token_budget && under_seq_budget) {
            state.first_in_line(uuid, Request::Active(active_sequence));
            break;
        }

        // Compute and store hit rate
        let hit_rate = if !active_sequence.is_empty() {
            1.0 - (new_tokens as f32 / active_sequence.len() as f32)
        } else {
            0.0
        };
        hit_rates.push(hit_rate);

        state.move_to_prefill(uuid, active_sequence, prefill_cost);
        scheduled_count += 1;
    }

    scheduled_count
}

/// Processes MoveBlock signals with the KvManager.
///
/// When a signal fails, this function verifies that the failure is for an expected case:
/// specifically a single signal attempting to create a single partial (generation) block.
/// This validation is important because in normal operation, the only legitimate failure
/// case should be when trying to acquire a new generation block - any other failures would
/// indicate an unexpected state in the system.
fn process_signals(kv_manager: &mut KvManager, signals: &[MoveBlock]) -> bool {
    for signal in signals {
        if kv_manager.process(signal) {
            continue;
        }

        // Check we have a Use signal with blocks
        let MoveBlock::Use(blocks, _hashes) = signal else {
            panic!(
                "Failed signal is Invalid. Has to fail on generation signal, but failed on {signal:?}"
            );
        };

        // Verify the signal contains exactly one block
        let num_blocks = blocks.len();
        let num_active_blocks = kv_manager.num_active_blocks();
        if num_blocks != 1 {
            panic!(
                "Failed signal is Invalid. Tried to create (prefill) {num_blocks} blocks on top of {num_active_blocks} active blocks."
            );
        }

        // Verify the block is a PartialBlock (generation block)
        if !matches!(blocks[0], UniqueBlock::PartialBlock(_)) {
            panic!("Failed signal is Invalid. Generation block has to be partial.");
        }

        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::time::Duration;
    use tokio::time::interval;

    /// Helper function to verify that the scheduler is idle (no active or waiting requests/resources)
    fn assert_scheduler_idle(metrics: &ForwardPassMetrics) {
        assert_eq!(
            metrics.worker_stats.request_active_slots, 0,
            "Expected 0 active slots, got {}",
            metrics.worker_stats.request_active_slots
        );
        assert_eq!(
            metrics.worker_stats.num_requests_waiting, 0,
            "Expected 0 waiting requests, got {}",
            metrics.worker_stats.num_requests_waiting
        );
        assert_eq!(
            metrics.kv_stats.kv_active_blocks, 0,
            "Expected 0 active blocks, got {}",
            metrics.kv_stats.kv_active_blocks
        );
        assert_eq!(
            metrics.kv_stats.gpu_cache_usage_perc, 0.0,
            "Expected 0% GPU cache usage, got {}",
            metrics.kv_stats.gpu_cache_usage_perc
        );
    }

    #[rstest]
    #[case::case_1(false, false, false)]
    #[case::case_2(false, true, false)]
    #[case::case_3(true, false, false)]
    #[case::case_4(true, true, false)]
    #[case::case_5(false, false, true)]
    #[case::case_6(false, true, true)]
    #[case::case_7(true, false, true)]
    #[case::case_8(true, true, true)]
    #[tokio::test]
    async fn test_scheduler_token_generation_patterns(
        #[case] use_shared_tokens: bool,
        #[case] enable_prefix_caching: bool,
        #[case] enable_chunked_prefill: bool,
    ) {
        unsafe { std::env::set_var("RUST_LOG", "debug") };

        let kv_capacity: usize = 500;
        let block_size: usize = 64;
        let num_requests: usize = 200;
        let input_len: usize = 1000;
        let max_output_tokens: usize = 100;

        // Create channel for token output
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

        // Create scheduler args using builder - now including enable_prefix_caching
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(kv_capacity)
            .block_size(block_size)
            .speedup_ratio(10.0)
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .build()
            .unwrap();

        // Create scheduler with new args struct
        let scheduler = Scheduler::new(args, 0, Some(output_tx), None, None);

        // Create shared tokens for caching case
        let shared_tokens = if use_shared_tokens {
            Some(
                (0..input_len / 2)
                    .map(|_| rand::random::<u32>() % 50000)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Create test requests
        for _ in 0..num_requests {
            let input_tokens = if let Some(ref shared) = shared_tokens {
                // For caching case: use shared tokens for first half, random for second half
                let mut tokens = shared.clone();
                tokens.extend((0..input_len / 2).map(|_| rand::random::<u32>() % 50000));
                tokens
            } else {
                // For random case: create unique random token vector for each request
                (0..input_len)
                    .map(|_| rand::random::<u32>() % 50000)
                    .collect::<Vec<_>>()
            };

            let request = DirectRequest {
                tokens: input_tokens,
                max_output_tokens,
                uuid: None,
                dp_rank: 0,
            };
            scheduler.receive(request).await;
        }

        let start_time = std::time::Instant::now();

        // Collect all generated tokens (should be num_requests * max_output_tokens)
        let expected_tokens = num_requests * max_output_tokens;
        let mut received_tokens = 0;

        // Set up a timeout that causes the test to panic if no tokens are received for 2 seconds
        let timeout = tokio::time::sleep(Duration::from_secs(2));
        tokio::pin!(timeout);

        // Get metrics receiver
        let metrics_rx = scheduler.metrics_receiver();

        // Set up debug ticker interval
        let mut debug_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                biased;

                // Manual debug ticker that prints forward pass metrics
                _ = debug_interval.tick() => {
                    let _metrics = metrics_rx.borrow().clone();
                    tracing::debug!("Forward Pass Metrics: {_metrics:#?}");
                }

                Some(_) = output_rx.recv() => {
                    received_tokens += 1;
                    // Reset timeout whenever we receive a token
                    timeout.set(tokio::time::sleep(Duration::from_secs(2)));
                }

                _ = &mut timeout => {
                    // Break instead of panicking when timeout occurs
                    break;
                }
            }
        }

        // Calculate and print elapsed time
        let elapsed = start_time.elapsed();
        println!(
            "Test completed in: {elapsed:?} for {} case with prefix_caching={enable_prefix_caching} and chunked_prefill={enable_chunked_prefill}",
            if use_shared_tokens {
                "caching"
            } else {
                "random"
            }
        );

        // Assert that we received the expected number of tokens
        assert!(
            received_tokens == expected_tokens,
            "Received {received_tokens} tokens but expected exactly {expected_tokens}"
        );

        // Wait a bit for final metrics update to propagate
        tokio::time::sleep(Duration::from_millis(100)).await;

        let metrics = scheduler.metrics_receiver().borrow().clone();
        assert_scheduler_idle(&metrics);
    }

    #[tokio::test]
    async fn test_cache_hit_rate_with_identical_requests() {
        let block_size: usize = 64;
        let max_output_tokens: usize = 10;
        let speedup_ratio = 10.0;
        let num_requests = 10;
        let token_length = 65;

        // Create channel for token output
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

        // Create scheduler args
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(100) // Large enough to not be a constraint
            .block_size(block_size)
            .speedup_ratio(speedup_ratio)
            .build()
            .unwrap();

        // Create scheduler
        let scheduler = Scheduler::new(args, 0, Some(output_tx), None, None);

        // Create identical tokens for all requests
        let identical_tokens: Vec<u32> = (0..token_length).map(|i| i as u32).collect();

        // Send all requests with identical tokens
        for _ in 0..num_requests {
            let request = DirectRequest {
                tokens: identical_tokens.clone(),
                max_output_tokens,
                uuid: None,
                dp_rank: 0,
            };
            scheduler.receive(request).await;
            // Sleep for 0.1 second after each request
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Collect all generated tokens
        let mut received_tokens = 0;

        // Set up a timeout that resets to 0.5 seconds on each received token
        let timeout = tokio::time::sleep(Duration::from_millis(500));
        tokio::pin!(timeout);

        // Get metrics receiver
        let metrics_rx = scheduler.metrics_receiver();

        // Set up debug ticker interval
        let mut debug_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                biased;

                // Manual debug ticker that prints forward pass metrics
                _ = debug_interval.tick() => {
                    let _metrics = metrics_rx.borrow().clone();
                    tracing::debug!("Forward Pass Metrics: {_metrics:#?}");
                }

                Some(_signal) = output_rx.recv() => {
                    received_tokens += 1;
                    // Reset timeout whenever we receive a token
                    timeout.set(tokio::time::sleep(Duration::from_millis(500)));
                }

                _ = &mut timeout => {
                    // Break when timeout occurs (no more tokens for 0.5 seconds)
                    break;
                }
            }
        }

        // Wait a bit for final metrics update
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify forward pass metrics
        let metrics = metrics_rx.borrow().clone();

        assert_scheduler_idle(&metrics);
        assert!(
            metrics.kv_stats.gpu_prefix_cache_hit_rate > 0.8,
            "Expected cache hit rate > 0.8, got {}",
            metrics.kv_stats.gpu_prefix_cache_hit_rate
        );

        println!(
            "Test passed! Cache hit rate: {:.3}",
            metrics.kv_stats.gpu_prefix_cache_hit_rate
        );
        println!("Received {received_tokens} tokens");
    }

    #[tokio::test]
    async fn test_receiver_drop_cleans_up_resources() {
        let block_size: usize = 64;
        let input_tokens = 256;
        let max_output_tokens = 200; // More than we'll receive

        // Create channel for token output
        let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

        // Create scheduler args
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(10) // Enough for 256 tokens (4 blocks)
            .block_size(block_size)
            .speedup_ratio(100.0) // Fast simulation
            .build()
            .unwrap();

        // Create scheduler
        let scheduler = Scheduler::new(args, 0, Some(output_tx), None, None);

        // Create request with 256 tokens
        let tokens: Vec<u32> = (0..input_tokens).map(|i| i as u32).collect();
        let request = DirectRequest {
            tokens,
            max_output_tokens,
            uuid: None,
            dp_rank: 0,
        };

        scheduler.receive(request).await;

        // Receive exactly 129 tokens
        let mut received_count = 0;
        while received_count < 129 {
            if let Some(_signal) = output_rx.recv().await {
                received_count += 1;
            } else {
                panic!("Channel closed before receiving 129 tokens");
            }
        }

        // Drop the receiver immediately
        drop(output_rx);

        // Wait for 1 second to allow cleanup
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Check forward pass metrics
        let metrics_rx = scheduler.metrics_receiver();
        let metrics = metrics_rx.borrow().clone();

        assert_scheduler_idle(&metrics);
    }
}
