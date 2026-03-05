// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous Scheduler for LLM Request Management
//!
//! This module implements an asynchronous scheduler that handles three main functions:
//! 1. Receiving new requests and placing them in the waiting queue
//! 2. Scheduling waiting requests against available KV cache resources
//! 3. Simulating the execution of running requests with realistic timing
//!
//! ## Scheduling Process
//! The scheduler checks direct block capacity to determine if there's sufficient
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
//! preemption strategy (LIFO by default, matching vLLM v1) where a running request
//! is evicted and placed at the front of the waiting queue to be rescheduled later.
//!
//! ## NOTE
//! The current prefill and decoding time simulations are not scientific at all and are WIP

use crate::common::perf_model::PerfModel;
use crate::common::protocols::{
    DirectRequest, KvCacheEventSink, MockEngineArgs, MoveBlock, OutputSignal, PreemptionMode,
    WorkerType,
};
use crate::common::running_mean::RunningMean;
use crate::common::sequence::ActiveSequence;
use crate::common::utils::sleep_until_precise;
use crate::kv_manager::KvManager;
use dynamo_kv_router::protocols::DpRank;
use dynamo_tokens::blocks::UniqueBlock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use validator::Validate;

/// Simple metrics struct for mocker's internal use
#[derive(Clone, Default, Debug)]
pub struct MockerMetrics {
    pub dp_rank: DpRank,
    pub active_decode_blocks: u64,
}

/// Enum representing either a direct request or an active sequence
pub enum Request {
    Direct(DirectRequest),
    Active(ActiveSequence),
}

#[derive(Default)]
struct SchedulerState {
    waiting: VecDeque<Uuid>,
    prefill: VecDeque<Uuid>,
    decode: VecDeque<Uuid>,
    requests: HashMap<Uuid, Request>,
}

impl SchedulerState {
    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn receive(&mut self, request: DirectRequest) -> Uuid {
        let uuid = request.uuid.unwrap_or_else(Uuid::new_v4);
        self.requests.insert(uuid, Request::Direct(request));
        self.waiting.push_back(uuid);
        uuid
    }

    /// Try to admit one request from waiting → prefill.
    /// Converts DirectRequest → ActiveSequence if needed. PrefillCost is computed
    /// later in simulate_prefill when the request reaches the front of the queue.
    fn admit_one(&mut self, args: &MockEngineArgs) -> bool {
        let Some(&uuid) = self.waiting.front() else {
            return false;
        };
        let num_active = self.prefill.len() + self.decode.len();
        if args.max_num_seqs.is_some_and(|limit| num_active >= limit) {
            return false;
        }

        self.waiting.pop_front();

        // Convert DirectRequest → ActiveSequence if needed
        if let Some(Request::Direct(_)) = self.requests.get(&uuid) {
            let Some(Request::Direct(direct)) = self.requests.remove(&uuid) else {
                unreachable!()
            };
            self.requests.insert(
                uuid,
                Request::Active(ActiveSequence::new(
                    direct.tokens,
                    direct.max_output_tokens,
                    Some(args.block_size),
                    args.enable_prefix_caching,
                    args.zmq_kv_events_port.is_some(),
                )),
            );
        }

        self.prefill.push_back(uuid);
        true
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

    /// Remove a UUID and its associated Request from collections.
    fn complete(&mut self, uuid: &Uuid) {
        tracing::trace!("Request {uuid} will complete");
        self.decode.retain(|u| u != uuid);
        self.requests.remove(uuid);
    }

    /// Preempt a running request by evicting it from decode, resetting the sequence,
    /// and adding it back to the front of the waiting queue.
    /// In LIFO mode, evicts the newest request (matches vLLM v1).
    /// In FIFO mode, evicts the oldest request.
    fn preempt(&mut self, mode: PreemptionMode) -> Vec<MoveBlock> {
        let uuid = match mode {
            PreemptionMode::Lifo => self.decode.pop_back(),
            PreemptionMode::Fifo => self.decode.pop_front(),
        }
        .expect("Nothing to evict for preemption.");
        let request = self
            .requests
            .remove(&uuid)
            .expect("Request does not exist.");
        tracing::warn!("Request {uuid} will be preempted");

        // Reset the sequence and re-queue for prefill
        let Request::Active(mut active_sequence) = request else {
            panic!("Expected ActiveSequence in running queue")
        };
        let signals = active_sequence.reset_with_signal();

        self.requests.insert(uuid, Request::Active(active_sequence));
        self.waiting.push_front(uuid);

        signals
    }
}

/// Cancels its token when dropped. Shared via Arc so the background task is
/// only cancelled when the last Scheduler clone is dropped.
struct CancelGuard(CancellationToken);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Manages scheduling of requests using KvManager resources
#[derive(Clone)]
pub struct Scheduler {
    request_tx: mpsc::UnboundedSender<DirectRequest>,
    metrics_rx: tokio::sync::watch::Receiver<MockerMetrics>,
    _cancel_guard: Arc<CancelGuard>,
}

impl Scheduler {
    /// Create a new Scheduler with the given parameters
    pub fn new(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        cancellation_token: Option<CancellationToken>,
    ) -> Self {
        args.validate().expect("invalid MockEngineArgs");

        // Create channel for request handling
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

        // Spawn main background task with cancellation token
        tokio::spawn(async move {
            // Create state and kv_manager as local variables owned by this task
            let mut state = SchedulerState::default();
            let mut kv_manager = KvManager::new_with_event_sink(
                args.num_gpu_blocks,
                args.block_size,
                kv_event_sink,
                dp_rank,
            );
            let mut hit_rates = RunningMean::new(1000);

            loop {
                // 1. Receive requests
                if receive_requests(&mut state, &mut request_rx, &cancel_token_clone)
                    .await
                    .is_none()
                {
                    break;
                }

                // 2. Simulate prefill + decode
                simulate_prefill(&mut state, &mut kv_manager, &mut hit_rates, &args).await;

                simulate_decode(
                    &mut state,
                    &mut kv_manager,
                    &output_tx,
                    &args.perf_model,
                    args.block_size,
                    args.speedup_ratio,
                    args.preemption_mode,
                )
                .await;

                // 3. Send metrics once per forward pass (after all prefill and decode processing)
                let _ = metrics_tx.send(MockerMetrics {
                    dp_rank,
                    active_decode_blocks: kv_manager.num_active_blocks() as u64,
                });
            }
        });

        Self {
            request_tx,
            metrics_rx,
            _cancel_guard: cancel_guard,
        }
    }
}

impl super::SchedulerHandle for Scheduler {
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

/// Receive requests from the channel.
/// Returns `Some(())` to continue the loop, `None` to break (on cancellation).
async fn receive_requests(
    state: &mut SchedulerState,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    cancel_token: &CancellationToken,
) -> Option<()> {
    if cancel_token.is_cancelled() {
        return None;
    }

    if state.is_empty() {
        // Fully idle - block until new request arrives or shutdown
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => {
                return None;
            }
            result = request_rx.recv() => {
                let Some(request) = result else {
                    return None; // channel closed
                };
                state.receive(request);
                return Some(());
            }
        }
    }

    // Has active/waiting work - collect any pending requests without blocking
    while let Ok(request) = request_rx.try_recv() {
        state.receive(request);
    }

    Some(())
}

/// Simulate prefill phase for all pending prefill requests.
///
/// Handles token budget, block allocation, and preemption inline.
/// Token budget: `max_num_batched_tokens - decode.len()` (1 token per decode request).
/// When blocks are unavailable, decode requests are preempted (LIFO by default)
/// to free capacity, matching vLLM v1 behavior.
async fn simulate_prefill(
    state: &mut SchedulerState,
    kv_manager: &mut KvManager,
    hit_rates: &mut RunningMean<f32>,
    args: &MockEngineArgs,
) -> Duration {
    let start_time = Instant::now();
    let mut total_time = Duration::ZERO;

    let mut token_budget = args
        .max_num_batched_tokens
        .map_or(usize::MAX, |t| t.saturating_sub(state.decode.len()));

    'prefill: while token_budget > 0 {
        // Drain prefill first, then pull from waiting one at a time
        if state.prefill.is_empty() && !state.admit_one(args) {
            break;
        }
        let uuid = state.prefill[0];

        let Some(Request::Active(seq)) = state.requests.get(&uuid) else {
            panic!("Request does not exist.");
        };
        let prefill_cost = kv_manager.get_prefill_cost(seq);
        let sequence_len = seq.len();
        let allocated_tokens = seq.num_allocated_tokens();
        let remaining = prefill_cost.new_tokens;

        // Token budget check
        let tokens_left = sequence_len - allocated_tokens;
        if !args.enable_chunked_prefill && tokens_left > token_budget {
            break;
        }
        let chunk = tokens_left.min(token_budget);
        let cumulative = allocated_tokens + chunk;

        // Allocate blocks. process() returns the number of blocks committed.
        // On partial success, preempt a decode request and retry — the next
        // loop iteration re-prepares from the updated num_allocated_tokens.
        let Some(Request::Active(seq)) = state.requests.get_mut(&uuid) else {
            panic!("Request does not exist.");
        };
        if let Some(signal) = seq.prepare_allocation(cumulative) {
            let expected = match &signal {
                MoveBlock::Use(blocks, ..) => blocks.len(),
                _ => unreachable!(),
            };
            let allocated = kv_manager.process(&signal);
            // Commit the blocks that were actually allocated
            let committed_tokens = if allocated == expected {
                cumulative
            } else {
                // Partial: compute token boundary from block count
                let prev_blocks = allocated_tokens
                    .div_ceil(seq.block_size())
                    .min(seq.unique_blocks().len());
                (prev_blocks + allocated) * seq.block_size()
            };
            seq.commit_allocation(committed_tokens.min(cumulative));

            if allocated < expected {
                if state.decode.is_empty() {
                    break;
                }
                for signal in state.preempt(args.preemption_mode) {
                    kv_manager.process(&signal);
                }
                continue 'prefill; // retry with freed capacity
            }
        } else {
            seq.commit_allocation(cumulative);
        }

        // Accumulate prefill compute time (only for the new tokens in this chunk)
        let new_tokens_in_chunk = chunk.min(remaining);
        if args.worker_type != WorkerType::Decode && new_tokens_in_chunk > 0 {
            total_time += Duration::from_secs_f64(
                prefill_cost.predict_prefill_compute(Some(new_tokens_in_chunk), &args.perf_model)
                    / 1000.0,
            );
        }

        // Hit rate: fraction of tokens that were already cached
        let hit_rate = if sequence_len > 0 {
            1.0 - (remaining as f32 / sequence_len as f32)
        } else {
            0.0
        };
        hit_rates.push(hit_rate);

        token_budget -= chunk;

        if cumulative >= sequence_len {
            // Fully prefilled — promote to decode queue
            state.prefill.pop_front();
            state.decode.push_back(uuid);
        } else {
            // Partially prefilled — resume next iteration with updated allocated_tokens
            break;
        }
    }

    if args.speedup_ratio > 0.0 && total_time > Duration::ZERO {
        let sleep_duration = Duration::from_secs_f64(total_time.as_secs_f64() / args.speedup_ratio);
        let deadline = start_time + sleep_duration;

        sleep_until_precise(deadline).await;
    }

    total_time
}

/// Simulate decode phase for all active decode requests.
/// Returns the total decode compute time.
async fn simulate_decode(
    state: &mut SchedulerState,
    kv_manager: &mut KvManager,
    output_tx: &Option<mpsc::UnboundedSender<OutputSignal>>,
    perf_model: &PerfModel,
    block_size: usize,
    speedup_ratio: f64,
    preemption_mode: PreemptionMode,
) -> Duration {
    let start_time = Instant::now();

    // Compute decode timing
    let active_kv_tokens = kv_manager.num_active_blocks() * block_size;

    // Compute average context length across all active decode requests
    let total_length: usize = state
        .decode
        .iter()
        .map(|uuid| {
            if let Request::Active(seq) = state.requests.get(uuid).unwrap() {
                seq.len()
            } else {
                0
            }
        })
        .sum();
    let count = state.decode.len();

    let context_length = if count > 0 { total_length / count } else { 0 };
    let decoding_time = perf_model.predict_decode_time(active_kv_tokens, context_length);

    let total_time = Duration::from_secs_f64(decoding_time / 1000.0);

    // Process decoding
    let uuids: Vec<Uuid> = state.decode.iter().copied().collect();
    for uuid in uuids {
        // Try to generate; if allocation fails, preempt until it succeeds
        // or nothing is left to preempt (matches vLLM v1 scheduler loop).
        // Reborrow sequence each iteration so the mutable ref doesn't
        // conflict with state.preempt().
        let mut allocated = false;
        loop {
            let Some(sequence) = state.run(uuid) else {
                break;
            };
            let signals = sequence.generate();
            if process_signals(kv_manager, &signals) {
                allocated = true;
                break;
            }
            sequence.pop(); // revert the failed generation

            if state.decode.is_empty() {
                break;
            }

            // Preempt one request and free its blocks
            for signal in state.preempt(preemption_mode) {
                kv_manager.process(&signal);
            }

            // If the current request was the one preempted, stop retrying
            if !state.decode.contains(&uuid) {
                break;
            }
        }

        if !allocated {
            continue;
        }

        let Some(sequence) = state.run(uuid) else {
            continue;
        };

        // Check completion and send notification
        let is_complete = sequence.generated_tokens() >= sequence.max_output_tokens();

        let send_failed = output_tx.as_ref().is_some_and(|tx| {
            tx.send(OutputSignal {
                uuid,
                completed: is_complete,
            })
            .is_err()
        });

        if send_failed {
            for signal in &sequence.free_signal() {
                kv_manager.process(signal);
            }
        }

        if send_failed || is_complete {
            state.complete(&uuid);
        }
    }

    if speedup_ratio > 0.0 && total_time > Duration::ZERO {
        let sleep_duration = Duration::from_secs_f64(total_time.as_secs_f64() / speedup_ratio);
        let deadline = start_time + sleep_duration;

        sleep_until_precise(deadline).await;
    }

    total_time
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
        if kv_manager.process(signal) > 0 {
            continue;
        }

        // Check we have a Use signal with blocks
        let MoveBlock::Use(blocks, _hashes, ..) = signal else {
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
    use crate::scheduler::SchedulerHandle;
    use rstest::rstest;
    use std::time::Duration;
    use tokio::time::interval;

    /// Helper function to verify that the scheduler is idle (no active KV blocks)
    fn assert_scheduler_idle(metrics: &MockerMetrics) {
        assert_eq!(
            metrics.active_decode_blocks, 0,
            "Expected 0 active blocks, got {}",
            metrics.active_decode_blocks
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
            scheduler.receive(request);
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
            scheduler.receive(request);
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

        // Verify forward pass metrics - scheduler should be idle after completing all requests
        let metrics = metrics_rx.borrow().clone();
        assert_scheduler_idle(&metrics);

        println!("Test passed! Received {received_tokens} tokens");
    }

    /// White-box unit test that directly creates SchedulerState + KvManager,
    /// manually invokes simulate_prefill / simulate_decode, and asserts on
    /// queue states and block counts after each step.
    #[tokio::test]
    async fn test_scheduler_internal_state_transitions() {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(6)
            .max_num_batched_tokens(Some(12))
            .max_num_seqs(Some(3))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();

        let mut state = SchedulerState::default();
        let mut kv_manager = KvManager::new(args.num_gpu_blocks, args.block_size);
        let mut hit_rates = RunningMean::new(1000);
        let output_tx: Option<mpsc::UnboundedSender<OutputSignal>> = None;

        let r1_uuid = Uuid::from_u128(1);
        let r2_uuid = Uuid::from_u128(2);
        let r3_uuid = Uuid::from_u128(3);

        // ── Step 1: Receive 3 requests ──
        // R1: 8 input, 2 max_output → 2 blocks
        // R2: 8 input, 2 max_output → 2 blocks
        // R3: 12 input, 2 max_output → 3 blocks
        state.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(r1_uuid),
            dp_rank: 0,
        });
        state.receive(DirectRequest {
            tokens: (100..108).collect(),
            max_output_tokens: 2,
            uuid: Some(r2_uuid),
            dp_rank: 0,
        });
        state.receive(DirectRequest {
            tokens: (200..212).collect(),
            max_output_tokens: 2,
            uuid: Some(r3_uuid),
            dp_rank: 0,
        });

        assert_eq!(state.waiting.len(), 3);
        assert_eq!(state.prefill.len(), 0);
        assert_eq!(state.decode.len(), 0);
        assert_eq!(kv_manager.num_active_blocks(), 0);

        // ── Step 2: First simulate_prefill ──
        // Budget=12. R1 takes 8 tokens (2 blocks), fully prefilled → decode.
        // R2 takes 4 tokens (1 block, chunked), partially prefilled → stays in prefill.
        simulate_prefill(&mut state, &mut kv_manager, &mut hit_rates, &args).await;

        assert_eq!(state.waiting.len(), 1);
        assert_eq!(state.prefill.len(), 1);
        assert_eq!(state.decode.len(), 1);
        assert_eq!(state.decode[0], r1_uuid);
        assert_eq!(state.prefill[0], r2_uuid);
        assert_eq!(state.waiting[0], r3_uuid);
        assert_eq!(kv_manager.num_active_blocks(), 3); // 2 for R1 + 1 for R2

        let seq = match state.requests.get(&r1_uuid).unwrap() {
            Request::Active(s) => s,
            _ => panic!("expected ActiveSequence"),
        };
        assert_eq!(seq.num_allocated_tokens(), 8);
        assert_eq!(seq.generated_tokens(), 0);

        let seq = match state.requests.get(&r2_uuid).unwrap() {
            Request::Active(s) => s,
            _ => panic!("expected ActiveSequence"),
        };
        assert_eq!(seq.num_allocated_tokens(), 4);
        assert_eq!(seq.generated_tokens(), 0);

        // ── Step 3: First simulate_decode ──
        // R1 generates 1 token, gains a partial block.
        simulate_decode(
            &mut state,
            &mut kv_manager,
            &output_tx,
            &args.perf_model,
            args.block_size,
            args.speedup_ratio,
            args.preemption_mode,
        )
        .await;

        assert_eq!(state.decode.len(), 1);
        assert_eq!(state.decode[0], r1_uuid);
        assert_eq!(kv_manager.num_active_blocks(), 4); // +1 partial for R1

        let seq = match state.requests.get(&r1_uuid).unwrap() {
            Request::Active(s) => s,
            _ => panic!("expected ActiveSequence"),
        };
        assert_eq!(seq.generated_tokens(), 1);

        // ── Step 4: Second simulate_prefill ──
        // Budget=11. R2 finishes (4 more tokens, 1 block → active=5, decode).
        // R3 admitted, needs 2 blocks for chunk of 7. Only 1 free slot → partial.
        // Preempt R2 (LIFO) → R2 back to waiting. Retry R3 → evicts R2's
        // inactive blocks, allocates 2 more → R3 allocated_tokens=11.
        simulate_prefill(&mut state, &mut kv_manager, &mut hit_rates, &args).await;

        assert_eq!(state.waiting.len(), 1, "R2 preempted back to waiting");
        assert_eq!(state.waiting[0], r2_uuid);
        assert_eq!(state.prefill.len(), 1, "R3 partially prefilled");
        assert_eq!(state.prefill[0], r3_uuid);
        assert_eq!(state.decode.len(), 1, "R1 still decoding");
        assert_eq!(state.decode[0], r1_uuid);
        assert_eq!(kv_manager.num_active_blocks(), 6); // at capacity

        let seq = match state.requests.get(&r3_uuid).unwrap() {
            Request::Active(s) => s,
            _ => panic!("expected ActiveSequence"),
        };
        assert_eq!(seq.num_allocated_tokens(), 11);

        // ── Step 5: Second simulate_decode ──
        // R1 generates 2nd token → complete. Frees 3 blocks (1 destroyed, 2 deactivated).
        simulate_decode(
            &mut state,
            &mut kv_manager,
            &output_tx,
            &args.perf_model,
            args.block_size,
            args.speedup_ratio,
            args.preemption_mode,
        )
        .await;

        assert!(!state.requests.contains_key(&r1_uuid), "R1 completed");
        assert_eq!(state.decode.len(), 0);
        assert_eq!(state.prefill.len(), 1);
        assert_eq!(state.waiting.len(), 1);
        assert_eq!(kv_manager.num_active_blocks(), 3); // only R3's 3 blocks

        // ── Step 6: Third simulate_prefill ──
        // R3 finishes prefill (1 token left, no new blocks) → decode.
        // R2 re-admitted, fully prefilled (2 blocks via inactive eviction) → decode.
        simulate_prefill(&mut state, &mut kv_manager, &mut hit_rates, &args).await;

        assert_eq!(state.waiting.len(), 0);
        assert_eq!(state.prefill.len(), 0);
        assert_eq!(state.decode.len(), 2);
        assert!(state.decode.contains(&r3_uuid));
        assert!(state.decode.contains(&r2_uuid));
        assert_eq!(kv_manager.num_active_blocks(), 5); // 3 for R3 + 2 for R2

        // ── Steps 7+: Cycle until all requests complete ──
        loop {
            simulate_prefill(&mut state, &mut kv_manager, &mut hit_rates, &args).await;
            simulate_decode(
                &mut state,
                &mut kv_manager,
                &output_tx,
                &args.perf_model,
                args.block_size,
                args.speedup_ratio,
                args.preemption_mode,
            )
            .await;

            if state.is_empty() {
                break;
            }
        }

        assert_eq!(state.waiting.len(), 0);
        assert_eq!(state.prefill.len(), 0);
        assert_eq!(state.decode.len(), 0);
        assert_eq!(kv_manager.num_active_blocks(), 0);
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

        scheduler.receive(request);

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
