// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the TRT-LLM `GUARANTEED_NO_EVICT` policy carried on the shared
//! vLLM scheduler core: reservation-gated admission and the no-preemption
//! invariant. See [`super::policy`].
//!
//! These drive the vLLM [`VllmCore`] directly (TRT-LLM routes to it) and read
//! its scheduler state through the test-only [`VllmCore::state`] accessor.

use uuid::Uuid;

use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs};
use crate::scheduler::vllm::{RequestStatus, VllmCore};

/// block_size 4, 6 GPU blocks (24 tokens). Each request below reserves
/// `ceil((prompt + max_output) / 4)` blocks to completion.
fn engine_args(engine_type: EngineType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(4)
        .num_gpu_blocks(6)
        // High enough that both prompts (8 + 8) fit in one pass, so the
        // capacity gate — not the token budget — is what limits admission.
        .max_num_batched_tokens(Some(16))
        .max_num_seqs(Some(4))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(false)
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

fn receive(core: &mut VllmCore, uuid: Uuid, tokens: std::ops::Range<u32>, max_output: usize) {
    core.receive(DirectRequest {
        tokens: tokens.collect(),
        max_output_tokens: max_output,
        uuid: Some(uuid),
        dp_rank: 0,
        arrival_timestamp_ms: None,
    });
}

/// Under GUARANTEED_NO_EVICT only the first request — whose
/// `prompt + max_output` footprint fits after reserving for running
/// requests — is admitted; the second halts at the gate and stays waiting.
#[test]
fn admits_only_what_fits_to_completion() {
    let mut core = VllmCore::new(engine_args(EngineType::Trtllm));
    let r1 = Uuid::from_u128(1);
    let r2 = Uuid::from_u128(2);
    // Each: 8 prompt + 8 output = 16 tokens = 4 blocks. Two need 8 > 6.
    receive(&mut core, r1, 0..8, 8);
    receive(&mut core, r2, 100..108, 8);

    let mut collector = crate::replay::TraceCollector::default();
    let pass = core.execute_pass(&mut collector, 0.0);

    assert_eq!(
        core.state().running.iter().copied().collect::<Vec<_>>(),
        vec![r1],
        "only r1 fits its to-completion reservation under no-evict"
    );
    assert!(
        core.state().waiting.contains(&r2),
        "r2 must remain waiting (no skip-ahead admission)"
    );
    assert_eq!(
        core.state().requests.get(&r2).unwrap().status,
        RequestStatus::Waiting,
    );
    assert_eq!(
        pass.mocker_metrics.vllm_preemptions_total, 0,
        "no-evict policy must never preempt"
    );
}

/// Contrast: with identical args, vLLM admits optimistically and runs both
/// requests concurrently (their prompts physically fit; only the reserved
/// to-completion footprint exceeds capacity, which vLLM ignores).
#[test]
fn vllm_admits_optimistically_unlike_trtllm() {
    let mut core = VllmCore::new(engine_args(EngineType::Vllm));
    let r1 = Uuid::from_u128(1);
    let r2 = Uuid::from_u128(2);
    receive(&mut core, r1, 0..8, 8);
    receive(&mut core, r2, 100..108, 8);

    let mut collector = crate::replay::TraceCollector::default();
    core.execute_pass(&mut collector, 0.0);

    let running: Vec<_> = core.state().running.iter().copied().collect();
    assert!(
        running.contains(&r1) && running.contains(&r2),
        "vLLM admits both requests optimistically, got {running:?}"
    );
}

/// A workload that over-commits KV during decode would preempt under vLLM.
/// Under no-evict the gate prevents over-admission, so the run completes
/// every request without ever calling the (hard-error) preemption path.
#[test]
fn preemption_inducing_workload_never_preempts() {
    // 4 GPU blocks (16 tokens). Each request reserves all 4 blocks to
    // completion (4 prompt + 12 output = 16 tokens), so only one can run
    // at a time.
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Trtllm)
        .block_size(4)
        .num_gpu_blocks(4)
        .max_num_batched_tokens(Some(8))
        .max_num_seqs(Some(4))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(false)
        .speedup_ratio(0.0)
        .build()
        .unwrap();
    let mut core = VllmCore::new(args);
    let r1 = Uuid::from_u128(1);
    let r2 = Uuid::from_u128(2);
    receive(&mut core, r1, 0..4, 12);
    receive(&mut core, r2, 100..104, 12);

    let mut collector = crate::replay::TraceCollector::default();
    let mut completed = 0usize;
    let mut now_ms = 0.0;
    let mut max_preemptions = 0u64;
    for _ in 0..300 {
        if core.state().requests.is_empty() {
            break;
        }
        // Would panic via debug_assert in `trtllm::report_no_evict_violation`
        // if the no-evict gate ever let the core over-admit.
        let pass = core.execute_pass(&mut collector, now_ms);
        now_ms = pass.end_ms.max(now_ms + 1.0);
        completed += pass
            .output_signals
            .iter()
            .filter(|signal| signal.completed)
            .count();
        max_preemptions = max_preemptions.max(pass.mocker_metrics.vllm_preemptions_total);
    }

    assert!(
        core.state().requests.is_empty(),
        "both requests should complete; {} left",
        core.state().requests.len()
    );
    assert_eq!(completed, 2, "both requests should finish");
    assert_eq!(max_preemptions, 0, "GUARANTEED_NO_EVICT must never preempt");
}

/// Hardware-parity test: reproduces a real `trtllm-serve` no-evict saturation
/// run (B200, MiniMax-M2.5-NVFP4, TP4). KV pool 7319 blocks (block_size 32),
/// 64 offered requests of ISL 1096 + max_output 7000 → each reserves
/// `ceil((1096+7000)/32) = 253` blocks → admission cap `floor(7319/253) = 28`.
/// Real engine measured a steady `num_scheduled_requests = 28` with the rest
/// queued and zero evictions; the mocker must match: running caps at 28, the
/// remainder stays waiting, and preemption never fires.
#[test]
fn no_evict_admission_cap_matches_hardware() {
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Trtllm)
        .block_size(32)
        .num_gpu_blocks(7319)
        .max_num_seqs(Some(256)) // batch-size cap is NOT the limiter; KV is
        .max_num_batched_tokens(Some(8192))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(false)
        .speedup_ratio(0.0)
        .build()
        .unwrap();
    let mut core = VllmCore::new(args);
    for i in 0..64u128 {
        // 1096 unique input tokens (no prefix reuse), max_output 7000
        let base = (i as u32 + 1) * 100_000;
        receive(&mut core, Uuid::from_u128(i + 1), base..(base + 1096), 7000);
    }
    let mut collector = crate::replay::TraceCollector::default();
    let mut now_ms = 0.0;
    let mut max_preemptions = 0u64;
    // Run enough passes to finish all prefills; long OSL means none complete,
    // so the running set fills to the KV cap and then holds.
    for _ in 0..40 {
        let pass = core.execute_pass(&mut collector, now_ms);
        now_ms = pass.end_ms.max(now_ms + 1.0);
        max_preemptions = max_preemptions.max(pass.mocker_metrics.vllm_preemptions_total);
    }
    let running = core.state().running.len();
    let waiting = core.state().waiting.len();
    eprintln!(
        "no-evict cap: running={running} waiting={waiting} max_preemptions={max_preemptions} (hardware=28)"
    );
    assert_eq!(max_preemptions, 0, "GUARANTEED_NO_EVICT must never preempt");
    assert_eq!(running, 28, "mocker admission cap must match hardware (28)");
    assert_eq!(
        running + waiting,
        64,
        "the rest must stay queued, not dropped"
    );
}
