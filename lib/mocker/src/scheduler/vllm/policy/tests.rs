// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for engine-specific behavior carried on the shared scheduler core.
//!
//! These drive the vLLM [`VllmCore`] directly (TRT-LLM routes to it) and read
//! its scheduler state through the test-only [`VllmCore::state`] accessor.

use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, EngineType, KvEventPublishers, MockEngineArgs, SchedulingPolicy,
};
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::KvManager;
use crate::scheduler::vllm::{RequestStatus, VllmCore};

use super::{AdmissionDecision, decide_waiting_admission};

mod vllm {
    use super::*;

    fn kv_manager(capacity: usize) -> KvManager {
        KvManager::new_with_event_sink(capacity, 4, KvEventPublishers::default(), 0)
    }

    #[test]
    fn admits_when_current_sequence_fits_without_reserving_future_output() {
        let manager = kv_manager(4);
        let sequence = ActiveSequence::new((0..8).collect(), 32, Some(4), false, false);

        let decision = decide_waiting_admission(
            SchedulingPolicy::Vllm,
            &sequence,
            true,
            std::iter::empty(),
            4,
            4,
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Admit { .. }));
    }

    #[test]
    fn waits_when_current_sequence_does_not_fit_available_kv() {
        let mut manager = kv_manager(4);
        let mut holder = ActiveSequence::new((100..112).collect(), 1, Some(4), false, false);
        let signal = holder.take_creation_signal().unwrap();
        assert_eq!(manager.process(&signal), 3);
        let sequence = ActiveSequence::new((0..8).collect(), 32, Some(4), false, false);

        let decision = decide_waiting_admission(
            SchedulingPolicy::Vllm,
            &sequence,
            true,
            std::iter::empty(),
            4,
            4,
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Wait));
    }

    #[test]
    fn rejects_fresh_sequence_that_exceeds_total_kv() {
        let manager = kv_manager(4);
        let sequence = ActiveSequence::new((0..20).collect(), 1, Some(4), false, false);

        let decision = decide_waiting_admission(
            SchedulingPolicy::Vllm,
            &sequence,
            true,
            std::iter::empty(),
            4,
            4,
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Reject));
    }

    #[test]
    fn discounts_active_cached_prefix() {
        let mut manager = kv_manager(3);
        let mut holder = ActiveSequence::new((0..8).collect(), 1, Some(4), true, false);
        let signal = holder.take_creation_signal().unwrap();
        assert_eq!(manager.process(&signal), 2);
        let sequence = ActiveSequence::new((0..12).collect(), 1, Some(4), true, false);

        let decision = decide_waiting_admission(
            SchedulingPolicy::Vllm,
            &sequence,
            true,
            std::iter::empty(),
            3,
            4,
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Admit { .. }));
    }

    #[test]
    fn does_not_discount_inactive_cached_prefix() {
        let mut manager = kv_manager(3);
        let mut seeder = ActiveSequence::new((0..8).collect(), 1, Some(4), true, false);
        let signal = seeder.take_creation_signal().unwrap();
        assert_eq!(manager.process(&signal), 2);
        for signal in seeder.free_signal() {
            manager.process(&signal);
        }
        let mut holder = ActiveSequence::new((100..104).collect(), 1, Some(4), true, false);
        let signal = holder.take_creation_signal().unwrap();
        assert_eq!(manager.process(&signal), 1);
        let sequence = ActiveSequence::new((0..12).collect(), 1, Some(4), true, false);

        let decision = decide_waiting_admission(
            SchedulingPolicy::Vllm,
            &sequence,
            true,
            std::iter::empty(),
            3,
            4,
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Wait));
    }
}

mod trtllm {
    use super::*;

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
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
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
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
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
            // Would panic via the policy invariant if the no-evict gate ever let
            // the core over-admit.
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

    fn drain(core: &mut VllmCore) -> usize {
        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut completed = 0usize;
        for _ in 0..100 {
            if core.state().requests.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            completed += pass
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count();
        }
        completed
    }

    fn capacity_args() -> MockEngineArgs {
        // 4 GPU blocks * block_size 4 = 16-token per-request capacity.
        MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    /// Enqueue normalization clamps an over-long output to the room left in the KV
    /// pool, so a request asking for more than fits still runs to the clamped
    /// length instead of being dropped. (Without clamping, r1's 4+40=44 tokens =
    /// 11 blocks exceed the 4-block pool and it could never run.)
    #[test]
    fn enqueue_clamps_excess_output_to_capacity() {
        let mut core = VllmCore::new(capacity_args());
        let r1 = Uuid::from_u128(1);
        receive(&mut core, r1, 0..4, 40); // clamped to 4 + (16-4)=12 = 16 tokens (4 blocks)

        assert!(
            core.state().requests.contains_key(&r1),
            "r1 fits after clamping and is admitted, not rejected"
        );
        let completed = drain(&mut core);
        assert_eq!(completed, 1, "clamped r1 runs to completion");
        assert!(core.state().requests.is_empty(), "queue fully drains");
    }

    /// Scheduler-level regression for the active-vs-inactive cached-prefix split:
    /// a request reusing an INACTIVE cached prefix must NOT discount it from the
    /// no-evict reservation, so it stays waiting while a holder occupies capacity
    /// and is admitted only once that capacity frees. (With the old all-cached
    /// discount it would be over-admitted into a pool that cannot hold it.)
    #[test]
    fn inactive_cached_prefix_not_discounted_keeps_request_waiting() {
        // 8 blocks * block_size 4, prefix caching on so reuse is modeled.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(8))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let holder = Uuid::from_u128(1);
        let seeder = Uuid::from_u128(2);
        let reuser = Uuid::from_u128(3);
        // holder: full = ceil((4+12)/4) = 4 blocks, long output -> holds capacity;
        //   while it runs, free-for-others = 8 - 4 = 4 blocks.
        receive(&mut core, holder, 0..4, 12);
        // seeder: a 2-block prefix (100..108), short output -> completes and leaves
        //   that prefix INACTIVE-but-registered.
        receive(&mut core, seeder, 100..108, 4);
        // reuser: SAME 2-block prefix + output -> full = ceil((8+12)/4) = 5 blocks.
        //   Discounting the inactive prefix (the bug) -> needs 5-2=3 <= 4 -> admitted;
        //   discounting only ACTIVE reuse -> needs 5 > 4 -> must wait for the holder.
        receive(&mut core, reuser, 100..108, 12);

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut max_preemptions = 0u64;
        let mut checked = false;
        for _ in 0..400 {
            if core.state().requests.is_empty() {
                break;
            }
            // Once the seeder has completed (prefix now inactive) but the holder is
            // still running, the reuser must remain waiting.
            if !checked
                && !core.state().requests.contains_key(&seeder)
                && core.state().requests.contains_key(&holder)
            {
                assert!(
                    !core.state().running.contains(&reuser),
                    "reuser hits the seeder's INACTIVE prefix; un-discounted it needs 5 > 4 free, \
                 so it must wait while the holder holds capacity"
                );
                assert_eq!(
                    core.state().requests.get(&reuser).map(|r| r.status),
                    Some(RequestStatus::Waiting),
                );
                checked = true;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            max_preemptions = max_preemptions.max(pass.mocker_metrics.vllm_preemptions_total);
        }
        assert!(
            checked,
            "test must observe the seeder-done / holder-running window"
        );
        assert!(
            core.state().requests.is_empty(),
            "reuser is admitted once the holder frees capacity; all requests drain"
        );
        assert_eq!(max_preemptions, 0, "GUARANTEED_NO_EVICT must never preempt");
    }

    /// An oversized request whose to-completion footprint can never fit the whole
    /// KV pool (even when empty) must be terminally rejected at the admission gate,
    /// not left stalling the FIFO head. Without rejection the no-evict gate halts at
    /// the oversized head (FIFO, no skip-ahead), so the valid follower behind it
    /// never runs — which is what hangs offline (`in_flight`) and live (`waiter`)
    /// replay.
    #[test]
    fn oversized_request_is_rejected_so_followers_run() {
        let mut core = VllmCore::new(capacity_args()); // 4 blocks * 4 = 16-token cap
        let oversized = Uuid::from_u128(1);
        let valid = Uuid::from_u128(2);
        // oversized: 20-token prompt = 5 blocks > 4-block pool, so
        //   ceil((20 + max_output) / 4) always exceeds the pool — unschedulable.
        receive(&mut core, oversized, 0..20, 8);
        // valid: 4-token prompt + 4 output = 2 blocks, fits comfortably.
        receive(&mut core, valid, 100..104, 4);

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut valid_completed = false;
        for _ in 0..100 {
            if core.state().requests.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            if pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == valid && signal.completed)
            {
                valid_completed = true;
            }
            assert_eq!(
                pass.mocker_metrics.vllm_preemptions_total, 0,
                "no-evict policy must never preempt"
            );
        }

        assert!(
            !core.state().requests.contains_key(&oversized),
            "oversized request must be terminally rejected, not stall the FIFO head"
        );
        assert!(
            valid_completed,
            "the valid follower must run to completion once the oversized head is rejected"
        );
        assert!(core.state().requests.is_empty(), "queue fully drains");
    }

    /// The rejection is an EXPLICIT terminal outcome: the oversized request's
    /// terminal signal carries `rejected = true` (so replay drivers free and advance
    /// without counting it as a real completion), while the valid request's terminal
    /// signal is an ordinary completion (`rejected = false`).
    #[test]
    fn rejection_emits_explicit_terminal_signal() {
        let mut core = VllmCore::new(capacity_args());
        let oversized = Uuid::from_u128(1);
        let valid = Uuid::from_u128(2);
        receive(&mut core, oversized, 0..20, 8);
        receive(&mut core, valid, 100..104, 4);

        let mut collector = crate::replay::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut oversized_rejected = false;
        let mut valid_completed_cleanly = false;
        for _ in 0..100 {
            if core.state().requests.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms.max(now_ms + 1.0);
            for signal in &pass.output_signals {
                if signal.uuid == oversized && signal.completed && signal.rejected {
                    oversized_rejected = true;
                }
                if signal.uuid == valid && signal.completed && !signal.rejected {
                    valid_completed_cleanly = true;
                }
            }
        }

        assert!(
            oversized_rejected,
            "oversized request must emit a terminal rejection signal (completed + rejected)"
        );
        assert!(
            valid_completed_cleanly,
            "valid request must emit an ordinary completion (completed, not rejected)"
        );
    }

    /// Terminal rejection must be decided on the UNDISCOUNTED full footprint, not the
    /// prefix-discounted `needed`. A request reusing a running holder's active prefix
    /// gets its `needed` discounted (a "can admit now" quantity), but the reused
    /// blocks are still physically resident — so its true footprint can exceed the
    /// whole pool and it can never run. Discounting would leave it stalling the FIFO
    /// head until the holder frees; the undiscounted footprint rejects it outright.
    #[test]
    fn active_prefix_reuse_oversized_request_rejected_on_full_footprint() {
        // 8 GPU blocks * block_size 4, prefix caching on so the active reuse is modeled.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Trtllm)
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(8))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let holder = Uuid::from_u128(1);
        let reuser = Uuid::from_u128(2);
        // holder: 2-block prefix (0..8) + output 8 -> full = ceil(16/4) = 4, fits the
        //   8-block pool and runs, keeping the 0..8 prefix ACTIVE.
        receive(&mut core, holder, 0..8, 8);
        // reuser: shares the holder's 0..8 prefix, full footprint = ceil((32+8)/4) = 10
        //   > 8-block pool. Reusing the active prefix discounts `needed` to 10-2 = 8,
        //   which is NOT > 8 — so the discounted check would let it stall — but its
        //   physical footprint (10) can never fit, so it must be terminally rejected.
        receive(&mut core, reuser, 0..32, 8);

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert!(
            core.state().running.contains(&holder),
            "holder fits its footprint and is admitted"
        );
        assert!(
            pass.output_signals
                .iter()
                .any(|signal| signal.uuid == reuser && signal.completed && signal.rejected),
            "reuser's 10-block footprint can never fit the 8-block pool (reused prefix is still \
         resident), so it must emit a terminal rejection even while reusing the active prefix"
        );
        assert!(
            !core.state().requests.contains_key(&reuser),
            "reuser must be rejected, not left stalling the FIFO head behind the holder"
        );
        assert_eq!(
            pass.mocker_metrics.vllm_preemptions_total, 0,
            "no-evict policy must never preempt"
        );
    }
}
