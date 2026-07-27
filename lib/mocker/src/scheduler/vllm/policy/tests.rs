// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for engine-specific behavior carried on the shared scheduler core.
//!
//! These drive the vLLM [`VllmCore`] directly (TRT-LLM routes to it) and read
//! its scheduler state through the test-only [`VllmCore::state`] accessor.

use dynamo_kv_router::protocols::KvCacheEventData;
use rstest::rstest;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, EngineType, G1Backend, KvEventPublishers, MockEngineArgs, PrefillCost,
    SchedulingPolicy,
};
use crate::common::sequence::ActiveSequence;
use crate::kv_manager::G1Acquire;
use crate::kv_manager::G1Manager;
use crate::scheduler::vllm::{RequestStatus, VllmCore};

use super::{
    AdmissionDecision, WaitingAdmissionConfig, apply_mtp_prefix_recompute, apply_prefix_recompute,
    decide_waiting_admission, should_reject_for_model_len,
};

mod vllm {
    use super::*;

    fn kv_manager(capacity: usize) -> G1Manager {
        G1Manager::new_with_event_sink(capacity, 4, KvEventPublishers::default(), 0)
    }

    #[test]
    fn mtp_recomputes_exactly_one_cached_prefix_block() {
        let adjusted = apply_mtp_prefix_recompute(
            SchedulingPolicy::Vllm,
            4,
            true,
            PrefillCost {
                new_blocks: 1,
                new_tokens: 4,
                cached_tokens: 8,
                active_cached_tokens: 8,
            },
        );

        assert_eq!(adjusted.cached_tokens, 4);
        assert_eq!(adjusted.active_cached_tokens, 4);
        assert_eq!(adjusted.new_tokens, 8);
        assert_eq!(adjusted.new_blocks, 2);
    }

    #[test]
    fn mtp_recompute_does_not_change_trtllm_accounting() {
        let original = PrefillCost {
            new_blocks: 1,
            new_tokens: 4,
            cached_tokens: 8,
            active_cached_tokens: 8,
        };
        let adjusted = apply_mtp_prefix_recompute(
            SchedulingPolicy::TrtllmGuaranteedNoEvict,
            4,
            true,
            original.clone(),
        );

        assert_eq!(adjusted.new_blocks, original.new_blocks);
        assert_eq!(adjusted.new_tokens, original.new_tokens);
        assert_eq!(adjusted.cached_tokens, original.cached_tokens);
        assert_eq!(adjusted.active_cached_tokens, original.active_cached_tokens);
    }

    #[test]
    fn exact_block_aligned_full_prompt_hit_recomputes_last_block() {
        let adjusted = apply_prefix_recompute(
            SchedulingPolicy::Vllm,
            8,
            4,
            false,
            true,
            PrefillCost {
                new_blocks: 0,
                new_tokens: 0,
                cached_tokens: 8,
                active_cached_tokens: 8,
            },
        );

        assert_eq!(adjusted.cached_tokens, 4);
        assert_eq!(adjusted.active_cached_tokens, 4);
        assert_eq!(adjusted.new_tokens, 4);
        assert_eq!(adjusted.new_blocks, 1);
    }

    #[test]
    fn ordinary_vllm_prefix_recompute_does_not_change_trtllm_accounting() {
        let original = PrefillCost {
            new_blocks: 0,
            new_tokens: 0,
            cached_tokens: 8,
            active_cached_tokens: 8,
        };
        let adjusted = apply_prefix_recompute(
            SchedulingPolicy::TrtllmGuaranteedNoEvict,
            8,
            4,
            false,
            true,
            original.clone(),
        );

        assert_eq!(adjusted.new_blocks, original.new_blocks);
        assert_eq!(adjusted.new_tokens, original.new_tokens);
        assert_eq!(adjusted.cached_tokens, original.cached_tokens);
        assert_eq!(adjusted.active_cached_tokens, original.active_cached_tokens);
    }

    #[test]
    fn preempted_non_aligned_prompt_uses_complete_known_context() {
        for backend in [G1Backend::Kvbm, G1Backend::Native] {
            let owner = Uuid::from_u128(700 + backend as u128);
            let mut manager =
                G1Manager::new_with_backend(8, 4, KvEventPublishers::default(), 0, backend);
            // The six-token prompt is not block aligned. Retained generation
            // extends the known context to nine tokens: two reusable full
            // blocks plus one partial block.
            let mut sequence = ActiveSequence::new((0..6).collect(), 8, Some(4), true, false);
            let creation = sequence.take_creation_signal().unwrap();
            assert!(matches!(
                manager.process_for_request(owner, &creation, 0),
                G1Acquire::Ready(_)
            ));
            assert!(sequence.push(6).is_none());
            assert!(sequence.push(7).is_none());
            for signal in sequence.push(8).unwrap() {
                assert!(matches!(
                    manager.process_for_request(owner, &signal, 0),
                    G1Acquire::Ready(_)
                ));
            }
            sequence.commit_allocation(sequence.len());
            manager.finalize_computed_prefix(owner, 0, sequence.len(), &mut sequence);

            for signal in sequence.reset_with_signal() {
                assert!(matches!(
                    manager.process_for_request(owner, &signal, 0),
                    G1Acquire::Ready(_)
                ));
            }

            let raw = manager.get_prefill_cost(&sequence);
            assert_eq!(raw.cached_tokens, 8, "backend={backend:?}");
            assert_eq!(raw.new_tokens, 1, "backend={backend:?}");
            assert_eq!(raw.active_cached_tokens, 0, "backend={backend:?}");

            let adjusted =
                apply_prefix_recompute(SchedulingPolicy::Vllm, sequence.len(), 4, false, true, raw);
            assert_eq!(adjusted.cached_tokens, 8, "backend={backend:?}");
            assert_eq!(adjusted.new_tokens, 1, "backend={backend:?}");
        }
    }

    #[test]
    fn admits_when_current_sequence_fits_without_reserving_future_output() {
        let manager = kv_manager(4);
        let sequence = ActiveSequence::new((0..8).collect(), 32, Some(4), false, false);

        let decision = decide_waiting_admission(
            WaitingAdmissionConfig {
                policy: SchedulingPolicy::Vllm,
                num_gpu_blocks: 4,
                block_size: 4,
                mtp_enabled: false,
            },
            &sequence,
            true,
            std::iter::empty(),
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Admit { .. }));
    }

    #[test]
    fn waits_when_current_sequence_does_not_fit_available_kv() {
        let mut manager = kv_manager(4);
        let mut holder = ActiveSequence::new((100..112).collect(), 1, Some(4), false, false);
        let signal = holder.take_creation_signal().unwrap();
        assert!(matches!(manager.process(&signal), G1Acquire::Ready(3)));
        let sequence = ActiveSequence::new((0..8).collect(), 32, Some(4), false, false);

        let decision = decide_waiting_admission(
            WaitingAdmissionConfig {
                policy: SchedulingPolicy::Vllm,
                num_gpu_blocks: 4,
                block_size: 4,
                mtp_enabled: false,
            },
            &sequence,
            true,
            std::iter::empty(),
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Wait));
    }

    #[test]
    fn rejects_fresh_sequence_that_exceeds_total_kv() {
        let manager = kv_manager(4);
        let sequence = ActiveSequence::new((0..20).collect(), 1, Some(4), false, false);

        let decision = decide_waiting_admission(
            WaitingAdmissionConfig {
                policy: SchedulingPolicy::Vllm,
                num_gpu_blocks: 4,
                block_size: 4,
                mtp_enabled: false,
            },
            &sequence,
            true,
            std::iter::empty(),
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Reject));
    }

    #[test]
    fn rejects_prompt_at_max_model_len() {
        let sequence = ActiveSequence::new((0..8).collect(), 1, Some(4), false, false);

        assert!(should_reject_for_model_len(
            SchedulingPolicy::Vllm,
            &sequence,
            Some(8)
        ));
    }

    #[test]
    fn rejects_prompt_above_max_model_len() {
        let sequence = ActiveSequence::new((0..9).collect(), 1, Some(4), false, false);

        assert!(should_reject_for_model_len(
            SchedulingPolicy::Vllm,
            &sequence,
            Some(8)
        ));
    }

    #[test]
    fn aligned_repeat_physical_occupancy_matches_vllm_and_exposes_kvbm_gap() {
        let run = |backend| {
            let args = MockEngineArgs::builder()
                .engine_type(EngineType::Vllm)
                .g1_backend(backend)
                .block_size(4)
                .num_gpu_blocks(8)
                .max_num_batched_tokens(Some(64))
                .max_num_seqs(Some(4))
                .enable_chunked_prefill(true)
                .enable_prefix_caching(true)
                .speedup_ratio(0.0)
                .build()
                .unwrap();
            let mut core = VllmCore::new(args);
            let seed = Uuid::from_u128(101);
            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 1,
                uuid: Some(seed),
                ..Default::default()
            });

            let mut collector = crate::replay::TraceCollector::default();
            let first = core.execute_pass(&mut collector, 0.0);
            assert_eq!(first.admissions[0].reused_input_tokens, 0);

            let mut now_ms = first.end_ms.max(1.0);
            for _ in 0..16 {
                if !core.state().requests.contains_key(&seed) {
                    break;
                }
                let pass = core.execute_pass(&mut collector, now_ms);
                now_ms = pass.end_ms.max(now_ms + 1.0);
            }
            assert!(!core.state().requests.contains_key(&seed));

            let repeated = Uuid::from_u128(102);
            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 1,
                uuid: Some(repeated),
                ..Default::default()
            });
            let pass = core.execute_pass(&mut collector, now_ms);
            let reused_tokens = pass
                .admissions
                .iter()
                .find(|admission| admission.uuid == repeated)
                .expect("repeated request must be admitted")
                .reused_input_tokens;
            (
                reused_tokens,
                core.kv_manager.num_active_blocks(),
                core.kv_manager.num_inactive_blocks(),
            )
        };

        // The saved vLLM 0.25.1/B200 oracle has the same shape at block size
        // 16: cold [H0,H1,H2], then repeat reuses H0/H1 and creates another H2.
        let native = run(G1Backend::Native);
        assert_eq!(native, (4, 0, 3));

        // KVBM has the same request-visible cache hit, but canonicalizes the
        // recomputed tail and therefore undercounts physical occupancy by one.
        let kvbm = run(G1Backend::Kvbm);
        assert_eq!(kvbm, (4, 0, 2));
    }

    #[test]
    fn native_g1_exposes_chunked_prefill_block_only_after_it_is_computed() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .g1_backend(G1Backend::Native)
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(2))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, 1);
        let uuid = Uuid::from_u128(103);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            uuid: Some(uuid),
            ..Default::default()
        });

        let probe = ActiveSequence::new((0..8).collect(), 1, Some(4), true, false);
        let mut collector = crate::replay::TraceCollector::default();

        let first = core.execute_pass(&mut collector, 0.0);
        assert_eq!(core.state().requests[&uuid].num_computed_tokens, 2);
        assert_eq!(core.kv_manager.get_prefill_cost(&probe).cached_tokens, 0);
        assert!(
            first
                .kv_events
                .iter()
                .all(|event| !matches!(&event.event.data, KvCacheEventData::Stored(_))),
            "an allocated but only half-computed block must not be router-visible"
        );

        let second = core.execute_pass(&mut collector, first.end_ms);
        assert_eq!(core.state().requests[&uuid].num_computed_tokens, 4);
        assert_eq!(core.kv_manager.get_prefill_cost(&probe).cached_tokens, 4);
        let stored = second
            .kv_events
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(data) => Some(data),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stored.len(), 1, "the completed block must emit one Stored");
        assert_eq!(
            stored[0].blocks.len(),
            1,
            "the Stored event must contain exactly the completed first block"
        );
    }

    #[test]
    fn native_g1_finalizes_in_scheduler_order_for_same_pass_reuse() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .g1_backend(G1Backend::Native)
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, 1);
        let first_uuid = Uuid::from_u128(104);
        let second_uuid = Uuid::from_u128(105);
        for uuid in [first_uuid, second_uuid] {
            core.receive(DirectRequest {
                tokens: (0..8).collect(),
                max_output_tokens: 2,
                uuid: Some(uuid),
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass.admissions.len(), 2);
        assert_eq!(pass.admissions[0].uuid, first_uuid);
        assert_eq!(pass.admissions[0].reused_input_tokens, 0);
        assert_eq!(pass.admissions[1].uuid, second_uuid);
        assert_eq!(
            pass.admissions[1].reused_input_tokens, 4,
            "vLLM exposes the first request's completed block before scheduling the next request"
        );

        assert_eq!(core.request_block_count(first_uuid), 2);
        assert_eq!(core.request_block_count(second_uuid), 2);
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            3,
            "the requests share one prefix copy but own distinct equal-hash tails"
        );

        let stored = pass
            .kv_events
            .iter()
            .filter_map(|event| match &event.event.data {
                KvCacheEventData::Stored(data) => Some(data),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            stored.len(),
            1,
            "Minimal Physical publishes router hash-presence, not raw per-copy events"
        );
        assert_eq!(stored[0].blocks.len(), 2);
    }

    #[test]
    fn native_g1_exposes_generated_block_at_computed_boundary_in_same_pass() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .g1_backend(G1Backend::Native)
            .block_size(4)
            .num_gpu_blocks(8)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_kv_capture(args, 1);
        let running = Uuid::from_u128(106);
        core.receive(DirectRequest {
            tokens: vec![0, 1, 2],
            max_output_tokens: 2,
            output_token_ids: Some(vec![3, 9]),
            uuid: Some(running),
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let first = core.execute_pass(&mut collector, 0.0);
        let running_request = &core.state().requests[&running];
        assert_eq!(running_request.num_computed_tokens, 3);
        assert_eq!(running_request.sequence.len(), 4);

        let waiting = Uuid::from_u128(107);
        core.receive(DirectRequest {
            tokens: vec![0, 1, 2, 3, 4],
            max_output_tokens: 1,
            uuid: Some(waiting),
            ..Default::default()
        });
        let second = core.execute_pass(&mut collector, first.end_ms);
        let admission = second
            .admissions
            .iter()
            .find(|admission| admission.uuid == waiting)
            .expect("waiting request must be admitted in the boundary pass");
        assert_eq!(
            admission.reused_input_tokens, 4,
            "vLLM caches a newly computed generated block before later admissions"
        );

        let stored = second
            .kv_events
            .iter()
            .filter(|event| matches!(&event.event.data, KvCacheEventData::Stored(_)))
            .count();
        assert_eq!(stored, 1, "the completed generated block emits one Stored");
    }

    #[rstest]
    #[case::kvbm(G1Backend::Kvbm)]
    #[case::native(G1Backend::Native)]
    fn sampled_boundary_allocation_is_delayed_until_the_next_pass(#[case] g1_backend: G1Backend) {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .g1_backend(g1_backend)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let first = Uuid::from_u128(108);
        let second = Uuid::from_u128(109);
        for (uuid, base) in [(first, 0), (second, 100)] {
            core.receive(DirectRequest {
                tokens: (base..base + 4).collect(),
                max_output_tokens: 3,
                output_token_ids: Some(vec![base + 4, base + 5, base + 6]),
                uuid: Some(uuid),
                ..Default::default()
            });
        }

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        for (uuid, expected_token) in [(first, 4), (second, 104)] {
            assert!(pass.output_signals.iter().any(|signal| {
                signal.uuid == uuid
                    && signal.token_id == Some(expected_token)
                    && !signal.completed
                    && !signal.rejected
            }));
            let request = &core.state().requests[&uuid];
            assert_eq!(request.num_computed_tokens, 4);
            assert_eq!(request.sequence.num_allocated_tokens(), 4);
            assert_eq!(request.sequence.len(), 5);
            assert_eq!(core.request_block_count(uuid), 1);
        }
        assert_eq!(core.state().preemptions_total, 0);
        assert_eq!(core.kv_manager.num_active_blocks(), 2);

        let next_pass = core.execute_pass(&mut collector, pass.end_ms);
        for (uuid, expected_token) in [(first, 5), (second, 105)] {
            assert!(next_pass.output_signals.iter().any(|signal| {
                signal.uuid == uuid
                    && signal.token_id == Some(expected_token)
                    && !signal.completed
                    && !signal.rejected
            }));
            let request = &core.state().requests[&uuid];
            assert_eq!(request.num_computed_tokens, 5);
            assert_eq!(request.sequence.num_allocated_tokens(), 5);
            assert_eq!(request.sequence.len(), 6);
            assert_eq!(core.request_block_count(uuid), 2);
        }
        assert_eq!(core.state().preemptions_total, 0);
        assert_eq!(core.kv_manager.num_active_blocks(), 4);
    }

    #[rstest]
    #[case::kvbm(G1Backend::Kvbm)]
    #[case::native(G1Backend::Native)]
    fn terminal_sample_does_not_allocate_a_dangling_slot(#[case] g1_backend: G1Backend) {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .g1_backend(g1_backend)
            .block_size(4)
            .num_gpu_blocks(2)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let mut collector = crate::replay::TraceCollector::default();

        let seed = Uuid::from_u128(110);
        core.receive(DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            output_token_ids: Some(vec![8]),
            uuid: Some(seed),
            ..Default::default()
        });
        let seed_pass = core.execute_pass(&mut collector, 0.0);
        assert!(
            seed_pass.output_signals.iter().any(|signal| {
                signal.uuid == seed
                    && signal.token_id == Some(8)
                    && signal.completed
                    && !signal.rejected
            }),
            "an exact-capacity request must not livelock trying to allocate KV for its terminal sample"
        );
        assert!(!core.state().requests.contains_key(&seed));
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert_eq!(core.kv_manager.num_inactive_blocks(), 2);

        // Allocating one unrelated prompt block should evict only the older
        // seed tail. If the terminal sample above acquired a dangling partial
        // slot, it would also evict the reusable seed head.
        let unrelated = Uuid::from_u128(111);
        core.receive(DirectRequest {
            tokens: (100..104).collect(),
            max_output_tokens: 1,
            output_token_ids: Some(vec![104]),
            uuid: Some(unrelated),
            ..Default::default()
        });
        let unrelated_pass = core.execute_pass(&mut collector, seed_pass.end_ms + 1.0);
        assert!(
            unrelated_pass
                .output_signals
                .iter()
                .any(|signal| signal.uuid == unrelated && signal.completed)
        );

        let probe = Uuid::from_u128(112);
        core.receive(DirectRequest {
            tokens: (0..5).collect(),
            max_output_tokens: 1,
            output_token_ids: Some(vec![5]),
            uuid: Some(probe),
            ..Default::default()
        });
        let probe_pass = core.execute_pass(&mut collector, unrelated_pass.end_ms + 1.0);
        let admission = probe_pass
            .admissions
            .iter()
            .find(|admission| admission.uuid == probe)
            .expect("probe request must be admitted");
        assert_eq!(
            admission.reused_input_tokens, 4,
            "the seed's first block must survive the unrelated one-block allocation"
        );
    }

    #[test]
    fn trtllm_does_not_apply_vllm_max_model_len() {
        let sequence = ActiveSequence::new((0..9).collect(), 1, Some(4), false, false);

        assert!(!should_reject_for_model_len(
            SchedulingPolicy::TrtllmGuaranteedNoEvict,
            &sequence,
            Some(8)
        ));
    }

    #[test]
    fn core_rejects_prompt_above_max_model_len() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_model_len(Some(8))
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let uuid = Uuid::from_u128(1);
        core.receive(DirectRequest {
            tokens: (0..9).collect(),
            max_output_tokens: 1,
            uuid: Some(uuid),
            dp_rank: 0,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert!(
            pass.output_signals
                .iter()
                .any(|signal| signal.uuid == uuid && signal.completed && signal.rejected)
        );
        assert!(!core.state().requests.contains_key(&uuid));
    }

    #[test]
    fn core_completes_at_max_model_len_without_rejecting() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_model_len(Some(8))
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let uuid = Uuid::from_u128(2);
        core.receive(DirectRequest {
            tokens: (0..7).collect(),
            max_output_tokens: 4,
            uuid: Some(uuid),
            dp_rank: 0,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(pass.output_signals.len(), 1);
        let terminal = &pass.output_signals[0];
        assert_eq!(terminal.uuid, uuid);
        assert!(terminal.token_id.is_some());
        assert!(terminal.completed);
        assert!(!terminal.rejected);
        assert!(!core.state().requests.contains_key(&uuid));
    }

    #[test]
    fn speculative_decode_does_not_burst_past_max_model_len() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Vllm)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_model_len(Some(8))
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let uuid = Uuid::from_u128(3);
        core.receive(DirectRequest {
            tokens: (0..5).collect(),
            max_output_tokens: 8,
            uuid: Some(uuid),
            dp_rank: 0,
            ..Default::default()
        });

        let mut collector = crate::replay::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);

        assert_eq!(pass.output_signals.len(), 3);
        assert!(
            pass.output_signals
                .iter()
                .take(2)
                .all(|signal| !signal.completed)
        );
        let terminal = pass.output_signals.last().unwrap();
        assert!(terminal.completed);
        assert!(!terminal.rejected);
        assert!(!core.state().requests.contains_key(&uuid));
    }

    #[test]
    fn discounts_active_cached_prefix() {
        let mut manager = kv_manager(3);
        let mut holder = ActiveSequence::new((0..8).collect(), 1, Some(4), true, false);
        let signal = holder.take_creation_signal().unwrap();
        assert!(matches!(manager.process(&signal), G1Acquire::Ready(2)));
        let sequence = ActiveSequence::new((0..12).collect(), 1, Some(4), true, false);

        let decision = decide_waiting_admission(
            WaitingAdmissionConfig {
                policy: SchedulingPolicy::Vllm,
                num_gpu_blocks: 3,
                block_size: 4,
                mtp_enabled: false,
            },
            &sequence,
            true,
            std::iter::empty(),
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Admit { .. }));
    }

    #[test]
    fn mtp_recompute_requires_one_additional_available_block() {
        let mut manager = kv_manager(3);
        let mut holder = ActiveSequence::new((0..8).collect(), 1, Some(4), true, false);
        let signal = holder.take_creation_signal().unwrap();
        assert!(matches!(manager.process(&signal), G1Acquire::Ready(2)));
        let sequence = ActiveSequence::new((0..12).collect(), 1, Some(4), true, false);

        let decision = decide_waiting_admission(
            WaitingAdmissionConfig {
                policy: SchedulingPolicy::Vllm,
                num_gpu_blocks: 3,
                block_size: 4,
                mtp_enabled: true,
            },
            &sequence,
            true,
            std::iter::empty(),
            &manager,
        );

        assert!(matches!(decision, AdmissionDecision::Wait));
    }

    #[test]
    fn does_not_discount_inactive_cached_prefix() {
        let mut manager = kv_manager(3);
        let mut seeder = ActiveSequence::new((0..8).collect(), 1, Some(4), true, false);
        let signal = seeder.take_creation_signal().unwrap();
        assert!(matches!(manager.process(&signal), G1Acquire::Ready(2)));
        for signal in seeder.free_signal() {
            manager.process(&signal);
        }
        let mut holder = ActiveSequence::new((100..104).collect(), 1, Some(4), true, false);
        let signal = holder.take_creation_signal().unwrap();
        assert!(matches!(manager.process(&signal), G1Acquire::Ready(1)));
        let sequence = ActiveSequence::new((0..12).collect(), 1, Some(4), true, false);

        let decision = decide_waiting_admission(
            WaitingAdmissionConfig {
                policy: SchedulingPolicy::Vllm,
                num_gpu_blocks: 3,
                block_size: 4,
                mtp_enabled: false,
            },
            &sequence,
            true,
            std::iter::empty(),
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

    #[test]
    fn native_g1_runs_under_trtllm_no_evict_policy() {
        let mut args = capacity_args();
        args.g1_backend = G1Backend::Native;
        let mut core = VllmCore::new(args);
        receive(&mut core, Uuid::from_u128(201), 0..4, 4);
        receive(&mut core, Uuid::from_u128(202), 100..104, 4);

        assert_eq!(drain(&mut core), 2);
        assert!(core.state().requests.is_empty());
        assert_eq!(
            core.state().preemptions_total,
            0,
            "TRT-LLM GUARANTEED_NO_EVICT must remain in force with native G1"
        );
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
