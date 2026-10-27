// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A custom scorer can distinguish cache membership from request eligibility.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use dynamo_kv_router::{
    ActiveSequencesMultiWorker, KvRouterConfig, NoopSequencePublisher, RouterQueuePolicy,
    WorkerCandidates, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    indexer::TieredMatchDetails,
    protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank},
    scheduling::{
        LocalScheduler, NoopOverlapScoresRefresh, OverlapAnalysis, PolicyProfile, ScheduleMode,
        ScheduleRequest,
    },
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

#[derive(Clone, PartialEq)]
struct Config;

impl WorkerConfigLike for Config {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        1
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(2048)
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(1024)
    }
    fn prefix_cache_read_policy(&self) -> Option<&str> {
        Some("contract-test")
    }
}

struct Probe(Arc<AtomicBool>);

impl WorkerScorer for Probe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate.cache().expect("declared CACHE inputs");
            assert_eq!(cache.prefix_cache_read_allowed(), Some(false));
            assert_eq!(cache.device_overlap_blocks(), 4.0);
            assert_eq!(cache.accounting_cache_estimate(), (0.0, 0));
            self.0.store(true, Ordering::Release);
            *cost = 0.0;
        }
        Ok(())
    }
}

struct First;
impl WorkerPicker for First {
    fn pick(
        &mut self,
        _: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        if input.candidates().is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no candidates"));
        }
        Ok(0)
    }
}

#[tokio::test]
async fn scorer_reads_eligibility_without_losing_raw_overlap() {
    let worker = WorkerWithDpRank::new(1, 0);
    let config = KvRouterConfig::default();
    let mut tiered = TieredMatchDetails::default();
    tiered.device.overlap_scores.scores.insert(worker, 4);
    let overlap = OverlapAnalysis::new(&config, 16, &tiered).signals();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        16,
        HashMap::from([(1, (0, 1))]),
        false,
        0,
        "cache-read-contract",
    ));
    let (_tx, rx) = watch::channel(HashMap::from([(1, Config)]));
    let seen = Arc::new(AtomicBool::new(false));
    let policy = WorkerSelectionPolicy::new(
        config,
        "contract",
        vec![Box::new(Probe(Arc::clone(&seen)))],
        Box::new(First),
    );
    let cancel = CancellationToken::new();
    let scheduler = LocalScheduler::new(
        Arc::clone(&slots),
        rx,
        PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
        16,
        policy,
        None,
        None::<Arc<NoopOverlapScoresRefresh>>,
        None,
        None,
        Duration::from_secs(60),
        true,
        cancel.clone(),
        "contract",
        false,
    );
    let response = scheduler
        .schedule_request_admitted(ScheduleRequest {
            prefix_cache_read_resolver: Some(Arc::new(|policy| {
                (policy == Some("contract-test")).then_some(false)
            })),
            mode: ScheduleMode::TrackedWithLifecycle {
                request_id: "cache-read-contract".to_string(),
            },
            token_seq: None,
            block_hashes: None,
            isl_tokens: 65,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap,
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
        })
        .await
        .unwrap();
    assert!(seen.load(Ordering::Acquire));
    assert_eq!(response.response.cached_tokens, 0);
    scheduler.free("cache-read-contract").await.unwrap();
    scheduler.admission_barrier().await.unwrap();
    slots.assert_completely_drained(tokio::time::Instant::now());
    cancel.cancel();
}
