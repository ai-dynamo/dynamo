// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The default policy credits blocks a worker can fetch from a KV transfer source.

mod support;
use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use dynamo_kv_router::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvHintTransferWorkerMetadata, KvTransferEnforcement,
    WorkerConfigLike, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::SchedulingRequest;
use dynamo_kv_router::{KvRouterConfig, WorkerLoadProjection, WorkerSelector};
use std::collections::HashMap;

const BLOCK_SIZE: u32 = 16;
const PROMPT_BLOCKS: usize = 10;

#[derive(Clone, Default)]
struct HintWorker {
    topology: HashMap<String, String>,
    transfer_domain: Option<String>,
}

impl HintWorker {
    fn in_island(island: &str) -> Self {
        Self {
            topology: HashMap::from([("rdma-island".to_string(), island.to_string())]),
            transfer_domain: Some("rdma-island".to_string()),
        }
    }
}

impl WorkerConfigLike for HintWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        None
    }

    fn kv_hint_transfer_metadata_for_dp_rank(
        &self,
        _dp_rank: u32,
    ) -> Option<KvHintTransferWorkerMetadata<'_>> {
        Some(KvHintTransferWorkerMetadata {
            worker_type: "aggregated",
            source_control_endpoint: Some("tcp://worker:23280"),
        })
    }

    fn topology_domains(&self) -> Option<&HashMap<String, String>> {
        Some(&self.topology)
    }

    fn kv_transfer_domain(&self) -> Option<&str> {
        self.transfer_domain.as_deref()
    }

    fn kv_transfer_enforcement(&self) -> Option<KvTransferEnforcement> {
        self.transfer_domain
            .as_ref()
            .map(|_| KvTransferEnforcement::Required)
    }
}

fn worker(id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::from_worker_id(id)
}

/// `(id, device prefix blocks, active prefill tokens, active decode blocks)`.
fn request(workers: &[(u64, usize, usize, usize)]) -> SchedulingRequest {
    let (_, mut request) = support::fixture(0, PROMPT_BLOCKS * BLOCK_SIZE as usize);
    for &(id, prefix_blocks, active_prefill_tokens, active_decode_blocks) in workers {
        let overlap = &mut request.overlap;
        overlap
            .tier_overlap_blocks
            .device
            .insert(worker(id), prefix_blocks);
        overlap
            .effective_overlap_blocks
            .insert(worker(id), prefix_blocks as f64);
        overlap
            .effective_cached_tokens
            .insert(worker(id), prefix_blocks * BLOCK_SIZE as usize);
        request.worker_loads.insert(
            worker(id),
            WorkerLoadProjection {
                active_prefill_tokens,
                active_decode_blocks,
                additional_active_blocks: PROMPT_BLOCKS,
                ..Default::default()
            },
        );
    }
    request.kv_transfer_candidates = Some(KvTransferCandidates {
        block_hashes: (0..PROMPT_BLOCKS as u64)
            .map(ExternalSequenceBlockHash)
            .collect(),
        owner_prefix_blocks: workers
            .iter()
            .filter(|(_, prefix_blocks, ..)| *prefix_blocks > 0)
            .map(|&(id, prefix_blocks, ..)| (worker(id).into(), prefix_blocks))
            .collect(),
        routing_snapshot: None,
    });
    request
}

fn config(multiplier: f64) -> KvRouterConfig {
    KvRouterConfig {
        overlap_score_credit_decay: 0.0,
        shared_cache_multiplier: Some(multiplier),
        ..Default::default()
    }
}

/// The selected worker and the source its fetch hint would name.
fn select(
    workers: &HashMap<u64, HintWorker>,
    request: &SchedulingRequest,
    multiplier: f64,
) -> (u64, Option<(KvTransferCandidateSource, usize)>) {
    let selected = DefaultWorkerSelector::new(Some(config(multiplier)), "test")
        .select_worker(support::selection_input(workers, request, BLOCK_SIZE))
        .unwrap();
    (
        selected.worker.worker_id,
        selected
            .kv_transfer_source
            .map(|source| (source.source, source.prefix_blocks)),
    )
}

/// The design doc's example. A holds 6 of 10 blocks and is busy, C holds all 10
/// and is overloaded, B holds none and is idle.
#[test]
fn idle_worker_that_can_fetch_beats_a_busy_cached_one() {
    let workers = HashMap::from([
        (0, HintWorker::default()),
        (1, HintWorker::default()),
        (2, HintWorker::default()),
    ]);
    // A costs (2 + 10 - 6) + (3 + 10) = 19, B costs 10 + 10 = 20, C costs 60,
    // before fetch credit. A can fetch 4 blocks from C and B can fetch 10.
    let request = request(&[(0, 6, 32, 3), (1, 0, 0, 0), (2, 10, 320, 30)]);
    let from_c = Some((worker(2).into(), 10));

    // Without credit, A wins and fetches its missing blocks from C.
    assert_eq!(select(&workers, &request, 0.0), (0, from_c));
    // At half credit A costs 17 and B costs 15: B wins and fetches everything from C.
    assert_eq!(select(&workers, &request, 0.5), (1, from_c));
    // Fetch credit only scores; it does not count as cached for prefill booking.
    let selected = DefaultWorkerSelector::new(Some(config(0.5)), "test")
        .select_worker(support::selection_input(&workers, &request, BLOCK_SIZE))
        .unwrap();
    assert_eq!(selected.cached_tokens, 0);
}

#[test]
fn credit_follows_the_target_island() {
    let workers = HashMap::from([
        (0, HintWorker::in_island("x")),
        (1, HintWorker::in_island("y")),
        (2, HintWorker::in_island("x")),
        (3, HintWorker::in_island("y")),
    ]);
    // Two idle, cold workers; the busy sources hold 4 blocks in x and 8 in y.
    let request = request(&[(0, 0, 0, 0), (1, 0, 0, 0), (2, 4, 640, 60), (3, 8, 640, 60)]);
    assert_eq!(
        select(&workers, &request, 0.5),
        (1, Some((worker(3).into(), 8)))
    );

    // A required island ignores the longer source in the other island.
    let workers = HashMap::from([
        (0, HintWorker::in_island("x")),
        (3, HintWorker::in_island("y")),
    ]);
    let request = self::request(&[(0, 0, 0, 0), (3, 8, 640, 60)]);
    assert_eq!(select(&workers, &request, 0.5), (0, None));
}

#[test]
fn seeded_selection_matches_reference_with_fetchable_prefixes() {
    let workers: HashMap<_, _> = (0..8)
        .map(|id| {
            let island = if id % 2 == 0 { "x" } else { "y" };
            (id, HintWorker::in_island(island))
        })
        .collect();
    let shapes: Vec<_> = (0..8)
        .map(|id| {
            (
                id,
                (id as usize * 3) % 11,
                (id as usize % 5) * 48,
                id as usize % 7,
            )
        })
        .collect();
    let request = request(&shapes);
    for temperature in [0.0, 0.7] {
        for multiplier in [0.0, 0.5] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..config(multiplier)
            };
            let reference = dynamo_kv_router::DefaultWorkerSelector::new_seeded(
                Some(config.clone()),
                "test",
                42,
            );
            let plugin = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
            for _ in 0..32 {
                let expected = reference
                    .select_worker(support::selection_input(&workers, &request, BLOCK_SIZE))
                    .unwrap();
                let actual = plugin
                    .select_worker(support::selection_input(&workers, &request, BLOCK_SIZE))
                    .unwrap();
                assert_eq!(
                    (actual.worker, actual.kv_transfer_source),
                    (expected.worker, expected.kv_transfer_source),
                    "temperature={temperature} multiplier={multiplier}"
                );
            }
        }
    }
}
