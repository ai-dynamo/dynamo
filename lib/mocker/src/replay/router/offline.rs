// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, RouterEvent, WorkerConfigLike, WorkerId, WorkerWithDpRank,
    compute_block_hash_for_seq,
};
use dynamo_kv_router::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, RadixTree, RouterSchedulingPolicy,
    SchedulingPolicy, SchedulingRequest, SequenceRequest, WorkerSelector,
};
use dynamo_tokens::SequenceHash;
use uuid::Uuid;

use super::shared::{
    ReplayNoopPublisher, ReplayWorkerConfig, replay_policy, replay_router_config, replay_selector,
    replay_slots, replay_workers_with_configs,
};
use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::ReplayRequestHashes;

type ReplayQueueKey = <RouterSchedulingPolicy as SchedulingPolicy>::Key;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflinePendingRequestSnapshot {
    pub(crate) uuid: Uuid,
    pub(crate) overlap_blocks_by_worker: Vec<(usize, u32)>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineIndexerSnapshot {
    pub(crate) total_cached_blocks: usize,
    pub(crate) cached_blocks_by_worker: Vec<(usize, usize)>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineRouterSnapshot {
    pub(crate) pending: Vec<OfflinePendingRequestSnapshot>,
    pub(crate) active_blocks_by_worker: Vec<(usize, usize)>,
    pub(crate) active_tokens_by_worker: Vec<(usize, usize)>,
    pub(crate) indexer: OfflineIndexerSnapshot,
}

struct SyncReplayIndexer {
    block_size: u32,
    tree: RadixTree,
}

impl SyncReplayIndexer {
    fn new(block_size: u32) -> Self {
        Self {
            block_size,
            tree: RadixTree::new(),
        }
    }

    fn find_matches_for_request(&self, tokens: &[u32], lora_name: Option<&str>) -> OverlapScores {
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                lora_name,
                ..Default::default()
            },
        );
        self.tree.find_matches(sequence, false)
    }

    fn find_matches_for_hashes(&self, local_block_hashes: Vec<LocalBlockHash>) -> OverlapScores {
        self.tree.find_matches(local_block_hashes, false)
    }

    fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        self.tree.apply_event(event).map_err(Into::into)
    }

    #[cfg(test)]
    fn debug_snapshot(&self) -> OfflineIndexerSnapshot {
        let mut blocks_by_worker = HashMap::<usize, usize>::new();
        for event in self.tree.dump_tree_as_events() {
            *blocks_by_worker
                .entry(event.worker_id as usize)
                .or_default() += 1;
        }
        let mut cached_blocks_by_worker = blocks_by_worker.into_iter().collect::<Vec<_>>();
        cached_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        OfflineIndexerSnapshot {
            total_cached_blocks: self.tree.current_size(),
            cached_blocks_by_worker,
        }
    }
}

struct PendingRequest {
    uuid: Uuid,
    token_seq: Option<Vec<SequenceHash>>,
    isl_tokens: usize,
    overlaps: OverlapScores,
    track_prefill_tokens: bool,
    expected_output_tokens: Option<u32>,
}

impl PendingRequest {
    fn request_id(&self) -> String {
        self.uuid.to_string()
    }

    fn scheduling_request(
        &self,
        decode_blocks: HashMap<WorkerWithDpRank, usize>,
        prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    ) -> SchedulingRequest {
        SchedulingRequest {
            maybe_request_id: Some(self.request_id()),
            token_seq: self.token_seq.clone(),
            isl_tokens: self.isl_tokens,
            overlaps: self.overlaps.clone(),
            decode_blocks,
            prefill_tokens,
            track_prefill_tokens: self.track_prefill_tokens,
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: self.expected_output_tokens,
            allowed_worker_ids: None,
            resp_tx: None,
        }
    }
}

struct QueueEntry {
    key: ReplayQueueKey,
    _enqueue_time_ms: f64,
    enqueue_seq: u64,
    request: PendingRequest,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.enqueue_seq == other.enqueue_seq
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| other.enqueue_seq.cmp(&self.enqueue_seq))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    queue_threshold: Option<f64>,
    workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig>,
    slots: Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>>,
    selector: DefaultWorkerSelector,
    policy: RouterSchedulingPolicy,
    pending: BinaryHeap<QueueEntry>,
    next_enqueue_seq: u64,
    indexer: SyncReplayIndexer,
}

impl OfflineReplayRouter {
    pub(crate) fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        num_workers: usize,
    ) -> Result<Self> {
        let config = replay_router_config(args, router_config);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let selector = replay_selector(&config);
        let policy = replay_policy(&config, args);
        let queue_threshold = if num_workers > 1 {
            config.router_queue_threshold
        } else {
            None
        };

        Ok(Self {
            config,
            block_size: args.block_size as u32,
            queue_threshold,
            workers_with_configs,
            slots,
            selector,
            policy,
            pending: BinaryHeap::new(),
            next_enqueue_seq: 0,
            indexer: SyncReplayIndexer::new(args.block_size as u32),
        })
    }

    pub(crate) fn submit_request_with_hashes(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        now_ms: f64,
    ) -> Result<Option<usize>> {
        let pending = self.build_pending_request(request, replay_hashes)?;
        let should_queue = self
            .queue_threshold
            .is_some_and(|threshold| self.all_workers_busy(threshold));

        if should_queue {
            let key = self.enqueue_key(now_ms, &pending);
            self.pending.push(QueueEntry {
                key,
                _enqueue_time_ms: now_ms,
                enqueue_seq: self.next_enqueue_seq,
                request: pending,
            });
            self.next_enqueue_seq += 1;
            return Ok(None);
        }

        self.admit_request(pending).map(Some)
    }

    pub(crate) fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        self.indexer.apply_event(event)
    }

    pub(crate) fn mark_prefill_completed(&mut self, uuid: Uuid) -> Result<Vec<(Uuid, usize)>> {
        self.slots
            .mark_prefill_completed(&uuid.to_string())
            .map_err(anyhow::Error::from)?;
        self.drain_pending()
    }

    pub(crate) fn free(&mut self, uuid: Uuid) -> Result<Vec<(Uuid, usize)>> {
        self.slots
            .free(&uuid.to_string())
            .map_err(anyhow::Error::from)?;
        self.drain_pending()
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> OfflineRouterSnapshot {
        let mut pending = self
            .pending
            .iter()
            .map(|entry| {
                let mut overlap_blocks_by_worker = entry
                    .request
                    .overlaps
                    .scores
                    .iter()
                    .map(|(worker, overlap)| (worker.worker_id as usize, *overlap))
                    .collect::<Vec<_>>();
                overlap_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

                (
                    entry,
                    OfflinePendingRequestSnapshot {
                        uuid: entry.request.uuid,
                        overlap_blocks_by_worker,
                    },
                )
            })
            .collect::<Vec<_>>();
        pending.sort_unstable_by(|(left_entry, _), (right_entry, _)| {
            left_entry.cmp(right_entry).reverse()
        });

        let mut active_blocks_by_worker = self
            .slots
            .active_blocks()
            .into_iter()
            .map(|(worker, blocks)| (worker.worker_id as usize, blocks))
            .collect::<Vec<_>>();
        active_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        let mut active_tokens_by_worker = self
            .slots
            .active_tokens()
            .into_iter()
            .map(|(worker, tokens)| (worker.worker_id as usize, tokens))
            .collect::<Vec<_>>();
        active_tokens_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        OfflineRouterSnapshot {
            pending: pending.into_iter().map(|(_, snapshot)| snapshot).collect(),
            active_blocks_by_worker,
            active_tokens_by_worker,
            indexer: self.indexer.debug_snapshot(),
        }
    }

    fn enqueue_key(&self, now_ms: f64, request: &PendingRequest) -> ReplayQueueKey {
        let arrival_offset = Duration::from_secs_f64((now_ms.max(0.0)) / 1000.0);
        self.policy.enqueue_key(
            arrival_offset,
            &request.scheduling_request(HashMap::new(), HashMap::new()),
        )
    }

    fn build_pending_request(
        &self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
    ) -> Result<PendingRequest> {
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let (overlaps, token_seq) = match replay_hashes {
            Some(replay_hashes) => {
                let overlaps = self
                    .indexer
                    .find_matches_for_hashes(replay_hashes.local_block_hashes);
                let token_seq = if !self.config.router_track_active_blocks {
                    None
                } else if self.config.router_assume_kv_reuse {
                    Some(replay_hashes.sequence_hashes)
                } else {
                    self.config.compute_seq_hashes_for_tracking(
                        &request.tokens,
                        self.block_size,
                        None,
                        BlockHashOptions::default(),
                        None,
                    )
                };
                (overlaps, token_seq)
            }
            None => {
                let overlaps = self.indexer.find_matches_for_request(&request.tokens, None);
                let token_seq = self.config.compute_seq_hashes_for_tracking(
                    &request.tokens,
                    self.block_size,
                    None,
                    BlockHashOptions::default(),
                    None,
                );
                (overlaps, token_seq)
            }
        };

        Ok(PendingRequest {
            uuid,
            token_seq,
            isl_tokens: request.tokens.len(),
            overlaps,
            track_prefill_tokens: self.config.router_track_prefill_tokens,
            expected_output_tokens: Some(
                u32::try_from(request.max_output_tokens)
                    .context("max_output_tokens does not fit into u32")?,
            ),
        })
    }

    fn admit_request(&mut self, request: PendingRequest) -> Result<usize> {
        let (decode_blocks, prefill_tokens) = self
            .slots
            .potential_blocks_and_tokens_with_prefill_tracking(
                request.token_seq.as_deref(),
                request.isl_tokens,
                request.overlaps.clone(),
                request.track_prefill_tokens,
            );
        let scheduling_request = request.scheduling_request(decode_blocks, prefill_tokens);
        let selection = self.selector.select_worker(
            &self.workers_with_configs,
            &scheduling_request,
            self.block_size,
        )?;
        let worker_idx = usize::try_from(selection.worker.worker_id)
            .map_err(|_| anyhow!("selected worker id does not fit into usize"))?;
        let request_id = request.request_id();

        self.slots
            .add_request(SequenceRequest {
                request_id,
                token_sequence: request.token_seq,
                isl: request.isl_tokens,
                overlap: selection.overlap_blocks,
                track_prefill_tokens: request.track_prefill_tokens,
                expected_output_tokens: request.expected_output_tokens,
                worker: selection.worker,
                lora_name: None,
            })
            .map_err(anyhow::Error::from)?;

        Ok(worker_idx)
    }

    fn drain_pending(&mut self) -> Result<Vec<(Uuid, usize)>> {
        let Some(threshold) = self.queue_threshold else {
            return Ok(Vec::new());
        };

        let mut admissions = Vec::new();
        while !self.all_workers_busy(threshold) {
            let Some(QueueEntry { request, .. }) = self.pending.pop() else {
                break;
            };
            let uuid = request.uuid;
            let worker_idx = self.admit_request(request)?;
            admissions.push((uuid, worker_idx));
        }

        Ok(admissions)
    }

    fn all_workers_busy(&self, threshold: f64) -> bool {
        let mut checked_any = false;
        let any_worker_not_busy = self
            .slots
            .any_worker_matches_active_tokens(|worker, tokens| {
                let Some(config) = self.workers_with_configs.get(&worker.worker_id) else {
                    return false;
                };
                checked_any = true;
                let max_batched = config
                    .max_num_batched_tokens()
                    .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
                (tokens as f64) <= threshold * (max_batched as f64)
            });

        checked_any && !any_worker_not_busy
    }
}
