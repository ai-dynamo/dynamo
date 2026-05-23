// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G4/object-storage search machinery, salvaged near-verbatim from the removed
//! `leader::session::initiator` peer protocol.
//!
//! Only the local-tier + G4 portions were preserved; all peer/transport state
//! (`remote_g2_blocks`, `remote_g3_blocks`, `MessageTransport`, the peer arm of
//! the search select-loop) was dropped. The two internal channel messages
//! (`G4Results`, `G4LoadComplete`) — which were never wire messages — are now
//! the private [`G4Message`] enum.

use anyhow::Result;
use tokio::sync::{Mutex, mpsc, watch};
use tokio::task::JoinHandle;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::leader::blocks::BlockHolder;
use crate::leader::staging;
use crate::leader::{MatchBreakdown, OnboardingStatus};
use crate::{
    BlockId, G2, G3, SequenceHash, object::ObjectBlockOps, worker::group::ParallelWorkers,
};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::{blocks::ImmutableBlock, manager::BlockManager};

/// Internal channel message for the G4 search/load task results.
///
/// Replaces the former `OnboardMessage::{G4Results, G4LoadComplete}` variants,
/// which were only ever sent over a private in-process mpsc channel.
enum G4Message {
    /// Hashes found in object storage (with their byte sizes).
    Results {
        found_hashes: Vec<(SequenceHash, usize)>,
    },
    /// Result of loading G4 blocks into local G2.
    LoadComplete {
        success: Vec<SequenceHash>,
        failures: Vec<(SequenceHash, String)>,
        blocks: Arc<Vec<ImmutableBlock<G2>>>,
    },
}

/// Validate that sequence hashes have contiguous positions (X, X+1, X+2, ...).
///
/// The positions don't need to start at 0, but they must be monotonically
/// increasing with no gaps.
fn validate_contiguous_positions(seq_hashes: &[SequenceHash]) -> Result<()> {
    if seq_hashes.len() <= 1 {
        return Ok(());
    }

    let mut positions: Vec<u64> = seq_hashes.iter().map(|h| h.position()).collect();
    positions.sort();

    for window in positions.windows(2) {
        if window[1] != window[0] + 1 {
            anyhow::bail!(
                "Position gap detected in blocks: {} -> {} (expected {}). \
                 This indicates a block ordering bug.",
                window[0],
                window[1],
                window[0] + 1
            );
        }
    }

    Ok(())
}

/// Tracks G4/object storage search state for parallel search.
///
/// This state is used when G4 search runs in parallel with G2/G3 search.
/// The first responder (local or G4) wins for each hash.
#[derive(Default)]
pub struct G4SearchState {
    /// Hashes won by G4 in the first-responder-wins race
    won_hashes: HashSet<SequenceHash>,
    /// Hashes currently pending load (get_blocks in progress)
    pending_load: HashSet<SequenceHash>,
    /// Hashes that failed to load with error messages
    failed_hashes: HashMap<SequenceHash, String>,
    /// Block IDs allocated for G4→G2 loading (sequence_hash → block_id)
    allocated_blocks: HashMap<SequenceHash, BlockId>,
}

impl G4SearchState {
    fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.won_hashes.clear();
        self.pending_load.clear();
        self.failed_hashes.clear();
        self.allocated_blocks.clear();
    }
}

/// Local-tier + G4 search machinery.
///
/// Seeds the upcoming async remote-search path. Holds only local G2/G3 blocks
/// and G4 state — no peer/transport coupling.
pub struct AsyncSearch {
    session_id: uuid::Uuid,
    g2_manager: Arc<BlockManager<G2>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    parallel_worker: Option<Arc<dyn ParallelWorkers>>,
    status_tx: watch::Sender<OnboardingStatus>,

    // Held blocks from local search using BlockHolder for RAII semantics
    local_g2_blocks: BlockHolder<G2>,
    local_g3_blocks: BlockHolder<G3>,

    // Shared with FindMatchesResult for block access
    all_g2_blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
    match_breakdown: Arc<Mutex<MatchBreakdown>>,

    // G4/Object storage fields
    /// Object storage client for G4 search and load (leader-initiated)
    object_client: Option<Arc<dyn ObjectBlockOps>>,
    /// G4 search state tracking won hashes, pending loads, and failures
    g4_state: G4SearchState,
    /// Channel for receiving G4 search/load results
    g4_rx: Option<mpsc::Receiver<G4Message>>,
    /// Handle for G4 search task (for cancellation on drop)
    g4_task_handle: Option<JoinHandle<()>>,
}

impl AsyncSearch {
    /// Create a new async search context.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_id: uuid::Uuid,
        g2_manager: Arc<BlockManager<G2>>,
        g3_manager: Option<Arc<BlockManager<G3>>>,
        parallel_worker: Option<Arc<dyn ParallelWorkers>>,
        status_tx: watch::Sender<OnboardingStatus>,
        all_g2_blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
        match_breakdown: Arc<Mutex<MatchBreakdown>>,
        object_client: Option<Arc<dyn ObjectBlockOps>>,
    ) -> Self {
        Self {
            session_id,
            g2_manager,
            g3_manager,
            parallel_worker,
            status_tx,
            local_g2_blocks: BlockHolder::empty(),
            local_g3_blocks: BlockHolder::empty(),
            all_g2_blocks,
            match_breakdown,
            object_client,
            g4_state: G4SearchState::new(),
            g4_rx: None,
            g4_task_handle: None,
        }
    }

    fn publish_match_breakdown(&self) {
        if let Ok(mut breakdown) = self.match_breakdown.try_lock() {
            *breakdown = MatchBreakdown {
                host_blocks: self.local_g2_blocks.count(),
                disk_blocks: self.local_g3_blocks.count(),
                object_blocks: self.g4_state.won_hashes.len(),
            };
        }
    }

    /// Search local G2/G3, then G4 if configured. First-responder-wins per hash.
    async fn search_phase(&mut self, sequence_hashes: &[SequenceHash]) -> Result<()> {
        // Local G2 search
        self.local_g2_blocks = BlockHolder::new(self.g2_manager.match_blocks(sequence_hashes));

        let mut matched_hashes: HashSet<SequenceHash> =
            self.local_g2_blocks.sequence_hashes().into_iter().collect();

        // Local G3 search
        if let Some(ref g3_manager) = self.g3_manager {
            let remaining: Vec<_> = sequence_hashes
                .iter()
                .filter(|h| !matched_hashes.contains(h))
                .copied()
                .collect();

            if !remaining.is_empty() {
                self.local_g3_blocks = BlockHolder::new(g3_manager.match_blocks(&remaining));
                for hash in self.local_g3_blocks.sequence_hashes() {
                    matched_hashes.insert(hash);
                }
            }
        }

        // Continue to G4 only if blocks remain and object storage is configured.
        let remaining_hashes: Vec<_> = sequence_hashes
            .iter()
            .filter(|h| !matched_hashes.contains(h))
            .copied()
            .collect();

        if remaining_hashes.is_empty()
            || self.object_client.is_none()
            || self.parallel_worker.is_none()
        {
            return Ok(());
        }

        self.status_tx.send(OnboardingStatus::Searching).ok();

        let (tx, rx) = mpsc::channel(16);
        self.g4_rx = Some(rx);
        let handle = self.spawn_g4_search(remaining_hashes, tx.clone());
        self.g4_task_handle = Some(handle);

        self.process_g4_responses(&mut matched_hashes, tx).await?;

        Ok(())
    }

    /// Drain the G4 result channel: process search hits, kick loads, await loads.
    async fn process_g4_responses(
        &mut self,
        matched_hashes: &mut HashSet<SequenceHash>,
        g4_tx: mpsc::Sender<G4Message>,
    ) -> Result<()> {
        let mut pending_g4_search = self.g4_rx.is_some();
        let mut pending_g4_load = false;

        while pending_g4_search || pending_g4_load {
            let Some(msg) = (match self.g4_rx {
                Some(ref mut rx) => rx.recv().await,
                None => None,
            }) else {
                break;
            };

            match msg {
                G4Message::Results { found_hashes } => {
                    pending_g4_search = false;
                    let won_hashes = self.process_g4_results(found_hashes, matched_hashes);
                    if !won_hashes.is_empty() {
                        self.load_g4_blocks(won_hashes, g4_tx.clone()).await?;
                        pending_g4_load = true;
                    }
                }
                G4Message::LoadComplete {
                    success,
                    failures,
                    blocks,
                } => {
                    self.handle_g4_load_complete(success, failures, blocks);
                    pending_g4_load = false;
                }
            }
        }

        Ok(())
    }

    /// Apply "first hole" policy: trim results to first contiguous sequence.
    async fn apply_find_policy(&mut self, sequence_hashes: &[SequenceHash]) -> Result<()> {
        let mut matched_hashes: HashSet<SequenceHash> = HashSet::new();

        for hash in self.local_g2_blocks.sequence_hashes() {
            matched_hashes.insert(hash);
        }
        for hash in self.local_g3_blocks.sequence_hashes() {
            matched_hashes.insert(hash);
        }
        for hash in &self.g4_state.won_hashes {
            matched_hashes.insert(*hash);
        }

        let mut keep_count = 0;
        for hash in sequence_hashes {
            if matched_hashes.contains(hash) {
                keep_count += 1;
            } else {
                break;
            }
        }

        if keep_count == sequence_hashes.len() || keep_count == matched_hashes.len() {
            return Ok(());
        }

        let keep_hashes: Vec<SequenceHash> = sequence_hashes[..keep_count].to_vec();
        let keep_set: HashSet<&SequenceHash> = keep_hashes.iter().collect();

        // Filter local blocks
        self.local_g2_blocks.retain(&keep_hashes);
        self.local_g3_blocks.retain(&keep_hashes);

        // Filter G4 state - release allocated blocks beyond first hole
        let g4_release_hashes: Vec<SequenceHash> = self
            .g4_state
            .won_hashes
            .iter()
            .filter(|h| !keep_set.contains(h))
            .copied()
            .collect();

        for hash in &g4_release_hashes {
            self.g4_state.won_hashes.remove(hash);
            self.g4_state.pending_load.remove(hash);
            self.g4_state.allocated_blocks.remove(hash);
        }

        Ok(())
    }

    /// Stage local G3→G2.
    async fn stage_local_g3_to_g2(&mut self) -> Result<()> {
        if self.local_g3_blocks.is_empty() {
            return Ok(());
        }

        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ParallelWorker required for G3→G2 staging"))?;

        let result =
            staging::stage_g3_to_g2(&self.local_g3_blocks, &self.g2_manager, &**parallel_worker)
                .await?;

        let _ = self.local_g3_blocks.take_all();
        self.local_g2_blocks.extend(result.new_g2_blocks);

        Ok(())
    }

    /// Consolidate all G2 blocks into shared storage, sorted by position.
    async fn consolidate_blocks(&mut self) {
        let mut all_blocks = self.local_g2_blocks.take_all();

        all_blocks.sort_by_key(|b| b.sequence_hash().position());

        let seq_hashes: Vec<SequenceHash> = all_blocks.iter().map(|b| b.sequence_hash()).collect();
        if let Err(e) = validate_contiguous_positions(&seq_hashes) {
            tracing::warn!(
                session_id = %self.session_id,
                error = %e,
                "Block positions are not contiguous — proceeding with sorted order"
            );
        }

        let matched_blocks = all_blocks.len();
        *self.all_g2_blocks.lock().await = Some(all_blocks);

        self.status_tx
            .send(OnboardingStatus::Complete { matched_blocks })
            .ok();
    }

    // =========================================================================
    // G4/Object Storage Methods
    // =========================================================================

    /// Spawn a G4 search task. Calls `has_blocks` via parallel_worker which fans
    /// out to workers with rank-prefixed keys (so we must query through them).
    fn spawn_g4_search(
        &self,
        sequence_hashes: Vec<SequenceHash>,
        tx: mpsc::Sender<G4Message>,
    ) -> JoinHandle<()> {
        let session_id = self.session_id;
        let parallel_worker = self.parallel_worker.clone();

        tokio::spawn(async move {
            let Some(worker) = parallel_worker else {
                let _ = tx
                    .send(G4Message::Results {
                        found_hashes: vec![],
                    })
                    .await;
                return;
            };

            let results = worker.has_blocks(sequence_hashes).await;

            let found_hashes: Vec<(SequenceHash, usize)> = results
                .into_iter()
                .filter_map(|(hash, size_opt)| size_opt.map(|size| (hash, size)))
                .collect();

            tracing::debug!(
                session_id = %session_id,
                count = found_hashes.len(),
                "G4 search: found blocks in object storage"
            );

            let _ = tx.send(G4Message::Results { found_hashes }).await;
        })
    }

    /// Process G4 search results with first-responder-wins logic.
    ///
    /// Returns the hashes that G4 won (not already claimed by G2/G3).
    fn process_g4_results(
        &mut self,
        found_hashes: Vec<(SequenceHash, usize)>,
        matched_hashes: &mut HashSet<SequenceHash>,
    ) -> Vec<SequenceHash> {
        let mut won_hashes = Vec::new();

        for (hash, _size) in found_hashes {
            if matched_hashes.insert(hash) {
                won_hashes.push(hash);
                self.g4_state.won_hashes.insert(hash);
            }
        }

        won_hashes
    }

    /// Load G4 blocks into local G2 via workers.
    async fn load_g4_blocks(
        &mut self,
        won_hashes: Vec<SequenceHash>,
        g4_tx: mpsc::Sender<G4Message>,
    ) -> Result<()> {
        if won_hashes.is_empty() {
            return Ok(());
        }

        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ParallelWorkers required for G4 load"))?;

        for hash in &won_hashes {
            self.g4_state.pending_load.insert(*hash);
        }

        let dst_blocks = self
            .g2_manager
            .allocate_blocks(won_hashes.len())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to allocate {} G2 blocks for G4 load",
                    won_hashes.len()
                )
            })?;

        let dst_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

        for (hash, block_id) in won_hashes.iter().zip(dst_ids.iter()) {
            self.g4_state.allocated_blocks.insert(*hash, *block_id);
        }

        let session_id = self.session_id;
        let hashes = won_hashes.clone();
        let parallel_worker = parallel_worker.clone();
        let g2_manager = self.g2_manager.clone();

        // Spawn load task. dst_blocks is moved in to keep them alive during download.
        tokio::spawn(async move {
            let results = parallel_worker
                .get_blocks(hashes.clone(), LogicalLayoutHandle::G2, dst_ids.clone())
                .await;

            let mut success = Vec::new();
            let mut failures = Vec::new();
            let mut blocks = Vec::new();

            for ((result, dst_block), seq_hash) in
                results.into_iter().zip(dst_blocks).zip(hashes.iter())
            {
                match result {
                    Ok(hash) => {
                        let complete = dst_block
                            .stage(*seq_hash, g2_manager.block_size())
                            .expect("block size mismatch");
                        let immutable = g2_manager.register_block(complete);
                        blocks.push(immutable);
                        success.push(hash);
                    }
                    Err(hash) => {
                        failures.push((hash, "Failed to download block".to_string()));
                    }
                }
            }

            tracing::debug!(
                session_id = %session_id,
                success_count = success.len(),
                failure_count = failures.len(),
                "G4 load complete"
            );

            let _ = g4_tx
                .send(G4Message::LoadComplete {
                    success,
                    failures,
                    blocks: Arc::new(blocks),
                })
                .await;
        });

        Ok(())
    }

    /// Handle G4 load completion, updating state and adding blocks to local_g2_blocks.
    fn handle_g4_load_complete(
        &mut self,
        success: Vec<SequenceHash>,
        failures: Vec<(SequenceHash, String)>,
        blocks: Arc<Vec<ImmutableBlock<G2>>>,
    ) {
        for hash in &success {
            self.g4_state.pending_load.remove(hash);
            self.g4_state.allocated_blocks.remove(hash);
        }

        let blocks =
            Arc::try_unwrap(blocks).expect("G4LoadComplete should be the sole owner of blocks");

        self.local_g2_blocks.extend(blocks);

        for (hash, error) in failures {
            self.g4_state.pending_load.remove(&hash);
            self.g4_state.failed_hashes.insert(hash, error);
            self.g4_state.allocated_blocks.remove(&hash);
            self.g4_state.won_hashes.remove(&hash);
        }
    }
}
