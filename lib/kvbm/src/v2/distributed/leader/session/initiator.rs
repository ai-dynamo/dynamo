// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use tokio::sync::{Mutex, mpsc, watch};
use tokio::task::JoinHandle;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{
    logical::{blocks::ImmutableBlock, manager::BlockManager},
    physical::transfer::TransferOptions,
    v2::{
        BlockId, G2, G3, InstanceId, SequenceHash, distributed::object::ObjectBlockOps,
        distributed::parallelism::ParallelWorker, logical::LogicalLayoutHandle,
        physical::manager::LayoutHandle,
    },
};

use super::{
    super::{OnboardingStatus, SessionControl, StagingMode},
    BlockHolder, SessionId,
    messages::OnboardMessage,
    transport::MessageTransport,
};

/// Validate that sequence hashes have contiguous positions (X, X+1, X+2, ...).
///
/// The positions don't need to start at 0, but they must be monotonically
/// increasing with no gaps.
fn validate_contiguous_positions(seq_hashes: &[SequenceHash]) -> Result<()> {
    if seq_hashes.len() <= 1 {
        return Ok(());
    }

    // Collect and sort positions
    let mut positions: Vec<u64> = seq_hashes.iter().map(|h| h.position()).collect();
    positions.sort();

    // Check monotonically increasing with no holes: X, X+1, X+2, ...
    for window in positions.windows(2) {
        if window[1] != window[0] + 1 {
            anyhow::bail!(
                "Position gap detected in remote blocks: {} -> {} (expected {}). \
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
/// The first responder (local, remote, or G4) wins for each hash.
#[derive(Default)]
struct G4SearchState {
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

    /// Clear all state.
    #[expect(dead_code)]
    fn clear(&mut self) {
        self.won_hashes.clear();
        self.pending_load.clear();
        self.failed_hashes.clear();
        self.allocated_blocks.clear();
    }
}

/// Initiator-side session for coordinating distributed block search.
///
/// Supports three staging modes:
/// - Hold: Find and hold blocks (G2+G3), no staging
/// - Prepare: Stage G3→G2 everywhere, keep session alive
/// - Full: Stage G3→G2 + RDMA pull remote G2→local G2, session completes
pub struct InitiatorSession {
    session_id: SessionId,
    instance_id: InstanceId,
    mode: StagingMode,
    g2_manager: Arc<BlockManager<G2>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    parallel_worker: Option<Arc<dyn ParallelWorker>>,
    transport: Arc<MessageTransport>,
    status_tx: watch::Sender<OnboardingStatus>,

    // Held blocks from local search using BlockHolder for RAII semantics
    local_g2_blocks: BlockHolder<G2>,
    local_g3_blocks: BlockHolder<G3>,

    // Track remote blocks by tier
    remote_g2_blocks: HashMap<InstanceId, Vec<BlockId>>, // G2: track block IDs
    remote_g2_hashes: HashMap<InstanceId, Vec<SequenceHash>>, // G2: track sequence hashes (parallel to block_ids)
    remote_g3_blocks: HashMap<InstanceId, Vec<SequenceHash>>, // G3: track sequence hashes
    remote_g2_layouts: HashMap<InstanceId, LayoutHandle>,     // G2 layouts for RDMA

    // Shared with FindMatchesResult for block access
    all_g2_blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,

    // Control channel for deferred operations
    control_rx: mpsc::Receiver<SessionControl>,

    // G4/Object storage fields
    /// Object storage client for G4 search and load (leader-initiated)
    object_client: Option<Arc<dyn ObjectBlockOps>>,
    /// G4 search state tracking won hashes, pending loads, and failures
    g4_state: G4SearchState,
    /// Channel for receiving G4 search/load results
    g4_rx: Option<mpsc::Receiver<OnboardMessage>>,
    /// Handle for G4 search task (for cancellation on drop)
    #[allow(dead_code)]
    g4_task_handle: Option<JoinHandle<()>>,
}

impl InitiatorSession {
    /// Create a new initiator session.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session_id: SessionId,
        instance_id: InstanceId,
        mode: StagingMode,
        g2_manager: Arc<BlockManager<G2>>,
        g3_manager: Option<Arc<BlockManager<G3>>>,
        parallel_worker: Option<Arc<dyn ParallelWorker>>,
        transport: Arc<MessageTransport>,
        status_tx: watch::Sender<OnboardingStatus>,
        all_g2_blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
        control_rx: mpsc::Receiver<SessionControl>,
        object_client: Option<Arc<dyn ObjectBlockOps>>,
    ) -> Self {
        Self {
            session_id,
            instance_id,
            mode,
            g2_manager,
            g3_manager,
            parallel_worker,
            transport,
            status_tx,
            local_g2_blocks: BlockHolder::empty(),
            local_g3_blocks: BlockHolder::empty(),
            remote_g2_blocks: HashMap::new(),
            remote_g2_hashes: HashMap::new(),
            remote_g3_blocks: HashMap::new(),
            remote_g2_layouts: HashMap::new(),
            all_g2_blocks,
            control_rx,
            object_client,
            g4_state: G4SearchState::new(),
            g4_rx: None,
            g4_task_handle: None,
        }
    }

    /// Run the initiator session task.
    pub async fn run(
        mut self,
        mut rx: mpsc::Receiver<OnboardMessage>,
        remote_leaders: Vec<InstanceId>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<()> {
        tracing::debug!(
            session_id = %self.session_id,
            mode = ?self.mode,
            num_hashes = sequence_hashes.len(),
            num_remotes = remote_leaders.len(),
            "Starting initiator session"
        );

        // Phase 1: Search (local G2 and G3, then remote if needed)
        self.search_phase(&mut rx, &remote_leaders, &sequence_hashes)
            .await?;

        // Phase 1.5: Apply find policy (first-hole detection)
        // Trims results to first contiguous sequence from start
        self.apply_find_policy(&sequence_hashes).await?;

        tracing::debug!(
            session_id = %self.session_id,
            "search_phase complete, entering mode handler"
        );

        // Phase 2: Staging based on mode
        match self.mode {
            StagingMode::Hold => {
                tracing::debug!(session_id = %self.session_id, "Calling hold_mode()");
                self.hold_mode().await?;
                // Wait for control commands or shutdown
                self.await_commands(rx).await?;
            }
            StagingMode::Prepare => {
                self.prepare_mode().await?;
                // Wait for pull command or shutdown
                self.await_commands(rx).await?;
            }
            StagingMode::Full => {
                self.full_mode().await?;
                // Completes and exits
            }
        }

        Ok(())
    }

    /// Phase 1: Search for blocks locally and remotely.
    async fn search_phase(
        &mut self,
        rx: &mut mpsc::Receiver<OnboardMessage>,
        remote_leaders: &[InstanceId],
        sequence_hashes: &[SequenceHash],
    ) -> Result<()> {
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

        // Check if remote/G4 search needed
        // Continue if: not all matched locally AND (remote leaders exist OR object_client configured)
        let has_object_client = self.object_client.is_some();
        if matched_hashes.len() == sequence_hashes.len()
            || (remote_leaders.is_empty() && !has_object_client)
        {
            return Ok(());
        }

        // Remote search
        let remaining_hashes: Vec<_> = sequence_hashes
            .iter()
            .filter(|h| !matched_hashes.contains(h))
            .copied()
            .collect();

        if remaining_hashes.is_empty() {
            return Ok(());
        }

        self.status_tx.send(OnboardingStatus::Searching).ok();

        // Send CreateSession to all remotes FIRST
        for remote in remote_leaders {
            let msg = OnboardMessage::CreateSession {
                requester: self.instance_id,
                session_id: self.session_id,
                sequence_hashes: remaining_hashes.clone(),
            };
            self.transport.send(*remote, msg).await?;
        }

        // Then spawn G4 search task if object storage is configured and parallel_worker is available
        // We use parallel_worker.has_blocks() which fans out to workers with rank-prefixed keys
        let g4_tx = if self.object_client.is_some() && self.parallel_worker.is_some() {
            let (tx, rx) = mpsc::channel(16);
            self.g4_rx = Some(rx);
            // Spawn G4 search - searches the same remaining hashes as remote search
            let handle = self.spawn_g4_search(remaining_hashes.clone(), tx.clone());
            self.g4_task_handle = Some(handle);
            Some(tx)
        } else {
            None
        };

        // Process search responses (including G4 if configured)
        self.process_search_responses(rx, remote_leaders, &mut matched_hashes, g4_tx)
            .await?;

        Ok(())
    }

    /// Process G2Results, G3Results, and G4 results from responders.
    ///
    /// Uses `tokio::select!` to handle both remote messages and G4 results
    /// in parallel, applying first-responder-wins logic across all tiers.
    async fn process_search_responses(
        &mut self,
        rx: &mut mpsc::Receiver<OnboardMessage>,
        remote_leaders: &[InstanceId],
        matched_hashes: &mut HashSet<SequenceHash>,
        g4_tx: Option<mpsc::Sender<OnboardMessage>>,
    ) -> Result<()> {
        let mut pending_g2_responses = remote_leaders.len();
        let mut pending_g3_responses: HashSet<InstanceId> =
            remote_leaders.iter().copied().collect();
        let mut pending_search_complete: HashSet<InstanceId> =
            remote_leaders.iter().copied().collect();
        let mut pending_acknowledgments: HashSet<InstanceId> = HashSet::new();

        // G4 state tracking
        let mut pending_g4_search = self.g4_rx.is_some();
        let mut pending_g4_load = false;

        // Helper to check if all responses are complete
        let is_complete = |pending_g2: usize,
                           pending_g3: &HashSet<InstanceId>,
                           pending_ack: &HashSet<InstanceId>,
                           pending_search: &HashSet<InstanceId>,
                           pending_g4_s: bool,
                           pending_g4_l: bool| {
            pending_g2 == 0
                && pending_g3.is_empty()
                && pending_ack.is_empty()
                && pending_search.is_empty()
                && !pending_g4_s
                && !pending_g4_l
        };

        loop {
            // Check completion before waiting for more messages
            if is_complete(
                pending_g2_responses,
                &pending_g3_responses,
                &pending_acknowledgments,
                &pending_search_complete,
                pending_g4_search,
                pending_g4_load,
            ) {
                tracing::debug!(
                    session_id = %self.session_id,
                    "All responses received (including G4), exiting search_phase"
                );
                break;
            }

            tokio::select! {
                // Handle G4 messages from internal channel
                g4_msg = async {
                    if let Some(ref mut g4_rx) = self.g4_rx {
                        g4_rx.recv().await
                    } else {
                        std::future::pending::<Option<OnboardMessage>>().await
                    }
                } => {
                    let Some(msg) = g4_msg else {
                        // Channel closed unexpectedly
                        pending_g4_search = false;
                        pending_g4_load = false;
                        continue;
                    };

                    tracing::debug!(
                        session_id = %self.session_id,
                        msg = msg.variant_name(),
                        "process_search_responses received G4"
                    );

                    match msg {
                        OnboardMessage::G4Results { found_hashes, .. } => {
                            pending_g4_search = false;

                            // Process G4 results with first-responder-wins
                            let won_hashes = self.process_g4_results(found_hashes, matched_hashes);

                            // If G4 won any hashes, start loading them
                            if !won_hashes.is_empty() {
                                if let Some(ref tx) = g4_tx {
                                    self.load_g4_blocks(won_hashes, tx.clone()).await?;
                                    pending_g4_load = true;
                                }
                            }
                        }
                        OnboardMessage::G4LoadComplete { success, failures, blocks, .. } => {
                            self.handle_g4_load_complete(success, failures, blocks);
                            pending_g4_load = false;
                        }
                        _ => {}
                    }
                }

                // Handle remote messages
                remote_msg = rx.recv() => {
                    let Some(msg) = remote_msg else {
                        // Channel closed - exit loop
                        break;
                    };

                    tracing::debug!(
                        session_id = %self.session_id,
                        msg = msg.variant_name(),
                        "process_search_responses received"
                    );

                    match msg {
                        OnboardMessage::G2Results {
                            responder,
                            layout_handle,
                            sequence_hashes,
                            block_ids,
                            ..
                        } => {
                            tracing::debug!(
                                session_id = %self.session_id,
                                responder = %responder,
                                num_hashes = sequence_hashes.len(),
                                "Processing G2Results"
                            );
                            // Store layout for RDMA operations
                            self.remote_g2_layouts.insert(responder, layout_handle);

                            // First-responder-wins logic using sequence hashes
                            let mut hold_hashes = Vec::new();
                            let mut drop_hashes = Vec::new();

                            for (seq_hash, block_id) in sequence_hashes.iter().zip(block_ids.iter()) {
                                if matched_hashes.insert(*seq_hash) {
                                    hold_hashes.push(*seq_hash);
                                    self.remote_g2_blocks
                                        .entry(responder)
                                        .or_default()
                                        .push(*block_id);
                                    // Track sequence hash in parallel for block registration after RDMA pull
                                    self.remote_g2_hashes
                                        .entry(responder)
                                        .or_default()
                                        .push(*seq_hash);
                                } else {
                                    drop_hashes.push(*seq_hash);
                                }
                            }

                            // Send HoldBlocks decision
                            self.transport
                                .send(
                                    responder,
                                    OnboardMessage::HoldBlocks {
                                        requester: self.instance_id,
                                        session_id: self.session_id,
                                        hold_hashes,
                                        drop_hashes,
                                    },
                                )
                                .await?;

                            pending_acknowledgments.insert(responder);
                            pending_g2_responses -= 1;
                        }
                        OnboardMessage::G3Results {
                            responder,
                            sequence_hashes,
                            ..
                        } => {
                            // Store G3 sequence hashes for later staging
                            for seq_hash in sequence_hashes {
                                if matched_hashes.insert(seq_hash) {
                                    self.remote_g3_blocks
                                        .entry(responder)
                                        .or_default()
                                        .push(seq_hash);
                                }
                            }

                            pending_g3_responses.remove(&responder);
                        }
                        OnboardMessage::SearchComplete { responder, .. } => {
                            pending_search_complete.remove(&responder);
                            // SearchComplete means responder is done with G2 AND G3 search
                            pending_g3_responses.remove(&responder);

                            tracing::debug!(
                                session_id = %self.session_id,
                                responder = %responder,
                                g2_pending = pending_g2_responses,
                                g3_pending = pending_g3_responses.len(),
                                ack_pending = pending_acknowledgments.len(),
                                search_pending = pending_search_complete.len(),
                                g4_search = pending_g4_search,
                                g4_load = pending_g4_load,
                                "SearchComplete"
                            );
                        }
                        OnboardMessage::Acknowledged { responder, .. } => {
                            pending_acknowledgments.remove(&responder);
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply "first hole" policy: trim results to first contiguous sequence.
    ///
    /// This implements the policy where we only return blocks from position 0
    /// up to (but not including) the first missing block. Any blocks after the
    /// first hole are released.
    ///
    /// # Arguments
    /// * `sequence_hashes` - The original query hashes in order (position 0 to N)
    async fn apply_find_policy(&mut self, sequence_hashes: &[SequenceHash]) -> Result<()> {
        // Build set of all matched hashes (local + remote)
        let mut matched_hashes: HashSet<SequenceHash> = HashSet::new();

        // Local G2 blocks
        for hash in self.local_g2_blocks.sequence_hashes() {
            matched_hashes.insert(hash);
        }

        // Local G3 blocks
        for hash in self.local_g3_blocks.sequence_hashes() {
            matched_hashes.insert(hash);
        }

        // Remote G2 hashes
        for hashes in self.remote_g2_hashes.values() {
            for hash in hashes {
                matched_hashes.insert(*hash);
            }
        }

        // Remote G3 hashes
        for hashes in self.remote_g3_blocks.values() {
            for hash in hashes {
                matched_hashes.insert(*hash);
            }
        }

        // G4 won hashes (blocks successfully loaded from object storage)
        for hash in &self.g4_state.won_hashes {
            matched_hashes.insert(*hash);
        }

        // Find the first hole: count contiguous matches from start
        let mut keep_count = 0;
        for hash in sequence_hashes {
            if matched_hashes.contains(hash) {
                keep_count += 1;
            } else {
                // First hole found - stop here
                break;
            }
        }

        // If all hashes matched or first hole is at position 0, nothing to trim
        if keep_count == sequence_hashes.len() || keep_count == matched_hashes.len() {
            tracing::debug!(
                session_id = %self.session_id,
                matched = keep_count,
                total = sequence_hashes.len(),
                "apply_find_policy: no trimming needed"
            );
            return Ok(());
        }

        // Get the hashes to keep
        let keep_hashes: Vec<SequenceHash> = sequence_hashes[..keep_count].to_vec();
        let keep_set: HashSet<&SequenceHash> = keep_hashes.iter().collect();

        tracing::debug!(
            session_id = %self.session_id,
            from = matched_hashes.len(),
            to = keep_count,
            first_hole = keep_count,
            "apply_find_policy: trimming blocks"
        );

        // Filter local blocks
        self.local_g2_blocks.retain(&keep_hashes);
        self.local_g3_blocks.retain(&keep_hashes);

        // Filter remote G2 block tracking and send ReleaseBlocks messages
        for (remote_instance, block_ids) in &mut self.remote_g2_blocks {
            let hashes = self.remote_g2_hashes.get_mut(remote_instance);
            if let Some(hashes) = hashes {
                // Find indices of blocks to release
                let mut release_indices = Vec::new();
                for (i, hash) in hashes.iter().enumerate() {
                    if !keep_set.contains(hash) {
                        release_indices.push(i);
                    }
                }

                // Collect hashes to release for ReleaseBlocks message
                let release_hashes: Vec<SequenceHash> =
                    release_indices.iter().map(|&i| hashes[i]).collect();

                // Remove from tracking (reverse order to preserve indices)
                for i in release_indices.into_iter().rev() {
                    hashes.remove(i);
                    block_ids.remove(i);
                }

                // Send ReleaseBlocks message if any blocks need releasing
                if !release_hashes.is_empty() {
                    tracing::debug!(
                        session_id = %self.session_id,
                        count = release_hashes.len(),
                        instance = %remote_instance,
                        "Releasing G2 blocks beyond first hole"
                    );
                    self.transport
                        .send(
                            *remote_instance,
                            OnboardMessage::ReleaseBlocks {
                                requester: self.instance_id,
                                session_id: self.session_id,
                                release_hashes,
                            },
                        )
                        .await?;
                }
            }
        }

        // Filter remote G3 block tracking and send ReleaseBlocks messages
        for (remote_instance, hashes) in &mut self.remote_g3_blocks {
            // Find hashes to release
            let release_hashes: Vec<SequenceHash> = hashes
                .iter()
                .filter(|h| !keep_set.contains(h))
                .copied()
                .collect();

            // Remove from tracking
            hashes.retain(|h| keep_set.contains(h));

            // Send ReleaseBlocks message if any blocks need releasing
            if !release_hashes.is_empty() {
                tracing::debug!(
                    session_id = %self.session_id,
                    count = release_hashes.len(),
                    instance = %remote_instance,
                    "Releasing G3 blocks beyond first hole"
                );
                self.transport
                    .send(
                        *remote_instance,
                        OnboardMessage::ReleaseBlocks {
                            requester: self.instance_id,
                            session_id: self.session_id,
                            release_hashes,
                        },
                    )
                    .await?;
            }
        }

        // Filter G4 state - release allocated blocks and remove from tracking for hashes beyond first hole
        let g4_release_hashes: Vec<SequenceHash> = self
            .g4_state
            .won_hashes
            .iter()
            .filter(|h| !keep_set.contains(h))
            .copied()
            .collect();

        if !g4_release_hashes.is_empty() {
            tracing::debug!(
                session_id = %self.session_id,
                count = g4_release_hashes.len(),
                "Releasing G4 blocks beyond first hole"
            );

            for hash in &g4_release_hashes {
                // Remove from won_hashes
                self.g4_state.won_hashes.remove(hash);
                // Remove from pending_load (if still loading)
                self.g4_state.pending_load.remove(hash);
                // Remove allocated block (will be deallocated when dropped)
                self.g4_state.allocated_blocks.remove(hash);
            }
        }

        Ok(())
    }

    /// Hold mode: Just hold blocks without staging.
    async fn hold_mode(&mut self) -> Result<()> {
        let local_g2 = self.local_g2_blocks.count();
        let local_g3 = self.local_g3_blocks.count();
        let remote_g2: usize = self.remote_g2_blocks.values().map(|v| v.len()).sum();
        let remote_g3: usize = self.remote_g3_blocks.values().map(|v| v.len()).sum();

        // G4 state
        let pending_g4 = self.g4_state.pending_load.len();
        let loaded_g4 = self.g4_state.won_hashes.len();
        let failed_g4 = self.g4_state.failed_hashes.len();

        tracing::debug!(
            session_id = %self.session_id,
            local_g2,
            local_g3,
            remote_g2,
            remote_g3,
            pending_g4,
            loaded_g4,
            failed_g4,
            "hold_mode"
        );

        self.status_tx
            .send(OnboardingStatus::Holding {
                local_g2,
                local_g3,
                remote_g2,
                remote_g3,
                pending_g4,
                loaded_g4,
                failed_g4,
            })
            .ok();

        tracing::debug!(session_id = %self.session_id, "Sent Holding status");

        Ok(())
    }

    /// Prepare mode: Stage all G3→G2 but keep session alive.
    async fn prepare_mode(&mut self) -> Result<()> {
        // Stage local G3→G2
        self.stage_local_g3_to_g2().await?;

        // Send StageBlocks to remotes for their G3 sequence hashes
        for (remote, sequence_hashes) in &self.remote_g3_blocks {
            self.transport
                .send(
                    *remote,
                    OnboardMessage::StageBlocks {
                        requester: self.instance_id,
                        session_id: self.session_id,
                        stage_hashes: sequence_hashes.clone(),
                    },
                )
                .await?;
        }

        // Wait for BlocksReady from all remotes
        // (simplified - would need proper tracking in production)

        let local_g2 = self.local_g2_blocks.count();
        let remote_g2: usize = self
            .remote_g2_blocks
            .values()
            .map(|v| v.len())
            .sum::<usize>()
            + self
                .remote_g3_blocks
                .values()
                .map(|v| v.len())
                .sum::<usize>(); // G3 now staged to G2

        self.status_tx
            .send(OnboardingStatus::Prepared {
                local_g2,
                remote_g2,
            })
            .ok();

        Ok(())
    }

    /// Full mode: Stage G3→G2 + pull remote G2→local G2.
    async fn full_mode(&mut self) -> Result<()> {
        // Stage local G3→G2
        self.stage_local_g3_to_g2().await?;

        // Send StageBlocks to remotes for their G3 sequence hashes
        for (remote, sequence_hashes) in &self.remote_g3_blocks {
            self.transport
                .send(
                    *remote,
                    OnboardMessage::StageBlocks {
                        requester: self.instance_id,
                        session_id: self.session_id,
                        stage_hashes: sequence_hashes.clone(),
                    },
                )
                .await?;
        }

        // Pull remote G2→local G2 via RDMA (both original G2 and newly staged from G3)
        self.pull_remote_blocks().await?;

        // Consolidate all blocks
        self.consolidate_blocks().await;

        // Send CloseSession to all remotes
        let all_remotes: HashSet<InstanceId> = self
            .remote_g2_blocks
            .keys()
            .chain(self.remote_g3_blocks.keys())
            .copied()
            .collect();

        for remote in all_remotes {
            self.transport
                .send(
                    remote,
                    OnboardMessage::CloseSession {
                        requester: self.instance_id,
                        session_id: self.session_id,
                    },
                )
                .await?;
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

        let src_ids: Vec<BlockId> = self
            .local_g3_blocks
            .blocks()
            .iter()
            .map(|b| b.block_id())
            .collect();

        // Allocate G2 blocks
        let dst_blocks = self
            .g2_manager
            .allocate_blocks(src_ids.len())
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate G2 blocks"))?;

        let dst_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

        // Execute transfer
        let notification = parallel_worker.execute_local_transfer(
            LogicalLayoutHandle::G3,
            LogicalLayoutHandle::G2,
            Arc::from(src_ids),
            Arc::from(dst_ids.clone()),
            TransferOptions::default(),
        )?;

        notification.await?;

        // Register new G2 blocks with G3 metadata
        let new_g2_blocks: Vec<ImmutableBlock<G2>> = dst_blocks
            .into_iter()
            .zip(self.local_g3_blocks.blocks().iter())
            .map(|(dst, src)| {
                self.g2_manager
                    .register_mutable_block_from_existing(dst, src)
            })
            .collect();

        // Clear G3 blocks (take_all releases them) and add new G2 blocks
        let _ = self.local_g3_blocks.take_all();
        self.local_g2_blocks.extend(new_g2_blocks);

        Ok(())
    }

    /// Pull remote G2→local G2 via RDMA.
    ///
    /// This method:
    /// 1. Imports remote metadata for each instance (if not already imported)
    /// 2. Allocates local G2 blocks as destinations
    /// 3. Executes RDMA transfer via worker
    /// 4. Registers pulled blocks with their sequence hashes
    async fn pull_remote_blocks(&mut self) -> Result<()> {
        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ParallelWorker required for RDMA pull"))?;

        // Process each remote instance that has G2 blocks to pull
        for (remote_instance, block_ids) in self.remote_g2_blocks.clone() {
            // Skip if no blocks to pull
            if block_ids.is_empty() {
                continue;
            }

            // Get the parallel sequence hashes for registration
            let seq_hashes = self
                .remote_g2_hashes
                .get(&remote_instance)
                .cloned()
                .unwrap_or_default();
            if seq_hashes.len() != block_ids.len() {
                anyhow::bail!(
                    "Mismatch between block_ids ({}) and seq_hashes ({}) for instance {}",
                    block_ids.len(),
                    seq_hashes.len(),
                    remote_instance
                );
            }

            // Sort (block_id, seq_hash) pairs by position to ensure correct transfer order
            // This is a safety net in case responder sent blocks in wrong order
            let mut pairs: Vec<(BlockId, SequenceHash)> =
                block_ids.into_iter().zip(seq_hashes.into_iter()).collect();
            pairs.sort_by_key(|(_, hash)| hash.position());

            let block_ids: Vec<BlockId> = pairs.iter().map(|(id, _)| *id).collect();
            let seq_hashes: Vec<SequenceHash> = pairs.iter().map(|(_, hash)| *hash).collect();

            // Validate positions are contiguous (X, X+1, X+2, ...)
            // This catches any ordering bugs before data corruption occurs
            debug_assert!(validate_contiguous_positions(&seq_hashes).is_ok());

            // Step 1: Import remote metadata if not already done
            if !parallel_worker.has_remote_metadata(remote_instance) {
                tracing::debug!(
                    session_id = %self.session_id,
                    instance = %remote_instance,
                    "Requesting metadata from instance"
                );
                let metadata = self.transport.request_metadata(remote_instance).await?;
                parallel_worker
                    .connect_remote(remote_instance, metadata)?
                    .await?;
                tracing::debug!(
                    session_id = %self.session_id,
                    instance = %remote_instance,
                    "Metadata imported for instance"
                );
            }

            // Step 2: Allocate local G2 blocks as destinations
            let dst_blocks = self
                .g2_manager
                .allocate_blocks(block_ids.len())
                .ok_or_else(|| {
                    anyhow::anyhow!("Failed to allocate {} G2 blocks", block_ids.len())
                })?;
            let dst_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

            tracing::debug!(
                session_id = %self.session_id,
                count = block_ids.len(),
                instance = %remote_instance,
                "Pulling blocks via RDMA"
            );

            // Step 3: Execute RDMA transfer
            // Uses execute_remote_onboard_for_instance which looks up the stored handle mapping
            let notification = parallel_worker.execute_remote_onboard_for_instance(
                remote_instance,
                LogicalLayoutHandle::G2, // source is remote G2
                block_ids,
                LogicalLayoutHandle::G2, // destination is local G2
                Arc::from(dst_ids),
                TransferOptions::default(),
            )?;
            notification.await?;

            tracing::debug!(
                session_id = %self.session_id,
                instance = %remote_instance,
                "RDMA transfer complete"
            );

            // Step 4: Register pulled blocks with their sequence hashes
            // Note: We use register_mutable_block_with_hash to set the sequence hash
            // since we don't have the original block reference from the remote.
            let new_g2_blocks: Vec<ImmutableBlock<G2>> = dst_blocks
                .into_iter()
                .zip(seq_hashes.iter())
                .map(|(dst, seq_hash)| {
                    self.g2_manager
                        .register_mutable_block_with_hash(dst, *seq_hash)
                })
                .collect();

            // Add to local G2 blocks
            self.local_g2_blocks.extend(new_g2_blocks);
        }

        Ok(())
    }

    /// Consolidate all G2 blocks into shared storage.
    async fn consolidate_blocks(&mut self) {
        let all_blocks = self.local_g2_blocks.take_all();
        let matched_blocks = all_blocks.len();

        *self.all_g2_blocks.lock().await = Some(all_blocks);

        self.status_tx
            .send(OnboardingStatus::Complete { matched_blocks })
            .ok();
    }

    /// Wait for control commands (Hold/Prepare modes).
    async fn await_commands(&mut self, mut rx: mpsc::Receiver<OnboardMessage>) -> Result<()> {
        loop {
            tokio::select! {
                Some(cmd) = self.control_rx.recv() => {
                    match cmd {
                        SessionControl::Prepare => {
                            if self.mode == StagingMode::Hold {
                                self.prepare_mode().await?;
                                self.mode = StagingMode::Prepare;
                            }
                        }
                        SessionControl::Pull => {
                            if self.mode == StagingMode::Prepare {
                                self.pull_remote_blocks().await?;
                                self.consolidate_blocks().await;

                                // Send CloseSession to all remotes
                                let all_remotes: HashSet<InstanceId> = self
                                    .remote_g2_blocks
                                    .keys()
                                    .chain(self.remote_g3_blocks.keys())
                                    .copied()
                                    .collect();

                                for remote in all_remotes {
                                    self.transport.send(remote, OnboardMessage::CloseSession {
                                        requester: self.instance_id,
                                        session_id: self.session_id,
                                    }).await?;
                                }

                                break;
                            }
                        }
                        SessionControl::Cancel => {
                            // Release all blocks and exit
                            let all_remotes: HashSet<InstanceId> = self
                                .remote_g2_blocks
                                .keys()
                                .chain(self.remote_g3_blocks.keys())
                                .copied()
                                .collect();

                            for remote in all_remotes {
                                self.transport.send(remote, OnboardMessage::CloseSession {
                                    requester: self.instance_id,
                                    session_id: self.session_id,
                                }).await?;
                            }
                            break;
                        }
                        SessionControl::Shutdown => {
                            break;
                        }
                    }
                }
                // Also drain any remaining messages from responders
                Some(_msg) = rx.recv() => {
                    // Process any late messages if needed
                }
            }
        }

        Ok(())
    }

    // =========================================================================
    // G4/Object Storage Methods
    // =========================================================================

    /// Spawn a G4 search task that runs in parallel with remote G2/G3 search.
    ///
    /// This task calls `has_blocks` via parallel_worker which fans out to workers.
    /// Workers use rank-prefixed keys, so we must query through them (not directly to S3).
    fn spawn_g4_search(
        &self,
        sequence_hashes: Vec<SequenceHash>,
        tx: mpsc::Sender<OnboardMessage>,
    ) -> JoinHandle<()> {
        let session_id = self.session_id;
        // Use parallel_worker for has_blocks - it fans out to workers who use rank-prefixed keys
        let parallel_worker = self.parallel_worker.clone();

        tokio::spawn(async move {
            let Some(worker) = parallel_worker else {
                // No parallel worker configured, send empty results
                let _ = tx
                    .send(OnboardMessage::G4Results {
                        session_id,
                        found_hashes: vec![],
                    })
                    .await;
                return;
            };

            // Call has_blocks via parallel_worker (fans out to workers with rank-prefixed keys)
            let results = worker.has_blocks(sequence_hashes).await;

            // Filter to only blocks that exist (Some(size))
            let found_hashes: Vec<(SequenceHash, usize)> = results
                .into_iter()
                .filter_map(|(hash, size_opt)| size_opt.map(|size| (hash, size)))
                .collect();

            tracing::debug!(
                session_id = %session_id,
                count = found_hashes.len(),
                "G4 search: found blocks in object storage"
            );

            // Send results back to initiator
            let _ = tx
                .send(OnboardMessage::G4Results {
                    session_id,
                    found_hashes,
                })
                .await;
        })
    }

    /// Process G4 search results with first-responder-wins logic.
    ///
    /// Returns the hashes that G4 won (not already claimed by G2/G3/remote).
    fn process_g4_results(
        &mut self,
        found_hashes: Vec<(SequenceHash, usize)>,
        matched_hashes: &mut HashSet<SequenceHash>,
    ) -> Vec<SequenceHash> {
        let mut won_hashes = Vec::new();

        for (hash, _size) in found_hashes {
            // First-responder-wins: only claim if not already matched
            if matched_hashes.insert(hash) {
                won_hashes.push(hash);
                self.g4_state.won_hashes.insert(hash);
            }
        }

        tracing::debug!(
            session_id = %self.session_id,
            won_count = won_hashes.len(),
            "G4 won hashes (first-responder-wins)"
        );

        won_hashes
    }

    /// Load G4 blocks into local G2 via workers.
    ///
    /// Allocates G2 destination blocks and coordinates workers to download
    /// from object storage via `get_blocks`. After successful download, blocks
    /// are registered with the G2 manager and returned via G4LoadComplete message.
    async fn load_g4_blocks(
        &mut self,
        won_hashes: Vec<SequenceHash>,
        g4_tx: mpsc::Sender<OnboardMessage>,
    ) -> Result<()> {
        if won_hashes.is_empty() {
            return Ok(());
        }

        let parallel_worker = self
            .parallel_worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ParallelWorker required for G4 load"))?;

        // Mark hashes as pending load
        for hash in &won_hashes {
            self.g4_state.pending_load.insert(*hash);
        }

        // Allocate G2 destination blocks
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

        // Track allocated blocks (for cleanup on failure)
        for (hash, block_id) in won_hashes.iter().zip(dst_ids.iter()) {
            self.g4_state.allocated_blocks.insert(*hash, *block_id);
        }

        tracing::debug!(
            session_id = %self.session_id,
            count = won_hashes.len(),
            "Loading G4 blocks via workers"
        );

        // Clone values for the spawned task
        let session_id = self.session_id;
        let hashes = won_hashes.clone();
        let parallel_worker = parallel_worker.clone();
        let g2_manager = self.g2_manager.clone();

        // Spawn load task so we can continue processing other messages
        // IMPORTANT: dst_blocks is moved into the task to keep them alive during download
        tokio::spawn(async move {
            // Execute get_blocks via parallel worker
            let results = parallel_worker
                .get_blocks(hashes.clone(), LogicalLayoutHandle::G2, dst_ids.clone())
                .await;

            // Separate successes and failures, register successful blocks
            let mut success = Vec::new();
            let mut failures = Vec::new();
            let mut blocks = Vec::new();

            // Iterate over results alongside the dst_blocks and hashes
            for ((result, dst_block), seq_hash) in results
                .into_iter()
                .zip(dst_blocks.into_iter())
                .zip(hashes.iter())
            {
                match result {
                    Ok(hash) => {
                        // Register the block with its sequence hash
                        // This adds it to the BlockRegistry for presence filtering
                        let immutable = g2_manager.register_mutable_block_with_hash(dst_block, *seq_hash);
                        blocks.push(immutable);
                        success.push(hash);
                    }
                    Err(hash) => {
                        // Block will be returned to pool when dst_block is dropped
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

            // Send completion message with registered blocks
            let _ = g4_tx
                .send(OnboardMessage::G4LoadComplete {
                    session_id,
                    success,
                    failures,
                    blocks: std::sync::Arc::new(blocks),
                })
                .await;
        });

        Ok(())
    }

    /// Handle G4 load completion, updating state and adding blocks to local_g2_blocks.
    ///
    /// The blocks have already been registered with the G2 manager in the spawned task,
    /// so they are now visible in the BlockRegistry for presence filtering.
    fn handle_g4_load_complete(
        &mut self,
        success: Vec<SequenceHash>,
        failures: Vec<(SequenceHash, String)>,
        blocks: Arc<Vec<ImmutableBlock<G2>>>,
    ) {
        // Process successful loads - update state tracking
        for hash in &success {
            self.g4_state.pending_load.remove(hash);
            // Remove from allocated_blocks since we now have registered ImmutableBlocks
            self.g4_state.allocated_blocks.remove(hash);
        }

        // Unwrap the Arc to get the Vec (this is the only owner since the message was just received)
        let blocks = Arc::try_unwrap(blocks).expect("G4LoadComplete should be the sole owner of blocks");

        // Add the registered G4 blocks to local_g2_blocks
        // These blocks are now registered in the BlockRegistry and will be
        // detected by the PresenceFilter during G1→G2 offloading
        self.local_g2_blocks.extend(blocks);

        // Process failures
        for (hash, error) in failures {
            self.g4_state.pending_load.remove(&hash);
            self.g4_state.failed_hashes.insert(hash, error);

            // Remove from allocated_blocks on failure (block was already dropped)
            self.g4_state.allocated_blocks.remove(&hash);

            // Also remove from won_hashes since it failed to load
            self.g4_state.won_hashes.remove(&hash);
        }

        tracing::debug!(
            session_id = %self.session_id,
            won = self.g4_state.won_hashes.len(),
            pending = self.g4_state.pending_load.len(),
            failed = self.g4_state.failed_hashes.len(),
            local_g2 = self.local_g2_blocks.count(),
            "G4 load complete, blocks added to local_g2_blocks"
        );
    }
}
