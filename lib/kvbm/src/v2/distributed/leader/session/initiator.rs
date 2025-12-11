// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use tokio::sync::{Mutex, mpsc, watch};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{
    logical::{blocks::ImmutableBlock, manager::BlockManager},
    physical::transfer::TransferOptions,
    v2::{
        BlockId, G2, G3, InstanceId, SequenceHash, distributed::parallelism::ParallelWorker,
        logical::LogicalLayoutHandle, physical::manager::LayoutHandle,
    },
};

use super::{
    super::{OnboardingStatus, SessionControl, StagingMode},
    BlockHolder, SessionId,
    messages::OnboardMessage,
    transport::MessageTransport,
};

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
        }
    }

    /// Run the initiator session task.
    pub async fn run(
        mut self,
        mut rx: mpsc::Receiver<OnboardMessage>,
        remote_leaders: Vec<InstanceId>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<()> {
        eprintln!(
            "[INITIATOR {}] Starting in mode {:?} for {} hashes, {} remotes",
            self.session_id,
            self.mode,
            sequence_hashes.len(),
            remote_leaders.len()
        );

        // Phase 1: Search (local G2 and G3, then remote if needed)
        self.search_phase(&mut rx, &remote_leaders, &sequence_hashes)
            .await?;

        // Phase 1.5: Apply find policy (first-hole detection)
        // Trims results to first contiguous sequence from start
        self.apply_find_policy(&sequence_hashes).await?;

        eprintln!(
            "[INITIATOR {}] search_phase complete, entering mode handler",
            self.session_id
        );

        // Phase 2: Staging based on mode
        match self.mode {
            StagingMode::Hold => {
                eprintln!("[INITIATOR {}] Calling hold_mode()", self.session_id);
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

        // Check if remote search needed
        if matched_hashes.len() == sequence_hashes.len() || remote_leaders.is_empty() {
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

        // Send CreateSession to all remotes
        for remote in remote_leaders {
            let msg = OnboardMessage::CreateSession {
                requester: self.instance_id,
                session_id: self.session_id,
                sequence_hashes: remaining_hashes.clone(),
            };
            self.transport.send(*remote, msg).await?;
        }

        // Process search responses
        self.process_search_responses(rx, remote_leaders, &mut matched_hashes)
            .await?;

        Ok(())
    }

    /// Process G2Results and G3Results from responders.
    async fn process_search_responses(
        &mut self,
        rx: &mut mpsc::Receiver<OnboardMessage>,
        remote_leaders: &[InstanceId],
        matched_hashes: &mut HashSet<SequenceHash>,
    ) -> Result<()> {
        let mut pending_g2_responses = remote_leaders.len();
        let mut pending_g3_responses: HashSet<InstanceId> =
            remote_leaders.iter().copied().collect();
        let mut pending_search_complete: HashSet<InstanceId> =
            remote_leaders.iter().copied().collect();
        let mut pending_acknowledgments: HashSet<InstanceId> = HashSet::new();

        while let Some(msg) = rx.recv().await {
            eprintln!(
                "[INITIATOR {}] process_search_responses received: {:?}",
                self.session_id,
                std::mem::discriminant(&msg)
            );

            match msg {
                OnboardMessage::G2Results {
                    responder,
                    layout_handle,
                    sequence_hashes,
                    block_ids,
                    ..
                } => {
                    eprintln!(
                        "[INITIATOR {}] Processing G2Results from {} with {} hashes",
                        self.session_id,
                        responder,
                        sequence_hashes.len()
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

                    eprintln!(
                        "[INITIATOR {}] SearchComplete from {}: g2_pending={}, g3_pending={}, ack_pending={}, search_pending={}",
                        self.session_id,
                        responder,
                        pending_g2_responses,
                        pending_g3_responses.len(),
                        pending_acknowledgments.len(),
                        pending_search_complete.len()
                    );

                    // Check if search is complete
                    if pending_g2_responses == 0
                        && pending_g3_responses.is_empty()
                        && pending_acknowledgments.is_empty()
                        && pending_search_complete.is_empty()
                    {
                        eprintln!(
                            "[INITIATOR {}] All responses received, exiting search_phase",
                            self.session_id
                        );
                        break;
                    }
                }
                OnboardMessage::Acknowledged { responder, .. } => {
                    pending_acknowledgments.remove(&responder);

                    // Check if search is complete
                    if pending_g2_responses == 0
                        && pending_g3_responses.is_empty()
                        && pending_acknowledgments.is_empty()
                        && pending_search_complete.is_empty()
                    {
                        eprintln!(
                            "[INITIATOR {}] All responses received, exiting search_phase",
                            self.session_id
                        );
                        break;
                    }
                }
                _ => {}
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
            eprintln!(
                "[INITIATOR {}] apply_find_policy: no trimming needed ({} matched, {} total)",
                self.session_id,
                keep_count,
                sequence_hashes.len()
            );
            return Ok(());
        }

        // Get the hashes to keep
        let keep_hashes: Vec<SequenceHash> = sequence_hashes[..keep_count].to_vec();
        let keep_set: HashSet<&SequenceHash> = keep_hashes.iter().collect();

        eprintln!(
            "[INITIATOR {}] apply_find_policy: trimming from {} to {} blocks (first hole at position {})",
            self.session_id,
            matched_hashes.len(),
            keep_count,
            keep_count
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
                    eprintln!(
                        "[INITIATOR {}] Releasing {} G2 blocks from instance {} (beyond first hole)",
                        self.session_id,
                        release_hashes.len(),
                        remote_instance
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
                eprintln!(
                    "[INITIATOR {}] Releasing {} G3 blocks from instance {} (beyond first hole)",
                    self.session_id,
                    release_hashes.len(),
                    remote_instance
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

        Ok(())
    }

    /// Hold mode: Just hold blocks without staging.
    async fn hold_mode(&mut self) -> Result<()> {
        let local_g2 = self.local_g2_blocks.count();
        let local_g3 = self.local_g3_blocks.count();
        let remote_g2: usize = self.remote_g2_blocks.values().map(|v| v.len()).sum();
        let remote_g3: usize = self.remote_g3_blocks.values().map(|v| v.len()).sum();

        eprintln!(
            "[INITIATOR {}] hold_mode: local_g2={}, local_g3={}, remote_g2={}, remote_g3={}",
            self.session_id, local_g2, local_g3, remote_g2, remote_g3
        );

        self.status_tx
            .send(OnboardingStatus::Holding {
                local_g2,
                local_g3,
                remote_g2,
                remote_g3,
            })
            .ok();

        eprintln!("[INITIATOR {}] Sent Holding status", self.session_id);

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

            // Step 1: Import remote metadata if not already done
            if !parallel_worker.has_remote_metadata(remote_instance) {
                eprintln!(
                    "[INITIATOR {}] Requesting metadata from instance {}",
                    self.session_id, remote_instance
                );
                let metadata = self.transport.request_metadata(remote_instance).await?;
                parallel_worker
                    .connect_remote(remote_instance, metadata)?
                    .await?;
                eprintln!(
                    "[INITIATOR {}] Metadata imported for instance {}",
                    self.session_id, remote_instance
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

            eprintln!(
                "[INITIATOR {}] Pulling {} blocks from instance {} via RDMA",
                self.session_id,
                block_ids.len(),
                remote_instance
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

            eprintln!(
                "[INITIATOR {}] RDMA transfer complete from instance {}",
                self.session_id, remote_instance
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
}
