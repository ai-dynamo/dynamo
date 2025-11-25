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
        BlockId, InstanceId, SequenceHash,
        distributed::worker::Worker,
        integrations::{G2, G3},
        logical::LogicalLayoutHandle,
        physical::manager::LayoutHandle,
    },
};

use super::{
    super::{OnboardingStatus, SessionControl, StagingMode},
    SessionId,
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
    worker: Option<Arc<dyn Worker>>,
    transport: Arc<MessageTransport>,
    status_tx: watch::Sender<OnboardingStatus>,

    // Held blocks from local search
    local_g2_blocks: Vec<ImmutableBlock<G2>>,
    local_g3_blocks: Vec<ImmutableBlock<G3>>,

    // Track remote blocks by tier
    remote_g2_blocks: HashMap<InstanceId, Vec<BlockId>>, // G2: track block IDs
    remote_g3_blocks: HashMap<InstanceId, Vec<SequenceHash>>, // G3: track sequence hashes
    remote_g2_layouts: HashMap<InstanceId, LayoutHandle>, // G2 layouts for RDMA

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
        worker: Option<Arc<dyn Worker>>,
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
            worker,
            transport,
            status_tx,
            local_g2_blocks: Vec::new(),
            local_g3_blocks: Vec::new(),
            remote_g2_blocks: HashMap::new(),
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
        self.local_g2_blocks = self.g2_manager.match_blocks(sequence_hashes);

        let mut matched_hashes: HashSet<SequenceHash> = self
            .local_g2_blocks
            .iter()
            .map(|b| b.sequence_hash())
            .collect();

        // Local G3 search
        if let Some(ref g3_manager) = self.g3_manager {
            let remaining: Vec<_> = sequence_hashes
                .iter()
                .filter(|h| !matched_hashes.contains(h))
                .copied()
                .collect();

            if !remaining.is_empty() {
                self.local_g3_blocks = g3_manager.match_blocks(&remaining);
                for block in &self.local_g3_blocks {
                    matched_hashes.insert(block.sequence_hash());
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

    /// Hold mode: Just hold blocks without staging.
    async fn hold_mode(&mut self) -> Result<()> {
        let local_g2 = self.local_g2_blocks.len();
        let local_g3 = self.local_g3_blocks.len();
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

        let local_g2 = self.local_g2_blocks.len();
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

        let worker = self
            .worker
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Worker required for G3→G2 staging"))?;

        let src_ids: Vec<BlockId> = self.local_g3_blocks.iter().map(|b| b.block_id()).collect();

        // Allocate G2 blocks
        let dst_blocks = self
            .g2_manager
            .allocate_blocks(src_ids.len())
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate G2 blocks"))?;

        let dst_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();

        // Execute transfer
        let notification = worker.execute_local_transfer(
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
            .zip(self.local_g3_blocks.iter())
            .map(|(dst, src)| {
                self.g2_manager
                    .register_mutable_block_from_existing(dst, src)
            })
            .collect();

        // Replace G3 blocks with new G2 blocks
        self.local_g3_blocks.clear();
        self.local_g2_blocks.extend(new_g2_blocks);

        Ok(())
    }

    /// Pull remote G2→local G2 via RDMA.
    async fn pull_remote_blocks(&mut self) -> Result<()> {
        // TODO: Implement RDMA get for pulling remote G2 blocks to local G2
        //
        // Implementation steps:
        // 1. For each remote instance with held blocks (self.remote_block_refs):
        //    a. Wait for BlocksReady message containing NIXL metadata
        //    b. Extract block IDs and NIXL layout handle from message
        //
        // 2. For each remote:
        //    a. Allocate local G2 blocks:
        //       let dst_blocks = self.g2_manager.allocate_blocks(count)?;
        //       let dst_ids: Vec<BlockId> = dst_blocks.iter().map(|b| b.block_id()).collect();
        //
        //    b. Create RemoteDescriptor for the remote blocks:
        //       let remote_descriptor = RemoteDescriptor::Layout {
        //           handle: LayoutHandle {
        //               instance_id: remote_instance,
        //               layout_id: remote_layout_id,  // From BlocksReady message
        //           },
        //           block_ids: remote_block_ids,      // From BlocksReady message
        //       };
        //
        //    c. Execute RDMA transfer using Worker:
        //       let notification = worker.execute_remote_onboard(
        //           remote_descriptor,
        //           LogicalLayoutHandle::G2,
        //           dst_ids.clone(),
        //           TransferOptions::default(),
        //       )?;
        //       notification.await?;
        //
        //    d. Register transferred blocks with metadata from remote
        //       (metadata should come from BlocksReady or subsequent metadata exchange)
        //
        //    e. Add new G2 blocks to self.local_g2_blocks
        //
        // 3. Update status to track pulling progress:
        //    self.status_tx.send(OnboardingStatus::Staging {
        //        matched,
        //        staging_local: 0,
        //        staging_remote: 0,
        //        pulling: pending_pulls,
        //    }).ok();

        todo!("Implement RDMA get - needs NIXL metadata from BlocksReady messages")
    }

    /// Consolidate all G2 blocks into shared storage.
    async fn consolidate_blocks(&mut self) {
        let all_blocks = std::mem::take(&mut self.local_g2_blocks);
        let matched = all_blocks.len();

        *self.all_g2_blocks.lock().await = Some(all_blocks);

        self.status_tx
            .send(OnboardingStatus::Complete { matched })
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
