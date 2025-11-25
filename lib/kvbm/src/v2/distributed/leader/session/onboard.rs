// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;

use std::sync::Arc;

use crate::{
    logical::{BlockId, manager::BlockManager},
    v2::{
        InstanceId, SequenceHash,
        integrations::{G2, G3},
        logical::blocks::ImmutableBlock,
    },
};

pub type SessionId = uuid::Uuid;
pub type OnboardSessionTx = Sender<OnboardMessage>;

pub struct KvbmSystem {}

impl KvbmSystem {}

pub struct OnboardingSession {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteMatches {
    instance_id: InstanceId,
    session_id: SessionId,
    sequence_hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLocalMatchesRequest {
    pub requester: InstanceId,
    pub session_id: SessionId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotelyStagedBlocks {
    pub instance_id: InstanceId,
    pub session_id: SessionId,
    /// The locale (leader/worker) where these blocks live in G2.
    pub g2_locale: InstanceId,
    pub block_ids: Vec<BlockId>,
    /// false => expect at least one more message for the same session_id/instance_id.
    pub finished: bool,
}

/// Active messages exchanged between leaders for onboarding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnboardMessage {
    /// Initialize a session between two leaders.
    Init {
        requester: InstanceId,
        session_id: SessionId,
    },
    /// Ack a session init.
    Ack {
        responder: InstanceId,
        session_id: SessionId,
    },
    /// Ask a peer leader to search locally (G2/G3) and optionally pin and/or stage.
    FindLocalMatches {
        requester: InstanceId,
        session_id: SessionId,
        sequence_hashes: Vec<SequenceHash>,
        options: FindOptions,
    },
    /// Return the set of matches from a local search.
    LocalMatches(RemoteMatches),
    /// Instruct a peer to stage its local matches into G2.
    StageLocalMatches(StageLocalMatchesRequest),
    /// Notify that a batch of blocks is ready (can stream multiple with finished=false).
    Ready(RemotelyStagedBlocks),
    /// Release pinned blocks for a session once the initiator is done pulling them.
    DropPinnedBlocks {
        requester: InstanceId,
        session_id: SessionId,
        block_ids: Vec<BlockId>,
    },
    /// Ack for DropPinnedBlocks when cleanup is done.
    DropPinnedBlocksAck {
        responder: InstanceId,
        session_id: SessionId,
        released: Vec<BlockId>,
    },
}

fn session_id_from_message(msg: &OnboardMessage) -> SessionId {
    match msg {
        OnboardMessage::Init { session_id, .. }
        | OnboardMessage::Ack { session_id, .. }
        | OnboardMessage::FindLocalMatches { session_id, .. }
        | OnboardMessage::LocalMatches(RemoteMatches { session_id, .. })
        | OnboardMessage::StageLocalMatches(StageLocalMatchesRequest { session_id, .. })
        | OnboardMessage::Ready(RemotelyStagedBlocks { session_id, .. })
        | OnboardMessage::DropPinnedBlocks { session_id, .. }
        | OnboardMessage::DropPinnedBlocksAck { session_id, .. } => session_id.clone(),
    }
}

/// Dispatch an inbound active message to the per-session task via its channel.
/// Each session's channel serializes message handling for that session.
pub async fn dispatch_onboard_message(
    sessions: &DashMap<SessionId, OnboardSessionTx>,
    message: OnboardMessage,
) -> Result<()> {
    let session_id = session_id_from_message(&message);
    if let Some(entry) = sessions.get(&session_id) {
        entry
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("failed to send to session {session_id}: {e}"))?;
        return Ok(());
    }

    anyhow::bail!("no session task registered for session {session_id}");
}

pub struct DistributedKvbmSystem {}

impl DistributedKvbmSystem {
    pub async fn find_matches(
        &mut self,
        sequence_hashes: Arc<Vec<SequenceHash>>,
    ) -> Result<RemoteMatches> {
        unimplemented!()
    }
}
pub struct G4System {}

pub enum FindMatchesStatus {
    Searching,

    /// Staging blocks to G2
    /// - `matched`: number of blocks matched during the search
    /// - `staging`: number of blocks being staged to G2
    Staging {
        matched: usize,
        staging: usize,
    },

    /// Complete
    /// - `matched`: number of blocks matched during the search
    ///
    /// There is no ongoing activity for this find operation.
    Complete {
        matched: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindOptions {
    /// If true, the immutable blocks will be acquired and held in either the local or remote sessions
    /// until the session is complete or the match set is unpinned.
    pin: bool,
    stage: bool,
    use_hub: bool,
    remote_instances: Vec<InstanceId>,
}

pub enum BlockHandle {
    G1,
    G2,
    G3,
    G4,
}

pub struct Leader {}

impl Leader {
    pub async fn execute_transfer(
        &self,
        src: BlockHandle,
        dst: BlockHandle,
        src_block_ids: Vec<BlockId>,
        dst_block_ids: Vec<BlockId>,
    ) -> Result<()> {
        todo!("implement execute_transfer - drive 1 or more workers over the nova comms")
    }
}

pub struct LocalMatches {
    g2_blocks: Vec<ImmutableBlock<G2>>,
    g3_blocks: Vec<ImmutableBlock<G3>>,
}

pub struct OnboardSessionInner {
    instance_id: InstanceId,
    session_id: SessionId,
    leader: Leader,
    g2_manager: BlockManager<G2>,
    g3_manager: BlockManager<G3>,
    local_matches: LocalMatches,
    remote_matches: RemoteMatches,
}

impl OnboardSessionInner {
    pub async fn find_matches(
        &mut self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<FindMatchesStatus> {
        // search g2 for matches, acquire immutable blocks
        let num_local_matches = self.find_local_matches(sequence_hashes)?;

        // we capture a shared ptr to the remaining sequence hashes to avoid cloning the vector
        // we will pass these off to a number of tasks to search for remote matches in parallel
        let remaining_sequence_hashes = Arc::new(sequence_hashes[num_local_matches..].to_vec());

        // search g3 for remaining matches, if stage is true, acquire mutable g2 blocks and being staging
        // otherwise, wait add task to the onboarding operation object

        // search remote instances for remaining matches; this happens in parallel after the g3 matches are found
        // if use_hub is true, first narrow down the search to the suggested instances in the hub,
        // otherwise, search all the remote_instances -- if empty, search all known remote instances
        // remote matches should feed a mpsc channel which will record the first successful matches and either being
        // staging them or adding them to the onboarding operation object
        unimplemented!()
    }

    /// Handle a single inbound active message and return zero or more outbound messages to send.
    pub async fn handle_message(&mut self, message: OnboardMessage) -> Result<Vec<OnboardMessage>> {
        match message {
            OnboardMessage::Init {
                requester,
                session_id,
            } => {
                let _ = requester;
                // Accept session; record session_id if desired.
                Ok(vec![OnboardMessage::Ack {
                    responder: self.instance_id,
                    session_id,
                }])
            }
            OnboardMessage::Ack {
                responder: _,
                session_id: _,
            } => {
                // No follow-up; establish session locally.
                Ok(vec![])
            }
            OnboardMessage::FindLocalMatches {
                session_id,
                sequence_hashes,
                options,
                requester: _,
            } => {
                // Drive local search (G2/G3) and optionally pin/stage per options.
                let _ = self.find_matches(&sequence_hashes).await?;

                Ok(vec![OnboardMessage::LocalMatches(RemoteMatches {
                    instance_id: self.instance_id,
                    session_id,
                    sequence_hashes,
                })])
            }
            OnboardMessage::StageLocalMatches(req) => {
                let staged = self.stage_local_matches_messages().await?;

                let messages = staged
                    .into_iter()
                    .map(OnboardMessage::Ready)
                    .collect::<Vec<_>>();

                if messages.is_empty() {
                    anyhow::bail!("no staged blocks available")
                }

                let _ = req;
                Ok(messages)
            }
            OnboardMessage::Ready(_) => {
                // Ready should be consumed by the initiator, not handled here.
                Ok(vec![])
            }
            OnboardMessage::DropPinnedBlocks {
                requester,
                session_id,
                block_ids,
            } => {
                // Drop pinned blocks for this session; TODO: actually interact with pinset.
                let _ = requester;
                let _ = block_ids;
                Ok(vec![OnboardMessage::DropPinnedBlocksAck {
                    responder: self.instance_id,
                    session_id,
                    released: Vec::new(),
                }])
            }
            OnboardMessage::DropPinnedBlocksAck { .. } => Ok(vec![]),
            OnboardMessage::LocalMatches(_) => {
                // LocalMatches should be consumed by the initiator, not handled here.
                Ok(vec![])
            }
        }
    }

    /// Find local matches for the given sequence hashes.
    ///
    /// Returns the number of matches found.
    ///
    /// If the function returns a non-zero result, then the [`OnboardingSession`] will acquire the immutable blocks
    /// and hold a reference to them until the session is complete or the match set is unpinned.
    fn find_local_matches(&mut self, sequence_hashes: &[SequenceHash]) -> Result<usize> {
        unimplemented!()
    }

    async fn stage_local_matches_messages(&mut self) -> Result<Vec<RemotelyStagedBlocks>> {
        let mut messages = Vec::new();

        if !self.local_matches.g2_blocks.is_empty() {
            let block_ids = self
                .local_matches
                .g2_blocks
                .iter()
                .map(|b| b.block_id())
                .collect::<Vec<_>>();

            messages.push(RemotelyStagedBlocks {
                instance_id: self.instance_id,
                session_id: self.session_id,
                g2_locale: self.instance_id,
                block_ids,
                finished: self.local_matches.g3_blocks.is_empty(),
            });
        }

        if self.local_matches.g3_blocks.is_empty() {
            return Ok(messages);
        }

        let src_block_ids = self
            .local_matches
            .g3_blocks
            .iter()
            .map(|b| b.block_id())
            .collect::<Vec<_>>();

        let dst_blocks = self
            .g2_manager
            .allocate_blocks(src_block_ids.len())
            .ok_or(anyhow::anyhow!("failed to allocate g2 blocks"))?;

        if dst_blocks.len() != src_block_ids.len() {
            anyhow::bail!("failed to allocate g2 blocks");
        }

        let dst_block_ids = dst_blocks.iter().map(|b| b.block_id()).collect::<Vec<_>>();

        self.leader
            .execute_transfer(
                BlockHandle::G3,
                BlockHandle::G2,
                src_block_ids,
                dst_block_ids,
            )
            .await?;

        let g3_blocks = std::mem::take(&mut self.local_matches.g3_blocks);

        // register new g2 blocks and drop g3 blocks
        let new_g2_blocks = dst_blocks
            .into_iter()
            .zip(g3_blocks.into_iter())
            .map(|(dst, src)| {
                self.g2_manager
                    .register_mutable_block_from_existing(dst, &src)
            })
            .collect::<Vec<_>>();

        self.local_matches.g2_blocks.extend(new_g2_blocks);
        self.local_matches.g3_blocks.clear();

        let new_block_ids = self
            .local_matches
            .g2_blocks
            .iter()
            .map(|b| b.block_id())
            .collect::<Vec<_>>();

        messages.push(RemotelyStagedBlocks {
            instance_id: self.instance_id,
            session_id: self.session_id,
            g2_locale: self.instance_id,
            block_ids: new_block_ids,
            finished: true,
        });

        Ok(messages)
    }

    fn find_remote_matches(&mut self, sequence_hashes: &[SequenceHash]) -> Result<usize> {
        unimplemented!()
    }
}
