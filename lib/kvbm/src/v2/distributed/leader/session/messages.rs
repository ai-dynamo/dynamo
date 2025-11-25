// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::v2::{BlockId, InstanceId, SequenceHash, physical::manager::LayoutHandle};

use super::SessionId;

/// Messages exchanged between leaders during onboarding sessions.
///
/// Phase 2 protocol (G2-only):
/// 1. Initiator sends CreateSession to multiple responders
/// 2. Each responder searches local G2 and sends G2Results back
/// 3. Initiator applies first-responder-wins and sends HoldBlocks to each
/// 4. Responders send Acknowledged after releasing unwanted blocks
///
/// Phase 3 protocol (G3 staging):
/// 5. Responders search G3 and send G3Results
/// 6. Initiator sends StageBlocks with blocks to stage G3->G2
/// 7. Responders stage blocks and send BlocksReady when complete
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnboardMessage {
    /// Initiator creates a new onboarding session.
    CreateSession {
        requester: InstanceId,
        session_id: SessionId,
        sequence_hashes: Vec<SequenceHash>,
    },

    /// Responder signals local search (G2 and G3) is complete.
    SearchComplete {
        responder: InstanceId,
        session_id: SessionId,
    },

    /// Responder reports G2 search results.
    /// - layout_handle: G2 manager layout for RDMA operations
    /// - sequence_hashes: ordered list of matched sequence hashes
    /// - block_ids: parallel list of block IDs (can be zipped with sequence_hashes)
    G2Results {
        responder: InstanceId,
        session_id: SessionId,
        layout_handle: LayoutHandle,
        sequence_hashes: Vec<SequenceHash>,
        block_ids: Vec<BlockId>,
    },

    /// Responder reports G3 search results.
    /// - sequence_hashes: ordered list of matched sequence hashes (no block IDs)
    G3Results {
        responder: InstanceId,
        session_id: SessionId,
        sequence_hashes: Vec<SequenceHash>,
    },

    /// Initiator tells responder which sequence hashes to hold/drop.
    /// Works across G2 and G3 tiers.
    HoldBlocks {
        requester: InstanceId,
        session_id: SessionId,
        hold_hashes: Vec<SequenceHash>,
        drop_hashes: Vec<SequenceHash>,
    },

    /// Initiator tells responder which G3 sequence hashes to stage to G2.
    /// Any G3 blocks with these hashes should be staged to G2.
    StageBlocks {
        requester: InstanceId,
        session_id: SessionId,
        stage_hashes: Vec<SequenceHash>,
    },

    /// Responder reports newly staged blocks are ready in G2 (after G3->G2 staging).
    /// Only reports blocks that were just staged, not all G2 blocks.
    /// - layout_handle: G2 manager layout for RDMA pull
    /// - sequence_hashes: newly staged blocks
    /// - block_ids: parallel to sequence_hashes
    BlocksReady {
        responder: InstanceId,
        session_id: SessionId,
        layout_handle: LayoutHandle,
        sequence_hashes: Vec<SequenceHash>,
        block_ids: Vec<BlockId>,
    },

    /// Responder acknowledges hold/drop request.
    Acknowledged {
        responder: InstanceId,
        session_id: SessionId,
    },

    /// Initiator tells responder to release specific sequence hashes that weren't selected.
    /// Works across G2 and G3 tiers.
    ReleaseBlocks {
        requester: InstanceId,
        session_id: SessionId,
        release_hashes: Vec<SequenceHash>,
    },

    /// Initiator tells responder session is complete, responder can cleanup.
    CloseSession {
        requester: InstanceId,
        session_id: SessionId,
    },
    // TODO: Add heartbeat/TTL mechanism for handling unresponsive initiators
    // Heartbeat {
    //     requester: InstanceId,
    //     session_id: SessionId,
    //     timestamp: u64,
    // },
    // TTL resets with each heartbeat. If TTL expires:
    // - Responder releases all held blocks
    // - Responder cleans up session state
    // - Session task exits
}

/// Represents a block match found during search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMatch {
    pub sequence_hash: SequenceHash,
    pub block_id: BlockId,
}

impl OnboardMessage {
    /// Extract the session ID from any message variant.
    pub fn session_id(&self) -> SessionId {
        match self {
            OnboardMessage::CreateSession { session_id, .. }
            | OnboardMessage::SearchComplete { session_id, .. }
            | OnboardMessage::G2Results { session_id, .. }
            | OnboardMessage::G3Results { session_id, .. }
            | OnboardMessage::HoldBlocks { session_id, .. }
            | OnboardMessage::StageBlocks { session_id, .. }
            | OnboardMessage::BlocksReady { session_id, .. }
            | OnboardMessage::Acknowledged { session_id, .. }
            | OnboardMessage::ReleaseBlocks { session_id, .. }
            | OnboardMessage::CloseSession { session_id, .. } => *session_id,
        }
    }

    /// Extract the requester/responder instance ID from the message.
    pub fn instance_id(&self) -> InstanceId {
        match self {
            OnboardMessage::CreateSession { requester, .. }
            | OnboardMessage::HoldBlocks { requester, .. }
            | OnboardMessage::StageBlocks { requester, .. }
            | OnboardMessage::ReleaseBlocks { requester, .. }
            | OnboardMessage::CloseSession { requester, .. } => *requester,
            OnboardMessage::SearchComplete { responder, .. }
            | OnboardMessage::G2Results { responder, .. }
            | OnboardMessage::G3Results { responder, .. }
            | OnboardMessage::BlocksReady { responder, .. }
            | OnboardMessage::Acknowledged { responder, .. } => *responder,
        }
    }
}
