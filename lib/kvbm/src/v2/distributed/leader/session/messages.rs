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

// =============================================================================
// Inverted Control Pattern (Prefill-Decode) Messages
// =============================================================================
//
// These messages support the "inverted control pattern" where:
// 1. Decode creates a local session (finds local matches, holds blocks)
// 2. Decode sends the session_id to Prefill (out-of-band)
// 3. Prefill attaches to the session on Decode via Nova
// 4. Prefill controls the session remotely (queries state, triggers staging, pulls blocks)

/// Messages for the inverted control pattern (remote session attachment).
///
/// Protocol:
/// 1. Decode creates ControllableSession via `create_controllable_session()`
/// 2. Prefill sends AttachSession to join
/// 3. Decode sends SessionState with current block info
/// 4. If G3 blocks exist and not auto-staging, Prefill sends TriggerStaging
/// 5. Decode sends BlocksStaged as G3->G2 completes
/// 6. Prefill pulls blocks via RDMA, sends BlocksPulled when done
/// 7. Prefill sends DetachSession to close
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteSessionMessage {
    /// Prefill attaches to an existing session on Decode.
    /// Triggers initial state emission.
    AttachSession {
        controller: InstanceId,
        session_id: SessionId,
    },

    /// Session state snapshot sent after attachment or on state change.
    /// Contains full metadata for G2-ready blocks and counts for pending G3.
    SessionState {
        session_id: SessionId,
        /// Blocks currently in G2 (ready for RDMA pull)
        g2_blocks: Vec<G2BlockInfo>,
        /// Count of blocks pending G3->G2 staging
        g3_pending_count: usize,
        /// Optional: Full info for G3 blocks (for future direct G3 RDMA)
        g3_blocks: Option<Vec<G3BlockInfo>>,
        /// Current session phase
        phase: RemoteSessionPhase,
    },

    /// Push notification when G3->G2 staging completes for specific blocks.
    BlocksStaged {
        session_id: SessionId,
        /// Newly staged blocks now available in G2
        staged_blocks: Vec<G2BlockInfo>,
        /// Remaining G3 blocks pending
        g3_remaining_count: usize,
    },

    /// Controller commands session to stage all G3 blocks to G2.
    /// Idempotent: no-op if already staging or staged.
    TriggerStaging {
        controller: InstanceId,
        session_id: SessionId,
    },

    /// Controller signals it has pulled specific blocks and they can be released.
    BlocksPulled {
        controller: InstanceId,
        session_id: SessionId,
        pulled_hashes: Vec<SequenceHash>,
    },

    /// Controller closes the session (all done).
    DetachSession {
        controller: InstanceId,
        session_id: SessionId,
    },

    /// Session reports an error to controller.
    SessionError {
        session_id: SessionId,
        error: String,
    },
}

/// Full metadata for a G2 block (ready for RDMA pull).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2BlockInfo {
    pub block_id: BlockId,
    pub sequence_hash: SequenceHash,
    pub layout_handle: LayoutHandle,
}

/// Metadata for a G3 block (for future direct RDMA from G3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3BlockInfo {
    pub sequence_hash: SequenceHash,
    // No block_id or layout_handle yet - G3 blocks need staging first
}

/// Phase of a remote-controlled session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RemoteSessionPhase {
    /// Session created, awaiting controller attachment.
    #[default]
    AwaitingAttachment,
    /// Controller attached, may still be staging.
    Attached,
    /// G3->G2 staging in progress (auto or triggered).
    Staging,
    /// All blocks in G2, ready for RDMA pull.
    Ready,
    /// Session complete (all blocks pulled or released).
    Complete,
}

/// Options for controllable session creation.
#[derive(Debug, Clone)]
pub struct ControllableSessionOptions {
    /// If true (default), immediately start G3→G2 staging.
    /// If false, wait for controller to call trigger_staging().
    pub auto_stage: bool,
}

impl Default for ControllableSessionOptions {
    fn default() -> Self {
        Self { auto_stage: true }
    }
}

impl RemoteSessionMessage {
    /// Extract the session ID from any message variant.
    pub fn session_id(&self) -> SessionId {
        match self {
            RemoteSessionMessage::AttachSession { session_id, .. }
            | RemoteSessionMessage::SessionState { session_id, .. }
            | RemoteSessionMessage::BlocksStaged { session_id, .. }
            | RemoteSessionMessage::TriggerStaging { session_id, .. }
            | RemoteSessionMessage::BlocksPulled { session_id, .. }
            | RemoteSessionMessage::DetachSession { session_id, .. }
            | RemoteSessionMessage::SessionError { session_id, .. } => *session_id,
        }
    }

    /// Extract the controller instance ID if present.
    pub fn controller(&self) -> Option<InstanceId> {
        match self {
            RemoteSessionMessage::AttachSession { controller, .. }
            | RemoteSessionMessage::TriggerStaging { controller, .. }
            | RemoteSessionMessage::BlocksPulled { controller, .. }
            | RemoteSessionMessage::DetachSession { controller, .. } => Some(*controller),
            RemoteSessionMessage::SessionState { .. }
            | RemoteSessionMessage::BlocksStaged { .. }
            | RemoteSessionMessage::SessionError { .. } => None,
        }
    }
}
