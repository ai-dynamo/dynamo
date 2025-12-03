// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use derive_builder::Builder;
use dynamo_tokens::TokenBlockSequence;

use super::Request;
use crate::distributed::leader::SessionId;

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct RequestSlot {
    request: Request,

    /// The sequence of tokens organized by blocks. This will grow tokens are decoded.
    sequence: TokenBlockSequence,

    /// The current state of the transaction being issued on behalf of the request.
    /// This includes onboarding and offloading operations.
    #[builder(default = "TransactionState::Inactive")]
    txn_state: TransactionState,

    /// If true, then the `request_finished` has been issued for the given request,
    /// but the [`TransactionState`] is not yet [`TransactionState::Inactive`] and
    /// therefore, the deletion of the slot is delayed until all outstanding operations
    /// are complete. If the slot is marked for deletion, no new transactions will
    /// be accepted, and any queued but not yet started transactions will be canceled.
    #[builder(default = "false")]
    marked_for_deletion: bool,
}

#[derive(Debug, Clone)]
pub enum TransactionState {
    /// No active onboarding or offloading.
    Inactive,

    /// The slot is actively onboarding blocks from the remote kv storage to the worker g1 memory.
    Onboarding(SessionId),

    /// The slot is actively offloading blocks from the worker g1 memory to the remote kv storage.
    Offloading(SessionId),
}

pub enum SlotActions {
    MatchRequest {
        num_sequence_hashes: usize,
        num_computed_tokens: usize,
        num_matched_tokens: Option<usize>,
    },
    OnboardingFinished {
        session_id: SessionId,
    },
    OffloadingFinished {
        session_id: SessionId,
    },
    RequestFinished {
        finish_status: FinishedStatus,
    },
}

/// Return value for the [`RequestSlot::marked_as_finished`] method.
/// Returns [`FinishedStatus::Finished`] if the slot is already in the inactive state,
/// otherwise returns [`FinishedStatus::Pending`].
pub enum FinishedStatus {
    /// The slot is already in the inactive state, so the request is finished and can be deleted.
    Finished,

    /// The slot is not in the inactive state, we must await the completion of the outstanding operations.
    Pending,

    /// The request is not tracked by the leader. There is no slot for the request.
    UntrackedRequest,
}

impl RequestSlot {
    pub fn builder() -> RequestSlotBuilder {
        RequestSlotBuilder::default()
    }

    pub fn new(request: Request, block_size: usize) -> Result<Self> {
        let sequence = TokenBlockSequence::new(
            request.tokens.clone().into(),
            block_size as u32,
            Some(request.salt_hash),
        );
        Ok(Self::builder()
            .request(request)
            .sequence(sequence)
            .build()?)
    }

    pub fn request_id(&self) -> &str {
        &self.request.request_id
    }

    pub fn txn_state(&self) -> &TransactionState {
        &self.txn_state
    }

    pub fn is_marked_for_deletion(&self) -> bool {
        self.marked_for_deletion
    }

    pub fn mark_finished_onboarding(&mut self) -> Result<SessionId> {
        let current_state = self.txn_state.clone();
        let session_id = match current_state {
            TransactionState::Onboarding(session_id) => {
                self.txn_state = TransactionState::Inactive;
                session_id
            }
            _ => anyhow::bail!("Slot is not in the onboarding state"),
        };

        Ok(session_id)
    }

    pub fn mark_finished_offloading(&mut self) -> Result<SessionId> {
        let current_state = self.txn_state.clone();
        let session_id = match current_state {
            TransactionState::Offloading(session_id) => {
                self.txn_state = TransactionState::Inactive;
                session_id
            }
            _ => anyhow::bail!("Slot is not in the offloading state"),
        };

        Ok(session_id)
    }

    /// Mark the slot as finished.
    pub fn marked_as_finished(&mut self) -> FinishedStatus {
        self.marked_for_deletion = true;
        if matches!(self.txn_state, TransactionState::Inactive) {
            return FinishedStatus::Finished;
        } else {
            return FinishedStatus::Pending;
        }
    }
}
