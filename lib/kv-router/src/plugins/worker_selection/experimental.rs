// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unstable worker-selection extensions.
//!
//! Items in this module have no compatibility guarantee. They may change or be removed in any
//! release.

use super::{WorkerSelectionContext, WorkerSelectionPolicy};
use crate::protocols::WorkerWithDpRank;

/// What the router does with its KV fetch hint for the selected worker.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvTransferAction {
    /// Keep the router's existing hint behavior.
    #[default]
    Default,
    /// Do not attach the router's `kv.fetch` hint to this request.
    Skip,
}

/// Prefix coverage of the selected worker and the router's best fetch source.
#[derive(Clone, Copy, Debug)]
pub struct KvTransferInput {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) local_prefix_blocks: u32,
    pub(crate) source_prefix_blocks: u32,
}

impl KvTransferInput {
    /// The selected worker and rank.
    pub fn worker(self) -> WorkerWithDpRank {
        self.worker
    }

    /// Root-aligned prefix blocks the selected worker holds in its device and host-pinned tiers.
    pub fn local_prefix_blocks(self) -> u32 {
        self.local_prefix_blocks
    }

    /// Root-aligned prefix blocks held by the best eligible source. Always greater than
    /// [`Self::local_prefix_blocks`].
    pub fn source_prefix_blocks(self) -> u32 {
        self.source_prefix_blocks
    }

    /// Prefix blocks the fetch can add beyond the selected worker's own prefix.
    pub fn additional_blocks(self) -> u32 {
        self.source_prefix_blocks - self.local_prefix_blocks
    }
}

/// Decides whether to keep the router's KV fetch hint for one selection.
pub trait KvTransferPolicy: Send {
    /// Called once per booked selection after the picker, only when the router would attach a
    /// fetch hint. The decision does not change the selected worker.
    fn decide(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: KvTransferInput,
    ) -> KvTransferAction;
}

/// Attach a fetch-hint policy to a worker-selection policy.
pub fn with_kv_transfer_policy(
    policy: WorkerSelectionPolicy,
    transfer: Box<dyn KvTransferPolicy>,
) -> WorkerSelectionPolicy {
    policy.with_kv_transfer_policy(transfer)
}
