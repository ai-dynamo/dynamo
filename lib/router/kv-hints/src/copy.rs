// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV copy a booked worker can make: which source it copies from, and where that source's
//! blocks are. The router computes both for a booking and hands them to the worker-selection
//! policy's hint emitter, which turns them into actions such as [`KvHintAction::kv_fetch`].
//!
//! [`KvHintAction::kv_fetch`]: crate::wire::KvHintAction::kv_fetch

use dynamo_kv_hashing::ExternalSequenceBlockHash;
use dynamo_kv_indexer::KvTransferCandidateSource;
use dynamo_router_protocol::worker::WorkerWithDpRank;

use crate::transfer::KvTransferSourceChoice;

/// A copy `target` can make from a source: the source selection chose for it, with the local
/// prefix the choice was made against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCopySource {
    target: WorkerWithDpRank,
    choice: KvTransferSourceChoice,
    local_prefix_blocks: usize,
}

impl KvCopySource {
    /// The router builds one per booking; policies only read it.
    pub fn new(
        target: WorkerWithDpRank,
        choice: KvTransferSourceChoice,
        local_prefix_blocks: usize,
    ) -> Self {
        Self {
            target,
            choice,
            local_prefix_blocks,
        }
    }

    /// The worker that would copy.
    pub fn target(&self) -> WorkerWithDpRank {
        self.target
    }

    /// Who it would copy from.
    pub fn source(&self) -> KvTransferCandidateSource {
        self.choice.source
    }

    /// The source choice as selection made it.
    pub fn choice(&self) -> KvTransferSourceChoice {
        self.choice
    }

    /// How many request blocks, from the root, the source holds.
    pub fn source_prefix_blocks(&self) -> usize {
        self.choice.prefix_blocks
    }

    /// How many request blocks, from the root, the target already holds.
    pub fn local_prefix_blocks(&self) -> usize {
        self.local_prefix_blocks
    }

    /// The blocks the copy would add beyond the target's own prefix.
    pub fn additional_blocks(&self) -> usize {
        self.choice
            .prefix_blocks
            .saturating_sub(self.local_prefix_blocks)
    }
}

/// Where the blocks of a [`KvCopySource`] are: the source's control endpoint and the request's
/// block chain, which the fetch payload cites through the source's prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCopyLocations<'a> {
    source_address: String,
    chain: &'a [ExternalSequenceBlockHash],
    local_prefix_blocks: usize,
    source_prefix_blocks: usize,
}

impl<'a> KvCopyLocations<'a> {
    /// The router builds one per booking from the worker registry; policies only read it.
    pub fn new(
        source_address: String,
        chain: &'a [ExternalSequenceBlockHash],
        local_prefix_blocks: usize,
        source_prefix_blocks: usize,
    ) -> Self {
        Self {
            source_address,
            chain,
            local_prefix_blocks,
            source_prefix_blocks,
        }
    }

    /// The source's control endpoint a target fetches from.
    pub fn source_address(&self) -> &str {
        &self.source_address
    }

    /// The request's root-aligned block chain.
    pub fn chain(&self) -> &'a [ExternalSequenceBlockHash] {
        self.chain
    }

    /// How many request blocks, from the root, the target already holds.
    pub fn local_prefix_blocks(&self) -> usize {
        self.local_prefix_blocks
    }

    /// How many request blocks, from the root, the source holds.
    pub fn source_prefix_blocks(&self) -> usize {
        self.source_prefix_blocks
    }

    /// The chain through the source's prefix: what a fetch cites.
    pub fn fetchable_chain(&self) -> &'a [ExternalSequenceBlockHash] {
        &self.chain[..self.source_prefix_blocks.min(self.chain.len())]
    }
}
