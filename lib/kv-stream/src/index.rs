// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The seam between the ledger and a prefix index.

use crate::{BlockHash, BlockKey, Parent, Tier, TierSet};

/// A prefix index kept in step with a rank's ledger. The stream calls it from the same fold
/// that updates the ledger, in event order.
///
/// What reaches it, and why:
///
/// - **Every store**, repeated or not. The tries re-cover a worker's edge on a repeated store,
///   which is what heals them after a mid-chain eviction truncated it, so repeats are not
///   filtered out. The one exception: at the ledger's block cap a store is cut at the first
///   block the ledger refused.
/// - **A remove only when the tier's last reference goes** (or the ledger never saw a store
///   for it). On a counted tier the earlier removes of a duplicated block are absorbed here,
///   which is the job Dynamo's `EventDedupFilter` does with its ref-counted policy.
/// - **A clear**, with the tiers it covers.
///
/// A trie re-covers only the blocks a store names, so after a mid-chain remove and a store of
/// that one block again it holds less than the ledger: the ledger still chains the blocks
/// after it. A snapshot exports the ledger, and an index rebuilt from one can therefore hold
/// more than the live index did. Both are within what the rank holds.
///
/// An index keeps one engine hash per content path, so the host must not present two engine
/// hashes for the same content under one parent (for example by narrowing a digest
/// differently between events).
pub trait StreamIndex {
    /// False for an index that ignores everything: the stream then skips collecting the
    /// slices these methods take.
    const PROJECTS: bool = true;

    /// `blocks` is a chain: `parent` precedes the first block and each block precedes the next.
    fn stored(&mut self, tier: Tier, parent: Parent, blocks: &[BlockKey]);

    fn removed(&mut self, tier: Tier, hashes: &[BlockHash]);

    fn cleared(&mut self, tiers: TierSet);
}

/// No index: the ledger alone. This is the engine-frontend role, where nothing asks the rank
/// for prefix matches.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoIndex;

impl StreamIndex for NoIndex {
    const PROJECTS: bool = false;

    fn stored(&mut self, _tier: Tier, _parent: Parent, _blocks: &[BlockKey]) {}

    fn removed(&mut self, _tier: Tier, _hashes: &[BlockHash]) {}

    fn cleared(&mut self, _tiers: TierSet) {}
}
