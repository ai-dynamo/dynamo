// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`UniversalBlock`] — the per-block result of hashing a [`crate::Request`].
//!
//! `UniversalBlock` carries the universal identifier ([`PositionalLineageHash`])
//! and the per-block content hash. The `u64` sequence hash needed for chain
//! extension is embedded in PLH itself
//! ([`PositionalLineageHash::current_sequence_hash`]) and accessible via
//! [`UniversalBlock::sequence_hash`], so callers no longer need a side table
//! mapping PLH → u64.
//!
//! Salt is *not* a per-block field: it is a per-request constant, identical for
//! every block in a sequence. Callers that need it should read it once from
//! [`crate::Request::salt_hash`].

use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash, TokenBlock};
use serde::{Deserialize, Serialize};

/// Per-block hashing result. PLH is self-contained for chain extension via
/// [`PositionalLineageHash::extend`]; no out-of-band tracking required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalBlock {
    /// XXH3 over the per-slot byte buffer of this block.
    pub block_hash: BlockHash,
    /// Universal identifier; transport- and lookup-friendly. Carries the full u64
    /// sequence hash inline.
    pub plh: PositionalLineageHash,
}

impl UniversalBlock {
    /// Block index in the sequence (zero-based).
    #[inline]
    pub fn position(&self) -> u64 {
        self.plh.position()
    }

    /// Parent-chained sequence hash for this block (full u64).
    #[inline]
    pub fn sequence_hash(&self) -> SequenceHash {
        self.plh.current_sequence_hash()
    }

    /// Lossy conversion: produces a [`PositionalLineageHash`] for transport / indexing.
    pub fn into_plh(self) -> PositionalLineageHash {
        self.plh
    }
}

impl From<&TokenBlock> for UniversalBlock {
    fn from(b: &TokenBlock) -> Self {
        Self {
            block_hash: b.block_hash(),
            plh: b.positional_lineage_hash(),
        }
    }
}
