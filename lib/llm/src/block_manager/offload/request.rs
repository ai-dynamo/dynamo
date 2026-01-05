// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::sync::Weak;
use tokio::sync::oneshot;

use crate::block_manager::block::{
    BlockMetadata, ImmutableBlock, MutableBlock, locality::LocalityProvider,
};
use crate::block_manager::pool::BlockPoolError;
use crate::block_manager::storage::Storage;

/// Higher priority offloads are done first.
/// If two offloads have the same priority, the one that was requested first is done first.
#[derive(PartialEq, Eq)]
pub struct OffloadRequestKey {
    pub priority: u64,
    pub timestamp: u64,
}

impl PartialOrd for OffloadRequestKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OffloadRequestKey {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then(self.timestamp.cmp(&other.timestamp))
    }
}

/// Data needed to offload a block.
/// While the block is in the offload queue, we hold a weak reference to it.
/// This way, we don't prevent the block from being reused if needed.
pub struct OffloadRequest<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    pub key: OffloadRequestKey,
    pub block: Weak<MutableBlock<S, L, M>>,
    pub sequence_hash: u64,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> PartialOrd for OffloadRequest<S, L, M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Order offload requests by priority, high to low.
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Ord for OffloadRequest<S, L, M> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Equality is based on sequence hash, priority, and location.
impl<S: Storage, L: LocalityProvider, M: BlockMetadata> PartialEq for OffloadRequest<S, L, M> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Eq for OffloadRequest<S, L, M> {}

pub type BlockResult<Target, Locality, Metadata> =
    Result<Vec<ImmutableBlock<Target, Locality, Metadata>>, BlockPoolError>;

pub type ResponseSender<Target, Locality, Metadata> =
    oneshot::Sender<Result<Vec<ImmutableBlock<Target, Locality, Metadata>>, BlockPoolError>>;

/// Data needed for onboarding.
/// Unlike offloading, we need a means to return the resulting blocks to the caller.
pub struct OnboardRequest<
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    M: BlockMetadata,
> {
    pub blocks: Vec<ImmutableBlock<Source, Locality, M>>,
    pub response_tx: ResponseSender<Target, Locality, M>,
    pub targets: Option<Vec<MutableBlock<Target, Locality, M>>>,
}

impl<Source: Storage, Target: Storage, Locality: LocalityProvider, M: BlockMetadata>
    OnboardRequest<Source, Target, Locality, M>
{
    pub fn new(
        blocks: Vec<ImmutableBlock<Source, Locality, M>>,
        response_tx: ResponseSender<Target, Locality, M>,
        targets: Option<Vec<MutableBlock<Target, Locality, M>>>,
    ) -> Self {
        Self {
            blocks,
            response_tx,
            targets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offload_request_key_ordering() {
        let key1 = OffloadRequestKey {
            priority: 1,
            timestamp: 1,
        };

        let key2 = OffloadRequestKey {
            priority: 2,
            timestamp: 2,
        };

        assert!(key2 < key1);

        let key3 = OffloadRequestKey {
            priority: 2,
            timestamp: 3,
        };

        assert!(key2 < key3);
    }
}
