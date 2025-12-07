// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy trait and built-in implementations for offload filtering.
//!
//! Policies determine which blocks should be offloaded. They are evaluated
//! as filters - blocks that fail any filter are removed from the transfer.
//!
//! # Performance Optimization
//!
//! This module uses `Either<Ready, BoxFuture>` instead of `#[async_trait]` to
//! avoid heap allocations for synchronous policies. Policies that only perform
//! local, synchronous operations (like `PresenceFilter`, `PassAllPolicy`) return
//! `Either::Left(ready(...))` which requires zero heap allocation. Policies that
//! need actual async operations return `Either::Right(Box::pin(...))`.
//!
//! # Built-in Policies
//!
//! - `PresenceFilter<Src, Dst>`: Skip blocks already present in destination tier
//! - `PresenceAndLFUFilter<Src, Dst>`: Presence check + LFU count threshold
//! - `PassAllPolicy`: No filtering (pass all blocks)
//! - `AllOfPolicy`: Composite AND policy
//! - `AnyOfPolicy`: Composite OR policy

use std::future::{Future, Ready, ready};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use futures::future::Either;

use crate::v2::logical::blocks::{BlockMetadata, BlockRegistry, ImmutableBlock};
use crate::v2::{BlockId, SequenceHash};

/// Boxed future type for async policy evaluation.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Future type for single-block policy evaluation.
///
/// - `Left(Ready<...>)`: Synchronous result, zero heap allocation
/// - `Right(BoxFuture<...>)`: Async result, requires heap allocation
pub type PolicyFuture<'a> = Either<Ready<Result<bool>>, BoxFuture<'a, Result<bool>>>;

/// Future type for batch policy evaluation.
///
/// - `Left(Ready<...>)`: Synchronous result, zero heap allocation
/// - `Right(BoxFuture<...>)`: Async result, requires heap allocation
pub type PolicyBatchFuture<'a> = Either<Ready<Result<Vec<bool>>>, BoxFuture<'a, Result<Vec<bool>>>>;

/// Create a synchronous policy result (zero allocation).
#[inline]
pub fn sync_result(result: Result<bool>) -> PolicyFuture<'static> {
    Either::Left(ready(result))
}

/// Create a synchronous batch policy result (zero allocation).
#[inline]
pub fn sync_batch_result(result: Result<Vec<bool>>) -> PolicyBatchFuture<'static> {
    Either::Left(ready(result))
}

/// Create an async policy result (boxes the future).
#[inline]
pub fn async_result<'a, F>(future: F) -> PolicyFuture<'a>
where
    F: Future<Output = Result<bool>> + Send + 'a,
{
    Either::Right(Box::pin(future))
}

/// Create an async batch policy result (boxes the future).
#[inline]
pub fn async_batch_result<'a, F>(future: F) -> PolicyBatchFuture<'a>
where
    F: Future<Output = Result<Vec<bool>>> + Send + 'a,
{
    Either::Right(Box::pin(future))
}

/// Context provided to policies for block evaluation.
#[derive(Debug)]
pub struct EvalContext<T: BlockMetadata> {
    /// Block ID
    pub block_id: BlockId,
    /// Sequence hash for this block
    pub sequence_hash: SequenceHash,
    /// Optional strong reference to the block.
    /// - Some: Strong blocks (held during evaluation)
    /// - None: Weak blocks (deferred upgrade)
    pub block: Option<ImmutableBlock<T>>,
}

impl<T: BlockMetadata> EvalContext<T> {
    /// Create a new evaluation context from a strong block reference.
    pub fn new(block: ImmutableBlock<T>) -> Self {
        Self {
            block_id: block.block_id(),
            sequence_hash: block.sequence_hash(),
            block: Some(block),
        }
    }

    /// Create a context for weak block evaluation (deferred upgrade).
    ///
    /// Used when evaluating weak blocks - we have the metadata
    /// but defer the actual upgrade until just before transfer.
    pub fn from_weak(block_id: BlockId, sequence_hash: SequenceHash) -> Self {
        Self {
            block_id,
            sequence_hash,
            block: None,
        }
    }
}

/// Trait for offload policies that filter blocks.
///
/// Policies are evaluated as a chain - a block must pass ALL policies to proceed.
/// Each policy receives an `EvalContext` with block information and returns
/// `Ok(true)` to pass or `Ok(false)` to filter out.
///
/// # Performance
///
/// This trait uses `Either<Ready, BoxFuture>` instead of `#[async_trait]` to
/// avoid heap allocations for synchronous policies. Implement using:
/// - `sync_result(Ok(true))` for synchronous policies (zero allocation)
/// - `async_result(async { ... })` for async policies (boxes the future)
///
/// # Batch Evaluation
///
/// The `evaluate_batch` method provides a default implementation that calls
/// `evaluate` for each block. Override for efficiency when the policy can
/// benefit from batching (e.g., batch registry lookups).
pub trait OffloadPolicy<T: BlockMetadata>: Send + Sync {
    /// Unique name for this policy (for logging/debugging).
    fn name(&self) -> &str;

    /// Evaluate whether a block should be offloaded.
    ///
    /// Returns:
    /// - `Ok(true)`: Block passes this filter, continue to next policy
    /// - `Ok(false)`: Block filtered out, remove from transfer
    /// - `Err(_)`: Fatal error, fail the entire transfer
    fn evaluate<'a>(&'a self, ctx: &'a EvalContext<T>) -> PolicyFuture<'a>;

    /// Batch evaluate multiple blocks.
    ///
    /// Default implementation calls `evaluate` for each block.
    /// Override for efficiency when batching is beneficial.
    fn evaluate_batch<'a>(&'a self, contexts: &'a [EvalContext<T>]) -> PolicyBatchFuture<'a> {
        // Default: sequential evaluation
        let contexts_clone: Vec<_> = contexts.iter().collect();
        async_batch_result(async move {
            let mut results = Vec::with_capacity(contexts_clone.len());
            for ctx in contexts_clone {
                // This calls the sync or async evaluate
                let result = match self.evaluate(ctx) {
                    Either::Left(ready) => ready.await,
                    Either::Right(boxed) => boxed.await,
                };
                results.push(result?);
            }
            Ok(results)
        })
    }
}

/// G1→G2 filter: skip blocks already present in destination tier.
///
/// Uses `BlockRegistry::check_presence` to determine if a block exists
/// in the destination tier without acquiring a full block reference.
/// This is efficient because it only checks the registry metadata.
///
/// # Performance
///
/// This policy is fully synchronous and returns `Either::Left(Ready)`,
/// avoiding any heap allocation per evaluation.
///
/// # Example
/// ```ignore
/// let filter = PresenceFilter::<G1, G2>::new(registry.clone());
/// // Blocks already in G2 will be filtered out
/// ```
pub struct PresenceFilter<Src: BlockMetadata, Dst: BlockMetadata> {
    registry: Arc<BlockRegistry>,
    _marker: PhantomData<(Src, Dst)>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> PresenceFilter<Src, Dst> {
    /// Create a new presence filter.
    pub fn new(registry: Arc<BlockRegistry>) -> Self {
        Self {
            registry,
            _marker: PhantomData,
        }
    }
}

impl<Src: BlockMetadata, Dst: BlockMetadata> OffloadPolicy<Src> for PresenceFilter<Src, Dst> {
    fn name(&self) -> &str {
        "PresenceFilter"
    }

    fn evaluate<'a>(&'a self, ctx: &'a EvalContext<Src>) -> PolicyFuture<'a> {
        // Purely synchronous - uses Left(Ready), zero heap allocation
        let presence = self.registry.check_presence::<Dst>(&[ctx.sequence_hash]);
        // Return false (filter out) if already present in Dst
        // presence[0].1 is true if block exists in Dst
        sync_result(Ok(!presence[0].1))
    }

    fn evaluate_batch<'a>(&'a self, contexts: &'a [EvalContext<Src>]) -> PolicyBatchFuture<'a> {
        if contexts.is_empty() {
            return sync_batch_result(Ok(Vec::new()));
        }

        // Batch lookup for efficiency - still synchronous
        let hashes: Vec<SequenceHash> = contexts.iter().map(|ctx| ctx.sequence_hash).collect();
        let presence = self.registry.check_presence::<Dst>(&hashes);

        // Invert presence: true means "not present" (pass the filter)
        sync_batch_result(Ok(presence
            .into_iter()
            .map(|(_, present)| !present)
            .collect()))
    }
}

/// G2→G3 filter: presence check + LFU count threshold.
///
/// Combines two filter conditions:
/// 1. Skip blocks already present in destination tier
/// 2. Only offload blocks with LFU count above threshold
///
/// The LFU threshold ensures we only offload "hot" blocks that have been
/// accessed frequently, avoiding wasted transfers for rarely-used blocks.
///
/// # Performance
///
/// This policy is fully synchronous and returns `Either::Left(Ready)`,
/// avoiding any heap allocation per evaluation.
///
/// # Example
/// ```ignore
/// // Only offload blocks with LFU count > 8 that aren't in G3
/// let filter = PresenceAndLFUFilter::<G2, G3>::new(registry.clone(), 8);
/// ```
pub struct PresenceAndLFUFilter<Src: BlockMetadata, Dst: BlockMetadata> {
    registry: Arc<BlockRegistry>,
    min_lfu_count: u32,
    _marker: PhantomData<(Src, Dst)>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> PresenceAndLFUFilter<Src, Dst> {
    /// Create a new presence + LFU filter with specified threshold.
    pub fn new(registry: Arc<BlockRegistry>, min_lfu_count: u32) -> Self {
        Self {
            registry,
            min_lfu_count,
            _marker: PhantomData,
        }
    }

    /// Create with default threshold of 8.
    pub fn with_default_threshold(registry: Arc<BlockRegistry>) -> Self {
        Self::new(registry, 8)
    }
}

impl<Src: BlockMetadata, Dst: BlockMetadata> OffloadPolicy<Src> for PresenceAndLFUFilter<Src, Dst> {
    fn name(&self) -> &str {
        "PresenceAndLFUFilter"
    }

    fn evaluate<'a>(&'a self, ctx: &'a EvalContext<Src>) -> PolicyFuture<'a> {
        // 1. Skip if already in Dst
        let presence = self.registry.check_presence::<Dst>(&[ctx.sequence_hash]);
        if presence[0].1 {
            return sync_result(Ok(false));
        }

        // 2. Check LFU count > threshold
        if let Some(tracker) = self.registry.frequency_tracker() {
            // Convert SequenceHash to u128 for the tracker
            let count = tracker.count(ctx.sequence_hash.as_u128());
            return sync_result(Ok(count > self.min_lfu_count));
        }

        // No frequency tracker = pass all (conservative default)
        sync_result(Ok(true))
    }

    fn evaluate_batch<'a>(&'a self, contexts: &'a [EvalContext<Src>]) -> PolicyBatchFuture<'a> {
        if contexts.is_empty() {
            return sync_batch_result(Ok(Vec::new()));
        }

        // Batch presence lookup
        let hashes: Vec<SequenceHash> = contexts.iter().map(|ctx| ctx.sequence_hash).collect();
        let presence = self.registry.check_presence::<Dst>(&hashes);

        // Get tracker once
        let tracker = self.registry.frequency_tracker();
        let min_lfu = self.min_lfu_count;

        let results: Vec<bool> = presence
            .into_iter()
            .zip(contexts.iter())
            .map(|((_, present), ctx)| {
                // Skip if present in Dst
                if present {
                    return false;
                }

                // Check LFU count
                if let Some(ref t) = tracker {
                    let count = t.count(ctx.sequence_hash.as_u128());
                    count > min_lfu
                } else {
                    true // No tracker = pass
                }
            })
            .collect();

        sync_batch_result(Ok(results))
    }
}

/// Composite policy that requires ALL sub-policies to pass (AND logic).
pub struct AllOfPolicy<T: BlockMetadata> {
    policies: Vec<Arc<dyn OffloadPolicy<T>>>,
}

impl<T: BlockMetadata> AllOfPolicy<T> {
    /// Create a new AND composite policy.
    pub fn new(policies: Vec<Arc<dyn OffloadPolicy<T>>>) -> Self {
        Self { policies }
    }

    /// Add a policy to the composite.
    pub fn with(mut self, policy: Arc<dyn OffloadPolicy<T>>) -> Self {
        self.policies.push(policy);
        self
    }
}

impl<T: BlockMetadata> OffloadPolicy<T> for AllOfPolicy<T> {
    fn name(&self) -> &str {
        "AllOfPolicy"
    }

    fn evaluate<'a>(&'a self, ctx: &'a EvalContext<T>) -> PolicyFuture<'a> {
        // Must use async because sub-policies might be async
        let policies = &self.policies;
        async_result(async move {
            for policy in policies {
                let result = match policy.evaluate(ctx) {
                    Either::Left(ready) => ready.await,
                    Either::Right(boxed) => boxed.await,
                };
                if !result? {
                    return Ok(false);
                }
            }
            Ok(true)
        })
    }
}

/// Composite policy that requires ANY sub-policy to pass (OR logic).
pub struct AnyOfPolicy<T: BlockMetadata> {
    policies: Vec<Arc<dyn OffloadPolicy<T>>>,
}

impl<T: BlockMetadata> AnyOfPolicy<T> {
    /// Create a new OR composite policy.
    pub fn new(policies: Vec<Arc<dyn OffloadPolicy<T>>>) -> Self {
        Self { policies }
    }

    /// Add a policy to the composite.
    pub fn with(mut self, policy: Arc<dyn OffloadPolicy<T>>) -> Self {
        self.policies.push(policy);
        self
    }
}

impl<T: BlockMetadata> OffloadPolicy<T> for AnyOfPolicy<T> {
    fn name(&self) -> &str {
        "AnyOfPolicy"
    }

    fn evaluate<'a>(&'a self, ctx: &'a EvalContext<T>) -> PolicyFuture<'a> {
        if self.policies.is_empty() {
            return sync_result(Ok(true)); // No policies = pass
        }

        // Must use async because sub-policies might be async
        let policies = &self.policies;
        async_result(async move {
            for policy in policies {
                let result = match policy.evaluate(ctx) {
                    Either::Left(ready) => ready.await,
                    Either::Right(boxed) => boxed.await,
                };
                if result? {
                    return Ok(true);
                }
            }
            Ok(false)
        })
    }
}

/// A pass-all policy (no filtering).
///
/// # Performance
///
/// This policy is fully synchronous and returns `Either::Left(Ready)`,
/// avoiding any heap allocation per evaluation.
pub struct PassAllPolicy<T: BlockMetadata> {
    _marker: PhantomData<T>,
}

impl<T: BlockMetadata> PassAllPolicy<T> {
    /// Create a new pass-all policy.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: BlockMetadata> Default for PassAllPolicy<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BlockMetadata> OffloadPolicy<T> for PassAllPolicy<T> {
    fn name(&self) -> &str {
        "PassAllPolicy"
    }

    fn evaluate<'a>(&'a self, _ctx: &'a EvalContext<T>) -> PolicyFuture<'a> {
        // Zero allocation - just returns ready(Ok(true))
        sync_result(Ok(true))
    }

    fn evaluate_batch<'a>(&'a self, contexts: &'a [EvalContext<T>]) -> PolicyBatchFuture<'a> {
        sync_batch_result(Ok(vec![true; contexts.len()]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require BlockRegistry infrastructure which needs
    // tokio runtime and complex setup. Basic API tests here.

    #[test]
    fn test_pass_all_policy() {
        let _policy: PassAllPolicy<()> = PassAllPolicy::new();
        // Would test evaluate with proper setup
    }

    #[test]
    fn test_all_of_policy_creation() {
        let policies: Vec<Arc<dyn OffloadPolicy<()>>> = vec![Arc::new(PassAllPolicy::new())];
        let composite = AllOfPolicy::new(policies);
        assert_eq!(composite.name(), "AllOfPolicy");
    }

    #[test]
    fn test_any_of_policy_creation() {
        let policies: Vec<Arc<dyn OffloadPolicy<()>>> = vec![Arc::new(PassAllPolicy::new())];
        let composite = AnyOfPolicy::new(policies);
        assert_eq!(composite.name(), "AnyOfPolicy");
    }

    #[tokio::test]
    async fn test_sync_result_zero_alloc() {
        // Verify sync_result returns Left variant
        let future = sync_result(Ok(true));
        assert!(matches!(future, Either::Left(_)));

        let result = match future {
            Either::Left(ready) => ready.await,
            Either::Right(_) => unreachable!(),
        };
        assert!(result.unwrap());
    }

    #[tokio::test]
    async fn test_async_result_boxes() {
        // Verify async_result returns Right variant
        let future = async_result(async { Ok(false) });
        assert!(matches!(future, Either::Right(_)));

        let result = match future {
            Either::Left(_) => unreachable!(),
            Either::Right(boxed) => boxed.await,
        };
        assert!(!result.unwrap());
    }
}
