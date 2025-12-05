// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy trait and built-in implementations for offload filtering.
//!
//! Policies determine which blocks should be offloaded. They are evaluated
//! as filters - blocks that fail any filter are removed from the transfer.
//!
//! # Built-in Policies
//!
//! - `PresenceFilter<Src, Dst>`: Skip blocks already present in destination tier
//! - `PresenceAndLFUFilter<Src, Dst>`: Presence check + LFU count threshold

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::v2::logical::blocks::{BlockMetadata, BlockRegistry, ImmutableBlock};
use crate::v2::{BlockId, SequenceHash};

/// Context provided to policies for block evaluation.
#[derive(Debug)]
pub struct EvalContext<T: BlockMetadata> {
    /// Block ID
    pub block_id: BlockId,
    /// Sequence hash for this block
    pub sequence_hash: SequenceHash,
    /// Strong reference to the block (held during evaluation)
    pub block: ImmutableBlock<T>,
}

impl<T: BlockMetadata> EvalContext<T> {
    /// Create a new evaluation context.
    pub fn new(block: ImmutableBlock<T>) -> Self {
        Self {
            block_id: block.block_id(),
            sequence_hash: block.sequence_hash(),
            block,
        }
    }
}

/// Trait for offload policies that filter blocks.
///
/// Policies are evaluated as a chain - a block must pass ALL policies to proceed.
/// Each policy receives an `EvalContext` with block information and returns
/// `Ok(true)` to pass or `Ok(false)` to filter out.
///
/// # Async Evaluation
///
/// The `evaluate` method is async to support policies that need to make
/// external calls (e.g., hub client for remote availability checks).
/// For simple policies, implement synchronously and wrap in `async {}`.
///
/// # Batch Evaluation
///
/// The `evaluate_batch` method provides a default implementation that calls
/// `evaluate` for each block. Override for efficiency when the policy can
/// benefit from batching (e.g., batch registry lookups).
#[async_trait]
pub trait OffloadPolicy<T: BlockMetadata>: Send + Sync {
    /// Unique name for this policy (for logging/debugging).
    fn name(&self) -> &str;

    /// Evaluate whether a block should be offloaded.
    ///
    /// Returns:
    /// - `Ok(true)`: Block passes this filter, continue to next policy
    /// - `Ok(false)`: Block filtered out, remove from transfer
    /// - `Err(_)`: Fatal error, fail the entire transfer
    async fn evaluate(&self, ctx: &EvalContext<T>) -> Result<bool>;

    /// Batch evaluate multiple blocks.
    ///
    /// Default implementation calls `evaluate` for each block.
    /// Override for efficiency when batching is beneficial.
    async fn evaluate_batch(&self, contexts: &[EvalContext<T>]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(contexts.len());
        for ctx in contexts {
            results.push(self.evaluate(ctx).await?);
        }
        Ok(results)
    }
}

/// G1→G2 filter: skip blocks already present in destination tier.
///
/// Uses `BlockRegistry::check_presence` to determine if a block exists
/// in the destination tier without acquiring a full block reference.
/// This is efficient because it only checks the registry metadata.
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

#[async_trait]
impl<Src: BlockMetadata, Dst: BlockMetadata> OffloadPolicy<Src> for PresenceFilter<Src, Dst> {
    fn name(&self) -> &str {
        "PresenceFilter"
    }

    async fn evaluate(&self, ctx: &EvalContext<Src>) -> Result<bool> {
        let presence = self.registry.check_presence::<Dst>(&[ctx.sequence_hash]);
        // Return false (filter out) if already present in Dst
        // presence[0].1 is true if block exists in Dst
        Ok(!presence[0].1)
    }

    async fn evaluate_batch(&self, contexts: &[EvalContext<Src>]) -> Result<Vec<bool>> {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }

        // Batch lookup for efficiency
        let hashes: Vec<SequenceHash> = contexts.iter().map(|ctx| ctx.sequence_hash).collect();
        let presence = self.registry.check_presence::<Dst>(&hashes);

        // Invert presence: true means "not present" (pass the filter)
        Ok(presence.into_iter().map(|(_, present)| !present).collect())
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

#[async_trait]
impl<Src: BlockMetadata, Dst: BlockMetadata> OffloadPolicy<Src> for PresenceAndLFUFilter<Src, Dst> {
    fn name(&self) -> &str {
        "PresenceAndLFUFilter"
    }

    async fn evaluate(&self, ctx: &EvalContext<Src>) -> Result<bool> {
        // 1. Skip if already in Dst
        let presence = self.registry.check_presence::<Dst>(&[ctx.sequence_hash]);
        if presence[0].1 {
            return Ok(false);
        }

        // 2. Check LFU count > threshold
        if let Some(tracker) = self.registry.frequency_tracker() {
            // Convert SequenceHash to u128 for the tracker
            let count = tracker.count(ctx.sequence_hash.as_u128());
            return Ok(count > self.min_lfu_count);
        }

        // No frequency tracker = pass all (conservative default)
        Ok(true)
    }

    async fn evaluate_batch(&self, contexts: &[EvalContext<Src>]) -> Result<Vec<bool>> {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }

        // Batch presence lookup
        let hashes: Vec<SequenceHash> = contexts.iter().map(|ctx| ctx.sequence_hash).collect();
        let presence = self.registry.check_presence::<Dst>(&hashes);

        // Get tracker once
        let tracker = self.registry.frequency_tracker();

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
                    count > self.min_lfu_count
                } else {
                    true // No tracker = pass
                }
            })
            .collect();

        Ok(results)
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

#[async_trait]
impl<T: BlockMetadata> OffloadPolicy<T> for AllOfPolicy<T> {
    fn name(&self) -> &str {
        "AllOfPolicy"
    }

    async fn evaluate(&self, ctx: &EvalContext<T>) -> Result<bool> {
        for policy in &self.policies {
            if !policy.evaluate(ctx).await? {
                return Ok(false);
            }
        }
        Ok(true)
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

#[async_trait]
impl<T: BlockMetadata> OffloadPolicy<T> for AnyOfPolicy<T> {
    fn name(&self) -> &str {
        "AnyOfPolicy"
    }

    async fn evaluate(&self, ctx: &EvalContext<T>) -> Result<bool> {
        if self.policies.is_empty() {
            return Ok(true); // No policies = pass
        }

        for policy in &self.policies {
            if policy.evaluate(ctx).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// A pass-all policy (no filtering).
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

#[async_trait]
impl<T: BlockMetadata> OffloadPolicy<T> for PassAllPolicy<T> {
    fn name(&self) -> &str {
        "PassAllPolicy"
    }

    async fn evaluate(&self, _ctx: &EvalContext<T>) -> Result<bool> {
        Ok(true)
    }

    async fn evaluate_batch(&self, contexts: &[EvalContext<T>]) -> Result<Vec<bool>> {
        Ok(vec![true; contexts.len()])
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
}
