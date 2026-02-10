// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder and configuration types for [`BlockManager`](super::BlockManager).

use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::metrics::{BlockPoolMetrics, MetricsAggregator, short_type_name};
use crate::{BlockId, pools::backends::LineageBackend, tinylfu::TinyLFUTracker};

use crate::{
    blocks::{Block, BlockMetadata, state::Reset},
    pools::{
        ActivePool, BlockDuplicationPolicy, InactivePool, InactivePoolBackend, ResetPool,
        ReusePolicy, SequenceHash,
        backends::{HashMapBackend, LruBackend, MultiLruBackend},
    },
    registry::BlockRegistry,
};

use super::BlockManager;

/// Capacity settings for the TinyLFU frequency tracker used by
/// [`BlockRegistry`] and the multi-level LRU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyTrackingCapacity {
    /// Small capacity: 2^18 (262,144) entries
    Small,
    /// Medium capacity: 2^21 (2,097,152) entries - default
    Medium,
    /// Large capacity: 2^24 (16,777,216) entries
    Large,
}

impl FrequencyTrackingCapacity {
    /// Get the size in number of entries.
    pub fn size(&self) -> usize {
        match self {
            Self::Small => 1 << 18,
            Self::Medium => 1 << 21,
            Self::Large => 1 << 24,
        }
    }

    /// Create a new [`TinyLFUTracker`] with this capacity.
    pub fn create_tracker(&self) -> Arc<TinyLFUTracker<u128>> {
        Arc::new(TinyLFUTracker::new(self.size()))
    }
}

impl Default for FrequencyTrackingCapacity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Configuration for the inactive pool backend.
pub enum InactiveBackendConfig {
    /// HashMap with configurable reuse policy
    HashMap { reuse_policy: Box<dyn ReusePolicy> },
    /// Simple LRU - capacity automatically set to block_count
    Lru,
    /// Multi-level LRU with 4 fixed levels - capacity automatically set to block_count
    MultiLru {
        /// Frequency thresholds: [cold->warm, warm->hot, hot->very_hot]
        /// Default: [3, 8, 15]
        frequency_thresholds: [u8; 3],
    },
    /// Lineage backend
    Lineage,
}

/// Error types for [`BlockManager`] builder validation.
#[derive(Debug, thiserror::Error)]
pub enum BlockManagerBuilderError {
    #[error("Block count must be greater than 0")]
    InvalidBlockCount,
    #[error("Block size mismatch: expected {expected} tokens, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },
    #[error("Invalid backend configuration: {0}")]
    InvalidBackend(String),
    #[error("Builder validation failed: {0}")]
    ValidationError(String),
}

/// Error types for [`BlockManager::reset_inactive_pool`].
#[derive(Debug, thiserror::Error)]
pub enum BlockManagerResetError {
    #[error("Reset pool count mismatch: expected {expected}, got {actual}")]
    BlockCountMismatch { expected: usize, actual: usize },
}

/// Builder for [`BlockManager`] configuration.
///
/// Construct via [`BlockManager::builder()`] and finish with [`build()`](Self::build).
pub struct BlockManagerConfigBuilder<T: BlockMetadata> {
    /// Number of blocks in the pool
    block_count: Option<usize>,

    /// Size of each block in tokens (must be power of 2, 1-1024)
    /// Default: 16
    block_size: Option<usize>,

    /// Block registry for tracking blocks and frequency
    registry: Option<BlockRegistry>,

    /// Inactive pool backend configuration
    inactive_backend: Option<InactiveBackendConfig>,

    /// Policy for handling duplicate sequence hashes
    duplication_policy: Option<BlockDuplicationPolicy>,

    /// Optional metrics aggregator for prometheus export
    aggregator: Option<MetricsAggregator>,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: BlockMetadata> Default for BlockManagerConfigBuilder<T> {
    fn default() -> Self {
        Self {
            block_count: None,
            block_size: Some(16), // Default to 16 tokens per block
            registry: None,
            inactive_backend: None,
            duplication_policy: None,
            aggregator: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: BlockMetadata> BlockManagerConfigBuilder<T> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of blocks in the pool.
    pub fn block_count(mut self, count: usize) -> Self {
        self.block_count = Some(count);
        self
    }

    /// Set the block size (number of tokens per block).
    ///
    /// # Requirements
    /// - Must be >= 1 and <= 1024
    /// - Must be a power of 2
    ///
    /// # Panics
    /// Panics if the block size doesn't meet requirements.
    pub fn block_size(mut self, size: usize) -> Self {
        assert!(
            (1..=1024).contains(&size),
            "block_size must be between 1 and 1024, got {}",
            size
        );
        assert!(
            size.is_power_of_two(),
            "block_size must be a power of 2, got {}",
            size
        );
        self.block_size = Some(size);
        self
    }

    /// Set the duplication policy.
    pub fn duplication_policy(mut self, policy: BlockDuplicationPolicy) -> Self {
        self.duplication_policy = Some(policy);
        self
    }

    /// Set the block registry.
    pub fn registry(mut self, registry: BlockRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Use simple LRU backend (capacity automatically set to block_count).
    pub fn with_lru_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lru);
        self
    }

    /// Use multi-level LRU backend with 4 fixed priority levels.
    ///
    /// Default thresholds: `[3, 8, 15]` for transitions between:
    /// Cold (0-2 hits) -> Warm (3-7) -> Hot (8-14) -> Very Hot (15+).
    pub fn with_multi_lru_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds: [3, 8, 15],
        });
        self
    }

    /// Use multi-level LRU with custom frequency thresholds.
    ///
    /// # Requirements
    /// - Thresholds must be in ascending order: cold_to_warm < warm_to_hot < hot_to_very_hot
    /// - hot_to_very_hot must be <= 15 (4-bit counter maximum)
    /// - cold_to_warm must be >= 1 (to distinguish from never-accessed blocks)
    ///
    /// # Arguments
    /// * `cold_to_warm` - Minimum frequency to move from Cold to Warm level
    /// * `warm_to_hot` - Minimum frequency to move from Warm to Hot level
    /// * `hot_to_very_hot` - Minimum frequency to move from Hot to Very Hot level
    ///
    /// # Panics
    /// Panics if thresholds don't meet the requirements above.
    pub fn with_multi_lru_backend_custom_thresholds(
        mut self,
        cold_to_warm: u8,
        warm_to_hot: u8,
        hot_to_very_hot: u8,
    ) -> Self {
        // Validate ascending order
        assert!(
            cold_to_warm < warm_to_hot && warm_to_hot < hot_to_very_hot,
            "Thresholds must be in ascending order: {} < {} < {} failed",
            cold_to_warm,
            warm_to_hot,
            hot_to_very_hot
        );

        // Validate maximum value (4-bit counter limit)
        assert!(
            hot_to_very_hot <= 15,
            "hot_to_very_hot threshold ({}) must be <= 15 (4-bit counter maximum)",
            hot_to_very_hot
        );

        // Additional validation: ensure reasonable gaps between levels
        assert!(
            cold_to_warm >= 1,
            "cold_to_warm threshold must be >= 1 to distinguish from zero-access blocks"
        );

        self.inactive_backend = Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds: [cold_to_warm, warm_to_hot, hot_to_very_hot],
        });
        self
    }

    /// Use HashMap backend with custom reuse policy.
    pub fn with_hashmap_backend(mut self, reuse_policy: Box<dyn ReusePolicy>) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::HashMap { reuse_policy });
        self
    }

    /// Use lineage backend.
    pub fn with_lineage_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lineage);
        self
    }

    /// Set a metrics aggregator for prometheus export.
    ///
    /// The aggregator will automatically receive this manager's metrics source.
    pub fn aggregator(mut self, aggregator: MetricsAggregator) -> Self {
        self.aggregator = Some(aggregator);
        self
    }

    /// Validate the configuration.
    fn validate(&self) -> Result<(), String> {
        let registry = self.registry.as_ref().ok_or("registry is required")?;

        let block_count = self.block_count.ok_or("block_count is required")?;

        if block_count == 0 {
            return Err("block_count must be greater than 0".to_string());
        }

        // Validate block_size
        let block_size = self.block_size.unwrap_or(16);
        if !block_size.is_power_of_two() || !(1..=1024).contains(&block_size) {
            return Err(format!(
                "Invalid block_size {}: must be a power of 2 between 1 and 1024",
                block_size
            ));
        }

        // Additional validation for MultiLRU thresholds at build time
        if let Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds,
        }) = &self.inactive_backend
        {
            let [t1, t2, t3] = frequency_thresholds;
            if !(*t1 < *t2 && *t2 < *t3) {
                return Err(format!(
                    "Invalid thresholds [{}, {}, {}]: must be in ascending order",
                    t1, t2, t3
                ));
            }
            if *t3 > 15 {
                return Err(format!(
                    "Invalid threshold {}: maximum frequency is 15 (4-bit counter)",
                    t3
                ));
            }

            // Validate MultiLRU requires frequency tracking
            if !registry.has_frequency_tracking() {
                return Err(
                    "MultiLRU backend requires a registry with frequency tracking".to_string(),
                );
            }
        }

        Ok(())
    }

    /// Build the [`BlockManager`].
    ///
    /// Validates configuration and constructs all pools, the upgrade closure,
    /// and the metrics source. Returns an error if validation fails or
    /// backend construction fails.
    pub fn build(mut self) -> Result<BlockManager<T>, BlockManagerBuilderError> {
        // First validate the configuration
        self.validate()
            .map_err(BlockManagerBuilderError::ValidationError)?;

        let block_count = self.block_count.unwrap();
        let block_size = self.block_size.unwrap_or(16);

        // Use provided registry
        let registry = self.registry.unwrap();

        // Create metrics
        let metrics = Arc::new(BlockPoolMetrics::new(short_type_name::<T>()));

        // Create reset pool
        let blocks: Vec<Block<T, Reset>> = (0..block_count as BlockId)
            .map(|id| Block::new(id, block_size))
            .collect();
        let reset_pool = ResetPool::new(blocks, block_size, Some(metrics.clone()));
        metrics.set_reset_pool_size(block_count as i64);

        // Create backend based on configuration
        let backend: Box<dyn InactivePoolBackend<T>> = match self.inactive_backend.take() {
            Some(InactiveBackendConfig::HashMap { reuse_policy }) => {
                tracing::info!("Using HashMap for inactive pool");
                Box::new(HashMapBackend::new(reuse_policy))
            }
            Some(InactiveBackendConfig::Lru) => {
                // Capacity automatically set to block_count
                let capacity = NonZeroUsize::new(block_count).expect("block_count must be > 0");
                tracing::info!("Using LRU for inactive pool");
                Box::new(LruBackend::new(capacity))
            }
            Some(InactiveBackendConfig::MultiLru {
                frequency_thresholds,
            }) => {
                // Require frequency tracker for MultiLRU
                let frequency_tracker = registry.frequency_tracker().ok_or_else(|| {
                    BlockManagerBuilderError::InvalidBackend(
                        "MultiLRU backend requires a registry with frequency tracking".to_string(),
                    )
                })?;

                // Each level needs capacity for all blocks since the frequency
                // distribution is unpredictable — all blocks could land in one level.
                let level_capacity =
                    NonZeroUsize::new(block_count).expect("block_count must be > 0");

                tracing::info!(
                    "Using MultiLRU inactive backend with thresholds: {:?}",
                    frequency_thresholds
                );
                Box::new(
                    MultiLruBackend::new_with_thresholds(
                        level_capacity,
                        &frequency_thresholds,
                        frequency_tracker,
                    )
                    .map_err(|e| BlockManagerBuilderError::InvalidBackend(e.to_string()))?,
                )
            }
            Some(InactiveBackendConfig::Lineage) => {
                tracing::info!("Using Lineage inactive backend");
                Box::new(LineageBackend::default())
            }
            None => {
                tracing::info!("Using default inactive backend: Lineage");
                Box::new(LineageBackend::default())
            }
        };

        // Create pools
        let inactive_pool = InactivePool::new(backend, &reset_pool, Some(metrics.clone()));
        let active_pool = ActivePool::new(registry.clone(), inactive_pool.return_fn());

        // Create upgrade function that captures the necessary components
        let registry_clone = registry.clone();
        let inactive_pool_clone = inactive_pool.clone();
        let return_fn_clone = inactive_pool.return_fn();
        let upgrade_fn = Arc::new(
            move |seq_hash: SequenceHash| -> Option<Arc<dyn crate::blocks::RegisteredBlock<T>>> {
                // Try active pool first with touch=false (using registry directly)
                if let Some(handle) = registry_clone.match_sequence_hash(seq_hash, false)
                    && let Some(block) = handle.try_get_block::<T>(return_fn_clone.clone())
                {
                    return Some(block);
                }
                // Then try inactive pool with touch=false
                if let Some(block) = inactive_pool_clone
                    .find_blocks(&[seq_hash], false)
                    .into_iter()
                    .next()
                {
                    return Some(block);
                }
                None
            },
        );

        // Register with aggregator if provided
        if let Some(ref aggregator) = self.aggregator {
            aggregator.register_source(metrics.clone());
        }

        Ok(BlockManager {
            reset_pool,
            active_pool,
            inactive_pool,
            block_registry: registry,
            duplication_policy: self
                .duplication_policy
                .unwrap_or(BlockDuplicationPolicy::Allow),
            upgrade_fn,
            allocate_mutex: Mutex::new(()),
            total_blocks: block_count,
            block_size,
            metrics,
        })
    }
}
