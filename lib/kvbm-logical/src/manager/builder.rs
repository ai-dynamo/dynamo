// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Builder and configuration types for [`BlockManager`](super::BlockManager).

use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::metrics::{BlockPoolMetrics, MetricsAggregator, short_type_name};
use crate::tinylfu::TinyLFUTracker;

use crate::{
    blocks::BlockMetadata,
    pools::{
        BlockDuplicationPolicy, BlockStore, InactiveIndex,
        backends::{
            FifoReusePolicy, HashMapBackend, LeafPolicy, LineageBackend, LruBackend,
            MultiLruBackend,
        },
    },
    registry::BlockRegistry,
};

use super::BlockManager;

/// Capacity settings for the TinyLFU frequency tracker used by
/// [`BlockRegistry`] and the multi-level LRU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrequencyTrackingCapacity {
    /// Small capacity: 2^18 (262,144) entries
    Small,
    /// Medium capacity: 2^21 (2,097,152) entries - default
    #[default]
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

/// Configuration for the inactive pool backend.
pub enum InactiveBackendConfig {
    /// HashMap with FIFO reuse order.
    HashMap,
    /// Simple LRU — capacity automatically set to block_count.
    Lru,
    /// Multi-level LRU with 4 fixed levels — capacity automatically set to block_count.
    MultiLru {
        /// Frequency thresholds: [cold->warm, warm->hot, hot->very_hot].
        /// Default: [3, 8, 15].
        frequency_thresholds: [u8; 3],
    },
    /// Lineage backend with a selectable leaf-eviction policy.
    Lineage {
        /// Leaf-eviction ordering. Default: [`LineageEviction::Tick`].
        eviction: LineageEviction,
    },
}

/// Leaf-eviction ordering for the [`Lineage`](InactiveBackendConfig::Lineage)
/// inactive backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LineageEviction {
    /// `BTreeMap` ordered by a per-node insertion tick — a node that
    /// re-becomes a leaf returns to its original position. Historical
    /// behavior; the default. O(log n) per hook, with B-tree node churn.
    #[default]
    Tick,
    /// Intrusive FIFO over leaves — O(1) and allocation-free, but a node
    /// that re-becomes a leaf is appended at the tail.
    Fifo,
}

/// Build the runtime [`LeafPolicy`] for a [`LineageEviction`] selection.
fn lineage_leaf_policy(eviction: LineageEviction, capacity: usize) -> LeafPolicy {
    match eviction {
        LineageEviction::Tick => LeafPolicy::tick(capacity),
        LineageEviction::Fifo => LeafPolicy::fifo(capacity),
    }
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

    /// Default value of the per-block `reset_on_release` flag. When
    /// `Some(true)`, every `ImmutableBlock` constructed by this manager
    /// starts with the flag set, so its last drop bypasses the inactive
    /// pool and resets the slot directly. Individual blocks can still
    /// override via [`ImmutableBlock::set_evict_on_reset`].
    default_reset_on_release: Option<bool>,

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
            default_reset_on_release: None,
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

    /// Use HashMap backend with FIFO reuse order.
    pub fn with_hashmap_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::HashMap);
        self
    }

    /// Use the lineage backend with the default ([`Tick`](LineageEviction::Tick))
    /// leaf-eviction policy.
    pub fn with_lineage_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lineage {
            eviction: LineageEviction::default(),
        });
        self
    }

    /// Use the lineage backend with an explicit leaf-eviction policy.
    pub fn with_lineage_backend_eviction(mut self, eviction: LineageEviction) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lineage { eviction });
        self
    }

    /// Set a metrics aggregator for prometheus export.
    ///
    /// The aggregator will automatically receive this manager's metrics source.
    pub fn aggregator(mut self, aggregator: MetricsAggregator) -> Self {
        self.aggregator = Some(aggregator);
        self
    }

    /// Set the default value of the per-block `reset_on_release` flag.
    ///
    /// When `true`, every `ImmutableBlock` constructed by this manager
    /// starts with the flag set. On its last drop, the slot bypasses
    /// the inactive pool and is reset back to the free list directly
    /// (matching `release_duplicate` semantics for primary releases).
    ///
    /// Individual blocks can still override via
    /// [`crate::blocks::ImmutableBlock::set_evict_on_reset`].
    ///
    /// Default: `false`.
    pub fn with_default_reset_on_release(mut self, value: bool) -> Self {
        self.default_reset_on_release = Some(value);
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

        metrics.set_reset_pool_size(block_count as i64);

        // Create backend based on configuration
        let backend: Box<dyn InactiveIndex> = match self.inactive_backend.take() {
            Some(InactiveBackendConfig::HashMap) => {
                tracing::info!("Using HashMap for inactive pool");
                Box::new(HashMapBackend::new(Box::new(FifoReusePolicy::new())))
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
            Some(InactiveBackendConfig::Lineage { eviction }) => {
                tracing::info!("Using Lineage inactive backend ({eviction:?})");
                Box::new(LineageBackend::with_policy(
                    block_count,
                    lineage_leaf_policy(eviction, block_count),
                ))
            }
            None => {
                let eviction = LineageEviction::default();
                tracing::info!("Using default inactive backend: Lineage ({eviction:?})");
                Box::new(LineageBackend::with_policy(
                    block_count,
                    lineage_leaf_policy(eviction, block_count),
                ))
            }
        };

        // Construct unified store
        let store = BlockStore::new(
            block_count,
            block_size,
            backend,
            metrics.clone(),
            self.default_reset_on_release.unwrap_or(false),
        );

        // Register with aggregator if provided
        if let Some(ref aggregator) = self.aggregator {
            aggregator.register_source(metrics.clone());
        }

        Ok(BlockManager {
            store,
            block_registry: registry,
            duplication_policy: self
                .duplication_policy
                .unwrap_or(BlockDuplicationPolicy::Allow),
            total_blocks: block_count,
            block_size,
            metrics,
        })
    }
}
