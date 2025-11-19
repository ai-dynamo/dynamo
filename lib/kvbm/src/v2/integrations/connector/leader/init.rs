// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Initialization support for ConnectorLeader.
//!
//! Handles the leader-worker handshake and setup of block managers.
//! This module provides both synchronous (vLLM) and asynchronous (TensorRT-LLM)
//! initialization patterns.

use std::sync::Arc;

use anyhow::Result;

use crate::physical::manager::LayoutHandle;
use crate::v2::integrations::IntegrationsConfig;
use crate::v2::logical::manager::BlockManager;
use crate::v2::physical::layout::LayoutConfig;

use super::{G2, G3};

/// Initialized state ready for ConnectorLeader construction.
///
/// Contains all the resources gathered from worker initialization:
/// - Layout configuration derived from worker GPU layouts
/// - Worker layout handles for distributed operations
/// - Local block managers for CPU and disk tiers
#[derive(Clone)]
pub struct InitializedState {
    /// Layout configuration evaluated from worker G1 layouts
    pub layout_config: Arc<LayoutConfig>,

    /// Worker GPU layout handles (G1), rank-ordered
    pub g1_handles: Vec<LayoutHandle>,

    /// Worker CPU layout handles (G2), rank-ordered
    pub g2_handles: Vec<LayoutHandle>,

    /// Worker disk layout handles (G3), rank-ordered (optional)
    pub g3_handles: Option<Vec<LayoutHandle>>,

    /// Local CPU block manager (G2) for leader-side caching
    pub cpu_block_manager: Arc<BlockManager<G2>>,

    /// Local disk block manager (G3) for leader-side caching (optional)
    pub disk_block_manager: Option<Arc<BlockManager<G3>>>,
}

impl InitializedState {
    /// Create mock initialized state for testing/development.
    ///
    /// This bypasses the leader-worker handshake and creates minimal
    /// block managers using configuration hints.
    pub fn mock_from_config(config: &IntegrationsConfig) -> Result<Self> {
        // Create mock LayoutConfig from attention configuration
        let layout_config = Arc::new(Self::mock_layout_config(config));

        // Create mock CPU block manager
        let num_cpu_blocks = config.attention.num_cpu_blocks();
        let cpu_block_manager = if num_cpu_blocks > 0 {
            Arc::new(Self::create_block_manager(&layout_config, num_cpu_blocks)?)
        } else {
            // Default to small manager if not specified
            Arc::new(Self::create_block_manager(&layout_config, 128)?)
        };

        // Optionally create disk block manager (default: none for now)
        let disk_block_manager = None;

        Ok(Self {
            layout_config,
            g1_handles: vec![],
            g2_handles: vec![],
            g3_handles: None,
            cpu_block_manager,
            disk_block_manager,
        })
    }

    fn mock_layout_config(config: &IntegrationsConfig) -> LayoutConfig {
        let block_size = config.block_size();
        let head_size = config.attention.head_size();
        let num_heads = config.attention.num_heads();
        let dtype_bytes = config.attention.cache_dtype_bytes();

        // For now, create a simplified config
        // In real implementation, this would come from worker G1 layout discovery
        LayoutConfig::builder()
            .num_blocks(1) // Temporary - will be overridden
            .num_layers(32) // Mock - would come from model config
            .outer_dim(2) // K and V separate
            .page_size(block_size)
            .inner_dim(head_size * num_heads)
            .dtype_width_bytes(dtype_bytes)
            .build()
            .expect("Failed to build mock LayoutConfig")
    }

    fn create_block_manager<T: crate::v2::logical::blocks::BlockMetadata>(
        layout_config: &LayoutConfig,
        block_count: usize,
    ) -> Result<BlockManager<T>> {
        use crate::v2::logical::blocks::BlockRegistry;
        use crate::v2::logical::manager::FrequencyTrackingCapacity;

        let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
        let registry = BlockRegistry::with_frequency_tracker(tracker);

        BlockManager::builder()
            .block_count(block_count)
            .block_size(layout_config.page_size)
            .registry(registry)
            .with_multi_lru_backend()
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BlockManager: {:?}", e))
    }
}
