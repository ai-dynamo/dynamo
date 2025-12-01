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
use crate::v2::distributed::worker::{
    LeaderLayoutConfig, NixlBackendConfigMessage, NovaWorkerClient,
};
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

    /// Coordinate layout creation with workers via Nova RPC.
    ///
    /// This method implements Phase 3 of leader-worker coordination:
    /// 1. Gathers G1 layout configs from all workers
    /// 2. Validates all configs match (same dimensions, dtype, etc.)
    /// 3. Computes G2/G3 block counts from leader config
    /// 4. Sends configure_layouts to all workers to create G2/G3 tiers
    /// 5. Builds local block managers for leader-side caching
    ///
    /// # Arguments
    /// * `workers` - NovaWorkerClient instances for each worker, rank-ordered
    /// * `config` - Leader configuration (includes host_cache and disk_cache settings)
    ///
    /// # Returns
    /// Initialized state with layout handles and block managers
    pub async fn coordinate_with_workers(
        workers: &[NovaWorkerClient],
        config: &IntegrationsConfig,
    ) -> Result<Self> {
        use anyhow::{anyhow, bail};

        if workers.is_empty() {
            bail!("No workers to coordinate with");
        }

        tracing::info!(
            num_workers = workers.len(),
            "Phase 3: Coordinating layouts with workers"
        );

        // Step 1: Gather layout configs from all workers
        let mut layout_config_futures = Vec::with_capacity(workers.len());
        for worker in workers {
            layout_config_futures.push(worker.get_layout_config()?);
        }

        let mut layout_configs = Vec::with_capacity(workers.len());
        for (i, future) in layout_config_futures.into_iter().enumerate() {
            let config = future
                .await
                .map_err(|e| anyhow!("Failed to get layout config from worker {}: {}", i, e))?;
            layout_configs.push(config);
        }

        tracing::debug!(
            num_configs = layout_configs.len(),
            "Gathered layout configs from workers"
        );

        // Step 2: Validate all configs match
        let reference_config = &layout_configs[0];
        for (i, config) in layout_configs.iter().enumerate().skip(1) {
            if config.num_layers != reference_config.num_layers {
                bail!(
                    "Layout config mismatch: worker {} has {} layers, worker 0 has {}",
                    i,
                    config.num_layers,
                    reference_config.num_layers
                );
            }
            if config.outer_dim != reference_config.outer_dim {
                bail!(
                    "Layout config mismatch: worker {} has outer_dim {}, worker 0 has {}",
                    i,
                    config.outer_dim,
                    reference_config.outer_dim
                );
            }
            if config.page_size != reference_config.page_size {
                bail!(
                    "Layout config mismatch: worker {} has page_size {}, worker 0 has {}",
                    i,
                    config.page_size,
                    reference_config.page_size
                );
            }
            if config.inner_dim != reference_config.inner_dim {
                bail!(
                    "Layout config mismatch: worker {} has inner_dim {}, worker 0 has {}",
                    i,
                    config.inner_dim,
                    reference_config.inner_dim
                );
            }
            if config.dtype_width_bytes != reference_config.dtype_width_bytes {
                bail!(
                    "Layout config mismatch: worker {} has dtype_width_bytes {}, worker 0 has {}",
                    i,
                    config.dtype_width_bytes,
                    reference_config.dtype_width_bytes
                );
            }
        }

        let validated_config = Arc::new(reference_config.clone());
        tracing::info!(?validated_config, "All worker layout configs match");

        // Step 3: Compute G2/G3 block counts from leader config
        // Use bytes_per_block from the validated layout config
        let bytes_per_block = validated_config.required_bytes() / validated_config.num_blocks;

        let host_block_count = config
            .host_cache
            .as_ref()
            .and_then(|hc| hc.compute_num_blocks(bytes_per_block))
            .unwrap_or(0);

        let disk_block_count = config
            .disk_cache
            .as_ref()
            .and_then(|dc| dc.compute_num_blocks(bytes_per_block));

        let enable_gds = config.disk_cache.as_ref().is_some_and(|dc| dc.use_gds);

        tracing::info!(
            host_block_count,
            ?disk_block_count,
            enable_gds,
            bytes_per_block,
            "Computed block counts for G2/G3 tiers"
        );

        // Step 4: Build leader config and send to workers
        let leader_config = LeaderLayoutConfig {
            host_block_count,
            disk_block_count,
            backend_config: NixlBackendConfigMessage {
                enable_posix: host_block_count > 0,
                enable_gds,
                gds_params: None, // TODO: Add GDS params from config if needed
            },
        };

        let mut configure_futures = Vec::with_capacity(workers.len());
        for worker in workers {
            configure_futures.push(worker.configure_layouts(leader_config.clone())?);
        }

        let mut g1_handles = Vec::with_capacity(workers.len());
        let mut g2_handles = Vec::with_capacity(workers.len());
        let mut g3_handles_opt: Option<Vec<LayoutHandle>> = if disk_block_count.is_some() {
            Some(Vec::with_capacity(workers.len()))
        } else {
            None
        };

        for (i, future) in configure_futures.into_iter().enumerate() {
            let response = future
                .await
                .map_err(|e| anyhow!("Failed to configure layouts on worker {}: {}", i, e))?;

            tracing::debug!(
                worker = i,
                created = ?response.created_layouts,
                "Worker configured layouts"
            );

            // Extract handles from the metadata
            // The response contains metadata with logical layout descriptors
            // For now, store placeholder handles - the actual handles would be
            // extracted from the unpacked metadata
            // TODO: Extract actual handles from response.metadata.unpack()

            // For Phase 3 scaffolding, we track which layouts were created
            // but don't yet extract the handles (that requires metadata unpacking)
            if response
                .created_layouts
                .contains(&crate::v2::logical::LogicalLayoutHandle::G1)
            {
                g1_handles.push(LayoutHandle::new(i as u64, 0)); // Placeholder
            }
            if response
                .created_layouts
                .contains(&crate::v2::logical::LogicalLayoutHandle::G2)
            {
                g2_handles.push(LayoutHandle::new(i as u64, 1)); // Placeholder
            }
            if let Some(ref mut g3_handles) = g3_handles_opt {
                if response
                    .created_layouts
                    .contains(&crate::v2::logical::LogicalLayoutHandle::G3)
                {
                    g3_handles.push(LayoutHandle::new(i as u64, 2)); // Placeholder
                }
            }
        }

        tracing::info!(
            num_g1 = g1_handles.len(),
            num_g2 = g2_handles.len(),
            num_g3 = g3_handles_opt.as_ref().map(|v| v.len()),
            "Worker layout coordination complete"
        );

        // Step 5: Build local block managers
        let cpu_block_manager = if host_block_count > 0 {
            Arc::new(Self::create_block_manager(
                &validated_config,
                host_block_count,
            )?)
        } else {
            // Default minimal block manager
            Arc::new(Self::create_block_manager(&validated_config, 128)?)
        };

        let disk_block_manager = if let Some(disk_blocks) = disk_block_count {
            Some(Arc::new(Self::create_block_manager(
                &validated_config,
                disk_blocks,
            )?))
        } else {
            None
        };

        Ok(Self {
            layout_config: validated_config,
            g1_handles,
            g2_handles,
            g3_handles: g3_handles_opt,
            cpu_block_manager,
            disk_block_manager,
        })
    }
}
