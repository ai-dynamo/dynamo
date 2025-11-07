// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::block_manager::{block::factory::logical::LogicalBlockFactory, storage::StorageType};

/// The local block factories for the block manager
///
/// This struct will construct the factories in a consistent order and can be
/// used as an intermediate step before creating the block pools.
///
/// This is useful for debugging and for testing.
#[derive(Dissolve)]
pub struct LogicalBlockFactories<R: LogicalResources> {
    disk_factory: Option<LogicalBlockFactory<DiskStorage, R>>,
    host_factory: Option<LogicalBlockFactory<PinnedStorage, R>>,
    device_factory: Option<LogicalBlockFactory<DeviceStorage, R>>,
}

impl<R: LogicalResources> LogicalBlockFactories<R> {
    /// Construct the local block factories
    pub fn new(resources: &mut Resources, logical_resources: R) -> Result<Self> {
        let mut next_block_set_idx = 0;
        let layout_builder = resources.layout_builder();

        let logical_resources = Arc::new(logical_resources);

        let device_factory = if let Some(config) = resources.config.device_layout.take() {
            next_block_set_idx += 1;

            let offload_filter = config.offload_filter.clone();

            let mut builder = layout_builder.clone();
            let config = Arc::new(builder.num_blocks(config.num_blocks).build()?);

            let factory = LogicalBlockFactory::new(
                config,
                next_block_set_idx,
                resources.worker_id,
                logical_resources.clone(),
                StorageType::Device(0),
                offload_filter,
            );

            Some(factory)
        } else {
            None
        };

        let host_factory = if let Some(config) = resources.config.host_layout.take() {
            next_block_set_idx += 1;

            let offload_filter = config.offload_filter.clone();

            let mut builder = layout_builder.clone();
            let config = Arc::new(builder.num_blocks(config.num_blocks).build()?);
            let factory = LogicalBlockFactory::new(
                config,
                next_block_set_idx,
                resources.worker_id,
                logical_resources.clone(),
                StorageType::Pinned,
                offload_filter,
            );

            Some(factory)
        } else {
            None
        };

        let disk_factory = if let Some(config) = resources.config.disk_layout.take() {
            next_block_set_idx += 1;

            let offload_filter = config.offload_filter.clone();

            let mut builder = layout_builder.clone();
            let config = Arc::new(builder.num_blocks(config.num_blocks).build()?);
            let factory = LogicalBlockFactory::new(
                config,
                next_block_set_idx,
                resources.worker_id,
                logical_resources.clone(),
                StorageType::Disk(0),
                offload_filter,
            );

            Some(factory)
        } else {
            None
        };

        Ok(Self {
            disk_factory,
            host_factory,
            device_factory,
        })
    }
}
