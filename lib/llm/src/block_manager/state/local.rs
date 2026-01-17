// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// The local block factories for the block manager
///
/// This struct will construct the factories in a consistent order and can be
/// used as an intermediate step before creating the block pools.
///
/// This is useful for debugging and for testing.
#[derive(Dissolve)]
pub struct LocalBlockDataFactories {
    block_set: NixlBlockSet,
    disk_factory: Option<LocalBlockDataFactory<DiskStorage>>,
    host_factory: Option<LocalBlockDataFactory<PinnedStorage>>,
    device_factory: Option<LocalBlockDataFactory<DeviceStorage>>,
}

impl LocalBlockDataFactories {
    /// Construct the local block factories
    pub fn new(resources: &mut Resources) -> Result<Self> {
        let mut block_set = NixlBlockSet::new(resources.worker_id);
        let mut next_block_set_idx = 0;
        let layout_builder = resources.layout_builder();

        let device_factory = if let Some(config) = resources.config.device_layout.take() {
            next_block_set_idx += 1;

            let offload_filter = config.offload_filter.clone();

            tracing::debug!("Constructing device pool.");
            let layout = create_layout(
                layout_builder.clone(),
                config,
                resources.nixl_agent.as_ref().as_ref(),
            )?;
            block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            Some(LocalBlockDataFactory::new(
                layout,
                next_block_set_idx,
                resources.worker_id,
                offload_filter,
            ))
        } else {
            None
        };

        let host_factory = if let Some(config) = resources.config.host_layout.take() {
            next_block_set_idx += 1;

            let offload_filter = config.offload_filter.clone();

            tracing::debug!("Constructing host pool.");
            let layout = create_layout(
                layout_builder.clone(),
                config,
                resources.nixl_agent.as_ref().as_ref(),
            )?;
            block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            Some(LocalBlockDataFactory::new(
                layout,
                next_block_set_idx,
                resources.worker_id,
                offload_filter,
            ))
        } else {
            None
        };

        let disk_factory = if let Some(config) = resources.config.disk_layout.take() {
            let offload_filter = config.offload_filter.clone();

            if resources.nixl_agent.is_none() {
                tracing::warn!("NIXL is disabled; will not allocate disk blocks.");
                None
            } else {
                next_block_set_idx += 1;
                tracing::debug!("Constructing disk pool.");
                let layout = create_layout(
                    layout_builder.clone(),
                    config,
                    resources.nixl_agent.as_ref().as_ref(),
                )?;
                block_set.add_block_set(next_block_set_idx, layout.serialize()?);
                Some(LocalBlockDataFactory::new(
                    layout,
                    next_block_set_idx,
                    resources.worker_id,
                    offload_filter,
                ))
            }
        } else {
            None
        };

        Ok(Self {
            block_set,
            disk_factory,
            host_factory,
            device_factory,
        })
    }
}

fn create_layout<S: Storage + NixlRegisterableStorage>(
    mut builder: LayoutConfigBuilder,
    config: KvManagerLayoutConfig<S>,
    nixl_agent: Option<&NixlAgent>,
) -> Result<Arc<dyn NixlLayout<StorageType = S>>> {
    let layout = builder.num_blocks(config.num_blocks).build()?;

    if let Some(_logical) = config.logical {
        return Err(anyhow::anyhow!(
            "Logical layouts are not supported by the local builder"
        ));
    }

    if let Some(storage) = config.storage {
        let mut layout = layout.create_layout(config.layout_type, storage)?;
        if let Some(nixl_agent) = nixl_agent {
            layout.nixl_register(nixl_agent, None)?;
        }
        return Ok(layout.into());
    }

    if let Some(allocator) = config.allocator {
        let mut layout = layout.allocate_layout(config.layout_type, allocator)?;
        if let Some(nixl_agent) = nixl_agent {
            layout.nixl_register(nixl_agent, None)?;
        }
        return Ok(layout.into());
    }

    anyhow::bail!("failed to create layout");
}
