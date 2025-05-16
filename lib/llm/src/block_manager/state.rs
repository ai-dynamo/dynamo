// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use super::offload::OffloadManager;
use super::{
    block::{Block, ImmutableBlock},
    config::NixlOptions,
    pool::BlockPoolError,
};
use cudarc::driver::CudaStream;
use std::sync::Arc;

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    stream: Arc<CudaStream>,
}

impl TransferContext {
    pub fn new(nixl_agent: Arc<Option<NixlAgent>>, stream: Arc<CudaStream>) -> Self {
        Self { nixl_agent, stream }
    }

    pub fn nixl_agent(&self) -> Option<&NixlAgent> {
        self.nixl_agent.as_ref().as_ref()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[allow(dead_code)]
pub struct KvBlockManagerState<Metadata: BlockMetadata> {
    worker_id: WorkerID,
    cancellation_token: CancellationToken,

    nixl_agent: Arc<Option<NixlAgent>>,
    nixl_backends: HashMap<String, Arc<nixl_sys::Backend>>,

    disk_pool: Arc<Option<BlockPool<DiskStorage, Metadata>>>,
    host_pool: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
    device_pool: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,

    local_block_set: NixlBlockSet,
    remote_block_sets: RwLock<HashMap<WorkerID, HashMap<usize, RemoteBlocks>>>,

    offload_manager: Arc<OffloadManager<Metadata>>,
}

impl<Metadata: BlockMetadata> KvBlockManagerState<Metadata> {
    pub fn new(config: KvBlockManagerConfig) -> Result<Arc<Self>> {
        config
            .runtime
            .validate()
            .context("Validating runtime config")?;

        config.model.validate().context("Validating model config")?;

        let worker_id = config.runtime.worker_id;
        let cancellation_token = config.runtime.cancellation_token;

        // Create a map of NIXL backends
        let mut nixl_backends: HashMap<String, Arc<nixl_sys::Backend>> = HashMap::new();

        // Create a NIXL agent if NIXL is enabled and instantiate requested backends
        // TODO: Build a map of NIXL backends to block pools/sets
        let nixl_agent = Arc::new(match config.runtime.nixl {
            NixlOptions::Enabled => {
                tracing::debug!("Creating NIXL agent");
                let agent = NixlAgent::new(&worker_id.to_string())?;

                tracing::debug!("Creating NIXL backends");
                let (_ucx_mem_list1, ucx_params) = agent.get_plugin_params("UCX")?;
                let backend = agent.create_backend("UCX", &ucx_params)?;
                nixl_backends.insert("UCX".to_string(), Arc::new(backend));

                Some(agent)
            }
            NixlOptions::EnabledWithAgent(agent) => Some(agent),
            NixlOptions::Disabled => None,
        });

        // Initialize model-specific layout config. The layout_builder is incomplete at this point.
        // We will clone this builder and apply the storage-specific configs to each clone in the
        // following steps.
        let model = &config.model;
        let mut layout_builder = LayoutConfig::builder();

        layout_builder
            .num_layers(model.num_layers)
            .page_size(model.page_size)
            .inner_dim(model.inner_dim)
            .dtype(model.dtype);

        let mut next_block_set_idx = 0;
        let mut local_block_set = block::nixl::NixlBlockSet::new(worker_id);

        let (disk_pool, disk_blocks) = if let Some(config) = config.disk_layout {
            if nixl_agent.is_none() {
                tracing::warn!("NIXL is disabled; will not allocate disk blocks.");
                (Arc::new(None), None)
            } else {
                next_block_set_idx += 1;
                tracing::debug!("Constructing disk pool.");
                let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref().as_ref())?;
                local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
                let (pool, blocks) = create_block_pool::<_, Metadata>(
                    layout,
                    next_block_set_idx,
                    cancellation_token.clone(),
                    worker_id,
                )?;
                (Arc::new(Some(pool)), Some(blocks))
            }
        } else {
            tracing::debug!("No disk layout provided; will not allocate disk blocks.");
            (Arc::new(None), None)
        };

        // Create the host block pool if a host layout is provided
        let (host_pool, host_blocks) = if let Some(config) = config.host_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing host pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref().as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            let (pool, blocks) = create_block_pool::<_, Metadata>(
                layout,
                next_block_set_idx,
                cancellation_token.clone(),
                worker_id,
            )?;
            (Arc::new(Some(pool)), Some(blocks))
        } else {
            tracing::debug!("No host layout provided; will not allocate host blocks.");
            (Arc::new(None), None)
        };

        // Create the device block pool if a device layout is provided
        let (device_pool, device_blocks) = if let Some(config) = config.device_layout {
            next_block_set_idx += 1;
            tracing::debug!("Constructing device pool.");
            let layout = create_layout(layout_builder.clone(), config, nixl_agent.as_ref().as_ref())?;
            local_block_set.add_block_set(next_block_set_idx, layout.serialize()?);
            let (pool, blocks) = create_block_pool::<_, Metadata>(
                layout,
                next_block_set_idx,
                cancellation_token.clone(),
                worker_id,
            )?;
            (Arc::new(Some(pool)), Some(blocks))
        } else {
            tracing::debug!("No device layout provided; will not allocate device blocks.");
            (Arc::new(None), None)
        };

        // Finalize the local block set by adding NIXL metadata
        if let Some(nixl_agent) = nixl_agent.as_ref() {
            tracing::debug!("Finalize NixlBlockSet: adding NIXL metadata.");
            local_block_set.set_nixl_metadata(nixl_agent.get_local_md()?);
        }

        let offload_manager =
            OffloadManager::new(disk_pool.clone(), host_pool.clone(), device_pool.clone(), nixl_agent.clone())?;

        let state = Arc::new(Self {
            worker_id,
            cancellation_token,
            nixl_agent,
            nixl_backends,
            disk_pool,
            host_pool,
            device_pool,
            local_block_set,
            remote_block_sets: RwLock::new(HashMap::new()),
            offload_manager,
        });

        if let Some(mut blocks) = disk_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state
                .disk_pool
                .as_ref()
                .as_ref()
                .unwrap()
                .add_blocks_blocking(blocks)?;
        }

        if let Some(mut blocks) = host_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state
                .host_pool
                .as_ref()
                .as_ref()
                .unwrap()
                .add_blocks_blocking(blocks)?;
        }

        if let Some(mut blocks) = device_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state
                .device_pool
                .as_ref()
                .as_ref()
                .unwrap()
                .add_blocks_blocking(blocks)?;
        }

        Ok(state)
    }

    /// Exports the local blockset configuration as a serialized object.
    pub fn export_local_blockset(&self) -> Result<SerializedNixlBlockSet> {
        SerializedNixlBlockSet::try_from(&self.local_block_set)
            .context("Failed to serialize local blockset")
    }

    /// Imports a remote blockset configuration from a serialized object.
    // TODO: NIXL will validate the every descriptor list against the memory registration list for
    // a given agent; this is can be an expensive operation. To avoid this, NIXL offers the ability
    // to generate "partial pre-validated (PPV)" descriptor lists. However, to support per-block and per-layer
    // PPV lists we will need as many as `num_layers + 1` PPV lists per block:
    // - one for representing the entire block
    // - one for representing each layer individually
    //
    // A deeper dive into the performance impact of PPV lists is required to determine if this is
    // the best approach.
    //
    // If PPV are valuable, it might be beneficial to lazily instantiate PPV lists when they are
    // needed; alternatively, we could generate the entire PPV list for each block at import time.
    pub fn import_remote_blockset(
        &self,
        serialized_blockset: SerializedNixlBlockSet,
    ) -> Result<()> {
        let remote = NixlBlockSet::try_from(serialized_blockset)
            .context("Failed to deserialize remote blockset")?;

        let (block_sets, metadata, worker_id) = remote.dissolve();
        tracing::debug!("Importing remote blockset from worker {}", worker_id);

        assert_ne!(
            worker_id, self.worker_id,
            "Cannot import blockset from self"
        );

        let agent = self
            .nixl_agent
            .as_ref()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not initialized"))?;

        let mut remote_block_sets = self.remote_block_sets.write().unwrap();

        if remote_block_sets.contains_key(&worker_id) {
            anyhow::bail!(
                "Worker ID {} already exists; cannot update remote blockset",
                worker_id
            );
        }

        let mut inner_map = HashMap::new();

        for (block_set_idx, block_set_layout) in block_sets {
            // Deserialize the individual layout and create RemoteBlocks
            let remote_blocks =
                RemoteBlocks::from_serialized(block_set_layout.clone(), block_set_idx, worker_id)?;

            // check the storage type of the remote blocks
            let layout = remote_blocks.layout();
            let storage = layout.storage();

            let storage = storage
                .first()
                .ok_or_else(|| anyhow::anyhow!("No storage found in remote blockset"))?;

            match storage.mem_type() {
                MemType::Dram => {
                    tracing::trace!(block_set_idx, "Detected Host/DRAM remote descriptor");
                }
                MemType::Vram => {
                    tracing::trace!(block_set_idx, "Detected GPU/Device/VRAM remote descriptor");
                }
                _ => {
                    tracing::warn!(
                        block_set_idx,
                        "Detected unknown remote descriptor; skipping blockset..."
                    );
                    continue;
                }
            }

            inner_map.insert(block_set_idx, remote_blocks);
        }

        let agent_id = agent
            .load_remote_md(&metadata)
            .context("Loading remote metadata")?;

        // try to convert the agent_id (String) to a WorkerID (u64)
        let agent_id: WorkerID =
            agent_id // Assuming agent_id is String here
                .parse() // Parse the String into u64 (WorkerID)
                .context("Failed to parse agent ID string into WorkerID (u64)")?;

        assert_eq!(agent_id, worker_id, "Mismatch with remote worker ID");

        remote_block_sets.insert(worker_id, inner_map);

        Ok(())
    }

    /// Get a [`Vec<RemoteBlock<IsImmutable>>`] from a [`BlockDescriptorList`]
    pub fn get_remote_blocks_immutable(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<IsImmutable>>> {
        // no checks - we can always create an immutable remote block even if the bds is mutable
        self.get_remote_blocks::<IsImmutable>(bds)
    }

    /// Get a [`Vec<RemoteBlock<IsMutable>>`] from a [`BlockDescriptorList`]
    pub fn get_remote_blocks_mutable(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<IsMutable>>> {
        if bds.mutability() == BlockMutability::Mutable {
            self.get_remote_blocks::<IsMutable>(bds)
        } else {
            anyhow::bail!("Cannot get mutable remote blocks for immutable block descriptor set");
        }
    }

    /// Generate a [`Vec<RemoteBlock>`] from a [`BlockDescriptorList`]
    fn get_remote_blocks<M: MutabilityKind>(
        &self,
        bds: &BlockDescriptorList,
    ) -> Result<Vec<RemoteBlock<M>>> {
        // Get a read lock on the remote block sets
        let remote_block_sets = self.remote_block_sets.read().unwrap();

        // validate we have loaded a remote blockset for the worker and the specific block_set_idx
        let remote_blocks = remote_block_sets
            .get(&bds.worker_id())
            .and_then(|map| map.get(&bds.block_set_idx()))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No remote blockset found for worker {} and block_set_idx {}",
                    bds.worker_id(),
                    bds.block_set_idx()
                )
            })?;

        // Iterate through indices, call .block() for each, and collect results.
        // The collect::<Result<...>>() handles potential errors from .block()
        let blocks: Vec<block::nixl::RemoteBlock<M>> = bds
            .block_indices()
            .iter()
            .map(|block_idx| remote_blocks.block(*block_idx))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(blocks)
    }

    pub fn disk(&self) -> Option<&BlockPool<DiskStorage, Metadata>> {
        self.disk_pool.as_ref().as_ref()
    }

    pub fn host(&self) -> Option<&BlockPool<PinnedStorage, Metadata>> {
        self.host_pool.as_ref().as_ref()
    }

    pub fn device(&self) -> Option<&BlockPool<DeviceStorage, Metadata>> {
        self.device_pool.as_ref().as_ref()
    }

    pub fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    pub(crate) async fn enqueue_offload_block<S: Storage + 'static>(
        &self,
        block: &ImmutableBlock<S, Metadata>,
        priority: u64,
    ) -> Result<()> {
        self.offload_manager.offload(block, priority).await?;

        Ok(())
    }

    pub async fn onboard_blocks(
        &self,
        blocks: Vec<ImmutableBlock<PinnedStorage, Metadata>>,
    ) -> core::result::Result<Vec<ImmutableBlock<DeviceStorage, Metadata>>, BlockPoolError> {
        self.offload_manager.onboard(blocks).await
    }
}

impl<Metadata: BlockMetadata> std::fmt::Debug for KvBlockManagerState<Metadata> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KvBlockManagerState")
    }
}

fn create_layout<S: Storage + NixlRegisterableStorage>(
    mut builder: LayoutConfigBuilder,
    config: KvManagerLayoutConfig<S>,
    nixl_agent: Option<&NixlAgent>,
) -> Result<Arc<dyn NixlLayout<StorageType = S>>> {
    let layout = builder.num_blocks(config.num_blocks).build()?;
    if let Some(storage) = config.storage {
        let mut layout = layout.create_layout(config.layout_type, storage)?;
        if let Some(nixl_agent) = nixl_agent {
            layout.nixl_register(nixl_agent, None)?;
        }
        return Ok(Arc::new(layout));
    }

    if let Some(allocator) = config.allocator {
        let mut layout = layout.allocate_layout(config.layout_type, allocator)?;
        if let Some(nixl_agent) = nixl_agent {
            layout.nixl_register(nixl_agent, None)?;
        }
        return Ok(Arc::new(layout));
    }

    anyhow::bail!("failed to create layout");
}

#[expect(clippy::type_complexity)]
fn create_block_pool<S: Storage + NixlRegisterableStorage, M: BlockMetadata>(
    layout: Arc<dyn NixlLayout<StorageType = S>>,
    block_set_idx: usize,
    cancellation_token: CancellationToken,
    worker_id: WorkerID,
) -> Result<(BlockPool<S, M>, Vec<Block<S, M>>)> {
    let blocks = block::layout_to_blocks::<_, M>(layout, block_set_idx, worker_id)?;
    let pool = BlockPool::<S, M>::builder()
        .cancel_token(cancellation_token)
        .build()?;
    Ok((pool, blocks))
}
