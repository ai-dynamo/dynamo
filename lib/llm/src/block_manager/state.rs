// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod local;
mod logical;
mod resources;

use crate::block_manager::block::{MutableBlock, factory::IntoBlocks};
use crate::block_manager::locality::LogicalResources;
use crate::block_manager::offload::request::BlockResult;

use super::*;

// use super::offload::{OffloadManager, OffloadManagerConfig};
use super::{
    block::{
        Block, GlobalRegistry, ImmutableBlock, factory::LocalBlockDataFactory,
        locality::LocalityProvider,
    },
    config::NixlOptions,
    events::{EventManager, NullEventManager},
    metrics::BlockManagerMetrics,
    offload::{OffloadManager, OffloadManagerConfig},
};
use derive_getters::Dissolve;
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

pub(crate) struct Resources {
    pub worker_id: WorkerID,
    pub cancellation_token: CancellationToken,
    pub async_rt_handle: Handle,

    // nixl agent/backends for the block manager
    pub nixl_agent: Arc<Option<NixlAgent>>,
    #[expect(dead_code)]
    pub nixl_backends: HashMap<String, Arc<nixl_sys::Backend>>,

    // registry for blocks across all storage types
    pub global_registry: GlobalRegistry,

    // event manager for block manager events
    pub event_manager: Arc<dyn EventManager>,

    // metrics for the block manager
    pub metrics: Arc<BlockManagerMetrics>,

    // config for the block manager
    pub config: KvBlockManagerConfig,
}

#[allow(dead_code)]
pub struct KvBlockManagerState<Locality: LocalityProvider, Metadata: BlockMetadata> {
    resources: Arc<Resources>,

    disk_pool: Option<Arc<dyn BlockPool<DiskStorage, Locality, Metadata>>>,
    host_pool: Option<Arc<dyn BlockPool<PinnedStorage, Locality, Metadata>>>,
    device_pool: Option<Arc<dyn BlockPool<DeviceStorage, Locality, Metadata>>>,

    local_block_set: NixlBlockSet,
    remote_block_sets: RwLock<HashMap<WorkerID, HashMap<usize, RemoteBlocks>>>,
    offload_manager: Arc<OffloadManager<Locality, Metadata>>,
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> KvBlockManagerState<Locality, Metadata> {
    pub fn disk(&self) -> Option<&dyn BlockPool<DiskStorage, Locality, Metadata>> {
        self.disk_pool.as_ref().map(|pool| pool.as_ref())
    }

    pub fn host(&self) -> Option<&dyn BlockPool<PinnedStorage, Locality, Metadata>> {
        self.host_pool.as_ref().map(|pool| pool.as_ref())
    }

    pub fn device(&self) -> Option<&dyn BlockPool<DeviceStorage, Locality, Metadata>> {
        self.device_pool.as_ref().map(|pool| pool.as_ref())
    }

    pub fn worker_id(&self) -> WorkerID {
        self.resources.worker_id
    }

    pub(crate) async fn enqueue_offload_block<S: Storage + 'static>(
        &self,
        block: &ImmutableBlock<S, Locality, Metadata>,
        priority: u64,
    ) -> Result<()> {
        self.offload_manager.offload(block, priority).await?;

        Ok(())
    }

    pub fn onboard_blocks<S: Storage + 'static>(
        &self,
        blocks: Vec<ImmutableBlock<S, Locality, Metadata>>,
        targets: Option<Vec<MutableBlock<DeviceStorage, Locality, Metadata>>>,
    ) -> oneshot::Receiver<BlockResult<DeviceStorage, Locality, Metadata>> {
        self.offload_manager.onboard(blocks, targets)
    }
}

impl<R: LogicalResources, Metadata: BlockMetadata>
    KvBlockManagerState<locality::Logical<R>, Metadata>
{
    pub async fn new(config: KvBlockManagerConfig, logical_resources: R) -> Result<Arc<Self>> {
        let model_config = config.model.clone();
        let mut resources = Resources::new(config)?;
        let block_data_factories =
            logical::LogicalBlockFactories::new(&mut resources, logical_resources)?;

        let (disk_factory, host_factory, device_factory) = block_data_factories.dissolve();

        let (disk_pool, disk_blocks) = match disk_factory {
            Some(factory) => {
                let (pool, blocks) =
                    create_block_pool::<_, _, Metadata>(factory, &resources, "disk")?;
                (Some(pool), Some(blocks))
            }
            None => {
                tracing::debug!("No disk layout provided; will not allocate disk blocks.");
                (None, None)
            }
        };

        let (host_pool, host_blocks) = match host_factory {
            Some(factory) => {
                let (pool, blocks) =
                    create_block_pool::<_, _, Metadata>(factory, &resources, "host")?;
                (Some(pool), Some(blocks))
            }
            None => {
                tracing::debug!("No host layout provided; will not allocate host blocks.");
                (None, None)
            }
        };

        let (device_pool, device_blocks) = match device_factory {
            Some(factory) => {
                let (pool, blocks) =
                    create_block_pool::<_, _, Metadata>(factory, &resources, "device")?;
                (Some(pool), Some(blocks))
            }
            None => {
                tracing::debug!("No device layout provided; will not allocate device blocks.");
                (None, None)
            }
        };

        let offload_config = OffloadManagerConfig {
            nixl_agent: resources.nixl_agent.clone(),
            async_rt_handle: resources.async_rt_handle.clone(),
            metrics: resources.metrics.clone(),
            cancellation_token: resources.cancellation_token.clone(),
            model_config,
        };

        let offload_manager = OffloadManager::new(
            disk_pool.clone(),
            host_pool.clone(),
            device_pool.clone(),
            offload_config,
        )?;

        let resources = Arc::new(resources);

        let state = Arc::new(Self {
            resources: resources.clone(),
            disk_pool,
            host_pool,
            device_pool,
            local_block_set: NixlBlockSet::new(resources.worker_id),
            remote_block_sets: RwLock::new(HashMap::new()),
            offload_manager,
        });

        if let Some(mut blocks) = disk_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state.disk_pool.as_ref().unwrap().add_blocks(blocks).await?;
        }

        if let Some(mut blocks) = host_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state.host_pool.as_ref().unwrap().add_blocks(blocks).await?;
        }

        if let Some(mut blocks) = device_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state
                .device_pool
                .as_ref()
                .unwrap()
                .add_blocks(blocks)
                .await?;
        }

        Ok(state)
    }
}

// move into mod local
// move local block data factory into mod super::block
// create a method on locality to construct a block data factory from a layout builder and resources
// - this will allow us to use the locality abstraction to build our factories and block pools
impl<Metadata: BlockMetadata> KvBlockManagerState<locality::Local, Metadata> {
    pub async fn new(config: KvBlockManagerConfig) -> Result<Arc<Self>> {
        let model_config = config.model.clone();
        let mut resources = Resources::new(config)?;
        let block_data_factories = local::LocalBlockDataFactories::new(&mut resources)?;

        let (mut local_block_set, disk_factory, host_factory, device_factory) =
            block_data_factories.dissolve();

        let (disk_pool, disk_blocks) = match disk_factory {
            Some(factory) => {
                let (pool, blocks) =
                    create_block_pool::<_, _, Metadata>(factory, &resources, "disk")?;
                (Some(pool), Some(blocks))
            }
            None => {
                tracing::debug!("No disk layout provided; will not allocate disk blocks.");
                (None, None)
            }
        };

        let (host_pool, host_blocks) = match host_factory {
            Some(factory) => {
                let (pool, blocks) =
                    create_block_pool::<_, _, Metadata>(factory, &resources, "host")?;
                (Some(pool), Some(blocks))
            }
            None => {
                tracing::debug!("No disk layout provided; will not allocate disk blocks.");
                (None, None)
            }
        };

        let (device_pool, device_blocks) = match device_factory {
            Some(factory) => {
                let (pool, blocks) =
                    create_block_pool::<_, _, Metadata>(factory, &resources, "disk")?;
                (Some(pool), Some(blocks))
            }
            None => {
                tracing::debug!("No disk layout provided; will not allocate disk blocks.");
                (None, None)
            }
        };

        // Finalize the local block set by adding NIXL metadata
        if let Some(nixl_agent) = resources.nixl_agent.as_ref() {
            tracing::debug!("Finalize NixlBlockSet: adding NIXL metadata.");
            local_block_set.set_nixl_metadata(nixl_agent.get_local_md()?);
        }

        let offload_config = OffloadManagerConfig {
            nixl_agent: resources.nixl_agent.clone(),
            async_rt_handle: resources.async_rt_handle.clone(),
            metrics: resources.metrics.clone(),
            cancellation_token: resources.cancellation_token.clone(),
            model_config,
        };

        let offload_manager = OffloadManager::new(
            disk_pool.clone(),
            host_pool.clone(),
            device_pool.clone(),
            offload_config,
        )?;

        let resources = Arc::new(resources);

        let state = Arc::new(Self {
            resources: resources.clone(),
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

            state.disk_pool.as_ref().unwrap().add_blocks(blocks).await?;
        }

        if let Some(mut blocks) = host_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state.host_pool.as_ref().unwrap().add_blocks(blocks).await?;
        }

        if let Some(mut blocks) = device_blocks {
            blocks.iter_mut().for_each(|block| {
                block.set_manager(state.clone());
            });

            state
                .device_pool
                .as_ref()
                .unwrap()
                .add_blocks(blocks)
                .await?;
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
            worker_id, self.resources.worker_id,
            "Cannot import blockset from self"
        );

        let agent = self
            .resources
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
}

impl<Locality: LocalityProvider, Metadata: BlockMetadata> std::fmt::Debug
    for KvBlockManagerState<Locality, Metadata>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KvBlockManagerState")
    }
}

//     if let Some(storage) = config.storage {
//         let mut layout = layout.create_layout(config.layout_type, storage, false)?;
//         if let Some(nixl_agent) = nixl_agent {
//             layout.nixl_register(nixl_agent, None)?;
//         }
//         return Ok(layout.into());
//     }

//     if let Some(allocator) = config.allocator {
//         let mut layout = layout.allocate_layout(config.layout_type, allocator)?;
//         if let Some(nixl_agent) = nixl_agent {
//             layout.nixl_register(nixl_agent, None)?;
//         }
//         return Ok(layout.into());
//     }

//     anyhow::bail!("failed to create layout");
// }

#[expect(clippy::type_complexity)]
pub(crate) fn create_block_pool<S: Storage, L: LocalityProvider, M: BlockMetadata>(
    factory: impl IntoBlocks<S, L>,
    resources: &Resources,
    pool_name: &str,
) -> Result<(Arc<dyn BlockPool<S, L, M>>, Vec<Block<S, L, M>>)> {
    let pool = ManagedBlockPool::<S, L, M>::builder()
        .cancel_token(resources.cancellation_token.clone())
        .global_registry(resources.global_registry.clone())
        .async_runtime(resources.async_rt_handle.clone())
        .event_manager(resources.event_manager.clone())
        .pool_metrics(resources.metrics.pool(pool_name))
        .build()?;

    let blocks = factory.into_blocks()?;
    Ok((Arc::new(pool), blocks))
}

// Block state operations moved to block.rs for better organization and private field access
