// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    Storage,
    block::{
        BasicMetadata, Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock,
        WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
    },
    config::RemoteTransferContext,
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    offload::MAX_TRANSFER_BATCH_SIZE,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    v2::physical::{
        layout::PhysicalLayout, manager::TransportManager, transfer::LayoutHandle,
        transfer::options::TransferOptions,
    },
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};
use tokio_util::sync::CancellationToken;

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A batching wrapper for connector transfers to prevent resource exhaustion.
/// Splits large transfers into smaller batches that can be handled by the resource pools.
#[derive(Clone, Debug)]
pub struct ConnectorTransferBatcher {
    max_batch_size: usize,
}

impl ConnectorTransferBatcher {
    pub fn new() -> Self {
        Self {
            max_batch_size: MAX_TRANSFER_BATCH_SIZE,
        }
    }

    pub async fn execute_batched_transfer<T: BlockTransferDirectHandler>(
        &self,
        handler: &T,
        request: BlockTransferRequest,
    ) -> Result<()> {
        let blocks = request.blocks();
        let num_blocks = blocks.len();

        if num_blocks <= self.max_batch_size {
            return handler.execute_transfer_direct(request).await;
        }

        let batches = blocks.chunks(self.max_batch_size);

        let batch_futures: Vec<_> = batches
            .map(|batch| {
                let batch_request = BlockTransferRequest {
                    from_pool: *request.from_pool(),
                    to_pool: *request.to_pool(),
                    blocks: batch.to_vec(),
                    connector_req: None,
                    sequence_hashes: None,
                };
                handler.execute_transfer_direct(batch_request)
            })
            .collect();

        // Execute all batches concurrently
        tracing::debug!("Executing {} batches concurrently", batch_futures.len());

        match try_join_all(batch_futures).await {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("Batched connector transfer failed: {}", e);
                Err(e)
            }
        }
    }
}

#[async_trait]
pub trait BlockTransferHandler: Send + Sync {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()>;

    fn scheduler_client(&self) -> Option<TransferSchedulerClient>;

    /// Execute a remote transfer. Returns error if remote transfers not configured.
    async fn execute_remote_transfer(&self, _request: RemoteTransferRequest) -> Result<()> {
        Err(anyhow::anyhow!(
            "Remote transfers not supported by this handler"
        ))
    }
}

#[async_trait]
pub trait BlockTransferDirectHandler {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()>;
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
/// Also handles remote storage transfers (G4 object storage, remote disk) when configured.
#[derive(Clone)]
pub struct BlockTransferHandlerV1 {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    remote_context: Option<Arc<RemoteTransferContext>>,
    cancel_token: CancellationToken,
}

#[async_trait]
impl BlockTransferHandler for BlockTransferHandlerV1 {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    fn scheduler_client(&self) -> Option<TransferSchedulerClient> {
        self.scheduler_client.clone()
    }

    async fn execute_remote_transfer(&self, request: RemoteTransferRequest) -> Result<()> {
        let remote_context = self
            .remote_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Remote transfer context not configured"))?;

        let pipeline = request.to_pipeline();

        if pipeline.num_blocks() == 0 {
            tracing::debug!("Remote transfer request has no blocks, skipping");
            return Ok(());
        }

        let host_blocks = self
            .host
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Host blocks not available for bounce buffers"))?;

        tracing::debug!(
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            direction = ?pipeline.direction(),
            num_blocks = pipeline.num_blocks(),
            "Executing remote transfer"
        );

        // Direct pipelines should use pipeline.execute() locally, not be sent to workers.
        // All worker transfers require explicit bounce_block_ids from the connector.
        let bounce_ids = pipeline.bounce_block_ids().ok_or_else(|| {
            anyhow::anyhow!(
                "Remote transfer via worker requires explicit bounce_block_ids. \
                Direct pipelines should use pipeline.execute() locally instead."
            )
        })?;

        let bounce_block_list: Vec<LocalBlockData<PinnedStorage>> = bounce_ids
            .iter()
            .map(|&id| {
                host_blocks.get(id).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Host block {} not found (have {} blocks)",
                        id,
                        host_blocks.len()
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        pipeline
            .execute(&bounce_block_list, remote_context, &self.cancel_token)
            .await?;

        tracing::debug!(
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            "Remote transfer completed successfully"
        );

        // Notify the scheduler that this operation completed (if connector tracking is enabled)
        if let Some(connector_req) = request.connector_req {
            if let Some(scheduler_client) = self.scheduler_client.clone() {
                tracing::debug!(
                    target = "kvbm-g4",
                    request_id = %request.request_id,
                    operation_id = %request.operation_id,
                    "Notifying scheduler of transfer completion"
                );

                let handle = scheduler_client.schedule_transfer(connector_req).await?;
                handle.mark_complete(Ok(())).await;

                tracing::debug!(
                    target = "kvbm-g4",
                    request_id = %request.request_id,
                    operation_id = %request.operation_id,
                    "Scheduler notified successfully"
                );
            } else {
                tracing::warn!(
                    target = "kvbm-g4",
                    request_id = %request.request_id,
                    operation_id = %request.operation_id,
                    "No scheduler client available, cannot notify completion"
                );
            }
        }

        Ok(())
    }
}

#[async_trait]
impl BlockTransferDirectHandler for BlockTransferHandlerV1 {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing transfer of {} blocks from {:?} to {:?}",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool()
        );

        tracing::debug!("request: {request:#?}");

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => self.begin_transfer(&self.device, &self.host, request).await,
            (Device, Disk) => self.begin_transfer(&self.device, &self.disk, request).await,
            (Host, Device) => self.begin_transfer(&self.host, &self.device, request).await,
            (Host, Disk) => self.begin_transfer(&self.host, &self.disk, request).await,
            (Disk, Device) => self.begin_transfer(&self.disk, &self.device, request).await,
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        Ok(())
    }
}

impl BlockTransferHandlerV1 {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        remote_context: Option<Arc<RemoteTransferContext>>,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
            remote_context,
            cancel_token,
        })
    }

    fn get_local_data<S: Storage>(
        blocks: Option<Vec<LocalBlock<S, BasicMetadata>>>,
    ) -> Option<LocalBlockDataList<S>> {
        blocks.map(|blocks| {
            blocks
                .into_iter()
                .map(|b| {
                    let block_data = b.block_data() as &dyn Any;

                    block_data
                        .downcast_ref::<LocalBlockData<S>>()
                        .unwrap()
                        .clone()
                })
                .collect()
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target>(
        &self,
        source_pool_list: &Option<LocalBlockDataList<Source>>,
        target_pool_list: &Option<LocalBlockDataList<Target>>,
        request: BlockTransferRequest,
    ) -> Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage + NixlDescriptor,
        Target: Storage + NixlDescriptor,
        // Check that the source block is readable, local, and writable to the target block.
        LocalBlockData<Source>:
            ReadableBlock<StorageType = Source> + Local + WriteToStrategy<LocalBlockData<Target>>,
        // Check that the target block is writable.
        LocalBlockData<Target>: WritableBlock<StorageType = Target>,
        LocalBlockData<Source>: BlockDataProvider<Locality = locality::Local>,
        LocalBlockData<Target>: BlockDataProviderMut<Locality = locality::Local>,
    {
        let Some(source_pool_list) = source_pool_list else {
            return Err(anyhow::anyhow!("Source pool manager not initialized"));
        };
        let Some(target_pool_list) = target_pool_list else {
            return Err(anyhow::anyhow!("Target pool manager not initialized"));
        };

        // Extract the `from` and `to` indices from the request.
        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        // Get the blocks corresponding to the indices.
        let sources: Vec<LocalBlockData<Source>> = source_idxs
            .map(|idx| source_pool_list[idx].clone())
            .collect();
        let mut targets: Vec<LocalBlockData<Target>> = target_idxs
            .map(|idx| target_pool_list[idx].clone())
            .collect();

        // Perform the transfer, and return the notifying channel.
        match sources.write_to(&mut targets, self.context.clone()) {
            Ok(channel) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
        }
    }
}

#[derive(Clone)]
pub struct BlockTransferHandlerV2 {
    device_handle: Option<LayoutHandle>,
    host_handle: Option<LayoutHandle>,
    disk_handle: Option<LayoutHandle>,
    transport_manager: TransportManager,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
}

impl BlockTransferHandlerV2 {
    pub fn new(
        device_layout: Option<PhysicalLayout>,
        host_layout: Option<PhysicalLayout>,
        disk_layout: Option<PhysicalLayout>,
        transport_manager: TransportManager,
        scheduler_client: Option<TransferSchedulerClient>,
    ) -> Result<Self> {
        Ok(Self {
            device_handle: device_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            host_handle: host_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            disk_handle: disk_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            transport_manager,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
        })
    }
}

#[async_trait]
impl BlockTransferHandler for BlockTransferHandlerV2 {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    fn scheduler_client(&self) -> Option<TransferSchedulerClient> {
        self.scheduler_client.clone()
    }
}

#[async_trait]
impl BlockTransferDirectHandler for BlockTransferHandlerV2 {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        let (src, dst) = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => (self.device_handle.as_ref(), self.host_handle.as_ref()),
            (Device, Disk) => (self.device_handle.as_ref(), self.disk_handle.as_ref()),
            (Host, Device) => (self.host_handle.as_ref(), self.device_handle.as_ref()),
            (Host, Disk) => (self.host_handle.as_ref(), self.disk_handle.as_ref()),
            (Disk, Device) => (self.disk_handle.as_ref(), self.device_handle.as_ref()),
            _ => return Err(anyhow::anyhow!("Invalid transfer type.")),
        };

        if let (Some(src), Some(dst)) = (src, dst) {
            let src_block_ids = request
                .blocks()
                .iter()
                .map(|(from, _)| *from)
                .collect::<Vec<_>>();
            let dst_block_ids = request
                .blocks()
                .iter()
                .map(|(_, to)| *to)
                .collect::<Vec<_>>();

            self.transport_manager
                .execute_transfer(
                    *src,
                    &src_block_ids,
                    *dst,
                    &dst_block_ids,
                    TransferOptions::default(),
                )?
                .await?;
        } else {
            return Err(anyhow::anyhow!("Invalid transfer type."));
        }

        Ok(())
    }
}

#[async_trait]
impl<T: ?Sized + BlockTransferHandler> Handler for T {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let result = if let Some(req) = request.connector_req.take() {
            let operation_id = req.uuid;

            tracing::debug!(
                request_id = %req.request_id,
                operation_id = %operation_id,
                "scheduling transfer"
            );

            let client = self
                .scheduler_client()
                .expect("scheduler client is required");

            let handle = client.schedule_transfer(req).await?;

            // we don't support cancellation yet
            assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

            match self.execute_transfer(request).await {
                Ok(_) => {
                    handle.mark_complete(Ok(())).await;
                    Ok(())
                }
                Err(e) => {
                    handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await;
                    Err(e)
                }
            }
        } else {
            self.execute_transfer(request).await
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}

impl BlockTransferHandlerV1 {
    /// Public accessor for get_local_data for use by external callers.
    pub fn get_local_data_pub<S: Storage>(
        blocks: Option<Vec<LocalBlock<S, BasicMetadata>>>,
    ) -> Option<LocalBlockDataList<S>> {
        Self::get_local_data(blocks)
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::block::transfer::remote::{
        RemoteBlockDescriptor, RemoteKey, RemoteTransferPipeline,
    };

    /// Test that RemoteTransferRequest serializes and deserializes correctly via ZMQ path.
    #[test]
    fn test_remote_transfer_request_zmq_roundtrip() {
        let descs = vec![
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x1234, 4096),
            RemoteBlockDescriptor::object_from_hash("test-bucket", 0x5678, 4096),
        ];
        let pipeline = RemoteTransferPipeline::onboard_with_bounce(descs, vec![0, 1], vec![10, 11]);

        let request = RemoteTransferRequest::new_with_connector_req(
            "test-req-001".to_string(),
            uuid::Uuid::new_v4(),
            &pipeline,
            crate::block_manager::connector::protocol::LeaderTransferRequest {
                request_id: "test-req-001".to_string(),
                uuid: uuid::Uuid::new_v4(),
                requirement: None,
                request_type: crate::block_manager::connector::protocol::RequestType::Immediate,
            },
        );

        // Serialize as JSON (same as ZMQ transport)
        let json = serde_json::to_vec(&request).unwrap();

        // Deserialize back
        let restored: RemoteTransferRequest = serde_json::from_slice(&json).unwrap();

        assert_eq!(restored.request_id, "test-req-001");
        assert!(restored.is_onboard());
        assert_eq!(restored.num_blocks(), 2);
        assert!(restored.connector_req.is_some());

        // Verify pipeline can be reconstructed
        let restored_pipeline = restored.to_pipeline();
        assert!(restored_pipeline.has_bounce());
        assert_eq!(
            restored_pipeline.bounce_block_ids(),
            Some([0, 1].as_slice())
        );
        assert_eq!(
            restored_pipeline.device_block_ids(),
            Some([10, 11].as_slice())
        );
    }

    /// Test RemoteTransferPipeline accessors.
    #[test]
    fn test_remote_transfer_pipeline_accessors() {
        let descs = vec![
            RemoteBlockDescriptor::object_from_hash("bucket", 0xaaaa, 8192),
            RemoteBlockDescriptor::object_from_hash("bucket", 0xbbbb, 8192),
            RemoteBlockDescriptor::object_from_hash("bucket", 0xcccc, 8192),
        ];

        // With bounce
        let pipeline = RemoteTransferPipeline::offload_with_bounce(
            descs.clone(),
            vec![100, 101, 102],
            vec![200, 201, 202],
        );
        assert_eq!(
            pipeline.direction(),
            crate::block_manager::block::transfer::remote::RemoteTransferDirection::Offload
        );
        assert!(pipeline.has_bounce());
        assert_eq!(pipeline.num_blocks(), 3);
        assert_eq!(
            pipeline.bounce_block_ids(),
            Some([100, 101, 102].as_slice())
        );
        assert_eq!(
            pipeline.device_block_ids(),
            Some([200, 201, 202].as_slice())
        );

        // Direct (no bounce)
        let direct_pipeline = RemoteTransferPipeline::onboard_direct(descs);
        assert!(!direct_pipeline.has_bounce());
        assert_eq!(direct_pipeline.bounce_block_ids(), None);
        assert_eq!(direct_pipeline.device_block_ids(), None);
    }

    /// Test that disk key descriptors serialize correctly.
    #[test]
    fn test_disk_key_descriptor_roundtrip() {
        let disk_key =
            RemoteKey::Disk(crate::block_manager::block::transfer::remote::DiskKey::new(
                "/mnt/remote-nvme",
                "block_0x1234",
            ));
        let desc = RemoteBlockDescriptor::new(disk_key, 16384);

        let pipeline = RemoteTransferPipeline::offload_direct(vec![desc]);
        let request =
            RemoteTransferRequest::new("disk-test".to_string(), uuid::Uuid::new_v4(), &pipeline);

        let json = serde_json::to_vec(&request).unwrap();
        let restored: RemoteTransferRequest = serde_json::from_slice(&json).unwrap();

        let restored_pipeline = restored.to_pipeline();
        let restored_descs = restored_pipeline.descriptors();
        assert_eq!(restored_descs.len(), 1);
        assert_eq!(
            restored_descs[0].kind(),
            crate::block_manager::block::transfer::remote::RemoteStorageKind::Disk
        );
    }
}

/// Integration tests for remote transfers with checksum validation.
/// Requires NIXL with OBJ and POSIX backends.
#[cfg(all(test, feature = "testing-cuda", feature = "testing-remote-storage"))]
mod integration_tests {
    use super::*;
    use crate::block_manager::{
        BasicMetadata, LayoutConfig,
        block::transfer::remote::{RemoteBlockDescriptor, RemoteTransferPipeline},
        block::{Block, BlockData, data::BlockDataExt},
        config::RemoteTransferContext,
        layout::{FullyContiguous, nixl::NixlLayout as NixlLayoutTrait},
        storage::{PinnedAllocator, PinnedStorage},
    };
    use nixl_sys::Agent as NixlAgent;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    const BLOCK_SIZE: usize = 4096;
    const NUM_BLOCKS: usize = 4;

    fn create_agent_with_backends() -> anyhow::Result<NixlAgent> {
        let agent = NixlAgent::new("transfer-test")?;
        if let Ok((_, p)) = agent.get_plugin_params("OBJ") {
            let _ = agent.create_backend("OBJ", &p);
        }
        if let Ok((_, p)) = agent.get_plugin_params("POSIX") {
            let _ = agent.create_backend("POSIX", &p);
        }
        Ok(agent)
    }

    fn allocate_blocks(
        agent: &NixlAgent,
    ) -> anyhow::Result<Vec<Block<PinnedStorage, locality::Local, BasicMetadata>>> {
        let config = LayoutConfig::builder()
            .num_blocks(NUM_BLOCKS)
            .num_layers(1)
            .outer_dim(1)
            .page_size(BLOCK_SIZE / 2)
            .inner_dim(1)
            .dtype_width_bytes(2)
            .build()?;
        let mut layout = FullyContiguous::allocate(config, &PinnedAllocator::new()?)?;
        layout.nixl_register(agent, None)?;
        let layout = Arc::new(layout);
        Ok((0..NUM_BLOCKS)
            .map(|i| {
                Block::new(
                    BlockData::new(layout.clone(), i, 0, 0),
                    BasicMetadata::default(),
                )
                .unwrap()
            })
            .collect())
    }

    fn fill_pattern(
        blocks: &[Block<PinnedStorage, locality::Local, BasicMetadata>],
        seed: u64,
    ) -> u64 {
        let mut h = DefaultHasher::new();
        for (bi, b) in blocks.iter().enumerate() {
            let v = b.block_data().block_view().unwrap();
            let s = unsafe { std::slice::from_raw_parts_mut(v.as_ptr() as *mut u8, v.size()) };
            s.iter_mut()
                .enumerate()
                .for_each(|(j, x)| *x = ((seed ^ bi as u64 ^ j as u64) & 0xFF) as u8);
            s.hash(&mut h);
        }
        h.finish()
    }

    fn checksum(blocks: &[Block<PinnedStorage, locality::Local, BasicMetadata>]) -> u64 {
        let mut h = DefaultHasher::new();
        for b in blocks {
            let v = b.block_data().block_view().unwrap();
            unsafe { std::slice::from_raw_parts(v.as_ptr(), v.size()) }.hash(&mut h);
        }
        h.finish()
    }

    fn clear(blocks: &[Block<PinnedStorage, locality::Local, BasicMetadata>]) {
        for b in blocks {
            let v = b.block_data().block_view().unwrap();
            unsafe { std::slice::from_raw_parts_mut(v.as_ptr() as *mut u8, v.size()) }.fill(0);
        }
    }

    #[tokio::test]
    async fn test_object_storage_roundtrip() -> anyhow::Result<()> {
        let bucket =
            std::env::var("AWS_DEFAULT_BUCKET").unwrap_or_else(|_| "test-bucket".to_string());
        let agent = create_agent_with_backends()?;
        let blocks = allocate_blocks(&agent)?;
        let cancel = CancellationToken::new();

        let ctx = Arc::new(RemoteTransferContext::for_object(
            Arc::new(crate::block_manager::block::transfer::TransferContext::new(
                Arc::new(Some(agent)),
                crate::block_manager::storage::DeviceAllocator::new(0)?
                    .ctx()
                    .new_stream()?,
                tokio::runtime::Handle::current(),
                None,
            )),
            Some(bucket.clone()),
        ));

        let before = fill_pattern(&blocks, 0xDEADBEEF);
        let id = uuid::Uuid::new_v4().as_u128() as u64;
        let descs: Vec<_> = (0..NUM_BLOCKS)
            .map(|i| RemoteBlockDescriptor::object_from_hash(&bucket, id + i as u64, BLOCK_SIZE))
            .collect();

        RemoteTransferPipeline::offload_direct(descs.clone())
            .execute(&blocks, &ctx, &cancel)
            .await?;
        clear(&blocks);
        assert_ne!(before, checksum(&blocks));
        RemoteTransferPipeline::onboard_direct(descs)
            .execute(&blocks, &ctx, &cancel)
            .await?;

        assert_eq!(
            before,
            checksum(&blocks),
            "Object storage roundtrip: checksum mismatch"
        );
        println!("Object storage roundtrip: OK");
        Ok(())
    }

    #[tokio::test]
    async fn test_disk_storage_roundtrip() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().to_str().unwrap();
        let agent = create_agent_with_backends()?;
        let blocks = allocate_blocks(&agent)?;
        let cancel = CancellationToken::new();

        let ctx = Arc::new(RemoteTransferContext::for_disk(
            Arc::new(crate::block_manager::block::transfer::TransferContext::new(
                Arc::new(Some(agent)),
                crate::block_manager::storage::DeviceAllocator::new(0)?
                    .ctx()
                    .new_stream()?,
                tokio::runtime::Handle::current(),
                None,
            )),
            path.to_string(),
            false, // use_gds
        ));

        let before = fill_pattern(&blocks, 0xCAFEBABE);
        let id = uuid::Uuid::new_v4().as_u128() as u64;
        let descs: Vec<_> = (0..NUM_BLOCKS)
            .map(|i| RemoteBlockDescriptor::disk_from_hash(path, id + i as u64, BLOCK_SIZE))
            .collect();

        RemoteTransferPipeline::offload_direct(descs.clone())
            .execute(&blocks, &ctx, &cancel)
            .await?;
        clear(&blocks);
        assert_ne!(before, checksum(&blocks));
        RemoteTransferPipeline::onboard_direct(descs)
            .execute(&blocks, &ctx, &cancel)
            .await?;

        assert_eq!(
            before,
            checksum(&blocks),
            "Disk storage roundtrip: checksum mismatch"
        );
        println!("Disk storage roundtrip: OK");
        Ok(())
    }
}
