// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::utils::RemoteTransferRequest;
use super::*;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    BasicMetadata, Storage,
    block::{
        Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock, WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{
            TransferContext, WriteTo, WriteToStrategy, checksum::compute_checksum,
            remote::RemoteTransferPipeline,
        },
    },
    config::RemoteTransferContext,
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    distributed::vllm::g4_checksum_enabled,
    offload::MAX_TRANSFER_BATCH_SIZE,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
};
use tokio_util::sync::CancellationToken;

use anyhow::Result;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use std::{any::Any, sync::Arc};

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

static G4_PIPELINE_CHUNK_SIZE: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_PIPELINE_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
});

fn g4_pipeline_chunk_size() -> usize {
    *G4_PIPELINE_CHUNK_SIZE
}

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

    pub async fn execute_batched_transfer(
        &self,
        handler: &BlockTransferHandler,
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

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
/// Also handles remote storage transfers (G4 object storage, remote disk) when configured.
#[derive(Clone)]
pub struct BlockTransferHandler {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    /// Scheduler client for completion tracking.
    /// Public so `RemoteTransferDispatch` can propagate connector_req
    /// through the scheduler's completion system.
    pub scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    remote_context: Option<Arc<RemoteTransferContext>>,
    cancel_token: CancellationToken,
}

impl BlockTransferHandler {
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

    /// Execute transfer with batching to prevent resource exhaustion
    pub async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    /// Execute transfer directly without batching (used by the batcher)
    #[tracing::instrument(level = "debug", skip_all, fields(
        num_blocks = request.blocks().len(),
        from_pool = ?request.from_pool(),
        to_pool = ?request.to_pool(),
    ))]
    pub async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
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

    /// Execute a remote transfer (G4 object storage or remote disk).
    ///
    /// Handles both onboard (remote -> host -> device) and offload (device -> host -> remote)
    /// transfers, with optional checksum logging when G4 validation is enabled.
    #[tracing::instrument(level = "info", skip_all, fields(
        request_id = %request.request_id,
        operation_id = %request.operation_id,
        otel.name = "kvbm.remote_transfer",
    ))]
    pub async fn execute_remote_transfer(&self, request: RemoteTransferRequest) -> Result<()> {
        let remote_ctx = self
            .remote_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Remote transfer context not configured"))?;

        let host_blocks = self
            .host
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Host blocks required for remote transfers"))?;

        let pipeline = request.pipeline.to_pipeline();
        let num_blocks = pipeline.num_blocks();
        let direction = pipeline.direction();
        let is_onboard = direction.is_onboard();
        use crate::block_manager::config::DISK_FLAG_GDS_WRITE;
        let backend = match remote_ctx.config() {
            crate::block_manager::config::RemoteStorageConfig::Object { .. } => "object",
            crate::block_manager::config::RemoteStorageConfig::Disk { transfer_flags, .. } => {
                if transfer_flags & DISK_FLAG_GDS_WRITE != 0 {
                    "gds_mt"
                } else {
                    "posix"
                }
            }
        };

        tracing::debug!(
            target: "kvbm-g4",
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            num_blocks,
            backend = backend,
            direction = ?direction,
            "executing remote transfer"
        );

        // Get bounce buffer block IDs (host blocks used as staging)
        let bounce_ids = pipeline
            .bounce_block_ids()
            .ok_or_else(|| anyhow::anyhow!("Remote transfer requires bounce buffer block IDs"))?;

        // Get the host blocks for this transfer
        let bounce_blocks: Vec<LocalBlockData<PinnedStorage>> = bounce_ids
            .iter()
            .map(|&idx| host_blocks[idx].clone())
            .collect();

        let chunk_size = g4_pipeline_chunk_size();

        if chunk_size > 0 && num_blocks > chunk_size && is_onboard {
            self.execute_remote_transfer_chunked(
                &request,
                &pipeline,
                &bounce_blocks,
                bounce_ids,
                remote_ctx,
                backend,
                chunk_size,
            )
            .await?;
        } else {
            let transfer_start = std::time::Instant::now();

            if !is_onboard && g4_checksum_enabled() {
                self.log_checksums_for_offload(&bounce_blocks, &pipeline, &request.request_id);
            }

            {
                use tracing::Instrument;
                let r2h_span = tracing::info_span!(
                    "r2h_transfer",
                    otel.name = "kvbm.r2h",
                    request_id = %request.request_id,
                    num_blocks,
                    backend,
                );

                pipeline
                    .execute(&bounce_blocks, remote_ctx.as_ref(), &self.cancel_token)
                    .instrument(r2h_span)
                    .await
                    .map_err(|e| anyhow::anyhow!("Remote transfer failed: {}", e))?;
            }

            let r2h_elapsed = transfer_start.elapsed();

            if is_onboard && g4_checksum_enabled() {
                self.log_checksums_for_onboard(&bounce_blocks, &pipeline, &request.request_id);
            }

            if is_onboard {
                tracing::info!(
                    target: "kvbm-diag",
                    request_id = %request.request_id,
                    num_blocks,
                    r2h_ms = r2h_elapsed.as_millis(),
                    backend,
                    "R2H phase complete (serial path)"
                );
            }

            if let Some(device_ids) = pipeline.device_block_ids() {
                if !device_ids.is_empty() {
                    let block_pairs: Vec<(usize, usize)> = if is_onboard {
                        bounce_ids
                            .iter()
                            .copied()
                            .zip(device_ids.iter().copied())
                            .collect()
                    } else {
                        vec![]
                    };

                    if !block_pairs.is_empty() {
                        use tracing::Instrument;
                        let h2d_span = tracing::info_span!(
                            "h2d_transfer",
                            otel.name = "kvbm.h2d",
                            request_id = %request.request_id,
                            num_blocks = block_pairs.len(),
                        );

                        let local_request = BlockTransferRequest {
                            from_pool: if is_onboard { Host } else { Device },
                            to_pool: if is_onboard { Device } else { Host },
                            blocks: block_pairs,
                            connector_req: None,
                            sequence_hashes: None,
                        };
                        self.execute_transfer(local_request).instrument(h2d_span).await?;

                        if is_onboard {
                            tracing::info!(
                                target: "kvbm-diag",
                                request_id = %request.request_id,
                                num_blocks,
                                r2h_ms = r2h_elapsed.as_millis(),
                                h2d_ms = (transfer_start.elapsed() - r2h_elapsed).as_millis(),
                                total_ms = transfer_start.elapsed().as_millis(),
                                backend,
                                "R2H+H2D complete (serial path)"
                            );
                        }
                    }
                } else if is_onboard {
                    tracing::warn!(
                        target: "kvbm-diag",
                        request_id = %request.request_id,
                        num_blocks,
                        "H2D skipped: device_block_ids is empty"
                    );
                }
            } else if is_onboard {
                tracing::warn!(
                    target: "kvbm-diag",
                    request_id = %request.request_id,
                    num_blocks,
                    "H2D skipped: no device_block_ids in pipeline (Direct mode)"
                );
            }
        }

        tracing::debug!(
            target: "kvbm-g4",
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            num_blocks,
            backend = backend,
            "remote transfer completed"
        );

        Ok(())
    }

    /// Chunked pipelined remote transfer for onboard.
    ///
    /// Splits the block set into chunks and:
    /// 1. Spawns concurrent R2H (remote→host) tasks per chunk to saturate NFS bandwidth
    /// 2. Pipelines H2D (host→device) as each chunk's R2H completes
    ///
    /// This overlaps R2H of later chunks with H2D of earlier chunks, and enables
    /// concurrent NFS reads that would otherwise be serialized in a single NIXL request.
    #[tracing::instrument(level = "info", skip_all, fields(
        request_id = %request.request_id,
        chunk_size,
        backend,
        otel.name = "kvbm.remote_transfer_chunked",
    ))]
    async fn execute_remote_transfer_chunked(
        &self,
        request: &RemoteTransferRequest,
        pipeline: &RemoteTransferPipeline,
        bounce_blocks: &[LocalBlockData<PinnedStorage>],
        bounce_ids: &[usize],
        remote_ctx: &Arc<RemoteTransferContext>,
        backend: &str,
        chunk_size: usize,
    ) -> Result<()> {
        let num_blocks = pipeline.num_blocks();
        let descriptors = pipeline.descriptors();
        let device_ids = pipeline.device_block_ids();
        let num_chunks = (num_blocks + chunk_size - 1) / chunk_size;

        tracing::info!(
            target: "kvbm-g4",
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            num_blocks,
            chunk_size,
            num_chunks,
            backend,
            "starting chunked R2H→H2D pipeline"
        );

        let start_time = std::time::Instant::now();

        // Channel carries (chunk_index, start, end, result) from R2H tasks to the
        // H2D consumer loop running on this task.
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<(
            usize,
            usize,
            usize,
            std::result::Result<(), anyhow::Error>,
        )>(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(num_blocks);

            let chunk_descs = descriptors[start..end].to_vec();
            let chunk_bounce = bounce_blocks[start..end].to_vec();
            let ctx = Arc::clone(remote_ctx);
            let cancel = self.cancel_token.clone();
            let done_tx = done_tx.clone();

            tokio::spawn(async move {
                let sub_pipeline = RemoteTransferPipeline::onboard_direct(chunk_descs);
                let result = sub_pipeline
                    .execute(&chunk_bounce, &ctx, &cancel)
                    .await
                    .map_err(|e| anyhow::anyhow!("R2H chunk {}: {}", chunk_idx, e));
                let _ = done_tx.send((chunk_idx, start, end, result)).await;
            });
        }
        drop(done_tx);

        let mut r2h_completed = 0usize;
        let mut r2h_error: Option<anyhow::Error> = None;
        let mut h2d_tasks = tokio::task::JoinSet::new();
        let handler = self.clone();

        while let Some((chunk_idx, start, end, result)) = done_rx.recv().await {
            match result {
                Ok(()) => {
                    r2h_completed += 1;
                    tracing::debug!(
                        target: "kvbm-g4",
                        request_id = %request.request_id,
                        chunk_idx,
                        chunk_blocks = end - start,
                        r2h_completed,
                        num_chunks,
                        elapsed_ms = start_time.elapsed().as_millis(),
                        "chunk R2H done → spawning H2D"
                    );

                    if let Some(all_device_ids) = device_ids {
                        let block_pairs: Vec<(usize, usize)> = bounce_ids[start..end]
                            .iter()
                            .copied()
                            .zip(all_device_ids[start..end].iter().copied())
                            .collect();

                        if !block_pairs.is_empty() {
                            let h2d_handler = handler.clone();
                            h2d_tasks.spawn(async move {
                                let local_request = BlockTransferRequest {
                                    from_pool: Host,
                                    to_pool: Device,
                                    blocks: block_pairs,
                                    connector_req: None,
                                    sequence_hashes: None,
                                };
                                h2d_handler
                                    .execute_transfer(local_request)
                                    .await
                                    .map_err(|e| anyhow::anyhow!("H2D chunk {}: {}", chunk_idx, e))
                            });
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        target: "kvbm-g4",
                        request_id = %request.request_id,
                        chunk_idx,
                        error = %e,
                        "chunk R2H failed"
                    );
                    if r2h_error.is_none() {
                        r2h_error = Some(e);
                    }
                }
            }
        }

        // All R2H done (channel drained). Now wait for any remaining H2D tasks.
        while let Some(result) = h2d_tasks.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::error!(target: "kvbm-g4", error = %e, "H2D chunk failed");
                    if r2h_error.is_none() {
                        r2h_error = Some(e);
                    }
                }
                Err(e) => {
                    tracing::error!(target: "kvbm-g4", error = %e, "H2D task panicked");
                    if r2h_error.is_none() {
                        r2h_error = Some(anyhow::anyhow!("H2D task panicked: {}", e));
                    }
                }
            }
        }

        if let Some(e) = r2h_error {
            return Err(e);
        }

        tracing::info!(
            target: "kvbm-g4",
            request_id = %request.request_id,
            num_blocks,
            num_chunks,
            elapsed_ms = start_time.elapsed().as_millis(),
            "chunked R2H→H2D pipeline complete"
        );

        Ok(())
    }

    /// Log checksums for blocks being offloaded to remote storage.
    fn log_checksums_for_offload(
        &self,
        bounce_blocks: &[LocalBlockData<PinnedStorage>],
        pipeline: &RemoteTransferPipeline,
        request_id: &str,
    ) {
        use crate::block_manager::block::data::BlockDataViews;

        let descriptors = pipeline.descriptors();
        for (block, desc) in bounce_blocks.iter().zip(descriptors.iter()) {
            let remote_key = format!("{}/{}", desc.key().location(), desc.key().key_str());
            let seq_hash = desc.sequence_hash().unwrap_or(0);

            // Get block view and compute checksum from raw pointer
            match block.local_block_view() {
                Ok(view) => {
                    let checksum = unsafe {
                        let ptr = view.as_ptr();
                        let size = view.size();
                        let slice = std::slice::from_raw_parts(ptr, size);
                        compute_checksum(slice)
                    };

                    tracing::info!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        remote_key,
                        checksum,
                        phase = "pre-offload",
                        "checksum computed before remote write"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        error = %e,
                        "failed to get block view for checksum"
                    );
                }
            }
        }
    }

    /// Log checksums for blocks being onboarded from remote storage.
    fn log_checksums_for_onboard(
        &self,
        bounce_blocks: &[LocalBlockData<PinnedStorage>],
        pipeline: &RemoteTransferPipeline,
        request_id: &str,
    ) {
        use crate::block_manager::block::data::BlockDataViews;

        let descriptors = pipeline.descriptors();
        for (block, desc) in bounce_blocks.iter().zip(descriptors.iter()) {
            let remote_key = format!("{}/{}", desc.key().location(), desc.key().key_str());
            let seq_hash = desc.sequence_hash().unwrap_or(0);

            // Get block view and compute checksum from raw pointer
            match block.local_block_view() {
                Ok(view) => {
                    let checksum = unsafe {
                        let ptr = view.as_ptr();
                        let size = view.size();
                        let slice = std::slice::from_raw_parts(ptr, size);
                        compute_checksum(slice)
                    };

                    tracing::info!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        remote_key,
                        checksum,
                        phase = "post-onboard",
                        "checksum computed after remote read"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        error = %e,
                        "failed to get block view for checksum"
                    );
                }
            }
        }
    }
}

#[async_trait]
impl Handler for BlockTransferHandler {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        // Try to parse as RemoteTransferRequest first, then fall back to BlockTransferRequest
        let result = if let Ok(remote_request) =
            serde_json::from_slice::<RemoteTransferRequest>(&message.data[0])
        {
            // Handle remote transfer (G4 object storage)
            let operation_id = remote_request.operation_id;

            tracing::debug!(
                target: "kvbm-g4",
                request_id = %remote_request.request_id,
                operation_id = %operation_id,
                num_blocks = remote_request.num_blocks(),
                is_onboard = remote_request.is_onboard(),
                "handling remote transfer request"
            );

            if let Some(connector_req) = remote_request.connector_req.clone() {
                let client = self
                    .scheduler_client
                    .as_ref()
                    .expect("scheduler client is required")
                    .clone();

                let handle = client.schedule_transfer(connector_req).await?;
                assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

                match self.execute_remote_transfer(remote_request).await {
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
                self.execute_remote_transfer(remote_request).await
            }
        } else {
            // Handle local block transfer
            let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

            if let Some(req) = request.connector_req.take() {
                let operation_id = req.uuid;

                tracing::debug!(
                    request_id = %req.request_id,
                    operation_id = %operation_id,
                    "scheduling transfer"
                );

                let client = self
                    .scheduler_client
                    .as_ref()
                    .expect("scheduler client is required")
                    .clone();

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
            }
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}
