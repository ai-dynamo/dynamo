// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod context;
mod cuda;
mod memcpy;
mod nixl;
pub mod remote;
mod strategy;

use super::*;

use crate::block_manager::storage::{
    DeviceStorage, DiskStorage, PinnedStorage, SystemStorage,
    nixl::{NixlRegisterableStorage, NixlStorage},
};

use nixl_sys::NixlDescriptor;
use nixl_sys::XferOp::{Read, Write};
use std::ops::Range;
use tokio::sync::oneshot;

pub use crate::block_manager::storage::{CudaAccessible, Local, Remote};
pub use async_trait::async_trait;
pub use context::{PoolConfig, RemoteStorageConfig, RemoteTransferContext, TransferContext};
pub use remote::{
    DiskKey, ObjectKey, RemoteBlockDescriptor, RemoteBlockMetadata, RemoteKey,
    RemoteStorageKind, RemoteTransferDirection, RemoteTransferHandle, RemoteTransferPipeline,
    RemoteTransferStrategy,
};

/// A block that can be the target of a write
pub trait Writable {}

/// A block that can be the source of a read
pub trait Readable {}

pub trait Mutable: Readable + Writable {}

pub trait Immutable: Readable {}

#[derive(Debug)]
pub enum BlockTarget {
    Source,
    Destination,
}

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("Builder configuration error: {0}")]
    BuilderError(String),
    #[error("Transfer execution failed: {0}")]
    ExecutionError(String),
    #[error("Incompatible block types provided: {0}")]
    IncompatibleTypes(String),
    #[error("Mismatched source/destination counts: {0} sources, {1} destinations")]
    CountMismatch(usize, usize),
    #[error("Block operation failed: {0}")]
    BlockError(#[from] BlockError),
    // TODO: Add NIXL specific errors
    #[error("No blocks provided")]
    NoBlocksProvided,

    #[error("Mismatched {0:?} block set index: {1} != {2}")]
    MismatchedBlockSetIndex(BlockTarget, usize, usize),

    #[error("Mismatched {0:?} worker ID: {1} != {2}")]
    MismatchedWorkerID(BlockTarget, usize, usize),

    #[error("Transfer was cancelled")]
    Cancelled,

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl TransferError {
    /// Check if this error represents a cancellation.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, TransferError::Cancelled)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixlTransfer {
    Read,
    Write,
}

impl NixlTransfer {
    pub fn as_xfer_op(&self) -> nixl_sys::XferOp {
        match self {
            NixlTransfer::Read => Read,
            NixlTransfer::Write => Write,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaTransferMode {
    /// Use the custom CUDA kernel for G1 <-> G2 transfers
    Custom,
    /// Use the default CUDA async memcpy for G1 <-> G2 transfers
    Default,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    Memcpy,
    CudaAsyncH2D,
    CudaAsyncD2H,
    CudaAsyncD2D,
    CudaBlockingH2D,
    CudaBlockingD2H,
    Nixl(NixlTransfer),
    Invalid,
}

/// Trait for determining the transfer strategy for writing from a local
/// source to a target destination which could be local or remote
pub trait WriteToStrategy<Target> {
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

/// Trait for determining the transfer strategy for reading from a
/// `Source` which could be local or remote into `Self` which must
/// be both local and writable.
pub trait ReadFromStrategy<Source> {
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

impl<RB: ReadableBlock, WB: WritableBlock> WriteToStrategy<WB> for RB
where
    <RB as StorageTypeProvider>::StorageType:
        Local + WriteToStrategy<<WB as StorageTypeProvider>::StorageType>,
{
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        <<RB as StorageTypeProvider>::StorageType as WriteToStrategy<
            <WB as StorageTypeProvider>::StorageType,
        >>::write_to_strategy()
    }
}

impl<WB: WritableBlock, RB: ReadableBlock> ReadFromStrategy<RB> for WB
where
    <RB as StorageTypeProvider>::StorageType: Remote,
    <WB as StorageTypeProvider>::StorageType: NixlRegisterableStorage,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

#[inline]
fn resolve_cuda_transfer_mode(
    base_strategy: TransferStrategy,
    is_contiguous: bool,
) -> CudaTransferMode {
    match base_strategy {
        TransferStrategy::CudaAsyncH2D => {
            if is_contiguous {
                CudaTransferMode::Default
            } else {
                CudaTransferMode::Custom
            }
        }
        TransferStrategy::CudaAsyncD2H => {
            if is_contiguous {
                CudaTransferMode::Default
            } else {
                CudaTransferMode::Custom
            }
        }
        other => panic!(
            "resolve_cuda_strategy called with non-CUDA strategy: {:?}",
            other
        ),
    }
}

pub fn handle_local_transfer<RB, WB>(
    sources: &[RB],
    targets: &mut [WB],
    ctx: Arc<TransferContext>,
) -> Result<oneshot::Receiver<()>, TransferError>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    WB: WritableBlock,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    // Check for empty slices and length mismatch early
    if sources.is_empty() && targets.is_empty() {
        tracing::warn!(
            "handle_local_transfer called with both sources and targets empty, skipping transfer"
        );
        let (tx, rx) = oneshot::channel();
        tx.send(()).unwrap();
        return Ok(rx);
    }

    if sources.len() != targets.len() {
        return Err(TransferError::CountMismatch(sources.len(), targets.len()));
    }

    let (tx, rx) = oneshot::channel();

    match RB::write_to_strategy() {
        TransferStrategy::Memcpy => {
            for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                // TODO: Unlike all other transfer strategies, this is fully blocking.
                // We probably want some sort of thread pool to handle these.
                memcpy::copy_block(src, dst)?;
            }

            tx.send(()).unwrap();
            Ok(rx)
        }
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {
            tracing::debug!(
                "Transfer: Using CUDA strategy: {:?}",
                RB::write_to_strategy()
            );

            if RB::write_to_strategy() == TransferStrategy::CudaAsyncH2D
                || RB::write_to_strategy() == TransferStrategy::CudaAsyncD2H
            {
                let is_contiguous = sources[0].block_data().is_fully_contiguous()
                    && targets[0].block_data().is_fully_contiguous();
                let transfer_mode =
                    resolve_cuda_transfer_mode(RB::write_to_strategy(), is_contiguous);

                match transfer_mode {
                    CudaTransferMode::Custom => {
                        let selected_stream = ctx.stream();
                        cuda::copy_blocks_with_customized_kernel(
                            sources,
                            targets,
                            selected_stream.as_ref(),
                            &ctx,
                        )?;
                    }
                    CudaTransferMode::Default => {
                        for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                            cuda::copy_block(
                                src,
                                dst,
                                ctx.stream().as_ref(),
                                RB::write_to_strategy(),
                            )?;
                        }
                    }
                }
                ctx.cuda_event(tx)?;

                Ok(rx)
            } else {
                // Fall back to individual copy for D2Dblocks
                for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                    cuda::copy_block(src, dst, ctx.stream().as_ref(), RB::write_to_strategy())?;
                }
                ctx.cuda_event(tx)?;
                Ok(rx)
            }
        }
        TransferStrategy::Nixl(transfer_type) => {
            let transfer_fut = nixl::write_blocks_to(sources, targets, &ctx, transfer_type)?;

            ctx.async_rt_handle().spawn(async move {
                transfer_fut.await;
                tx.send(()).unwrap();
            });
            Ok(rx)
        }
        _ => Err(TransferError::IncompatibleTypes(format!(
            "Unsupported copy strategy: {:?}",
            RB::write_to_strategy()
        ))),
    }
}

pub trait WriteTo<Target> {
    fn write_to(
        &self,
        dst: &mut Vec<Target>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>;
}

impl<RB, WB, L: LocalityProvider> WriteTo<WB> for Vec<RB>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
    RB: BlockDataProvider<Locality = L>,
    WB: WritableBlock + BlockDataProviderMut<Locality = L>,
{
    fn write_to(
        &self,
        dst: &mut Vec<WB>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError> {
        L::handle_transfer(self, dst, ctx)
    }
}

use tokio_util::sync::CancellationToken;

/// Handle a transfer involving remote storage (object storage, remote disk).
///
/// This is the remote storage counterpart to `handle_local_transfer`.
/// Supports both direct transfers and full pipelines with bounce buffers.
///
/// # Transfer Modes
///
/// ## Direct Mode (Host <-> Remote)
/// - **Onboard**: Remote -> Host
/// - **Offload**: Host -> Remote
///
/// ## Pipeline Mode (Device <-> Bounce <-> Remote)
/// - **Onboard**: Remote -> Host(bounce) -> Device
/// - **Offload**: Device -> Host(bounce) -> Remote
///
/// # Remote Storage Types
///
/// - **Object Storage**:
/// - **Remote Disk**:
///
/// # Arguments
///
/// * `pipeline` - The transfer pipeline (direct or with bounce)
/// * `ctx` - Remote transfer context (config, registry, base context)
/// * `host_blocks` - Host blocks (bounce buffers for pipeline mode, source/dest for direct)
/// * `device_blocks` - Device blocks (only for pipeline mode with bounce)
/// * `cancel_token` - Optional cancellation token
///
/// # Example
///
/// ```rust,ignore
/// // Direct offload: Host -> Object Storage
/// let pipeline = RemoteTransferPipeline::offload_direct(descriptors);
/// let handle = handle_remote_transfer(
///     pipeline,
///     ctx.clone(),
///     &host_blocks,
///     None,  // No device blocks for direct
///     None,  // New cancel token
/// )?;
/// handle.wait().await?;
///
/// // Full pipeline: Remote -> Bounce -> Device
/// let pipeline = RemoteTransferPipeline::onboard_with_bounce(
///     descriptors,
///     bounce_block_ids,
///     device_block_ids,
/// );
/// let handle = handle_remote_transfer(
///     pipeline,
///     ctx.clone(),
///     &host_blocks,
///     Some(&device_blocks),
///     Some(request_cancel.child_token()),
/// )?;
/// handle.wait().await?;
/// ```
pub fn handle_remote_transfer<HB, DB>(
    pipeline: RemoteTransferPipeline,
    ctx: Arc<RemoteTransferContext>,
    host_blocks: &[HB],
    device_blocks: Option<&[DB]>,
    cancel_token: Option<CancellationToken>,
) -> Result<RemoteTransferHandle, TransferError>
where
    HB: ReadableBlock + WritableBlock + Local + Clone + Send + Sync + 'static,
    DB: ReadableBlock + WritableBlock + Local + Clone + Send + Sync + 'static,
    <HB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <DB as StorageTypeProvider>::StorageType: NixlDescriptor,
    HB: WriteToStrategy<DB>,
    DB: WriteToStrategy<HB>,
{
    let cancel_token = cancel_token.unwrap_or_else(CancellationToken::new);

    let descriptors = pipeline.descriptors();
    if descriptors.is_empty() {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(()));
        return Ok(RemoteTransferHandle::new(rx, cancel_token));
    }

    let expected_kind = descriptors[0].kind();
    if !descriptors.iter().all(|d| d.kind() == expected_kind) {
        return Err(TransferError::IncompatibleTypes(
            "All remote descriptors must have the same storage kind".to_string(),
        ));
    }

    let (tx, rx) = oneshot::channel();

    match pipeline {
        RemoteTransferPipeline::Direct {
            direction,
            remote_descriptors,
        } => {
            handle_direct_remote_transfer(
                direction,
                expected_kind,
                remote_descriptors,
                host_blocks,
                ctx,
                cancel_token.clone(),
                tx,
            )?;
        }

        RemoteTransferPipeline::WithBounce {
            direction,
            remote_descriptors,
            bounce_block_ids,
            device_block_ids,
        } => {
            let device_blocks = device_blocks.ok_or_else(|| {
                TransferError::IncompatibleTypes(
                    "Device blocks required for pipeline with bounce".to_string(),
                )
            })?;

            handle_pipeline_remote_transfer(
                direction,
                expected_kind,
                remote_descriptors,
                host_blocks,
                device_blocks,
                bounce_block_ids,
                device_block_ids,
                ctx,
                cancel_token.clone(),
                tx,
            )?;
        }
    }

    Ok(RemoteTransferHandle::new(rx, cancel_token))
}

/// Handle direct transfer (Host <-> Remote)
fn handle_direct_remote_transfer<HB>(
    direction: RemoteTransferDirection,
    kind: RemoteStorageKind,
    descriptors: Vec<remote::RemoteBlockDescriptor>,
    host_blocks: &[HB],
    ctx: Arc<RemoteTransferContext>,
    cancel_token: CancellationToken,
    tx: oneshot::Sender<Result<(), TransferError>>,
) -> Result<(), TransferError>
where
    HB: ReadableBlock + WritableBlock + Local + Clone + Send + Sync + 'static,
    <HB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    // Clone host blocks for async ownership
    let host_blocks: Vec<HB> = host_blocks.to_vec();

    // Clone the handle before spawning
    let rt_handle = ctx.async_rt_handle().clone();

    rt_handle.spawn(async move {
        let result = nixl::execute_remote_transfer(
            direction,
            kind,
            &descriptors,
            &host_blocks,
            &ctx,
            &cancel_token,
        )
        .await;

        let _ = tx.send(result);
    });

    Ok(())
}

/// Handle full pipeline transfer (Device <-> Bounce <-> Remote)
fn handle_pipeline_remote_transfer<HB, DB>(
    direction: RemoteTransferDirection,
    kind: RemoteStorageKind,
    descriptors: Vec<remote::RemoteBlockDescriptor>,
    host_blocks: &[HB],
    device_blocks: &[DB],
    bounce_ids: Vec<usize>,
    device_ids: Vec<usize>,
    ctx: Arc<RemoteTransferContext>,
    cancel_token: CancellationToken,
    tx: oneshot::Sender<Result<(), TransferError>>,
) -> Result<(), TransferError>
where
    HB: ReadableBlock + WritableBlock + Local + Clone + Send + Sync + 'static,
    DB: ReadableBlock + WritableBlock + Local + Clone + Send + Sync + 'static,
    <HB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <DB as StorageTypeProvider>::StorageType: NixlDescriptor,
    HB: WriteToStrategy<DB>,
    DB: WriteToStrategy<HB>,
{
    // Clone blocks for async ownership
    let bounce_blocks: Vec<HB> = bounce_ids
        .iter()
        .map(|&id| host_blocks.get(id).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            TransferError::ExecutionError("Invalid bounce block ID".to_string())
        })?;

    let target_device_blocks: Vec<DB> = device_ids
        .iter()
        .map(|&id| device_blocks.get(id).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            TransferError::ExecutionError("Invalid device block ID".to_string())
        })?;

    let base_ctx = ctx.base().clone();
    let rt_handle = ctx.async_rt_handle().clone();

    rt_handle.spawn(async move {
        let result = match direction {
            RemoteTransferDirection::Onboard => {
                // Remote → Bounce → Device
                execute_onboard_pipeline(
                    kind,
                    descriptors,
                    bounce_blocks,
                    target_device_blocks,
                    ctx,
                    base_ctx,
                    cancel_token,
                )
                .await
            }
            RemoteTransferDirection::Offload => {
                // Device → Bounce → Remote
                execute_offload_pipeline(
                    kind,
                    descriptors,
                    bounce_blocks,
                    target_device_blocks,
                    ctx,
                    base_ctx,
                    cancel_token,
                )
                .await
            }
        };
        let _ = tx.send(result);
    });

    Ok(())
}

/// Execute onboard pipeline: Remote -> Bounce -> Device
async fn execute_onboard_pipeline<HB, DB>(
    kind: RemoteStorageKind,
    descriptors: Vec<remote::RemoteBlockDescriptor>,
    bounce_blocks: Vec<HB>,
    mut device_blocks: Vec<DB>,
    ctx: Arc<RemoteTransferContext>,
    base_ctx: Arc<TransferContext>,
    cancel_token: CancellationToken,
) -> Result<(), TransferError>
where
    HB: ReadableBlock + WritableBlock + Local + Clone + Send + Sync + 'static,
    DB: WritableBlock + Clone + Send + Sync + 'static,
    <HB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <DB as StorageTypeProvider>::StorageType: NixlDescriptor,
    HB: WriteToStrategy<DB>,
{
    nixl::execute_remote_transfer(
        RemoteTransferDirection::Onboard,
        kind,
        &descriptors,
        &bounce_blocks,
        &ctx,
        &cancel_token,
    )
    .await?;

    if cancel_token.is_cancelled() {
        return Err(TransferError::Cancelled);
    }

    let rx = handle_local_transfer(&bounce_blocks, &mut device_blocks, base_ctx)?;
    rx.await.map_err(|_| {
        TransferError::ExecutionError("H2D transfer channel closed".to_string())
    })?;

    Ok(())
}

/// Execute offload pipeline: Device → Bounce → Remote
async fn execute_offload_pipeline<HB, DB>(
    kind: RemoteStorageKind,
    descriptors: Vec<remote::RemoteBlockDescriptor>,
    mut bounce_blocks: Vec<HB>,
    device_blocks: Vec<DB>,
    ctx: Arc<RemoteTransferContext>,
    base_ctx: Arc<TransferContext>,
    cancel_token: CancellationToken,
) -> Result<(), TransferError>
where
    HB: WritableBlock + ReadableBlock + Local + Clone + Send + Sync + 'static,
    DB: ReadableBlock + Local + Clone + Send + Sync + 'static,
    <HB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <DB as StorageTypeProvider>::StorageType: NixlDescriptor,
    DB: WriteToStrategy<HB>,
{
    // Phase 1: Device → Bounce
    let rx = handle_local_transfer(&device_blocks, &mut bounce_blocks, base_ctx)?;
    rx.await.map_err(|_| {
        TransferError::ExecutionError("D2H transfer channel closed".to_string())
    })?;

    // Check cancellation between phases
    if cancel_token.is_cancelled() {
        return Err(TransferError::Cancelled);
    }

    // Phase 2: Bounce → Remote
    nixl::execute_remote_transfer(
        RemoteTransferDirection::Offload,
        kind,
        &descriptors,
        &bounce_blocks,
        &ctx,
        &cancel_token,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2D
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
    }
}
