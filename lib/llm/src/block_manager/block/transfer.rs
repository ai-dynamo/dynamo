// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod context;
mod cuda;
mod memcpy;
mod nixl;
mod strategy;

use super::*;

use crate::block_manager::block::{
    BlockDataExt, BlockDataProvider, BlockDataProviderMut, BlockError, StorageTypeProvider,
};
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
pub use context::{PoolConfig, TransferContext};

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

    #[error(transparent)]
    Other(#[from] anyhow::Error),
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
            let src_block_type = sources[0].block_data();
            let src_contig = src_block_type.is_fully_contiguous();
            let src_disk = matches!(src_block_type.storage_type(), StorageType::Disk(_));
            let target_block_type = targets[0].block_data();
            let target_contig = target_block_type.is_fully_contiguous();
            let target_disk = matches!(target_block_type.storage_type(), StorageType::Disk(_));

            if (!src_contig && !src_disk && target_contig && target_disk)
                || (src_contig && src_disk && !target_contig && !target_disk)
            {
                // This is done to avoid small NIXL transfers done directly from
                // GPU to disk. We use temporary GPU buffers to rearrange layout to
                // FullyContiguous on egress, and to redistribute to LayerSeparate
                // on ingress.

                tracing::debug!(
                    "Using temporary GPU buffer for layout conversion: {} to {}",
                    if src_contig {
                        "contig source"
                    } else {
                        "non-contig source"
                    },
                    if target_contig {
                        "contig target"
                    } else {
                        "non-contig target"
                    },
                );

                // Use one temporary GPU buffer per source and target block pairs.
                let mut temp_bufs = Vec::with_capacity(targets.len());
                for _ in 0..targets.len() {
                    match ctx.acquire_temp_device_buffer() {
                        Ok(temp_buf) => {
                            temp_bufs.push(temp_buf);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to acquire temporary GPU buffer: {}, falling back to direct transfer",
                                e
                            );
                            break;
                        }
                    }
                }

                if temp_bufs.len() == targets.len() {
                    let mut dest_slice: Vec<_> =
                        temp_bufs.iter().map(|x| x.data.to_owned()).collect();
                    if target_disk {
                        // consodliate discontigious KV cache data to contigious file data
                        match cuda::copy_blocks_with_customized_kernel(
                            sources,
                            &mut dest_slice,
                            ctx.stream().as_ref(),
                            &ctx,
                        ) {
                            Ok(_) => {
                                let transfer_fut = nixl::write_blocks_to(
                                    &dest_slice,
                                    targets,
                                    &ctx,
                                    transfer_type,
                                )?;
                                ctx.async_rt_handle().spawn(async move {
                                    let temp_bufs = temp_bufs;
                                    transfer_fut.await;
                                    // temp_buffer will be automatically returned to pool when dropped
                                    tx.send(()).unwrap();
                                    drop(temp_bufs);
                                });
                                return Ok(rx);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Consolidation into temp buffer failed: {}, falling back to direct transfer",
                                    e
                                );
                            }
                        }
                    } else {
                        // src_disk == true, here we distribute to discontigious KV cache from contigious file data
                        let transfer_fut =
                            nixl::write_blocks_to(&sources, &mut dest_slice, &ctx, transfer_type)?;
                        let (src_addresses, dst_addresses, dims) =
                            cuda::get_address_pairs(&dest_slice, targets)?;

                        let ctx2 = ctx.clone();
                        ctx.async_rt_handle().spawn(async move {
                            let temp_bufs = temp_bufs;
                            transfer_fut.await;

                            let _context_guard = ctx2.stream().context().bind_to_thread();
                            match cuda::copy_pairs_with_customized_kernel(
                                src_addresses,
                                dst_addresses,
                                dims,
                                ctx2.stream().as_ref(),
                                &ctx2,
                            ) {
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::warn!("Scattering from temp buffer failed: {}", e);
                                    drop(tx);
                                    return;
                                }
                            }

                            // temp_buffer will be automatically returned to pool when dropped
                            tx.send(()).unwrap();
                            drop(temp_bufs);
                        });

                        return Ok(rx);
                    }
                }
            }

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
