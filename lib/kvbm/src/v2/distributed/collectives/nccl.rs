// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NCCL-based collective operations for GPU-to-GPU communication.
//!
//! This module provides [`NcclCollectives`], an implementation of [`CollectiveOps`]
//! that uses NVIDIA NCCL for efficient GPU collective communication.
//!
//! # Construction Paths
//!
//! NCCL communicators can be obtained via two paths:
//!
//! ## Path A: Bootstrap (tests and standalone Rust apps)
//!
//! Use [`NcclCollectives::from_bootstrap`] when creating communicators from scratch:
//!
//! ```rust,ignore
//! let bootstrap = NcclBootstrap::generate(world_size)?;
//! // ... distribute bootstrap to other ranks ...
//! let collectives = NcclCollectives::from_bootstrap(
//!     &bootstrap,
//!     rank,
//!     cuda_context,
//!     event_registrar,
//!     layout_resolver,
//! )?;
//! ```
//!
//! ## Path B: Borrowed handles (production with PyTorch/vLLM/TensorRT-LLM)
//!
//! Use [`NcclCollectives::from_borrowed`] when an external runtime provides the communicator:
//!
//! ```rust,ignore
//! // In Python: comm_ptr = dist._get_default_group()._get_backend()._nccl_comm.as_int()
//! let collectives = unsafe {
//!     NcclCollectives::from_borrowed(
//!         comm_ptr,
//!         stream_ptr,
//!         rank,
//!         world_size,
//!         cuda_context,
//!         event_registrar,
//!         layout_resolver,
//!     )
//! };
//! ```
//!
//! # Thread Safety
//!
//! NCCL operations are thread-safe when each thread uses its own stream. This
//! implementation uses a dedicated NCCL stream per `NcclCollectives` instance.

use std::ops::Range;
use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use cudarc::nccl::sys::{
    ncclBcast, ncclComm_t, ncclCommDestroy, ncclDataType_t, ncclGroupEnd, ncclGroupStart,
};
use dynamo_nova::events::LocalEventSystem;

use crate::physical::layout::PhysicalLayout;
use crate::physical::transfer::TransferCompleteNotification;
use crate::v2::BlockId;
use crate::v2::logical::LogicalLayoutHandle;

use super::CollectiveOps;
use super::bootstrap::{NcclBootstrap, check_nccl_result};

/// Trait for resolving logical layout handles to physical layouts.
///
/// This trait decouples [`NcclCollectives`] from [`PhysicalWorker`], allowing
/// the collective operations to work with any layout resolution strategy.
pub trait LayoutResolver: Send + Sync {
    /// Resolve a logical layout handle to a physical layout.
    ///
    /// # Arguments
    /// * `logical` - The logical layout handle (G1, G2, G3)
    ///
    /// # Returns
    /// The physical layout for the given logical handle, or an error if not found.
    fn resolve_layout(&self, logical: LogicalLayoutHandle) -> Result<PhysicalLayout>;
}

/// Trait for registering CUDA events for completion notification.
///
/// This trait abstracts the CUDA event registration mechanism, allowing
/// [`NcclCollectives`] to integrate with different event polling systems.
/// Implementations should use efficient background polling rather than
/// spawning individual tasks per event.
///
/// The primary implementation wraps `TransferContext::register_cuda_event`,
/// which uses a shared background task for polling multiple events.
pub trait CudaEventRegistrar: Send + Sync {
    /// Register a CUDA event for completion notification.
    ///
    /// The returned notification will complete when the CUDA event has been
    /// signaled (i.e., all operations recorded before the event have completed).
    ///
    /// # Arguments
    /// * `event` - The CUDA event to monitor
    ///
    /// # Returns
    /// A notification that completes when the event is signaled.
    fn register_cuda_event(&self, event: CudaEvent) -> TransferCompleteNotification;
}

/// Ownership mode for the NCCL communicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommOwnership {
    /// We own the communicator and must destroy it on drop.
    Owned,
    /// The communicator is borrowed from external code (PyTorch, etc.).
    Borrowed,
}

/// Stream wrapper that can be either owned or borrowed.
enum NcclStream {
    /// Owned CudaStream - we control its lifetime
    Owned(Arc<CudaStream>),
    /// Borrowed raw stream pointer - caller controls lifetime
    Borrowed(CUstream),
}

impl NcclStream {
    /// Get the raw CUDA stream pointer for NCCL calls.
    fn raw(&self) -> CUstream {
        match self {
            NcclStream::Owned(stream) => stream.cu_stream(),
            NcclStream::Borrowed(ptr) => *ptr,
        }
    }

    /// Get the owned stream (for recording events). Only available for owned streams.
    fn as_owned(&self) -> Option<&Arc<CudaStream>> {
        match self {
            NcclStream::Owned(stream) => Some(stream),
            NcclStream::Borrowed(_) => None,
        }
    }
}

/// NCCL-based collective operations for GPU-to-GPU communication.
///
/// This implementation uses NVIDIA NCCL for efficient broadcast operations
/// across GPUs. It supports both owned communicators (created via bootstrap)
/// and borrowed communicators (from PyTorch, vLLM, etc.).
///
/// # Performance
///
/// Broadcast operations use NCCL groups to batch multiple memory region
/// transfers into a single collective operation, minimizing synchronization
/// overhead.
pub struct NcclCollectives {
    /// NCCL communicator handle
    comm: ncclComm_t,

    /// Whether we own the communicator (and must destroy it on drop)
    ownership: CommOwnership,

    /// Rank of this worker in the collective group
    rank: usize,

    /// Total number of workers in the collective group
    world_size: usize,

    /// CUDA stream for NCCL operations (owned or borrowed)
    nccl_stream: NcclStream,

    /// CUDA context for stream/event management (only used for owned mode)
    #[allow(dead_code)]
    cuda_context: Arc<CudaContext>,

    /// Event system for completion notifications (used for borrowed stream fallback)
    event_system: Arc<LocalEventSystem>,

    /// CUDA event registrar for efficient completion notification
    event_registrar: Arc<dyn CudaEventRegistrar>,

    /// Layout resolver for mapping logical handles to physical layouts
    layout_resolver: Arc<dyn LayoutResolver>,
}

impl NcclCollectives {
    // =========================================================================
    // Path A: Create from scratch (used by tests, standalone Rust apps)
    // =========================================================================

    /// Create with a new NCCL communicator initialized from bootstrap info.
    ///
    /// This is a **collective operation** - all ranks must call simultaneously
    /// with the same bootstrap data for initialization to succeed.
    ///
    /// # Arguments
    /// * `bootstrap` - Bootstrap data containing the NCCL unique ID
    /// * `rank` - The rank of this worker (0 to world_size-1)
    /// * `cuda_context` - CUDA context for stream management
    /// * `event_system` - Event system for fallback completion notifications
    /// * `event_registrar` - Registrar for efficient CUDA event completion polling
    /// * `layout_resolver` - Resolver for mapping logical handles to physical layouts
    ///
    /// # Returns
    /// A new `NcclCollectives` instance that owns its communicator.
    ///
    /// # Errors
    /// Returns an error if NCCL initialization fails.
    pub fn from_bootstrap(
        bootstrap: &NcclBootstrap,
        rank: usize,
        cuda_context: Arc<CudaContext>,
        event_system: Arc<LocalEventSystem>,
        event_registrar: Arc<dyn CudaEventRegistrar>,
        layout_resolver: Arc<dyn LayoutResolver>,
    ) -> Result<Self> {
        let nccl_stream = cuda_context
            .new_stream()
            .context("Failed to create NCCL stream")?;

        let comm = bootstrap
            .init_communicator(rank, nccl_stream.cu_stream())
            .context("Failed to initialize NCCL communicator")?;

        Ok(Self {
            comm,
            ownership: CommOwnership::Owned,
            rank,
            world_size: bootstrap.world_size(),
            nccl_stream: NcclStream::Owned(nccl_stream),
            cuda_context,
            event_system,
            event_registrar,
            layout_resolver,
        })
    }

    // =========================================================================
    // Path B: Borrow existing communicator (production use with Python/C/C++)
    // =========================================================================

    /// Create from borrowed NCCL handles passed from external code.
    ///
    /// This is the primary production path when the NCCL communicator is
    /// initialized by Python (torch.distributed), C++, or another runtime.
    ///
    /// # Arguments
    /// * `comm_ptr` - Raw pointer to `ncclComm_t` handle (cast to usize)
    /// * `stream_ptr` - Raw pointer to `cudaStream_t` handle (cast to usize)
    /// * `rank` - The rank of this worker in the collective group
    /// * `world_size` - Total number of workers in the collective group
    /// * `cuda_context` - CUDA context for event management
    /// * `event_system` - Event system for fallback completion notifications
    /// * `event_registrar` - Registrar for efficient CUDA event completion polling
    /// * `layout_resolver` - Resolver for mapping logical handles to physical layouts
    ///
    /// # Safety
    /// - `comm_ptr` must be a valid `ncclComm_t` handle
    /// - `stream_ptr` must be a valid `cudaStream_t` handle
    /// - The caller must ensure the handles outlive this struct
    /// - The communicator must not be destroyed while this struct exists
    ///
    /// # FFI Example (Python via PyO3)
    /// ```python
    /// # In Python
    /// comm = torch.distributed.distributed_c10d._get_default_group()._get_backend()._nccl_comm
    /// stream = torch.cuda.current_stream()
    ///
    /// # Pass to Rust
    /// collectives = NcclCollectives.from_borrowed(
    ///     comm_ptr=comm.as_int(),
    ///     stream_ptr=stream.cuda_stream,
    ///     rank=rank,
    ///     world_size=world_size,
    /// )
    /// ```
    ///
    /// # FFI Example (C/C++)
    /// ```c
    /// // In C/C++
    /// ncclComm_t comm;
    /// ncclCommInitRank(&comm, world_size, id, rank);
    /// cudaStream_t stream;
    /// cudaStreamCreate(&stream);
    ///
    /// // Pass to Rust via FFI
    /// nccl_collectives_from_borrowed((uintptr_t)comm, (uintptr_t)stream, rank, world_size);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_borrowed(
        comm_ptr: usize,
        stream_ptr: usize,
        rank: usize,
        world_size: usize,
        cuda_context: Arc<CudaContext>,
        event_system: Arc<LocalEventSystem>,
        event_registrar: Arc<dyn CudaEventRegistrar>,
        layout_resolver: Arc<dyn LayoutResolver>,
    ) -> Self {
        Self {
            comm: comm_ptr as ncclComm_t,
            ownership: CommOwnership::Borrowed,
            rank,
            world_size,
            nccl_stream: NcclStream::Borrowed(stream_ptr as CUstream),
            cuda_context,
            event_system,
            event_registrar,
            layout_resolver,
        }
    }

    /// Broadcast memory regions using NCCL grouped operations.
    ///
    /// # Arguments
    /// * `regions` - Vector of (ptr, size) pairs for memory regions to broadcast
    /// * `root` - Root rank for the broadcast
    ///
    /// # Errors
    /// Returns an error if any NCCL operation fails.
    fn broadcast_regions(&self, regions: &[(usize, usize)], root: i32) -> Result<()> {
        if regions.is_empty() {
            return Ok(());
        }

        let stream = self.nccl_stream.raw();

        // Start NCCL group - batches operations for efficiency
        let result = unsafe { ncclGroupStart() };
        check_nccl_result(result).context("ncclGroupStart failed")?;

        // Queue all broadcasts within the group
        for (ptr, size) in regions {
            // SAFETY: We're calling NCCL with valid pointers within a group operation.
            // The stream cast is safe because both cudarc::driver::sys::CUstream and
            // cudarc::nccl::sys::CUstream are the same underlying CUDA type (*mut CUstream_st).
            let result = unsafe {
                ncclBcast(
                    *ptr as *mut std::ffi::c_void,
                    *size,
                    ncclDataType_t::ncclChar, // byte-level transfer
                    root,
                    self.comm,
                    stream.cast(),
                )
            };
            check_nccl_result(result).context("ncclBcast failed")?;
        }

        // End group - submits all queued ops to GPU
        let result = unsafe { ncclGroupEnd() };
        check_nccl_result(result).context("ncclGroupEnd failed")?;

        Ok(())
    }

    /// Collect memory regions for a set of blocks and layers.
    ///
    /// # Arguments
    /// * `layout` - Physical layout to query
    /// * `block_ids` - Block IDs to collect regions for
    /// * `layer_range` - Range of layers to include (None = all layers)
    ///
    /// # Returns
    /// Vector of (address, size) pairs for the requested regions.
    fn collect_regions(
        &self,
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<Vec<(usize, usize)>> {
        let num_layers = layout.layout().num_layers();
        let outer_dim = layout.layout().outer_dim();

        let layer_range = layer_range.unwrap_or(0..num_layers);

        let mut regions =
            Vec::with_capacity(block_ids.len() * (layer_range.end - layer_range.start) * outer_dim);

        for &block_id in block_ids {
            for layer_id in layer_range.clone() {
                for outer_id in 0..outer_dim {
                    let region = layout.memory_region(block_id, layer_id, outer_id)?;
                    regions.push((region.addr, region.size));
                }
            }
        }

        Ok(regions)
    }

    /// Create a completion notification by recording an event on the NCCL stream.
    fn create_completion_notification(&self) -> Result<TransferCompleteNotification> {
        // For owned streams, we can record an event and use the efficient registrar
        if let Some(stream) = self.nccl_stream.as_owned() {
            let cuda_event = stream
                .record_event(None)
                .context("Failed to record CUDA event")?;

            // Use the event registrar for efficient background polling
            Ok(self.event_registrar.register_cuda_event(cuda_event))
        } else {
            // For borrowed streams, we can't easily record events since we don't
            // have ownership. Return an immediate completion notification.
            // The caller is responsible for synchronization with the borrowed stream.
            tracing::warn!(
                "Using borrowed stream - returning immediate completion. \
                 Caller must ensure stream synchronization."
            );

            let nova_event = self.event_system.new_event()?;
            let handle = nova_event.handle();
            nova_event.trigger()?;
            let awaiter = self.event_system.awaiter(handle)?;
            Ok(TransferCompleteNotification::from_awaiter(awaiter))
        }
    }
}

impl CollectiveOps for NcclCollectives {
    fn broadcast(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification> {
        // Resolve layouts
        let src_layout = self.layout_resolver.resolve_layout(src)?;
        let dst_layout = self.layout_resolver.resolve_layout(dst)?;

        // For broadcast, rank 0 uses src, other ranks use dst
        let layout = if self.rank == 0 {
            &src_layout
        } else {
            &dst_layout
        };

        let block_ids = if self.rank == 0 {
            src_block_ids
        } else {
            dst_block_ids
        };

        // Collect memory regions for the broadcast
        let regions = self.collect_regions(layout, block_ids, layer_range)?;

        tracing::debug!(
            rank = self.rank,
            world_size = self.world_size,
            num_regions = regions.len(),
            total_bytes = regions.iter().map(|(_, size)| size).sum::<usize>(),
            "Starting NCCL broadcast"
        );

        // Execute grouped broadcast (rank 0 is always root for broadcast)
        self.broadcast_regions(&regions, 0)?;

        // Create completion notification
        self.create_completion_notification()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

impl Drop for NcclCollectives {
    fn drop(&mut self) {
        if self.ownership == CommOwnership::Owned {
            // SAFETY: We own this communicator and it's valid
            let result = unsafe { ncclCommDestroy(self.comm) };
            if let Err(e) = check_nccl_result(result) {
                tracing::warn!("Failed to destroy NCCL communicator: {:?}", e);
            }
        }
    }
}

// SAFETY: NcclCollectives can be sent between threads.
// The NCCL communicator itself is thread-safe when operations use
// the same stream (which we guarantee by having a dedicated stream).
unsafe impl Send for NcclCollectives {}

// SAFETY: NcclCollectives can be shared between threads.
// All mutable state is behind Arc or atomic operations, and NCCL
// operations are thread-safe when using the same stream.
unsafe impl Sync for NcclCollectives {}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
    use cudarc::nccl::sys::{ncclCommDestroy, ncclCommInitAll};
    use std::ffi::c_int;
    use std::sync::{Arc, Barrier};
    use std::thread;

    /// Get the number of CUDA devices available.
    fn cuda_device_count() -> usize {
        CudaContext::device_count().unwrap_or(0) as usize
    }

    /// Initialize NCCL communicators for all devices using ncclCommInitAll.
    ///
    /// This is the single-process multi-GPU initialization pattern.
    /// Returns a vector of communicator handles as usize (for Send).
    unsafe fn init_all_comms(num_devices: usize) -> Result<Vec<usize>> {
        let mut comms: Vec<ncclComm_t> = vec![std::ptr::null_mut(); num_devices];
        let devices: Vec<c_int> = (0..num_devices as c_int).collect();

        // SAFETY: ncclCommInitAll is safe to call with valid pointers
        let result =
            unsafe { ncclCommInitAll(comms.as_mut_ptr(), num_devices as c_int, devices.as_ptr()) };

        check_nccl_result(result).context("ncclCommInitAll failed")?;
        // Convert to usize for Send
        Ok(comms.into_iter().map(|c| c as usize).collect())
    }

    /// Clean up NCCL communicators.
    unsafe fn destroy_comms(comms: &[usize]) {
        for &comm in comms {
            // SAFETY: Converting back from usize and destroying
            unsafe {
                let _ = ncclCommDestroy(comm as ncclComm_t);
            }
        }
    }

    /// Helper to get device pointer from CudaSlice with stream.
    fn get_device_ptr(slice: &CudaSlice<u8>, stream: &CudaStream) -> usize {
        let (ptr, _guard) = slice.device_ptr(stream);
        ptr as usize
    }

    /// Initialize NCCL communicators for multiple ranks on a SINGLE GPU.
    ///
    /// This allows testing NCCL API correctness without multiple GPUs.
    /// All ranks use device 0, so broadcasts become memory copies on the same device.
    /// Returns a vector of communicator handles as usize (for Send).
    unsafe fn init_single_gpu_comms(num_ranks: usize) -> Result<Vec<usize>> {
        let mut comms: Vec<ncclComm_t> = vec![std::ptr::null_mut(); num_ranks];
        // All ranks use device 0
        let devices: Vec<c_int> = vec![0; num_ranks];

        // SAFETY: ncclCommInitAll supports duplicate device IDs for multi-rank single-GPU
        let result =
            unsafe { ncclCommInitAll(comms.as_mut_ptr(), num_ranks as c_int, devices.as_ptr()) };

        check_nccl_result(result).context("ncclCommInitAll (single-GPU) failed")?;
        // Convert to usize for Send
        Ok(comms.into_iter().map(|c| c as usize).collect())
    }

    // NOTE: These NCCL tests require a full NCCL installation with all symbols.
    // Some stripped NCCL builds (e.g., Lambda Labs' 2.26.2-0lambda1) are missing
    // ncclAlltoAll, ncclGather, ncclScatter, etc. which cudarc requires.
    // If tests fail with "undefined symbol: ncclAlltoAll", install official NVIDIA NCCL.

    #[test]
    #[ignore = "requires full NCCL and at least 1 GPU - run with: cargo test -p dynamo-kvbm nccl_broadcast_single_gpu -- --ignored"]
    fn test_nccl_broadcast_single_gpu_multi_rank() {
        // Skip if no GPU available
        let num_devices = cuda_device_count();
        if num_devices < 1 {
            println!("Skipping test: no GPU available");
            return;
        }

        // Use 2 ranks on the same GPU
        let world_size = 2;
        println!(
            "Testing NCCL broadcast with {} ranks on single GPU (device 0)",
            world_size
        );

        // Initialize communicators - all ranks use device 0
        let comms =
            unsafe { init_single_gpu_comms(world_size) }.expect("Failed to init NCCL comms");

        // All ranks share the same CUDA context (device 0)
        let context = CudaContext::new(0).expect("Failed to create CUDA context");

        // Each rank gets its own stream
        let streams: Vec<Arc<CudaStream>> = (0..world_size)
            .map(|_| context.new_stream().expect("Failed to create stream"))
            .collect();

        // Test data
        let test_size = 1024 * 1024; // 1 MB
        let test_pattern: u8 = 0xCD;

        // Allocate separate buffers for each "rank" (all on same GPU)
        let buffers: Vec<CudaSlice<u8>> = streams
            .iter()
            .map(|stream| {
                let zeros = vec![0u8; test_size];
                stream
                    .clone_htod(&zeros)
                    .expect("Failed to allocate buffer")
            })
            .collect();

        // Fill rank 0's buffer with test pattern
        {
            let host_data = vec![test_pattern; test_size];
            let src_buffer = streams[0]
                .clone_htod(&host_data)
                .expect("Failed to copy to device");

            let src_ptr = get_device_ptr(&src_buffer, &streams[0]);
            let dst_ptr = get_device_ptr(&buffers[0], &streams[0]);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr as u64,
                    src_ptr as u64,
                    test_size,
                    streams[0].cu_stream(),
                )
                .expect("dtod copy failed");
            }
            streams[0].synchronize().expect("sync failed");
        }

        // Get buffer pointers
        let buffer_ptrs: Vec<usize> = buffers
            .iter()
            .zip(streams.iter())
            .map(|(buf, stream)| get_device_ptr(buf, stream))
            .collect();

        // Synchronization barrier
        let barrier = Arc::new(Barrier::new(world_size));

        // Spawn threads to perform broadcast
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let comm = comms[rank];
                let stream = streams[rank].clone();
                let buffer_ptr = buffer_ptrs[rank];
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    let result = unsafe {
                        ncclBcast(
                            buffer_ptr as *mut std::ffi::c_void,
                            test_size,
                            ncclDataType_t::ncclChar,
                            0,                  // root rank
                            comm as ncclComm_t,
                            stream.cu_stream().cast(),
                        )
                    };

                    check_nccl_result(result).expect("ncclBcast failed");
                    stream.synchronize().expect("Stream sync failed");
                    println!("Rank {} completed broadcast", rank);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all buffers have the test pattern
        for (rank, (stream, buffer)) in streams.iter().zip(buffers.iter()).enumerate() {
            let host_data = stream
                .clone_dtoh(buffer)
                .expect("Failed to copy from device");

            assert_eq!(
                host_data[0], test_pattern,
                "Rank {} first byte mismatch",
                rank
            );
            assert_eq!(
                host_data[test_size - 1],
                test_pattern,
                "Rank {} last byte mismatch",
                rank
            );

            let mismatch_count = host_data.iter().filter(|&&b| b != test_pattern).count();
            assert_eq!(
                mismatch_count, 0,
                "Rank {} has {} mismatched bytes",
                rank, mismatch_count
            );

            println!("Rank {} verified: all {} bytes correct", rank, test_size);
        }

        unsafe { destroy_comms(&comms) };
        println!("Single-GPU multi-rank test passed!");
    }

    #[test]
    #[ignore = "requires multiple GPUs - run with: cargo test -p dynamo-kvbm nccl_broadcast_multi_gpu -- --ignored"]
    fn test_nccl_broadcast_multi_gpu_raw() {
        // Skip if < 2 GPUs available
        let num_devices = cuda_device_count();
        if num_devices < 2 {
            println!(
                "Skipping test: {} GPUs available, need at least 2",
                num_devices
            );
            return;
        }

        // Use 2 GPUs for the test
        let world_size = 2;
        println!("Testing NCCL broadcast with {} GPUs", world_size);

        // Initialize all communicators at once (single-process pattern)
        let comms = unsafe { init_all_comms(world_size) }.expect("Failed to init NCCL comms");

        // Create CUDA contexts and streams for each device
        let contexts: Vec<Arc<CudaContext>> = (0..world_size)
            .map(|i| CudaContext::new(i).expect("Failed to create CUDA context"))
            .collect();

        let streams: Vec<Arc<CudaStream>> = contexts
            .iter()
            .map(|ctx| ctx.new_stream().expect("Failed to create stream"))
            .collect();

        // Test data
        let test_size = 1024 * 1024; // 1 MB
        let test_pattern: u8 = 0xAB;

        // Allocate device buffers using streams
        let buffers: Vec<CudaSlice<u8>> = streams
            .iter()
            .map(|stream| {
                // Allocate zeroed buffer
                let zeros = vec![0u8; test_size];
                stream
                    .clone_htod(&zeros)
                    .expect("Failed to allocate buffer")
            })
            .collect();

        // Fill rank 0's buffer with test pattern
        {
            let host_data = vec![test_pattern; test_size];
            let buffer = streams[0]
                .clone_htod(&host_data)
                .expect("Failed to copy to device 0");
            // Copy to actual buffer location
            let src_ptr = get_device_ptr(&buffer, &streams[0]);
            let dst_ptr = get_device_ptr(&buffers[0], &streams[0]);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr as u64,
                    src_ptr as u64,
                    test_size,
                    streams[0].cu_stream(),
                )
                .expect("dtod copy failed");
            }
            streams[0].synchronize().expect("sync failed");
        }

        // Get buffer pointers before spawning threads (to avoid lifetime issues)
        let buffer_ptrs: Vec<usize> = buffers
            .iter()
            .zip(streams.iter())
            .map(|(buf, stream)| get_device_ptr(buf, stream))
            .collect();

        // Synchronization barrier for threads
        let barrier = Arc::new(Barrier::new(world_size));

        // Spawn threads to perform broadcast
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let comm = comms[rank]; // Already usize, which is Send
                let stream = streams[rank].clone();
                let buffer_ptr = buffer_ptrs[rank];
                let barrier = barrier.clone();

                thread::spawn(move || {
                    // Wait for all threads to be ready
                    barrier.wait();

                    // Perform broadcast (rank 0 is root)
                    let result = unsafe {
                        ncclBcast(
                            buffer_ptr as *mut std::ffi::c_void,
                            test_size,
                            ncclDataType_t::ncclChar,
                            0,                  // root rank
                            comm as ncclComm_t, // Convert back to ncclComm_t
                            stream.cu_stream().cast(),
                        )
                    };

                    check_nccl_result(result).expect("ncclBcast failed");

                    // Synchronize stream
                    stream.synchronize().expect("Stream sync failed");

                    println!("Rank {} completed broadcast", rank);
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all buffers have the test pattern
        for (rank, (stream, buffer)) in streams.iter().zip(buffers.iter()).enumerate() {
            let host_data = stream
                .clone_dtoh(buffer)
                .expect("Failed to copy from device");

            // Check first and last bytes, plus some random samples
            assert_eq!(
                host_data[0], test_pattern,
                "Rank {} first byte mismatch",
                rank
            );
            assert_eq!(
                host_data[test_size - 1],
                test_pattern,
                "Rank {} last byte mismatch",
                rank
            );
            assert_eq!(
                host_data[test_size / 2],
                test_pattern,
                "Rank {} middle byte mismatch",
                rank
            );

            // Verify all bytes
            let mismatch_count = host_data.iter().filter(|&&b| b != test_pattern).count();
            assert_eq!(
                mismatch_count, 0,
                "Rank {} has {} mismatched bytes",
                rank, mismatch_count
            );

            println!("Rank {} verified: all {} bytes correct", rank, test_size);
        }

        // Clean up
        unsafe { destroy_comms(&comms) };
        println!("Test passed!");
    }

    #[test]
    #[ignore = "requires multiple GPUs - run with: cargo test -p dynamo-kvbm nccl_grouped_broadcast -- --ignored"]
    fn test_nccl_grouped_broadcast_multi_gpu() {
        // Skip if < 2 GPUs available
        let num_devices = cuda_device_count();
        if num_devices < 2 {
            println!(
                "Skipping test: {} GPUs available, need at least 2",
                num_devices
            );
            return;
        }

        // Use 2 GPUs for the test
        let world_size = 2;
        println!("Testing NCCL grouped broadcast with {} GPUs", world_size);

        // Initialize all communicators at once
        let comms = unsafe { init_all_comms(world_size) }.expect("Failed to init NCCL comms");

        // Create CUDA contexts and streams
        let contexts: Vec<Arc<CudaContext>> = (0..world_size)
            .map(|i| CudaContext::new(i).expect("Failed to create CUDA context"))
            .collect();

        let streams: Vec<Arc<CudaStream>> = contexts
            .iter()
            .map(|ctx| ctx.new_stream().expect("Failed to create stream"))
            .collect();

        // Test multiple regions (simulating multiple blocks)
        let num_regions = 4;
        let region_size = 256 * 1024; // 256 KB per region

        // Allocate multiple buffers per device
        let buffers: Vec<Vec<CudaSlice<u8>>> = streams
            .iter()
            .map(|stream| {
                (0..num_regions)
                    .map(|_| {
                        let zeros = vec![0u8; region_size];
                        stream.clone_htod(&zeros).expect("Failed to allocate")
                    })
                    .collect()
            })
            .collect();

        // Fill rank 0's buffers with different patterns
        for (region_idx, buffer) in buffers[0].iter().enumerate() {
            let pattern = (region_idx + 1) as u8 * 0x11; // Different pattern per region
            let host_data = vec![pattern; region_size];
            let src_buffer = streams[0]
                .clone_htod(&host_data)
                .expect("Failed to allocate src");

            let src_ptr = get_device_ptr(&src_buffer, &streams[0]);
            let dst_ptr = get_device_ptr(buffer, &streams[0]);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr as u64,
                    src_ptr as u64,
                    region_size,
                    streams[0].cu_stream(),
                )
                .expect("dtod copy failed");
            }
        }
        streams[0].synchronize().expect("sync failed");

        // Synchronization barrier
        let barrier = Arc::new(Barrier::new(world_size));

        // Collect buffer pointers for each rank (as usize for Send)
        let buffer_ptrs: Vec<Vec<usize>> = buffers
            .iter()
            .zip(streams.iter())
            .map(|(rank_buffers, stream)| {
                rank_buffers
                    .iter()
                    .map(|b| get_device_ptr(b, stream))
                    .collect()
            })
            .collect();

        // Spawn threads for grouped broadcast
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let comm = comms[rank]; // Already usize, which is Send
                let stream = streams[rank].clone();
                let ptrs = buffer_ptrs[rank].clone();
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Use NCCL group for multiple broadcasts
                    unsafe {
                        check_nccl_result(ncclGroupStart()).expect("ncclGroupStart failed");

                        for ptr in &ptrs {
                            let result = ncclBcast(
                                *ptr as *mut std::ffi::c_void,
                                region_size,
                                ncclDataType_t::ncclChar,
                                0,
                                comm as ncclComm_t,
                                stream.cu_stream().cast(),
                            );
                            check_nccl_result(result).expect("ncclBcast failed");
                        }

                        check_nccl_result(ncclGroupEnd()).expect("ncclGroupEnd failed");
                    }

                    stream.synchronize().expect("Stream sync failed");
                    println!("Rank {} completed grouped broadcast", rank);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify all ranks have correct data
        for (rank, (stream, rank_buffers)) in streams.iter().zip(buffers.iter()).enumerate() {
            for (region_idx, buffer) in rank_buffers.iter().enumerate() {
                let expected_pattern = (region_idx + 1) as u8 * 0x11;
                let host_data = stream
                    .clone_dtoh(buffer)
                    .expect("Failed to copy from device");

                let mismatch_count = host_data.iter().filter(|&&b| b != expected_pattern).count();
                assert_eq!(
                    mismatch_count, 0,
                    "Rank {} region {} has {} mismatched bytes (expected 0x{:02x})",
                    rank, region_idx, mismatch_count, expected_pattern
                );
            }
            println!(
                "Rank {} verified: all {} regions correct",
                rank, num_regions
            );
        }

        unsafe { destroy_comms(&comms) };
        println!("Grouped broadcast test passed!");
    }

    #[test]
    #[ignore = "requires multiple GPUs - run with: cargo test -p dynamo-kvbm nccl_broadcast_large -- --ignored"]
    fn test_nccl_broadcast_large_transfer() {
        // Skip if < 2 GPUs available
        let num_devices = cuda_device_count();
        if num_devices < 2 {
            println!(
                "Skipping test: {} GPUs available, need at least 2",
                num_devices
            );
            return;
        }

        let world_size = 2;
        println!("Testing NCCL large broadcast with {} GPUs", world_size);

        let comms = unsafe { init_all_comms(world_size) }.expect("Failed to init NCCL comms");

        let contexts: Vec<Arc<CudaContext>> = (0..world_size)
            .map(|i| CudaContext::new(i).expect("Failed to create CUDA context"))
            .collect();

        let streams: Vec<Arc<CudaStream>> = contexts
            .iter()
            .map(|ctx| ctx.new_stream().expect("Failed to create stream"))
            .collect();

        // Large transfer: 64 MB (typical KV cache block size)
        let test_size = 64 * 1024 * 1024;
        println!("Transfer size: {} MB", test_size / (1024 * 1024));

        // Allocate buffers
        let buffers: Vec<CudaSlice<u8>> = streams
            .iter()
            .map(|stream| {
                let zeros = vec![0u8; test_size];
                stream.clone_htod(&zeros).expect("Failed to allocate")
            })
            .collect();

        // Fill rank 0 with pseudo-random pattern
        {
            let host_data: Vec<u8> = (0..test_size).map(|i| (i % 256) as u8).collect();
            let src_buffer = streams[0]
                .clone_htod(&host_data)
                .expect("Failed to copy to device 0");

            let src_ptr = get_device_ptr(&src_buffer, &streams[0]);
            let dst_ptr = get_device_ptr(&buffers[0], &streams[0]);
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr as u64,
                    src_ptr as u64,
                    test_size,
                    streams[0].cu_stream(),
                )
                .expect("dtod copy failed");
            }
            streams[0].synchronize().expect("sync failed");
        }

        // Get buffer pointers
        let buffer_ptrs: Vec<usize> = buffers
            .iter()
            .zip(streams.iter())
            .map(|(buf, stream)| get_device_ptr(buf, stream))
            .collect();

        let barrier = Arc::new(Barrier::new(world_size));

        let start = std::time::Instant::now();

        // Spawn threads for large transfer
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let comm = comms[rank]; // Already usize, which is Send
                let stream = streams[rank].clone();
                let buffer_ptr = buffer_ptrs[rank];
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    let result = unsafe {
                        ncclBcast(
                            buffer_ptr as *mut std::ffi::c_void,
                            test_size,
                            ncclDataType_t::ncclChar,
                            0,
                            comm as ncclComm_t,
                            stream.cu_stream().cast(),
                        )
                    };
                    check_nccl_result(result).expect("ncclBcast failed");
                    stream.synchronize().expect("Stream sync failed");
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let elapsed = start.elapsed();
        let throughput_gbps =
            (test_size as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed.as_secs_f64();
        println!(
            "Transfer completed in {:?} ({:.2} GB/s)",
            elapsed, throughput_gbps
        );

        // Verify data on rank 1
        {
            let host_data = streams[1]
                .clone_dtoh(&buffers[1])
                .expect("Failed to copy from device 1");

            // Sample verification (checking every byte would be slow)
            let samples = [
                0,
                test_size / 4,
                test_size / 2,
                test_size * 3 / 4,
                test_size - 1,
            ];
            for &idx in &samples {
                let expected = (idx % 256) as u8;
                assert_eq!(
                    host_data[idx], expected,
                    "Mismatch at index {}: expected {}, got {}",
                    idx, expected, host_data[idx]
                );
            }

            // Full verification with sampling
            let mismatch_count = host_data
                .iter()
                .enumerate()
                .filter(|(i, b)| **b != (*i % 256) as u8)
                .count();
            assert_eq!(
                mismatch_count, 0,
                "Found {} mismatched bytes",
                mismatch_count
            );
        }

        unsafe { destroy_comms(&comms) };
        println!("Large transfer test passed!");
    }
}
