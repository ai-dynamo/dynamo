// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! oneCCL-based collective operations for XPU-to-XPU communication.
//!
//! This module provides [`OneCclCollectives`], an implementation of [`CollectiveOps`]
//! that uses Intel oneCCL for efficient XPU collective communication.
//!
//! # Construction Paths
//!
//! oneCCL communicators can be obtained via two paths:
//!
//! ## Path A: Bootstrap (tests and standalone Rust apps)
//!
//! Use [`OneCclCollectives::from_bootstrap`] when creating communicators from scratch:
//!
//! ```rust,ignore
//! let bootstrap = OneCclBootstrap::generate(world_size)?;
//! // ... distribute bootstrap to other ranks ...
//! let collectives = OneCclCollectives::from_bootstrap(
//!     &bootstrap,
//!     rank,
//!     sycl_queue_ptr,
//!     event_system,
//!     layout_resolver,
//! )?;
//! ```
//!
//! ## Path B: Borrowed handles (production with PyTorch/vLLM)
//!
//! Use [`OneCclCollectives::from_borrowed`] when an external runtime provides
//! the communicator:
//!
//! ```rust,ignore
//! let collectives = unsafe {
//!     OneCclCollectives::from_borrowed(
//!         comm_ptr,
//!         stream_ptr,
//!         rank,
//!         world_size,
//!         event_system,
//!         layout_resolver,
//!     )
//! };
//! ```
//!
//! # Thread Safety
//!
//! oneCCL operations are thread-safe when each thread uses its own stream.
//! This implementation uses a dedicated CCL stream per `OneCclCollectives` instance.

use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use anyhow::{Context, Result};
use oneapi_rs::ccl::sys;
use velo::EventManager;

use crate::BlockId;
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::layout::PhysicalLayout;
use kvbm_physical::transfer::TransferCompleteNotification;

use super::CollectiveOps;
use super::oneccl_bootstrap::{OneCclBootstrap, check_ccl_result};
use super::LayoutResolver;


/// Ownership mode for the oneCCL communicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommOwnership {
    /// We own the communicator and must destroy it on drop.
    Owned,
    /// The communicator is borrowed from external code (PyTorch, etc.).
    Borrowed,
}

/// Stream wrapper that can be either owned or borrowed.
enum CclStream {
    /// Owned CCL stream - we control its lifetime
    Owned(*mut sys::ccl_rs_stream_t),
    /// Borrowed raw stream pointer - caller controls lifetime
    Borrowed(*mut sys::ccl_rs_stream_t),
}

impl CclStream {
    /// Get the raw CCL stream pointer for oneCCL calls.
    fn raw(&self) -> *mut sys::ccl_rs_stream_t {
        match self {
            CclStream::Owned(ptr) => *ptr,
            CclStream::Borrowed(ptr) => *ptr,
        }
    }
}

// SAFETY: The CCL stream handle is safe to send between threads.
unsafe impl Send for CclStream {}
unsafe impl Sync for CclStream {}

impl Drop for CclStream {
    fn drop(&mut self) {
        if let CclStream::Owned(ptr) = self {
            if !ptr.is_null() {
                unsafe { sys::ccl_rs_stream_destroy(*ptr) };
            }
        }
    }
}

/// oneCCL-based collective operations for XPU-to-XPU communication.
///
/// This implementation uses Intel oneCCL for efficient broadcast operations
/// across XPUs. It supports both owned communicators (created via bootstrap)
/// and borrowed communicators (from PyTorch, etc.).
///
/// # Performance
///
/// Broadcast operations submit oneCCL collectives on a dedicated SYCL queue
/// wrapped as a CCL stream. Completion is tracked via `ccl_rs_event_test`
/// polling, analogous to CUDA event polling in the NCCL path.
pub struct OneCclCollectives {
    /// oneCCL communicator handle
    comm: *mut sys::ccl_rs_comm_t,

    /// Whether we own the communicator (and must destroy it on drop)
    ownership: CommOwnership,

    /// Rank of this worker in the collective group
    rank: usize,

    /// Total number of workers in the collective group
    world_size: usize,

    /// oneCCL stream (wrapping a SYCL queue) for collective operations
    ccl_stream: CclStream,

    /// Event system for completion notifications
    event_system: EventManager,

    /// Layout resolver for mapping logical handles to physical layouts
    layout_resolver: Arc<dyn LayoutResolver>,
}

impl OneCclCollectives {
    // =========================================================================
    // Path A: Create from scratch (used by tests, standalone Rust apps)
    // =========================================================================

    /// Create with a new oneCCL communicator initialized from bootstrap info.
    ///
    /// This is a **collective operation** - all ranks must call simultaneously
    /// with the same bootstrap data for initialization to succeed.
    ///
    /// # Arguments
    /// * `bootstrap` - Bootstrap data containing the oneCCL KVS address
    /// * `rank` - The rank of this worker (0 to world_size-1)
    /// * `sycl_queue_ptr` - Raw `sycl::queue*` pointer for the dedicated CCL stream
    /// * `event_system` - Event system for completion notifications
    /// * `layout_resolver` - Resolver for mapping logical handles to physical layouts
    ///
    /// # Safety
    /// `sycl_queue_ptr` must be a valid `sycl::queue*` pointer that outlives this struct.
    pub unsafe fn from_bootstrap(
        bootstrap: &OneCclBootstrap,
        rank: usize,
        sycl_queue_ptr: *mut c_void,
        event_system: EventManager,
        layout_resolver: Arc<dyn LayoutResolver>,
    ) -> Result<Self> {
        let ccl_stream = unsafe { OneCclBootstrap::create_stream(sycl_queue_ptr) }
            .context("Failed to create CCL stream from SYCL queue")?;

        let comm = bootstrap
            .init_communicator(rank)
            .context("Failed to initialize oneCCL communicator")?;

        Ok(Self {
            comm,
            ownership: CommOwnership::Owned,
            rank,
            world_size: bootstrap.world_size(),
            ccl_stream: CclStream::Owned(ccl_stream),
            event_system,
            layout_resolver,
        })
    }

    // =========================================================================
    // Path B: Borrow existing communicator (production use with Python/C/C++)
    // =========================================================================

    /// Create from borrowed oneCCL handles passed from external code.
    ///
    /// This is the primary production path when the oneCCL communicator is
    /// initialized by Python (torch.distributed with ccl backend), or another runtime.
    ///
    /// # Arguments
    /// * `comm_ptr` - Raw pointer to `ccl_rs_comm_t` handle (cast to usize)
    /// * `stream_ptr` - Raw pointer to `ccl_rs_stream_t` handle (cast to usize)
    /// * `rank` - The rank of this worker in the collective group
    /// * `world_size` - Total number of workers in the collective group
    /// * `event_system` - Event system for completion notifications
    /// * `layout_resolver` - Resolver for mapping logical handles to physical layouts
    ///
    /// # Safety
    /// - `comm_ptr` must be a valid `ccl_rs_comm_t*` handle
    /// - `stream_ptr` must be a valid `ccl_rs_stream_t*` handle
    /// - The caller must ensure the handles outlive this struct
    pub unsafe fn from_borrowed(
        comm_ptr: usize,
        stream_ptr: usize,
        rank: usize,
        world_size: usize,
        event_system: EventManager,
        layout_resolver: Arc<dyn LayoutResolver>,
    ) -> Self {
        Self {
            comm: comm_ptr as *mut sys::ccl_rs_comm_t,
            ownership: CommOwnership::Borrowed,
            rank,
            world_size,
            ccl_stream: CclStream::Borrowed(stream_ptr as *mut sys::ccl_rs_stream_t),
            event_system,
            layout_resolver,
        }
    }

    /// Broadcast memory regions using oneCCL.
    ///
    /// Submission is batched via `group_start`/`group_end`. Every
    /// `ccl_rs_broadcast` returns an event and the API requires every
    /// returned event to be destroyed; we destroy them as we iterate.
    ///
    /// **Host completion via `ccl_rs_stream_wait`, not per-op events.**
    /// oneCCL warns that `ccl::event::wait()` on a collective submitted
    /// inside group API is not supported — per-op events in a group are
    /// not meaningful completion signals (the runtime may fuse/reorder
    /// them). The host-side wait for a group is a wait on
    /// the underlying `sycl::queue`, exposed via `ccl_rs_stream_wait`.
    fn broadcast_regions(&self, regions: &[(usize, usize)], root: c_int) -> Result<()> {
        if regions.is_empty() {
            return Ok(());
        }

        let stream = self.ccl_stream.raw();

        // Start a group call — batches all broadcasts
        check_ccl_result(unsafe { sys::ccl_rs_group_start() })
            .map_err(|e| anyhow::anyhow!("ccl_rs_group_start failed: {}", e))?;

        let mut last_event: *mut sys::ccl_rs_event_t = ptr::null_mut();
        let mut broadcast_err: Option<anyhow::Error> = None;

        for (ptr, size) in regions {
            let mut event: *mut sys::ccl_rs_event_t = ptr::null_mut();

            // SAFETY: We're calling oneCCL with valid pointers.
            // Using CCL_RS_DATATYPE_UINT8 for byte-level transfer (like ncclChar).
            let res = check_ccl_result(unsafe {
                sys::ccl_rs_broadcast(
                    *ptr as *mut c_void,
                    *size,
                    sys::ccl_rs_datatype_t::CCL_RS_DATATYPE_UINT8,
                    root,
                    self.comm,
                    stream,
                    &mut event,
                )
            });

            if let Err(e) = res {
                broadcast_err = Some(anyhow::anyhow!("ccl_rs_broadcast failed: {}", e));
                break;
            }

            // Destroy any previous event; we don't wait on per-op events
            // Always destroy the last one too.
            if !last_event.is_null() {
                unsafe { sys::ccl_rs_event_destroy(last_event) };
            }
            last_event = event;
        }

        // Always close the group — leaving it open corrupts oneCCL runtime state.
        let group_end_res = check_ccl_result(unsafe { sys::ccl_rs_group_end() });

        // Destroy the retained per-op event without waiting on it: per-op
        // events in a group are not valid sync points.
        if !last_event.is_null() {
            unsafe { sys::ccl_rs_event_destroy(last_event) };
        }

        // Propagate broadcast error first (root cause), then group_end error.
        if let Some(e) = broadcast_err {
            return Err(e);
        }
        group_end_res
            .map_err(|e| anyhow::anyhow!("ccl_rs_group_end failed: {}", e))?;

        // Block until all submitted work has completed on-device. This
        // delegates to `sycl::queue::wait()` on the queue wrapped by the
        // CCL stream.
        check_ccl_result(unsafe { sys::ccl_rs_stream_wait(stream) })
            .map_err(|e| anyhow::anyhow!("ccl_rs_stream_wait failed: {}", e))?;

        Ok(())
    }

    /// Collect memory regions for a set of blocks and layers.
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

    /// Create a completion notification.
    ///
    /// Returns an already-triggered notification. Callers see immediate
    /// completion because `broadcast_regions` has already blocked on
    /// `ccl_rs_stream_wait` — by the time this function runs, the
    /// broadcast is complete on-device.
    ///
    fn create_completion_notification(&self) -> Result<TransferCompleteNotification> {
        let nova_event = self.event_system.new_event()?;
        let handle = nova_event.handle();
        nova_event.trigger()?;
        let awaiter = self.event_system.awaiter(handle)?;
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

impl CollectiveOps for OneCclCollectives {
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
            "Starting oneCCL broadcast"
        );

        // Execute broadcasts (rank 0 is always root)
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

impl Drop for OneCclCollectives {
    fn drop(&mut self) {
        if self.ownership == CommOwnership::Owned && !self.comm.is_null() {
            unsafe { sys::ccl_rs_comm_destroy(self.comm) };
        }
    }
}

// SAFETY: OneCclCollectives can be sent between threads.
// The oneCCL communicator is thread-safe when operations use the same stream.
unsafe impl Send for OneCclCollectives {}

// SAFETY: OneCclCollectives can be shared between threads.
// All mutable state is behind Arc or atomic operations.
unsafe impl Sync for OneCclCollectives {}


#[cfg(test)]
mod tests {
    use super::*;
    use oneapi_rs::safe::{DevicePtr, SyclSlice, SyclDevice, SyclQueue};
    use std::sync::Arc;

    /// Get the number of XPU (SYCL) devices available.
    fn xpu_device_count() -> usize {
        SyclDevice::count().unwrap_or(0)
    }

    /// Allocate a device buffer filled with a byte pattern.
    fn alloc_filled(queue: &Arc<SyclQueue>, size: usize, pattern: u8) -> SyclSlice<u8> {
        let host_data = vec![pattern; size];
        let mut buffer = queue.alloc_zeros::<u8>(size).expect("Failed to allocate buffer");
        queue
            .memcpy_sync(host_data.as_slice(), &mut buffer)
            .expect("Failed to copy to device");
        buffer
    }

    /// Allocate a zeroed device buffer.
    fn alloc_zeroed(queue: &Arc<SyclQueue>, size: usize) -> SyclSlice<u8> {
        queue.alloc_zeros::<u8>(size).expect("Failed to allocate zeroed buffer")
    }

    /// Read device buffer back to host.
    fn read_back(queue: &Arc<SyclQueue>, buffer: &SyclSlice<u8>) -> Vec<u8> {
        queue.clone_dtoh(buffer).expect("Failed to copy from device")
    }

    // ---------------------------------------------------------------
    // Multi-process test infrastructure
    //
    // oneCCL's KVS rendezvous is designed for multi-process use.
    // Each coordinator test spawns child processes (one per rank)
    // by re-invoking the test binary with environment variables:
    //
    //   ONECCL_TEST_RANK       — rank of this child (0..world_size-1)
    //   ONECCL_TEST_WORLD_SIZE — total number of ranks
    //   ONECCL_TEST_BOOTSTRAP  — hex-encoded serialized bootstrap
    //   ONECCL_TEST_CASE       — which test case to run
    //   ONEAPI_DEVICE_SELECTOR — restricts the child to one XPU
    //
    // This matches production (torchrun / mpirun) exactly:
    //   generate → serialize → distribute → deserialize → init_communicator
    // ---------------------------------------------------------------

    /// Hex-encode bytes for safe env var transport.
    fn hex_encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Hex-decode bytes from env var.
    fn hex_decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Spawn child processes for a multi-process oneCCL test.
    fn spawn_ranks(
        test_case: &str,
        world_size: usize,
        bootstrap_hex: &str,
    ) -> Vec<std::process::Child> {
        let binary = std::env::current_exe()
            .expect("Failed to get test binary path");
        let mut children = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let child = std::process::Command::new(&binary)
                .arg("--exact")
                .arg("collectives::oneccl::tests::oneccl_worker")
                .arg("--nocapture")
                .env("ONECCL_TEST_RANK", rank.to_string())
                .env("ONECCL_TEST_WORLD_SIZE", world_size.to_string())
                .env("ONECCL_TEST_BOOTSTRAP", bootstrap_hex)
                .env("ONECCL_TEST_CASE", test_case)
                .env("ONEAPI_DEVICE_SELECTOR", format!("level_zero:{}", rank))
                .spawn()
                .unwrap_or_else(|e| panic!("Failed to spawn rank {}: {}", rank, e));

            children.push(child);
        }

        children
    }

    /// Wait for all child processes and assert they all succeeded.
    fn wait_for_all(children: Vec<std::process::Child>) {
        for (rank, child) in children.into_iter().enumerate() {
            let output = child.wait_with_output().unwrap_or_else(|e| {
                panic!("Failed to wait for rank {}: {}", rank, e);
            });
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                panic!(
                    "Rank {} exited with status: {}\nstdout:\n{}\nstderr:\n{}",
                    rank, output.status, stdout, stderr
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Worker entry point — runs inside each child process
    // ---------------------------------------------------------------

    #[test]
    fn oneccl_worker() {
        // Only execute if this is a spawned child process
        let rank_str = match std::env::var("ONECCL_TEST_RANK") {
            Ok(r) => r,
            Err(_) => return, // Not a child — skip silently
        };

        let rank: usize = rank_str.parse().expect("Invalid ONECCL_TEST_RANK");
        let world_size: usize = std::env::var("ONECCL_TEST_WORLD_SIZE")
            .expect("ONECCL_TEST_WORLD_SIZE not set")
            .parse()
            .expect("Invalid ONECCL_TEST_WORLD_SIZE");
        let bootstrap_hex =
            std::env::var("ONECCL_TEST_BOOTSTRAP").expect("ONECCL_TEST_BOOTSTRAP not set");
        let test_case =
            std::env::var("ONECCL_TEST_CASE").expect("ONECCL_TEST_CASE not set");

        let bootstrap_bytes = hex_decode(&bootstrap_hex);

        match test_case.as_str() {
            "broadcast_1mb" => worker_broadcast_1mb(rank, world_size, &bootstrap_bytes),
            "multi_region" => worker_multi_region(rank, world_size, &bootstrap_bytes),
            "large_transfer" => worker_large_transfer(rank, world_size, &bootstrap_bytes),
            other => panic!("Unknown test case: {}", other),
        }
    }

    // ---------------------------------------------------------------
    // Worker implementations
    // ---------------------------------------------------------------

    fn worker_broadcast_1mb(rank: usize, _world_size: usize, bootstrap_bytes: &[u8]) {
        let test_size = 1024 * 1024; // 1 MB
        let test_pattern: u8 = 0xAB;

        // ONEAPI_DEVICE_SELECTOR restricts us to one GPU → ordinal 0
        let queue = SyclQueue::new_for_device_ordinal(0)
            .expect("Failed to create SYCL queue");

        let buffer = if rank == 0 {
            alloc_filled(&queue, test_size, test_pattern)
        } else {
            alloc_zeroed(&queue, test_size)
        };

        let buffer_ptr = buffer.device_ptr() as usize;

        // Deserialize bootstrap (multi-process path, matches production)
        let bootstrap =
            OneCclBootstrap::deserialize(bootstrap_bytes).expect("Failed to deserialize bootstrap");

        let ccl_stream =
            unsafe { OneCclBootstrap::create_stream(queue.raw_queue_ptr()) }
                .expect("Failed to create CCL stream");

        // 3-arg init — correct for multi-process (one device per process)
        let comm = bootstrap
            .init_communicator(rank)
            .expect("Failed to init communicator");

        // Broadcast from root (rank 0)
        let mut event: *mut sys::ccl_rs_event_t = ptr::null_mut();
        check_ccl_result(unsafe {
            sys::ccl_rs_broadcast(
                buffer_ptr as *mut c_void,
                test_size,
                sys::ccl_rs_datatype_t::CCL_RS_DATATYPE_UINT8,
                0, // root rank
                comm,
                ccl_stream,
                &mut event,
            )
        })
        .expect("ccl_rs_broadcast failed");

        if !event.is_null() {
            check_ccl_result(unsafe { sys::ccl_rs_event_wait(event) })
                .expect("ccl_rs_event_wait failed");
            unsafe { sys::ccl_rs_event_destroy(event) };
        }

        // Verify
        let host_data = read_back(&queue, &buffer);
        let mismatch_count = host_data.iter().filter(|&&b| b != test_pattern).count();
        assert_eq!(
            mismatch_count, 0,
            "Rank {} has {} mismatched bytes out of {}",
            rank, mismatch_count, test_size
        );

        println!("Rank {} verified: all {} bytes correct", rank, test_size);

        unsafe {
            sys::ccl_rs_stream_destroy(ccl_stream);
            sys::ccl_rs_comm_destroy(comm);
        }
    }

    fn worker_multi_region(rank: usize, _world_size: usize, bootstrap_bytes: &[u8]) {
        let num_regions = 4;
        let region_size = 256 * 1024; // 256 KB

        let queue = SyclQueue::new_for_device_ordinal(0)
            .expect("Failed to create SYCL queue");

        let buffers: Vec<SyclSlice<u8>> = (0..num_regions)
            .map(|region_idx| {
                if rank == 0 {
                    let pattern = (region_idx + 1) as u8 * 0x11;
                    alloc_filled(&queue, region_size, pattern)
                } else {
                    alloc_zeroed(&queue, region_size)
                }
            })
            .collect();

        let buffer_ptrs: Vec<usize> = buffers.iter().map(|b| b.device_ptr() as usize).collect();

        let bootstrap =
            OneCclBootstrap::deserialize(bootstrap_bytes).expect("Failed to deserialize bootstrap");

        let ccl_stream =
            unsafe { OneCclBootstrap::create_stream(queue.raw_queue_ptr()) }
                .expect("Failed to create CCL stream");

        let comm = bootstrap
            .init_communicator(rank)
            .expect("Failed to init communicator");

        // Broadcast each region
        for &ptr in &buffer_ptrs {
            let mut event: *mut sys::ccl_rs_event_t = ptr::null_mut();
            check_ccl_result(unsafe {
                sys::ccl_rs_broadcast(
                    ptr as *mut c_void,
                    region_size,
                    sys::ccl_rs_datatype_t::CCL_RS_DATATYPE_UINT8,
                    0,
                    comm,
                    ccl_stream,
                    &mut event,
                )
            })
            .expect("ccl_rs_broadcast failed");

            if !event.is_null() {
                check_ccl_result(unsafe { sys::ccl_rs_event_wait(event) })
                    .expect("ccl_rs_event_wait failed");
                unsafe { sys::ccl_rs_event_destroy(event) };
            }
        }

        // Verify each region
        for (region_idx, buffer) in buffers.iter().enumerate() {
            let expected_pattern = (region_idx + 1) as u8 * 0x11;
            let host_data = read_back(&queue, buffer);
            let mismatch_count = host_data.iter().filter(|&&b| b != expected_pattern).count();
            assert_eq!(
                mismatch_count, 0,
                "Rank {} region {} has {} mismatched bytes (expected 0x{:02x})",
                rank, region_idx, mismatch_count, expected_pattern
            );
        }

        println!("Rank {} verified: all {} regions correct", rank, num_regions);

        unsafe {
            sys::ccl_rs_stream_destroy(ccl_stream);
            sys::ccl_rs_comm_destroy(comm);
        }
    }

    fn worker_large_transfer(rank: usize, _world_size: usize, bootstrap_bytes: &[u8]) {
        let test_size = 64 * 1024 * 1024; // 64 MB

        let queue = SyclQueue::new_for_device_ordinal(0)
            .expect("Failed to create SYCL queue");

        let buffer = if rank == 0 {
            let host_data: Vec<u8> = (0..test_size).map(|i| (i % 256) as u8).collect();
            let mut buf = queue
                .alloc_zeros::<u8>(test_size)
                .expect("Failed to allocate");
            queue
                .memcpy_sync(host_data.as_slice(), &mut buf)
                .expect("Failed to copy to device");
            buf
        } else {
            alloc_zeroed(&queue, test_size)
        };

        let buffer_ptr = buffer.device_ptr() as usize;

        let bootstrap =
            OneCclBootstrap::deserialize(bootstrap_bytes).expect("Failed to deserialize bootstrap");

        let ccl_stream =
            unsafe { OneCclBootstrap::create_stream(queue.raw_queue_ptr()) }
                .expect("Failed to create CCL stream");

        let comm = bootstrap
            .init_communicator(rank)
            .expect("Failed to init communicator");

        let start = std::time::Instant::now();

        let mut event: *mut sys::ccl_rs_event_t = ptr::null_mut();
        check_ccl_result(unsafe {
            sys::ccl_rs_broadcast(
                buffer_ptr as *mut c_void,
                test_size,
                sys::ccl_rs_datatype_t::CCL_RS_DATATYPE_UINT8,
                0,
                comm,
                ccl_stream,
                &mut event,
            )
        })
        .expect("ccl_rs_broadcast failed");

        if !event.is_null() {
            check_ccl_result(unsafe { sys::ccl_rs_event_wait(event) })
                .expect("ccl_rs_event_wait failed");
            unsafe { sys::ccl_rs_event_destroy(event) };
        }

        let elapsed = start.elapsed();

        // Verify on non-root rank
        if rank != 0 {
            let host_data = read_back(&queue, &buffer);

            let samples = [0, test_size / 4, test_size / 2, test_size * 3 / 4, test_size - 1];
            for &idx in &samples {
                let expected = (idx % 256) as u8;
                assert_eq!(
                    host_data[idx], expected,
                    "Rank {} mismatch at index {}: expected {}, got {}",
                    rank, idx, expected, host_data[idx]
                );
            }

            let mismatch_count = host_data
                .iter()
                .enumerate()
                .filter(|(i, b)| **b != (*i % 256) as u8)
                .count();
            assert_eq!(
                mismatch_count, 0,
                "Rank {} found {} mismatched bytes",
                rank, mismatch_count
            );
        }

        let throughput_gbps =
            (test_size as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed.as_secs_f64();
        println!(
            "Rank {} completed in {:?} ({:.2} GB/s)",
            rank, elapsed, throughput_gbps
        );

        unsafe {
            sys::ccl_rs_stream_destroy(ccl_stream);
            sys::ccl_rs_comm_destroy(comm);
        }
    }

    // ---------------------------------------------------------------
    // Coordinator tests (spawn children and verify exit codes)
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "testing-oneccl")]
    fn test_oneccl_broadcast_multi_xpu_raw() {
        let num_devices = xpu_device_count();
        if num_devices < 2 {
            println!("Skipping: {} XPUs available, need at least 2", num_devices);
            return;
        }

        // Skip if we're inside a child process
        if std::env::var("ONECCL_TEST_RANK").is_ok() {
            return;
        }

        let world_size = 2;
        println!(
            "Testing oneCCL broadcast with {} XPUs (multi-process)",
            world_size
        );

        let bootstrap =
            OneCclBootstrap::generate(world_size).expect("Failed to generate bootstrap");
        let bootstrap_hex = hex_encode(&bootstrap.serialize());

        let children = spawn_ranks("broadcast_1mb", world_size, &bootstrap_hex);
        wait_for_all(children);

        println!("Broadcast test passed! (all ranks verified)");
    }

    #[test]
    #[cfg(feature = "testing-oneccl")]
    fn test_oneccl_multi_region_broadcast() {
        let num_devices = xpu_device_count();
        if num_devices < 2 {
            println!("Skipping: {} XPUs available, need at least 2", num_devices);
            return;
        }

        if std::env::var("ONECCL_TEST_RANK").is_ok() {
            return;
        }

        let world_size = 2;
        println!(
            "Testing oneCCL multi-region broadcast with {} XPUs (multi-process)",
            world_size
        );

        let bootstrap =
            OneCclBootstrap::generate(world_size).expect("Failed to generate bootstrap");
        let bootstrap_hex = hex_encode(&bootstrap.serialize());

        let children = spawn_ranks("multi_region", world_size, &bootstrap_hex);
        wait_for_all(children);

        println!("Multi-region broadcast test passed!");
    }

    #[test]
    #[cfg(feature = "testing-oneccl")]
    fn test_oneccl_broadcast_large_transfer() {
        let num_devices = xpu_device_count();
        if num_devices < 2 {
            println!("Skipping: {} XPUs available, need at least 2", num_devices);
            return;
        }

        if std::env::var("ONECCL_TEST_RANK").is_ok() {
            return;
        }

        let world_size = 2;
        println!(
            "Testing oneCCL large broadcast with {} XPUs (multi-process)",
            world_size
        );

        let bootstrap =
            OneCclBootstrap::generate(world_size).expect("Failed to generate bootstrap");
        let bootstrap_hex = hex_encode(&bootstrap.serialize());

        let children = spawn_ranks("large_transfer", world_size, &bootstrap_hex);
        wait_for_all(children);

        println!("Large transfer test passed!");
    }
}
