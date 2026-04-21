// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA worker pool for memory allocation with first-touch policy.
//!
//! This module provides dedicated worker threads that are pinned to specific NUMA nodes.
//!
//! ## Architecture
//!
//! - One worker thread per NUMA node (spawned lazily)
//! - Workers pin themselves on startup (immune to application thread management)
//! - Channel-based communication for allocation requests
//! - First-touch page allocation ensures correct NUMA placement

use super::get_current_cpu_numa_node;
use cudarc::driver::result::malloc_host;
use cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP;
use nix::libc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::{NumaNode, get_device_numa_node};

/// Wrapper for raw pointer that can be sent between threads.
///
/// # Safety
///
/// This wrapper allows sending raw pointers across thread boundaries. The safety contract is:
/// - The pointer is allocated by the worker thread and returned to the caller
/// - The pointer is only dereferenced by the receiver (caller), never by the sender (worker)
/// - Ownership is transferred: the caller is responsible for deallocation
/// - The pointer remains valid for the lifetime expected by the caller
struct SendPtr(*mut u8);

// SAFETY: The pointer ownership is transferred from worker to caller.
// The worker never accesses the pointer after sending it.
unsafe impl Send for SendPtr {}

/// Request to allocate CUDA pinned memory on a specific NUMA node.
struct AllocRequest {
    /// Number of bytes to allocate.
    size: usize,
    /// Target NUMA node for allocation.
    node: NumaNode,
    /// CUDA device ID (for context binding).
    gpu_id: u32,
    /// Channel for sending back the allocation result.
    response: Sender<AllocResult>,
}

/// Result of allocation.
type AllocResult = Result<SendPtr, String>;

/// A dedicated worker thread pinned to a specific NUMA node.
struct NumaWorker {
    node: NumaNode,
    request_tx: Option<Sender<AllocRequest>>,
    handle: Option<JoinHandle<()>>,
}

impl NumaWorker {
    /// Spawn a new worker thread pinned to the specified NUMA node.
    fn spawn(node: NumaNode) -> Result<Self, String> {
        let (request_tx, request_rx) = channel();

        let handle = thread::Builder::new()
            .name(format!("numa-worker-{}", node.0))
            .spawn(move || {
                Self::worker_loop(node, request_rx);
            })
            .map_err(|e| format!("Failed to spawn worker thread: {}", e))?;

        Ok(Self {
            node,
            request_tx: Some(request_tx),
            handle: Some(handle),
        })
    }

    /// Worker thread main loop that processes allocation requests.
    ///
    /// On startup, the worker pins itself to the target NUMA node using
    /// `sched_setaffinity`. It then processes allocation requests in a loop
    /// until the channel is closed.
    fn worker_loop(node: NumaNode, requests: Receiver<AllocRequest>) {
        // First thing: pin this thread to the target NUMA node
        tracing::trace!("Pinning worker thread to node {}", node.0);
        if let Err(e) = super::pin_thread_to_numa_node(node) {
            tracing::error!("Failed to pin worker thread to node {}: {}", node.0, e);
            tracing::error!("Worker will continue but allocations may be suboptimal");
        } else {
            tracing::trace!("Successfully pinned worker thread to node {}", node.0);

            // `pin_thread_to_numa_node` uses `sched_setaffinity` to set the CPU affinity mask
            // but doesn't immediately migrate the thread. The scheduler will migrate at
            // the next opportunity (timer tick, yield, etc).
            // We yield once to give the scheduler a chance to migrate before we verify.
            // This is primarily for accurate logging - allocations will happen on the right CPU
            // regardless since the affinity mask prevents running on wrong CPUs.
            thread::yield_now();
            thread::sleep(Duration::from_millis(1));

            // Verify we're on the right node
            let current_node = super::get_current_cpu_numa_node();
            tracing::trace!("Current node after pinning: {}", current_node.0);
            if current_node != node {
                tracing::warn!(
                    "Worker thread on node {} after pinning (expected {})",
                    current_node.0,
                    node.0
                );
            } else {
                tracing::trace!("NUMA worker thread for node {} started and pinned", node.0);
            }
        }

        // Process allocation requests
        loop {
            tracing::trace!("Worker waiting for request on node {}", node.0);
            match requests.recv() {
                Ok(req) => {
                    tracing::trace!(
                        "Worker received CUDA pinned allocation request on node {}",
                        node.0
                    );
                    let result = Self::do_cuda_pinned_allocation(req.size, req.node, req.gpu_id);
                    match result {
                        Ok(SendPtr(ptr)) => {
                            if let Err(_e) = req.response.send(Ok(SendPtr(ptr))) {
                                // Receiver gone: free to avoid leak
                                tracing::warn!(
                                    "Receiver dropped before receiving allocation, freeing {} bytes at {:p}",
                                    req.size,
                                    ptr
                                );
                                unsafe {
                                    let _ = cudarc::driver::result::free_host(
                                        ptr as *mut std::ffi::c_void,
                                    );
                                }
                            }
                        }
                        Err(err) => {
                            let _ = req.response.send(Err(err));
                        }
                    }
                }
                Err(_) => {
                    // Channel closed, exit worker
                    tracing::trace!(
                        "NUMA worker for node {} shutting down (channel closed)",
                        node.0
                    );
                    break;
                }
            }
        }
    }

    /// Perform CUDA pinned memory allocation.
    fn do_cuda_pinned_allocation(size: usize, node: NumaNode, gpu_id: u32) -> AllocResult {
        if size == 0 {
            return Err("Cannot allocate zero bytes".to_string());
        }

        // Verify we're on the correct NUMA node BEFORE allocation
        let node_before = get_current_cpu_numa_node();
        if node_before != node {
            tracing::warn!(
                "Worker thread moved! Expected NUMA node {}, currently on node {}",
                node.0,
                node_before.0
            );
        }

        // Get or create CUDA context for this GPU
        let ctx = crate::device::cuda_context(gpu_id)
            .map_err(|e| format!("Failed to create CUDA context for device {}: {}", gpu_id, e))?;

        unsafe {
            // Bind CUDA context to this worker thread before allocation
            // This ensures malloc_host has a valid context to work with
            ctx.bind_to_thread()
                .map_err(|e| format!("Failed to bind CUDA context to worker thread: {:?}", e))?;

            // Verify thread is still on correct node after CUDA context binding
            let node_after_ctx = get_current_cpu_numa_node();
            if node_after_ctx != node {
                tracing::warn!(
                    "Thread moved after CUDA context bind! Expected node {}, now on node {}",
                    node.0,
                    node_after_ctx.0
                );
            }

            // Allocate CUDA pinned memory
            // This is called from the pinned worker thread, so pages will be
            // allocated on the correct NUMA node via first-touch
            let ptr = malloc_host(size, CU_MEMHOSTALLOC_DEVICEMAP)
                .map_err(|e| format!("malloc_host failed: {:?}", e))?;

            let ptr = ptr as *mut u8;

            if ptr.is_null() {
                return Err("malloc_host returned null".to_string());
            }

            // Verify thread is STILL on correct node before touching pages
            let node_before_touch = get_current_cpu_numa_node();
            if node_before_touch != node {
                tracing::error!(
                    "Thread on wrong node before first-touch! Expected {}, on node {} - memory will be misplaced!",
                    node.0,
                    node_before_touch.0
                );
            }

            // Touch one byte per page to trigger first-touch policy efficiently
            // This is much faster than zeroing the entire region for large allocations
            let page_size = match libc::sysconf(libc::_SC_PAGESIZE) {
                n if n > 0 => n as usize,
                _ => 4096,
            };
            let mut offset = 0usize;
            while offset < size {
                std::ptr::write_volatile(ptr.add(offset), 0);
                offset = offset.saturating_add(page_size);
            }
            // Ensure the last page is touched
            if size > 0 && !size.is_multiple_of(page_size) {
                std::ptr::write_volatile(ptr.add(size - 1), 0);
            }

            // Verify final node after touching
            let node_after_touch = get_current_cpu_numa_node();

            tracing::trace!(
                "Worker allocated {} bytes (CUDA pinned) on GPU {} (target NUMA node {}) at {:p} - thread nodes: before={} after_ctx={} before_touch={} after_touch={}",
                size,
                gpu_id,
                node.0,
                ptr,
                node_before.0,
                node_after_ctx.0,
                node_before_touch.0,
                node_after_touch.0
            );

            Ok(SendPtr(ptr))
        }
    }

    /// Request an allocation from this worker.
    fn allocate(&self, size: usize, gpu_id: u32) -> AllocResult {
        let (response_tx, response_rx) = channel();

        let request = AllocRequest {
            size,
            node: self.node,
            gpu_id,
            response: response_tx,
        };

        self.request_tx
            .as_ref()
            .ok_or_else(|| "Worker has been shut down".to_string())?
            .send(request)
            .map_err(|_| "Worker thread has died".to_string())?;

        // Wait for response with dynamic timeout based on allocation size
        // Large allocations take time: we account for ~1 second per GB to touch pages
        // Add 10 second base + 1 second per GB
        let timeout_secs = 10u64 + (size as u64 / (1024 * 1024 * 1024));
        let timeout = Duration::from_secs(timeout_secs.clamp(10, 300)); // Clamp to 10-300 seconds

        tracing::trace!(
            "Worker pool waiting for allocation of {} MB with timeout of {} seconds",
            size / (1024 * 1024),
            timeout.as_secs()
        );

        response_rx
            .recv_timeout(timeout)
            .map_err(|e| format!("Worker timeout after {} seconds: {}", timeout.as_secs(), e))?
    }
}

impl Drop for NumaWorker {
    fn drop(&mut self) {
        tracing::trace!("Dropping NUMA worker for node {}", self.node.0);

        // Drop request_tx FIRST to close the channel
        // This causes recv() in worker thread to return Err and exit
        self.request_tx.take();
        tracing::trace!("Channel closed for worker node {}", self.node.0);

        // Now the worker thread will exit its loop
        if let Some(handle) = self.handle.take() {
            tracing::trace!("Waiting for worker thread {} to join", self.node.0);
            let _ = handle.join();
            tracing::trace!("Worker thread {} joined", self.node.0);
        }
    }
}

/// Pool of NUMA workers, one per node.
///
/// This pool manages dedicated worker threads that are pinned to specific NUMA nodes.
/// When you request an allocation for a GPU, the pool automatically determines the
/// GPU's NUMA node and routes the request to the appropriate worker.
pub struct NumaWorkerPool {
    workers: Mutex<std::collections::HashMap<u32, Arc<NumaWorker>>>,
}

impl NumaWorkerPool {
    fn new() -> Self {
        Self {
            workers: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Get the global worker pool.
    ///
    /// The pool is created lazily on first access and lives for the entire process lifetime.
    pub fn global() -> &'static Self {
        static POOL: OnceLock<NumaWorkerPool> = OnceLock::new();
        POOL.get_or_init(NumaWorkerPool::new)
    }

    /// Get or create a worker for a NUMA node.
    fn get_or_spawn_worker(&self, node: NumaNode) -> Result<Arc<NumaWorker>, String> {
        let mut workers = self.workers.lock().unwrap();

        if let Some(worker) = workers.get(&node.0) {
            return Ok(worker.clone());
        }

        // Spawn new worker
        let worker = NumaWorker::spawn(node)?;
        let worker = Arc::new(worker);
        workers.insert(node.0, worker.clone());

        tracing::trace!("Spawned NUMA worker for node {}", node.0);

        Ok(worker)
    }

    /// Allocate CUDA pinned memory for a specific GPU (auto-detects NUMA node).
    ///
    /// This method:
    /// 1. Determines the GPU's NUMA node via CUDA driver PCI attributes + sysfs
    /// 2. Routes the allocation to a worker pinned to that node
    /// 3. The worker allocates and touches pages to ensure first-touch placement
    ///
    /// Returns `None` if the GPU's NUMA node cannot be determined, signaling
    /// the caller to fall back to non-NUMA allocation.
    ///
    /// # Arguments
    /// * `size` - Number of bytes to allocate
    /// * `gpu_id` - CUDA device ID
    ///
    /// # Returns
    /// `Some(ptr)` on success, `None` if NUMA node is unknown (caller should
    /// use non-NUMA allocation). Returns `Err` on allocation failure.
    pub fn allocate_pinned_for_gpu(
        &self,
        size: usize,
        gpu_id: u32,
    ) -> Result<Option<*mut u8>, String> {
        let node = match get_device_numa_node(gpu_id) {
            Some(node) => node,
            None => {
                tracing::debug!(
                    "NUMA node unknown for GPU {}, skipping NUMA-aware allocation",
                    gpu_id
                );
                return Ok(None);
            }
        };

        tracing::debug!(
            "Allocating {} bytes pinned memory for GPU {} (NUMA node {})",
            size,
            gpu_id,
            node.0
        );

        let worker = self.get_or_spawn_worker(node)?;
        worker
            .allocate(size, gpu_id)
            .map(|send_ptr| Some(send_ptr.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa::get_current_cpu_numa_node;

    #[test]
    fn test_worker_spawn() {
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node);
        assert!(worker.is_ok());
    }

    #[test]
    fn test_worker_pool_singleton() {
        let pool1 = NumaWorkerPool::global();
        let pool2 = NumaWorkerPool::global();
        assert!(std::ptr::eq(pool1, pool2));
    }

    #[test]
    fn test_get_current_cpu_numa_node() {
        let node = get_current_cpu_numa_node();
        if !node.is_unknown() {
            println!("Current CPU on NUMA node: {}", node.0);
        } else {
            println!("NUMA node detection unavailable (single-node or fake NUMA)");
        }
    }

    #[test]
    fn test_numa_node_display() {
        let node = NumaNode(0);
        assert_eq!(format!("{}", node), "NumaNode(0)");

        let unknown = NumaNode::UNKNOWN;
        assert_eq!(format!("{}", unknown), "UNKNOWN");
    }

    #[test]
    fn test_numa_node_is_unknown() {
        let valid = NumaNode(0);
        assert!(!valid.is_unknown());

        let unknown = NumaNode::UNKNOWN;
        assert!(unknown.is_unknown());
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod cuda_tests {
    use super::*;
    use crate::numa::get_device_numa_node;

    #[test]
    fn test_worker_allocate_pinned() {
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node).unwrap();

        let send_ptr = worker.allocate(4096, 0).unwrap();
        let ptr = send_ptr.0;
        assert!(!ptr.is_null());

        unsafe {
            cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
        }
    }

    #[test]
    fn test_worker_pool() {
        let pool = NumaWorkerPool::new();

        match pool.allocate_pinned_for_gpu(8192, 0).unwrap() {
            Some(ptr) => unsafe {
                assert!(!ptr.is_null());
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            },
            None => {
                println!(
                    "NUMA node unknown for GPU 0, allocation skipped (expected on single-socket)"
                );
            }
        }
    }

    #[test]
    fn test_worker_reuse() {
        let pool = NumaWorkerPool::new();

        // If NUMA node is unknown, both calls return None — that's fine
        let r1 = pool.allocate_pinned_for_gpu(1024, 0).unwrap();
        let r2 = pool.allocate_pinned_for_gpu(1024, 0).unwrap();

        match (r1, r2) {
            (Some(ptr1), Some(ptr2)) => unsafe {
                assert!(!ptr1.is_null());
                assert!(!ptr2.is_null());
                assert_ne!(ptr1, ptr2);
                cudarc::driver::result::free_host(ptr1 as *mut std::ffi::c_void).unwrap();
                cudarc::driver::result::free_host(ptr2 as *mut std::ffi::c_void).unwrap();
            },
            (None, None) => {
                println!("NUMA node unknown, both allocations skipped");
            }
            _ => panic!("inconsistent NUMA detection between two calls for same GPU"),
        }
    }

    #[test]
    fn test_zero_size_allocation_with_known_node() {
        // Zero-size is rejected by the worker, but only if NUMA node is known.
        // If NUMA node is unknown, allocate_pinned_for_gpu returns Ok(None) before
        // reaching the worker.
        let pool = NumaWorkerPool::new();
        let result = pool.allocate_pinned_for_gpu(0, 0);
        match result {
            Ok(None) => {
                println!("NUMA node unknown, zero-size check not reached");
            }
            Err(e) => {
                assert!(e.contains("zero"));
            }
            Ok(Some(_)) => panic!("zero-size allocation should not succeed"),
        }
    }

    #[test]
    fn test_get_device_numa_node() {
        let node = get_device_numa_node(0);
        match node {
            Some(n) => {
                assert!(n.0 < 16, "NUMA node {} seems unreasonably high", n.0);
                println!("GPU 0 on NUMA node: {}", n.0);
            }
            None => {
                println!("GPU 0 has no determinable NUMA node");
            }
        }
    }

    #[test]
    fn test_pinned_allocation_api() {
        let pool = NumaWorkerPool::new();

        if let Some(ptr) = pool.allocate_pinned_for_gpu(1024, 0).unwrap() {
            assert!(!ptr.is_null());
            unsafe {
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            }
        }
    }

    #[test]
    fn test_worker_channel_communication() {
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node).unwrap();

        let send_ptr = worker.allocate(1024, 0).unwrap();
        let ptr = send_ptr.0;
        assert!(!ptr.is_null());

        unsafe {
            cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
        }
    }
}
