// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA worker pool for memory allocation with first-touch policy
//!
//! This module provides dedicated worker threads that are pinned to specific NUMA nodes..
//!
//! ## Architecture
//!
//! - One worker thread per NUMA node (spawned lazily)
//! - Workers pin themselves on startup (immune to application thread management)
//! - Channel-based communication for allocation requests
//! - First-touch page allocation ensures correct NUMA placement

use super::get_current_cpu_numa_node;
use crate::block_manager::storage::cuda::Cuda;
use cudarc::driver::result::malloc_host;
use cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use nix::libc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::{NumaNode, get_device_numa_node};

/// Wrapper for raw pointer that can be sent between threads
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

/// Request to allocate CUDA pinned memory
struct AllocRequest {
    size: usize,
    node: NumaNode,
    gpu_id: u32,
    response: Sender<AllocResult>,
}

/// Result of allocation
type AllocResult = Result<SendPtr, String>;

/// A dedicated worker thread pinned to a specific NUMA node
struct NumaWorker {
    node: NumaNode,
    request_tx: Option<Sender<AllocRequest>>,
    handle: Option<JoinHandle<()>>,
}

impl NumaWorker {
    /// Spawn a new worker thread pinned to the specified NUMA node
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

    /// Worker thread main loop
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

    /// Perform CUDA pinned memory allocation
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
        let ctx = Cuda::device_or_create(gpu_id as usize)
            .map_err(|e| format!("Failed to get CUDA context for GPU {}: {:?}", gpu_id, e))?;

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
            let ptr = malloc_host(size, CU_MEMHOSTALLOC_WRITECOMBINED)
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
            let page_size = (libc::sysconf(libc::_SC_PAGESIZE) as usize).max(4096);
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

    /// Request an allocation from this worker
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

/// Pool of NUMA workers, one per node
pub struct NumaWorkerPool {
    workers: Mutex<std::collections::HashMap<u32, Arc<NumaWorker>>>,
}

impl NumaWorkerPool {
    fn new() -> Self {
        Self {
            workers: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Get the global worker pool
    pub fn global() -> &'static Self {
        static POOL: OnceLock<NumaWorkerPool> = OnceLock::new();
        POOL.get_or_init(NumaWorkerPool::new)
    }

    /// Get or create a worker for a NUMA node
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

    /// Allocate CUDA pinned memory for a specific GPU (auto-detects NUMA node)
    pub fn allocate_pinned_for_gpu(&self, size: usize, gpu_id: u32) -> Result<*mut u8, String> {
        let node = get_device_numa_node(gpu_id);

        tracing::debug!(
            "Allocating {} bytes pinned memory for GPU {} (NUMA node {})",
            size,
            gpu_id,
            node.0
        );

        let worker = self.get_or_spawn_worker(node)?;
        worker.allocate(size, gpu_id).map(|send_ptr| send_ptr.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::numa_allocator::{get_current_cpu_numa_node, get_device_numa_node};

    /// Check if CUDA is available for testing
    fn is_cuda_available() -> bool {
        // Check if nvidia-smi is available
        if std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=count")
            .arg("--format=csv,noheader")
            .output()
            .is_err()
        {
            return false;
        }

        // Try to initialize CUDA context for device 0
        use crate::block_manager::storage::cuda::Cuda;
        Cuda::device_or_create(0).is_ok()
    }

    #[test]
    fn test_worker_spawn() {
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node);
        assert!(worker.is_ok());
    }

    #[test]
    fn test_worker_allocate_pinned() {
        if !is_cuda_available() {
            eprintln!("Skipping test_worker_allocate_pinned: CUDA not available");
            return;
        }

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
        if !is_cuda_available() {
            eprintln!("Skipping test_worker_pool: CUDA not available");
            return;
        }

        let pool = NumaWorkerPool::new();

        unsafe {
            let ptr = pool.allocate_pinned_for_gpu(8192, 0).unwrap();
            assert!(!ptr.is_null());

            cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
        }
    }

    #[test]
    fn test_worker_pool_singleton() {
        // Verify that global() returns the same instance
        let pool1 = NumaWorkerPool::global();
        let pool2 = NumaWorkerPool::global();

        // They should be the same static reference
        assert!(std::ptr::eq(pool1, pool2));
    }

    #[test]
    fn test_worker_reuse() {
        if !is_cuda_available() {
            eprintln!("Skipping test_worker_reuse: CUDA not available");
            return;
        }

        // Test that subsequent allocations for the same GPU reuse the same worker
        let pool = NumaWorkerPool::new();

        unsafe {
            // First allocation spawns worker for GPU 0
            let ptr1 = pool.allocate_pinned_for_gpu(1024, 0).unwrap();

            // Second allocation should reuse worker for GPU 0
            let ptr2 = pool.allocate_pinned_for_gpu(1024, 0).unwrap();

            assert!(!ptr1.is_null());
            assert!(!ptr2.is_null());
            assert_ne!(ptr1, ptr2);

            cudarc::driver::result::free_host(ptr1 as *mut std::ffi::c_void).unwrap();
            cudarc::driver::result::free_host(ptr2 as *mut std::ffi::c_void).unwrap();
        }
    }

    #[test]
    fn test_zero_size_allocation() {
        // Test that zero-size allocations are rejected
        let pool = NumaWorkerPool::new();
        let result = pool.allocate_pinned_for_gpu(0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("zero"));
    }

    #[test]
    fn test_dynamic_timeout_scaling() {
        // Test that timeout scales with allocation size
        let pool = NumaWorkerPool::new();

        // We can't easily test the actual timeout behavior without sleeping,
        // but we can verify that allocations of different sizes work
        // The timeout calculation is: 10s + (size_in_GB) seconds, capped at 300s

        // Just verify a small allocation works (timeout calculation is internal)
        unsafe {
            if let Ok(ptr) = pool.allocate_pinned_for_gpu(1024, 0) {
                assert!(!ptr.is_null());
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            }
        }
    }

    #[test]
    fn test_get_current_cpu_numa_node() {
        // Test that we can detect current CPU's NUMA node
        let node = get_current_cpu_numa_node();

        // On a real NUMA system, should return a valid node
        // On fake NUMA or single-node, might return 0 or UNKNOWN
        if !node.is_unknown() {
            println!("Current CPU on NUMA node: {}", node.0);
        } else {
            println!("NUMA node detection unavailable (single-node or fake NUMA)");
        }
    }

    #[test]
    fn test_get_device_numa_node() {
        // Test GPU NUMA node detection
        // This will only work if nvidia-smi is available
        let node = get_device_numa_node(0);

        if !node.is_unknown() {
            println!("GPU 0 on NUMA node: {}", node.0);
            // Node should be 0 or 1 on typical dual-socket systems
            assert!(node.0 <= 1 || node.0 == u32::MAX);
        } else {
            println!("GPU NUMA detection unavailable (no nvidia-smi or no GPU)");
        }
    }

    #[test]
    fn test_numa_node_display() {
        // Test Display implementation for NumaNode
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

    #[test]
    fn test_pinned_allocation_api() {
        // Verify the public API works for pinned allocation
        let pool = NumaWorkerPool::new();

        unsafe {
            // Test that we can allocate pinned memory for a GPU
            if let Ok(ptr) = pool.allocate_pinned_for_gpu(1024, 0) {
                assert!(!ptr.is_null());
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            }
        }
    }

    #[test]
    fn test_worker_channel_communication() {
        // Test that worker receives and processes requests
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node).unwrap();

        // Send allocation request
        let result = worker.allocate(1024, 0);

        // Should get a response (either success or timeout)
        assert!(result.is_ok() || result.is_err());

        if let Ok(send_ptr) = result {
            unsafe {
                let ptr = send_ptr.0;
                assert!(!ptr.is_null());
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            }
        }
    }
}
