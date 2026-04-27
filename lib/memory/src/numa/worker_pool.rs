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
use nix::libc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::NumaNode;

/// Backend-agnostic allocator for NUMA-pinned host memory.
///
/// Implementations are called on a worker thread that is already pinned to
/// the target NUMA node. First-touch page walking is handled by the worker
/// pool after allocation.
pub trait PinnedAllocator: Send + Sync + 'static {
    /// Allocate `size` bytes of host memory accessible by the device.
    ///
    /// Called on a NUMA-pinned worker thread. The implementation should
    /// perform the backend-specific allocation (e.g., `cuMemHostAlloc`,
    /// `sycl::malloc_host`) but does NOT need to do first-touch — the
    /// worker handles that.
    fn alloc_pinned(&self, size: usize) -> Result<*mut u8, String>;

    /// Free a pointer previously returned by [`alloc_pinned`].
    fn free_pinned(&self, ptr: *mut u8) -> Result<(), String>;
}

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

/// Request to allocate pinned memory on a specific NUMA node.
struct AllocRequest {
    /// Number of bytes to allocate.
    size: usize,
    /// Target NUMA node for allocation.
    node: NumaNode,
    /// Backend-specific allocator (CUDA, SYCL, etc.).
    allocator: Arc<dyn PinnedAllocator>,
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
                        "Worker received pinned allocation request on node {}",
                        node.0
                    );
                    let result = Self::do_pinned_allocation(
                        req.size, req.node, &*req.allocator,
                    );
                    match result {
                        Ok(SendPtr(ptr)) => {
                            if let Err(_e) = req.response.send(Ok(SendPtr(ptr))) {
                                // Receiver gone: free to avoid leak
                                tracing::warn!(
                                    "Receiver dropped before receiving allocation, freeing {} bytes at {:p}",
                                    req.size,
                                    ptr
                                );
                                if let Err(e) = req.allocator.free_pinned(ptr) {
                                    tracing::error!("Failed to free leaked allocation: {}", e);
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

    /// Perform backend-agnostic pinned memory allocation with first-touch.
    ///
    /// The allocator handles the backend-specific allocation (CUDA/SYCL/etc),
    /// while this method handles NUMA verification and first-touch page walking.
    fn do_pinned_allocation(
        size: usize,
        node: NumaNode,
        allocator: &dyn PinnedAllocator,
    ) -> AllocResult {
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

        // Delegate to backend-specific allocator
        let ptr = allocator.alloc_pinned(size)?;

        if ptr.is_null() {
            return Err("Allocator returned null pointer".to_string());
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
        // SAFETY: ptr was just allocated with at least `size` bytes by the allocator.
        unsafe {
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
        }

        // Verify final node after touching
        let node_after_touch = get_current_cpu_numa_node();

        tracing::trace!(
            "Worker allocated {} bytes (target NUMA node {}) at {:p} - thread nodes: before={} before_touch={} after_touch={}",
            size,
            node.0,
            ptr,
            node_before.0,
            node_before_touch.0,
            node_after_touch.0
        );

        Ok(SendPtr(ptr))
    }

    /// Request an allocation from this worker.
    fn allocate(&self, size: usize, allocator: Arc<dyn PinnedAllocator>) -> AllocResult {
        let (response_tx, response_rx) = channel();

        let request = AllocRequest {
            size,
            node: self.node,
            allocator,
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

    /// Allocate pinned memory on a specific NUMA node using the provided allocator.
    ///
    /// This is the backend-agnostic entry point. The caller resolves the NUMA
    /// node (e.g., via PCI BDF → sysfs) and provides a [`PinnedAllocator`]
    /// for the backend-specific allocation.
    ///
    /// The worker thread pinned to `node` will:
    /// 1. Call `allocator.alloc_pinned(size)` on the pinned thread
    /// 2. First-touch all pages to ensure correct NUMA placement
    ///
    /// # Arguments
    /// * `size` - Number of bytes to allocate
    /// * `node` - Target NUMA node
    /// * `allocator` - Backend-specific allocator (CUDA, SYCL, etc.)
    pub fn allocate_pinned_on_node(
        &self,
        size: usize,
        node: NumaNode,
        allocator: Arc<dyn PinnedAllocator>,
    ) -> Result<*mut u8, String> {
        tracing::debug!(
            "Allocating {} bytes pinned memory on NUMA node {}",
            size,
            node.0
        );

        let worker = self.get_or_spawn_worker(node)?;
        worker
            .allocate(size, allocator)
            .map(|send_ptr| send_ptr.0)
    }

    /// Allocate pinned memory for a GPU/device (NUMA node auto-detected from PCI address).
    ///
    /// Unified entry point for ALL backends (CUDA, SYCL/XPU, etc.):
    /// 1. Resolves NUMA node from PCI BDF address (sysfs → vendor-smi fallback)
    /// 2. Routes allocation to the NUMA-pinned worker thread
    /// 3. Caller provides a [`PinnedAllocator`] for backend-specific allocation
    ///
    /// Both CUDA and SYCL callers use the identical pattern:
    /// ```ignore
    /// // cuda/mod.rs
    /// allocate_pinned_for_gpu(size, &pci, Arc::new(CudaPinnedAllocator { .. }))
    /// // sycl/mod.rs
    /// allocate_pinned_for_gpu(size, &pci, Arc::new(SyclPinnedAllocator { .. }))
    /// ```
    ///
    /// Returns `None` if the NUMA node cannot be determined, signaling
    /// the caller to fall back to non-NUMA allocation.
    ///
    /// # Arguments
    /// * `size` - Number of bytes to allocate
    /// * `pci_address` - PCI BDF address
    /// * `allocator` - Backend-specific allocator (CUDA, SYCL, etc.)
    pub fn allocate_pinned_for_gpu(
        &self,
        size: usize,
        pci_address: &str,
        allocator: Arc<dyn PinnedAllocator>,
    ) -> Result<Option<*mut u8>, String> {
        let node = match super::get_numa_node_for_pci_address(pci_address) {
            Some(node) => node,
            None => {
                tracing::debug!(
                    "NUMA node unknown for PCI {}, skipping NUMA-aware allocation",
                    pci_address
                );
                return Ok(None);
            }
        };

        tracing::debug!(
            "Allocating {} bytes pinned memory for PCI {} (NUMA node {})",
            size,
            pci_address,
            node.0
        );

        self.allocate_pinned_on_node(size, node, allocator)
            .map(Some)
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

    /// Test CUDA allocator that mirrors the real CudaPinnedAllocator
    /// in kvbm-physical::device::cuda.
    struct TestCudaAllocator {
        gpu_id: u32,
    }

    impl PinnedAllocator for TestCudaAllocator {
        fn alloc_pinned(&self, size: usize) -> Result<*mut u8, String> {
            use cudarc::driver::result::malloc_host;
            use cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP;

            let ctx = crate::numa::cuda_context(self.gpu_id)
                .map_err(|e| format!("CUDA context: {}", e))?;
            unsafe {
                ctx.bind_to_thread()
                    .map_err(|e| format!("bind_to_thread: {:?}", e))?;
                malloc_host(size, CU_MEMHOSTALLOC_DEVICEMAP)
                    .map(|p| p as *mut u8)
                    .map_err(|e| format!("malloc_host: {:?}", e))
            }
        }

        fn free_pinned(&self, ptr: *mut u8) -> Result<(), String> {
            unsafe {
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void)
                    .map_err(|e| format!("free_host: {:?}", e))
            }
        }
    }

    fn test_allocator() -> Arc<dyn PinnedAllocator> {
        Arc::new(TestCudaAllocator { gpu_id: 0 })
    }

    fn test_pci_and_allocator() -> Option<(String, Arc<dyn PinnedAllocator>)> {
        let pci = crate::numa::get_pci_bus_address_from_cuda(0)?;
        Some((pci, test_allocator()))
    }

    #[test]
    fn test_worker_allocate_pinned() {
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node).unwrap();

        let send_ptr = worker.allocate(4096, test_allocator()).unwrap();
        let ptr = send_ptr.0;
        assert!(!ptr.is_null());

        unsafe {
            cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
        }
    }

    #[test]
    fn test_worker_pool() {
        let pool = NumaWorkerPool::new();
        let (pci, allocator) = match test_pci_and_allocator() {
            Some(v) => v,
            None => { println!("No PCI address for GPU 0, skipping"); return; }
        };

        match pool.allocate_pinned_for_gpu(8192, &pci, allocator).unwrap() {
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
        let pci = match crate::numa::get_pci_bus_address_from_cuda(0) {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        // If NUMA node is unknown, both calls return None — that’s fine
        let r1 = pool.allocate_pinned_for_gpu(1024, &pci, test_allocator()).unwrap();
        let r2 = pool.allocate_pinned_for_gpu(1024, &pci, test_allocator()).unwrap();

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
        let pool = NumaWorkerPool::new();
        let (pci, allocator) = match test_pci_and_allocator() {
            Some(v) => v,
            None => { println!("No PCI address, skipping"); return; }
        };
        let result = pool.allocate_pinned_for_gpu(0, &pci, allocator);
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
        let (pci, allocator) = match test_pci_and_allocator() {
            Some(v) => v,
            None => { println!("No PCI address, skipping"); return; }
        };

        if let Some(ptr) = pool.allocate_pinned_for_gpu(1024, &pci, allocator).unwrap() {
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

        let send_ptr = worker.allocate(1024, test_allocator()).unwrap();
        let ptr = send_ptr.0;
        assert!(!ptr.is_null());

        unsafe {
            cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
        }
    }
}


#[cfg(all(test, feature = "testing-sycl"))]
mod sycl_tests {
    use super::*;
    use oneapi_rs::safe::{SyclDevice, SyclQueue};

    /// Test SYCL allocator that mirrors the real SyclPinnedAllocator
    /// in kvbm-physical::device::sycl.
    struct TestSyclAllocator {
        queue: Arc<SyclQueue>,
    }

    impl PinnedAllocator for TestSyclAllocator {
        fn alloc_pinned(&self, size: usize) -> Result<*mut u8, String> {
            self.queue
                .malloc_host(size)
                .map(|p| p as *mut u8)
                .map_err(|e| format!("SYCL malloc_host: {}", e))
        }

        fn free_pinned(&self, ptr: *mut u8) -> Result<(), String> {
            self.queue
                .free_raw(ptr as *mut std::ffi::c_void)
                .map_err(|e| format!("SYCL free_raw: {}", e))
        }
    }

    fn sycl_queue() -> Option<Arc<SyclQueue>> {
        SyclQueue::new_for_device_ordinal(0).ok()
    }

    fn test_allocator(queue: &Arc<SyclQueue>) -> Arc<dyn PinnedAllocator> {
        Arc::new(TestSyclAllocator {
            queue: Arc::clone(queue),
        })
    }

    fn test_pci_address() -> Option<String> {
        let dev = SyclDevice::by_ordinal(0).ok()?;
        dev.info().ok()?.pci_address
    }

    #[test]
    fn test_sycl_worker_allocate_pinned() {
        let queue = match sycl_queue() {
            Some(q) => q,
            None => { println!("No SYCL device, skipping"); return; }
        };
        let node = NumaNode(0);
        let worker = NumaWorker::spawn(node).unwrap();

        let send_ptr = worker.allocate(4096, test_allocator(&queue)).unwrap();
        let ptr = send_ptr.0;
        assert!(!ptr.is_null());

        // Free via the allocator
        queue.free_raw(ptr as *mut std::ffi::c_void).unwrap();
    }

    #[test]
    fn test_sycl_allocate_pinned_for_gpu() {
        let queue = match sycl_queue() {
            Some(q) => q,
            None => { println!("No SYCL device, skipping"); return; }
        };
        let pci = match test_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address for XPU 0, skipping"); return; }
        };

        let pool = NumaWorkerPool::new();
        match pool.allocate_pinned_for_gpu(8192, &pci, test_allocator(&queue)).unwrap() {
            Some(ptr) => {
                assert!(!ptr.is_null());
                println!("SYCL NUMA-aware allocation succeeded for PCI {}", pci);
                queue.free_raw(ptr as *mut std::ffi::c_void).unwrap();
            }
            None => {
                println!(
                    "NUMA node unknown for XPU PCI {}, allocation skipped (expected on single-socket)",
                    pci
                );
            }
        }
    }

    #[test]
    fn test_sycl_worker_reuse() {
        let queue = match sycl_queue() {
            Some(q) => q,
            None => { println!("No SYCL device, skipping"); return; }
        };
        let pci = match test_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        let pool = NumaWorkerPool::new();
        let r1 = pool.allocate_pinned_for_gpu(1024, &pci, test_allocator(&queue)).unwrap();
        let r2 = pool.allocate_pinned_for_gpu(1024, &pci, test_allocator(&queue)).unwrap();

        match (r1, r2) {
            (Some(ptr1), Some(ptr2)) => {
                assert!(!ptr1.is_null());
                assert!(!ptr2.is_null());
                assert_ne!(ptr1, ptr2);
                queue.free_raw(ptr1 as *mut std::ffi::c_void).unwrap();
                queue.free_raw(ptr2 as *mut std::ffi::c_void).unwrap();
            }
            (None, None) => {
                println!("NUMA node unknown, both allocations skipped");
            }
            _ => panic!("inconsistent NUMA detection between two calls for same XPU"),
        }
    }

    #[test]
    fn test_sycl_zero_size_allocation() {
        let queue = match sycl_queue() {
            Some(q) => q,
            None => { println!("No SYCL device, skipping"); return; }
        };
        let pci = match test_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        let pool = NumaWorkerPool::new();
        let result = pool.allocate_pinned_for_gpu(0, &pci, test_allocator(&queue));
        match result {
            Ok(None) => println!("NUMA node unknown, zero-size check not reached"),
            Err(e) => assert!(e.contains("zero")),
            Ok(Some(_)) => panic!("zero-size allocation should not succeed"),
        }
    }

    #[test]
    fn test_sycl_pci_address_available() {
        match test_pci_address() {
            Some(pci) => {
                println!("XPU 0 PCI address: {}", pci);
                // Verify format: DDDD:BB:DD.F
                assert!(pci.contains(':'), "PCI address should contain ':'");
                assert!(pci.contains('.'), "PCI address should contain '.'");
            }
            None => {
                println!("No PCI address for XPU 0 (PCI extension not available)");
            }
        }
    }

    #[test]
    fn test_sycl_numa_node_for_xpu() {
        let pci = match test_pci_address() {
            Some(p) => p,
            None => { println!("No PCI address, skipping"); return; }
        };

        match crate::numa::get_numa_node_for_pci_address(&pci) {
            Some(node) => {
                assert!(node.0 < 16, "NUMA node {} seems unreasonably high", node.0);
                println!("XPU 0 (PCI {}) on NUMA node: {}", pci, node.0);
            }
            None => {
                println!("XPU 0 (PCI {}) has no determinable NUMA node", pci);
            }
        }
    }
}

