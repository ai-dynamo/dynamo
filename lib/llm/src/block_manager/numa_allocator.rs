// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-export NUMA utilities from dynamo-memory.
pub use dynamo_memory::numa::*;

#[cfg(test)]
mod tests {
    use super::*;

    // ── NumaNode tests ──────────────────────────────────────────────────

    #[test]
    fn test_numa_node_equality() {
        let node0a = NumaNode(0);
        let node0b = NumaNode(0);
        let node1 = NumaNode(1);

        assert_eq!(node0a, node0b);
        assert_ne!(node0a, node1);
    }

    #[test]
    fn test_numa_node_unknown() {
        let unknown = NumaNode::UNKNOWN;
        assert!(unknown.is_unknown());
        assert_eq!(unknown.0, u32::MAX);

        let valid = NumaNode(0);
        assert!(!valid.is_unknown());
    }

    #[test]
    fn test_numa_node_display() {
        assert_eq!(format!("{}", NumaNode(0)), "NumaNode(0)");
        assert_eq!(format!("{}", NumaNode(7)), "NumaNode(7)");
        assert_eq!(format!("{}", NumaNode::UNKNOWN), "UNKNOWN");
    }

    #[test]
    fn test_numa_node_serialization() {
        let node = NumaNode(1);
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: NumaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, deserialized);
    }

    #[test]
    fn test_numa_node_hash() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(NumaNode(0), "node0");
        map.insert(NumaNode(1), "node1");

        assert_eq!(map.get(&NumaNode(0)), Some(&"node0"));
        assert_eq!(map.get(&NumaNode(1)), Some(&"node1"));
        assert_eq!(map.get(&NumaNode(2)), None);
    }

    #[test]
    fn test_numa_node_copy_clone() {
        let node1 = NumaNode(5);
        let node2 = node1; // Copy
        let node3 = node1; // Clone

        assert_eq!(node1, node2);
        assert_eq!(node1, node3);
        assert_eq!(node2, node3);
    }

    // ── System detection tests ──────────────────────────────────────────

    #[test]
    fn test_get_current_cpu_numa_node() {
        let node = get_current_cpu_numa_node();

        if !node.is_unknown() {
            assert!(node.0 < 8, "NUMA node {} seems unreasonably high", node.0);
        }
    }

    #[test]
    fn test_get_device_numa_node_valid_gpu() {
        let node = get_device_numa_node(0);
        println!("GPU 0 detected on NUMA node: {}", node.0);
    }

    // ── Worker pool tests ───────────────────────────────────────────────
    //
    // NumaWorker and NumaWorkerPool::new() are private in dynamo-memory,
    // so these tests go through the public NumaWorkerPool::global() API.

    /// Check if CUDA is available for testing
    fn is_cuda_available() -> bool {
        if std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=count")
            .arg("--format=csv,noheader")
            .output()
            .is_err()
        {
            return false;
        }

        crate::block_manager::storage::cuda::Cuda::device_or_create(0).is_ok()
    }

    #[test]
    fn test_worker_pool_singleton() {
        let pool1 = worker_pool::NumaWorkerPool::global();
        let pool2 = worker_pool::NumaWorkerPool::global();
        assert!(std::ptr::eq(pool1, pool2));
    }

    #[test]
    fn test_worker_pool_allocate() {
        if !is_cuda_available() {
            eprintln!("Skipping test_worker_pool_allocate: CUDA not available");
            return;
        }

        let pool = worker_pool::NumaWorkerPool::global();

        unsafe {
            let ptr = pool.allocate_pinned_for_gpu(8192, 0).unwrap();
            assert!(!ptr.is_null());

            cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
        }
    }

    #[test]
    fn test_worker_pool_reuse() {
        if !is_cuda_available() {
            eprintln!("Skipping test_worker_pool_reuse: CUDA not available");
            return;
        }

        let pool = worker_pool::NumaWorkerPool::global();

        unsafe {
            let ptr1 = pool.allocate_pinned_for_gpu(1024, 0).unwrap();
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
        if !is_cuda_available() {
            eprintln!("Skipping test_zero_size_allocation: CUDA not available");
            return;
        }

        let pool = worker_pool::NumaWorkerPool::global();
        let result = pool.allocate_pinned_for_gpu(0, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("zero"));
    }

    #[test]
    fn test_pinned_allocation_api() {
        let pool = worker_pool::NumaWorkerPool::global();

        unsafe {
            if let Ok(ptr) = pool.allocate_pinned_for_gpu(1024, 0) {
                assert!(!ptr.is_null());
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            }
        }
    }
}
