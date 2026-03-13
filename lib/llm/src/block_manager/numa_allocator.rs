// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-export NUMA utilities from dynamo-memory.
pub use dynamo_memory::numa::*;

/// Check if NUMA optimization is explicitly opted-in for the block manager.
///
/// Set `DYN_KVBM_ENABLE_NUMA=1` to enable NUMA-aware allocation in the
/// KV cache block manager. This is opt-in because the block manager
/// manages its own pinned memory allocations separately from `PinnedStorage`.
///
/// The global kill switch `DYN_MEMORY_DISABLE_NUMA` always takes precedence:
/// if it is set truthy, this function returns `false` regardless of
/// `DYN_KVBM_ENABLE_NUMA`.
///
/// TODO(KVBM-336): remove this function in the future
#[deprecated(
    since = "1.0.0",
    note = "Use dynamo_memory::numa::is_numa_enabled instead"
)]
pub fn is_numa_enabled() -> bool {
    // Global kill switch always wins
    if is_numa_disabled() {
        return false;
    }
    matches!(
        std::env::var("DYN_KVBM_ENABLE_NUMA").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let node2 = node1;
        let node3 = node1;

        assert_eq!(node1, node2);
        assert_eq!(node1, node3);
        assert_eq!(node2, node3);
    }

    #[test]
    fn test_get_current_cpu_numa_node() {
        let node = get_current_cpu_numa_node();
        if !node.is_unknown() {
            assert!(node.0 < 8, "NUMA node {} seems unreasonably high", node.0);
        }
    }

    #[test]
    fn test_worker_pool_singleton() {
        let pool1 = worker_pool::NumaWorkerPool::global();
        let pool2 = worker_pool::NumaWorkerPool::global();
        assert!(std::ptr::eq(pool1, pool2));
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod cuda_tests {
    use super::*;

    #[test]
    fn test_get_device_numa_node_valid_gpu() {
        match get_device_numa_node(0) {
            Some(node) => println!("GPU 0 detected on NUMA node: {}", node.0),
            None => println!("GPU 0 has no determinable NUMA node"),
        }
    }

    #[test]
    fn test_worker_pool_allocate() {
        let pool = worker_pool::NumaWorkerPool::global();

        match pool.allocate_pinned_for_gpu(8192, 0).unwrap() {
            Some(ptr) => unsafe {
                assert!(!ptr.is_null());
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            },
            None => {
                println!("NUMA node unknown for GPU 0, allocation skipped");
            }
        }
    }

    #[test]
    fn test_worker_pool_reuse() {
        let pool = worker_pool::NumaWorkerPool::global();

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
    fn test_zero_size_allocation() {
        let pool = worker_pool::NumaWorkerPool::global();
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
    fn test_pinned_allocation_api() {
        let pool = worker_pool::NumaWorkerPool::global();

        if let Ok(Some(ptr)) = pool.allocate_pinned_for_gpu(1024, 0) {
            assert!(!ptr.is_null());
            unsafe {
                cudarc::driver::result::free_host(ptr as *mut std::ffi::c_void).unwrap();
            }
        }
    }
}
