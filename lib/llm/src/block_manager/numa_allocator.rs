// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NUMA-aware memory allocation utilities.
//!
//! This module re-exports the NUMA utilities from `dynamo-memory` for use in the block manager.
//! See [`dynamo_memory::numa`] for full documentation.

// Re-export everything from dynamo-memory's numa module
pub use dynamo_memory::numa::topology;
pub use dynamo_memory::numa::worker_pool;
pub use dynamo_memory::numa::{
    NumaNode, get_current_cpu_numa_node, get_device_numa_node, is_numa_enabled,
    pin_thread_to_numa_node,
};

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
        // Verify NumaNode can be serialized (important for benchmarking)
        let node = NumaNode(1);
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: NumaNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, deserialized);
    }

    #[test]
    fn test_get_current_cpu_numa_node() {
        // Should either return a valid node or UNKNOWN
        let node = get_current_cpu_numa_node();

        // If not unknown, should be a reasonable NUMA node number (< 8 on most systems)
        if !node.is_unknown() {
            assert!(node.0 < 8, "NUMA node {} seems unreasonably high", node.0);
        }
    }

    #[test]
    fn test_get_device_numa_node_valid_gpu() {
        // Test GPU 0 detection
        let node = get_device_numa_node(0);

        // Should return either a valid node (0-7) or use heuristic (gpu_id % 2)
        // On dual-socket systems, GPU 0 typically on node 0 or 1
        println!("GPU 0 detected on NUMA node: {}", node.0);
    }

    #[test]
    fn test_numa_node_hash() {
        // Verify NumaNode can be used as a HashMap key
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
        // Verify NumaNode is Copy and Clone
        let node1 = NumaNode(5);
        let node2 = node1; // Copy
        let node3 = node1; // Clone

        assert_eq!(node1, node2);
        assert_eq!(node1, node3);
        assert_eq!(node2, node3);
    }
}
