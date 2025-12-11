// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message types for leader-worker cohort coordination.
//!
//! This module defines the request and response types used for communication
//! between the leader and workers during cohort formation, layout creation,
//! transfer execution, and metadata exchange.

use crate::v2::physical::layout::{LayoutConfig, LayoutDescriptor};
use crate::v2::physical::manager::{LayoutHandle, SerializedLayout};
use crate::v2::physical::transfer::TransferOptions;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Type of physical layout structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutType {
    /// All blocks, layers, and outer dimensions stored contiguously.
    FullyContiguous,

    /// Each layer stored in separate memory regions.
    LayerSeparate,
}

/// Type of memory storage for the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Device (GPU) memory - requires valid CUDA device.
    Device,

    /// Pinned (page-locked) host memory - faster for DMA transfers.
    Pinned,

    /// Regular system memory - pageable host memory.
    System,

    /// Disk-backed memory - for very large datasets.
    Disk,
}

/// Request from worker to leader to join the cohort.
///
/// This message is sent during the initial handshake when a worker
/// wants to join the leader's cohort.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCohortRequest {
    /// Optional rank for this worker (e.g., from torch.distributed or MPI)
    /// If provided, all workers must provide ranks and they must form a contiguous sequence [0, N)
    pub rank: Option<usize>,

    /// Total world size expected (number of workers including this one)
    pub world_size: usize,
    // Future fields could include:
    // pub vllm_config: Option<VllmWorkerConfig>,
    // pub capabilities: WorkerCapabilities,
}

/// Response from leader to worker for cohort join request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCohortResponse {
    /// Whether the worker was accepted into the cohort
    pub accepted: bool,

    /// Position assigned to this worker in the cohort (if accepted)
    pub position: Option<usize>,

    /// Human-readable reason for rejection (if not accepted)
    pub reason: Option<String>,
}

/// Request from leader to worker to create a physical layout.
///
/// This message is broadcast to all workers after the cohort is complete,
/// instructing them to allocate memory according to the provided configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateLayoutRequest {
    /// Configuration for the memory layout to create
    pub layout_config: LayoutConfig,

    /// Type of layout structure (contiguous vs layer-separate)
    pub layout_type: LayoutType,

    /// Type of memory storage (device, pinned, system, disk)
    pub memory_type: MemoryType,

    /// Name/identifier for this layout (for tracking multiple layouts)
    pub name: String,
}

/// Response from worker to leader after layout creation.
///
/// Contains the layout handle on success, or an error message on failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateLayoutResponse {
    /// Result of the layout creation: Ok with handle, or Err with error message
    pub result: Result<LayoutHandle, String>,
}

/// Request from leader to worker to fetch memory descriptors.
///
/// This is sent after all workers have completed memory creation
/// to collect the NIXL-compatible layout descriptors for remote access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDescriptorsRequest {
    // Currently empty - just a fetch operation
    // Future fields could include:
    // pub filter: Option<DescriptorFilter>,
}

/// Response from worker to leader with memory descriptors.
///
/// Contains all layout descriptors created by this worker,
/// including NIXL metadata needed for remote memory access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDescriptorsResponse {
    /// All layout descriptors from this worker
    pub descriptors: Vec<LayoutDescriptor>,
}

/// Request from worker to leader indicating arrival at a named barrier.
///
/// Workers send this message to signal they have reached a synchronization point.
/// The leader tracks barrier arrivals and unblocks when all workers have reached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierReachedRequest {
    /// Name of the barrier that was reached
    pub barrier_name: String,
}

/// Response from leader acknowledging barrier arrival.
///
/// Currently just confirms receipt. Future versions could include
/// information about how many workers have reached the barrier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierReachedResponse {
    /// Whether the barrier arrival was acknowledged
    pub acknowledged: bool,
}

/// Layout set identifiers for grouping layouts by memory type.
///
/// Used to organize layout collections when coordinating transfers
/// across multiple workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutSets {
    /// Device (GPU) layouts
    G1,
    /// Host (pinned or system) layouts
    G2,
    /// Disk (posix or gds) layouts
    G3,
}

/// Wire-compatible version of TransferOptions for network transmission.
///
/// This is a simplified version of TransferOptions that can be serialized.
/// The bounce_buffer field is omitted since it contains Arc<dyn> which
/// cannot be serialized and is not needed for local worker transfers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferOptionsWire {
    /// Range of layers to transfer (None = all layers)
    pub layer_range: Option<Range<usize>>,
    /// NIXL write notification value delivered after RDMA write completes
    pub nixl_write_notification: Option<u64>,
}

impl From<TransferOptionsWire> for TransferOptions {
    fn from(wire: TransferOptionsWire) -> Self {
        TransferOptions {
            layer_range: wire.layer_range,
            nixl_write_notification: wire.nixl_write_notification,
            bounce_buffer: None,
        }
    }
}

/// Request from leader to worker to execute a local transfer.
///
/// The leader sends this to all workers with their worker-specific handles,
/// instructing each to execute the same block transfer locally using its
/// TransportManager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteTransferRequest {
    /// Source layout handle (worker-specific)
    pub src_handle: LayoutHandle,
    /// Destination layout handle (worker-specific)
    pub dst_handle: LayoutHandle,
    /// Source block IDs to transfer
    pub src_blocks: Vec<usize>,
    /// Destination block IDs to transfer
    pub dst_blocks: Vec<usize>,
    /// Transfer options (layer range, notifications, etc.)
    pub options: TransferOptionsWire,
}

/// Response from worker after executing a local transfer.
///
/// Contains the result of the transfer operation: success or error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteTransferResponse {
    /// Result of the transfer: Ok(()) on success, Err(message) on failure
    pub result: Result<(), String>,
}

/// Request from leader to worker to export all layout metadata.
///
/// This is sent after workers have created their layouts to collect
/// the serialized layout data including NIXL metadata for remote access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportLayoutsRequest {
    // Currently empty - just a fetch operation
    // Future fields could include:
    // pub filter: Option<Vec<LayoutHandle>>,
}

/// Response from worker to leader with exported layout metadata.
///
/// Contains the serialized layout data including all layouts created by this worker
/// and the NIXL metadata needed for remote memory access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportLayoutsResponse {
    /// Serialized layout metadata including NIXL registration data
    pub metadata: SerializedLayout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cohort_request_serialization() {
        let request = CreateCohortRequest {
            rank: Some(0),
            world_size: 4,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: CreateCohortRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.rank, Some(0));
        assert_eq!(deserialized.world_size, 4);
    }

    #[test]
    fn test_create_cohort_response_serialization() {
        let response = CreateCohortResponse {
            accepted: true,
            position: Some(2),
            reason: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: CreateCohortResponse = serde_json::from_str(&json).unwrap();

        assert!(deserialized.accepted);
        assert_eq!(deserialized.position, Some(2));
    }

    #[test]
    fn test_create_layout_request_serialization() {
        let config = LayoutConfig::builder()
            .num_blocks(10)
            .num_layers(4)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let request = CreateLayoutRequest {
            layout_config: config.clone(),
            layout_type: LayoutType::FullyContiguous,
            memory_type: MemoryType::System,
            name: "test_layout".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: CreateLayoutRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.layout_config, config);
        assert_eq!(deserialized.layout_type, LayoutType::FullyContiguous);
        assert_eq!(deserialized.memory_type, MemoryType::System);
        assert_eq!(deserialized.name, "test_layout");
    }

    #[test]
    fn test_memory_descriptors_response_empty() {
        let response = MemoryDescriptorsResponse {
            descriptors: vec![],
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: MemoryDescriptorsResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.descriptors.len(), 0);
    }
}
