// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for layout serialization.
//!
//! These tests verify the complete serialization and deserialization flow,
//! ensuring that layouts can be transmitted to remote nodes and reconstructed
//! with all necessary metadata intact.

use crate::block_manager::v2::memory::{
    MemoryRegion, NixlDescriptor, OwnedMemoryRegion, StorageKind,
};
use crate::block_manager::v2::physical::layout::physical::PhysicalLayout;
use crate::block_manager::v2::physical::layout::{BlockDimension, LayoutConfig, LayoutDescriptor};
use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;
use std::any::Any;
use std::sync::Arc;

// Simple mock implementation for testing
#[derive(Debug)]
pub struct MockMemory {
    addr: usize,
    size: usize,
}

impl MockMemory {
    pub fn new(addr: usize, size: usize) -> Arc<Self> {
        Arc::new(Self { addr, size })
    }
}

impl MemoryRegion for MockMemory {
    fn addr(&self) -> usize {
        self.addr
    }
    fn size(&self) -> usize {
        self.size
    }
    fn storage_kind(&self) -> StorageKind {
        StorageKind::System
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Mock memory region for testing serialization
#[derive(Debug)]
struct TestMemoryRegion {
    addr: usize,
    size: usize,
    kind: StorageKind,
    descriptor: NixlDescriptor,
}

impl TestMemoryRegion {
    fn new(addr: usize, size: usize, kind: StorageKind) -> Arc<Self> {
        Arc::new(Self {
            addr,
            size,
            kind,
            descriptor: NixlDescriptor {
                addr: addr as u64,
                size,
                mem_type: nixl_sys::MemType::Dram,
                device_id: 0,
            },
        })
    }
}

impl MemoryRegion for TestMemoryRegion {
    fn addr(&self) -> usize {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn storage_kind(&self) -> StorageKind {
        self.kind
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        Some(self.descriptor.clone())
    }
}

fn make_test_config() -> LayoutConfig {
    LayoutConfig::builder()
        .num_blocks(10)
        .num_layers(4)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(128)
        .dtype_width_bytes(2)
        .build()
        .unwrap()
}

#[test]
fn test_fully_contiguous_layout_serialization_roundtrip() {
    let agent = NixlAgent::require_backends("test-fc-serialize", &[])
        .expect("failed to create wrapped agent");
    let config = make_test_config();

    // Calculate required size
    let required_size = config.num_blocks
        * config.num_layers
        * config.outer_dim
        * config.page_size
        * config.inner_dim
        * config.dtype_width_bytes;

    // Create test memory region
    let memory = TestMemoryRegion::new(0x10000, required_size, StorageKind::System);
    let regions = vec![memory as OwnedMemoryRegion];

    // Build physical layout
    let original_layout = PhysicalLayout::builder(agent)
        .with_config(config.clone())
        .fully_contiguous()
        .with_registered_regions(regions)
        .expect("failed to provide regions")
        .build()
        .expect("failed to build layout");

    // Serialize to LayoutDescriptor
    let serialized = original_layout
        .to_descriptor()
        .expect("failed to serialize layout");

    // Verify serialized data
    assert_eq!(serialized.version, LayoutDescriptor::CURRENT_VERSION);
    assert_eq!(serialized.layout_config, config);
    assert_eq!(serialized.location, StorageKind::System);
    assert_eq!(serialized.memory_descriptors.len(), 1);
    assert_eq!(serialized.memory_descriptors[0].addr, 0x10000);
    assert_eq!(serialized.memory_descriptors[0].size, required_size);

    // Serialize to JSON
    let json = serialized.to_json().expect("failed to serialize to JSON");
    assert!(json.contains("\"version\":1"));
    assert!(json.contains("\"num_blocks\":10"));

    // Deserialize from JSON
    let deserialized = LayoutDescriptor::from_json(&json).expect("failed to deserialize from JSON");

    // Verify deserialized matches original
    assert_eq!(deserialized.version, serialized.version);
    assert_eq!(deserialized.layout_config, serialized.layout_config);
    assert_eq!(deserialized.location, serialized.location);
    assert_eq!(
        deserialized.memory_descriptors.len(),
        serialized.memory_descriptors.len()
    );

    // Reconstruct layout from serialized data
    let reconstructed =
        PhysicalLayout::from_descriptor(deserialized).expect("failed to reconstruct layout");

    // Verify reconstructed layout has same configuration
    assert_eq!(reconstructed.layout().config(), &config);
    assert_eq!(reconstructed.location(), StorageKind::System);
    assert_eq!(reconstructed.layout().num_blocks(), 10);
    assert_eq!(reconstructed.layout().num_layers(), 4);
    assert!(reconstructed.layout().is_fully_contiguous());
}

#[test]
fn test_layer_separate_layout_serialization_roundtrip() {
    let agent = NixlAgent::require_backends("test-ls-serialize", &[])
        .expect("failed to create wrapped agent");
    let config = make_test_config();

    // Calculate per-layer size
    let per_layer_size = config.num_blocks
        * config.outer_dim
        * config.page_size
        * config.inner_dim
        * config.dtype_width_bytes;

    // Create memory regions (one per layer)
    let regions: Vec<OwnedMemoryRegion> = (0..config.num_layers)
        .map(|i| {
            TestMemoryRegion::new(
                0x10000 + i * per_layer_size,
                per_layer_size,
                StorageKind::System,
            ) as OwnedMemoryRegion
        })
        .collect();

    // Build physical layout
    let original_layout = PhysicalLayout::builder(agent)
        .with_config(config.clone())
        .layer_separate(BlockDimension::BlockIsFirstDim)
        .with_registered_regions(regions)
        .expect("failed to provide regions")
        .build()
        .expect("failed to build layout");

    // Serialize to LayoutDescriptor
    let serialized = original_layout
        .to_descriptor()
        .expect("failed to serialize layout");

    // Verify serialized data
    assert_eq!(serialized.version, LayoutDescriptor::CURRENT_VERSION);
    assert_eq!(serialized.layout_config, config);
    assert_eq!(serialized.memory_descriptors.len(), 4); // One per layer

    // Verify memory descriptors
    for (i, desc) in serialized.memory_descriptors.iter().enumerate() {
        assert_eq!(desc.addr, 0x10000 + i * per_layer_size);
        assert_eq!(desc.size, per_layer_size);
    }

    // Serialize to JSON bytes
    let json_bytes = serialized
        .to_json_bytes()
        .expect("failed to serialize to JSON bytes");

    // Deserialize from JSON bytes
    let deserialized = LayoutDescriptor::from_json_bytes(&json_bytes)
        .expect("failed to deserialize from JSON bytes");

    // Verify deserialized matches original
    assert_eq!(deserialized.version, serialized.version);
    assert_eq!(deserialized.layout_config, serialized.layout_config);
    assert_eq!(
        deserialized.memory_descriptors.len(),
        serialized.memory_descriptors.len()
    );

    // Reconstruct layout from serialized data
    let reconstructed =
        PhysicalLayout::from_descriptor(deserialized).expect("failed to reconstruct layout");

    // Verify reconstructed layout has same configuration
    assert_eq!(reconstructed.layout().config(), &config);
    assert_eq!(reconstructed.location(), StorageKind::System);
    assert_eq!(reconstructed.layout().num_blocks(), 10);
    assert_eq!(reconstructed.layout().num_layers(), 4);
    assert!(!reconstructed.layout().is_fully_contiguous());
}

#[test]
fn test_memory_region_calculation_after_deserialization() {
    let agent = NixlAgent::require_backends("test-memory-calc", &[])
        .expect("failed to create wrapped agent");
    let config = LayoutConfig::builder()
        .num_blocks(2)
        .num_layers(2)
        .outer_dim(2)
        .page_size(4)
        .inner_dim(8)
        .dtype_width_bytes(2)
        .build()
        .unwrap();

    let required_size = config.num_blocks
        * config.num_layers
        * config.outer_dim
        * config.page_size
        * config.inner_dim
        * config.dtype_width_bytes;

    let memory = TestMemoryRegion::new(0x1000, required_size, StorageKind::System);
    let regions = vec![memory as OwnedMemoryRegion];

    let original_layout = PhysicalLayout::builder(agent)
        .with_config(config.clone())
        .fully_contiguous()
        .with_registered_regions(regions)
        .expect("failed to provide regions")
        .build()
        .expect("failed to build layout");

    // Serialize and deserialize
    let serialized = original_layout
        .to_descriptor()
        .expect("failed to serialize");
    let reconstructed = PhysicalLayout::from_descriptor(serialized).expect("failed to reconstruct");

    // Verify memory region calculations
    let region = reconstructed
        .memory_region(0, 0, 0)
        .expect("failed to get memory region");
    assert_eq!(region.addr, 0x1000);

    let region_size = config.page_size * config.inner_dim * config.dtype_width_bytes;
    assert_eq!(region.size, region_size);

    // Test different block/layer/outer indices
    let region = reconstructed
        .memory_region(1, 1, 1)
        .expect("failed to get memory region");
    // Address should be: base + block_stride + layer_stride + outer_stride
    let layer_stride = config.outer_dim * region_size;
    let block_stride = config.num_layers * layer_stride;
    let expected_addr = 0x1000 + block_stride + layer_stride + region_size;
    assert_eq!(region.addr, expected_addr);
}

#[test]
fn test_version_check_on_deserialization() {
    let config = make_test_config();

    // Calculate required size for fully contiguous layout
    let required_size = config.num_blocks
        * config.num_layers
        * config.outer_dim
        * config.page_size
        * config.inner_dim
        * config.dtype_width_bytes;

    let mut serialized = LayoutDescriptor {
        version: 999, // Future version
        layout_config: config.clone(),
        location: StorageKind::System,
        nixl_metadata: crate::block_manager::v2::physical::layout::physical::NixlMetadata::new(
            "test".to_string(),
            nixl_sys::MemType::Dram,
            0,
        ),
        memory_descriptors: vec![],
        layout_type_details:
            crate::block_manager::v2::physical::layout::LayoutTypeDetails::FullyContiguous(
                crate::block_manager::v2::physical::layout::FullyContiguousDetails {
                    block_format:
                        crate::block_manager::v2::physical::layout::BlockFormat::Operational,
                },
            ),
    };

    // Should fail with unsupported version
    let result = PhysicalLayout::from_descriptor(serialized.clone());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Unsupported serialization version")
    );

    // Should succeed with supported version
    serialized.version = LayoutDescriptor::CURRENT_VERSION;
    serialized.memory_descriptors = vec![crate::block_manager::v2::memory::MemoryDescriptor::new(
        0x1000,
        required_size,
    )];
    let result = PhysicalLayout::from_descriptor(serialized);
    if let Err(ref e) = result {
        eprintln!("Error during deserialization: {}", e);
    }
    assert!(
        result.is_ok(),
        "Expected successful deserialization, got error: {:?}",
        result.err()
    );
}
