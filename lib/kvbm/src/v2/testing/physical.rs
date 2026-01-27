// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical layout and transfer testing utilities.
//!
//! This module provides reusable test infrastructure for:
//! - Creating physical layouts with various storage types
//! - Creating TransferManagers with UCX backend for RDMA tests
//! - Filling blocks with test patterns and computing checksums
//! - Verifying data integrity after transfers

use anyhow::Result;
use std::collections::HashMap;

use crate::{
    BlockId,
    v2::physical::{
        layout::{BlockDimension, LayoutConfig, PhysicalLayout},
        manager::{LayoutHandle, TransferManager},
        transfer::{
            BlockChecksum, FillPattern, NixlAgent, StorageKind, TransferCapabilities,
            compute_block_checksums, compute_layer_checksums, fill_blocks, fill_layers,
        },
    },
};

// =============================================================================
// Flexible Backend Agent Builder
// =============================================================================

/// A NixlAgent wrapper that tracks which backends were successfully initialized.
///
/// This wrapper allows tests to check backend availability and conditionally
/// skip tests that require unavailable backends.
///
/// # Example
///
/// ```ignore
/// // Flexible - won't fail if UCX unavailable
/// let agent = TestAgentBuilder::new("test")
///     .try_backend("UCX")
///     .try_backend("POSIX")  // Always available
///     .build()?;
///
/// // Check what's available
/// if !agent.has_backend("UCX") {
///     eprintln!("Skipping RDMA test - UCX unavailable");
///     return Ok(());
/// }
/// ```
pub struct TestAgent {
    agent: NixlAgent,
    available_backends: Vec<String>,
}

impl TestAgent {
    /// Returns true if the specified backend was successfully initialized.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.available_backends
            .iter()
            .any(|b| b.eq_ignore_ascii_case(backend))
    }

    /// Returns the list of successfully initialized backends.
    pub fn available_backends(&self) -> &[String] {
        &self.available_backends
    }

    /// Consumes self and returns the underlying NixlAgent.
    pub fn into_nixl_agent(self) -> NixlAgent {
        self.agent
    }

    /// Returns a reference to the underlying NixlAgent.
    pub fn nixl_agent(&self) -> &NixlAgent {
        &self.agent
    }
}

impl std::ops::Deref for TestAgent {
    type Target = NixlAgent;

    fn deref(&self) -> &Self::Target {
        &self.agent
    }
}

/// Builder for TestAgent with flexible backend handling.
///
/// Unlike `NixlAgent::with_backends()` which fails if ANY backend is unavailable,
/// `TestAgentBuilder` allows graceful degradation by distinguishing between
/// required and optional backends.
///
/// # Example
///
/// ```ignore
/// // RDMA tests - UCX is required
/// let agent = TestAgentBuilder::new("rdma-test")
///     .require_backend("UCX")  // Fails if unavailable
///     .try_backend("POSIX")    // Optional
///     .build()?;
///
/// // Disk tests - POSIX only, no GDS requirement
/// let agent = TestAgentBuilder::new("disk-test")
///     .try_backend("POSIX")
///     .build()?;
/// ```
#[derive(Default)]
pub struct TestAgentBuilder {
    name: Option<String>,
    try_backends: Vec<String>,
    required_backends: Vec<String>,
}

impl TestAgentBuilder {
    /// Creates a new builder with the given agent name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            try_backends: Vec::new(),
            required_backends: Vec::new(),
        }
    }

    /// Attempts to add a backend. If unavailable, build() will still succeed
    /// but `has_backend(name)` will return false.
    pub fn try_backend(mut self, backend: impl Into<String>) -> Self {
        self.try_backends.push(backend.into());
        self
    }

    /// Requires a backend. If unavailable, build() will fail.
    ///
    /// Use this for tests that cannot function without specific backends,
    /// like RDMA tests requiring UCX.
    pub fn require_backend(mut self, backend: impl Into<String>) -> Self {
        self.required_backends.push(backend.into());
        self
    }

    /// Builds the TestAgent.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Agent creation fails
    /// - Any required backend fails to initialize
    pub fn build(self) -> Result<TestAgent> {
        let name = self
            .name
            .ok_or_else(|| anyhow::anyhow!("Agent name is required"))?;

        let mut agent = NixlAgent::new(&name)?;
        let mut available_backends = Vec::new();

        // Initialize required backends first - fail on error
        for backend in &self.required_backends {
            let backend_upper = backend.to_uppercase();
            agent.add_backend(&backend_upper).map_err(|e| {
                anyhow::anyhow!(
                    "Required backend {} unavailable: {}. \
                     Use try_backend() if this backend is optional.",
                    backend_upper,
                    e
                )
            })?;
            available_backends.push(backend_upper);
        }

        // Initialize optional backends - log warning but continue
        for backend in &self.try_backends {
            let backend_upper = backend.to_uppercase();
            // Skip if already added as required
            if available_backends
                .iter()
                .any(|b| b.eq_ignore_ascii_case(&backend_upper))
            {
                continue;
            }
            match agent.add_backend(&backend_upper) {
                Ok(_) => {
                    tracing::debug!("Initialized optional backend: {}", backend_upper);
                    available_backends.push(backend_upper);
                }
                Err(e) => {
                    tracing::debug!(
                        "Optional backend {} unavailable: {} - continuing without it",
                        backend_upper,
                        e
                    );
                }
            }
        }

        Ok(TestAgent {
            agent,
            available_backends,
        })
    }
}

// =============================================================================
// Transfer Checksums Helper
// =============================================================================

/// Captures checksums for source blocks to enable verification after transfer.
///
/// This provides a cleaner pattern for the fill→transfer→verify workflow.
///
/// # Example
///
/// ```ignore
/// // Capture source checksums
/// let src = TransferChecksums::fill_and_capture(&src_layout, &src_ids, FillPattern::Sequential)?;
///
/// // ... execute transfer ...
///
/// // Verify destination matches
/// src.verify_against(&dst_layout, &dst_ids)?;
/// ```
pub struct TransferChecksums {
    checksums: HashMap<BlockId, BlockChecksum>,
    block_ids: Vec<BlockId>,
}

impl TransferChecksums {
    /// Fill blocks with a pattern and capture their checksums.
    pub fn fill_and_capture(
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
        pattern: FillPattern,
    ) -> Result<Self> {
        let checksums = fill_and_checksum(layout, block_ids, pattern)?;
        Ok(Self {
            checksums,
            block_ids: block_ids.to_vec(),
        })
    }

    /// Capture checksums without filling (for already-filled blocks).
    pub fn capture(layout: &PhysicalLayout, block_ids: &[BlockId]) -> Result<Self> {
        let checksums = crate::v2::physical::transfer::compute_block_checksums(layout, block_ids)?;
        Ok(Self {
            checksums,
            block_ids: block_ids.to_vec(),
        })
    }

    /// Returns the captured checksums.
    pub fn checksums(&self) -> &HashMap<BlockId, BlockChecksum> {
        &self.checksums
    }

    /// Returns the block IDs for which checksums were captured.
    pub fn block_ids(&self) -> &[BlockId] {
        &self.block_ids
    }

    /// Verify that destination blocks match the captured source checksums.
    ///
    /// The destination block IDs must be the same length as source block IDs.
    /// Comparison is done positionally: src_blocks[i] is compared with dst_ids[i].
    pub fn verify_against(&self, dst_layout: &PhysicalLayout, dst_ids: &[BlockId]) -> Result<()> {
        verify_checksums_by_position(&self.checksums, &self.block_ids, dst_layout, dst_ids)
    }

    /// Verify that destination blocks match using specific layers only.
    pub fn verify_layers_against(
        &self,
        dst_layout: &PhysicalLayout,
        dst_ids: &[BlockId],
        layer_range: std::ops::Range<usize>,
    ) -> Result<()> {
        verify_layer_checksums_by_position(
            &self.checksums,
            &self.block_ids,
            dst_layout,
            dst_ids,
            layer_range,
        )
    }
}

// =============================================================================
// Layout Types
// =============================================================================

/// Layout kind for parameterized testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    /// Fully contiguous layout
    FC,
    /// Layer-wise (layer-separate) layout
    LW,
}

/// Storage and layout specification for creating test layouts.
#[derive(Debug, Clone, Copy)]
pub struct LayoutSpec {
    pub kind: LayoutKind,
    pub storage: StorageKind,
}

impl LayoutSpec {
    pub fn new(kind: LayoutKind, storage: StorageKind) -> Self {
        Self { kind, storage }
    }
}

/// Standard layout configuration for tests.
///
/// Uses standard dimensions suitable for most tests:
/// - 2 layers
/// - outer_dim=2 (K&V separate)
/// - page_size=16
/// - inner_dim=128
/// - dtype_width=2 (bf16)
pub fn standard_config(num_blocks: usize) -> LayoutConfig {
    LayoutConfig::builder()
        .num_blocks(num_blocks)
        .num_layers(2)
        .outer_dim(2)
        .page_size(16)
        .inner_dim(128)
        .dtype_width_bytes(2)
        .build()
        .expect("standard config should build")
}

/// Create a custom layout configuration for RDMA tests.
///
/// # Arguments
/// * `num_blocks` - Number of blocks in the layout
/// * `num_layers` - Number of transformer layers
/// * `outer_dim` - Outer dimension (2 for K&V separate)
/// * `page_size` - Tokens per block/page
/// * `inner_dim` - Hidden dimension
/// * `dtype_width` - Data type width in bytes
pub fn custom_config(
    num_blocks: usize,
    num_layers: usize,
    outer_dim: usize,
    page_size: usize,
    inner_dim: usize,
    dtype_width: usize,
) -> LayoutConfig {
    LayoutConfig::builder()
        .num_blocks(num_blocks)
        .num_layers(num_layers)
        .outer_dim(outer_dim)
        .page_size(page_size)
        .inner_dim(inner_dim)
        .dtype_width_bytes(dtype_width)
        .build()
        .expect("custom config should build")
}

/// Create a test NIXL agent with no backends.
///
/// Use this for tests that don't require specific NIXL backends.
pub fn create_test_agent(name: &str) -> NixlAgent {
    NixlAgent::new(name).expect("Failed to create agent")
}

/// Create a test NIXL agent with specific backends (strict - all must succeed).
///
/// # Arguments
/// * `name` - Agent name (must be unique for RDMA addressing)
/// * `backends` - List of backends to enable (e.g., &["UCX"])
pub fn create_test_agent_with_backends(name: &str, backends: &[&str]) -> Result<NixlAgent> {
    NixlAgent::with_backends(name, backends)
}

/// Create a fully contiguous physical layout with the specified storage type.
pub fn create_fc_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    create_fc_layout_with_config(agent, storage_kind, standard_config(num_blocks))
}

/// Create a fully contiguous physical layout with custom config.
pub fn create_fc_layout_with_config(
    agent: NixlAgent,
    storage_kind: StorageKind,
    config: LayoutConfig,
) -> PhysicalLayout {
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous();

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

/// Create a layer-separate physical layout with the specified storage type.
pub fn create_lw_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    create_lw_layout_with_config(agent, storage_kind, standard_config(num_blocks))
}

/// Create a layer-separate physical layout with custom config.
pub fn create_lw_layout_with_config(
    agent: NixlAgent,
    storage_kind: StorageKind,
    config: LayoutConfig,
) -> PhysicalLayout {
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .layer_separate(BlockDimension::BlockIsFirstDim);

    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(None).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

/// Create a physical layout based on the specification.
pub fn create_layout(agent: NixlAgent, spec: LayoutSpec, num_blocks: usize) -> PhysicalLayout {
    match spec.kind {
        LayoutKind::FC => create_fc_layout(agent, spec.storage, num_blocks),
        LayoutKind::LW => create_lw_layout(agent, spec.storage, num_blocks),
    }
}

/// Create a physical layout based on specification with custom config.
pub fn create_layout_with_config(
    agent: NixlAgent,
    spec: LayoutSpec,
    config: LayoutConfig,
) -> PhysicalLayout {
    match spec.kind {
        LayoutKind::FC => create_fc_layout_with_config(agent, spec.storage, config),
        LayoutKind::LW => create_lw_layout_with_config(agent, spec.storage, config),
    }
}

/// Create a TransferManager for testing.
///
/// # Arguments
/// * `agent` - NIXL agent (should have backends configured)
/// * `capabilities` - Optional transfer capabilities
pub fn create_transfer_manager(
    agent: NixlAgent,
    capabilities: Option<TransferCapabilities>,
) -> Result<TransferManager> {
    TransferManager::builder()
        .capabilities(capabilities.unwrap_or_default())
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()
}

/// Create a TransferManager with UCX backend for RDMA tests.
///
/// # Arguments
/// * `agent_name` - Unique agent name for RDMA addressing
///
/// Note: The worker_id is derived from the event system. For explicit worker_id
/// control, use the TransferManager builder directly with a custom event system.
pub fn create_rdma_transfer_manager(agent_name: &str) -> Result<TransferManager> {
    let agent = create_test_agent_with_backends(agent_name, &["UCX"])?;
    TransferManager::builder()
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()
}

/// Fill blocks and compute checksums.
///
/// This can only be called on System or Pinned layouts.
pub fn fill_and_checksum(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    pattern: FillPattern,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    fill_blocks(layout, block_ids, pattern)?;
    compute_block_checksums(layout, block_ids)
}

/// Fill specific layers and compute checksums.
///
/// This can only be called on System or Pinned layouts.
pub fn fill_layers_and_checksum(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    layer_range: std::ops::Range<usize>,
    pattern: FillPattern,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    fill_layers(layout, block_ids, layer_range.clone(), pattern)?;
    compute_layer_checksums(layout, block_ids, layer_range)
}

/// Verify that destination block checksums match the expected source checksums.
///
/// This function compares checksums in order, assuming the source and destination
/// block arrays have a 1:1 correspondence (src[i] was transferred to dst[i]).
pub fn verify_checksums_by_position(
    src_checksums: &HashMap<BlockId, BlockChecksum>,
    src_block_ids: &[BlockId],
    dst_layout: &PhysicalLayout,
    dst_block_ids: &[BlockId],
) -> Result<()> {
    assert_eq!(
        src_block_ids.len(),
        dst_block_ids.len(),
        "Source and destination block arrays must have same length"
    );

    let dst_checksums = compute_block_checksums(dst_layout, dst_block_ids)?;

    for (src_id, dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums
            .get(src_id)
            .unwrap_or_else(|| panic!("Missing source checksum for block {}", src_id));
        let dst_checksum = dst_checksums
            .get(dst_id)
            .unwrap_or_else(|| panic!("Missing destination checksum for block {}", dst_id));

        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch: src[{}] != dst[{}]: {} != {}",
            src_id, dst_id, src_checksum, dst_checksum
        );
    }

    Ok(())
}

/// Verify checksums for specific layers.
pub fn verify_layer_checksums_by_position(
    src_checksums: &HashMap<BlockId, BlockChecksum>,
    src_block_ids: &[BlockId],
    dst_layout: &PhysicalLayout,
    dst_block_ids: &[BlockId],
    layer_range: std::ops::Range<usize>,
) -> Result<()> {
    assert_eq!(
        src_block_ids.len(),
        dst_block_ids.len(),
        "Source and destination block arrays must have same length"
    );

    let dst_checksums = compute_layer_checksums(dst_layout, dst_block_ids, layer_range)?;

    for (src_id, dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums
            .get(src_id)
            .unwrap_or_else(|| panic!("Missing source checksum for block {}", src_id));
        let dst_checksum = dst_checksums
            .get(dst_id)
            .unwrap_or_else(|| panic!("Missing destination checksum for block {}", dst_id));

        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch: src[{}] != dst[{}]: {} != {}",
            src_id, dst_id, src_checksum, dst_checksum
        );
    }

    Ok(())
}

/// Fill guard blocks and return their checksums for later verification.
///
/// Guard blocks are blocks adjacent to transfer destinations that should
/// remain unchanged during transfers.
pub fn create_guard_blocks(
    layout: &PhysicalLayout,
    guard_block_ids: &[BlockId],
    pattern: FillPattern,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    fill_blocks(layout, guard_block_ids, pattern)?;
    compute_block_checksums(layout, guard_block_ids)
}

/// Verify that guard blocks remain unchanged after transfers.
pub fn verify_guard_blocks_unchanged(
    layout: &PhysicalLayout,
    guard_block_ids: &[BlockId],
    expected_checksums: &HashMap<BlockId, BlockChecksum>,
) -> Result<()> {
    let current_checksums = compute_block_checksums(layout, guard_block_ids)?;

    for &block_id in guard_block_ids {
        let expected = expected_checksums
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing expected checksum for guard block {}", block_id));
        let current = current_checksums
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing current checksum for guard block {}", block_id));

        if expected != current {
            anyhow::bail!(
                "Guard block {} was modified during transfer! Expected: {}, Got: {}",
                block_id,
                expected,
                current
            );
        }
    }

    Ok(())
}

// =============================================================================
// TransferManager-based helpers (for registered layouts)
// =============================================================================

/// Fill blocks in a registered layout via TransferManager.
///
/// Accesses the internal registry directly (only available in-crate).
/// This can only be called on System or Pinned layouts.
pub fn fill_manager_blocks(
    manager: &TransferManager,
    handle: LayoutHandle,
    block_ids: &[BlockId],
    pattern: FillPattern,
) -> Result<()> {
    let registry = manager.registry().read().unwrap();
    let layout = registry
        .get_layout(handle)
        .ok_or_else(|| anyhow::anyhow!("Layout not found: {:?}", handle))?;
    fill_blocks(layout, block_ids, pattern)
}

/// Compute checksums for blocks in a registered layout.
///
/// Accesses the internal registry directly (only available in-crate).
pub fn compute_manager_checksums(
    manager: &TransferManager,
    handle: LayoutHandle,
    block_ids: &[BlockId],
) -> Result<HashMap<BlockId, BlockChecksum>> {
    let registry = manager.registry().read().unwrap();
    let layout = registry
        .get_layout(handle)
        .ok_or_else(|| anyhow::anyhow!("Layout not found: {:?}", handle))?;
    compute_block_checksums(layout, block_ids)
}

/// Fill blocks and compute checksums via TransferManager.
///
/// Accesses the internal registry directly (only available in-crate).
/// This can only be called on System or Pinned layouts.
pub fn fill_and_checksum_manager(
    manager: &TransferManager,
    handle: LayoutHandle,
    block_ids: &[BlockId],
    pattern: FillPattern,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    let registry = manager.registry().read().unwrap();
    let layout = registry
        .get_layout(handle)
        .ok_or_else(|| anyhow::anyhow!("Layout not found: {:?}", handle))?;
    fill_blocks(layout, block_ids, pattern)?;
    compute_block_checksums(layout, block_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_fc_layout_system() {
        let agent = create_test_agent("test_fc_system");
        let layout = create_fc_layout(agent, StorageKind::System, 4);
        assert!(layout.layout().as_ref().is_fully_contiguous());
    }

    #[test]
    fn test_create_lw_layout_system() {
        let agent = create_test_agent("test_lw_system");
        let layout = create_lw_layout(agent, StorageKind::System, 4);
        assert!(!layout.layout().as_ref().is_fully_contiguous());
    }

    #[test]
    fn test_fill_and_checksum() {
        let agent = create_test_agent("test_fill_checksum");
        let layout = create_fc_layout(agent, StorageKind::System, 4);

        let block_ids = vec![0, 1, 2];
        let checksums = fill_and_checksum(&layout, &block_ids, FillPattern::Sequential).unwrap();

        assert_eq!(checksums.len(), 3);
        // Each block should have a unique checksum with sequential pattern
        let values: Vec<_> = checksums.values().collect();
        assert!(values[0] != values[1] || values[1] != values[2]);
    }

    #[test]
    fn test_custom_config() {
        let config = custom_config(32, 3, 2, 4, 64, 2);
        assert_eq!(config.num_blocks, 32);
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.outer_dim, 2);
        assert_eq!(config.page_size, 4);
        assert_eq!(config.inner_dim, 64);
        assert_eq!(config.dtype_width_bytes, 2);
    }
}
