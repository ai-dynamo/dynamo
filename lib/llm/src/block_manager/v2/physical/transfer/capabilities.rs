// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer capability flags for controlling direct path enablement.
//!
//! By default, the transfer system uses a conservative staging policy where:
//! - Device can only transfer to/from Host
//! - Disk can only transfer to/from Host
//! - Host can transfer to Device, Disk, or Remote
//! - Device ↔ Device is allowed (native CUDA)
//!
//! These capability flags enable optional direct paths that bypass host staging.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::block_manager::v2::physical::{
    layout::LayoutConfig,
    transfer::{
        PhysicalLayout, TransferOptions, TransportManager, executor::execute_transfer,
        nixl_agent::NixlAgent,
    },
};

/// Transfer capability flags controlling which direct paths are enabled.
///
/// # Default Policy (Conservative)
///
/// With all flags disabled (default), the system uses host staging:
/// - **Device → Remote**: Device → Host → Remote (2 hops)
/// - **Disk → Remote**: Disk → Host → Remote (2 hops)
/// - **Device ↔ Disk**: Device → Host → Disk (2 hops)
///
/// # Optional Direct Paths
///
/// - `allow_gds`: Enables GPU Direct Storage (Disk ↔ Device without host)
/// - `allow_gpu_rdma`: Enables GPU RDMA (Device → Remote without host)
///
/// # Example
///
/// ```
/// # use dynamo_kvbm::v2::physical::transfer::TransferCapabilities;
/// // Default conservative policy
/// let caps = TransferCapabilities::default();
/// assert!(!caps.allow_gds);
/// assert!(!caps.allow_gpu_rdma);
///
/// // Enable GDS for high-performance disk I/O
/// let caps = TransferCapabilities::default().with_gds(true);
/// ```
static GDS_SUPPORTED: OnceLock<bool> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TransferCapabilities {
    /// Enable GPU Direct Storage (Disk ↔ Device without host staging).
    ///
    /// When enabled:
    /// - Disk → Device: Direct transfer (requires GDS support)
    /// - Device → Disk: Direct transfer (requires GDS support)
    ///
    /// When disabled (default):
    /// - Disk → Device: Disk → Host → Device (2 hops)
    /// - Device → Disk: Device → Host → Disk (2 hops)
    pub allow_gds: bool,

    /// Enable GPU RDMA (Device → Remote without host staging).
    ///
    /// When enabled:
    /// - Device → Remote: Direct NIXL transfer
    ///
    /// When disabled (default):
    /// - Device → Remote: Device → Host → Remote (2 hops)
    ///
    /// Note: This only affects Device → Remote. Host → Remote is always direct.
    pub allow_gpu_rdma: bool,
}

impl TransferCapabilities {
    /// Create capabilities with default conservative policy (all direct paths disabled).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create capabilities with all direct paths enabled (high performance mode).
    pub fn all_enabled() -> Self {
        Self {
            allow_gds: true,
            allow_gpu_rdma: true,
        }
    }

    /// Set the GDS (GPU Direct Storage) capability.
    pub fn with_gds(mut self, enabled: bool) -> Self {
        self.allow_gds = enabled;
        self
    }

    fn test_gds_transfer(&self) -> anyhow::Result<()> {
        let agent = NixlAgent::require_backends("agent", &["GDS_MT"])?;

        // Try a little test transfer and see if it works.
        let config = LayoutConfig::builder()
            .num_blocks(1)
            .num_layers(1)
            .outer_dim(1)
            .page_size(1)
            .inner_dim(4096)
            .build()?;

        let src = PhysicalLayout::builder(agent.clone())
            .with_config(config.clone())
            .fully_contiguous()
            .allocate_device(0)
            .build()?;
        let dst = PhysicalLayout::builder(agent.clone())
            .with_config(config)
            .fully_contiguous()
            .allocate_disk(None)
            .build()?;

        let src_blocks = vec![0];
        let dst_blocks = vec![0];

        let ctx = TransportManager::builder()
            .worker_id(0)
            .nixl_agent(agent)
            .cuda_device_id(0)
            .build()?;

        execute_transfer(
            &src,
            &dst,
            &src_blocks,
            &dst_blocks,
            TransferOptions::default(),
            ctx.context(),
        )?;

        Ok(())
    }

    pub fn with_gds_if_supported(mut self) -> Self {
        self.allow_gds = *GDS_SUPPORTED.get_or_init(|| self.test_gds_transfer().is_ok());

        self
    }

    /// Set the GPU RDMA capability.
    pub fn with_gpu_rdma(mut self, enabled: bool) -> Self {
        self.allow_gpu_rdma = enabled;
        self
    }

    /// Check if a direct path from Device to Disk is allowed.
    pub fn allows_device_disk_direct(&self) -> bool {
        self.allow_gds
    }

    /// Check if a direct path from Device to Remote is allowed.
    pub fn allows_device_remote_direct(&self) -> bool {
        self.allow_gpu_rdma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_capabilities() {
        let caps = TransferCapabilities::default();
        assert!(!caps.allow_gds);
        assert!(!caps.allow_gpu_rdma);
        assert!(!caps.allows_device_disk_direct());
        assert!(!caps.allows_device_remote_direct());
    }

    #[test]
    fn test_all_enabled() {
        let caps = TransferCapabilities::all_enabled();
        assert!(caps.allow_gds);
        assert!(caps.allow_gpu_rdma);
        assert!(caps.allows_device_disk_direct());
        assert!(caps.allows_device_remote_direct());
    }

    #[test]
    fn test_builder_pattern() {
        let caps = TransferCapabilities::new()
            .with_gds(true)
            .with_gpu_rdma(false);

        assert!(caps.allow_gds);
        assert!(!caps.allow_gpu_rdma);
    }

    #[test]
    fn test_selective_enablement() {
        // Enable only GDS
        let caps = TransferCapabilities::new().with_gds(true);
        assert!(caps.allows_device_disk_direct());
        assert!(!caps.allows_device_remote_direct());

        // Enable only GPU RDMA
        let caps = TransferCapabilities::new().with_gpu_rdma(true);
        assert!(!caps.allows_device_disk_direct());
        assert!(caps.allows_device_remote_direct());
    }
}
