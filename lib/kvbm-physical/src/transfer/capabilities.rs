// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use crate::{
    layout::LayoutConfig,
    transfer::{
        PhysicalLayout, TransferManager,
        executor::{TransferOptionsInternal, execute_transfer},
    },
};
use dynamo_memory::nixl::NixlAgent;

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
/// # use kvbm_physical::transfer::TransferCapabilities;
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

    /// PR-7.4: Enable CUDA graph capture/replay for repeated same-shape
    /// transfers on the Cuda* route family.
    ///
    /// When enabled:
    /// - Same-shape Cuda* transfers may be executed via a pre-captured
    ///   `cudaGraphExec_t` with per-launch address rebinding, reducing
    ///   kernel-launch overhead for frequently repeated shapes.
    ///
    /// When disabled (default):
    /// - All Cuda* transfers use the standard launch path. This is the
    ///   safe default because graph capture has CUDA-side preconditions
    ///   (stream capture mode, no blocking ops, etc.) that callers must
    ///   verify before enabling.
    ///
    /// **Status (PR-7.4):** Scaffolding only. No path emits
    /// `Candidate::CudaGraphReplay` today, so this flag has no runtime
    /// effect. The capture/replay executor wiring is deferred to PR-7.4.1.
    #[serde(default)]
    pub cuda_graph_replay: bool,

    /// PR-7.5: Enable optional startup benchmarking.
    ///
    /// When enabled:
    /// - Callers may invoke `TransferContext::benchmark_pair` to empirically
    ///   measure candidate submit latencies for a given layout-pair key.
    /// - The scorer consults `BenchmarkCache` and applies a +500 bonus to
    ///   the empirically fastest candidate, overriding the static score
    ///   constants when a cache entry is present.
    ///
    /// When disabled (default):
    /// - `BenchmarkCache` is never populated or consulted; the scorer uses
    ///   the existing static constants unchanged.  Production behaviour is
    ///   identical to pre-PR-7.5.
    ///
    /// **Correctness invariant:** the benchmark result only influences
    /// selection, not dispatch.  A wrong or stale benchmark entry can
    /// cause a suboptimal candidate to be chosen but cannot corrupt data.
    ///
    /// **Status (PR-7.5):** DirectDma candidates only; NIXL + transform
    /// benchmarking deferred to PR-7.5.1.
    #[serde(default)]
    pub startup_benchmark: bool,
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
            cuda_graph_replay: false, // remains opt-in; requires capturable stream
            startup_benchmark: false, // opt-in; caller decides when to benchmark
        }
    }

    /// Set the GDS (GPU Direct Storage) capability.
    pub fn with_gds(mut self, enabled: bool) -> Self {
        self.allow_gds = enabled;
        self
    }

    fn test_gds_transfer(&self) -> anyhow::Result<()> {
        let agent = NixlAgent::with_backends("agent", &["GDS_MT"])?;

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

        let ctx = TransferManager::builder()
            .nixl_agent(agent)
            .cuda_device_id(0)
            .build()?;

        execute_transfer(
            &src,
            &dst,
            &src_blocks,
            &dst_blocks,
            TransferOptionsInternal::default(),
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

    /// PR-7.4: Set the CUDA graph capture/replay capability.
    ///
    /// Defaulting to `false` is intentional — graph capture has
    /// CUDA-side preconditions (stream capture mode, no blocking ops, etc.)
    /// that the caller must verify. Enable only when the caller is certain
    /// the transfer stream is graph-capturable and the shapes are stable
    /// enough to amortise the capture cost.
    ///
    /// **Status (PR-7.4):** Flag scaffolding only; no executor path exists yet.
    /// Enabling this flag currently has no runtime effect (no path emits
    /// `Candidate::CudaGraphReplay`). Full wiring deferred to PR-7.4.1.
    pub fn with_cuda_graph_replay(mut self, enabled: bool) -> Self {
        self.cuda_graph_replay = enabled;
        self
    }

    /// PR-7.5: Set the optional startup benchmarking capability.
    ///
    /// When enabled, the scorer will consult `BenchmarkCache` and apply a
    /// +500 bonus to empirically measured winners.  Populate the cache by
    /// calling `TransferContext::benchmark_pair` at startup.
    ///
    /// Correctness is not affected — the benchmark only influences
    /// candidate selection, not dispatch.  Disable (default) in production
    /// unless you have verified that startup benchmarking on your hardware
    /// reliably identifies a performance winner.
    pub fn with_startup_benchmark(mut self, enabled: bool) -> Self {
        self.startup_benchmark = enabled;
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

#[cfg(all(test, feature = "testing-kvbm"))]
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
