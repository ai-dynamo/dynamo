// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer options for configuring block and layer transfers.

use super::BounceBuffer;
use crate::device::DeviceStream;
use crate::layout::KvBlockLayout;
use derive_builder::Builder;
use derive_getters::Dissolve;
use kvbm_common::KvbmTransferRoute;
use std::ops::Range;
use std::sync::Arc;

/// Options for configuring transfer operations.
///
/// This structure provides configuration for block and layer transfers,
/// including layer ranges, NIXL write notifications, and bounce buffers.
///
/// Caller-provided streams flow through `device_stream` — a unified
/// multi-backend `Arc<DeviceStream>` consumed by both the XPU/CUDA
/// device executor and the CUDA planner path. The planner downcasts
/// to a concrete `Arc<CudaStream>` internally.
///
/// # Examples
///
/// ```rust,ignore
/// let options = TransferOptions::builder()
///     .nixl_write_notification(42)
///     .layer_range(0..10)
///     .build();
/// ```
#[derive(Clone, Default, Builder, Dissolve)]
#[builder(pattern = "owned", default)]
pub struct TransferOptions {
    /// Range of layers to transfer (None = all layers).
    ///
    /// When specified, only the layers in this range will be transferred.
    /// This is useful for partial block transfers or layer-specific operations.
    #[builder(default, setter(strip_option))]
    pub layer_range: Option<Range<usize>>,

    /// NIXL write notification value delivered after RDMA write completes.
    ///
    /// When specified, NIXL will deliver this notification value to the remote
    /// node after the RDMA write operation completes. This enables efficient
    /// notification of transfer completion without requiring polling.
    #[builder(default, setter(strip_option))]
    pub nixl_write_notification: Option<u64>,

    /// Bounce buffer specification for multi-hop transfers.
    ///
    /// When direct transfers are not allowed or efficient, this specifies
    /// an intermediate staging area. The transfer will be split into two hops:
    /// source → bounce buffer → destination.
    #[builder(default, setter(strip_option, into))]
    pub bounce_buffer: Option<BounceBuffer>,

    /// Override source block layout interpretation.
    ///
    /// When set, the transfer executor will treat source blocks as having
    /// this layout instead of the layout's default block_layout().
    /// This enables transferring blocks that are stored in one format
    /// but should be interpreted as another (e.g., operational → universal).
    #[builder(default, setter(strip_option))]
    pub src_kv_layout: Option<KvBlockLayout>,

    /// Override destination block layout interpretation.
    ///
    /// When set, the transfer executor will treat destination blocks as having
    /// this layout instead of the layout's default block_layout().
    /// This enables writing blocks in a different format than the destination
    /// layout's native format.
    #[builder(default, setter(strip_option))]
    pub dst_kv_layout: Option<KvBlockLayout>,

    /// Caller-provided device stream.
    ///
    /// When set, the transfer executor uses this stream instead of acquiring
    /// one from the context pool, and skips event recording (caller manages
    /// synchronization). Returns `completed()` immediately. Used by both the
    /// multi-backend device executor and the CUDA planner path.
    #[builder(default, setter(strip_option))]
    pub device_stream: Option<Arc<DeviceStream>>,

    /// Optional logical route for transfer metrics.
    ///
    /// This is attached by higher layers that know the semantic tier movement
    /// so the physical executor can emit compatibility metrics on completion.
    #[builder(default, setter(strip_option))]
    pub metric_route: Option<KvbmTransferRoute>,

    /// Route the transfer through the stride-aware planner
    /// (`transfer::plan` → `transfer::lower` → `executor::planner`).
    ///
    /// Default `false` — the legacy `select_strategy` /
    /// `execute_direct_transfer` path runs unchanged. When `true`, the
    /// transfer goes through `plan_copy`, lowers `CopyPlan::Direct` to
    /// `Candidate::DirectDma`, and executes via the planner-aware
    /// Async path. CUDA backend only — under `xpu-sycl` builds the
    /// internal options builder bails when `use_planner=true` is combined
    /// with kernel transforms or non-empty `axis_slices`.
    #[builder(default)]
    pub use_planner: bool,
}

impl TransferOptions {
    /// Create a new builder for transfer options.
    pub fn builder() -> TransferOptionsBuilder {
        TransferOptionsBuilder::default()
    }

    /// Create transfer options from an optional layer range.
    pub fn from_layer_range(layer_range: Option<Range<usize>>) -> Self {
        Self {
            layer_range,
            ..Self::default()
        }
    }

    /// Create default transfer options.
    ///
    /// This transfers all layers with no special configuration.
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let options = TransferOptions::default();
        assert!(options.layer_range.is_none());
        assert!(options.nixl_write_notification.is_none());
        assert!(options.bounce_buffer.is_none());
        assert!(options.device_stream.is_none());
        assert!(options.metric_route.is_none());
        assert!(!options.use_planner);
    }

    #[test]
    fn test_builder_with_notification() {
        let options = TransferOptions::builder()
            .nixl_write_notification(42)
            .build()
            .unwrap();

        assert_eq!(options.nixl_write_notification, Some(42));
        assert!(options.layer_range.is_none());
    }

    #[test]
    fn test_builder_with_layer_range() {
        let options = TransferOptions::builder()
            .layer_range(0..10)
            .build()
            .unwrap();

        assert_eq!(options.layer_range, Some(0..10));
        assert!(options.nixl_write_notification.is_none());
    }

    #[test]
    fn test_builder_with_all_options() {
        let options = TransferOptions::builder()
            .nixl_write_notification(100)
            .layer_range(5..15)
            .use_planner(true)
            .build()
            .unwrap();

        assert_eq!(options.nixl_write_notification, Some(100));
        assert_eq!(options.layer_range, Some(5..15));
        assert!(options.use_planner);
    }
}
