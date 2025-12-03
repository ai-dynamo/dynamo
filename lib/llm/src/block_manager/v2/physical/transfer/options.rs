// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer options for configuring block and layer transfers.

use super::BounceBufferSpec;
use derive_builder::Builder;
use std::{ops::Range, sync::Arc};

/// Hints for how to build transfer descriptors.
///
/// These hints control how blocks are mapped to descriptors, applicable to
/// any backend that uses descriptors (NIXL object storage, GDS disk, RDMA, etc.)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DescriptorHint {
    /// Auto-detect from layout characteristics (default).
    #[default]
    Auto,

    /// One descriptor per block, each at its own offset within a shared target.
    ///
    /// Use cases:
    /// - Object: byte-range reads from one object key (TP partitions)
    /// - Disk: reads from different offsets in one file
    /// - RDMA: reads from different offsets in remote memory
    ///
    /// ```text
    /// Shared Target (object / file / remote buffer):
    /// ┌───────┬───────┬───────┬───────┐
    /// │  0    │  1    │  2    │  3    │  (offsets)
    /// └───────┴───────┴───────┴───────┘
    ///    ↓       ↓       ↓       ↓
    /// Block 0  Block 1  Block 2  Block 3
    /// ```
    PerBlockWithOffset,

    /// One descriptor per block, each targeting a unique destination.
    ///
    /// Use cases:
    /// - Object: write to separate object keys (one-to-one mapping)
    /// - Disk: write to separate files
    /// - RDMA: write to separate remote buffers
    ///
    /// ```text
    /// Block 0 → Target A (key/file/buffer 0)
    /// Block 1 → Target B (key/file/buffer 1)
    /// Block 2 → Target C (key/file/buffer 2)
    /// ```
    PerBlockUniqueTarget,

    /// All blocks coalesced into a single descriptor to one target.
    ///
    /// Use cases:
    /// - Object: write all blocks to one object key
    /// - Disk: write all blocks to one file region
    /// - RDMA: single large transfer
    ///
    /// ```text
    /// Block 0  Block 1  Block 2  Block 3
    ///    └────────┴────────┴────────┘
    ///                 ↓
    ///          Single Target
    /// ```
    Coalesced,

    /// Batch contiguous block ranges into fewer descriptors.
    ///
    /// Example: [0,1,2, 10,11, 50] → 3 descriptors instead of 6
    ///
    /// Use cases:
    /// - Host memory transfers with fragmented block lists
    /// - Reducing descriptor overhead for large transfers
    BatchedRanges,
}

/// Options for configuring transfer operations.
///
/// This structure provides configuration for block and layer transfers,
/// including layer ranges, NIXL write notifications, and bounce buffers.
///
/// # Examples
///
/// ```rust,ignore
/// let options = TransferOptions::builder()
///     .nixl_write_notification(42)
///     .layer_range(0..10)
///     .build();
/// ```
#[derive(Clone, Default, Builder)]
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
    pub bounce_buffer: Option<Arc<dyn BounceBufferSpec>>,

    /// Hint for how to build transfer descriptors.
    ///
    /// When `Auto` (default), the executor will analyze layout characteristics
    /// to determine the optimal descriptor pattern. Providing explicit hints
    /// can skip detection overhead and ensure correct behavior.
    ///
    /// # Example
    /// ```rust,ignore
    /// // TP partition read: byte-range reads from shared object
    /// let options = TransferOptions::builder()
    ///     .descriptor_hint(DescriptorHint::PerBlockWithOffset)
    ///     .build()?;
    /// ```
    #[builder(default)]
    pub descriptor_hint: DescriptorHint,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let options = TransferOptions::default();
        assert!(options.layer_range.is_none());
        assert!(options.nixl_write_notification.is_none());
        assert!(options.bounce_buffer.is_none());
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
            .build()
            .unwrap();

        assert_eq!(options.nixl_write_notification, Some(100));
        assert_eq!(options.layer_range, Some(5..15));
    }
}
