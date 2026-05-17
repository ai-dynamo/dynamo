// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical (un-sharded) block shape used by cross-leader compatibility
//! checks under [`BlockLayoutMode`].
//!
//! [`CanonicalBlockShape`] is the pure-scalar form of the un-sharded
//! aggregate tensor produced by re-collecting every worker's slice along
//! the sharded axis (heads for TP, layers for PP). It is the *minimum*
//! information needed for the universal-mode equality check; the
//! operational-mode predicate additionally compares per-worker shape via
//! `kvbm_protocols::control::layout_compat::LayoutCompatPayload`.
//!
//! See `kvbm_common::block_layout_mode` for the full mode-level
//! semantics.
//!
//! [`BlockLayoutMode`]: crate::BlockLayoutMode

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Canonical (un-sharded) block descriptor.
///
/// All shape-bearing fields of the canonical aggregate tensor that a
/// universal-mode compatibility check has to compare. Pure scalars; no
/// dependency on physical / wire types so this type can be reused at the
/// hub, the engine, and anywhere on the control plane.
///
/// - `num_layers_total` — un-sharded layer count (== sum of every PP
///   rank's `layer_ownership.len()`).
/// - `num_heads_total` — un-sharded head count (== sum of every TP rank's
///   per-worker `num_heads`).
/// - `head_dim` — per-head feature dimension (== `inner_dim / num_heads`).
/// - `outer_dim`, `page_size`, `dtype_width_bytes` — un-sharded fields
///   copied verbatim from any worker's `LayoutConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalBlockShape {
    pub num_layers_total: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub num_heads_total: usize,
    pub head_dim: usize,
    pub dtype_width_bytes: usize,
}

impl CanonicalBlockShape {
    /// Compare two canonical shapes and return a precise per-field error
    /// when they differ.
    ///
    /// The error message names exactly which dimension diverged
    /// (`canonical num_heads_total differs (local=64, remote=48)`) so the
    /// operator can act on the rejection without re-decoding two opaque
    /// shapes.
    pub fn require_equal(&self, other: &Self) -> Result<()> {
        if self.num_layers_total != other.num_layers_total {
            bail!(
                "canonical num_layers_total differs (local={}, remote={})",
                self.num_layers_total,
                other.num_layers_total,
            );
        }
        if self.outer_dim != other.outer_dim {
            bail!(
                "canonical outer_dim differs (local={}, remote={})",
                self.outer_dim,
                other.outer_dim,
            );
        }
        if self.page_size != other.page_size {
            bail!(
                "canonical page_size differs (local={}, remote={})",
                self.page_size,
                other.page_size,
            );
        }
        if self.num_heads_total != other.num_heads_total {
            bail!(
                "canonical num_heads_total differs (local={}, remote={})",
                self.num_heads_total,
                other.num_heads_total,
            );
        }
        if self.head_dim != other.head_dim {
            bail!(
                "canonical head_dim differs (local={}, remote={})",
                self.head_dim,
                other.head_dim,
            );
        }
        if self.dtype_width_bytes != other.dtype_width_bytes {
            bail!(
                "canonical dtype_width_bytes differs (local={}, remote={})",
                self.dtype_width_bytes,
                other.dtype_width_bytes,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(layers: usize, heads: usize) -> CanonicalBlockShape {
        CanonicalBlockShape {
            num_layers_total: layers,
            outer_dim: 2,
            page_size: 16,
            num_heads_total: heads,
            head_dim: 128,
            dtype_width_bytes: 2,
        }
    }

    #[test]
    fn require_equal_passes_on_identical() {
        s(32, 64).require_equal(&s(32, 64)).unwrap();
    }

    #[test]
    fn require_equal_names_diverging_num_layers() {
        let err = s(32, 64).require_equal(&s(28, 64)).unwrap_err();
        let m = format!("{err}");
        assert!(m.contains("num_layers_total"), "got: {m}");
        assert!(
            m.contains("local=32") && m.contains("remote=28"),
            "got: {m}"
        );
    }

    #[test]
    fn require_equal_names_diverging_num_heads() {
        let err = s(32, 64).require_equal(&s(32, 48)).unwrap_err();
        let m = format!("{err}");
        assert!(m.contains("num_heads_total"), "got: {m}");
    }

    #[test]
    fn require_equal_names_diverging_head_dim() {
        let mut a = s(32, 64);
        let mut b = s(32, 64);
        a.head_dim = 128;
        b.head_dim = 64;
        let err = a.require_equal(&b).unwrap_err();
        assert!(format!("{err}").contains("head_dim"));
    }

    #[test]
    fn require_equal_names_diverging_outer_dim() {
        let mut a = s(32, 64);
        let mut b = s(32, 64);
        a.outer_dim = 2;
        b.outer_dim = 1;
        let err = a.require_equal(&b).unwrap_err();
        assert!(format!("{err}").contains("outer_dim"));
    }

    #[test]
    fn require_equal_names_diverging_page_size() {
        let mut a = s(32, 64);
        let mut b = s(32, 64);
        a.page_size = 16;
        b.page_size = 32;
        let err = a.require_equal(&b).unwrap_err();
        assert!(format!("{err}").contains("page_size"));
    }

    #[test]
    fn require_equal_names_diverging_dtype_width() {
        let mut a = s(32, 64);
        let mut b = s(32, 64);
        a.dtype_width_bytes = 2;
        b.dtype_width_bytes = 4;
        let err = a.require_equal(&b).unwrap_err();
        assert!(format!("{err}").contains("dtype_width_bytes"));
    }

    #[test]
    fn serde_round_trip() {
        let a = s(32, 64);
        let s = serde_json::to_string(&a).unwrap();
        let back: CanonicalBlockShape = serde_json::from_str(&s).unwrap();
        assert_eq!(back, a);
    }
}
