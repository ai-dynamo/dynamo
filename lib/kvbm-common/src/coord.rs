// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-axis coordinate keyed by [`KvDim`] label.
//!
//! [`CoordByLabel`] is the input type for label-driven address arithmetic
//! over a [`KvDimLayout`] / `LayoutView`. Each axis can be `Some(v)` to
//! supply a coordinate or `None` to indicate that axis is not part of the
//! caller's layout (e.g. MLA has no `HeadCount`).
//!
//! Backed by a fixed array indexed by [`KvDim`] ordinal: `KvDim` is a
//! closed enum, so adding a new variant is already a load-bearing semantic
//! change that demands recompiling every consumer of `CoordByLabel`.

use crate::tensor::KvDim;

/// Total number of [`KvDim`] variants.
///
/// Kept as a hand-maintained constant; incrementing this requires
/// extending [`KvDim::ALL`] and the `From<KvDim> for usize` mapping.
const KV_DIM_COUNT: usize = 7;

const fn kv_dim_index(d: KvDim) -> usize {
    match d {
        KvDim::Block => 0,
        KvDim::Layer => 1,
        KvDim::Outer => 2,
        KvDim::Page => 3,
        KvDim::HeadCount => 4,
        KvDim::HeadSize => 5,
        KvDim::Payload => 6,
    }
}

/// Label-keyed coordinate. Set axes the layout uses; leave others `None`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoordByLabel {
    coords: [Option<usize>; KV_DIM_COUNT],
}

impl CoordByLabel {
    /// Empty coordinate (every axis `None`).
    pub fn new() -> Self {
        Self {
            coords: [None; KV_DIM_COUNT],
        }
    }

    /// Builder-style setter.
    pub fn with(mut self, dim: KvDim, value: usize) -> Self {
        self.coords[kv_dim_index(dim)] = Some(value);
        self
    }

    /// In-place setter.
    pub fn set(&mut self, dim: KvDim, value: usize) {
        self.coords[kv_dim_index(dim)] = Some(value);
    }

    /// Read the coordinate for `dim`, if any.
    pub fn get(&self, dim: KvDim) -> Option<usize> {
        self.coords[kv_dim_index(dim)]
    }

    /// True iff the caller has supplied a coordinate for `dim`.
    pub fn has(&self, dim: KvDim) -> bool {
        self.coords[kv_dim_index(dim)].is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_all_axes() {
        let mut c = CoordByLabel::new();
        for (i, d) in [
            KvDim::Block,
            KvDim::Layer,
            KvDim::Outer,
            KvDim::Page,
            KvDim::HeadCount,
            KvDim::HeadSize,
            KvDim::Payload,
        ]
        .iter()
        .enumerate()
        {
            c.set(*d, i);
        }
        assert_eq!(c.get(KvDim::Block), Some(0));
        assert_eq!(c.get(KvDim::Layer), Some(1));
        assert_eq!(c.get(KvDim::Outer), Some(2));
        assert_eq!(c.get(KvDim::Page), Some(3));
        assert_eq!(c.get(KvDim::HeadCount), Some(4));
        assert_eq!(c.get(KvDim::HeadSize), Some(5));
        assert_eq!(c.get(KvDim::Payload), Some(6));
    }

    #[test]
    fn defaults_to_none() {
        let c = CoordByLabel::new();
        assert!(!c.has(KvDim::Block));
        assert_eq!(c.get(KvDim::Block), None);
    }

    #[test]
    fn builder_style_with() {
        let c = CoordByLabel::new()
            .with(KvDim::Block, 42)
            .with(KvDim::Page, 7);
        assert_eq!(c.get(KvDim::Block), Some(42));
        assert_eq!(c.get(KvDim::Page), Some(7));
        assert!(!c.has(KvDim::HeadCount));
    }
}
