// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-axis byte strides paired with a [`KvDimLayout`].
//!
//! [`KvDimStrides`] carries the *physical* access pattern that
//! [`KvDimLayout`] (the *schema*) is silent about: how far apart adjacent
//! elements are in memory along each labelled axis. The two travel
//! together — strides without their layout are uninterpretable, and a
//! layout without strides cannot address memory.
//!
//! Strides are stored in **bytes** so that downstream address arithmetic
//! can mix them with raw pointer offsets without re-multiplying by
//! `elem_size`. PyTorch reports element strides; convert at construction
//! via [`KvDimStrides::from_element_strides`].

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::tensor::KvDimLayout;

/// Per-axis byte strides parallel to a [`KvDimLayout`]'s `dims`/`sizes`.
///
/// Carries `elem_size` alongside the stride vector so contiguity queries
/// can't be passed a mismatched dtype width — strides and `elem_size`
/// came from the same tensor, they should travel together.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDimStrides {
    bytes: Vec<usize>,
    elem_size: usize,
}

impl KvDimStrides {
    /// Build from PyTorch-style element strides (`tensor.stride()`).
    ///
    /// Validates that the stride vector is non-empty, that `elem_size`
    /// is positive, and that all entries are positive — zero stride
    /// would imply a broadcast view, which KV caches do not expose.
    pub fn from_element_strides(elem_strides: &[usize], elem_size: usize) -> Result<Self> {
        if elem_strides.is_empty() {
            bail!("KvDimStrides: element strides must be non-empty");
        }
        if elem_size == 0 {
            bail!("KvDimStrides: elem_size must be > 0");
        }
        for (i, &s) in elem_strides.iter().enumerate() {
            if s == 0 {
                bail!(
                    "KvDimStrides: stride at axis {i} is zero; broadcast \
                     views are not supported for KV caches"
                );
            }
        }
        let bytes = elem_strides.iter().map(|s| s * elem_size).collect();
        Ok(Self { bytes, elem_size })
    }

    /// Build directly from byte strides (already multiplied by element
    /// size). Same validation rules as [`from_element_strides`].
    pub fn from_byte_strides(byte_strides: Vec<usize>, elem_size: usize) -> Result<Self> {
        if byte_strides.is_empty() {
            bail!("KvDimStrides: byte strides must be non-empty");
        }
        if elem_size == 0 {
            bail!("KvDimStrides: elem_size must be > 0");
        }
        for (i, &s) in byte_strides.iter().enumerate() {
            if s == 0 {
                bail!("KvDimStrides: stride at axis {i} is zero");
            }
        }
        Ok(Self {
            bytes: byte_strides,
            elem_size,
        })
    }

    /// Strides as bytes per axis.
    pub fn as_bytes(&self) -> &[usize] {
        &self.bytes
    }

    /// Element size (bytes) the strides were built against.
    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    /// Row-major-from-sizes byte strides — the strides a tensor would
    /// have if it were fully contiguous in `layout`'s labelled order.
    ///
    /// `bytes[k] = elem_size * Π sizes[k+1..]`.
    pub fn contiguous_for(layout: &KvDimLayout, elem_size: usize) -> Self {
        let sizes = layout.sizes();
        let n = sizes.len();
        let mut bytes = vec![0usize; n];
        let mut acc = elem_size;
        for k in (0..n).rev() {
            bytes[k] = acc;
            acc *= sizes[k];
        }
        Self { bytes, elem_size }
    }

    /// Length (in bytes) of the contiguous suffix when measured against
    /// `layout`. Walks innermost axes outward; stops at the first axis
    /// `k` where `bytes[k] != bytes[k+1] * sizes[k+1]`, or where the
    /// trailing stride differs from `elem_size`.
    ///
    /// Returns `0` when even the innermost axis is non-contiguous (its
    /// stride disagrees with `elem_size`); returns the full layout
    /// extent when every axis is contiguous.
    ///
    /// Errors when `layout.dims().len() != self.bytes.len()`.
    pub fn contiguous_tail_bytes(&self, layout: &KvDimLayout) -> Result<usize> {
        let sizes = layout.sizes();
        if sizes.len() != self.bytes.len() {
            bail!(
                "KvDimStrides: rank mismatch — layout has {} axes, strides have {}",
                sizes.len(),
                self.bytes.len()
            );
        }
        let n = sizes.len();
        // Innermost axis: stride must equal elem_size.
        if self.bytes[n - 1] != self.elem_size {
            return Ok(0);
        }
        let mut tail_bytes = self.elem_size * sizes[n - 1];
        let mut last_stride = self.bytes[n - 1];
        let mut last_size = sizes[n - 1];
        for k in (0..n - 1).rev() {
            let expected = last_stride * last_size;
            if self.bytes[k] != expected {
                break;
            }
            tail_bytes *= sizes[k];
            last_stride = self.bytes[k];
            last_size = sizes[k];
        }
        Ok(tail_bytes)
    }

    /// True iff the strides describe a tensor fully row-major-contiguous
    /// in `layout`'s labelled order.
    ///
    /// Errors on rank disagreement.
    pub fn is_fully_contiguous(&self, layout: &KvDimLayout) -> Result<bool> {
        if layout.dims().len() != self.bytes.len() {
            bail!(
                "KvDimStrides: rank mismatch — layout has {} axes, strides have {}",
                layout.dims().len(),
                self.bytes.len()
            );
        }
        let expected = Self::contiguous_for(layout, self.elem_size);
        Ok(expected.bytes == self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::KvDim;

    fn nhd_layout() -> KvDimLayout {
        KvDimLayout::new(
            vec![
                KvDim::Outer,
                KvDim::Block,
                KvDim::Page,
                KvDim::HeadCount,
                KvDim::HeadSize,
            ],
            vec![2, 1024, 16, 8, 128],
        )
        .unwrap()
    }

    #[test]
    fn from_element_strides_converts_to_bytes() {
        let s = KvDimStrides::from_element_strides(&[16384, 16, 1], 2).unwrap();
        assert_eq!(s.as_bytes(), &[32768, 32, 2]);
        assert_eq!(s.elem_size(), 2);
    }

    #[test]
    fn rejects_empty_strides() {
        assert!(KvDimStrides::from_element_strides(&[], 2).is_err());
    }

    #[test]
    fn rejects_zero_stride() {
        assert!(KvDimStrides::from_element_strides(&[0, 1], 2).is_err());
    }

    #[test]
    fn rejects_zero_elem_size() {
        assert!(KvDimStrides::from_element_strides(&[1], 0).is_err());
    }

    #[test]
    fn contiguous_for_matches_row_major() {
        let layout = nhd_layout();
        // sizes = [2, 1024, 16, 8, 128], elem_size = 2 (fp16)
        // expected byte strides = [1024*16*8*128*2, 16*8*128*2, 8*128*2, 128*2, 2]
        //                       = [33554432, 32768, 2048, 256, 2]
        let s = KvDimStrides::contiguous_for(&layout, 2);
        assert_eq!(s.as_bytes(), &[33554432, 32768, 2048, 256, 2]);
        assert_eq!(s.elem_size(), 2);
    }

    #[test]
    fn fully_contiguous_round_trip() {
        let layout = nhd_layout();
        let s = KvDimStrides::contiguous_for(&layout, 2);
        assert!(s.is_fully_contiguous(&layout).unwrap());
        assert_eq!(
            s.contiguous_tail_bytes(&layout).unwrap(),
            2 * 1024 * 16 * 8 * 128 * 2
        );
    }

    #[test]
    fn hnd_permutation_collapses_inner_tail() {
        // FlashAttention HND: stride order (0, 1, 3, 2, 4) over the
        // logical NHD shape [Outer, Block, Page, HeadCount, HeadSize].
        // After the permutation, HeadCount sits between Block and Page in
        // memory, so the contiguous tail is just [HeadSize].
        // sizes = [2, 1024, 16, 8, 128], elem_size = 2
        // physical: outermost stride 1024*8*16*128 = 16777216
        //                            8*16*128       = 16384
        //                            128 (Page rows of 128 elements skip  HeadCount*HeadSize? no
        //
        // Reconstruct from "physical layout"
        // Physical order is HND: [Outer, Block, HeadCount, Page, HeadSize]
        //   contiguous strides over physical shape [2, 1024, 8, 16, 128]:
        //     [1024*8*16*128, 8*16*128, 16*128, 128, 1]
        //   = [16777216, 16384, 2048, 128, 1] (elements)
        //   Then we map back to the *logical* [Outer, Block, Page, HeadCount, HeadSize]:
        //     stride_logical[0=Outer]     = 16777216
        //     stride_logical[1=Block]     = 16384
        //     stride_logical[2=Page]      = 128       (Page sits inside HeadCount in memory)
        //     stride_logical[3=HeadCount] = 2048      (HeadCount jumps over Page*HeadSize)
        //     stride_logical[4=HeadSize]  = 1
        let layout = nhd_layout();
        let s = KvDimStrides::from_element_strides(&[16777216, 16384, 128, 2048, 1], 2).unwrap();
        // Innermost axis stride matches elem_size, so tail starts at 1 * HeadSize.
        // Next axis (HeadCount) stride is 2048 elem = 4096 bytes; expected
        // for contiguous = 1 * 128 elem = 256 bytes. They disagree, so the
        // contiguous tail is exactly the HeadSize axis.
        assert_eq!(s.contiguous_tail_bytes(&layout).unwrap(), 128 * 2);
        assert!(!s.is_fully_contiguous(&layout).unwrap());
    }

    #[test]
    fn non_contiguous_innermost_returns_zero_tail() {
        let layout = nhd_layout();
        // Stride 4 instead of elem_size=2 on the innermost axis ⇒ no tail.
        let s = KvDimStrides::from_byte_strides(vec![8388608, 65536, 4096, 512, 4], 2).unwrap();
        assert_eq!(s.contiguous_tail_bytes(&layout).unwrap(), 0);
    }

    #[test]
    fn rank_mismatch_errors() {
        let layout = nhd_layout();
        let s = KvDimStrides::from_element_strides(&[1, 1], 2).unwrap();
        assert!(s.contiguous_tail_bytes(&layout).is_err());
        assert!(s.is_fully_contiguous(&layout).is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let layout = nhd_layout();
        let s = KvDimStrides::contiguous_for(&layout, 2);
        let json = serde_json::to_string(&s).unwrap();
        let de: KvDimStrides = serde_json::from_str(&json).unwrap();
        assert_eq!(s, de);
    }
}
