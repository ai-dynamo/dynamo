// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kernel catalog for the planner-driven CudaAsync executor.
//!
//! Maps `(src KvBlockLayout, dst KvBlockLayout, dtype) → KernelKind`
//! and bundles the launch parameters into a [`KernelInvocation`] the
//! executor can dispatch.
//!
//! ## Match-key choice (plan deviation)
//!
//! The PR-6 plan called for matching by `LayoutSignature` pair. The
//! registered kernels (`universal_from_block` / `block_from_universal`)
//! parameterise dimensions at launch time and don't care about size
//! equality, so the discrimination collapses to `KvBlockLayout` —
//! the same enum already carried by every `PhysicalLayout`. Using
//! `(KvBlockLayout, KvBlockLayout, dtype)` directly avoids a redundant
//! signature build at every catalog lookup and keeps the dispatch
//! table small and readable. Future PRs can promote this to
//! signature-based matching if a kernel needs finer discrimination
//! (e.g. tiled vs scalar variants of the same shape pair).
//!
//! ## Concrete-enum dispatch (plan-confirmed)
//!
//! `KernelKind` is a concrete enum, not a trait object. Adding a new
//! kernel = adding a variant + a match arm. At 2 kernels (PR-6.1) and
//! at most ~5 anticipated through PR-7, the abstraction overhead of
//! `Box<dyn KernelImpl>` would earn nothing.

use kvbm_kernels::TensorDataType;

use crate::layout::KvBlockLayout;

/// One of the kernels registered in the catalog.
///
/// Each variant corresponds to a single FFI entrypoint in
/// `kvbm-kernels`. The variant carries no data — `KernelInvocation`
/// holds the launch params; this enum just tags which kernel to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelKind {
    /// `universal_from_block`: operational (NHD/HND) → universal.
    UniversalFromBlock,
    /// `block_from_universal`: universal → operational (NHD/HND).
    BlockFromUniversal,
    /// `nhd_hnd_transpose`: operational ↔ operational (NHD ↔ HND).
    /// One FFI symbol with a `src_layout` flag; the dispatcher reads
    /// `KernelInvocation::block_layout` for the direction.
    NhdHndTranspose,
}

/// Launch parameters for a registered kernel.
///
/// Constructed by [`build_invocation`] at planning time once
/// `match_kernel` has resolved `(src_kv, dst_kv, dtype) → KernelKind`.
/// The executor reads `src_regions` / `dst_regions` (chunk-level
/// pointer arrays for the operational side, per-block bases for the
/// universal side) directly from the layouts at dispatch.
///
/// Shape scalars are duplicated from `LayoutConfig` for two reasons:
/// (1) the `block_layout` field discriminates NHD vs HND for the
/// operational side without re-projecting; (2) the executor doesn't
/// need to re-parse the `LayoutConfig` to issue a launch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelInvocation {
    pub kind: KernelKind,
    pub num_layers: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dtype: TensorDataType,
    /// NHD or HND inner-token shape on the operational side. The
    /// kernel selects its template from this enum value at launch.
    pub block_layout: kvbm_kernels::BlockLayout,
}

/// Look up a kernel for the (src, dst, dtype) tuple.
///
/// Returns `None` when the layouts are identical (no transform
/// required — caller should stay on the Direct path) or when no
/// registered kernel covers the pair (catalog miss; caller should
/// surface a precise error).
pub(crate) fn match_kernel(
    src_kv: KvBlockLayout,
    dst_kv: KvBlockLayout,
    dtype: TensorDataType,
) -> Option<KernelKind> {
    use KvBlockLayout::*;
    // FP8 has no kernel template specialization today; reject at the
    // catalog so the launcher never sees an unmappable dtype. When
    // FP8 kernel support lands this guard goes away.
    if matches!(dtype, TensorDataType::FP8) {
        return None;
    }
    match (src_kv, dst_kv) {
        // Operational → Universal: universal_from_block kernel.
        // The template selects NHD vs HND from `block_layout` at
        // launch; the universal side is the canonical
        // `[Head, Layer, Outer, Page]` axis order.
        (OperationalNHD | OperationalHND, Universal) => Some(KernelKind::UniversalFromBlock),
        // Universal → Operational: block_from_universal.
        (Universal, OperationalNHD | OperationalHND) => Some(KernelKind::BlockFromUniversal),
        // Operational ↔ Operational (PR-6.3): nhd_hnd_transpose.
        // Direction is carried on `KernelInvocation::block_layout`
        // (the kernel's `src_layout` template parameter).
        (OperationalNHD, OperationalHND) | (OperationalHND, OperationalNHD) => {
            Some(KernelKind::NhdHndTranspose)
        }
        // Same layout: no transform needed (defensive — `requires_transform`
        // also reports false here, so plan_copy would emit Direct).
        (a, b) if a == b => None,
        // Custom layouts are not covered by the catalog.
        _ => None,
    }
}

/// Convert a `KvBlockLayout` to the kernel-side `BlockLayout` enum.
///
/// Operational variants map to NHD/HND. Universal variants don't have
/// a "block layout" in the kernel's sense; this returns `None` so
/// callers know they must consult the *operational* side of a
/// transform pair to derive the kernel's template parameter.
pub(crate) fn to_kernel_block_layout(kv: KvBlockLayout) -> Option<kvbm_kernels::BlockLayout> {
    match kv {
        KvBlockLayout::OperationalNHD => Some(kvbm_kernels::BlockLayout::NHD),
        KvBlockLayout::OperationalHND => Some(kvbm_kernels::BlockLayout::HND),
        _ => None,
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use crate::layout::KvBlockLayout::*;

    #[test]
    fn matches_operational_to_universal() {
        assert_eq!(
            match_kernel(OperationalNHD, Universal, TensorDataType::F16),
            Some(KernelKind::UniversalFromBlock),
        );
        assert_eq!(
            match_kernel(OperationalHND, Universal, TensorDataType::BF16),
            Some(KernelKind::UniversalFromBlock),
        );
    }

    #[test]
    fn matches_universal_to_operational() {
        assert_eq!(
            match_kernel(Universal, OperationalNHD, TensorDataType::F32),
            Some(KernelKind::BlockFromUniversal),
        );
        assert_eq!(
            match_kernel(Universal, OperationalHND, TensorDataType::F16),
            Some(KernelKind::BlockFromUniversal),
        );
    }

    /// Same-layout pairs return None — there's no transform required,
    /// and the planner should stay on the Direct path.
    #[test]
    fn same_layout_returns_none() {
        for kv in [OperationalNHD, OperationalHND, Universal] {
            assert_eq!(match_kernel(kv, kv, TensorDataType::F16), None);
        }
    }

    /// Operational ↔ Operational (NHD ↔ HND): PR-6.3 added a dedicated
    /// transpose kernel. Both directions resolve to `NhdHndTranspose`;
    /// the dispatcher reads the direction off
    /// `KernelInvocation::block_layout`.
    #[test]
    fn matches_nhd_hnd_transpose() {
        assert_eq!(
            match_kernel(OperationalNHD, OperationalHND, TensorDataType::F16),
            Some(KernelKind::NhdHndTranspose),
        );
        assert_eq!(
            match_kernel(OperationalHND, OperationalNHD, TensorDataType::F16),
            Some(KernelKind::NhdHndTranspose),
        );
    }

    /// Reproducer for c1: single Universal variant dispatches the
    /// universal_from_block / block_from_universal kernels in both
    /// directions for the operational pair.
    #[test]
    fn matches_universal_round_trip() {
        assert_eq!(
            match_kernel(OperationalNHD, Universal, TensorDataType::F16),
            Some(KernelKind::UniversalFromBlock),
        );
        assert_eq!(
            match_kernel(Universal, OperationalNHD, TensorDataType::F16),
            Some(KernelKind::BlockFromUniversal),
        );
    }

    /// Custom layouts are out of scope for the catalog.
    #[test]
    fn custom_returns_none() {
        use kvbm_common::BlockDim;
        let custom = Custom([
            BlockDim::Layer,
            BlockDim::Outer,
            BlockDim::Page,
            BlockDim::Head,
        ]);
        assert_eq!(
            match_kernel(custom, OperationalNHD, TensorDataType::F16),
            None,
        );
    }

    #[test]
    fn to_kernel_block_layout_round_trip() {
        assert_eq!(
            to_kernel_block_layout(OperationalNHD),
            Some(kvbm_kernels::BlockLayout::NHD)
        );
        assert_eq!(
            to_kernel_block_layout(OperationalHND),
            Some(kvbm_kernels::BlockLayout::HND)
        );
        assert_eq!(to_kernel_block_layout(Universal), None);
    }
}
