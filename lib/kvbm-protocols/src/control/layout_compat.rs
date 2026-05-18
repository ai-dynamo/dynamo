// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-leader block-layout compatibility wire payload and predicate.
//!
//! [`LayoutCompatPayload`] is the wire-stable description of a leader's
//! block layout shipped at hub-registration time and reused by the
//! engine's per-import enforcement. [`check_layout_compat`] is the
//! single predicate that both call sites delegate to so the hub-side
//! gate and the engine-side gate cannot drift.
//!
//! Operational mode requires per-worker `(KvBlockLayout, LayoutConfig)`
//! equality on top of canonical equality. Universal mode requires only
//! canonical equality — per-worker permutation may differ. See
//! `kvbm_common::block_layout_mode` for the full semantic write-up.

use anyhow::{Result, bail};
use kvbm_common::{BlockLayoutMode, CanonicalBlockShape, KvBlockLayout};
use serde::{Deserialize, Serialize};

use super::LayoutConfigDescription;

/// Compatibility payload one leader sends to declare its block layout to
/// peers. Carried inside the hub's `Feature::P2P` config at register
/// time (c2 — was previously inside `Feature::ConditionalDisagg`), and
/// built by the engine from each `SerializedLayout` during per-import
/// compatibility checks.
///
/// The hub stores the first-seen payload as the baseline and rejects
/// subsequent registrations whose payload is not compatible. The engine
/// builds the same payload from local + remote `SerializedLayout` and
/// runs [`check_layout_compat`] as defence-in-depth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutCompatPayload {
    /// Local startup policy. `BlockLayoutMode::Operational` is strict /
    /// bit-for-bit; `BlockLayoutMode::Universal` accepts any
    /// canonical-equal peer.
    pub mode: BlockLayoutMode,
    /// Un-sharded canonical aggregate, when the leader could derive it.
    /// `Some` for any leader whose worker metadata carries a
    /// `ParallelismDescriptor`; `None` for legacy / single-worker
    /// leaders with no global extents stamped.
    ///
    /// Universal mode requires `Some` on both sides — without canonical
    /// extents the universal predicate has nothing to compare.
    /// Operational mode treats canonical equality as defence-in-depth
    /// on top of the per-worker config check: when both sides carry
    /// `Some` they must agree; when either is `None` the canonical
    /// check is skipped and equality reduces to the per-worker fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical: Option<CanonicalBlockShape>,
    /// The leader's per-worker [`KvBlockLayout`] — operational equality
    /// is structural via the type's `PartialEq` so two distinct `Custom`
    /// permutations are correctly rejected, where a string label like
    /// `"custom"` would silently conflate them.
    pub per_worker_layout: KvBlockLayout,
    /// Wire mirror of the leader's per-worker `LayoutConfig`. SPMD
    /// leaders carry identical config across workers (verified by
    /// `validate_remote_metadata`), so a rank-0 sample is sufficient.
    pub per_worker_config: LayoutConfigDescription,
    /// Tensor-parallel size. Operational equality requires identical
    /// `tp_size` because per-worker shape varies with TP fan-out.
    pub tp_size: usize,
    /// Pipeline-parallel size. Operational equality requires identical
    /// `pp_size` for the same reason.
    pub pp_size: usize,
}

impl LayoutCompatPayload {
    /// Reject a payload that is internally inconsistent with its mode.
    ///
    /// Universal mode requires both a derivable canonical aggregate and
    /// a labeled per-worker layout — those are the universal-mode
    /// preconditions documented in
    /// [`BlockLayoutMode::Universal`](kvbm_common::BlockLayoutMode::Universal).
    /// Operational mode tolerates both being absent (legacy / unstamped
    /// leaders fall back to per-worker config comparison).
    ///
    /// This guard is called by [`check_layout_compat`] on both sides
    /// before any cross-payload comparison, and by the hub's
    /// `on_register` gate before storing the *first* payload as the
    /// baseline — without it, a malformed first universal registration
    /// could set an unverifiable baseline that subsequent peers are
    /// silently measured against.
    pub fn validate_self(&self) -> Result<()> {
        if self.mode == BlockLayoutMode::Universal {
            if self.canonical.is_none() {
                bail!(
                    "universal layout_compat payload missing canonical block shape; \
                     universal mode requires global_extents (set ParallelismTemplate \
                     or stamp ParallelismDescriptor on worker metadata)"
                );
            }
            if matches!(self.per_worker_layout, KvBlockLayout::Unknown) {
                bail!(
                    "universal layout_compat payload has KvBlockLayout::Unknown; \
                     universal mode requires every axis to be labeled"
                );
            }
        }
        Ok(())
    }
}

/// Reject `candidate` if it is not compatible with `baseline`.
///
/// - **Cross-mode** (different `BlockLayoutMode`) always rejects.
/// - **Universal × Universal** requires canonical equality only.
/// - **Operational × Operational** requires canonical equality plus
///   identical `per_worker_layout`, `per_worker_config` (every field of
///   [`LayoutConfigDescription`] except `num_blocks`, which is capacity),
///   `tp_size`, and `pp_size`.
///
/// The first divergent field is named in the error message so the
/// operator can act without re-decoding both payloads.
pub fn check_layout_compat(
    baseline: &LayoutCompatPayload,
    candidate: &LayoutCompatPayload,
) -> Result<()> {
    baseline
        .validate_self()
        .map_err(|e| anyhow::anyhow!("baseline {e}"))?;
    candidate
        .validate_self()
        .map_err(|e| anyhow::anyhow!("candidate {e}"))?;
    if baseline.mode != candidate.mode {
        bail!(
            "block_layout mode differs (baseline={}, candidate={}); \
             cross-mode peers are incompatible",
            baseline.mode,
            candidate.mode,
        );
    }
    match (baseline.canonical.as_ref(), candidate.canonical.as_ref()) {
        (Some(b), Some(c)) => b
            .require_equal(c)
            .map_err(|e| anyhow::anyhow!("canonical block shape mismatch: {e}"))?,
        // Universal mode requires canonical on both sides — it has nothing
        // else to compare. Operational tolerates missing canonical as a
        // legacy-stamp path and falls through to the per-worker fields.
        (None, _) | (_, None) if baseline.mode == BlockLayoutMode::Universal => {
            bail!(
                "universal compat: canonical block shape missing \
                 (baseline_has={}, candidate_has={}); universal mode \
                 requires global_extents on both sides",
                baseline.canonical.is_some(),
                candidate.canonical.is_some(),
            );
        }
        _ => {}
    }

    if baseline.mode == BlockLayoutMode::Operational {
        if baseline.per_worker_layout != candidate.per_worker_layout {
            bail!(
                "operational KvBlockLayout differs (baseline={:?}, candidate={:?})",
                baseline.per_worker_layout,
                candidate.per_worker_layout,
            );
        }
        if baseline.tp_size != candidate.tp_size {
            bail!(
                "operational tp_size differs (baseline={}, candidate={})",
                baseline.tp_size,
                candidate.tp_size,
            );
        }
        if baseline.pp_size != candidate.pp_size {
            bail!(
                "operational pp_size differs (baseline={}, candidate={})",
                baseline.pp_size,
                candidate.pp_size,
            );
        }
        require_layout_config_eq(&baseline.per_worker_config, &candidate.per_worker_config)?;
    }
    Ok(())
}

/// Per-field equality for [`LayoutConfigDescription`], ignoring
/// `num_blocks` (capacity, not shape). Names the first divergent field
/// in the error so the operator sees exactly which dimension differs.
fn require_layout_config_eq(
    baseline: &LayoutConfigDescription,
    candidate: &LayoutConfigDescription,
) -> Result<()> {
    macro_rules! check {
        ($field:ident) => {
            if baseline.$field != candidate.$field {
                bail!(
                    "operational per_worker_config {} differs (baseline={:?}, candidate={:?})",
                    stringify!($field),
                    baseline.$field,
                    candidate.$field,
                );
            }
        };
    }
    check!(num_layers);
    check!(outer_dim);
    check!(page_size);
    check!(inner_dim);
    check!(num_heads);
    check!(alignment);
    check!(dtype_width_bytes);
    // num_blocks intentionally not compared — it's capacity, not shape.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical() -> CanonicalBlockShape {
        CanonicalBlockShape {
            num_layers_total: 32,
            outer_dim: 2,
            page_size: 16,
            num_heads_total: 64,
            head_dim: 128,
            dtype_width_bytes: 2,
        }
    }

    fn cfg() -> LayoutConfigDescription {
        LayoutConfigDescription {
            num_blocks: 1024,
            num_layers: 32,
            outer_dim: 2,
            page_size: 16,
            inner_dim: 32 * 128,
            alignment: 256,
            dtype_width_bytes: 2,
            num_heads: Some(32),
        }
    }

    fn payload(mode: BlockLayoutMode) -> LayoutCompatPayload {
        LayoutCompatPayload {
            mode,
            canonical: Some(canonical()),
            per_worker_layout: match mode {
                BlockLayoutMode::Operational => KvBlockLayout::OperationalNHD,
                BlockLayoutMode::Universal => KvBlockLayout::Universal,
            },
            per_worker_config: cfg(),
            tp_size: 2,
            pp_size: 1,
        }
    }

    #[test]
    fn cross_mode_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let b = payload(BlockLayoutMode::Universal);
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("mode"));
    }

    #[test]
    fn universal_identical_accepted() {
        let a = payload(BlockLayoutMode::Universal);
        check_layout_compat(&a, &a).unwrap();
    }

    #[test]
    fn universal_canonical_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Universal);
        let mut b = payload(BlockLayoutMode::Universal);
        b.canonical.as_mut().unwrap().head_dim = 64;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("head_dim"));
    }

    #[test]
    fn universal_accepts_different_per_worker_layout() {
        // Universal-mode policy: aggregate-canonical equality decides
        // compat; per_worker_layout is intentionally not compared.
        // (UniversalPP collapsed into Universal in c1; cross-topology
        // decompositions get their own cases in c4.) Use Custom on one
        // side as the layout-difference probe.
        use kvbm_common::BlockDim;
        let a = payload(BlockLayoutMode::Universal);
        let mut b = payload(BlockLayoutMode::Universal);
        b.per_worker_layout = KvBlockLayout::Custom([
            BlockDim::Head,
            BlockDim::Page,
            BlockDim::Layer,
            BlockDim::Outer,
        ]);
        check_layout_compat(&a, &b).unwrap();
    }

    #[test]
    fn operational_identical_accepted() {
        let a = payload(BlockLayoutMode::Operational);
        check_layout_compat(&a, &a).unwrap();
    }

    #[test]
    fn operational_per_worker_layout_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.per_worker_layout = KvBlockLayout::OperationalHND;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("KvBlockLayout"));
    }

    /// Regression for the second Codex finding: the manager's
    /// first-baseline path stored the payload unconditionally, so a
    /// malformed universal payload (canonical=None) became the
    /// baseline and silently accepted every subsequent peer. The
    /// predicate now runs [`LayoutCompatPayload::validate_self`] on
    /// both sides before any cross-payload comparison.
    #[test]
    fn validate_self_rejects_universal_without_canonical() {
        let mut p = payload(BlockLayoutMode::Universal);
        p.canonical = None;
        let err = p
            .validate_self()
            .expect_err("universal without canonical must fail self-validation");
        assert!(format!("{err}").to_lowercase().contains("canonical"));
    }

    #[test]
    fn validate_self_rejects_universal_with_unknown_layout() {
        let mut p = payload(BlockLayoutMode::Universal);
        p.per_worker_layout = KvBlockLayout::Unknown;
        let err = p
            .validate_self()
            .expect_err("universal with Unknown KvBlockLayout must fail self-validation");
        assert!(format!("{err}").contains("Unknown"));
    }

    #[test]
    fn validate_self_accepts_operational_without_canonical() {
        // Operational has no universal-mode preconditions; legacy /
        // unstamped leaders fall back to per-worker comparison.
        let mut p = payload(BlockLayoutMode::Operational);
        p.canonical = None;
        p.validate_self().unwrap();
    }

    #[test]
    fn check_layout_compat_rejects_malformed_baseline() {
        let mut bad_baseline = payload(BlockLayoutMode::Universal);
        bad_baseline.canonical = None;
        let good_candidate = payload(BlockLayoutMode::Universal);
        let err = check_layout_compat(&bad_baseline, &good_candidate).unwrap_err();
        let m = format!("{err}");
        assert!(
            m.contains("baseline") && m.to_lowercase().contains("canonical"),
            "predicate must reject a malformed baseline with a clear reason (got: {m})"
        );
    }

    #[test]
    fn check_layout_compat_rejects_malformed_candidate() {
        let baseline = payload(BlockLayoutMode::Universal);
        let mut bad_candidate = payload(BlockLayoutMode::Universal);
        bad_candidate.canonical = None;
        let err = check_layout_compat(&baseline, &bad_candidate).unwrap_err();
        let m = format!("{err}");
        assert!(
            m.contains("candidate") && m.to_lowercase().contains("canonical"),
            "predicate must reject a malformed candidate with a clear reason (got: {m})"
        );
    }

    /// Regression for the bug Codex caught at stop-time review:
    /// `KvBlockLayout::name()` flattens every `Custom([..])` variant to
    /// `"custom"`, so a string-typed `per_worker_layout` would silently
    /// accept two leaders with different inner permutations. With
    /// `KvBlockLayout` carried directly the structural `PartialEq`
    /// distinguishes them.
    #[test]
    fn operational_rejects_distinct_custom_permutations() {
        use kvbm_common::BlockDim;
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        let mut c = payload(BlockLayoutMode::Operational);
        b.per_worker_layout = KvBlockLayout::Custom([
            BlockDim::Layer,
            BlockDim::Outer,
            BlockDim::Page,
            BlockDim::Head,
        ]);
        c.per_worker_layout = KvBlockLayout::Custom([
            BlockDim::Outer,
            BlockDim::Layer,
            BlockDim::Page,
            BlockDim::Head,
        ]);
        let err = check_layout_compat(&b, &c)
            .expect_err("two distinct Custom permutations must reject under operational mode");
        assert!(format!("{err}").contains("KvBlockLayout"));
        // Same Custom permutation on both sides must still accept.
        check_layout_compat(&b, &b).unwrap();
        // Custom vs non-Custom must also reject.
        check_layout_compat(&a, &b).unwrap_err();
    }

    #[test]
    fn operational_canonical_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.canonical.as_mut().unwrap().num_heads_total = 48;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("num_heads_total"));
    }

    #[test]
    fn operational_tp_size_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.tp_size = 4;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("tp_size"));
    }

    #[test]
    fn operational_pp_size_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.pp_size = 2;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("pp_size"));
    }

    #[test]
    fn operational_config_inner_dim_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.per_worker_config.inner_dim = 8192;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("inner_dim"));
    }

    #[test]
    fn operational_config_num_layers_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.per_worker_config.num_layers = 28;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("num_layers"));
    }

    #[test]
    fn operational_config_num_blocks_ignored() {
        // num_blocks is capacity, not shape — must not gate compat.
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.per_worker_config.num_blocks = 9999;
        check_layout_compat(&a, &b).unwrap();
    }

    #[test]
    fn operational_config_alignment_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.per_worker_config.alignment = 512;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("alignment"));
    }

    #[test]
    fn operational_config_dtype_width_mismatch_rejected() {
        let a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        b.per_worker_config.dtype_width_bytes = 4;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("dtype_width_bytes"));
    }

    #[test]
    fn operational_tolerates_missing_canonical_on_both_sides() {
        // Legacy / unstamped leaders that can't produce a canonical
        // aggregate still gate on per-worker config + KvBlockLayout.
        let mut a = payload(BlockLayoutMode::Operational);
        let mut b = payload(BlockLayoutMode::Operational);
        a.canonical = None;
        b.canonical = None;
        check_layout_compat(&a, &b).unwrap();
    }

    #[test]
    fn operational_tolerates_one_sided_canonical() {
        // Mixed stamping (one side has ParallelismDescriptor, the other
        // doesn't) is accepted as long as per-worker fields agree.
        let mut a = payload(BlockLayoutMode::Operational);
        let b = payload(BlockLayoutMode::Operational);
        a.canonical = None;
        check_layout_compat(&a, &b).unwrap();
    }

    #[test]
    fn universal_rejects_missing_canonical() {
        let a = payload(BlockLayoutMode::Universal);
        let mut b = payload(BlockLayoutMode::Universal);
        b.canonical = None;
        let err = check_layout_compat(&a, &b).unwrap_err();
        assert!(format!("{err}").contains("canonical"));
    }

    #[test]
    fn payload_serde_round_trip() {
        let p = payload(BlockLayoutMode::Universal);
        let s = serde_json::to_string(&p).unwrap();
        let back: LayoutCompatPayload = serde_json::from_str(&s).unwrap();
        assert_eq!(back, p);
    }
}
