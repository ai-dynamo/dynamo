// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block-layout compatibility mode for cross-leader metadata import.
//!
//! When a KVBM leader imports another leader's metadata (conditional
//! prefill/decode disaggregation, cross-instance onboard, future
//! leader-to-leader exchange), it has to decide whether the remote layout
//! is *compatible* with this leader's local layout. [`BlockLayoutMode`] is
//! the startup-time policy that answers that question.
//!
//! # Modes
//!
//! ## `Operational` — per-worker exact equality
//!
//! Block layout is treated as **opaque bytes in a fixed permutation**.
//! This leader's [`KvBlockLayout`] is whatever the engine probe stamped
//! at startup. On import, every routed local↔remote worker pair must have
//! byte-identical layouts:
//!
//! - Same [`KvBlockLayout`] variant (including `Custom` axis order).
//! - Same `LayoutConfig` for every shape-bearing field: `num_layers`,
//!   `outer_dim`, `page_size`, `inner_dim`, `num_heads`,
//!   `dtype_width_bytes`, `alignment`. `num_blocks` may differ (capacity,
//!   not shape).
//!
//! Any mismatch → reject. The intent: users who commit to a layout at
//! startup refuse silent permutation through the kernel catalog even when
//! the bytes would correctly transform. This is the strict / bit-for-bit
//! mode.
//!
//! ## `Universal` — canonical aggregate equality
//!
//! Block layout is treated as a **labeled canonical tensor**. The per-worker
//! permutation may differ between local and remote leaders; what must
//! match is the *canonical aggregate tensor* — the un-sharded block you
//! would reconstruct by re-collecting every worker's slice along its
//! sharded axis (heads for TP, layers for PP).
//!
//! **Mode-dominant G2 selection (c3).** Under Universal the connector
//! pins G2 (and G3) to [`KvBlockLayout::Universal`] regardless of G1.
//! G1 stays in the engine's operational layout (`OperationalNHD` /
//! `OperationalHND`); G1↔G2 transfers dispatch the fused permute
//! kernel from the kvbm-kernels catalog. Cross-leader transfers all
//! see the canonical Universal layout on the wire — peers' G2's are
//! agreed up front, even when their G1's diverge.
//!
//! ### Startup precondition (Universal only)
//!
//! Every local worker's [`KvBlockLayout`] must be *labeled* — i.e. not
//! [`KvBlockLayout::Unknown`]. `Custom([BlockDim; 4])` is acceptable
//! because every axis is named. Workers that fail this gate are rejected
//! at registration, before any import.
//!
//! ### Import predicate (Universal)
//!
//! 1. Both sides must be labeled. Remote workers whose layout is
//!    `Unknown` are rejected — universal mode cannot reason about an
//!    opaque remote.
//! 2. Build the *canonical block descriptor* from each side. The
//!    descriptor is the tuple
//!
//!    ```text
//!    (num_layers_total, outer_dim, page_size, num_heads_total, head_dim, dtype_width_bytes)
//!    ```
//!
//!    where `num_layers_total` is the un-sharded layer count
//!    (== `ParallelismDescriptor.global_extents[Layer]` == sum of every PP
//!    rank's `layer_ownership.len()`), `num_heads_total` is the un-sharded
//!    head count (== `global_extents[HeadCount]` == sum of every TP rank's
//!    per-worker `num_heads`), and `head_dim = inner_dim / num_heads`.
//! 3. Local and remote canonical descriptors must be equal in every
//!    component. If they differ, reject with the diverging field named
//!    explicitly (`canonical num_heads differs (local=64, remote=48)`).
//! 4. Per-worker [`KvBlockLayout`] permutation may differ between local
//!    and remote — the always-available permute kernels can re-pack on
//!    the fly during transfer.
//! 5. Per-worker extents along the sharded axis may differ (e.g. local
//!    TP=4 with `num_heads=16` per worker importing from remote TP=8 with
//!    `num_heads=8` per worker is fine as long as `num_heads_total=64` on
//!    both sides). The existing many-to-one / one-to-many routing in
//!    `route_local_to_remote` already handles the cross-TP cases.
//!
//! The intent: users willing to absorb the cost of permutation kernels
//! and TP/PP re-routing in exchange for accepting any remote leader whose
//! un-sharded canonical tensor matches.
//!
//! # Where enforcement lives
//!
//! Two gates apply this mode, with different visibility and different
//! coverage:
//!
//! - **Hub gate (cross-mode, CD-opt-in only).** At
//!   `POST /v1/instances`, the first ConditionalDisagg instance that
//!   sends a `LayoutCompatPayload` defines the baseline (mode +
//!   canonical shape + per-worker layout); subsequent registrations
//!   whose payload doesn't match are rejected with `400 Bad Request`
//!   before the leader becomes discoverable. This is the **only**
//!   gate that catches cross-mode, because only the hub sees both
//!   leaders' *declared* `BlockLayoutMode` (each leader stamps its
//!   own inside the payload; the wire `SerializedLayout` does not
//!   carry mode).
//!
//!   Coverage (c2): every `Feature::P2P` registration is gated.
//!   `Feature::ConditionalDisagg` requires `Feature::P2P` to also be
//!   present in the same register request (CD is a specialisation of
//!   P2P), so all CD instances pass through the gate. Standalone
//!   leaders that don't register `Feature::P2P` (observer-only,
//!   future use cases) don't get the hub-level gate; session-layer
//!   re-validation for those paths is tracked separately.
//!
//! - **Engine gate (same-mode shape check, every connect path).** At
//!   `connect_remote` the local leader runs
//!   [`crate::shape::CanonicalBlockShape`]-based equality against the
//!   remote worker's metadata using *its own* startup mode. This gate
//!   fires for **every** import path — CD or not, with or without a
//!   hub gate having run first — but it cannot independently catch a
//!   cross-mode peer because the wire metadata is mode-less. When
//!   the hub gate has run (CD with `layout_compat`), the engine gate
//!   is redundant defence-in-depth; when it hasn't (everything
//!   else), the engine gate is the only shape-equality check, and a
//!   cross-mode peer in those paths is undetectable by today's wire
//!   protocol.
//!
//! Acceptance matrix at the hub:
//!
//! - operational + operational + identical per-worker layouts → accept.
//! - universal   + universal   + matching canonicals          → accept.
//! - operational + universal   + anything                     → **reject**
//!   (cross-mode peers are incompatible by policy even when their bytes
//!   happen to agree, because the two modes carry different downstream
//!   contracts — operational refuses silent permutation through the
//!   kernel catalog; universal opts into it).
//! - operational + operational + non-identical layouts        → reject.
//! - universal   + universal   + non-matching canonicals      → reject.
//!
//! The baseline is cleared when both the prefill and decode sets on
//! the hub become empty, so an intentional reconfig (full teardown +
//! relaunch) can adopt a new mode without bouncing the hub.
//!
//! [`KvBlockLayout`]: crate::block::KvBlockLayout
//! [`KvBlockLayout::Unknown`]: crate::block::KvBlockLayout::Unknown

use serde::{Deserialize, Serialize};

/// Block-layout compatibility mode applied at cross-leader metadata import.
///
/// See module docs for full semantics. Selected at leader startup
/// (`KvbmConfig.block_layout` / env `KVBM_BLOCK_LAYOUT`) and immutable for
/// the leader's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockLayoutMode {
    /// Per-worker `(KvBlockLayout, LayoutConfig)` must match exactly on
    /// import. Strict / bit-for-bit. **Default.**
    #[default]
    Operational,

    /// Canonical aggregate tensor must match; per-worker permutation and
    /// shard extents may differ. Requires every local `KvBlockLayout` to
    /// be labeled (not `Unknown`) at startup.
    Universal,
}

impl BlockLayoutMode {
    /// Stable lowercase identifier suitable for logs / metrics.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Operational => "operational",
            Self::Universal => "universal",
        }
    }
}

impl std::fmt::Display for BlockLayoutMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_operational() {
        assert_eq!(BlockLayoutMode::default(), BlockLayoutMode::Operational);
    }

    #[test]
    fn serde_uses_snake_case() {
        let s = serde_json::to_string(&BlockLayoutMode::Operational).unwrap();
        assert_eq!(s, "\"operational\"");
        let u = serde_json::to_string(&BlockLayoutMode::Universal).unwrap();
        assert_eq!(u, "\"universal\"");

        let parsed: BlockLayoutMode = serde_json::from_str("\"universal\"").unwrap();
        assert_eq!(parsed, BlockLayoutMode::Universal);
    }

    #[test]
    fn label_matches_serde() {
        assert_eq!(BlockLayoutMode::Operational.as_label(), "operational");
        assert_eq!(BlockLayoutMode::Universal.as_label(), "universal");
    }
}
