// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-side cross-parallelism pull planner.
//!
//! [`plan_pull`] turns a `(local ParallelismTemplate, remote
//! [ParallelismDescriptor], refs)` triple into a list of per-local-rank
//! [`WorkerPullPlan`]s. Each plan tells one local worker which remote
//! ranks it must pull from, with a coordinate-space slice on each side
//! describing what portion of every block to transfer.
//!
//! The planner is pure CPU: no GPU, no NIXL, no network. Workers are
//! "dumb" — they receive a fully resolved [`WorkerPullPlan`] and execute
//! it via [`kvbm_physical::layout::LayoutView::slice`] +
//! [`kvbm_physical::manager::TransferManager::execute_transfer_selection`].
//!
//! ## Cross-TP arithmetic
//!
//! All sides use a single shard axis (typically [`KvDim::HeadCount`]).
//! The peer-side compatibility gate ([`crate::leader::parallelism::validate_remote_metadata`])
//! has already enforced:
//!
//! - `tp_local | tp_remote` or `tp_remote | tp_local` (divisibility),
//! - shard axes agree,
//! - `global_extents` agree on every axis including the shard axis,
//! - `pp_size == 1` on both sides.
//!
//! Let `G` be the global shard-axis extent and let `L = tp_local`,
//! `R = tp_remote`.
//!
//! - **Symmetric (`L == R`)**: local rank `i` pulls from remote rank
//!   `i` with no shard-axis slicing on either side.
//! - **Local smaller (`L < R`, with `N = R / L`)**: local rank `i` pulls
//!   from remote ranks `i*N .. (i+1)*N`. Each remote rank contributes
//!   its full per-rank extent (`G/R` heads); they land at successive
//!   offsets `0*G/R, 1*G/R, ...` within local rank `i`'s `G/L`-sized
//!   HeadCount tensor. `N` shards per local plan.
//! - **Local larger (`L > R`, with `N = L / R`)**: local rank `i` pulls
//!   from remote rank `i / N`. Local owns `G/L` heads; this is a
//!   sub-slice `[(i mod N) * G/L, (i mod N + 1) * G/L)` of the remote
//!   rank's `G/R`-sized HeadCount tensor. The local side has no
//!   shard-axis restriction (the full local extent is filled). Exactly
//!   one shard per local plan.
//!
//! PP is reserved (`pp_size == 1`). When PP lands, the planner will
//! consume `layer_ownership` and add a `Layer`-axis entry to
//! [`LayoutSlice::axes`] for each shard whose remote rank only owns a
//! sub-range of layers. The wire format already reserves this room.

use std::collections::HashSet;
use std::ops::Range;

use anyhow::{Result, bail};
use kvbm_common::{AxisIntersection, BlockId, KvDim, KvbmTransferRoute, LogicalLayoutHandle};
use kvbm_physical::manager::ParallelismDescriptor;
use serde::{Deserialize, Serialize};
use velo::InstanceId;

use crate::leader::parallelism::ParallelismTemplate;

/// A single block pair to pull: `src_block_id` from the remote leader's
/// per-leader block-id space, `dst_block_id` into the local leader's.
///
/// The same `src_block_id` is meaningful on every remote rank under
/// `remote_instance` — the remote leader assigns block-ids uniformly
/// across its workers. Likewise `dst_block_id` is meaningful on every
/// local rank. The planner therefore copies the same `(src_block_ids,
/// dst_block_ids)` vectors onto every [`WorkerPullPlan`] it emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullRef {
    pub src_block_id: BlockId,
    pub dst_block_id: BlockId,
}

/// Coordinate-space slice over a block tensor's local axes.
///
/// Empty `axes` means "full block" (every axis covers its full local
/// extent). A `(dim, start..end)` entry restricts that dim to the
/// half-open range `[start, end)` of the *local* (post-shard) coord
/// space on the side this slice applies to.
///
/// Today only the shard axis (typically [`KvDim::HeadCount`]) is ever
/// restricted. Layer slicing is reserved for the first PP PR.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutSlice {
    pub axes: Vec<(KvDim, Range<usize>)>,
}

impl LayoutSlice {
    /// Empty slice — full block on every axis.
    pub fn full() -> Self {
        Self::default()
    }

    /// `true` when no axis is restricted.
    pub fn is_full(&self) -> bool {
        self.axes.is_empty()
    }

    /// Convenience constructor: restrict a single axis.
    pub fn on_axis(dim: KvDim, range: Range<usize>) -> Self {
        Self {
            axes: vec![(dim, range)],
        }
    }
}

/// One shard within a [`WorkerPullPlan`]: pull `local_slice` of every
/// `dst_block_id` from `remote_slice` of every `src_block_id` on the
/// `remote_rank`-th worker of `remote_instance`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullShard {
    pub remote_rank: usize,
    /// Coordinate-space restriction applied on the local destination side.
    pub local_slice: LayoutSlice,
    /// Coordinate-space restriction applied on the remote source side.
    pub remote_slice: LayoutSlice,
}

/// Strict subset of [`kvbm_physical::transfer::TransferOptions`]
/// permitted over the wire for cross-parallelism pulls.
///
/// Excluded by design:
///
/// - `use_planner` — per-side optimisation toggle, decided locally.
/// - `layer_range` — incompatible with the planner today; the first PP
///   PR will reintroduce a layer-range field and reconcile with
///   `transfer/executor/mod.rs`'s `use_planner + layer_range`
///   rejection.
/// - `bounce_buffer`, `device_stream`, `src_kv_layout`, `dst_kv_layout` —
///   per-allocation references that cannot be serialised meaningfully.
///
/// Every field is `#[serde(default)]` so a newer sender can add fields
/// to the wire shape without breaking older receivers — the receiver
/// silently fills the missing field with `Default::default()`. The
/// inverse (receiver expects a field the sender doesn't know about) is
/// not a concern because all current fields are `Option`-shaped and
/// default to `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WirePullOptions {
    #[serde(default)]
    pub nixl_write_notification: Option<u64>,
    #[serde(default)]
    pub metric_route: Option<KvbmTransferRoute>,
}

/// Fully-resolved pull plan for one local worker.
///
/// Shared across all `shards`: `remote_instance`, `source_layout`,
/// `dst_layout`, `src_block_ids`, `dst_block_ids`, `options`. The
/// `shards` vec carries the per-remote-rank slicing.
///
/// `options`, `shards`, and the block-id vectors are tagged
/// `#[serde(default)]` for forward compatibility: a sender that omits
/// any of these (because a future schema split them off, or a default
/// is acceptable) decodes cleanly on the receiver. The core targeting
/// fields (`remote_instance`, `source_layout`, `dst_layout`) are
/// required; missing any of them is a genuine wire-shape error and
/// should fail the decode loudly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerPullPlan {
    pub remote_instance: InstanceId,
    pub source_layout: LogicalLayoutHandle,
    pub dst_layout: LogicalLayoutHandle,
    #[serde(default)]
    pub src_block_ids: Vec<BlockId>,
    #[serde(default)]
    pub dst_block_ids: Vec<BlockId>,
    #[serde(default)]
    pub shards: Vec<PullShard>,
    #[serde(default)]
    pub options: WirePullOptions,
}

/// Map a cross-parallelism pull onto per-local-rank worker plans.
///
/// Returns `Vec<(local_rank, WorkerPullPlan)>` sorted by `local_rank`.
/// Local ranks with no shards are omitted entirely (no RPC bytes
/// burned on no-op plans).
///
/// `remote` must be the full set of per-rank descriptors from one peer
/// leader (length `tp_remote * pp_remote`), already validated by
/// [`crate::leader::parallelism::validate_remote_metadata`]. The
/// planner re-checks the few invariants its arithmetic depends on —
/// divisibility, consistent remote tp_size, presence of the shard axis
/// in `global_extents` — but does not duplicate the full compat suite.
///
/// Caller invariants:
///
/// - `local.shard_axis == remote[0].shard_axis` (caller-enforced via
///   compat gate; the planner panics in debug builds if violated).
/// - `local.global_extents` lists the shard axis with the same value
///   as `remote[0].global_extents`.
/// - `local.pp_size == 1` and `remote[*].pp_size == 1`.
pub fn plan_pull(
    local: &ParallelismTemplate,
    remote: &[ParallelismDescriptor],
    remote_instance: InstanceId,
    source_layout: LogicalLayoutHandle,
    dst_layout: LogicalLayoutHandle,
    refs: &[PullRef],
    opts: &WirePullOptions,
) -> Result<Vec<(usize, WorkerPullPlan)>> {
    if local.tp_size == 0 {
        bail!("plan_pull: local.tp_size must be > 0");
    }
    if local.pp_size != 1 {
        bail!(
            "plan_pull: local.pp_size = {} is not supported (PP is a non-goal)",
            local.pp_size
        );
    }
    if remote.is_empty() {
        bail!("plan_pull: remote descriptor vec is empty");
    }
    let head = &remote[0];
    if head.tp_size == 0 {
        bail!("plan_pull: remote.tp_size must be > 0");
    }
    if head.pp_size != 1 {
        bail!(
            "plan_pull: remote.pp_size = {} is not supported (PP is a non-goal)",
            head.pp_size
        );
    }
    // Internal consistency: all remote descriptors agree on tp_size.
    // The full compat gate (validate_remote_metadata) enforces shard
    // axis / global_extents agreement too; we trust it for those but
    // re-check tp_size since divisibility hinges on it.
    for (i, d) in remote.iter().enumerate().skip(1) {
        if d.tp_size != head.tp_size {
            bail!(
                "plan_pull: remote rank {i} reports tp_size={} but rank 0 reports {}",
                d.tp_size,
                head.tp_size
            );
        }
    }
    debug_assert_eq!(
        local.shard_axis, head.shard_axis,
        "plan_pull: shard axis mismatch — caller must run validate_remote_metadata first",
    );

    let local_tp = local.tp_size;
    let remote_tp = head.tp_size;
    if !local_tp.is_multiple_of(remote_tp) && !remote_tp.is_multiple_of(local_tp) {
        bail!(
            "plan_pull: coprime tp sizes (local={local_tp}, remote={remote_tp}); \
             one must divide the other",
        );
    }

    if refs.is_empty() {
        // No work — every local plan would be empty.
        return Ok(Vec::new());
    }

    // Global shard-axis extent. We could equally read it from `local`;
    // both have been validated to match.
    let shard_axis = local.shard_axis;
    let global_shard = extent_for(&head.global_extents, shard_axis).ok_or_else(|| {
        anyhow::anyhow!("plan_pull: shard axis {shard_axis:?} missing from remote global_extents")
    })?;
    let local_extent = global_shard
        .checked_div(local_tp)
        .ok_or_else(|| anyhow::anyhow!("plan_pull: local_tp == 0"))?;
    let remote_extent = global_shard
        .checked_div(remote_tp)
        .ok_or_else(|| anyhow::anyhow!("plan_pull: remote_tp == 0"))?;
    if local_extent * local_tp != global_shard {
        bail!(
            "plan_pull: global shard extent {global_shard} not divisible by local tp_size {local_tp}",
        );
    }
    if remote_extent * remote_tp != global_shard {
        bail!(
            "plan_pull: global shard extent {global_shard} not divisible by remote tp_size {remote_tp}",
        );
    }

    let (src_block_ids, dst_block_ids): (Vec<BlockId>, Vec<BlockId>) = refs
        .iter()
        .map(|r| (r.src_block_id, r.dst_block_id))
        .unzip();

    // Note: the arithmetic below addresses remote ranks by *value*, not
    // by index into the `remote` slice. We deliberately do not consult
    // `descriptor.rank` after the consistency checks above — the
    // descriptor vec is only used to extract shape (tp_size, shard_axis,
    // global_extents) which `validate_remote_metadata` proved uniform
    // across ranks. The set of valid rank values is `0..remote_tp`,
    // covered by gate 3.5 in the compat suite.
    let mut out: Vec<(usize, WorkerPullPlan)> = Vec::with_capacity(local_tp);
    for local_rank in 0..local_tp {
        let shards = compute_shards(
            local_rank,
            local_tp,
            remote_tp,
            shard_axis,
            local_extent,
            remote_extent,
        )?;
        if shards.is_empty() {
            continue;
        }
        out.push((
            local_rank,
            WorkerPullPlan {
                remote_instance,
                source_layout,
                dst_layout,
                src_block_ids: src_block_ids.clone(),
                dst_block_ids: dst_block_ids.clone(),
                shards,
                options: opts.clone(),
            },
        ));
    }
    Ok(out)
}

/// Compute the shard list for one local rank under the divisibility
/// constraint. Pure arithmetic — no allocations beyond the returned
/// `Vec`.
fn compute_shards(
    local_rank: usize,
    local_tp: usize,
    remote_tp: usize,
    shard_axis: KvDim,
    local_extent: usize,
    remote_extent: usize,
) -> Result<Vec<PullShard>> {
    if local_tp == remote_tp {
        // Symmetric. Local rank pulls from the identically-numbered
        // remote rank with full extents on both sides.
        Ok(vec![PullShard {
            remote_rank: local_rank,
            local_slice: LayoutSlice::full(),
            remote_slice: LayoutSlice::full(),
        }])
    } else if local_tp < remote_tp {
        // Local rank `i` pulls from remote ranks [i*N, (i+1)*N) where
        // N = remote_tp / local_tp. Each remote rank contributes its
        // full per-rank tensor; remote-rank-k-th-of-N (k=0..N) lands at
        // local offset `k * remote_extent`.
        let n = remote_tp / local_tp;
        let mut shards = Vec::with_capacity(n);
        for k in 0..n {
            let remote_rank = local_rank * n + k;
            let local_start = k * remote_extent;
            let local_end = local_start + remote_extent;
            shards.push(PullShard {
                remote_rank,
                local_slice: LayoutSlice::on_axis(shard_axis, local_start..local_end),
                remote_slice: LayoutSlice::full(),
            });
        }
        Ok(shards)
    } else {
        // local_tp > remote_tp. N = local_tp / remote_tp. Local rank
        // `i` pulls from remote rank `i / N`. Local owns `local_extent`
        // heads which is `(i mod N)` slot of size local_extent within
        // remote rank's larger tensor.
        let n = local_tp / remote_tp;
        let remote_rank = local_rank / n;
        let slot = local_rank % n;
        let remote_start = slot * local_extent;
        let remote_end = remote_start + local_extent;
        Ok(vec![PullShard {
            remote_rank,
            local_slice: LayoutSlice::full(),
            remote_slice: LayoutSlice::on_axis(shard_axis, remote_start..remote_end),
        }])
    }
}

fn extent_for(extents: &[(KvDim, usize)], axis: KvDim) -> Option<usize> {
    extents.iter().find(|(d, _)| *d == axis).map(|(_, v)| *v)
}

/// Translate a planner-emitted [`PullShard`] into the per-axis
/// [`AxisIntersection`]s that [`kvbm_physical::transfer::TransferSelection`]
/// expects.
///
/// The translation is determined entirely by the two extent lookups —
/// the planner's [`LayoutSlice`] only carries the *restricted-side*
/// range, and we fill the unrestricted side's range from the
/// corresponding per-worker layout extent. Closures are taken (rather
/// than concrete maps) so unit tests can stub them without
/// materialising real [`kvbm_physical::layout::LayoutView`]s.
///
/// The planner emits at most one of `local_slice` / `remote_slice` per
/// axis, with `len == the other side's per-worker extent` on that
/// axis. If the planner ever violates that invariant this function
/// surfaces it as a hard error rather than silently mis-routing a
/// partial transfer.
pub fn build_axis_intersections<L, R>(
    shard: &PullShard,
    local_extent_of: L,
    remote_extent_of: R,
) -> Result<Vec<AxisIntersection>>
where
    L: Fn(KvDim) -> Option<usize>,
    R: Fn(KvDim) -> Option<usize>,
{
    let mut seen: HashSet<KvDim> = HashSet::new();
    let mut out: Vec<AxisIntersection> = Vec::new();

    for (dim, dst_range) in &shard.local_slice.axes {
        let remote = remote_extent_of(*dim).ok_or_else(|| {
            anyhow::anyhow!(
                "build_axis_intersections: axis {dim:?} missing from remote per-worker layout"
            )
        })?;
        if dst_range.len() != remote {
            bail!(
                "build_axis_intersections: PullShard.local_slice on {dim:?} has length {} \
                 but remote per-worker extent is {} — planner emitted invalid pair",
                dst_range.len(),
                remote,
            );
        }
        out.push(AxisIntersection {
            dim: *dim,
            src_local: 0..remote,
            dst_local: dst_range.clone(),
        });
        seen.insert(*dim);
    }
    for (dim, src_range) in &shard.remote_slice.axes {
        if seen.contains(dim) {
            bail!(
                "build_axis_intersections: PullShard restricts axis {dim:?} on both sides — \
                 planner output is malformed",
            );
        }
        let local = local_extent_of(*dim).ok_or_else(|| {
            anyhow::anyhow!(
                "build_axis_intersections: axis {dim:?} missing from local per-worker layout"
            )
        })?;
        if src_range.len() != local {
            bail!(
                "build_axis_intersections: PullShard.remote_slice on {dim:?} has length {} \
                 but local per-worker extent is {} — planner emitted invalid pair",
                src_range.len(),
                local,
            );
        }
        out.push(AxisIntersection {
            dim: *dim,
            src_local: src_range.clone(),
            dst_local: 0..local,
        });
    }
    Ok(out)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use kvbm_config::ParallelismMode;

    const GLOBAL_HEADS: usize = 32;

    fn make_local(tp_size: usize) -> ParallelismTemplate {
        ParallelismTemplate {
            tp_size,
            pp_size: 1,
            parallelism_mode: ParallelismMode::TensorParallel,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![
                (KvDim::Layer, 24),
                (KvDim::Page, 8),
                (KvDim::HeadCount, GLOBAL_HEADS),
                (KvDim::HeadSize, 64),
            ],
            num_layers: 24,
            dtype_width_bytes: 2,
        }
    }

    fn make_remote(tp_size: usize) -> Vec<ParallelismDescriptor> {
        (0..tp_size)
            .map(|rank| ParallelismDescriptor {
                tp_size,
                pp_size: 1,
                rank,
                shard_axis: KvDim::HeadCount,
                global_extents: vec![
                    (KvDim::Layer, 24),
                    (KvDim::Page, 8),
                    (KvDim::HeadCount, GLOBAL_HEADS),
                    (KvDim::HeadSize, 64),
                ],
                layer_ownership: 0..24,
            })
            .collect()
    }

    fn refs(n: usize) -> Vec<PullRef> {
        (0..n)
            .map(|i| PullRef {
                src_block_id: 100 + i,
                dst_block_id: 200 + i,
            })
            .collect()
    }

    fn opts() -> WirePullOptions {
        WirePullOptions::default()
    }

    fn instance() -> InstanceId {
        InstanceId::new_v4()
    }

    fn handle() -> LogicalLayoutHandle {
        LogicalLayoutHandle::G2
    }

    #[test]
    fn symmetric_tp_one_shard_per_rank_full_extents() {
        let local = make_local(4);
        let remote = make_remote(4);
        let plans = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(3),
            &opts(),
        )
        .unwrap();
        assert_eq!(plans.len(), 4, "every local rank produces a plan");
        for (rank, plan) in &plans {
            assert_eq!(plan.shards.len(), 1, "symmetric → one shard per local rank");
            assert_eq!(plan.shards[0].remote_rank, *rank);
            assert!(plan.shards[0].local_slice.is_full());
            assert!(plan.shards[0].remote_slice.is_full());
            assert_eq!(plan.src_block_ids, vec![100, 101, 102]);
            assert_eq!(plan.dst_block_ids, vec![200, 201, 202]);
        }
    }

    #[test]
    fn asymmetric_local_smaller_pulls_from_n_remotes() {
        // local TP=2, remote TP=4. N = 2. Each local rank touches 2
        // remote ranks, each contributing its full per-rank extent of
        // 8 heads, landing at local offsets 0..8 and 8..16 of the
        // local rank's 16-head HeadCount tensor.
        let local = make_local(2);
        let remote = make_remote(4);
        let plans = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap();
        assert_eq!(plans.len(), 2);

        let (lr0, plan0) = &plans[0];
        assert_eq!(*lr0, 0);
        assert_eq!(plan0.shards.len(), 2);
        assert_eq!(plan0.shards[0].remote_rank, 0);
        assert_eq!(plan0.shards[0].remote_slice, LayoutSlice::full());
        assert_eq!(
            plan0.shards[0].local_slice,
            LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
        );
        assert_eq!(plan0.shards[1].remote_rank, 1);
        assert_eq!(
            plan0.shards[1].local_slice,
            LayoutSlice::on_axis(KvDim::HeadCount, 8..16),
        );

        let (lr1, plan1) = &plans[1];
        assert_eq!(*lr1, 1);
        assert_eq!(plan1.shards[0].remote_rank, 2);
        assert_eq!(plan1.shards[1].remote_rank, 3);
        assert_eq!(
            plan1.shards[0].local_slice,
            LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
        );
        assert_eq!(
            plan1.shards[1].local_slice,
            LayoutSlice::on_axis(KvDim::HeadCount, 8..16),
        );
    }

    #[test]
    fn asymmetric_local_larger_one_shard_per_rank_remote_sliced() {
        // local TP=4, remote TP=2. N = 2. Local rank 0,1 → remote rank
        // 0; local rank 2,3 → remote rank 1. Each pulls an 8-head
        // sub-slice of the remote rank's 16-head HeadCount tensor.
        let local = make_local(4);
        let remote = make_remote(2);
        let plans = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(2),
            &opts(),
        )
        .unwrap();
        assert_eq!(plans.len(), 4);

        let expected = [
            (0usize, 0usize, 0..8),
            (1, 0, 8..16),
            (2, 1, 0..8),
            (3, 1, 8..16),
        ];
        for ((lr, plan), (exp_lr, exp_rr, exp_remote)) in plans.iter().zip(expected.iter()) {
            assert_eq!(*lr, *exp_lr);
            assert_eq!(plan.shards.len(), 1);
            assert_eq!(plan.shards[0].remote_rank, *exp_rr);
            assert!(plan.shards[0].local_slice.is_full());
            assert_eq!(
                plan.shards[0].remote_slice,
                LayoutSlice::on_axis(KvDim::HeadCount, exp_remote.clone()),
            );
        }
    }

    #[test]
    fn single_remote_worker_pulls_quartered_slice() {
        // local TP=4, remote TP=1. Every local rank pulls from remote
        // rank 0, slicing the remote's full 32-head HeadCount tensor
        // into quarters.
        let local = make_local(4);
        let remote = make_remote(1);
        let plans = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap();
        assert_eq!(plans.len(), 4);
        for (lr, plan) in &plans {
            assert_eq!(plan.shards.len(), 1);
            assert_eq!(plan.shards[0].remote_rank, 0);
            assert!(plan.shards[0].local_slice.is_full());
            let start = lr * 8;
            assert_eq!(
                plan.shards[0].remote_slice,
                LayoutSlice::on_axis(KvDim::HeadCount, start..start + 8),
            );
        }
    }

    #[test]
    fn single_local_worker_pulls_from_every_remote() {
        // local TP=1, remote TP=4. The single local rank touches all
        // 4 remote ranks, each contributing its full per-rank extent.
        let local = make_local(1);
        let remote = make_remote(4);
        let plans = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap();
        assert_eq!(plans.len(), 1);
        let (lr, plan) = &plans[0];
        assert_eq!(*lr, 0);
        assert_eq!(plan.shards.len(), 4);
        for (k, shard) in plan.shards.iter().enumerate() {
            assert_eq!(shard.remote_rank, k);
            assert!(shard.remote_slice.is_full());
            assert_eq!(
                shard.local_slice,
                LayoutSlice::on_axis(KvDim::HeadCount, k * 8..(k + 1) * 8),
            );
        }
    }

    #[test]
    fn empty_refs_returns_no_plans() {
        let local = make_local(2);
        let remote = make_remote(2);
        let plans = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &[],
            &opts(),
        )
        .unwrap();
        assert!(plans.is_empty());
    }

    #[test]
    fn block_ids_and_options_replicated_onto_every_plan() {
        let local = make_local(2);
        let remote = make_remote(4);
        let custom_opts = WirePullOptions {
            nixl_write_notification: Some(0xCAFE),
            metric_route: None,
        };
        let inst = instance();
        let plans = plan_pull(
            &local,
            &remote,
            inst,
            handle(),
            handle(),
            &refs(3),
            &custom_opts,
        )
        .unwrap();
        for (_, plan) in &plans {
            assert_eq!(plan.remote_instance, inst);
            assert_eq!(plan.source_layout, handle());
            assert_eq!(plan.dst_layout, handle());
            assert_eq!(plan.src_block_ids, vec![100, 101, 102]);
            assert_eq!(plan.dst_block_ids, vec![200, 201, 202]);
            assert_eq!(plan.options, custom_opts);
        }
    }

    #[test]
    fn rejects_coprime_tp() {
        let local = make_local(3);
        let remote = make_remote(2);
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("coprime"), "msg: {err}");
    }

    #[test]
    fn rejects_zero_local_tp() {
        let local = make_local(0);
        let remote = make_remote(2);
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("local.tp_size"), "msg: {err}");
    }

    #[test]
    fn rejects_empty_remote() {
        let local = make_local(2);
        let err = plan_pull(
            &local,
            &[],
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("remote descriptor vec"),
            "msg: {err}"
        );
    }

    #[test]
    fn rejects_pp_local() {
        let mut local = make_local(2);
        local.pp_size = 2;
        let remote = make_remote(2);
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("pp_size"), "msg: {err}");
    }

    #[test]
    fn rejects_pp_remote() {
        let local = make_local(2);
        let mut remote = make_remote(2);
        for d in &mut remote {
            d.pp_size = 2;
        }
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("pp_size"), "msg: {err}");
    }

    #[test]
    fn rejects_inconsistent_remote_tp() {
        let local = make_local(2);
        let mut remote = make_remote(2);
        remote[1].tp_size = 4;
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("rank 1 reports tp_size"),
            "msg: {err}"
        );
    }

    #[test]
    fn rejects_missing_shard_axis_extent() {
        let local = make_local(2);
        let mut remote = make_remote(2);
        for d in &mut remote {
            d.global_extents.retain(|(a, _)| *a != KvDim::HeadCount);
        }
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("shard axis"), "msg: {err}");
    }

    // -- build_axis_intersections tests --

    fn local_extent(map: &[(KvDim, usize)]) -> impl Fn(KvDim) -> Option<usize> + '_ {
        |d| map.iter().find(|(k, _)| *k == d).map(|(_, v)| *v)
    }

    #[test]
    fn build_axis_intersections_local_smaller_pattern() {
        // Local TP=2 (HeadCount per-worker 16) pulling from remote TP=4
        // (per-worker 8). Shard touches remote rank 0; local_slice
        // lands at the front 8 heads of local rank 0's 16-head tensor.
        let shard = PullShard {
            remote_rank: 0,
            local_slice: LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
            remote_slice: LayoutSlice::full(),
        };
        let local = vec![(KvDim::HeadCount, 16)];
        let remote = vec![(KvDim::HeadCount, 8)];
        let inters =
            build_axis_intersections(&shard, local_extent(&local), local_extent(&remote)).unwrap();
        assert_eq!(inters.len(), 1);
        assert_eq!(inters[0].dim, KvDim::HeadCount);
        assert_eq!(inters[0].src_local, 0..8);
        assert_eq!(inters[0].dst_local, 0..8);
    }

    #[test]
    fn build_axis_intersections_local_larger_pattern() {
        // Local TP=4 (per-worker 8) pulling from remote TP=2 (per-worker
        // 16). Local rank 1 takes remote rank 0's heads [8, 16) into
        // its full 0..8 local tensor.
        let shard = PullShard {
            remote_rank: 0,
            local_slice: LayoutSlice::full(),
            remote_slice: LayoutSlice::on_axis(KvDim::HeadCount, 8..16),
        };
        let local = vec![(KvDim::HeadCount, 8)];
        let remote = vec![(KvDim::HeadCount, 16)];
        let inters =
            build_axis_intersections(&shard, local_extent(&local), local_extent(&remote)).unwrap();
        assert_eq!(inters.len(), 1);
        assert_eq!(inters[0].dim, KvDim::HeadCount);
        assert_eq!(inters[0].src_local, 8..16);
        assert_eq!(inters[0].dst_local, 0..8);
    }

    #[test]
    fn build_axis_intersections_symmetric_pattern_empty() {
        // Symmetric TP: both sides full; no axis_slice needed. The
        // helper must emit an empty Vec (TransferSelection treats
        // that as full-extent).
        let shard = PullShard {
            remote_rank: 3,
            local_slice: LayoutSlice::full(),
            remote_slice: LayoutSlice::full(),
        };
        let local = vec![(KvDim::HeadCount, 8)];
        let remote = vec![(KvDim::HeadCount, 8)];
        let inters =
            build_axis_intersections(&shard, local_extent(&local), local_extent(&remote)).unwrap();
        assert!(inters.is_empty());
    }

    #[test]
    fn build_axis_intersections_rejects_local_slice_remote_extent_mismatch() {
        // Planner mistake: emits local_slice 0..8 but remote per-worker
        // HeadCount is 7 — the unrestricted side's full extent disagrees
        // with the slice length, so the resulting intersection lengths
        // would be (7, 8). Helper must bail.
        let shard = PullShard {
            remote_rank: 0,
            local_slice: LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
            remote_slice: LayoutSlice::full(),
        };
        let local = vec![(KvDim::HeadCount, 16)];
        let remote = vec![(KvDim::HeadCount, 7)];
        let err = build_axis_intersections(&shard, local_extent(&local), local_extent(&remote))
            .unwrap_err();
        assert!(
            err.to_string().contains("local_slice on HeadCount"),
            "msg: {err}"
        );
    }

    #[test]
    fn build_axis_intersections_rejects_remote_slice_local_extent_mismatch() {
        let shard = PullShard {
            remote_rank: 0,
            local_slice: LayoutSlice::full(),
            remote_slice: LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
        };
        let local = vec![(KvDim::HeadCount, 7)];
        let remote = vec![(KvDim::HeadCount, 16)];
        let err = build_axis_intersections(&shard, local_extent(&local), local_extent(&remote))
            .unwrap_err();
        assert!(
            err.to_string().contains("remote_slice on HeadCount"),
            "msg: {err}"
        );
    }

    #[test]
    fn build_axis_intersections_rejects_double_restriction() {
        let shard = PullShard {
            remote_rank: 0,
            local_slice: LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
            remote_slice: LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
        };
        let local = vec![(KvDim::HeadCount, 8)];
        let remote = vec![(KvDim::HeadCount, 8)];
        let err = build_axis_intersections(&shard, local_extent(&local), local_extent(&remote))
            .unwrap_err();
        assert!(err.to_string().contains("both sides"), "msg: {err}");
    }

    #[test]
    fn build_axis_intersections_rejects_missing_axis_in_remote_layout() {
        let shard = PullShard {
            remote_rank: 0,
            local_slice: LayoutSlice::on_axis(KvDim::HeadCount, 0..8),
            remote_slice: LayoutSlice::full(),
        };
        let local = vec![(KvDim::HeadCount, 16)];
        let remote: Vec<(KvDim, usize)> = vec![]; // remote layout has no HeadCount
        let err = build_axis_intersections(&shard, local_extent(&local), local_extent(&remote))
            .unwrap_err();
        assert!(
            err.to_string().contains("missing from remote"),
            "msg: {err}"
        );
    }

    #[test]
    fn rejects_indivisible_global_shard_extent() {
        let mut local = make_local(2);
        let mut remote = make_remote(2);
        // Set global HeadCount to a value that doesn't divide by 2.
        for (axis, v) in local.global_extents.iter_mut() {
            if *axis == KvDim::HeadCount {
                *v = 7;
            }
        }
        for d in &mut remote {
            for (axis, v) in d.global_extents.iter_mut() {
                if *axis == KvDim::HeadCount {
                    *v = 7;
                }
            }
        }
        let err = plan_pull(
            &local,
            &remote,
            instance(),
            handle(),
            handle(),
            &refs(1),
            &opts(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not divisible"), "msg: {err}");
    }
}
