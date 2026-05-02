// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request state for the unified conditional-disagg coordinator
//! (R-B Slice 2).
//!
//! Today the decode side ([`super::decode::RemotePrefillCoordinator`]
//! + [`super::decode_leader::DecodeDisaggLeader`]) and the prefill
//! side ([`super::prefill_coordinator::PrefillCoordinatorImpl`])
//! each maintain their own per-request structs. The R-B refactor
//! collapses them onto one [`CdRequest`] type whose role-conditional
//! fields are carried by a [`CdRequestRole`] enum.
//!
//! ## Slice 2 status
//!
//! These types are introduced but **not yet wired**.  Slice 3
//! (prefill flow) and Slice 4 (decode flow) plug the new
//! [`super::coordinator::ConditionalDisaggCoordinator`] in against
//! these types; Slice 5 retires the per-side types.  Until then the
//! production hot path uses the existing per-side coordinators —
//! consult `/home/ryan/.claude/plans/cd-unified-leader-rb.md` §5.
//!
//! ## Field layout — what's shared vs role-conditional
//!
//! Shared at top level: `request_id`, `session_id`,
//! `peer_instance_id`, the [`Session`] handle (lazily attached on
//! the prefill side, holder-opened on the decode side), and a
//! coarse [`CdRequestStatus`] used by lifecycle escalation.
//!
//! Role-specific: every other field, captured by [`DecodeBits`] /
//! [`PrefillBits`].  The asymmetric per-side state machines retain
//! their own granular status enums + atomics inside the bits types
//! so that role-specific code can keep its existing semantics
//! during the migration; collapsing the status enums (if it makes
//! sense) is deferred to Slice 4 cleanup.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kvbm_disagg_protocol::SessionId;
use kvbm_engine::disagg::session::Session;
use kvbm_logical::blocks::ImmutableBlock;
use parking_lot::Mutex;

use super::prefill_coordinator::{ObserverHandle, PrefillStatus};
use crate::{BlockId, G2, InstanceId, SequenceHash};

/// Coarse top-level status used by lifecycle escalation.  Per-role
/// state machines live inside [`DecodeBits`] / [`PrefillBits`] —
/// this enum only distinguishes the request-is-live vs
/// request-is-terminal state that the watcher cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdRequestStatus {
    /// Request is in-flight; lifecycle escalation should fire on
    /// peer detach / Failed / watchdog.
    Active,
    /// Request has been released cooperatively (cleanup completed
    /// normally or `cleanup_failed_request` ran).  Lifecycle
    /// watcher's eviction is a no-op idempotent removal.
    Released,
}

/// Per-request state for a CD-bound request.  Shared fields live at
/// the top level; the role enum carries per-side state.
pub struct CdRequest {
    pub request_id: String,
    pub session_id: SessionId,
    /// Peer endpoint identifier — for decode this is
    /// `prefill_instance_id` (whoever picks up our queued request);
    /// for prefill this is `initiator_instance_id` (the decode peer
    /// that issued the [`SessionId`] we attach to).
    pub peer_instance_id: InstanceId,
    /// Session handle.  Decode opens it synchronously in
    /// `begin_remote_prefill`; prefill attaches asynchronously in
    /// `run_setup` after peer-resolve, so both sides hold an
    /// `Option` until attach lands.
    pub session: Mutex<Option<Arc<dyn Session>>>,
    /// Coarse lifecycle status.
    pub status: Mutex<CdRequestStatus>,
    /// Per-role state.
    pub role: CdRequestRole,
}

/// Role-conditional state.  Only constructed by the role-specific
/// `gnmt` handler (decode: `commit_gnmt_remote`; prefill:
/// `ensure_started`); never mutated to switch roles after creation.
pub enum CdRequestRole {
    Decode(DecodeBits),
    Prefill(PrefillBits),
}

/// Decode-side state.  Mirrors the union of today's
/// `RemotePrefillState` ([`super::decode::RemotePrefillState`]) and
/// `CdRequestState` (private to `decode_leader.rs`).
///
/// `reserved_tokens` is the count consumed from the
/// `InflightBudget` at gnmt time; the budget release is the leader
/// wrapper's responsibility (not the coordinator's), retained
/// post-collapse.
pub struct DecodeBits {
    pub reserved_tokens: usize,
    pub failure_reason: Mutex<Option<String>>,

    /// Pinned local-match G2 entries — held until the local
    /// G2→G1 kick resolves so the entries stay live for the copy.
    pub local_match_g2_pins: Mutex<Option<Vec<ImmutableBlock<G2>>>>,
    pub local_match_g2_block_ids: Vec<BlockId>,
    pub local_match_g1_block_ids: Vec<BlockId>,
    pub local_onboard_complete: AtomicBool,

    /// Per-position remote-slice metadata, in expected order.  Built
    /// at USAA-1 and read-only afterward.
    pub remote_slots: Vec<RemoteSlotMeta>,
    /// `expected_hash → index in remote_slots`; built once.
    pub remote_slot_index: HashMap<SequenceHash, usize>,
    pub remote_pipeline_complete: AtomicBool,
    pub completed: AtomicBool,
}

/// Prefill-side state.  Mirrors today's
/// `PrefillCoordinatorImpl::RequestState` field-for-field; the only
/// move during R-B is the location, not the semantics.
pub struct PrefillBits {
    pub num_external_tokens: usize,
    pub expected_hashes: Vec<SequenceHash>,
    /// Decode-side `computed_blocks` offset; lets prefill translate
    /// `expected_hashes[i]` back to absolute token-block indices —
    /// `expected_hashes[i]` is at `computed_blocks_offset + i`.
    pub computed_blocks_offset: usize,
    /// G1 destinations from USAA, stashed until pull/register
    /// completes so the G2→G1 kick has them.
    pub pending_g1: Mutex<Option<Vec<BlockId>>>,
    /// FULL G1 window from USAA (entire prefill window vLLM
    /// allocated, not just the local-match prefix).  Captured once
    /// on the first non-no-op USAA so `cleanup_failed_request` can
    /// surface every slot to vLLM.  Stays `None` for pre-USAA
    /// failures.
    pub g1_block_ids: Mutex<Option<Vec<BlockId>>>,
    /// All registered G2 blocks across chunks.
    pub registered_g2: Mutex<Vec<ImmutableBlock<G2>>>,
    pub request_finished_seen: AtomicBool,
    /// `Drained` seen OR all expected hashes filled.
    pub pulls_complete: AtomicBool,
    /// Onboard kick was scheduled (idempotent).
    pub onboarding_scheduled: AtomicBool,
    /// Granular per-side state machine.  Retained on the bits type
    /// during R-B so prefill code keeps existing semantics; a
    /// future slice may collapse it into [`CdRequestStatus`].
    pub status: Mutex<PrefillStatus>,
    /// RAII observer-eviction handle.  Dropping `CdRequest` (last
    /// `Arc`) drops this, evicting the residual entry from
    /// `ConditionalDecodeG2Observer::pending`.  No explicit untrack
    /// needed in any failure path.
    pub observer_handle: ObserverHandle,
}

/// Per-position metadata for a remote-prefill block decode expects
/// the prefill peer to compute + publish.  Kept module-local for
/// Slice 2 (mirrors the private type in `decode_leader.rs`); Slice
/// 4 retires the duplicate.
#[derive(Debug, Clone)]
pub struct RemoteSlotMeta {
    pub expected_hash: SequenceHash,
    pub g1_dst_block_id: BlockId,
    pub sequence_index: usize,
}

impl CdRequest {
    /// Construct a decode-role request.
    pub fn new_decode(
        request_id: String,
        session_id: SessionId,
        peer_instance_id: InstanceId,
        bits: DecodeBits,
    ) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            session_id,
            peer_instance_id,
            session: Mutex::new(None),
            status: Mutex::new(CdRequestStatus::Active),
            role: CdRequestRole::Decode(bits),
        })
    }

    /// Construct a prefill-role request.
    pub fn new_prefill(
        request_id: String,
        session_id: SessionId,
        peer_instance_id: InstanceId,
        bits: PrefillBits,
    ) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            session_id,
            peer_instance_id,
            session: Mutex::new(None),
            status: Mutex::new(CdRequestStatus::Active),
            role: CdRequestRole::Prefill(bits),
        })
    }

    /// Borrow the decode-side bits if the request is decode-role.
    pub fn as_decode(&self) -> Option<&DecodeBits> {
        match &self.role {
            CdRequestRole::Decode(b) => Some(b),
            CdRequestRole::Prefill(_) => None,
        }
    }

    /// Borrow the prefill-side bits if the request is prefill-role.
    pub fn as_prefill(&self) -> Option<&PrefillBits> {
        match &self.role {
            CdRequestRole::Prefill(b) => Some(b),
            CdRequestRole::Decode(_) => None,
        }
    }

    /// Compute the failed G1 block-ids for this request, in the
    /// shape `mark_failed_onboarding` expects.  Decode reports the
    /// unfilled subset; prefill reports the full G1 window (or
    /// empty if pre-USAA).  Returns an empty Vec for pre-USAA
    /// prefill failures — the worker handler still pairs the
    /// request_id with `get_finished()` so vLLM unblocks.
    pub fn failed_g1_block_ids(&self) -> Vec<BlockId> {
        match &self.role {
            CdRequestRole::Decode(bits) => bits.unfilled_g1_block_ids(),
            CdRequestRole::Prefill(bits) => {
                bits.g1_block_ids.lock().clone().unwrap_or_default()
            }
        }
    }
}

impl DecodeBits {
    /// G1 destinations vLLM still expects to be filled on this
    /// request — the union of (a) the local-match slice if local
    /// onboard hasn't completed and (b) the remote slice if the
    /// remote pull pipeline hasn't completed.  Used by failure-path
    /// `mark_failed_onboarding` so vLLM aborts only the slots
    /// actually unfilled.
    pub fn unfilled_g1_block_ids(&self) -> Vec<BlockId> {
        use std::sync::atomic::Ordering;
        let mut out = Vec::new();
        if !self.local_onboard_complete.load(Ordering::Acquire) {
            out.extend(self.local_match_g1_block_ids.iter().copied());
        }
        if !self.remote_pipeline_complete.load(Ordering::Acquire) {
            out.extend(self.remote_slots.iter().map(|s| s.g1_dst_block_id));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_decode_bits() -> DecodeBits {
        DecodeBits {
            reserved_tokens: 32,
            failure_reason: Mutex::new(None),
            local_match_g2_pins: Mutex::new(None),
            local_match_g2_block_ids: vec![1, 2],
            local_match_g1_block_ids: vec![100, 101],
            local_onboard_complete: AtomicBool::new(false),
            remote_slots: vec![RemoteSlotMeta {
                expected_hash: SequenceHash::new(0, None, 0),
                g1_dst_block_id: 200,
                sequence_index: 2,
            }],
            remote_slot_index: HashMap::new(),
            remote_pipeline_complete: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        }
    }

    #[test]
    fn unfilled_returns_full_set_when_neither_pipeline_done() {
        let bits = fresh_decode_bits();
        let unfilled = bits.unfilled_g1_block_ids();
        assert_eq!(unfilled, vec![100, 101, 200]);
    }

    #[test]
    fn unfilled_excludes_local_when_local_onboard_done() {
        use std::sync::atomic::Ordering;
        let bits = fresh_decode_bits();
        bits.local_onboard_complete.store(true, Ordering::Release);
        let unfilled = bits.unfilled_g1_block_ids();
        assert_eq!(unfilled, vec![200]);
    }

    #[test]
    fn unfilled_excludes_remote_when_remote_pipeline_done() {
        use std::sync::atomic::Ordering;
        let bits = fresh_decode_bits();
        bits.remote_pipeline_complete.store(true, Ordering::Release);
        let unfilled = bits.unfilled_g1_block_ids();
        assert_eq!(unfilled, vec![100, 101]);
    }

    #[test]
    fn unfilled_empty_when_both_pipelines_done() {
        use std::sync::atomic::Ordering;
        let bits = fresh_decode_bits();
        bits.local_onboard_complete.store(true, Ordering::Release);
        bits.remote_pipeline_complete.store(true, Ordering::Release);
        let unfilled = bits.unfilled_g1_block_ids();
        assert!(unfilled.is_empty());
    }

    #[test]
    fn cd_request_role_borrows_correct_bits() {
        let session_id = uuid::Uuid::new_v4();
        let peer_instance_id: InstanceId = uuid::Uuid::new_v4().into();
        let req = CdRequest::new_decode(
            "req-d".to_string(),
            session_id,
            peer_instance_id,
            fresh_decode_bits(),
        );
        assert!(req.as_decode().is_some());
        assert!(req.as_prefill().is_none());
        assert_eq!(req.failed_g1_block_ids(), vec![100, 101, 200]);
    }
}
