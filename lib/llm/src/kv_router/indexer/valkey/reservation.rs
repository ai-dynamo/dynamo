// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission request and lease state.

use super::*;

/// Caller-stable idempotency identity for one module admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ReservationNonce {
    pub(super) client_nonce: u64,
    pub(super) request_nonce: u64,
}

impl ReservationNonce {
    pub(super) fn random() -> Self {
        let bytes = Uuid::new_v4().into_bytes();
        Self {
            client_nonce: u64::from_be_bytes(bytes[..8].try_into().expect("UUID is 16 bytes")),
            request_nonce: u64::from_be_bytes(bytes[8..].try_into().expect("UUID is 16 bytes")),
        }
    }
}

/// A worker/rank accepted by frontend-local routing constraints.  Capacity is
/// latched by the module on first use and subsequently validated there; it is
/// not a frontend-provided load signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValkeyReservationCandidate {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) capacity: u32,
}

#[derive(Clone, Debug)]
pub(super) struct ReservationRequest {
    pub(super) domain: Vec<u8>,
    pub(super) nonce: ReservationNonce,
    pub(super) lease_ms: u64,
    pub(super) block_hashes: Vec<LocalBlockHash>,
    pub(super) candidates: Vec<ValkeyReservationCandidate>,
}

impl ReservationRequest {
    pub(super) fn retained_heap_bytes(&self) -> usize {
        self.domain
            .capacity()
            .saturating_add(
                self.block_hashes
                    .capacity()
                    .saturating_mul(std::mem::size_of::<LocalBlockHash>()),
            )
            .saturating_add(
                self.candidates
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ValkeyReservationCandidate>()),
            )
    }

    pub(super) fn take_lifecycle_request(&mut self) -> ReservationLifecycleRequest {
        ReservationLifecycleRequest {
            domain: std::mem::take(&mut self.domain),
            nonce: self.nonce,
            lease_ms: self.lease_ms,
        }
    }
}

/// Compact request identity retained after SELECT_RESERVE succeeds. Prefix
/// hashes and candidates are one-shot selection inputs and must not stay live
/// for the complete request/lease lifetime.
#[derive(Clone, Debug)]
pub(super) struct ReservationLifecycleRequest {
    pub(super) domain: Vec<u8>,
    pub(super) nonce: ReservationNonce,
    pub(super) lease_ms: u64,
}

impl ReservationLifecycleRequest {
    pub(super) fn retained_heap_bytes(&self) -> usize {
        self.domain.capacity()
    }
}

impl From<&ReservationRequest> for ReservationLifecycleRequest {
    fn from(request: &ReservationRequest) -> Self {
        Self {
            domain: request.domain.clone(),
            nonce: request.nonce,
            lease_ms: request.lease_ms,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ReservationGrant {
    pub(super) worker: WorkerWithDpRank,
    pub(super) expires_at_ms: u64,
    pub(super) matched_blocks: u32,
    pub(super) active_reservations_at_grant: u32,
}

#[derive(Debug)]
pub(super) struct ReservationLeaseState {
    pub(super) expires_at_ms: u64,
    pub(super) released: bool,
}

pub(super) struct ReservationLeaseInner {
    pub(super) indexer: ValkeyIndexer,
    pub(super) request: ReservationLifecycleRequest,
    pub(super) grant: ReservationGrant,
    pub(super) state: Mutex<ReservationLeaseState>,
    /// Serialize the one remote lifecycle mutation for this lease without
    /// holding the state snapshot lock across network I/O.
    pub(super) lifecycle: Mutex<()>,
    pub(super) renewal_cancel: CancellationToken,
    pub(super) renewal_started: AtomicBool,
    pub(super) release_started: AtomicBool,
}

/// A module-owned reservation that follows a selected request through
/// dispatch, streaming, cancellation, and cleanup.  It is intentionally not
/// cloneable: only one request guard owns the release responsibility.
pub(crate) struct ValkeyReservationLease {
    pub(super) inner: Arc<ReservationLeaseInner>,
}

/// Owns an idempotency key before the selection reply is known.  If a network
/// response is lost or selection is cancelled, its drop path replays the same
/// request to discover and release a possible committed reservation.
pub(crate) struct PendingValkeyReservation {
    pub(super) indexer: ValkeyIndexer,
    pub(super) request: ReservationRequest,
    pub(super) armed: bool,
}
