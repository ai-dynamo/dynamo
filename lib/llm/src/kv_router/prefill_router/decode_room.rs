// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-pool room for the prefill hop's policy-class queue.
//!
//! In disaggregated KV routing the prefill router's queue is the only admission gate:
//! it judges each class's prefill wait limit on the prefill workers and its decode wait
//! limit on the decode workers, through this pool. The decode router's queue books the
//! decode leg without waiting, because by then prefill is already running.

use std::sync::{Arc, OnceLock, Weak};

use dynamo_kv_router::scheduling::{BusyThresholds, PairedPoolRoom, SchedulingRequest};
use tokio::sync::watch;

use crate::kv_router::KvRouter;

/// The decode router a prefill router's queue judges decode wait limits on.
///
/// A prefill binding can activate before the frontend installs its decode
/// `RoutingHost`, so the decode router resolves lazily. Until it does, or once it is
/// gone, the pool reports room and the prefill queue waits on its prefill limits alone.
#[derive(Default)]
pub(crate) struct DecodePoolRoom {
    decode: OnceLock<Weak<KvRouter>>,
}

impl DecodePoolRoom {
    /// Resolve the pool to `decode`. The first router wins: the prefill router owning
    /// this pool accepts exactly one decode host.
    pub(crate) fn set_decode_router(&self, decode: &Arc<KvRouter>) {
        let _ = self.decode.set(Arc::downgrade(decode));
    }

    fn decode(&self) -> Option<Arc<KvRouter>> {
        self.decode.get().and_then(Weak::upgrade)
    }
}

impl PairedPoolRoom for DecodePoolRoom {
    fn has_room(&self, request: &SchedulingRequest, limits: &BusyThresholds) -> bool {
        // Coarse by design: decode selection still enforces the leg's exact eligibility
        // (`RoutingEligibility`), so a request this gate admits is routed at decode exactly
        // as it would be with no decode wait limit.
        self.decode()
            .is_none_or(|decode| decode.has_room_for_prefill_request(request, limits))
    }

    fn room_changed(&self) -> Option<watch::Receiver<()>> {
        self.decode().map(|decode| decode.subscribe_room_changed())
    }
}
