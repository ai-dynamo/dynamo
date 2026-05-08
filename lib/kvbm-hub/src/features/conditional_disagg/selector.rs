// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side prefill-worker selection.
//!

use crate::protocol::PrefillRequest;
use anyhow::Result;
use dashmap::DashMap;
use futures::future::BoxFuture;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use velo_common::InstanceId;

pub struct LoadPermit {
    inner: Option<LoadPermitInner>, // enable no-op variant (e.g. for round robin)
}

struct LoadPermitInner {
    in_flight_counter: Arc<DashMap<InstanceId, AtomicUsize>>,
    instance_id: InstanceId,
}

impl LoadPermit {
    pub fn new(
        in_flight_counter: Arc<DashMap<InstanceId, AtomicUsize>>,
        instance_id: InstanceId,
    ) -> Self {
        Self {
            inner: Some(LoadPermitInner {
                in_flight_counter,
                instance_id,
            }),
        }
    }
    pub fn new_noop() -> Self {
        Self { inner: None }
    }
}

impl Drop for LoadPermit {
    fn drop(&mut self) {
        if let Some(inner) = &self.inner {
            if let Some(counter) = inner.in_flight_counter.get(&inner.instance_id) {
                counter.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

// Entry of currently-live prefill peers
#[derive(Clone, Debug)]
pub struct PrefillPeerEntry {
    pub instance_id: InstanceId,
    pub engine_url: String,
}

/// Read-only "what prefill peers are alive currently" interface, implemented
/// by the manager. Decoupled from the manager type so selectors don't need
/// to know how role membership and addresses are tracked.
///
/// FUTURE-WORK: this trait is the seam where a focused `CdPeerRegistry` type
/// could plug in (mirroring dynamo's `Client`/`PushRouter` split). Today the
/// `ConditionalDisaggManager` implements it directly; if a second consumer
/// of CD per-peer state appears, extracting the implementation into a
/// dedicated registry struct is a natural follow-up.
pub trait PrefillPeerSource: Send + Sync {
    fn prefill_peers(&self) -> Vec<PrefillPeerEntry>;
}

/// Selection policy for prefill workers.
///
/// The selector receives a fresh slice of candidate peers per call — it
/// holds no long-lived reference to the peer source. This keeps selectors
/// stateless w.r.t. discovery (and avoids any reference cycle between
/// manager → selector → manager). Per-policy state (e.g. an in-flight
/// counter for `LeastLoadedSelector`) lives on the selector itself.
pub trait PrefillWorkerSelector: Send + Sync {
    fn select<'a>(
        &'a self,
        request: &'a PrefillRequest,
        peers: &'a [PrefillPeerEntry],
    ) -> BoxFuture<'a, Result<SelectedWorker>>;
}

pub struct SelectedWorker {
    pub instance_id: InstanceId,
    pub engine_url: String,
    pub permit: LoadPermit, // RAII guard
}

/// (1) Round-robin selector — cycles through peers in slice order.
///
/// Holds only a counter; peer list comes in per call.
pub struct RoundRobinSelector {
    counter: AtomicUsize,
}

impl Default for RoundRobinSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobinSelector {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl PrefillWorkerSelector for RoundRobinSelector {
    fn select<'a>(
        &'a self,
        _request: &'a PrefillRequest,
        peers: &'a [PrefillPeerEntry],
    ) -> BoxFuture<'a, Result<SelectedWorker>> {
        Box::pin(async move {
            if peers.is_empty() {
                return Err(anyhow::anyhow!("no prefill peers registered"));
            }
            // round robin indexing for routing choice
            let idx = self.counter.fetch_add(1, Ordering::Relaxed) % peers.len();
            let target = peers[idx].clone();
            Ok(SelectedWorker {
                instance_id: target.instance_id,
                engine_url: target.engine_url,
                permit: LoadPermit::new_noop(),
            })
        })
    }
}

/// (2) Least-loaded selector — picks the peer with the fewest in-flight
/// requests.
///
/// Holds an internal `DashMap<InstanceId, AtomicUsize>` for load tracking;
/// peer list comes in per call. Returned `LoadPermit` decrements the
/// counter on drop, so the in-flight count tracks request lifetimes via
/// RAII. Read-then-increment race is accepted (brief over-allocation
/// self-corrects as permits drop) — same trade-off dynamo's
/// `PushRouter::least_loaded` makes.
pub struct LeastLoadedSelector {
    in_flight_counter: Arc<DashMap<InstanceId, AtomicUsize>>,
}

impl Default for LeastLoadedSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl LeastLoadedSelector {
    pub fn new() -> Self {
        Self {
            in_flight_counter: Arc::new(DashMap::new()),
        }
    }
}

impl PrefillWorkerSelector for LeastLoadedSelector {
    fn select<'a>(
        &'a self,
        _request: &'a PrefillRequest,
        peers: &'a [PrefillPeerEntry],
    ) -> BoxFuture<'a, Result<SelectedWorker>> {
        Box::pin(async move {
            if peers.is_empty() {
                return Err(anyhow::anyhow!("no prefill peers registered"));
            }
            // Pick the least-loaded peer.
            let target = peers
                .iter()
                .min_by_key(|peer| {
                    self.in_flight_counter
                        .get(&peer.instance_id)
                        .map(|c| c.load(Ordering::Relaxed))
                        .unwrap_or(0)
                })
                .expect("non-empty checked above")
                .clone();
            // Atomically initialize-and-increment the chosen worker's
            // in-flight count. Existing entry: +1; missing: becomes 1.
            self.in_flight_counter
                .entry(target.instance_id)
                .or_insert_with(|| AtomicUsize::new(0))
                .fetch_add(1, Ordering::Relaxed);
            Ok(SelectedWorker {
                instance_id: target.instance_id,
                engine_url: target.engine_url,
                permit: LoadPermit::new(self.in_flight_counter.clone(), target.instance_id),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_disagg_protocol::DISAGG_PROTOCOL_VERSION;

    fn fake_request() -> PrefillRequest {
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "test-req".to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            sequence_hashes: vec![],
            token_ids: vec![],
            num_computed_tokens: 0,
        }
    }

    fn peer(url: &str) -> PrefillPeerEntry {
        PrefillPeerEntry {
            instance_id: InstanceId::new_v4(),
            engine_url: url.to_string(),
        }
    }

    // ---- RoundRobinSelector ------------------------------------------------

    #[tokio::test]
    async fn round_robin_cycles_in_order_and_wraps() {
        let peers = vec![peer("http://a"), peer("http://b"), peer("http://c")];
        let sel = RoundRobinSelector::new();

        let req = fake_request();
        let p0 = sel.select(&req, &peers).await.unwrap();
        let p1 = sel.select(&req, &peers).await.unwrap();
        let p2 = sel.select(&req, &peers).await.unwrap();
        let p3 = sel.select(&req, &peers).await.unwrap();
        let p4 = sel.select(&req, &peers).await.unwrap();

        assert_eq!(p0.instance_id, peers[0].instance_id);
        assert_eq!(p1.instance_id, peers[1].instance_id);
        assert_eq!(p2.instance_id, peers[2].instance_id);
        // wraps around
        assert_eq!(p3.instance_id, peers[0].instance_id);
        assert_eq!(p4.instance_id, peers[1].instance_id);
    }

    #[tokio::test]
    async fn round_robin_returns_noop_permit() {
        let peers = vec![peer("http://only")];
        let sel = RoundRobinSelector::new();
        let selected = sel.select(&fake_request(), &peers).await.unwrap();
        // Round-robin doesn't track load; permit must be a no-op so dropping
        // it doesn't try to decrement a counter that doesn't exist.
        assert!(selected.permit.inner.is_none());
    }

    #[tokio::test]
    async fn round_robin_errors_on_empty_peers() {
        let sel = RoundRobinSelector::new();
        let result = sel.select(&fake_request(), &[]).await;
        let err = match result {
            Ok(_) => panic!("expected error on empty peers"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("no prefill peers"));
    }

    // ---- LeastLoadedSelector -----------------------------------------------

    #[tokio::test]
    async fn least_loaded_picks_lowest_count() {
        let peers = vec![peer("http://a"), peer("http://b"), peer("http://c")];
        let sel = LeastLoadedSelector::new();
        let req = fake_request();

        // First pick: all peers have count=0; min_by_key returns the first.
        let p0 = sel.select(&req, &peers).await.unwrap();
        assert_eq!(p0.instance_id, peers[0].instance_id);

        // Now peers[0] has 1 in-flight; next pick should go to peers[1] (still 0).
        let p1 = sel.select(&req, &peers).await.unwrap();
        assert_eq!(p1.instance_id, peers[1].instance_id);

        // peers[0]=1, peers[1]=1, peers[2]=0 — should pick peers[2].
        let p2 = sel.select(&req, &peers).await.unwrap();
        assert_eq!(p2.instance_id, peers[2].instance_id);
    }

    #[tokio::test]
    async fn least_loaded_permit_decrements_on_drop() {
        let peers = vec![peer("http://a"), peer("http://b")];
        let sel = LeastLoadedSelector::new();
        let req = fake_request();

        // Hand out 3 permits to peers[0]: pick, drop, pick, drop, pick, drop.
        // After each drop, count returns to 0; next select keeps picking peers[0]
        // (tie-breaker prefers first in min_by_key).
        for _ in 0..3 {
            let s = sel.select(&req, &peers).await.unwrap();
            assert_eq!(s.instance_id, peers[0].instance_id);
            drop(s); // permit drops, counter decrements 1→0
        }

        // Pick again — peers[0] should still be at count=0 (drops took it back down).
        let s = sel.select(&req, &peers).await.unwrap();
        assert_eq!(s.instance_id, peers[0].instance_id);

        // Now hold the first permit. Next select should switch to peers[1].
        let _hold = s;
        let next = sel.select(&req, &peers).await.unwrap();
        assert_eq!(next.instance_id, peers[1].instance_id);
    }

    #[tokio::test]
    async fn least_loaded_concurrent_holds_distribute() {
        let peers = vec![peer("http://a"), peer("http://b"), peer("http://c")];
        let sel = LeastLoadedSelector::new();
        let req = fake_request();

        // Hold all three permits simultaneously — each peer should have count=1.
        let s0 = sel.select(&req, &peers).await.unwrap();
        let s1 = sel.select(&req, &peers).await.unwrap();
        let s2 = sel.select(&req, &peers).await.unwrap();
        let picked: std::collections::HashSet<_> =
            [s0.instance_id, s1.instance_id, s2.instance_id]
                .into_iter()
                .collect();
        assert_eq!(picked.len(), 3, "each peer should be picked once");
    }

    #[tokio::test]
    async fn least_loaded_errors_on_empty_peers() {
        let sel = LeastLoadedSelector::new();
        let result = sel.select(&fake_request(), &[]).await;
        let err = match result {
            Ok(_) => panic!("expected error on empty peers"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("no prefill peers"));
    }
}
