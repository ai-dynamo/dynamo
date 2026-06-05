// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The dispatcher contract the CD prefill queue feeds.
//!
//! The disagg feature owns the queue plumbing — it dequeues a
//! [`PrefillRequest`] and hands it to a [`PrefillRequestDispatcher`].
//! What that dispatcher *does* with the request is the boundary between
//! "queue mechanics" and "where the request actually goes". This crate
//! keeps the trait here because the queue and the router are
//! orthogonal features; they only meet through this trait.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::protocol::PrefillRequest;

/// Outcome reported by a dispatcher implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// The request was accepted for dispatch by the downstream peer.
    Accepted,
    /// The dispatcher could not place the request — e.g. no eligible
    /// peer, peer rejected, transport error. The CD queue worker
    /// surfaces this to whoever cares (logging today; failure-marker
    /// callback in a future iteration).
    Rejected {
        /// Human-readable reason.
        reason: String,
    },
}

/// Action that consumes a dequeued [`PrefillRequest`] from the CD prefill
/// queue and routes it to a prefill participant.
///
/// Implementations must be cheap to clone behind `Arc` and safe to call
/// concurrently from multiple worker tasks (today the CD pump spawns one,
/// but the trait is shaped to allow scaling out without changing call
/// sites).
pub trait PrefillRequestDispatcher: Send + Sync {
    /// Hand a single dequeued request to the dispatcher. The dispatcher
    /// is responsible for any selection + transport work; returning
    /// `Ok(Accepted)` does not imply the request is *complete*, only that
    /// it has been handed off.
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>>;
}

/// Test-only dispatcher that records every received [`PrefillRequest`]
/// for later assertion. Returns `Accepted` for every request.
pub struct RecordingDispatcher {
    received: Mutex<VecDeque<PrefillRequest>>,
}

impl RecordingDispatcher {
    /// Build a fresh recorder behind an [`Arc`].
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            received: Mutex::new(VecDeque::new()),
        })
    }

    /// Snapshot of all dispatched requests, in arrival order. Does not
    /// drain the recorder — repeat calls return the same items.
    pub fn recorded(&self) -> Vec<PrefillRequest> {
        self.received.lock().iter().cloned().collect()
    }

    /// Pop the oldest recorded request, blocking on a Mutex but never
    /// awaiting. Returns `None` if nothing has arrived yet.
    pub fn pop(&self) -> Option<PrefillRequest> {
        self.received.lock().pop_front()
    }

    /// Number of requests recorded but not popped.
    pub fn len(&self) -> usize {
        self.received.lock().len()
    }

    /// True iff no requests are pending.
    pub fn is_empty(&self) -> bool {
        self.received.lock().is_empty()
    }
}

impl Default for RecordingDispatcher {
    fn default() -> Self {
        Self {
            received: Mutex::new(VecDeque::new()),
        }
    }
}

impl PrefillRequestDispatcher for RecordingDispatcher {
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>> {
        Box::pin(async move {
            self.received.lock().push_back(request);
            Ok(DispatchOutcome::Accepted)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_protocols::disagg::DISAGG_PROTOCOL_VERSION;
    use velo_ext::InstanceId;

    fn make_request(id: &str) -> PrefillRequest {
        use kvbm_protocols::disagg::KvHashingRequestEnvelope;
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            token_ids: vec![1, 2, 3],
            num_provided_tokens: 0,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: None,
        }
    }

    #[tokio::test]
    async fn recording_dispatcher_accepts_and_records() {
        let d = RecordingDispatcher::new();
        let req = make_request("req-1");
        let outcome = d.dispatch(req.clone()).await.unwrap();
        assert_eq!(outcome, DispatchOutcome::Accepted);
        assert_eq!(d.len(), 1);
        assert_eq!(d.recorded()[0].request_id, "req-1");
    }

    #[tokio::test]
    async fn recording_dispatcher_preserves_order() {
        let d = RecordingDispatcher::new();
        for n in 0..5 {
            let req = make_request(&format!("req-{n}"));
            d.dispatch(req).await.unwrap();
        }
        let recorded = d.recorded();
        assert_eq!(recorded.len(), 5);
        for (i, r) in recorded.iter().enumerate() {
            assert_eq!(r.request_id, format!("req-{i}"));
        }
    }

    #[tokio::test]
    async fn recording_dispatcher_pop_drains() {
        let d = RecordingDispatcher::new();
        d.dispatch(make_request("a")).await.unwrap();
        d.dispatch(make_request("b")).await.unwrap();
        assert_eq!(d.pop().unwrap().request_id, "a");
        assert_eq!(d.pop().unwrap().request_id, "b");
        assert!(d.pop().is_none());
    }
}
