// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How many continuations each prefill worker is running right now.
//!
//! The router already has a prefill-busy interlock, and it cannot do this job.
//! That interlock reads the worker's active prefill tokens, and the router
//! clears that figure when a request produces its **first token**. One token
//! into a four-thousand-token continuation the worker therefore reports no
//! prefill load at all, so the interlock bounds whether a continuation *starts*
//! and never bounds how many are *running*.
//!
//! This census is the bound that actually holds. It counts what the router
//! itself handed out, so nothing it counts can be cleared underneath it.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dynamo_kv_router::protocols::WorkerId;
use dynamo_runtime::pipeline::{ManyOut, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::stream::Stream;
use parking_lot::Mutex;

use crate::protocols::common::llm_backend::LLMEngineOutput;

type LlmResponse = Annotated<LLMEngineOutput>;

/// Continuations in flight, per prefill worker.
#[derive(Default)]
pub(super) struct ContinuationCensus {
    in_flight: Mutex<HashMap<WorkerId, usize>>,
}

impl ContinuationCensus {
    /// Take a place on `worker_id` if the cap leaves one.
    ///
    /// Tests the cap and takes the place under one lock. Reading the count and
    /// then incrementing it would let two requests arriving together both see
    /// the last free place and both take it.
    ///
    /// The cap is required, not optional. Startup validation asks for one, but
    /// it does not run on every path a router can be built from, so a bound
    /// this could waive would be no bound at all.
    pub(super) fn try_admit(
        self: &Arc<Self>,
        worker_id: WorkerId,
        cap: usize,
    ) -> Option<ContinuationPermit> {
        let mut in_flight = self.in_flight.lock();
        // Read rather than `entry`, which would insert a zero row that the
        // refusal below then leaves behind.
        let running = in_flight.get(&worker_id).copied().unwrap_or(0);
        if running >= cap {
            return None;
        }
        in_flight.insert(worker_id, running + 1);
        Some(ContinuationPermit {
            census: Arc::clone(self),
            worker_id,
        })
    }

    pub(super) fn in_flight(&self, worker_id: WorkerId) -> usize {
        self.in_flight.lock().get(&worker_id).copied().unwrap_or(0)
    }

    /// The emptiest routable worker's count, or `None` when there is nothing to
    /// route to.
    ///
    /// This is what the pre-routing decision can honestly ask. It cannot ask
    /// "has *the* worker room", because the worker is not chosen yet; it can
    /// ask "has *any* worker room", and the per-worker bound is then applied
    /// for real at dispatch.
    pub(super) fn min_in_flight(&self, routable: &[WorkerId]) -> Option<usize> {
        let in_flight = self.in_flight.lock();
        routable
            .iter()
            .map(|worker_id| in_flight.get(worker_id).copied().unwrap_or(0))
            .min()
    }

    fn release(&self, worker_id: WorkerId) {
        let mut in_flight = self.in_flight.lock();
        let Some(running) = in_flight.get_mut(&worker_id) else {
            return;
        };
        *running -= 1;
        // Drop the key at zero, so a fleet that churns workers does not grow
        // this map forever.
        if *running == 0 {
            in_flight.remove(&worker_id);
        }
    }
}

/// One continuation's place in the census, released when this is dropped.
pub(super) struct ContinuationPermit {
    census: Arc<ContinuationCensus>,
    worker_id: WorkerId,
}

impl ContinuationPermit {
    /// Tie this permit's life to the stream it accounts for.
    ///
    /// A continuation ends when its stream ends, deep in the client's read
    /// loop, so nothing the router holds has the right lifetime. The place goes
    /// back at the end of the stream, and also if the stream is dropped first,
    /// which is what a client disconnect should do.
    pub(super) fn into_stream(self, stream: ManyOut<LlmResponse>) -> ManyOut<LlmResponse> {
        let context = stream.context();
        ResponseStream::new(
            Box::pin(CountedContinuation {
                stream,
                permit: Some(self),
            }),
            context,
        )
    }
}

impl Drop for ContinuationPermit {
    fn drop(&mut self) {
        self.census.release(self.worker_id);
    }
}

/// A continuation's response stream, holding its place in the census.
struct CountedContinuation {
    stream: ManyOut<LlmResponse>,
    permit: Option<ContinuationPermit>,
}

impl Stream for CountedContinuation {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(None) => {
                // Give the place back as soon as the worker is done, rather
                // than waiting for a client that may hold a finished stream.
                drop(self.permit.take());
                Poll::Ready(None)
            }
            poll => poll,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn census() -> Arc<ContinuationCensus> {
        Arc::new(ContinuationCensus::default())
    }

    #[test]
    fn a_permit_is_counted_until_it_is_dropped() {
        let census = census();

        let permit = census.try_admit(7, 2).expect("first place is free");
        assert_eq!(census.in_flight(7), 1);

        drop(permit);
        assert_eq!(census.in_flight(7), 0);
        assert!(
            census.in_flight.lock().is_empty(),
            "a worker at zero must not keep a row, or the map grows with fleet churn"
        );
    }

    #[test]
    fn the_cap_refuses_the_place_past_it() {
        let census = census();

        let _first = census.try_admit(7, 2).expect("first");
        let second = census.try_admit(7, 2).expect("second");
        assert!(
            census.try_admit(7, 2).is_none(),
            "a third continuation must be refused at a cap of two"
        );

        // Releasing one frees exactly one place.
        drop(second);
        assert!(census.try_admit(7, 2).is_some());
    }

    #[test]
    fn workers_are_counted_separately() {
        let census = census();

        let _seven = census.try_admit(7, 1).expect("worker 7");
        // Bound, not a temporary: a temporary would release its own place.
        let _eight = census
            .try_admit(8, 1)
            .expect("one worker being full must not refuse another");

        assert_eq!(census.in_flight(7), 1);
        assert_eq!(census.in_flight(8), 1);
    }

    #[test]
    fn min_in_flight_reports_the_emptiest_worker() {
        let census = census();
        let _seven = census.try_admit(7, 4).expect("worker 7");
        let _also_seven = census.try_admit(7, 4).expect("worker 7 again");
        let _eight = census.try_admit(8, 4).expect("worker 8");

        // 9 has never been admitted, so it is empty and it is the minimum.
        assert_eq!(census.min_in_flight(&[7, 8, 9]), Some(0));
        assert_eq!(census.min_in_flight(&[7, 8]), Some(1));
        assert_eq!(census.min_in_flight(&[7]), Some(2));
    }

    #[test]
    fn min_in_flight_of_nothing_is_unknown() {
        // Not zero: an empty pool has no emptiest worker, and reporting zero
        // would read as "there is room".
        assert_eq!(census().min_in_flight(&[]), None);
    }

    #[test]
    fn a_cap_of_zero_refuses_everything() {
        let census = census();

        assert!(census.try_admit(7, 0).is_none());
        assert!(
            census.in_flight.lock().is_empty(),
            "a refused admission must not leave a row behind"
        );
    }
}
