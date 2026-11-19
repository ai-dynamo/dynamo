// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request dispatch over the SGLang Rust listeners of one node. A
//! listener runs every request on its own DP rank, so the router's rank
//! picks the listener.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::Duration;

use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, PreprocessedRequest,
};
use dynamo_sidecar_common::GrpcEndpoint;
use futures::stream::{BoxStream, FuturesUnordered};
use futures::{Stream, StreamExt};

use crate::client::{self, AbortRpc, Pool};
use crate::native_http::NativeHttp;
use crate::protocol::validate_dp_rank;

/// Where one request runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dispatch {
    /// The listener, by its index in rank order.
    pub index: usize,
    /// The rank a prefill request's rendezvous carries, in its bootstrap
    /// port and room. A single-rank deployment carries none.
    pub prefill_rank: Option<u32>,
}

/// One Rust listener of the node.
pub(crate) struct Listener {
    pub dp_rank: u32,
    pub endpoint: GrpcEndpoint,
    pub pool: Pool,
    /// Native `/generate`, when every listener of the node serves it.
    pub native_http: Option<NativeHttp>,
    /// Its prefill bootstrap registry's port (its HTTP port).
    pub bootstrap_port: Option<u16>,
}

/// Each running dispatch's listener index and rid suffix, keyed by the
/// address of its context allocation.
type Dispatches = Arc<Mutex<HashMap<usize, (usize, u32)>>>;

/// Each critical section is one map operation, so a panic elsewhere cannot
/// leave the map half-updated, and a poisoned lock is still safe to use.
fn lock(
    dispatches: &Mutex<HashMap<usize, (usize, u32)>>,
) -> MutexGuard<'_, HashMap<usize, (usize, u32)>> {
    dispatches.lock().unwrap_or_else(|e| e.into_inner())
}

/// A dispatch's rid: its public ID plus `suffix` in hex.
fn dispatch_rid(id: &str, suffix: u32) -> String {
    format!("{id}-{suffix:08x}")
}

fn context_key(ctx: &Arc<dyn AsyncEngineContext>) -> usize {
    Arc::as_ptr(ctx).addr()
}

/// The node's listeners, by DP rank `[start_rank, start_rank + len)`.
pub(crate) struct NodeListeners {
    pub start_rank: u32,
    /// Every DP rank of the deployment.
    pub global_dp_size: u32,
    pub listeners: Vec<Listener>,
    /// Round-robin position for requests that name no rank.
    next: AtomicUsize,
    /// Where each running request was sent, so that its abort reaches only
    /// that dispatch.
    dispatches: Dispatches,
}

/// Records one dispatch while it runs. See [`NodeListeners::dispatch`].
pub(crate) struct Dispatched {
    dispatches: Dispatches,
    // Holding the allocation keeps its address, the map key, from being
    // reused before the entry is removed.
    context: Arc<dyn AsyncEngineContext>,
    rid: String,
    suffix: u32,
}

impl Dispatched {
    /// The rid SGLang knows this dispatch by.
    pub(crate) fn rid(&self) -> &str {
        &self.rid
    }

    /// Keep the dispatch recorded for as long as `stream` exists, not just
    /// while it runs: the framework holds a cancelled request's stream until
    /// its abort has run, and the abort looks the dispatch up.
    pub(crate) fn attach<T: 'static>(self, stream: BoxStream<'static, T>) -> BoxStream<'static, T> {
        Box::pin(Attached {
            stream,
            _dispatched: self,
        })
    }
}

impl Drop for Dispatched {
    fn drop(&mut self) {
        let mut dispatches = lock(&self.dispatches);
        let key = context_key(&self.context);
        // A later dispatch of the same context owns the entry.
        if dispatches
            .get(&key)
            .is_some_and(|&(_, suffix)| suffix == self.suffix)
        {
            dispatches.remove(&key);
        }
    }
}

struct Attached<T> {
    stream: BoxStream<'static, T>,
    _dispatched: Dispatched,
}

impl<T> Stream for Attached<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl NodeListeners {
    pub(crate) fn new(
        start_rank: u32,
        global_dp_size: u32,
        listeners: Vec<Listener>,
    ) -> Result<Self, DynamoError> {
        if listeners.is_empty() {
            return Err(client::protocol_error(
                "SGLang reported no listeners on this node",
            ));
        }
        Ok(Self {
            start_rank,
            global_dp_size,
            listeners,
            next: AtomicUsize::new(0),
            dispatches: Dispatches::default(),
        })
    }

    fn ranks(&self) -> std::ops::Range<u32> {
        self.start_rank..self.start_rank + self.listeners.len() as u32
    }

    /// Choose the listener for `request` by the rank it names: the KV
    /// router's choice, or a rank the client pinned. A rank this node does
    /// not serve is refused before any stream opens, which keeps the worker
    /// routable. Requests that name no rank go round-robin over the node's
    /// listeners.
    pub(crate) fn select(
        &self,
        request: &PreprocessedRequest,
        mode: DisaggregationMode,
    ) -> Result<Dispatch, DynamoError> {
        let index = match validate_dp_rank(request, mode, self.ranks())? {
            Some(rank) => (rank - self.start_rank) as usize,
            None => self.next.fetch_add(1, Ordering::Relaxed) % self.listeners.len(),
        };
        let carries_rank = mode.is_prefill() && self.global_dp_size > 1;
        Ok(Dispatch {
            index,
            prefill_rank: carries_rank.then_some(self.listeners[index].dp_rank),
        })
    }

    /// Record that `ctx` runs on listener `index`, under a rid of its own: its
    /// public ID plus a random suffix. A retry keeps the public ID and can
    /// run beside the failed attempt, on this listener or another, and SGLang
    /// aborts every generation that shares a rid. The record lasts until the
    /// returned guard drops.
    pub(crate) fn dispatch(&self, ctx: Arc<dyn AsyncEngineContext>, index: usize) -> Dispatched {
        let suffix = rand::random::<u32>();
        let rid = dispatch_rid(ctx.id(), suffix);
        lock(&self.dispatches).insert(context_key(&ctx), (index, suffix));
        Dispatched {
            dispatches: self.dispatches.clone(),
            context: ctx,
            rid,
            suffix,
        }
    }

    /// The listener index and rid that `ctx` runs under, while it runs.
    fn running(&self, ctx: &Arc<dyn AsyncEngineContext>) -> Option<(usize, String)> {
        let (index, suffix) = *lock(&self.dispatches).get(&context_key(ctx))?;
        Some((index, dispatch_rid(ctx.id(), suffix)))
    }

    /// Abort the request of `ctx` by its own rid, on the listener that runs
    /// it, within `timeout`. Another request with the same public ID is left
    /// alone. A request that is no longer running has nothing to abort.
    pub(crate) async fn abort(&self, ctx: &Arc<dyn AsyncEngineContext>, timeout: Duration) {
        let Some((index, rid)) = self.running(ctx) else {
            return;
        };
        AbortRpc::new(self.listeners[index].pool.control_client())
            .abort(&rid, timeout)
            .await;
    }

    /// Polls every listener and resolves when the first one stops serving,
    /// naming it: the node is lost with any of its ranks. Dropping the
    /// future stops every poll.
    pub(crate) fn watch_serving(
        &self,
        attempt_timeout: Duration,
    ) -> impl Future<Output = String> + Send + 'static {
        let mut watches: FuturesUnordered<_> = self
            .listeners
            .iter()
            .map(|listener| {
                let polls = client::watch_serving(listener.pool.control_client(), attempt_timeout);
                let (dp_rank, endpoint) = (listener.dp_rank, listener.endpoint.clone());
                async move {
                    let why = polls.await;
                    format!("listener for DP rank {dp_rank} at {endpoint}: {why}")
                }
            })
            .collect();
        async move {
            watches
                .next()
                .await
                .unwrap_or_else(|| "the node has no listeners".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::engine::RoutingHints;
    use dynamo_backend_common::{
        BackendError, ErrorType, OutputOptions, SamplingOptions, StopConditions,
    };

    use super::*;

    fn endpoint(port: u16) -> GrpcEndpoint {
        GrpcEndpoint::parse(&format!("127.0.0.1:{port}"), "test").unwrap()
    }

    /// A node with ranks `[start, start + count)` of a `global`-rank
    /// deployment, whose listeners never connect.
    fn node(start: u32, count: u32, global: u32) -> NodeListeners {
        let listeners = (0..count)
            .map(|offset| {
                let endpoint = endpoint(1 + offset as u16);
                Listener {
                    dp_rank: start + offset,
                    endpoint: endpoint.clone(),
                    pool: Pool::lazy(&endpoint),
                    native_http: None,
                    bootstrap_port: None,
                }
            })
            .collect();
        NodeListeners::new(start, global, listeners).unwrap()
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions::default())
            .build()
            .unwrap()
    }

    fn routed(dp_rank: Option<u32>, prefill_dp_rank: Option<u32>) -> PreprocessedRequest {
        let mut request = request();
        request.routing = Some(RoutingHints {
            dp_rank,
            prefill_dp_rank,
            ..Default::default()
        });
        request
    }

    fn assert_invalid(error: DynamoError, message: &str) {
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }

    #[tokio::test]
    async fn routing_rank_selects_listener() {
        use DisaggregationMode::{Aggregated, Decode, Prefill};
        // Node 1 of two: ranks [2, 4) of four.
        let listeners = node(2, 2, 4);
        for (mode, request, rank, prefill_rank) in [
            (Aggregated, routed(Some(3), None), 3, None),
            (Decode, routed(Some(2), Some(3)), 2, None),
            // Prefill takes its own rank first.
            (Prefill, routed(Some(2), Some(3)), 3, Some(3)),
        ] {
            let index = (rank - 2) as usize;
            assert_eq!(
                listeners.select(&request, mode).unwrap(),
                Dispatch {
                    index,
                    prefill_rank,
                },
                "{mode:?}"
            );
        }
        for rank in [1, 4] {
            assert_invalid(
                listeners
                    .select(&routed(Some(rank), None), Aggregated)
                    .unwrap_err(),
                "serves DP ranks [2, 4)",
            );
        }
        assert_invalid(
            listeners
                .select(&routed(None, Some(4)), Prefill)
                .unwrap_err(),
            "serves DP ranks [2, 4)",
        );
        // Requests that name no rank alternate over the node's listeners, and
        // a prefill carries the rank it lands on. A fresh node starts the turn
        // at its first listener.
        let listeners = node(2, 2, 4);
        for (mode, rank, prefill_rank) in [
            (Aggregated, 2, None),
            (Prefill, 3, Some(3)),
            (Decode, 2, None),
        ] {
            assert_eq!(
                listeners.select(&request(), mode).unwrap(),
                Dispatch {
                    index: (rank - 2) as usize,
                    prefill_rank,
                },
                "{mode:?}"
            );
        }
        // A single-rank deployment carries no rank.
        assert_eq!(
            node(0, 1, 1)
                .select(&request(), Prefill)
                .unwrap()
                .prefill_rank,
            None
        );
    }

    #[tokio::test]
    async fn repeated_public_ids_have_their_own_abort_targets() {
        use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
        let listeners = node(0, 2, 2);
        let context =
            || Context::with_id_and_metadata((), "same-id".into(), Default::default()).context();
        let (failed, retry) = (context(), context());
        assert_eq!(failed.id(), retry.id());
        // The failed attempt runs on rank 0 and its retry on rank 1.
        let first = listeners.dispatch(failed.clone(), 0);
        let second = listeners.dispatch(retry.clone(), 1);
        assert!(first.rid().starts_with("same-id-") && second.rid().starts_with("same-id-"));
        assert_ne!(first.rid(), second.rid());
        assert_eq!(listeners.running(&failed), Some((0, first.rid().into())));
        assert_eq!(listeners.running(&retry), Some((1, second.rid().into())));
        drop(first);
        assert_eq!(listeners.running(&failed), None);
        assert_eq!(listeners.running(&retry), Some((1, second.rid().into())));
        drop(second);
        assert!(lock(&listeners.dispatches).is_empty());

        // A later dispatch of the same context owns its entry.
        let ctx = dynamo_backend_common::testing::mock_context();
        let earlier = listeners.dispatch(ctx.clone(), 0);
        let later = listeners.dispatch(ctx.clone(), 1);
        drop(earlier);
        assert_eq!(listeners.running(&ctx), Some((1, later.rid().into())));
        drop(later);
        assert!(lock(&listeners.dispatches).is_empty());
    }

    #[tokio::test]
    async fn dispatch_stays_recorded_until_its_stream_drops() {
        let listeners = node(0, 1, 1);
        let ctx = dynamo_backend_common::testing::mock_context();
        let mut stream = listeners
            .dispatch(ctx.clone(), 0)
            .attach(futures::stream::empty::<()>().boxed());
        assert_eq!(stream.next().await, None);
        // A cancelled request's abort runs after its stream ends.
        assert!(listeners.running(&ctx).is_some());
        drop(stream);
        assert_eq!(listeners.running(&ctx), None);
    }
}
