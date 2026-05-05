// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side bidirectional-streaming entry point.
//!
//! Sits as a parallel path to [`PushRouter::generate`] (unary). The same
//! [`PushRouter`] instance can be used for both — discovery, instance
//! selection, and the velo request plane handle are reused. The bidi path:
//!
//! 1. Picks a target instance using the existing routing strategy.
//! 2. Resolves its `velo://...` address from the discovery `Instance`.
//! 3. Creates a local `StreamAnchor<BidiFrame<U>>` (consumer of server→client
//!    response items).
//! 4. Issues a velo unary RPC to [`BIDI_INIT_HANDLER`] carrying the local
//!    anchor handle.
//! 5. Decodes the [`BidiInitResponse`]; on `Ok`, attaches a local
//!    `StreamSender<BidiFrame<T>>` to the server's anchor handle.
//! 6. Spawns the [`BidiSession::run_outgoing`] pump for client→server data.
//! 7. Returns a `ManyOut<U>` that:
//!    - decodes `BidiFrame<U>` into pure `U` items,
//!    - swallows `Done` (firing the closer notify) and `Finalized`,
//!    - and on drop, calls `controller.kill()` to cascade-tear-down the
//!      upstream pump and both velo streams.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use anyhow::{Context as AnyhowContext, Result, anyhow};
use bytes::Bytes;
use futures::Stream;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::warn;

use velo::streaming::{StreamAnchor, StreamSender};

use crate::component::TransportType;
use crate::engine::{
    AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data,
    DataStream,
};
use crate::pipeline::network::bidi::{
    BIDI_INIT_HANDLER, BIDI_UNATTACHED_TIMEOUT, BidiFrame, BidiInitRequest, BidiInitResponse,
};
use crate::pipeline::network::bidi::session::{BidiSession, SessionGuard, decode_bidi_anchor};
use crate::pipeline::network::egress::push_router::{PushRouter, RouterMode};
use crate::pipeline::network::velo::{
    ENDPOINT_HEADER, REQUEST_ID_HEADER, current_velo, decode_velo_address,
};
use crate::pipeline::{Context, ManyIn, ManyOut, ResponseStream};
use crate::protocols::maybe_error::MaybeError;

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize + DeserializeOwned,
    U: Data + Serialize + DeserializeOwned + MaybeError,
{
    /// Issue a bidirectional-streaming request.
    ///
    /// `input` is the request stream — the user's [`ManyIn<T>`] holding a
    /// `Stream<T> + Send`. Items are pumped to the selected backend over a
    /// velo stream; the returned [`ManyOut<U>`] yields response items as
    /// they arrive.
    ///
    /// Instance selection respects the router's [`RouterMode`] (random and
    /// round-robin shipped in v1; other modes fall back to random).
    pub async fn bidi_generate(&self, input: ManyIn<T>) -> Result<ManyOut<U>> {
        self.bidi_generate_with::<()>(input, ()).await
    }

    /// Like [`bidi_generate`], but the user supplies a typed `init: I`
    /// payload that the server-side handler can read from the
    /// [`Context`]'s registry under `BIDI_INIT_KEY`.
    pub async fn bidi_generate_with<I>(&self, input: ManyIn<T>, init: I) -> Result<ManyOut<U>>
    where
        I: Send + Sync + 'static + Serialize,
    {
        // 1. Pick a target instance.
        let instance_id = self.select_for_bidi()?;

        // 2. Resolve to a velo address.
        let address = self.resolve_velo_address(instance_id).with_context(|| {
            format!(
                "bidi: instance {instance_id} for endpoint {} has no velo address",
                self.client.endpoint.id()
            )
        })?;
        let parsed = decode_velo_address(&address)
            .with_context(|| format!("bidi: parsing velo address {address}"))?;

        // 3. Borrow the per-process velo handle.
        let velo = current_velo()?;

        // 4. Create our recv anchor (server -> client response items).
        //    Set a TTL so a failed unary RPC doesn't leak this anchor.
        let client_recv: StreamAnchor<BidiFrame<U>> = velo.create_anchor::<BidiFrame<U>>();
        client_recv.set_timeout(Some(BIDI_UNATTACHED_TIMEOUT));
        let client_handle = client_recv.handle();

        // 5. Take the input stream out of the holder.
        let (stream_holder, ctx_unit) = input.into_parts();
        let request_id = ctx_unit.id().to_string();
        let data_stream: DataStream<T> = stream_holder
            .take()
            .ok_or_else(|| anyhow!("bidi: ManyIn stream already taken"))?;

        // 6. Build BidiInitRequest payload.
        let init_payload = BidiInitRequest::<I> {
            client_handle,
            request_id: request_id.clone(),
            init,
            frontend_send_ts_ns: now_ns(),
        };
        let payload = rmp_serde::to_vec(&init_payload)
            .map_err(|e| anyhow!("bidi: serialize BidiInitRequest: {e}"))?;

        // 7. Issue the velo unary kick-off. Use the local velo handle directly
        //    so we don't need to round-trip through the unified
        //    `RequestPlaneClient` (which would erase velo-specific bits).
        let mut headers: HashMap<String, String> = HashMap::new();
        headers.insert(ENDPOINT_HEADER.to_string(), parsed.endpoint_key.clone());
        headers.insert(REQUEST_ID_HEADER.to_string(), request_id.clone());

        // Make sure velo knows how to dial the peer.
        if parsed.velo_instance != velo.instance_id() {
            velo.discover_and_register_peer(parsed.velo_instance)
                .await
                .with_context(|| {
                    format!(
                        "bidi: velo discover_and_register_peer({})",
                        parsed.velo_instance
                    )
                })?;
        }

        let ack: Bytes = velo
            .unary(BIDI_INIT_HANDLER)
            .map_err(|e| anyhow!("bidi: creating velo unary builder: {e}"))?
            .raw_payload(Bytes::from(payload))
            .headers(headers)
            .instance(parsed.velo_instance)
            .send()
            .await
            .with_context(|| {
                format!(
                    "bidi: velo unary send to {} (key {})",
                    parsed.velo_instance, parsed.endpoint_key
                )
            })?;

        // 8. Decode response.
        let resp: BidiInitResponse = rmp_serde::from_slice(&ack)
            .map_err(|e| anyhow!("bidi: decode BidiInitResponse: {e}"))?;
        let server_handle = match resp {
            BidiInitResponse::Ok { server_handle } => server_handle,
            BidiInitResponse::Err { reason } => {
                drop(client_recv); // unattached -> reaped by TTL
                return Err(anyhow!("bidi: server rejected init: {reason}"));
            }
        };

        // 9. Attach our send to the server's recv handle.
        let client_send: StreamSender<BidiFrame<T>> = velo
            .attach_anchor::<BidiFrame<T>>(server_handle)
            .await
            .map_err(|e| anyhow!("bidi: attach to server anchor failed: {e}"))?;

        // 10. Build session and spawn the upstream pump.
        let session = BidiSession::new(request_id);
        let pump_session = session.clone();
        tokio::spawn(async move {
            let outcome = pump_session.run_outgoing(data_stream, client_send).await;
            tracing::debug!(outcome = ?outcome, "bidi client: upstream pump ended");
        });

        // 11. Build the user-visible response stream:
        //     - decode BidiFrame<U> via decode_bidi_anchor (drainer task
        //       owns the anchor; ends user-visible mpsc on peer Done; keeps
        //       polling for Finalized to terminate the anchor cleanly)
        //     - wrap in BidiResponseStream<U> so dropping it kills the
        //       session controller (which terminates the upstream pump too)
        let user_stream =
            decode_bidi_anchor::<U>(client_recv, session.peer_done(), session.controller());
        let guarded: Pin<Box<dyn AsyncEngineStream<U>>> = Box::pin(BidiResponseStream::new(
            user_stream,
            session.context(),
            SessionGuard::new(session.controller()),
        ));
        Ok(guarded)
    }

    /// Resolve `instance_id` → `velo://...` address from the discovery list.
    /// Returns `None` if the instance isn't currently visible or isn't using
    /// the velo transport.
    fn resolve_velo_address(&self, instance_id: u64) -> Option<String> {
        let instances = self.client.instances();
        instances
            .iter()
            .find(|i| i.instance_id == instance_id)
            .and_then(|i| match &i.transport {
                TransportType::Velo(addr) => Some(addr.clone()),
                _ => None,
            })
    }

    /// Pick an instance for a bidi request.
    ///
    /// v1 honours `RouterMode::RoundRobin` and `RouterMode::Random`;
    /// other modes fall back to random. Bidi uses a single kick-off RPC, so
    /// occupancy-tracked routing modes (P2C / least-loaded /
    /// device-aware-weighted) would need a separate hook anyway — out of
    /// scope for v1.
    fn select_for_bidi(&self) -> Result<u64> {
        let avail = self.client.instance_ids_avail();
        if avail.is_empty() {
            return Err(anyhow!(
                "bidi: no instances found for endpoint {}",
                self.client.endpoint.id()
            ));
        }
        let id = match self.router_mode() {
            RouterMode::RoundRobin => {
                let counter = self.bump_round_robin();
                avail[counter % avail.len()]
            }
            _ => {
                use rand::Rng;
                let idx = rand::rng().random::<u64>() as usize % avail.len();
                avail[idx]
            }
        };
        Ok(id)
    }
}

fn now_ns() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos() as u64)
}

/// User-visible response stream that owns the session guard. On drop, the
/// guard kills the session controller, which cascades to the upstream pump
/// (both velo streams are torn down).
struct BidiResponseStream<U> {
    inner: DataStream<U>,
    ctx: Arc<dyn AsyncEngineContext>,
    _guard: SessionGuard,
}

impl<U> BidiResponseStream<U> {
    fn new(
        inner: DataStream<U>,
        ctx: Arc<dyn AsyncEngineContext>,
        guard: SessionGuard,
    ) -> Self {
        Self {
            inner,
            ctx,
            _guard: guard,
        }
    }
}

impl<U: Data> Stream for BidiResponseStream<U> {
    type Item = U;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<U>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<U: Data> AsyncEngineStream<U> for BidiResponseStream<U> {}

impl<U: Data> AsyncEngineContextProvider for BidiResponseStream<U> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.ctx.clone()
    }
}

impl<U> std::fmt::Debug for BidiResponseStream<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BidiResponseStream")
            .field("ctx", &self.ctx)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tiny helpers we add to PushRouter to avoid leaking internals.
// ---------------------------------------------------------------------------

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + DeserializeOwned + MaybeError,
{
    fn router_mode(&self) -> RouterMode {
        // The router_mode field is private; expose via a public accessor.
        self.router_mode_value()
    }
    fn bump_round_robin(&self) -> usize {
        self.bump_round_robin_value()
    }
}
