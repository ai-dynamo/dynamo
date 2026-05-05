// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-side bidi handler.
//!
//! - [`BidiPushWorkHandler`]: trait the velo demux handler invokes for the
//!   `dynamo-bidi-init` velo handler. Returns the encoded
//!   [`BidiInitResponse`] bytes (carries the server's anchor handle) which
//!   ride back to the client as the velo unary ACK.
//! - [`BidiIngress<T, U, I>`]: the concrete impl. Holds an
//!   `AsyncEngine<ManyIn<T>, ManyOut<U>, Error>` and the per-process
//!   `Arc<Velo>` it needs to create anchors / attach senders.
//!
//! Lifecycle on `handle_bidi_init`:
//! 1. Decode `BidiInitRequest<I>`; reject (return `BidiInitResponse::Err`)
//!    if attach to the client's anchor fails.
//! 2. Create `server_recv: StreamAnchor<BidiFrame<T>>` (consumes client→server).
//! 3. Attach `server_send: StreamSender<BidiFrame<U>>` to the client's handle
//!    (produces server→client).
//! 4. Build a [`BidiSession`] with the request id; share the controller with
//!    the user-facing `ManyIn<T>` so cancellation cascades.
//! 5. Stuff the user's `init: I` into the context registry under
//!    [`BIDI_INIT_KEY`] so the handler can `clone_unique` it.
//! 6. Spawn a driver task: invoke the user `engine.generate(many_in)`, pump
//!    the resulting `ManyOut<U>` through `server_send` via
//!    `session.run_outgoing`.
//! 7. Return `BidiInitResponse::Ok { server_handle }` immediately so the
//!    client can attach its sender and start streaming.

use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::{debug, warn};

use velo::Velo;
use velo::streaming::{StreamAnchor, StreamSender};

use crate::engine::{AsyncEngine, AsyncEngineContext, Data, DataStream, EngineStream};
use crate::pipeline::{
    AsyncRequestStream, Context, ManyIn, ManyOut, PipelineError, ServiceEngine,
    network::bidi::{
        BIDI_INIT_KEY, BIDI_UNATTACHED_TIMEOUT, BidiFrame, BidiInitRequest, BidiInitResponse,
    },
    network::bidi::session::{BidiSession, decode_bidi_anchor},
};

/// Trait the velo bidi-init demux handler invokes. Symmetric in shape to
/// [`PushWorkHandler`] but returns response bytes (the encoded
/// [`BidiInitResponse`]) instead of `()`, since the velo unary ACK carries
/// the server's anchor handle back to the client.
///
/// [`PushWorkHandler`]: crate::pipeline::network::PushWorkHandler
#[async_trait]
pub trait BidiPushWorkHandler: Send + Sync {
    /// Handle a bidi-init request. `payload` is the rmp_serde-encoded
    /// `BidiInitRequest<I>` (the concrete impl knows what `I` is).
    /// Return value is the rmp_serde-encoded [`BidiInitResponse`].
    async fn handle_bidi_init(
        &self,
        payload: Bytes,
        request_id: Option<String>,
    ) -> Result<Bytes, PipelineError>;

    /// Hook for endpoint metrics — symmetric with `PushWorkHandler::add_metrics`.
    fn add_metrics(
        &self,
        _endpoint: &crate::component::Endpoint,
        _metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<()> {
        Ok(())
    }

    /// Hook for endpoint health-check timer reset notifier — symmetric with
    /// `PushWorkHandler::set_endpoint_health_check_notifier`.
    fn set_endpoint_health_check_notifier(
        &self,
        _notifier: Arc<tokio::sync::Notify>,
    ) -> Result<()> {
        Ok(())
    }
}

/// Server-side bidi ingress. Wraps a user `AsyncEngine<ManyIn<T>, ManyOut<U>>`
/// and the per-process velo handle.
pub struct BidiIngress<T: Send + 'static, U, I = ()> {
    engine: OnceLock<ServiceEngine<ManyIn<T>, ManyOut<U>>>,
    velo: Arc<Velo>,
    _phantom: PhantomData<fn(I)>,
}

impl<T, U, I> BidiIngress<T, U, I>
where
    T: Data + Serialize + DeserializeOwned,
    U: Data + Serialize + DeserializeOwned,
    I: Data + DeserializeOwned,
{
    pub fn new(velo: Arc<Velo>) -> Arc<Self> {
        Arc::new(Self {
            engine: OnceLock::new(),
            velo,
            _phantom: PhantomData,
        })
    }

    /// Attach the user engine. Must be called before the first
    /// `handle_bidi_init`.
    pub fn attach(&self, engine: ServiceEngine<ManyIn<T>, ManyOut<U>>) -> Result<()> {
        self.engine
            .set(engine)
            .map_err(|_| anyhow::anyhow!("BidiIngress: engine already set"))
    }

    /// Convenience: build an ingress + attach in one shot.
    pub fn for_engine(
        engine: ServiceEngine<ManyIn<T>, ManyOut<U>>,
        velo: Arc<Velo>,
    ) -> Result<Arc<Self>> {
        let ingress = Self::new(velo);
        ingress.attach(engine)?;
        Ok(ingress)
    }
}

#[async_trait]
impl<T, U, I> BidiPushWorkHandler for BidiIngress<T, U, I>
where
    T: Data + Serialize + DeserializeOwned,
    U: Data + Serialize + DeserializeOwned,
    I: Data + DeserializeOwned + Clone,
{
    async fn handle_bidi_init(
        &self,
        payload: Bytes,
        _request_id: Option<String>,
    ) -> Result<Bytes, PipelineError> {
        // 1. Decode the init request.
        let req: BidiInitRequest<I> = rmp_serde::from_slice(&payload).map_err(|e| {
            PipelineError::DeserializationError(format!("BidiInitRequest decode: {e}"))
        })?;

        // 2. Create our recv anchor (client -> server data direction).
        //    Set a TTL so a client that vanishes between unary response and
        //    attach doesn't leak the anchor.
        let server_recv: StreamAnchor<BidiFrame<T>> = self.velo.create_anchor::<BidiFrame<T>>();
        server_recv.set_timeout(Some(BIDI_UNATTACHED_TIMEOUT));
        let server_handle = server_recv.handle();

        // 3. Attach our send to the client's recv anchor (server -> client).
        let server_send: StreamSender<BidiFrame<U>> =
            match self.velo.attach_anchor::<BidiFrame<U>>(req.client_handle).await {
                Ok(s) => s,
                Err(e) => {
                    // Drop server_recv; it'll be reaped by unattached_timeout.
                    drop(server_recv);
                    let resp = BidiInitResponse::Err {
                        reason: format!("server attach to client anchor failed: {e}"),
                    };
                    let bytes = rmp_serde::to_vec(&resp).map_err(|e| {
                        PipelineError::Generic(format!(
                            "serialize BidiInitResponse::Err: {e}"
                        ))
                    })?;
                    return Ok(bytes.into());
                }
            };

        // 4. Build session keyed by the request id.
        let session = BidiSession::new(req.request_id.clone());
        let peer_done = session.peer_done();

        // 5. Decode anchor frames into a flat DataStream<T>. The drainer
        //    spawned inside takes ownership of `server_recv` for its full
        //    lifetime, draining past Done until Finalized so the anchor
        //    terminates cleanly without a spurious cancel cascade.
        let data_stream: DataStream<T> =
            decode_bidi_anchor::<T>(server_recv, peer_done, session.controller());

        // 6. Build user-visible ManyIn<T> with the session's controller, and
        //    stash the init payload into the registry under BIDI_INIT_KEY.
        let holder = AsyncRequestStream::new(data_stream);
        let mut many_in: ManyIn<T> = Context::with_controller_arc(holder, session.controller());
        many_in.insert_unique(BIDI_INIT_KEY, req.init);

        // 7. Spawn the user-handler driver. It takes ownership of:
        //    - the user engine (shared via Arc),
        //    - the session (Arc),
        //    - server_send (consumed by run_outgoing).
        let engine = self
            .engine
            .get()
            .cloned()
            .ok_or_else(|| PipelineError::Generic("BidiIngress: engine not attached".into()))?;
        let session_for_task = session.clone();
        tokio::spawn(async move {
            match engine.generate(many_in).await {
                Ok(many_out) => {
                    let stream: DataStream<U> = engine_stream_to_data_stream(many_out);
                    let outcome = session_for_task.run_outgoing(stream, server_send).await;
                    debug!(outcome = ?outcome, "bidi server: session ended");
                }
                Err(e) => {
                    warn!(error = %e, "bidi server: handler returned Err");
                    // Drop server_send -> velo emits a Dropped sentinel to
                    // the client's anchor, which propagates the failure
                    // there. We deliberately do NOT call controller.kill()
                    // here: that would tear down the server_recv drainer
                    // immediately, racing with the client's attach to
                    // server_handle (the client may not have attached yet).
                    // The natural cleanup path:
                    //  - client sees Dropped on its response stream,
                    //  - client's pump finishes its Done/finalize sequence,
                    //  - client's finalize emits Finalized to server_recv,
                    //  - server_recv's drainer reads Finalized and exits
                    //    cleanly (terminated=true, no cascade cancel).
                    drop(server_send);
                }
            }
        });

        // 8. Return our handle to the client so it can attach its sender.
        let resp = BidiInitResponse::Ok { server_handle };
        let bytes = rmp_serde::to_vec(&resp)
            .map_err(|e| PipelineError::Generic(format!("serialize BidiInitResponse::Ok: {e}")))?;
        Ok(bytes.into())
    }
}

/// Adapt an `EngineStream<U>` (user handler return) into a flat
/// `DataStream<U>` for the outgoing pump.
fn engine_stream_to_data_stream<U: Data>(stream: EngineStream<U>) -> DataStream<U> {
    // EngineStream<U> already satisfies Stream<Item = U>; just re-pin under
    // the DataStream type alias.
    Box::pin(stream)
}
