// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-backed implementation of [`RequestPlaneServer`].
//!
//! Velo lacks a `unregister_handler` API, so we register exactly two velo
//! handlers at construction time:
//!
//! - `dynamo-request-plane` — the unary path; demuxes to per-Dynamo-endpoint
//!   [`PushWorkHandler`]s via an internal `DashMap` keyed by
//!   `"{instance_id_hex}/{endpoint_name}"`.
//! - `dynamo-bidi-init` — the bidi handshake path; demuxes via the same
//!   key into [`BidiPushWorkHandler`]s. The velo unary ACK carries the
//!   server's [`StreamAnchorHandle`] back to the client.
//!
//! Endpoints register through either [`register_endpoint`] (unary) or
//! [`register_bidi_endpoint`] (bidi); the demux table stores both kinds via
//! [`HandlerKind`].

use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::Mutex;

use ::velo::{Context as VeloContext, Handler, UnifiedResponse, Velo};

use super::bidi_handler::BidiPushWorkHandler;
use super::unified_server::RequestPlaneServer;
use super::*;
use crate::SystemHealth;
use crate::pipeline::network::PushWorkHandler;
use crate::pipeline::network::bidi::BIDI_INIT_HANDLER;
use crate::pipeline::network::velo::{
    ENDPOINT_HEADER, REQUEST_ID_HEADER, REQUEST_PLANE_HANDLER, encode_velo_node_prefix,
};

/// Velo `RequestPlaneServer` implementation.
///
/// Holds a clone of the process-wide [`Velo`] handle and the demux table.
/// Both unary and bidi handlers live in the same `DashMap` keyed by
/// `{instance_id_hex}/{endpoint_name}`; [`HandlerKind`] discriminates.
pub struct VeloRequestPlaneServer {
    velo: Arc<Velo>,
    handlers: Arc<DashMap<String, EndpointHandler>>,
}

#[derive(Clone)]
struct EndpointHandler {
    kind: HandlerKind,
    endpoint_name: String,
    system_health: Arc<Mutex<SystemHealth>>,
}

#[derive(Clone)]
enum HandlerKind {
    Unary(Arc<dyn PushWorkHandler>),
    Bidi(Arc<dyn BidiPushWorkHandler>),
}

impl VeloRequestPlaneServer {
    /// Construct a new server bound to the given velo instance and register
    /// the two multiplexed velo handlers (unary + bidi-init). Idempotent
    /// registration is the caller's responsibility — invoke this exactly
    /// once per process (the `NetworkManager` global takes care of that).
    pub fn new(velo: Arc<Velo>) -> Result<Arc<Self>> {
        let handlers: Arc<DashMap<String, EndpointHandler>> = Arc::new(DashMap::new());

        // ---- Unary handler: returns Ok(None) -> empty velo ACK. ----
        let dispatch_handlers = handlers.clone();
        let unary_handler =
            Handler::unary_handler_async(REQUEST_PLANE_HANDLER, move |ctx: VeloContext| {
                let handlers = dispatch_handlers.clone();
                async move { dispatch_unary(handlers, ctx).await }
            })
            .build();
        velo.register_handler(unary_handler).map_err(|e| {
            anyhow!("registering velo demux handler {REQUEST_PLANE_HANDLER}: {e}")
        })?;

        // ---- Bidi-init handler: returns Ok(Some(BidiInitResponse bytes)). ----
        let dispatch_handlers_bidi = handlers.clone();
        let bidi_handler =
            Handler::unary_handler_async(BIDI_INIT_HANDLER, move |ctx: VeloContext| {
                let handlers = dispatch_handlers_bidi.clone();
                async move { dispatch_bidi(handlers, ctx).await }
            })
            .build();
        velo.register_handler(bidi_handler)
            .map_err(|e| anyhow!("registering velo demux handler {BIDI_INIT_HANDLER}: {e}"))?;

        tracing::info!(
            velo_instance = %velo.instance_id().as_uuid(),
            unary_handler = REQUEST_PLANE_HANDLER,
            bidi_handler = BIDI_INIT_HANDLER,
            "VeloRequestPlaneServer ready"
        );

        Ok(Arc::new(Self { velo, handlers }))
    }

    /// Cloned reference to the underlying velo instance (mainly for tests).
    pub fn velo(&self) -> &Arc<Velo> {
        &self.velo
    }

    fn endpoint_key(instance_id: u64, endpoint_name: &str) -> String {
        format!("{instance_id:x}/{endpoint_name}")
    }
}

/// Look up the EndpointHandler for the demux key in `ctx.headers`.
fn lookup<'a>(
    handlers: &'a DashMap<String, EndpointHandler>,
    headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<(EndpointHandler, Option<String>)> {
    let headers = headers.ok_or_else(|| {
        anyhow!("velo request missing headers (need `{ENDPOINT_HEADER}`)")
    })?;
    let endpoint_key = headers
        .get(ENDPOINT_HEADER)
        .cloned()
        .ok_or_else(|| anyhow!("velo request missing `{ENDPOINT_HEADER}` header"))?;
    let request_id = headers.get(REQUEST_ID_HEADER).cloned();
    let entry = handlers
        .get(&endpoint_key)
        .map(|e| e.value().clone())
        .ok_or_else(|| anyhow!("no handler registered for velo endpoint key {endpoint_key}"))?;
    Ok((entry, request_id))
}

/// Demux dispatch for the unary `dynamo-request-plane` handler.
async fn dispatch_unary(
    handlers: Arc<DashMap<String, EndpointHandler>>,
    ctx: VeloContext,
) -> UnifiedResponse {
    let (entry, request_id) = lookup(&handlers, ctx.headers.as_ref())?;
    match entry.kind {
        HandlerKind::Unary(h) => {
            h.handle_payload(ctx.payload, request_id)
                .await
                .map_err(|e| anyhow!("dynamo unary handler for {} failed: {e}", entry.endpoint_name))?;
            // ACK with empty payload — streaming responses ride the dedicated
            // TCP `ResponseService`.
            Ok(None)
        }
        HandlerKind::Bidi(_) => Err(anyhow!(
            "endpoint {} is registered as bidi; received via unary path",
            entry.endpoint_name
        )),
    }
}

/// Demux dispatch for the bidi-init `dynamo-bidi-init` handler.
async fn dispatch_bidi(
    handlers: Arc<DashMap<String, EndpointHandler>>,
    ctx: VeloContext,
) -> UnifiedResponse {
    let (entry, request_id) = lookup(&handlers, ctx.headers.as_ref())?;
    match entry.kind {
        HandlerKind::Bidi(b) => {
            let resp_bytes = b
                .handle_bidi_init(ctx.payload, request_id)
                .await
                .map_err(|e| anyhow!("dynamo bidi handler for {} failed: {e}", entry.endpoint_name))?;
            Ok(Some(resp_bytes))
        }
        HandlerKind::Unary(_) => Err(anyhow!(
            "endpoint {} is registered as unary; received via bidi-init path",
            entry.endpoint_name
        )),
    }
}

#[async_trait]
impl RequestPlaneServer for VeloRequestPlaneServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let key = Self::endpoint_key(instance_id, &endpoint_name);
        let fqn = format!("{namespace}.{component_name}.{endpoint_name}");

        self.handlers.insert(
            key.clone(),
            EndpointHandler {
                kind: HandlerKind::Unary(service_handler),
                endpoint_name: endpoint_name.clone(),
                system_health: system_health.clone(),
            },
        );

        system_health.lock().set_endpoint_registered(&endpoint_name);

        tracing::info!(
            fqn = %fqn,
            endpoint_key = %key,
            kind = "unary",
            velo_instance = %self.velo.instance_id().as_uuid(),
            "Registered endpoint with velo request plane"
        );
        Ok(())
    }

    async fn register_bidi_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn BidiPushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let key = Self::endpoint_key(instance_id, &endpoint_name);
        let fqn = format!("{namespace}.{component_name}.{endpoint_name}");

        self.handlers.insert(
            key.clone(),
            EndpointHandler {
                kind: HandlerKind::Bidi(service_handler),
                endpoint_name: endpoint_name.clone(),
                system_health: system_health.clone(),
            },
        );

        system_health.lock().set_endpoint_registered(&endpoint_name);

        tracing::info!(
            fqn = %fqn,
            endpoint_key = %key,
            kind = "bidi",
            velo_instance = %self.velo.instance_id().as_uuid(),
            "Registered endpoint with velo request plane"
        );
        Ok(())
    }

    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()> {
        let suffix = format!("/{endpoint_name}");
        let keys: Vec<String> = self
            .handlers
            .iter()
            .filter(|e| e.key().ends_with(&suffix))
            .map(|e| e.key().clone())
            .collect();

        for key in keys {
            if let Some((_, h)) = self.handlers.remove(&key) {
                h.system_health
                    .lock()
                    .set_endpoint_health_status(&h.endpoint_name, crate::HealthStatus::NotReady);
                tracing::info!(
                    endpoint_name = %h.endpoint_name,
                    endpoint_key = %key,
                    "Unregistered velo endpoint handler"
                );
            }
        }
        Ok(())
    }

    fn address(&self) -> String {
        encode_velo_node_prefix(self.velo.instance_id())
    }

    fn transport_name(&self) -> &'static str {
        "velo"
    }

    fn is_healthy(&self) -> bool {
        // The velo node is considered healthy as long as it has been built. velo
        // itself does not currently expose a top-level health flag; if we add
        // one in the future this is the place to wire it.
        true
    }
}

// Hint to the linker that the type alias [`Bytes`] is used in places where we
// only mention it transitively (rmp_serde decode of payload bytes lives in
// the bidi handler, but signatures here pass `Bytes` through).
#[allow(dead_code)]
fn _bytes_alias_marker(_: Bytes) {}
