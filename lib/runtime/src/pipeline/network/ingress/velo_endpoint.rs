// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-backed implementation of [`RequestPlaneServer`].
//!
//! Velo lacks a `unregister_handler` API, so we register exactly one velo handler
//! (`dynamo-request-plane`) at construction time and demux to per-Dynamo-endpoint
//! [`PushWorkHandler`]s via an internal `DashMap` keyed by
//! `"{instance_id_hex}/{endpoint_name}"`. This matches the routing key format used
//! by [`SharedTcpServer`].

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::Mutex;

use ::velo::{Context as VeloContext, Handler, Velo};

use super::unified_server::RequestPlaneServer;
use super::*;
use crate::SystemHealth;
use crate::pipeline::network::PushWorkHandler;
use crate::pipeline::network::velo::{
    ENDPOINT_HEADER, REQUEST_ID_HEADER, REQUEST_PLANE_HANDLER, encode_velo_node_prefix,
    endpoint_key,
};

/// Velo `RequestPlaneServer` implementation.
///
/// Holds a clone of the process-wide [`Velo`] handle and the demux table.
pub struct VeloRequestPlaneServer {
    velo: Arc<Velo>,
    handlers: Arc<DashMap<String, EndpointHandler>>,
}

struct EndpointHandler {
    service_handler: Arc<dyn PushWorkHandler>,
    endpoint_name: String,
    system_health: Arc<Mutex<SystemHealth>>,
}

impl VeloRequestPlaneServer {
    /// Construct a new server bound to the given velo instance and register the
    /// single multiplexed velo handler.
    ///
    /// Must be called **at most once per process**: velo's `register_handler`
    /// rejects a duplicate handler name, so a second call returns an error.
    /// `NetworkManager` upholds this by building the server through a
    /// `OnceCell` (and the velo instance itself through `GLOBAL_VELO`).
    pub fn new(velo: Arc<Velo>) -> Result<Arc<Self>> {
        let handlers: Arc<DashMap<String, EndpointHandler>> = Arc::new(DashMap::new());

        let dispatch_handlers = handlers.clone();
        let handler =
            Handler::unary_handler_async(REQUEST_PLANE_HANDLER, move |ctx: VeloContext| {
                let handlers = dispatch_handlers.clone();
                async move { dispatch(handlers, ctx).await }
            })
            .build();

        velo.register_handler(handler)
            .map_err(|e| anyhow!("registering velo demux handler {REQUEST_PLANE_HANDLER}: {e}"))?;

        tracing::info!(
            velo_instance = %velo.instance_id().as_uuid(),
            handler = REQUEST_PLANE_HANDLER,
            "VeloRequestPlaneServer ready"
        );

        Ok(Arc::new(Self { velo, handlers }))
    }
}

/// The single demux handler. Reads the demux key from headers, looks up the
/// corresponding [`PushWorkHandler`] in the DashMap, and invokes it.
///
/// The velo handler return type is `anyhow::Result<Option<bytes::Bytes>>`
/// (velo's `UnifiedResponse`): `Ok(None)` is an empty ACK, `Err(_)` propagates
/// back to the caller.
async fn dispatch(
    handlers: Arc<DashMap<String, EndpointHandler>>,
    ctx: VeloContext,
) -> anyhow::Result<Option<bytes::Bytes>> {
    let headers = match ctx.headers.as_ref() {
        Some(h) => h,
        None => {
            return Err(anyhow!(
                "velo request missing headers (need `{ENDPOINT_HEADER}`)"
            ));
        }
    };
    let endpoint_key = match headers.get(ENDPOINT_HEADER) {
        Some(k) => k.clone(),
        None => return Err(anyhow!("velo request missing `{ENDPOINT_HEADER}` header")),
    };
    let request_id = headers.get(REQUEST_ID_HEADER).cloned();

    let handler = handlers
        .get(&endpoint_key)
        .map(|entry| entry.service_handler.clone())
        .ok_or_else(|| {
            // Log on the server so operators can see misrouted / misspelled
            // requests without having to correlate from the client error.
            tracing::warn!(
                endpoint_key = %endpoint_key,
                "velo dispatch: no handler registered for endpoint key"
            );
            anyhow!("no handler registered for velo endpoint key {endpoint_key}")
        })?;

    handler
        .handle_payload(ctx.payload, request_id)
        .await
        .with_context(|| format!("dynamo handler for {endpoint_key} failed"))?;

    // ACK with empty payload — streaming responses flow over the dedicated
    // `ResponseService` (TCP), not over the velo unary path.
    Ok(None)
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
        let key = endpoint_key(instance_id, &endpoint_name);
        let fqn = format!("{namespace}.{component_name}.{endpoint_name}");

        self.handlers.insert(
            key.clone(),
            EndpointHandler {
                service_handler,
                endpoint_name: endpoint_name.clone(),
                system_health: system_health.clone(),
            },
        );

        system_health.lock().set_endpoint_registered(&endpoint_name);

        tracing::info!(
            fqn = %fqn,
            endpoint_key = %key,
            velo_instance = %self.velo.instance_id().as_uuid(),
            "Registered endpoint with velo request plane"
        );
        Ok(())
    }

    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()> {
        // The `RequestPlaneServer` trait only passes `endpoint_name` (no
        // instance_id), so we match every demux key ending in `/{name}`. This
        // mirrors `SharedTcpServer` / `SharedHttpServer` and is safe because a
        // Dynamo process registers a single `instance_id`, so at most one key
        // per endpoint name exists.
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
        // TODO(KVBM-424 follow-up): velo does not yet expose a top-level health
        // flag. When it does, wire it in here so liveness probes / LB gates
        // reflect actual node state instead of unconditional `true`.
        true
    }
}
