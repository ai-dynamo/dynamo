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

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::Mutex;

use ::velo::{Context as VeloContext, Handler, UnifiedResponse, Velo};

use super::unified_server::RequestPlaneServer;
use super::*;
use crate::SystemHealth;
use crate::pipeline::network::PushWorkHandler;
use crate::pipeline::network::velo::{
    ENDPOINT_HEADER, REQUEST_ID_HEADER, REQUEST_PLANE_HANDLER, encode_velo_node_prefix,
};

/// Velo `RequestPlaneServer` implementation.
///
/// Holds a clone of the process-wide [`Velo`] handle and the demux table.
pub struct VeloRequestPlaneServer {
    velo: Arc<Velo>,
    handlers: Arc<DashMap<String, EndpointHandler>>,
}

#[derive(Clone)]
struct EndpointHandler {
    service_handler: Arc<dyn PushWorkHandler>,
    endpoint_name: String,
    system_health: Arc<Mutex<SystemHealth>>,
}

impl VeloRequestPlaneServer {
    /// Construct a new server bound to the given velo instance and register the
    /// single multiplexed velo handler. Idempotent registration is the caller's
    /// responsibility — invoke this exactly once per process (the `NetworkManager`
    /// global takes care of that).
    pub fn new(velo: Arc<Velo>) -> Result<Arc<Self>> {
        let handlers: Arc<DashMap<String, EndpointHandler>> = Arc::new(DashMap::new());

        let dispatch_handlers = handlers.clone();
        let handler = Handler::unary_handler_async(REQUEST_PLANE_HANDLER, move |ctx: VeloContext| {
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

    /// Cloned reference to the underlying velo instance (mainly for tests).
    pub fn velo(&self) -> &Arc<Velo> {
        &self.velo
    }

    fn endpoint_key(instance_id: u64, endpoint_name: &str) -> String {
        format!("{instance_id:x}/{endpoint_name}")
    }
}

/// The single demux handler. Reads the demux key from headers, looks up the
/// corresponding [`PushWorkHandler`] in the DashMap, and invokes it.
async fn dispatch(
    handlers: Arc<DashMap<String, EndpointHandler>>,
    ctx: VeloContext,
) -> UnifiedResponse {
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
        .ok_or_else(|| anyhow!("no handler registered for velo endpoint key {endpoint_key}"))?;

    handler
        .handle_payload(ctx.payload, request_id)
        .await
        .map_err(|e| anyhow!("dynamo handler for {endpoint_key} failed: {e}"))?;

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
        let key = Self::endpoint_key(instance_id, &endpoint_name);
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
