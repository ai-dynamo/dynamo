// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-backed implementation of [`RequestPlaneClient`].
//!
//! The client is stateless aside from a peer-registration dedupe map. Each
//! `send_request` parses the `velo://...` address, ensures the peer is known
//! to the local velo instance (one-shot `discover_and_register_peer` per
//! `InstanceId`), forwards the demux key as a header, and issues a velo
//! `unary` call. The returned ACK [`Bytes`] is bubbled back to the caller —
//! streaming responses flow over the dedicated TCP `ResponseService`, not
//! through this client.

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;

use ::velo::{InstanceId, Velo};

use super::unified_client::{Headers, RequestPlaneClient};
use crate::pipeline::network::velo::{ENDPOINT_HEADER, REQUEST_PLANE_HANDLER, decode_velo_address};

/// Velo `RequestPlaneClient` implementation.
pub struct VeloRequestPlaneClient {
    velo: Arc<Velo>,
    /// Peers we have already asked velo to resolve and connect. We dedupe to
    /// avoid hammering discovery on every `send_request`.
    known_peers: Arc<DashMap<InstanceId, ()>>,
}

impl VeloRequestPlaneClient {
    /// Create a new client backed by the given velo instance.
    pub fn new(velo: Arc<Velo>) -> Arc<Self> {
        Arc::new(Self {
            velo,
            known_peers: Arc::new(DashMap::new()),
        })
    }

    async fn ensure_peer(&self, target: InstanceId) -> Result<()> {
        if target == self.velo.instance_id() {
            // Local dispatch — velo will short-circuit transport.
            return Ok(());
        }
        if self.known_peers.contains_key(&target) {
            return Ok(());
        }
        self.velo
            .discover_and_register_peer(target)
            .await
            .with_context(|| format!("velo discover_and_register_peer({target})"))?;
        self.known_peers.insert(target, ());
        Ok(())
    }
}

#[async_trait]
impl RequestPlaneClient for VeloRequestPlaneClient {
    async fn send_request(
        &self,
        address: String,
        payload: Bytes,
        mut headers: Headers,
    ) -> Result<Bytes> {
        let parsed = decode_velo_address(&address)
            .with_context(|| format!("parsing velo address {address}"))?;

        self.ensure_peer(parsed.velo_instance).await?;

        // Inject the demux key so the velo handler can route to the right
        // Dynamo `PushWorkHandler` on the server side.
        headers.insert(ENDPOINT_HEADER.to_string(), parsed.endpoint_key.clone());

        let response = self
            .velo
            .unary(REQUEST_PLANE_HANDLER)
            .map_err(|e| anyhow!("creating velo unary builder for {REQUEST_PLANE_HANDLER}: {e}"))?
            .raw_payload(payload)
            .headers(headers)
            .instance(parsed.velo_instance)
            .send()
            .await
            .with_context(|| {
                format!(
                    "velo unary send to {} (key {})",
                    parsed.velo_instance, parsed.endpoint_key
                )
            })?;

        Ok(response)
    }

    fn transport_name(&self) -> &'static str {
        "velo"
    }

    fn is_healthy(&self) -> bool {
        // velo doesn't currently expose a top-level health flag. The peer-level
        // health is observable via `Velo::available_handlers` per InstanceId,
        // which is too expensive for a hot-path probe; treat the client as
        // healthy whenever it has been constructed.
        true
    }
}
