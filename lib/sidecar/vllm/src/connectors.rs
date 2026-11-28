// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-connector PD-coordination strategies for the vLLM sidecar.
//!
//! vLLM's KV connectors disagree on who builds the prefill->decode handoff:
//! NixlConnector is pull-based (the engine's terminal response carries the
//! decode-side params), while MooncakeConnector is push-based (the routing
//! layer allocates a per-request `transfer_id` up front and publishes the
//! decode-side address itself). This module isolates each contract behind
//! [`KvConnectorProtocol`], mirroring `MooncakeConnectorProtocol` in
//! `components/src/dynamo/vllm/kv_connector_protocols.py`.
//!
//! The decode path is connector-agnostic in the sidecar (the handoff travels
//! through `prefill_result.disaggregated_params` untouched), so the protocol
//! only engages on prefill workers.

use async_trait::async_trait;
use dynamo_backend_common::{DisaggregationMode, DynamoError};
use dynamo_sidecar_common::cannot_connect;
use serde_json::{Map, Value};
use tokio::time::Instant;

use crate::client;

/// Per-request connector state. The connector set is closed (registered at
/// compile time), so this stays a concrete enum rather than a trait object.
#[derive(Clone, Debug)]
pub(crate) enum RequestHandoff {
    None,
    Mooncake(MooncakeHandoff),
}

/// Identity resolved once at startup. Nixl needs none; Mooncake carries the
/// bootstrap address and the engine's Mooncake `engine_id`.
#[derive(Clone, Debug)]
pub(crate) enum ResolvedConnector {
    None,
    Mooncake(MooncakeBootstrap),
}

/// One PD-coordination strategy for a vLLM KV connector.
///
/// Hooks 1-2 run outside the per-response hot path; hooks 3-4 are cheap
/// (a map insert and an enum match respectively).
#[async_trait]
pub(crate) trait KvConnectorProtocol: Send + Sync {
    /// The vLLM connector name, for logging and diagnostics.
    fn name(&self) -> &'static str;

    /// Hook 1, startup: resolve the connector's identity. Network I/O is only
    /// ever allowed here, never per request or per response.
    async fn resolve(&self, _deadline: Instant) -> Result<ResolvedConnector, DynamoError> {
        Ok(ResolvedConnector::None)
    }

    /// Hook 2, request admission on a prefill worker: allocate this request's
    /// handoff state. Only called for prefill-mode requests whose startup
    /// resolution succeeded.
    fn new_request(&self, _resolved: &ResolvedConnector) -> RequestHandoff {
        RequestHandoff::None
    }

    /// Hook 3, request build on a prefill worker: inject this connector's
    /// fields into the caller's `kv_transfer_params`.
    fn inject_prefill_params(&self, _handoff: &RequestHandoff, params: &mut Map<String, Value>) {
        params.insert("do_remote_decode".to_string(), Value::Bool(true));
    }

    /// Hook 4, prefill terminal: produce the `disaggregated_params` published
    /// to the decode worker. Returning the engine's params is the pull-based
    /// default; a push-based connector assembles the handoff itself.
    fn prefill_terminal_params(
        &self,
        _handoff: &RequestHandoff,
        engine: Option<Value>,
    ) -> Option<Value> {
        engine
    }
}

/// Pull-based: the engine's terminal response carries the decode-side params,
/// so the default hooks (passthrough) are exactly the right behavior.
#[derive(Clone)]
pub(crate) struct NixlConnectorProtocol;

impl KvConnectorProtocol for NixlConnectorProtocol {
    fn name(&self) -> &'static str {
        "NixlConnector"
    }
}

/// Push-based: vLLM never generates a `transfer_id` and never returns
/// `kv_transfer_params` from a producer's terminal response, so this
/// connector allocates the transfer_id per request and publishes the
/// decode-side handoff itself.
#[derive(Clone)]
pub(crate) struct MooncakeConnectorProtocol {
    /// Host the decode worker uses to reach this pod's bootstrap server.
    host: String,
    /// Bootstrap server port, shared with the engine's
    /// VLLM_MOONCAKE_BOOTSTRAP_PORT.
    port: u16,
}

impl MooncakeConnectorProtocol {
    pub(crate) fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
}

#[async_trait]
impl KvConnectorProtocol for MooncakeConnectorProtocol {
    fn name(&self) -> &'static str {
        "MooncakeConnector"
    }

    async fn resolve(&self, deadline: Instant) -> Result<ResolvedConnector, DynamoError> {
        let engine_id = resolve_mooncake_engine_id(self.port, deadline).await?;
        Ok(ResolvedConnector::Mooncake(MooncakeBootstrap::new(
            format!("http://{}:{}", self.host, self.port),
            engine_id,
        )))
    }

    fn new_request(&self, resolved: &ResolvedConnector) -> RequestHandoff {
        match resolved {
            ResolvedConnector::Mooncake(bootstrap) => {
                RequestHandoff::Mooncake(bootstrap.new_handoff())
            }
            ResolvedConnector::None => RequestHandoff::None,
        }
    }

    fn inject_prefill_params(&self, handoff: &RequestHandoff, params: &mut Map<String, Value>) {
        params.insert("do_remote_decode".to_string(), Value::Bool(true));
        let RequestHandoff::Mooncake(handoff) = handoff else {
            return;
        };
        params.insert("do_remote_prefill".to_string(), Value::Bool(false));
        params.insert(
            "transfer_id".to_string(),
            handoff.transfer_id.clone().into(),
        );
    }

    fn prefill_terminal_params(
        &self,
        handoff: &RequestHandoff,
        _engine: Option<Value>,
    ) -> Option<Value> {
        // A Mooncake producer's terminal never carries kv_transfer_params;
        // publish the decode-side handoff allocated when the request was
        // built instead.
        match handoff {
            RequestHandoff::Mooncake(handoff) => Some(handoff.decode_params()),
            RequestHandoff::None => None,
        }
    }
}

/// Resolve the protocol for the engine's configured KV connector, keyed by
/// `KVTransferConfig.kv_connector`. An unknown name is a misconfiguration,
/// not a benign default: silently falling back to NIXL would emit the wrong
/// wire shape and surface as opaque decode failures.
pub(crate) fn make_kv_connector(
    name: &str,
    mode: DisaggregationMode,
    mooncake_bootstrap_host: Option<String>,
    mooncake_bootstrap_port: u16,
) -> Result<Box<dyn KvConnectorProtocol>, DynamoError> {
    match name {
        "NixlConnector" | "NeuronNixlConnector" => {
            if mooncake_bootstrap_host.is_some() {
                tracing::warn!(
                    "--mooncake-bootstrap-host is ignored without `--kv-connector MooncakeConnector`"
                );
            }
            Ok(Box::new(NixlConnectorProtocol))
        }
        "MooncakeConnector" => {
            if !mode.is_prefill() {
                tracing::warn!(
                    mode = %mode,
                    "`--kv-connector MooncakeConnector` only changes prefill workers; this mode needs no handoff changes"
                );
                return Ok(Box::new(NixlConnectorProtocol));
            }
            let host = mooncake_bootstrap_host
                .filter(|host| !host.is_empty())
                .ok_or_else(|| {
                    client::invalid_argument(
                        "`--kv-connector MooncakeConnector` in prefill mode requires \
                         --mooncake-bootstrap-host (or DYN_VLLM_MOONCAKE_BOOTSTRAP_HOST), \
                         typically injected from status.podIP",
                    )
                })?;
            Ok(Box::new(MooncakeConnectorProtocol::new(
                host,
                mooncake_bootstrap_port,
            )))
        }
        other => Err(client::invalid_argument(format!(
            "unsupported --kv-connector `{other}`; supported: NixlConnector, MooncakeConnector"
        ))),
    }
}

/// This worker's resolved Mooncake bootstrap identity.
///
/// The `engine_id` is read from the local bootstrap server because vLLM
/// never reports it over gRPC; the decode worker validates it against its
/// own `GET <remote_bootstrap_addr>/query` before pulling KV.
#[derive(Clone, Debug)]
pub(crate) struct MooncakeBootstrap {
    /// Address the decode worker queries for this worker's bootstrap server,
    /// e.g. `http://10.0.0.1:8998`.
    pub remote_bootstrap_addr: String,
    /// The engine's Mooncake engine_id, read from its local bootstrap server.
    pub engine_id: String,
}

impl MooncakeBootstrap {
    pub(crate) fn new(remote_bootstrap_addr: String, engine_id: String) -> Self {
        Self {
            remote_bootstrap_addr,
            engine_id,
        }
    }

    /// Per-request Mooncake state: a fresh transfer_id bound to this worker.
    fn new_handoff(&self) -> MooncakeHandoff {
        MooncakeHandoff {
            transfer_id: uuid::Uuid::new_v4().to_string(),
            remote_bootstrap_addr: self.remote_bootstrap_addr.clone(),
            remote_engine_id: self.engine_id.clone(),
        }
    }
}

/// One prefill request's Mooncake handoff, threaded from request build to the
/// terminal response so both carry the same transfer_id.
#[derive(Clone, Debug)]
pub(crate) struct MooncakeHandoff {
    pub transfer_id: String,
    remote_bootstrap_addr: String,
    remote_engine_id: String,
}

impl MooncakeHandoff {
    /// The decode-side kv_transfer_params the prefill terminal publishes.
    /// Matches the engine-side contract: decode validates transfer_id,
    /// remote_bootstrap_addr, and remote_engine_id before pulling KV.
    fn decode_params(&self) -> Value {
        serde_json::json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "transfer_id": self.transfer_id,
            "remote_bootstrap_addr": self.remote_bootstrap_addr,
            "remote_engine_id": self.remote_engine_id,
        })
    }
}

const MOONCAKE_QUERY_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Read the engine's Mooncake engine_id from its local bootstrap server.
///
/// The bootstrap server starts with the engine and answers `GET /query`
/// with it. Retry like the gRPC startup wait because the server can lag
/// gRPC readiness.
async fn resolve_mooncake_engine_id(
    port: u16,
    startup_deadline: Instant,
) -> Result<String, DynamoError> {
    let http = reqwest::Client::new();
    let query = format!("http://127.0.0.1:{port}/query");
    let mut last_error = String::new();
    while Instant::now() < startup_deadline {
        match query_mooncake_engine_id(&http, &query).await {
            Ok(engine_id) => return Ok(engine_id),
            Err(error) => {
                tracing::debug!(%query, %error, "Mooncake bootstrap server is not ready; retrying");
                last_error = error;
                tokio::time::sleep(MOONCAKE_QUERY_RETRY_INTERVAL).await;
            }
        }
    }
    Err(cannot_connect(format!(
        "Mooncake bootstrap server at {query} did not report an engine_id \
         before the startup deadline: {last_error}"
    )))
}

async fn query_mooncake_engine_id(http: &reqwest::Client, query: &str) -> Result<String, String> {
    let body: Value = http
        .get(query)
        .send()
        .await
        .and_then(|response| response.error_for_status())
        .map_err(|error| error.to_string())?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    // vLLM answers with the engines keyed by global rank, e.g.
    // {"0": {"engine_id": ..., "worker_addr": ...}}. Accept a flat
    // {"engine_id": ...} shape too.
    body.get("engine_id")
        .and_then(Value::as_str)
        .or_else(|| {
            body.as_object()?
                .values()
                .find_map(|entry| entry.get("engine_id").and_then(Value::as_str))
        })
        .filter(|engine_id| !engine_id.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("response is missing a non-empty engine_id: {body}"))
}
