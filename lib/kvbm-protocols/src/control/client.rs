// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public velo client for the leader control plane (`--features client`).
//!
//! [`LeaderControlClient`] targets a single leader instance and exposes one
//! sub-client per control module — `client.core()`,
//! `client.transfer()` — plus the always-on `list_modules` query. Each
//! sub-client's methods speak the protocol types defined alongside it.

use std::sync::Arc;

use bytes::Bytes;
use serde::Serialize;
use serde::de::DeserializeOwned;
use velo::Messenger;
use velo_ext::InstanceId;

use super::modules::metrics::MetricsClient;
use super::modules::transfer::TransferClient;
use super::{
    ControlError, ControlReply, DESCRIBE_INSTANCE_HANDLER, DescribeInstanceRequest,
    InstanceDescription, LIST_MODULES_HANDLER, ListModulesRequest, ListModulesResponse, ModuleId,
    RESET_HANDLER, ResetRequest, ResetResponse,
};

/// A velo unary channel to one leader instance, shared by every sub-client.
///
/// `call` serializes the request, sends it to `handler` on `instance_id`,
/// decodes the [`ControlReply`] envelope, and unwraps it.
#[derive(Clone)]
pub(crate) struct ControlChannel {
    messenger: Arc<Messenger>,
    instance_id: InstanceId,
}

impl ControlChannel {
    pub(crate) async fn call<Req, Resp>(
        &self,
        handler: &str,
        req: &Req,
    ) -> Result<Resp, ControlError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let bytes = serde_json::to_vec(req)
            .map_err(|e| ControlError::Internal(format!("encode {handler} request: {e}")))?;
        let raw: Bytes = self
            .messenger
            .unary(handler)
            .map_err(|e| ControlError::Internal(format!("build {handler} call: {e:#}")))?
            .raw_payload(Bytes::from(bytes))
            .instance(self.instance_id)
            .send()
            .await
            .map_err(|e| ControlError::Internal(format!("{handler} transport: {e:#}")))?;
        let reply: ControlReply<Resp> = serde_json::from_slice(&raw)
            .map_err(|e| ControlError::Internal(format!("decode {handler} reply: {e}")))?;
        reply.into_result()
    }
}

/// Public client for one leader's control plane.
#[derive(Clone)]
pub struct LeaderControlClient {
    chan: ControlChannel,
}

impl LeaderControlClient {
    /// Create a client targeting `instance_id` over `messenger`.
    pub fn new(messenger: Arc<Messenger>, instance_id: InstanceId) -> Self {
        Self {
            chan: ControlChannel {
                messenger,
                instance_id,
            },
        }
    }

    /// The instance this client targets.
    pub fn instance_id(&self) -> InstanceId {
        self.chan.instance_id
    }

    /// Sub-client for the always-on `core` module (`describe_instance`).
    pub fn core(&self) -> CoreClient {
        CoreClient {
            chan: self.chan.clone(),
        }
    }

    /// Sub-client for the opt-in `dev` module (`reset`).
    pub fn dev(&self) -> DevClient {
        DevClient {
            chan: self.chan.clone(),
        }
    }

    /// Sub-client for the always-on `transfer` module (`search_prefix`,
    /// `search_scatter`).
    pub fn transfer(&self) -> TransferClient {
        TransferClient::new(self.chan.clone())
    }

    /// Sub-client for the opt-in `metrics` module (`snapshot`).
    pub fn metrics(&self) -> MetricsClient {
        MetricsClient::new(self.chan.clone())
    }

    /// Ask the leader which control modules are enabled on this instance.
    pub async fn list_modules(&self) -> Result<Vec<ModuleId>, ControlError> {
        let resp: ListModulesResponse = self
            .chan
            .call(LIST_MODULES_HANDLER, &ListModulesRequest::default())
            .await?;
        Ok(resp.modules)
    }

    /// Generic passthrough for handlers without a typed binding.
    ///
    /// Returns the raw [`ControlReply`] envelope so the caller can decide
    /// whether to unwrap to `Ok(JsonValue)` or surface the [`ControlError`].
    /// Used by the hub for forward-compat HTTP routes that proxy bytes through
    /// without knowing the schema.
    ///
    /// An empty `payload` is coerced to `{}` — control handlers expect a JSON
    /// object even for "default" cases (e.g. `ResetRequest::default()`).
    pub async fn call_raw(
        &self,
        handler: &str,
        payload: Bytes,
    ) -> Result<ControlReply<serde_json::Value>, ControlError> {
        let payload = if payload.is_empty() {
            Bytes::from_static(b"{}")
        } else {
            payload
        };
        let raw: Bytes = self
            .chan
            .messenger
            .unary(handler)
            .map_err(|e| ControlError::Internal(format!("build {handler} call: {e:#}")))?
            .raw_payload(payload)
            .instance(self.chan.instance_id)
            .send()
            .await
            .map_err(|e| ControlError::Internal(format!("{handler} transport: {e:#}")))?;
        serde_json::from_slice(&raw)
            .map_err(|e| ControlError::Internal(format!("decode {handler} reply: {e}")))
    }
}

/// Client for the always-on `core` control module.
#[derive(Clone)]
pub struct CoreClient {
    chan: ControlChannel,
}

impl CoreClient {
    /// Pull the leader's structured topology snapshot.
    ///
    /// **Fallback path.** In steady state the leader pushes
    /// [`InstanceDescription`] to the hub; the hub's cache is the read source
    /// of truth. This call is used by the hub when its cache is cold (after
    /// hub restart) or when an operator triggers a forced re-fetch.
    pub async fn describe(&self) -> Result<InstanceDescription, ControlError> {
        self.chan
            .call(
                DESCRIBE_INSTANCE_HANDLER,
                &DescribeInstanceRequest::default(),
            )
            .await
    }
}

/// Client for the opt-in `dev` control module.
#[derive(Clone)]
pub struct DevClient {
    chan: ControlChannel,
}

impl DevClient {
    /// Reset the inactive pools of the requested (or all configured) tiers.
    pub async fn reset(&self, req: ResetRequest) -> Result<ResetResponse, ControlError> {
        self.chan.call(RESET_HANDLER, &req).await
    }
}
