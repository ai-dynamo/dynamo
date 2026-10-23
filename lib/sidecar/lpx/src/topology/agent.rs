// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reads the allocation leader's early OpenEngine ServerInfo extension.
#![allow(clippy::result_large_err)]
use super::{payload::decode_payload, proto::Snapshot};
use crate::pb::{GetServerInfoRequest, control_client::ControlClient};
use std::{path::Path, time::Duration};
use tonic::{
    Status,
    codegen::http::Uri,
    transport::{Channel, Endpoint},
};

#[derive(Clone)]
pub(crate) struct AgentClient {
    client: ControlClient<Channel>,
}

impl AgentClient {
    pub(crate) fn new(socket: &Path) -> Result<Self, String> {
        let socket = socket
            .to_str()
            .ok_or("the Agent OpenEngine socket is not UTF-8")?;
        let endpoint = Endpoint::from_shared(format!("unix://{socket}"))
            .map_err(|error| format!("invalid Agent OpenEngine socket: {error}"))?
            .origin(Uri::from_static("http://localhost"));
        Ok(Self {
            client: ControlClient::new(endpoint.connect_lazy()),
        })
    }

    pub(crate) async fn get_snapshot(
        &self,
        timeout: Duration,
    ) -> Result<(String, Snapshot), Status> {
        let mut client = self.client.clone();
        let info = tokio::time::timeout(timeout, client.get_server_info(GetServerInfoRequest {}))
            .await
            .map_err(|_| Status::unavailable("Agent ServerInfo timed out"))?
            .map_err(|status| {
                if std::error::Error::source(&status).is_some() {
                    Status::unavailable(format!("Agent ServerInfo unreachable: {status}"))
                } else {
                    status
                }
            })?
            .into_inner();
        let value = info
            .extra
            .and_then(|mut extra| extra.fields.remove("lpx_topology"))
            .ok_or_else(|| Status::unavailable("Agent has no complete lpx_topology yet"))?;
        let Some(prost_types::value::Kind::StringValue(payload)) = value.kind else {
            return Err(Status::failed_precondition(
                "lpx_topology must be a base64 string",
            ));
        };
        let snapshot = decode_payload(&payload)?;
        Ok((payload, snapshot))
    }
}
