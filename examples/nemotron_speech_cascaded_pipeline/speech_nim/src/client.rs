// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::time::Duration;

use anyhow::Context;
use tokio::time::{Instant, sleep, timeout_at};
use tonic::{
    Request,
    metadata::MetadataMap,
    transport::{Certificate, Channel, ClientTlsConfig, Endpoint},
};

use crate::config::ConnectionConfig;

/// Backend-only transport; contains no Dynamo or OpenAI request mapping.
#[derive(Clone)]
pub struct SpeechNimConnection {
    pub channel: Channel,
    metadata: MetadataMap,
}

impl SpeechNimConnection {
    pub async fn connect(config: &ConnectionConfig) -> anyhow::Result<Self> {
        // Dynamo's dependency graph enables both rustls providers. Tonic 0.13
        // needs an explicit process default; preserve one chosen by an embedder.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let tls = config.nim_use_ssl || config.nim_ssl_root_cert.is_some();
        let server = config.nim_server.trim();
        let address = if server.contains("://") {
            server.to_owned()
        } else {
            format!("{}://{server}", if tls { "https" } else { "http" })
        };
        let url = url::Url::parse(&address).context("invalid NIM server URL")?;
        anyhow::ensure!(
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
                && url.path() == "/",
            "NIM server must be an HTTP(S) host and port without credentials or a path"
        );
        anyhow::ensure!(
            !tls || url.scheme() == "https",
            "TLS configuration requires an HTTPS NIM endpoint"
        );
        let mut endpoint = Endpoint::from_shared(address)?.connect_timeout(Duration::from_secs(5));
        if tls || url.scheme() == "https" {
            let mut options = ClientTlsConfig::new().with_native_roots();
            if let Some(path) = &config.nim_ssl_root_cert {
                options =
                    options.ca_certificate(Certificate::from_pem(tokio::fs::read(path).await?));
            }
            endpoint = endpoint.tls_config(options)?;
        }
        let mut metadata = MetadataMap::new();
        if let Some(key) = config.nim_api_key.as_deref().filter(|key| !key.is_empty()) {
            let mut authorization: tonic::metadata::MetadataValue<_> = format!("Bearer {key}")
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid NIM authorization metadata"))?;
            authorization.set_sensitive(true);
            metadata.insert("authorization", authorization);
        }
        if let Some(id) = config
            .nim_function_id
            .as_deref()
            .filter(|id| !id.is_empty())
        {
            metadata.insert(
                "function-id",
                id.parse().context("invalid NIM function-id metadata")?,
            );
        }
        let deadline = Instant::now()
            .checked_add(config.nim_startup_timeout_s)
            .context("NIM startup timeout is out of range")?;
        let channel = timeout_at(deadline, async {
            loop {
                match endpoint.connect().await {
                    Ok(channel) => break channel,
                    Err(error) => {
                        tracing::debug!(%error, "Waiting for the Speech NIM gRPC channel");
                        sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        })
        .await
        .context("Speech NIM gRPC channel did not become ready before the startup deadline")?;
        Ok(Self { channel, metadata })
    }

    pub fn request<T>(&self, message: T) -> Request<T> {
        let mut request = Request::new(message);
        *request.metadata_mut() = self.metadata.clone();
        request
    }
}
