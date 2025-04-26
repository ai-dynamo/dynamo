// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! NATS transport
//!
//! The following environment variables are used to configure the NATS client:
//!
//! - `NATS_SERVER`: the NATS server address
//!
//! For authentication, the following environment variables are used and prioritized in the following order:
//!
//! - `NATS_AUTH_USERNAME`: the username for authentication
//! - `NATS_AUTH_PASSWORD`: the password for authentication
//! - `NATS_AUTH_TOKEN`: the token for authentication
//! - `NATS_AUTH_NKEY`: the nkey for authentication
//! - `NATS_AUTH_CREDENTIALS_FILE`: the path to the credentials file
//!
//! Note: `NATS_AUTH_USERNAME` and `NATS_AUTH_PASSWORD` must be used together.
use crate::Result;

use async_nats::{client, jetstream, Subscriber};
use bytes::Bytes;
use derive_builder::Builder;
use futures::{StreamExt, TryStreamExt};
use std::path::{Path, PathBuf};
use tokio::fs::File as TokioFile;
use tokio::time;
use url::Url;
use validator::{Validate, ValidationError};

pub use crate::slug::Slug;
use tracing as log;

pub const URL_PREFIX: &str = "nats://";

#[derive(Clone)]
pub struct Client {
    client: client::Client,
    js_ctx: jetstream::Context,
}

impl Client {
    /// Create a NATS [`ClientOptionsBuilder`].
    pub fn builder() -> ClientOptionsBuilder {
        ClientOptionsBuilder::default()
    }

    /// Returns a reference to the underlying [`async_nats::client::Client`] instance
    pub fn client(&self) -> &client::Client {
        &self.client
    }

    /// Returns a reference to the underlying [`async_nats::jetstream::Context`] instance
    pub fn jetstream(&self) -> &jetstream::Context {
        &self.js_ctx
    }

    /// host:port of NATS
    pub fn addr(&self) -> String {
        let info = self.client.server_info();
        format!("{}:{}", info.host, info.port)
    }

    /// fetch the list of streams
    pub async fn list_streams(&self) -> Result<Vec<String>> {
        let names = self.js_ctx.stream_names();
        let stream_names: Vec<String> = names.try_collect().await?;
        Ok(stream_names)
    }

    /// fetch the list of consumers for a given stream
    pub async fn list_consumers(&self, stream_name: &str) -> Result<Vec<String>> {
        let stream = self.js_ctx.get_stream(stream_name).await?;
        let consumers: Vec<String> = stream.consumer_names().try_collect().await?;
        Ok(consumers)
    }

    pub async fn stream_info(&self, stream_name: &str) -> Result<jetstream::stream::State> {
        let mut stream = self.js_ctx.get_stream(stream_name).await?;
        let info = stream.info().await?;
        Ok(info.state.clone())
    }

    pub async fn get_stream(&self, name: &str) -> Result<jetstream::stream::Stream> {
        let stream = self.js_ctx.get_stream(name).await?;
        Ok(stream)
    }

    /// Issues a broadcast request for all services with the provided `service_name` to report their
    /// current stats. Each service will only respond once. The service may have customized the reply
    /// so the caller should select which endpoint and what concrete data model should be used to
    /// extract the details.
    ///
    /// Note: Because each endpoint will only reply once, the caller must drop the subscription after
    /// some time or it will await forever.
    pub async fn scrape_service(&self, service_name: &str) -> Result<Subscriber> {
        let subject = format!("$SRV.STATS.{}", service_name);
        let reply_subject = format!("_INBOX.{}", nuid::next());
        let subscription = self.client.subscribe(reply_subject.clone()).await?;

        // Publish the request with the reply-to subject
        self.client
            .publish_with_reply(subject, reply_subject, "".into())
            .await?;

        Ok(subscription)
    }

    /// Upload file to NATS at this URL
    pub async fn object_store_upload(&self, filepath: &Path, nats_url: Url) -> anyhow::Result<()> {
        let mut disk_file = TokioFile::open(filepath).await?;

        let (bucket_name, key) = url_to_bucket_and_key(&nats_url)?;
        let context = self.jetstream();

        let bucket = match context.get_object_store(&bucket_name).await {
            Ok(bucket) => bucket,
            Err(err) if err.to_string().contains("stream not found") => {
                // err.source() is GetStreamError, which has a kind() which
                // is GetStreamErrorKind::JetStream which wraps a jetstream::Error
                // which has code 404. Phew. So yeah check the string for now.

                tracing::debug!("Creating NATS bucket {bucket_name}");
                context
                    .create_object_store(jetstream::object_store::Config {
                        bucket: bucket_name.to_string(),
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed creating bucket / object store: {e}"))?
            }
            Err(err) => {
                anyhow::bail!("NATS get_object_store error: {err}");
            }
        };

        let key_meta = async_nats::jetstream::object_store::ObjectMetadata {
            name: key.to_string(),
            ..Default::default()
        };
        bucket.put(key_meta, &mut disk_file).await.map_err(|e| {
            anyhow::anyhow!("Failed uploading to bucket / object store {bucket_name}/{key}: {e}")
        })?;

        Ok(())
    }

    /// Delete a bucket and all it's contents from the NATS object store
    pub async fn object_store_delete_bucket(&self, bucket_name: &str) -> anyhow::Result<()> {
        let context = self.jetstream();
        match context.delete_object_store(&bucket_name).await {
            Ok(_) => Ok(()),
            Err(err) if err.to_string().contains("stream not found") => {
                tracing::trace!(bucket_name, "NATS bucket already gone");
                Ok(())
            }
            Err(err) => Err(anyhow::anyhow!("NATS get_object_store error: {err}")),
        }
    }
}

/// NATS client options
///
/// This object uses the builder pattern with default values that are evaluates
/// from the environment variables if they are not explicitly set by the builder.
#[derive(Debug, Clone, Builder, Validate)]
pub struct ClientOptions {
    #[builder(setter(into), default = "default_server()")]
    #[validate(custom(function = "validate_nats_server"))]
    server: String,

    #[builder(default)]
    auth: NatsAuth,
}

fn default_server() -> String {
    if let Ok(server) = std::env::var("NATS_SERVER") {
        return server;
    }

    "nats://localhost:4222".to_string()
}

fn validate_nats_server(server: &str) -> Result<(), ValidationError> {
    if server.starts_with("nats://") {
        Ok(())
    } else {
        Err(ValidationError::new("server must start with 'nats://'"))
    }
}

#[allow(dead_code)]
impl ClientOptions {
    /// Create a new [`ClientOptionsBuilder`]
    pub fn builder() -> ClientOptionsBuilder {
        ClientOptionsBuilder::default()
    }

    /// Validate the config and attempt to connection to the NATS server
    pub async fn connect(self) -> Result<Client> {
        self.validate()?;

        let client = match self.auth {
            NatsAuth::UserPass(username, password) => {
                async_nats::ConnectOptions::with_user_and_password(username, password)
            }
            NatsAuth::Token(token) => async_nats::ConnectOptions::with_token(token),
            NatsAuth::NKey(nkey) => async_nats::ConnectOptions::with_nkey(nkey),
            NatsAuth::CredentialsFile(path) => {
                async_nats::ConnectOptions::with_credentials_file(path).await?
            }
        };

        let client = client.connect(self.server).await?;
        let js_ctx = jetstream::new(client.clone());

        Ok(Client { client, js_ctx })
    }
}

impl Default for ClientOptions {
    fn default() -> Self {
        ClientOptions {
            server: default_server(),
            auth: NatsAuth::default(),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum NatsAuth {
    UserPass(String, String),
    Token(String),
    NKey(String),
    CredentialsFile(PathBuf),
}

impl std::fmt::Debug for NatsAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NatsAuth::UserPass(user, _pass) => {
                write!(f, "UserPass({}, <redacted>)", user)
            }
            NatsAuth::Token(_token) => write!(f, "Token(<redacted>)"),
            NatsAuth::NKey(_nkey) => write!(f, "NKey(<redacted>)"),
            NatsAuth::CredentialsFile(path) => write!(f, "CredentialsFile({:?})", path),
        }
    }
}

impl Default for NatsAuth {
    fn default() -> Self {
        if let (Ok(username), Ok(password)) = (
            std::env::var("NATS_AUTH_USERNAME"),
            std::env::var("NATS_AUTH_PASSWORD"),
        ) {
            return NatsAuth::UserPass(username, password);
        }

        if let Ok(token) = std::env::var("NATS_AUTH_TOKEN") {
            return NatsAuth::Token(token);
        }

        if let Ok(nkey) = std::env::var("NATS_AUTH_NKEY") {
            return NatsAuth::NKey(nkey);
        }

        if let Ok(path) = std::env::var("NATS_AUTH_CREDENTIALS_FILE") {
            return NatsAuth::CredentialsFile(PathBuf::from(path));
        }

        NatsAuth::UserPass("user".to_string(), "user".to_string())
    }
}

/// Is this file name / url in the NATS object store?
/// Checks the name only, does not go to the store.
pub fn is_nats_url(s: &str) -> bool {
    s.starts_with(URL_PREFIX)
}

/// Extract NATS bucket and key from a nats URL of the form:
/// nats://host[:port]/bucket/key
pub fn url_to_bucket_and_key(url: &Url) -> anyhow::Result<(String, String)> {
    let Some(mut path_segments) = url.path_segments() else {
        anyhow::bail!("No path in NATS URL: {url}");
    };
    let Some(bucket) = path_segments.next() else {
        anyhow::bail!("No bucket in NATS URL: {url}");
    };
    let Some(key) = path_segments.next() else {
        anyhow::bail!("No key in NATS URL: {url}");
    };
    Ok((bucket.to_string(), key.to_string()))
}

#[cfg(test)]
mod tests {

    use super::*;
    use figment::Jail;

    #[test]
    fn test_client_options_builder() {
        Jail::expect_with(|_jail| {
            let opts = ClientOptions::builder().build();
            assert!(opts.is_ok());
            Ok(())
        });

        Jail::expect_with(|jail| {
            jail.set_env("NATS_SERVER", "nats://localhost:5222");
            jail.set_env("NATS_AUTH_USERNAME", "user");
            jail.set_env("NATS_AUTH_PASSWORD", "pass");

            let opts = ClientOptions::builder().build();
            assert!(opts.is_ok());
            let opts = opts.unwrap();

            assert_eq!(opts.server, "nats://localhost:5222");
            assert_eq!(
                opts.auth,
                NatsAuth::UserPass("user".to_string(), "pass".to_string())
            );

            Ok(())
        });

        Jail::expect_with(|jail| {
            jail.set_env("NATS_SERVER", "nats://localhost:5222");
            jail.set_env("NATS_AUTH_USERNAME", "user");
            jail.set_env("NATS_AUTH_PASSWORD", "pass");

            let opts = ClientOptions::builder()
                .server("nats://localhost:6222")
                .auth(NatsAuth::Token("token".to_string()))
                .build();
            assert!(opts.is_ok());
            let opts = opts.unwrap();

            assert_eq!(opts.server, "nats://localhost:6222");
            assert_eq!(opts.auth, NatsAuth::Token("token".to_string()));

            Ok(())
        });
    }
}
