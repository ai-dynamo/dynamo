// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use crate::metrics::MetricsHierarchy;
use crate::protocols::EndpointId;
use crate::traits::events::EventPublisher;

use anyhow::Result;
use async_nats::connection::State;
use async_nats::{Subscriber, client, jetstream};
use async_trait::async_trait;
use bytes::Bytes;
use derive_builder::Builder;
use futures::{StreamExt, TryStreamExt};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use tokio::fs::File as TokioFile;
use tokio::io::AsyncRead;
use tokio::time;
use url::Url;
use validator::{Validate, ValidationError};

use crate::config::environment_names::nats as env_nats;
use crate::metrics::prometheus_names::nats_client as nats_metrics;
pub use crate::slug::Slug;
use tracing as log;

use super::utils::build_in_runtime;

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

    /// Helper method to get or optionally create an object store bucket
    ///
    /// # Arguments
    /// * `bucket_name` - The name of the bucket to retrieve
    /// * `create_if_not_found` - If true, creates the bucket when it doesn't exist
    ///
    /// # Returns
    /// The object store bucket or an error
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        create_if_not_found: bool,
    ) -> anyhow::Result<jetstream::object_store::ObjectStore> {
        let context = self.jetstream();

        match context.get_object_store(bucket_name).await {
            Ok(bucket) => Ok(bucket),
            Err(err) if err.to_string().contains("stream not found") => {
                // err.source() is GetStreamError, which has a kind() which
                // is GetStreamErrorKind::JetStream which wraps a jetstream::Error
                // which has code 404. Phew. So yeah check the string for now.

                if create_if_not_found {
                    tracing::debug!("Creating NATS bucket {bucket_name}");
                    context
                        .create_object_store(jetstream::object_store::Config {
                            bucket: bucket_name.to_string(),
                            ..Default::default()
                        })
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed creating bucket / object store: {e}"))
                } else {
                    anyhow::bail!(
                        "NATS get_object_store bucket does not exist: {bucket_name}. {err}."
                    );
                }
            }
            Err(err) => {
                anyhow::bail!("NATS get_object_store error: {err}");
            }
        }
    }

    /// Upload file to NATS at this URL
    pub async fn object_store_upload(&self, filepath: &Path, nats_url: &Url) -> anyhow::Result<()> {
        let mut disk_file = TokioFile::open(filepath).await?;

        let (bucket_name, key) = url_to_bucket_and_key(nats_url)?;
        let bucket = self.get_or_create_bucket(&bucket_name, true).await?;

        let key_meta = async_nats::jetstream::object_store::ObjectMetadata {
            name: key.to_string(),
            ..Default::default()
        };
        bucket.put(key_meta, &mut disk_file).await.map_err(|e| {
            anyhow::anyhow!("Failed uploading to bucket / object store {bucket_name}/{key}: {e}")
        })?;

        Ok(())
    }

    /// Download file from NATS at this URL
    pub async fn object_store_download(
        &self,
        nats_url: &Url,
        filepath: &Path,
    ) -> anyhow::Result<()> {
        let mut disk_file = TokioFile::create(filepath).await?;

        let (bucket_name, key) = url_to_bucket_and_key(nats_url)?;
        let bucket = self.get_or_create_bucket(&bucket_name, false).await?;

        let mut obj_reader = bucket.get(&key).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed downloading from bucket / object store {bucket_name}/{key}: {e}"
            )
        })?;
        let _bytes_copied = tokio::io::copy(&mut obj_reader, &mut disk_file).await?;

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

    /// Upload a serializable struct to NATS object store using bincode
    pub async fn object_store_upload_data<T>(&self, data: &T, nats_url: &Url) -> anyhow::Result<()>
    where
        T: Serialize,
    {
        // Serialize the data using bincode (more efficient binary format)
        let binary_data = bincode::serialize(data)
            .map_err(|e| anyhow::anyhow!("Failed to serialize data with bincode: {e}"))?;

        let (bucket_name, key) = url_to_bucket_and_key(nats_url)?;
        let bucket = self.get_or_create_bucket(&bucket_name, true).await?;

        let key_meta = async_nats::jetstream::object_store::ObjectMetadata {
            name: key.to_string(),
            ..Default::default()
        };

        // Upload the serialized bytes
        let mut cursor = std::io::Cursor::new(binary_data);
        bucket.put(key_meta, &mut cursor).await.map_err(|e| {
            anyhow::anyhow!("Failed uploading to bucket / object store {bucket_name}/{key}: {e}")
        })?;

        Ok(())
    }

    /// Download and deserialize a struct from NATS object store using bincode
    pub async fn object_store_download_data<T>(&self, nats_url: &Url) -> anyhow::Result<T>
    where
        T: DeserializeOwned,
    {
        let (bucket_name, key) = url_to_bucket_and_key(nats_url)?;
        let bucket = self.get_or_create_bucket(&bucket_name, false).await?;

        let mut obj_reader = bucket.get(&key).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed downloading from bucket / object store {bucket_name}/{key}: {e}"
            )
        })?;

        // Read all bytes into memory
        let mut buffer = Vec::new();
        tokio::io::copy(&mut obj_reader, &mut buffer)
            .await
            .map_err(|e| anyhow::anyhow!("Failed reading object data: {e}"))?;
        tracing::debug!("Downloaded {} bytes from {bucket_name}/{key}", buffer.len());

        // Deserialize from bincode
        let data = bincode::deserialize(&buffer)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize data with bincode: {e}"))?;

        Ok(data)
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
    if let Ok(server) = std::env::var(env_nats::NATS_SERVER) {
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

// TODO(jthomson04): We really shouldn't be hardcoding this.
const NATS_WORKER_THREADS: usize = 4;

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

        let (client, _) = build_in_runtime(
            async move {
                client
                    .connect(self.server)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to connect to NATS: {e}. Verify NATS server is running and accessible."))
            },
            NATS_WORKER_THREADS,
        )
        .await?;

        let js_ctx = jetstream::new(client.clone());

        // Validate JetStream is available
        js_ctx
            .query_account()
            .await
            .map_err(|e| anyhow::anyhow!("JetStream not available: {e}"))?;

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
            std::env::var(env_nats::auth::NATS_AUTH_USERNAME),
            std::env::var(env_nats::auth::NATS_AUTH_PASSWORD),
        ) {
            return NatsAuth::UserPass(username, password);
        }

        if let Ok(token) = std::env::var(env_nats::auth::NATS_AUTH_TOKEN) {
            return NatsAuth::Token(token);
        }

        if let Ok(nkey) = std::env::var(env_nats::auth::NATS_AUTH_NKEY) {
            return NatsAuth::NKey(nkey);
        }

        if let Ok(path) = std::env::var(env_nats::auth::NATS_AUTH_CREDENTIALS_FILE) {
            return NatsAuth::CredentialsFile(PathBuf::from(path));
        }

        NatsAuth::UserPass("user".to_string(), "user".to_string())
    }
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

/// Default queue name for publishing events
pub const QUEUE_NAME: &str = "queue";

/// A queue implementation using NATS JetStream
pub struct NatsQueue {
    /// The name of the stream to use for the queue
    stream_name: String,
    /// The NATS server URL
    nats_server: String,
    /// Timeout for dequeue operations in seconds
    dequeue_timeout: time::Duration,
    /// The NATS client
    client: Option<Client>,
    /// The subject pattern used for this queue
    subject: String,
    /// The subscriber for pull-based consumption
    subscriber: Option<jetstream::consumer::PullConsumer>,
    /// Optional consumer name for broadcast pattern (if None, uses "worker-group")
    consumer_name: Option<String>,
    /// Message stream for efficient message consumption
    message_stream: Option<jetstream::consumer::pull::Stream>,
}

impl NatsQueue {
    /// Create a new NatsQueue with the default "worker-group" consumer
    pub fn new(stream_name: String, nats_server: String, dequeue_timeout: time::Duration) -> Self {
        // Sanitize stream name to remove path separators (like in Python version)
        // rupei: are we sure NATs stream name accepts '_'?
        let sanitized_stream_name = Slug::slugify(&stream_name).to_string();
        let subject = format!("{sanitized_stream_name}.*");

        Self {
            stream_name: sanitized_stream_name,
            nats_server,
            dequeue_timeout,
            client: None,
            subject,
            subscriber: None,
            consumer_name: Some("worker-group".to_string()),
            message_stream: None,
        }
    }

    /// Create a new NatsQueue without a consumer (publisher-only mode)
    pub fn new_without_consumer(
        stream_name: String,
        nats_server: String,
        dequeue_timeout: time::Duration,
    ) -> Self {
        let sanitized_stream_name = Slug::slugify(&stream_name).to_string();
        let subject = format!("{sanitized_stream_name}.*");

        Self {
            stream_name: sanitized_stream_name,
            nats_server,
            dequeue_timeout,
            client: None,
            subject,
            subscriber: None,
            consumer_name: None,
            message_stream: None,
        }
    }

    /// Create a new NatsQueue with a specific consumer name for broadcast pattern
    /// Each consumer with a unique name will receive all messages independently
    pub fn new_with_consumer(
        stream_name: String,
        nats_server: String,
        dequeue_timeout: time::Duration,
        consumer_name: String,
    ) -> Self {
        let sanitized_stream_name = Slug::slugify(&stream_name).to_string();
        let subject = format!("{sanitized_stream_name}.*");

        Self {
            stream_name: sanitized_stream_name,
            nats_server,
            dequeue_timeout,
            client: None,
            subject,
            subscriber: None,
            consumer_name: Some(consumer_name),
            message_stream: None,
        }
    }

    /// Connect to the NATS server and set up the stream and consumer
    pub async fn connect(&mut self) -> Result<()> {
        self.connect_with_reset(false).await
    }

    /// Connect to the NATS server and set up the stream and consumer, optionally resetting the stream
    pub async fn connect_with_reset(&mut self, reset_stream: bool) -> Result<()> {
        if self.client.is_none() {
            // Create a new client
            let client_options = Client::builder().server(self.nats_server.clone()).build()?;

            let client = client_options.connect().await?;

            // messages older than a hour in the stream will be automatically purged
            let max_age = std::env::var(env_nats::stream::DYN_NATS_STREAM_MAX_AGE)
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(time::Duration::from_secs)
                .unwrap_or_else(|| time::Duration::from_secs(60 * 60));

            let stream_config = jetstream::stream::Config {
                name: self.stream_name.clone(),
                subjects: vec![self.subject.clone()],
                max_age,
                ..Default::default()
            };

            // Get or create the stream
            let stream = client
                .jetstream()
                .get_or_create_stream(stream_config)
                .await?;

            log::debug!("Stream {} is ready", self.stream_name);

            // If reset_stream is true, purge all messages from the stream
            if reset_stream {
                match stream.purge().await {
                    Ok(purge_info) => {
                        log::info!(
                            "Successfully purged {} messages from NATS stream {}",
                            purge_info.purged,
                            self.stream_name
                        );
                    }
                    Err(e) => {
                        log::warn!("Failed to purge NATS stream '{}': {e}", self.stream_name);
                    }
                }
            }

            // Create persistent subscriber only if consumer_name is set
            if let Some(ref consumer_name) = self.consumer_name {
                let consumer_config = jetstream::consumer::pull::Config {
                    durable_name: Some(consumer_name.clone()),
                    inactive_threshold: std::time::Duration::from_secs(3600), // 1 hour
                    ..Default::default()
                };

                let subscriber = stream.create_consumer(consumer_config).await?;

                // Create the message stream for efficient consumption
                let message_stream = subscriber.messages().await?;

                self.subscriber = Some(subscriber);
                self.message_stream = Some(message_stream);
            }

            self.client = Some(client);
        }

        Ok(())
    }

    /// Ensure we have an active connection
    pub async fn ensure_connection(&mut self) -> Result<()> {
        if self.client.is_none() {
            self.connect().await?;
        }
        Ok(())
    }

    /// Close the connection when done
    pub async fn close(&mut self) -> Result<()> {
        self.message_stream = None;
        self.subscriber = None;
        self.client = None;
        Ok(())
    }

    /// Shutdown the consumer by deleting it from the stream and closing the connection
    /// This permanently removes the consumer from the server
    ///
    /// If `consumer_name` is provided, that specific consumer will be deleted instead of the
    /// current consumer. This allows deletion of other consumers on the same stream.
    pub async fn shutdown(&mut self, consumer_name: Option<String>) -> Result<()> {
        // Determine which consumer to delete
        let target_consumer = consumer_name.as_ref().or(self.consumer_name.as_ref());

        // Warn if deleting our own consumer via explicit parameter
        if let Some(ref passed_name) = consumer_name
            && self.consumer_name.as_ref() == Some(passed_name)
        {
            log::warn!(
                "Deleting our own consumer '{}' via explicit consumer_name parameter. \
                Consider calling shutdown without arguments instead.",
                passed_name
            );
        }

        if let (Some(client), Some(consumer_to_delete)) = (&self.client, target_consumer) {
            // Get the stream and delete the consumer
            let stream = client.jetstream().get_stream(&self.stream_name).await?;
            stream
                .delete_consumer(consumer_to_delete)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("Failed to delete consumer {}: {}", consumer_to_delete, e)
                })?;
            log::debug!(
                "Deleted consumer {} from stream {}",
                consumer_to_delete,
                self.stream_name
            );
        } else {
            log::debug!(
                "Cannot shutdown consumer: client or target consumer is None (client: {:?}, target_consumer: {:?})",
                self.client.is_some(),
                target_consumer.is_some()
            );
        }

        // Only close the connection if we deleted our own consumer
        if consumer_name.is_none() {
            self.close().await
        } else {
            Ok(())
        }
    }

    /// Count the number of consumers for the stream
    pub async fn count_consumers(&mut self) -> Result<usize> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            let mut stream = client.jetstream().get_stream(&self.stream_name).await?;
            let info = stream.info().await?;
            Ok(info.state.consumer_count)
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// List all consumer names for the stream
    pub async fn list_consumers(&mut self) -> Result<Vec<String>> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            client.list_consumers(&self.stream_name).await
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// Enqueue a task using the provided data
    pub async fn enqueue_task(&mut self, task_data: Bytes) -> Result<()> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            let subject = format!("{}.queue", self.stream_name);
            client.jetstream().publish(subject, task_data).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// Dequeue and return a task as raw bytes
    pub async fn dequeue_task(&mut self, timeout: Option<time::Duration>) -> Result<Option<Bytes>> {
        self.ensure_connection().await?;

        let Some(ref mut stream) = self.message_stream else {
            return Err(anyhow::anyhow!("Message stream not initialized"));
        };

        let timeout_duration = timeout.unwrap_or(self.dequeue_timeout);

        // Try to get next message from the stream with timeout
        let message = tokio::time::timeout(timeout_duration, stream.next()).await;

        match message {
            Ok(Some(Ok(msg))) => {
                msg.ack()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to ack message: {}", e))?;
                Ok(Some(msg.payload.clone()))
            }

            Ok(Some(Err(e))) => Err(anyhow::anyhow!("Failed to get message from stream: {}", e)),

            Ok(None) => Err(anyhow::anyhow!("Message stream ended unexpectedly")),

            // Timeout - no messages available
            Err(_) => Ok(None),
        }
    }

    /// Get the number of messages currently in the queue
    pub async fn get_queue_size(&mut self) -> Result<u64> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            // Get consumer info to get pending messages count
            let stream = client.jetstream().get_stream(&self.stream_name).await?;
            let consumer_name = self
                .consumer_name
                .clone()
                .unwrap_or_else(|| "worker-group".to_string());
            let mut consumer: jetstream::consumer::PullConsumer = stream
                .get_consumer(&consumer_name)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get consumer: {}", e))?;
            let info = consumer.info().await?;

            Ok(info.num_pending)
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// Get the total number of messages currently in the stream
    pub async fn get_stream_messages(&mut self) -> Result<u64> {
        self.ensure_connection().await?;

        if let Some(client) = &self.client {
            let mut stream = client.jetstream().get_stream(&self.stream_name).await?;
            let info = stream.info().await?;
            Ok(info.state.messages)
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// Purge messages from the stream up to (but not including) the specified sequence number
    /// This permanently removes messages and affects all consumers of the stream
    pub async fn purge_up_to_sequence(&self, sequence: u64) -> Result<()> {
        if let Some(client) = &self.client {
            let stream = client.jetstream().get_stream(&self.stream_name).await?;

            // NOTE: this purge excludes the sequence itself
            // https://docs.rs/nats/latest/nats/jetstream/struct.PurgeRequest.html
            stream.purge().sequence(sequence).await.map_err(|e| {
                anyhow::anyhow!("Failed to purge stream up to sequence {}: {}", sequence, e)
            })?;

            log::debug!(
                "Purged stream {} up to sequence {}",
                self.stream_name,
                sequence
            );
            Ok(())
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }

    /// Purge messages from the stream up to the minimum acknowledged sequence across all consumers
    /// This finds the lowest acknowledged sequence number across all consumers and purges up to that point
    pub async fn purge_acknowledged(&mut self) -> Result<()> {
        self.ensure_connection().await?;

        let Some(client) = &self.client else {
            return Err(anyhow::anyhow!("Client not connected"));
        };

        let stream = client.jetstream().get_stream(&self.stream_name).await?;

        // Get all consumer names for the stream
        let consumer_names: Vec<String> = stream
            .consumer_names()
            .try_collect()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to list consumers: {}", e))?;

        if consumer_names.is_empty() {
            log::debug!("No consumers found for stream {}", self.stream_name);
            return Ok(());
        }

        // Find the minimum acknowledged sequence across all consumers
        let mut min_ack_sequence = u64::MAX;

        for consumer_name in &consumer_names {
            let mut consumer: jetstream::consumer::PullConsumer = stream
                .get_consumer(consumer_name)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get consumer {}: {}", consumer_name, e))?;

            let info = consumer.info().await.map_err(|e| {
                anyhow::anyhow!("Failed to get consumer info for {}: {}", consumer_name, e)
            })?;

            // The ack_floor contains the stream sequence of the highest contiguously acknowledged message
            // If stream_sequence is 0, it means no messages have been acknowledged yet
            if info.ack_floor.stream_sequence > 0 {
                min_ack_sequence = min_ack_sequence.min(info.ack_floor.stream_sequence);
                log::debug!(
                    "Consumer {} has ack_floor at sequence {}",
                    consumer_name,
                    info.ack_floor.stream_sequence
                );
            }
        }

        // Only purge if we found a valid minimum acknowledged sequence
        if min_ack_sequence < u64::MAX && min_ack_sequence > 0 {
            // Purge up to (but not including) the minimum acknowledged sequence + 1
            // We add 1 because we want to include the minimum acknowledged message in the purge
            let purge_sequence = min_ack_sequence + 1;

            self.purge_up_to_sequence(purge_sequence).await?;

            log::debug!(
                "Purged stream {} up to acknowledged sequence {} (purged up to sequence {})",
                self.stream_name,
                min_ack_sequence,
                purge_sequence
            );
        } else {
            log::debug!(
                "No messages to purge for stream {} (min_ack_sequence: {})",
                self.stream_name,
                min_ack_sequence
            );
        }

        Ok(())
    }
}

#[async_trait]
impl EventPublisher for NatsQueue {
    fn subject(&self) -> String {
        self.stream_name.clone()
    }

    async fn publish(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        event: &(impl Serialize + Send + Sync),
    ) -> Result<()> {
        let bytes = serde_json::to_vec(event)?;
        self.publish_bytes(event_name, bytes).await
    }

    async fn publish_bytes(
        &self,
        event_name: impl AsRef<str> + Send + Sync,
        bytes: Vec<u8>,
    ) -> Result<()> {
        // We expect the stream to be always suffixed with "queue"
        // This suffix itself is nothing special, just a repo standard
        if event_name.as_ref() != QUEUE_NAME {
            tracing::warn!(
                "Expected event_name to be '{}', but got '{}'",
                QUEUE_NAME,
                event_name.as_ref()
            );
        }

        let subject = format!("{}.{}", self.subject(), event_name.as_ref());

        // Note: enqueue_task requires &mut self, but EventPublisher requires &self
        // We need to ensure the client is connected and use it directly
        if let Some(client) = &self.client {
            client.jetstream().publish(subject, bytes.into()).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Client not connected"))
        }
    }
}

/// Prometheus metrics that mirror the NATS client statistics (in primitive types)
/// to be used for the System Status Server.
///
/// ⚠️  IMPORTANT: These Prometheus Gauges are COPIES of NATS client data, not live references!
///
/// How it works:
/// 1. NATS client provides source data via client.statistics() and connection_state()
/// 2. set_from_client_stats() reads current NATS values and updates these Prometheus Gauges
/// 3. Prometheus scrapes these Gauge values (snapshots, not live data)
///
/// Flow: NATS Client → Client Statistics → set_from_client_stats() → Prometheus Gauge
/// Note: These are snapshots updated when set_from_client_stats() is called.
#[derive(Debug, Clone)]
pub struct DRTNatsClientPrometheusMetrics {
    nats_client: client::Client,
    /// Number of bytes received (excluding protocol overhead)
    pub in_bytes: IntGauge,
    /// Number of bytes sent (excluding protocol overhead)
    pub out_bytes: IntGauge,
    /// Number of messages received
    pub in_messages: IntGauge,
    /// Number of messages sent
    pub out_messages: IntGauge,
    /// Number of times connection was established
    pub connects: IntGauge,
    /// Current connection state (0 = disconnected, 1 = connected, 2 = reconnecting)
    pub connection_state: IntGauge,
}

impl DRTNatsClientPrometheusMetrics {
    /// Create a new instance of NATS client metrics using a DistributedRuntime's Prometheus constructors
    pub fn new(drt: &crate::DistributedRuntime, nats_client: client::Client) -> Result<Self> {
        let metrics = drt.metrics();
        let in_bytes = metrics.create_intgauge(
            nats_metrics::IN_TOTAL_BYTES,
            "Total number of bytes received by NATS client",
            &[],
        )?;
        let out_bytes = metrics.create_intgauge(
            nats_metrics::OUT_OVERHEAD_BYTES,
            "Total number of bytes sent by NATS client",
            &[],
        )?;
        let in_messages = metrics.create_intgauge(
            nats_metrics::IN_MESSAGES,
            "Total number of messages received by NATS client",
            &[],
        )?;
        let out_messages = metrics.create_intgauge(
            nats_metrics::OUT_MESSAGES,
            "Total number of messages sent by NATS client",
            &[],
        )?;
        let connects = metrics.create_intgauge(
            nats_metrics::CURRENT_CONNECTIONS,
            "Current number of active connections for NATS client",
            &[],
        )?;
        let connection_state = metrics.create_intgauge(
            nats_metrics::CONNECTION_STATE,
            "Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)",
            &[],
        )?;

        Ok(Self {
            nats_client,
            in_bytes,
            out_bytes,
            in_messages,
            out_messages,
            connects,
            connection_state,
        })
    }

    /// Copy statistics from the stored NATS client to these Prometheus metrics
    pub fn set_from_client_stats(&self) {
        let stats = self.nats_client.statistics();

        // Get current values from the client statistics
        let in_bytes = stats.in_bytes.load(Ordering::Relaxed);
        let out_bytes = stats.out_bytes.load(Ordering::Relaxed);
        let in_messages = stats.in_messages.load(Ordering::Relaxed);
        let out_messages = stats.out_messages.load(Ordering::Relaxed);
        let connects = stats.connects.load(Ordering::Relaxed);

        // Get connection state
        let connection_state = match self.nats_client.connection_state() {
            State::Connected => 1,
            // treat Disconnected and Pending as "down"
            State::Disconnected | State::Pending => 0,
        };

        // Update Prometheus metrics
        // Using gauges allows us to set absolute values directly
        self.in_bytes.set(in_bytes as i64);
        self.out_bytes.set(out_bytes as i64);
        self.in_messages.set(in_messages as i64);
        self.out_messages.set(out_messages as i64);
        self.connects.set(connects as i64);
        self.connection_state.set(connection_state);
    }
}

/// The NATS subject / inbox to talk to an instance on.
/// TODO: Do we need to sanitize the names?
pub(crate) fn instance_subject(endpoint_id: &EndpointId, instance_id: u64) -> String {
    format!(
        "{}_{}.{}-{:x}",
        endpoint_id.namespace, endpoint_id.component, endpoint_id.name, instance_id,
    )
}

#[cfg(test)]
mod tests {

    use super::*;
    use figment::Jail;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestData {
        id: u32,
        name: String,
        values: Vec<f64>,
    }

    #[test]
    fn test_client_options_builder() {
        Jail::expect_with(|_jail| {
            let opts = ClientOptions::builder().build();
            assert!(opts.is_ok());
            Ok(())
        });

        Jail::expect_with(|jail| {
            jail.set_env(env_nats::NATS_SERVER, "nats://localhost:5222");
            jail.set_env(env_nats::auth::NATS_AUTH_USERNAME, "user");
            jail.set_env(env_nats::auth::NATS_AUTH_PASSWORD, "pass");

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
            jail.set_env(env_nats::NATS_SERVER, "nats://localhost:5222");
            jail.set_env(env_nats::auth::NATS_AUTH_USERNAME, "user");
            jail.set_env(env_nats::auth::NATS_AUTH_PASSWORD, "pass");

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

    // Integration test for object store data operations using bincode
    #[tokio::test]
    #[ignore] // Requires NATS server to be running
    async fn test_object_store_data_operations() {
        // Create test data
        let test_data = TestData {
            id: 42,
            name: "test_item".to_string(),
            values: vec![1.0, 2.5, 3.7, 4.2],
        };

        // Set up client
        let client_options = ClientOptions::builder()
            .server("nats://localhost:4222")
            .build()
            .expect("Failed to build client options");

        let client = client_options
            .connect()
            .await
            .expect("Failed to connect to NATS");

        // Test URL (using .bin extension to indicate binary format)
        let url =
            Url::parse("nats://localhost/test-bucket/test-data.bin").expect("Failed to parse URL");

        // Upload the data
        client
            .object_store_upload_data(&test_data, &url)
            .await
            .expect("Failed to upload data");

        // Download the data
        let downloaded_data: TestData = client
            .object_store_download_data(&url)
            .await
            .expect("Failed to download data");

        // Verify the data matches
        assert_eq!(test_data, downloaded_data);

        // Clean up
        client
            .object_store_delete_bucket("test-bucket")
            .await
            .expect("Failed to delete bucket");
    }

    // Integration test for broadcast pattern with purging
    #[tokio::test]
    #[ignore]
    async fn test_nats_queue_broadcast_with_purge() {
        use uuid::Uuid;

        // Create unique stream name for this test
        let stream_name = format!("test-broadcast-{}", Uuid::new_v4());
        let nats_server = "nats://localhost:4222".to_string();
        let timeout = time::Duration::from_secs(0);

        // Connect to NATS client first to delete stream if it exists
        let client_options = Client::builder()
            .server(nats_server.clone())
            .build()
            .expect("Failed to build client options");

        let client = client_options
            .connect()
            .await
            .expect("Failed to connect to NATS");

        // Delete the stream if it exists (to ensure clean start)
        let _ = client.jetstream().delete_stream(&stream_name).await;

        // Create two consumers with different names for the same stream
        let consumer1_name = format!("consumer-{}", Uuid::new_v4());
        let consumer2_name = format!("consumer-{}", Uuid::new_v4());

        let mut queue1 = NatsQueue::new_with_consumer(
            stream_name.clone(),
            nats_server.clone(),
            timeout,
            consumer1_name,
        );

        // Connect queue1 first (it will create the stream)
        queue1.connect().await.expect("Failed to connect queue1");

        // Send 4 messages using the EventPublisher trait
        let message_strings = [
            "message1".to_string(),
            "message2".to_string(),
            "message3".to_string(),
            "message4".to_string(),
        ];

        // Using the EventPublisher trait to publish messages
        for (idx, msg) in message_strings.iter().enumerate() {
            queue1
                .publish("queue", msg)
                .await
                .unwrap_or_else(|_| panic!("Failed to publish message {}", idx + 1));
        }

        // Convert messages to JSON-serialized Bytes for comparison
        let messages: Vec<Bytes> = message_strings
            .iter()
            .map(|s| Bytes::from(serde_json::to_vec(s).unwrap()))
            .collect();

        // Give JetStream a moment to persist the messages
        tokio::time::sleep(time::Duration::from_millis(100)).await;

        // Now create and connect queue2 and queue3 AFTER messages are published (to test persistence)
        let mut queue2 = NatsQueue::new_with_consumer(
            stream_name.clone(),
            nats_server.clone(),
            timeout,
            consumer2_name,
        );

        // Create a third queue without consumer (publisher-only)
        let mut queue3 =
            NatsQueue::new_without_consumer(stream_name.clone(), nats_server.clone(), timeout);

        // Connect queue2 and queue3 after messages are already published
        queue2.connect().await.expect("Failed to connect queue2");
        queue3.connect().await.expect("Failed to connect queue3");

        // Purge the first two messages (sequence 1 and 2)
        // Note: JetStream sequences start at 1, and purge is exclusive of the sequence number
        queue1
            .purge_up_to_sequence(3)
            .await
            .expect("Failed to purge messages");

        // Give JetStream a moment to process the purge
        tokio::time::sleep(time::Duration::from_millis(100)).await;

        // Consumer 1 dequeues one message (message3)
        let msg3_consumer1 = queue1
            .dequeue_task(Some(time::Duration::from_millis(500)))
            .await
            .expect("Failed to dequeue from queue1");
        assert_eq!(
            msg3_consumer1,
            Some(messages[2].clone()),
            "Consumer 1 should get message3"
        );

        // Give JetStream a moment to process acknowledgments
        tokio::time::sleep(time::Duration::from_millis(100)).await;

        // Now run purge_acknowledged
        // At this point:
        // - Consumer 1 has ack'd message 3 (ack_floor = 3)
        // - Consumer 2 hasn't consumed anything yet (ack_floor = 0)
        // - Min ack_floor = 0, so nothing will be purged
        queue1
            .purge_acknowledged()
            .await
            .expect("Failed to purge acknowledged messages");

        // Give JetStream a moment to process the purge
        tokio::time::sleep(time::Duration::from_millis(100)).await;

        // Now collect remaining messages from both consumers
        let mut consumer1_remaining = Vec::new();
        let mut consumer2_remaining = Vec::new();

        // Collect remaining messages from consumer 1
        while let Some(msg) = queue1
            .dequeue_task(None)
            .await
            .expect("Failed to dequeue from queue1")
        {
            consumer1_remaining.push(msg);
        }

        // Collect remaining messages from consumer 2
        while let Some(msg) = queue2
            .dequeue_task(None)
            .await
            .expect("Failed to dequeue from queue2")
        {
            consumer2_remaining.push(msg);
        }

        // Verify consumer 1 gets 1 remaining message (message4)
        assert_eq!(
            consumer1_remaining.len(),
            1,
            "Consumer 1 should have 1 remaining message"
        );
        assert_eq!(
            consumer1_remaining[0], messages[3],
            "Consumer 1 should get message4"
        );

        // Verify consumer 2 gets 2 messages (message3 and message4)
        assert_eq!(
            consumer2_remaining.len(),
            2,
            "Consumer 2 should have 2 messages"
        );
        assert_eq!(
            consumer2_remaining[0], messages[2],
            "Consumer 2 should get message3"
        );
        assert_eq!(
            consumer2_remaining[1], messages[3],
            "Consumer 2 should get message4"
        );

        // Test consumer count and shutdown behavior
        // First verify via consumer 1 that there are two consumers
        let consumer_count = queue1
            .count_consumers()
            .await
            .expect("Failed to count consumers");
        assert_eq!(consumer_count, 2, "Should have 2 consumers initially");

        // Close consumer 1 and verify via consumer 2 that there are still two consumers
        queue1.close().await.expect("Failed to close queue1");

        let consumer_count = queue2
            .count_consumers()
            .await
            .expect("Failed to count consumers");
        assert_eq!(
            consumer_count, 2,
            "Should still have 2 consumers after closing queue1"
        );

        // Reconnect queue1 to be able to shutdown
        queue1.connect().await.expect("Failed to reconnect queue1");

        // Shutdown consumer 1 and verify via consumer 2 that there is only one consumer left
        queue1
            .shutdown(None)
            .await
            .expect("Failed to shutdown queue1");

        let consumer_count = queue2
            .count_consumers()
            .await
            .expect("Failed to count consumers");
        assert_eq!(
            consumer_count, 1,
            "Should have only 1 consumer after shutting down queue1"
        );

        // Clean up by deleting the stream
        client
            .jetstream()
            .delete_stream(&stream_name)
            .await
            .expect("Failed to delete test stream");
    }
}
