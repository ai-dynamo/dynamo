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
use std::path::PathBuf;
use tokio::time;
use validator::{Validate, ValidationError};

pub use crate::slug::Slug;
use tracing as log;

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

    // /// create a new stream
    // async fn get_or_create_work_queue_stream(
    //     &self,
    //     name: &super::Namespace,
    // ) -> Result<jetstream::stream::Stream> {
    //     let stream = self
    //         .js_ctx
    //         .get_or_create_stream(async_nats::jetstream::stream::Config {
    //             name: name.to_string(),
    //             retention: async_nats::jetstream::stream::RetentionPolicy::WorkQueue,
    //             subjects: vec![format!("{name}.>")],
    //             ..Default::default()
    //         })
    //         .await?;
    //     Ok(stream)
    // }

    // // get work queue
    // pub async fn get_or_create_work_queue(
    //     &self,
    //     namespace: &super::Namespace,
    //     queue_name: &Slug,
    // ) -> Result<WorkQueue> {
    //     let stream = self.get_or_create_work_queue_stream(namespace).await?;

    //     let consumer_name = single_name(namespace, queue_name);
    //     let subject_name = subject_name(namespace, queue_name);
    //     let subject_name = format!("{}.*", subject_name);

    //     tracing::trace!(
    //         durable_name = consumer_name,
    //         filter_subject = subject_name,
    //         "get_or_create_work_queue"
    //     );
    //     let consumer = stream
    //         .get_or_create_consumer(
    //             &consumer_name,
    //             jetstream::consumer::pull::Config {
    //                 durable_name: Some(consumer_name.clone()),
    //                 filter_subject: subject_name,
    //                 ack_policy: jetstream::consumer::AckPolicy::Explicit,
    //                 ..Default::default()
    //             },
    //         )
    //         .await?;
    //     Ok(WorkQueue::new(consumer))
    // }

    // pub async fn get_or_create_work_queue_publisher(
    //     &self,
    //     namespace: &super::Namespace,
    //     queue_name: &Slug,
    // ) -> Result<WorkQueuePublisher> {
    //     let _stream = self.get_or_create_work_queue_stream(namespace).await?;
    //     let _subject = subject_name(namespace, queue_name);
    //     Ok(WorkQueuePublisher {
    //         client: self.clone(),
    //         namespace: namespace.clone(),
    //         queue_name: queue_name.clone(),
    //     })
    // }

    // pub async fn list_work_queues(
    //     &self,
    //     namespace: &super::Namespace,
    // ) -> Result<Vec<String>> {
    //     let stream = self.get_stream(namespace.as_ref()).await?;
    //     let consumers: Vec<String> = stream.consumer_names().try_collect().await?;
    //     Ok(consumers)
    // }

    // /// remove a work queue
    // pub async fn remove_work_queue(
    //     &self,
    //     namespace: &super::Namespace,
    //     queue_name: &Slug,
    // ) -> Result<()> {
    //     let stream = self.get_stream(namespace.as_ref()).await?;
    //     let consumer_name = single_name(namespace, queue_name);
    //     let consumers = self.list_consumers(namespace.as_ref()).await?;
    //     if consumers.contains(&consumer_name) {
    //         stream.delete_consumer(&consumer_name).await?;
    //     }
    //     Ok(())
    // }

    // /// publish a message to a subject
    // pub async fn publish(&self, subject: String, msg: Vec<u8>) -> Result<()> {
    //     self.client.publish(subject, msg.into()).await?;
    //     Ok(())
    // }

    // /// subscribe to a subject
    // pub async fn subscribe(
    //     &self,
    //     subject: String,
    // ) -> Result<async_nats::Subscriber> {
    //     let sub = self.client.subscribe(subject).await?;
    //     Ok(sub)
    // }

    // pub async fn enqueue(
    //     &self,
    //     namespace: &super::Namespace,
    //     queue_name: &Slug,
    //     payload: Bytes,
    // ) -> Result<String> {
    //     // let mut headers = HeaderMap::new();
    //     let subject = subject_name(namespace, queue_name);
    //     let request_id = uuid::Uuid::new_v4().to_string();
    //     let subject = format!("{}.{}", subject, request_id);

    //     self.client.publish(subject, payload).await?;

    //     // self.client
    //     //     .publish_with_headers(subject, headers, payload.into())
    //     //     .await?;

    //     Ok(request_id)
    // }

    // pub async fn enqueue_with_id(
    //     &self,
    //     namespace: &super::Namespace,
    //     queue_name: &Slug,
    //     request_id: &str,
    //     payload: Vec<u8>,
    // ) -> Result<()> {
    //     let subject = subject_name(namespace, queue_name);
    //     let subject = format!("{}.{}", subject, request_id);

    //     self.client.publish(subject, payload.into()).await?;
    //     Ok(())
    // }

    // pub fn frontend_client(&self, request_id: String) -> SpecializedClient {
    //     SpecializedClient::new(self.client.clone(), ClientKind::Frontend, request_id)
    // }

    // pub fn backend_client(&self, request_id: String) -> SpecializedClient {
    //     SpecializedClient::new(self.client.clone(), ClientKind::Backend, request_id)
    // }
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

    // const TEST_STREAM: &str = "test_async_nats_stream";

    // #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    // struct Request {
    //     id: String,
    // }

    // async fn nats_client() -> Result<Client> {
    //     Client::builder()
    //         .server("nats://localhost:4222")
    //         .username("user")
    //         .password("user")
    //         .build()
    //         .await
    // }

    // #[tokio::test]
    // async fn test_list_streams() {
    //     let client = match nats_client().await.ok() {
    //         Some(client) => client,
    //         None => {
    //             println!("Failed to create client; skipping nats tests");
    //             return;
    //         }
    //     };

    //     let streams = client.list_streams().await.expect("failed to list streams");

    //     for stream in streams {
    //         let info = client
    //             .stream_info(&stream)
    //             .await
    //             .expect("failed to get stream info");
    //         assert_eq!(info.messages, 0, "stream {} not empty", stream);
    //     }
    // }

    // #[tokio::test]
    // async fn test_workq_pull_and_response_stream() {
    //     let ns: Namespace = TEST_STREAM.try_into().unwrap();
    //     let _client = match nats_client().await.ok() {
    //         Some(client) => client,
    //         None => {
    //             println!("Failed to create client; skipping nats tests");
    //             return;
    //         }
    //     };

    //     let client = Client::builder()
    //         .server("nats://localhost:4222")
    //         .username("user")
    //         .password("user")
    //         .build()
    //         .await
    //         .expect("failed to create client");

    //     let _streams = client.list_streams().await.expect("failed to list streams");
    //     // assert!(!streams.contains(&TEST_STREAM.to_string()));

    //     let _stream = client
    //         .get_or_create_work_queue_stream(&ns)
    //         .await
    //         .expect("failed to create stream");

    //     let model_name: Slug = "foo".try_into().unwrap();
    //     let request_id = "bar";

    //     let request = Request {
    //         id: request_id.to_string(),
    //     };

    //     let request_payload = serde_json::to_vec(&request).expect("failed to serialize request");

    //     //     let request = CompletionRequest {
    //     //         prompt: CompletionContext::from_prompt("deep learning is".to_string()).into(),
    //     //         stop_conditions: None,
    //     //         sampling_options: None,
    //     //     };

    //     // remove work queue if it exists
    //     client
    //         .remove_work_queue(&ns, &model_name)
    //         .await
    //         .expect("remove work queue does not fail if queue does not exist");

    //     // get the count of the work queues
    //     let initial_work_queue_count = client
    //         .list_work_queues(&ns)
    //         .await
    //         .expect("failed to list work queues")
    //         .len();

    //     // create work queue
    //     let workq = client
    //         .get_or_create_work_queue(&ns, &model_name)
    //         .await
    //         .expect("failed to get work queue");

    //     // new work queue count
    //     let work_queue_count = client
    //         .list_work_queues(&ns)
    //         .await
    //         .expect("failed to list work queues")
    //         .len();

    //     assert_eq!(initial_work_queue_count, work_queue_count - 1);

    //     client
    //         .enqueue(&ns, &model_name, request_payload.into())
    //         .await
    //         .expect("failed to enqueue completion request");

    //     let mut messages = workq
    //         .pull(1, std::time::Duration::from_secs(1))
    //         .await
    //         .expect("failed to pull messages from work queue");

    //     assert_eq!(1, messages.len());

    //     let msg = messages.pop().expect("no message received");
    //     msg.ack().await.expect("failed to ack");

    //     let request: Request =
    //         serde_json::from_slice(&msg.payload).expect("failed to deserialize message");

    //     assert_eq!(request.id, request_id);

    //     // clean up and delete nats work queue and stream
    //     client
    //         .remove_work_queue(&ns, &model_name)
    //         .await
    //         .expect("failed to remove work queue");

    //     // client
    //     //     .delete_stream(TEST_STREAM)
    //     //     .await
    //     //     .expect("failed to delete stream");
    // }
}
// let frontend_client = client.frontend_client("test".to_string());

//     // the represents the frontend response subscription
//     let mut frontend_sub = frontend_client
//         .subscribe()
//         .await
//         .expect("failed to subscribe");

//     let backend_client = client.backend_client("test".to_string());

//     let mut backend_sub = backend_client
//         .subscribe()
//         .await
//         .expect("failed to subscribe");

//     let msg = messages[0].clone();
//     let req = serde_json::from_slice::<CompletionRequest>(&msg.payload)
//         .expect("failed to deserialize message");

//     msg.ack().await.expect("failed to ack");

//     assert_eq!(req.prompt, request.prompt);

//     // ping pong message between backend and frontend

//     // backend publishes to frontend
//     backend_client
//         .publish(&MessageKind::Initialize(Prologue {
//             formatted_prompt: None,
//             input_token_ids: None,
//         }))
//         .await
//         .expect("failed to publish");

//     // frontend receives initialize message
//     let msg = frontend_sub.next().await.expect("msg not received");
//     let msg = serde_json::from_slice::<MessageKind>(&msg.payload)
//         .expect("failed to deserialize message");

//     match msg {
//         MessageKind::Initialize(_) => {}
//         _ => panic!("unexpected message"),
//     }

//     // frontend publishes to backend
//     frontend_client
//         .publish(&MessageKind::Finalize(Epilogue {}))
//         .await
//         .expect("failed to publish");

//     // backend receives finalize message
//     let msg = backend_sub.next().await.expect("msg not received");
//     let msg = serde_json::from_slice::<MessageKind>(&msg.payload)
//         .expect("failed to deserialize message");

//     match &msg {
//         MessageKind::Finalize(_) => {}
//         _ => panic!("unexpected message"),
//     }

//     // delete the work queue
//     client
//         .remove_work_queue(model_name, TEST_STREAM)
//         .await
//         .expect("failed to remove work queue");

//     // new work queue count
//     let work_queue_count = client
//         .list_work_queues(TEST_STREAM)
//         .await
//         .expect("failed to list work queues")
//         .len();

//     // compare against the initial work queue count
//     assert_eq!(initial_work_queue_count, work_queue_count);
// }

// async fn connect(config: Arc<Config>) -> Result<NatsClient> {
//     let client = ClientOptions::builder()
//         .server(config.nats_address.clone())
//         .build()
//         .await
//         .context("Creating NATS Client")?;

//     Ok(client)
// }

// async fn create_service(
//     nats: NatsClient,
//     config: Arc<Config>,
//     observer: ServiceObserver,
// ) -> Result<NatsService> {
//     let service = nats
//         .client()
//         .service_builder()
//         .description(config.service_description.as_str())
//         .stats_handler(move |_name, _stats| {
//             let stats = InstanceStats {
//                 stage: observer.stage(),
//             };
//             serde_json::to_value(&stats).unwrap()
//         })
//         .start(
//             config.service_name.as_str(),
//             config.service_version.as_str(),
//         )
//         .await
//         .map_err(|e| anyhow::anyhow!("Failed to start service: {e}"))?;
//     Ok(service)
// }

// async fn create_endpoint(
//     endpoint_name: impl Into<String>,
//     service: &NatsService,
// ) -> Result<Endpoint> {
//     let info = service.info().await;
//     let group_name = format!("{}-{}", info.name, info.id);
//     let group = service.group(group_name);

//     let endpoint = group
//         .endpoint(endpoint_name.into())
//         .await
//         .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

//     Ok(endpoint)
// }

// async fn shutdown_endpoint_handler(
//     controller: ServiceController,
//     endpoint: Endpoint,
// ) -> Result<()> {
//     let mut endpoint = endpoint;

//     // note: this is a child cancellation token, canceling it will not cancel the parent
//     // but the parent will cancel the child -- we only use this to observe if another
//     // controller has cancelled the service
//     let cancellation_token = controller.cancel_token();

//     loop {
//         let req = tokio::select! {
//             _ = cancellation_token.cancelled() => {
//                 // tracing::trace!(worker_id, "Shutting down service {}", self.endpoint.name);
//                 return Ok(());
//             }

//             // await on service request
//             req = endpoint.next() => {
//                 req
//             }
//         };

//         if let Some(req) = req {
//             let response = "DONE".to_string();
//             if let Err(e) = req.respond(Ok(response.into())).await {
//                 tracing::warn!("Failed to respond to the shutdown request: {:?}", e);
//             }

//             controller.set_stage(ServiceStage::ShuttingDown);
//         }
//     }
// }

// #[derive(Debug, Clone, Builder)]
// pub struct Config {
//     /// The NATS server address
//     #[builder(default = "String::from(\"nats://localhost:4222\")")]
//     pub nats_address: String,

//     #[builder(setter(into), default = "String::from(SERVICE_NAME)")]
//     pub service_name: String,

//     #[builder(setter(into), default = "String::from(SERVICE_VERSION)")]
//     pub service_version: String,

//     #[builder(setter(into), default = "String::from(SERVICE_DESCRIPTION)")]
//     pub service_description: String,
// }

// impl Config {
//     pub fn new() -> Result<Config> {
//         Ok(ConfigBuilder::default().build()?)
//     }

//     /// Create a new [`ConfigBuilder`]
//     pub fn builder() -> ConfigBuilder {
//         ConfigBuilder::default()
//     }
// }

// // todo: move to icp - transports

// #[derive(Clone, Debug)]
// pub struct NatsClient {
//     client: Client,
//     js_ctx: jetstream::Context,
// }

// impl NatsClient {
//     pub fn client(&self) -> &Client {
//         &self.client
//     }

//     pub fn jetstream(&self) -> &jetstream::Context {
//         &self.js_ctx
//     }

//     pub fn service_builder(&self) -> NatsServiceBuilder {
//         self.client.service_builder()
//     }
// }

// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct ServiceInfo {
//     pub name: String,
//     pub id: String,
//     pub version: String,
//     pub started: String,
//     pub endpoints: Vec<EndpointInfo>,
// }

// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct EndpointInfo {
//     pub name: String,
//     pub subject: String,
//     pub data: serde_json::Value,
// }

// impl EndpointInfo {
//     pub fn get<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
//         serde_json::from_value(self.data.clone()).map_err(Into::into)
//     }
// }

// #[derive(Clone, Debug, Builder)]
// #[builder(build_fn(private, name = "build_internal"))]
// pub struct ClientOptions {
//     #[builder(setter(into))]
//     server: String,

//     #[builder(setter(into, strip_option), default)]
//     username: Option<String>,

//     #[builder(setter(into, strip_option), default)]
//     password: Option<String>,
// }

// #[allow(dead_code)]
// impl ClientOptions {
//     pub fn builder() -> ClientOptionsBuilder {
//         ClientOptionsBuilder::default()
//     }
// }

// impl ClientOptionsBuilder {
//     pub async fn build(&self) -> Result<NatsClient> {
//         let opts = self.build_internal()?;

//         // Create an unauthenticated connection to NATS.
//         let client = async_nats::ConnectOptions::new();

//         let client = if let (Some(username), Some(password)) = (opts.username, opts.password) {
//             client.user_and_password(username, password)
//         } else {
//             client
//         };

//         let client = client.connect(&opts.server).await?;

//         let js_ctx = jetstream::new(client.clone());

//         Ok(NatsClient { client, js_ctx })
//     }
// }
