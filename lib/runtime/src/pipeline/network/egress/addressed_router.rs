// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_nats::client::Client;
use async_nats::{HeaderMap, HeaderValue};
use tracing as log;

use super::*;
use super::http_router::{HttpRequestClient, HttpAddressedRouter};
use crate::config::RequestPlaneMode;
use crate::logging::DistributedTraceContext;
use crate::logging::get_distributed_tracing_context;
use crate::logging::inject_otel_context_into_nats_headers;
use crate::{Result, protocols::maybe_error::MaybeError};
use tokio_stream::{StreamExt, StreamNotifyClose, wrappers::ReceiverStream};
use tracing::Instrument;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RequestType {
    SingleIn,
    ManyIn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseType {
    SingleOut,
    ManyOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestControlMessage {
    id: String,
    request_type: RequestType,
    response_type: ResponseType,
    connection_info: ConnectionInfo,
}

pub struct AddressedRequest<T> {
    request: T,
    address: String,
}

impl<T> AddressedRequest<T> {
    pub fn new(request: T, address: String) -> Self {
        Self { request, address }
    }

    pub(crate) fn into_parts(self) -> (T, String) {
        (self.request, self.address)
    }
}

/// Transport mode for AddressedPushRouter
enum RequestTransport {
    Nats(Client),
    Http(Arc<HttpRequestClient>),
}

pub struct AddressedPushRouter {
    // Request transport (NATS or HTTP)
    req_transport: RequestTransport,

    // Response transport (TCP streaming - unchanged)
    resp_transport: Arc<tcp::server::TcpStreamServer>,
}

impl AddressedPushRouter {
    /// Create a new router with NATS request transport
    pub fn new(
        req_transport: Client,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            req_transport: RequestTransport::Nats(req_transport),
            resp_transport,
        }))
    }

    /// Create a new router with HTTP request transport
    pub fn new_http(
        http_client: Arc<HttpRequestClient>,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            req_transport: RequestTransport::Http(http_client),
            resp_transport,
        }))
    }

    /// Create router based on request plane mode from environment
    pub fn from_mode(
        mode: RequestPlaneMode,
        nats_client: Option<Client>,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        match mode {
            RequestPlaneMode::Nats => {
                let nats_client = nats_client
                    .ok_or_else(|| anyhow::anyhow!("NATS client required for NATS mode"))?;
                Self::new(nats_client, resp_transport)
            }
            RequestPlaneMode::Http => {
                let http_client = Arc::new(HttpRequestClient::from_env()?);
                Self::new_http(http_client, resp_transport)
            }
        }
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<AddressedRequest<T>>, ManyOut<U>, Error> for AddressedPushRouter
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        let request_id = request.context().id().to_string();
        let (addressed_request, context) = request.transfer(());
        let (request, address) = addressed_request.into_parts();
        let engine_ctx = context.context();
        let engine_ctx_ = engine_ctx.clone();

        // registration options for the data plane in a singe in / many out configuration
        let options = StreamOptions::builder()
            .context(engine_ctx.clone())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        // register our needs with the data plane
        // todo - generalize this with a generic data plane object which hides the specific transports
        let pending_connections: PendingConnections = self.resp_transport.register(options).await;

        // validate and unwrap the RegisteredStream object
        let pending_response_stream = match pending_connections.into_parts() {
            (None, Some(recv_stream)) => recv_stream,
            _ => {
                panic!("Invalid data plane registration for a SingleIn/ManyOut transport");
            }
        };

        // separate out the connection info and the stream provider from the registered stream
        let (connection_info, response_stream_provider) = pending_response_stream.into_parts();

        // package up the connection info as part of the "header" component of the two part message
        // used to issue the request on the
        // todo -- this object should be automatically created by the register call, and achieved by to the two into_parts()
        // calls. all the information here is provided by the [`StreamOptions`] object and/or the dataplane object
        let control_message = RequestControlMessage {
            id: engine_ctx.id().to_string(),
            request_type: RequestType::SingleIn,
            response_type: ResponseType::ManyOut,
            connection_info,
        };

        // next build the two part message where we package the connection info and the request into
        // a single Vec<u8> that can be sent over the wire.
        // --- package this up in the WorkQueuePublisher ---
        let ctrl = serde_json::to_vec(&control_message)?;
        let data = serde_json::to_vec(&request)?;

        log::trace!(
            request_id,
            "packaging two-part message; ctrl: {} bytes, data: {} bytes",
            ctrl.len(),
            data.len()
        );

        let msg = TwoPartMessage::from_parts(ctrl.into(), data.into());

        // the request plane / work queue should provide a two part message codec that can be used
        // or it should take a two part message directly
        // todo - update this
        let codec = TwoPartCodec::default();
        let buffer = codec.encode_message(msg)?;

        // TRANSPORT ABSTRACT REQUIRED - END HERE

        // Send request based on transport mode
        match &self.req_transport {
            RequestTransport::Nats(client) => {
                log::trace!(request_id, "enqueueing two-part message to NATS");

                // Prepare trace headers using the OpenTelemetry injector pattern
                // This handles traceparent and tracestate headers according to W3C Trace Context standard
                let mut headers = HeaderMap::new();
                inject_otel_context_into_nats_headers(&mut headers, None);

                // Add additional custom headers that aren't handled by the OpenTelemetry propagator
                if let Some(trace_context) = get_distributed_tracing_context() {
                    if let Some(x_request_id) = trace_context.x_request_id {
                        headers.insert("x-request-id", x_request_id);
                    }
                    if let Some(x_dynamo_request_id) = trace_context.x_dynamo_request_id {
                        headers.insert("x-dynamo-request-id", x_dynamo_request_id);
                    }
                }

                let _response = client
                    .request_with_headers(address.to_string(), headers, buffer)
                    .await?;
            }
            RequestTransport::Http(http_client) => {
                log::trace!(request_id, "sending HTTP request to {}", address);

                // Insert Trace Context into Headers
                let mut headers = std::collections::HashMap::new();
                if let Some(trace_context) = get_distributed_tracing_context() {
                    headers.insert(
                        "traceparent".to_string(),
                        trace_context.create_traceparent(),
                    );
                    if let Some(tracestate) = trace_context.tracestate {
                        headers.insert("tracestate".to_string(), tracestate);
                    }
                    if let Some(x_request_id) = trace_context.x_request_id {
                        headers.insert("x-request-id".to_string(), x_request_id);
                    }
                    if let Some(x_dynamo_request_id) = trace_context.x_dynamo_request_id {
                        headers.insert("x-dynamo-request-id".to_string(), x_dynamo_request_id);
                    }
                }

                use crate::pipeline::network::request_plane::RequestPlaneClient;
                let _response = http_client
                    .send_request(address, buffer, headers)
                    .await?;
            }
        }

        log::trace!(request_id, "awaiting transport handshake");
        let response_stream = response_stream_provider
            .await
            .map_err(|_| PipelineError::DetachedStreamReceiver)?
            .map_err(PipelineError::ConnectionFailed)?;

        // TODO: Detect end-of-stream using Server-Sent Events (SSE)
        let mut is_complete_final = false;
        let stream = tokio_stream::StreamNotifyClose::new(
            tokio_stream::wrappers::ReceiverStream::new(response_stream.rx),
        )
        .filter_map(move |res| {
            if let Some(res_bytes) = res {
                if is_complete_final {
                    return Some(U::from_err(
                        Error::msg(
                            "Response received after generation ended - this should never happen",
                        )
                        .into(),
                    ));
                }
                match serde_json::from_slice::<NetworkStreamWrapper<U>>(&res_bytes) {
                    Ok(item) => {
                        is_complete_final = item.complete_final;
                        if let Some(data) = item.data {
                            Some(data)
                        } else if is_complete_final {
                            None
                        } else {
                            Some(U::from_err(
                                Error::msg("Empty response received - this should never happen")
                                    .into(),
                            ))
                        }
                    }
                    Err(err) => {
                        // legacy log print
                        let json_str = String::from_utf8_lossy(&res_bytes);
                        log::warn!(%err, %json_str, "Failed deserializing JSON to response");

                        Some(U::from_err(Error::new(err).into()))
                    }
                }
            } else if is_complete_final {
                // end of stream
                None
            } else if engine_ctx_.is_stopped() {
                // Gracefully end the stream if 'stop_generating()' was called. Do NOT check for
                // 'is_killed()' here because it implies the stream ended abnormally which should be
                // handled by the error branch below.
                log::debug!("Request cancelled and then trying to read a response");
                None
            } else {
                // stream ended unexpectedly
                log::debug!("{STREAM_ERR_MSG}");
                Some(U::from_err(Error::msg(STREAM_ERR_MSG).into()))
            }
        });

        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}
