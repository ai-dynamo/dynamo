// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::unified_client::RequestPlaneClient;
use super::*;
use crate::engine::{AsyncEngine, AsyncEngineContextProvider, Data};
use crate::logging::inject_trace_headers_into_map;
use crate::pipeline::network::ConnectionInfo;
use crate::pipeline::network::NetworkStreamWrapper;
use crate::pipeline::network::PendingConnections;
use crate::pipeline::network::STREAM_ERR_MSG;
use crate::pipeline::network::StreamOptions;
use crate::pipeline::network::TwoPartCodec;
use crate::pipeline::network::codec::TwoPartMessage;
use crate::pipeline::network::tcp;
use crate::pipeline::{ManyOut, PipelineError, ResponseStream, SingleIn};
use crate::protocols::maybe_error::MaybeError;

use anyhow::{Error, Result};
use serde::Deserialize;
use serde::Serialize;
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

pub struct AddressedPushRouter {
    // Request transport (unified trait object - works with all transports)
    req_client: Arc<dyn RequestPlaneClient>,

    // Response transport (TCP streaming - unchanged)
    resp_transport: Arc<tcp::server::TcpStreamServer>,
}

impl AddressedPushRouter {
    /// Create a new router with a request plane client
    ///
    /// This is the unified constructor that works with any transport type.
    /// The client is provided as a trait object, hiding the specific implementation.
    pub fn new(
        req_client: Arc<dyn RequestPlaneClient>,
        resp_transport: Arc<tcp::server::TcpStreamServer>,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            req_client,
            resp_transport,
        }))
    }
}

#[async_trait::async_trait]
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

        tracing::trace!(
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

        // Send request using unified client interface
        tracing::trace!(
            request_id,
            transport = self.req_client.transport_name(),
            address = %address,
            "Sending request via request plane client"
        );

        // Prepare trace headers using shared helper
        let mut headers = std::collections::HashMap::new();
        inject_trace_headers_into_map(&mut headers);

        // Send request (works for all transport types)
        let _response = self
            .req_client
            .send_request(address, buffer, headers)
            .await?;

        tracing::trace!(request_id, "awaiting transport handshake");
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
                        tracing::warn!(%err, %json_str, "Failed deserializing JSON to response");

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
                tracing::debug!("Request cancelled and then trying to read a response");
                None
            } else {
                // stream ended unexpectedly
                tracing::debug!("{STREAM_ERR_MSG}");
                Some(U::from_err(Error::msg(STREAM_ERR_MSG).into()))
            }
        });

        Ok(ResponseStream::new(Box::pin(stream), engine_ctx))
    }
}
