// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A [`StreamingDispatch`] for hosts where something else sends the request.
//!
//! Behind an Envoy gateway the EPP chooses the worker but Envoy forwards the
//! request and owns the response stream. [`GatewayDispatch`] lets such a host
//! run the frontend's [`RoutingHost`](super::RoutingHost) unchanged: at the
//! dispatch point it records the selected worker instead of sending, and it
//! returns a response stream that the host feeds with the progress it observes
//! on the gateway's copy of the response. Every routing-host stage (session
//! affinity, request classifiers, booking, output tracking, cleanup) then runs
//! exactly as it does in the frontend.
//!
//! The host cannot see output token IDs, only how many tokens arrived, so the
//! stream carries placeholder IDs. Routing modes that hash output tokens
//! (approximate KV indexing without KV events) must not use this dispatch.

use dashmap::DashMap;
use dynamo_protocols::types::CompletionUsage;
use dynamo_runtime::{
    component::Instance,
    engine::ResponseStream,
    error::{DynamoError, ErrorType},
    pipeline::{
        AddressedRequest, AsyncEngineContextProvider, Error, ManyIn, ManyOut, SingleIn,
        StreamingDispatch,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::protocols::common::{
    FinishReason,
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

/// The worker the routing host selected for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayRoute {
    pub worker_id: u64,
    /// The worker's address for the gateway (`host:port`).
    pub endpoint: String,
    pub dp_rank: Option<u32>,
    /// The prompt tokens routed on, for the host to forward to the worker.
    pub token_ids: Vec<u32>,
}

/// Response progress a host observed on the gateway's copy of the response.
#[derive(Debug, Clone, PartialEq)]
pub enum GatewayProgress {
    /// This many output tokens arrived.
    Tokens(usize),
    /// The response finished.
    Finish {
        reason: Option<FinishReason>,
        usage: Option<CompletionUsage>,
    },
}

struct GatewayRequest {
    route: Option<GatewayRoute>,
    responses: mpsc::UnboundedSender<Annotated<LLMEngineOutput>>,
}

/// Maps a worker id to the address the gateway forwards to (`host:port`).
pub type EndpointResolver = Box<dyn Fn(u64) -> Option<String> + Send + Sync>;

/// Records each selected worker by request id and streams host-reported
/// progress back into the routing host. See the module docs.
pub struct GatewayDispatch {
    requests: DashMap<String, GatewayRequest>,
    resolve_endpoint: EndpointResolver,
}

impl GatewayDispatch {
    /// `resolve_endpoint` runs at dispatch: a worker it cannot resolve fails
    /// the send, so the host never commits a session to an unreachable worker.
    pub fn new(resolve_endpoint: impl Fn(u64) -> Option<String> + Send + Sync + 'static) -> Self {
        Self {
            requests: DashMap::new(),
            resolve_endpoint: Box::new(resolve_endpoint),
        }
    }

    /// Take the worker selected for `request_id`. Present once the routing
    /// host's `generate` has returned a stream for that request.
    pub fn take_route(&self, request_id: &str) -> Option<GatewayRoute> {
        self.requests.get_mut(request_id)?.route.take()
    }

    /// Feed observed progress into the request's response stream. Returns
    /// `false` when the request is unknown (never dispatched, or finished).
    pub fn progress(&self, request_id: &str, progress: GatewayProgress) -> bool {
        let Some(request) = self.requests.get(request_id) else {
            return false;
        };
        let output = match progress {
            GatewayProgress::Tokens(0) => return true,
            GatewayProgress::Tokens(count) => LLMEngineOutput {
                token_ids: vec![0; count],
                ..Default::default()
            },
            GatewayProgress::Finish { reason, usage } => LLMEngineOutput {
                finish_reason: reason,
                completion_usage: usage,
                ..Default::default()
            },
        };
        request.responses.send(Annotated::from_data(output)).is_ok()
    }

    /// End the request's response stream normally.
    pub fn complete(&self, request_id: &str) {
        self.requests.remove(request_id);
    }

    /// End the request's response stream with a cancellation. Cancellation is
    /// not a worker fault, so the worker is neither reported down nor retried.
    pub fn abort(&self, request_id: &str, reason: &str) {
        if let Some((_, request)) = self.requests.remove(request_id) {
            let _ = request
                .responses
                .send(Annotated::from_err(cancelled(reason)));
        }
    }

    /// Number of requests whose response stream is still open.
    pub fn open_requests(&self) -> usize {
        self.requests.len()
    }
}

fn cancelled(reason: &str) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(reason.to_string())
        .build()
}

#[async_trait::async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for GatewayDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (addressed, context) = request.transfer(());
        let request_id = context.id().to_string();
        let (request, _address, instance) = addressed.into_parts();
        let worker_id = instance
            .as_ref()
            .map(Instance::id)
            .ok_or_else(|| anyhow::anyhow!("gateway dispatch requires a selected worker"))?;
        // Unavailable rather than a worker fault: the worker may be healthy and
        // merely not yet indexed by the gateway's endpoint view.
        let endpoint = (self.resolve_endpoint)(worker_id).ok_or_else(|| {
            DynamoError::builder()
                .error_type(ErrorType::Unavailable)
                .message(format!(
                    "selected worker {worker_id} has no gateway endpoint"
                ))
                .build()
        })?;
        let route = GatewayRoute {
            worker_id,
            endpoint,
            dp_rank: request.routing.as_ref().and_then(|routing| routing.dp_rank),
            token_ids: request.token_ids.to_vec(),
        };

        let (responses, receiver) = mpsc::unbounded_channel();
        let previous = self.requests.insert(
            request_id.clone(),
            GatewayRequest {
                route: Some(route),
                responses,
            },
        );
        if previous.is_some() {
            tracing::warn!(%request_id, "Gateway dispatch replaced an open request");
        }

        Ok(ResponseStream::new(
            Box::pin(UnboundedReceiverStream::new(receiver)),
            context.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        anyhow::bail!("gateway dispatch does not support streaming input")
    }
}

#[cfg(test)]
mod tests;
