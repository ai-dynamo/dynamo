// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;
use std::sync::Arc;
use std::time::Duration;

use crate::{PoolSelector, PoolSnapshot, PoolTarget, RoutingRequest};
use async_trait::async_trait;
use dynamo_llm::http::service::error::HttpError;
use dynamo_llm::protocols::agents::HEADER_DYNAMO_SESSION_ID;
use dynamo_llm::protocols::codec::{Message, SseLineCodec};
use dynamo_llm::protocols::common::extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_llm::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};
use dynamo_protocols::types::ChatCompletionStreamOptions;
use dynamo_runtime::engine::{
    AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, ResponseStream,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::header::ACCEPT;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::watch;
use tokio_util::codec::FramedRead;
use tokio_util::io::StreamReader;
use url::Url;

/// A request the engine can forward: the route it goes to on a pool
/// frontend, and the fields the engine reads or rewrites.
pub trait ForwardedRequest: Serialize + Send + Sync + 'static {
    /// One event of the pool's streamed response.
    type Response: DeserializeOwned + Send + Sync + 'static;

    /// Route below a pool frontend's root, as path segments.
    const ROUTE: &'static [&'static str];

    fn model(&self) -> &str;

    /// Ask the pool to stream its response. The HTTP handler consumes every
    /// engine as a stream and folds it for clients that did not ask to
    /// stream, so those clients' usage has to come back in the stream.
    fn request_stream(&mut self);
}

impl ForwardedRequest for NvCreateChatCompletionRequest {
    type Response = NvCreateChatCompletionStreamResponse;

    const ROUTE: &'static [&'static str] = &["v1", "chat", "completions"];

    fn model(&self) -> &str {
        &self.inner.model
    }

    fn request_stream(&mut self) {
        if self.inner.stream != Some(true) {
            self.inner.stream = Some(true);
            self.inner.stream_options = Some(usage_in_stream());
        }
    }
}

impl ForwardedRequest for NvCreateCompletionRequest {
    type Response = NvCreateCompletionResponse;

    const ROUTE: &'static [&'static str] = &["v1", "completions"];

    fn model(&self) -> &str {
        &self.inner.model
    }

    fn request_stream(&mut self) {
        if self.inner.stream != Some(true) {
            self.inner.stream = Some(true);
            self.inner.stream_options = Some(usage_in_stream());
        }
    }
}

fn usage_in_stream() -> ChatCompletionStreamOptions {
    ChatCompletionStreamOptions {
        include_usage: true,
        continuous_usage_stats: false,
    }
}

/// Forwards each request to the pool frontend its selector picks and streams
/// the pool's response back.
///
/// The engine reads the model from each request, so one instance serves
/// every model.
pub struct PoolForwardingEngine {
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    selector: Arc<dyn PoolSelector>,
    http: reqwest::Client,
}

impl PoolForwardingEngine {
    /// `http` must not set a total request timeout, because a response
    /// streams for as long as the pool generates.
    pub fn new(
        pools: watch::Receiver<Arc<PoolSnapshot>>,
        selector: Arc<dyn PoolSelector>,
        http: reqwest::Client,
    ) -> Self {
        Self {
            pools,
            selector,
            http,
        }
    }

    fn select(&self, model: &str) -> Result<PoolTarget, DynamoError> {
        let snapshot = self.pools.borrow().clone();
        let candidates = snapshot.pools_for(model).collect::<Vec<_>>();
        self.selector
            .select(&RoutingRequest::new(model), &candidates)
            .ok_or_else(|| unavailable(format!("no pool can serve model {model}")))
    }

    async fn forward<R: ForwardedRequest>(
        &self,
        request: SingleIn<R>,
    ) -> Result<ManyOut<Annotated<R::Response>>, Error> {
        // The HTTP handler moves the session header into the request context,
        // so it has to be put back on the forwarded request.
        let session = request
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .ok();
        let (mut body, context) = request.into_parts();
        let ctx = context.context();
        let target = self.select(body.model())?;
        let url = route_url(&target.frontend, R::ROUTE)?;
        body.request_stream();

        let mut pool_request = self
            .http
            .post(url.clone())
            .header(ACCEPT, "text/event-stream")
            .json(&body);
        if let Some(session) = session {
            pool_request = pool_request.header(HEADER_DYNAMO_SESSION_ID, session.as_str());
        }
        tracing::debug!(request_id = ctx.id(), pool = %target.pool, %url, "forwarding request");
        // TODO: retry the next pool when this one rejects the request before
        // its first byte (HLD 2.2.2, priority retry).
        let response = pool_request
            .send()
            .await
            .map_err(|error| unavailable(format!("pool frontend {url} failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            // The HTTP service applies its existing status and sanitization policy.
            let message = read_error_body(response).await;
            return Err(HttpError {
                code: status.as_u16(),
                message,
            }
            .into());
        }
        let events = pool_events::<R::Response>(response, ctx.clone());
        Ok(ResponseStream::new(Box::pin(events), ctx))
    }
}

#[async_trait]
impl<R: ForwardedRequest> AsyncEngine<SingleIn<R>, ManyOut<Annotated<R::Response>>, Error>
    for PoolForwardingEngine
{
    async fn generate(
        &self,
        request: SingleIn<R>,
    ) -> Result<ManyOut<Annotated<R::Response>>, Error> {
        self.forward(request).await
    }
}

/// Error responses must not buffer arbitrary data or wait indefinitely.
async fn read_error_body(mut response: reqwest::Response) -> String {
    const MAX_ERROR_BYTES: usize = 16 * 1024;
    let read = async {
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            let remaining = MAX_ERROR_BYTES - bytes.len();
            bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            if bytes.len() == MAX_ERROR_BYTES {
                return Ok::<_, reqwest::Error>(format!(
                    "{} [truncated]",
                    String::from_utf8_lossy(&bytes)
                ));
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    };
    match tokio::time::timeout(Duration::from_secs(5), read).await {
        Ok(Ok(message)) => message,
        Ok(Err(_)) => "could not read pool error response".to_owned(),
        Err(_) => "timed out reading pool error response".to_owned(),
    }
}

fn unavailable(message: String) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Unavailable)
        .message(message)
        .build()
}

/// `frontend` with `route` appended, keeping any path it already has.
fn route_url(frontend: &Url, route: &[&str]) -> Result<Url, DynamoError> {
    let mut url = frontend.clone();
    url.path_segments_mut()
        .map_err(|()| {
            DynamoError::builder()
                .error_type(ErrorType::Internal)
                .message(format!("pool frontend {frontend} cannot be a base URL"))
                .build()
        })?
        .pop_if_empty()
        .extend(route);
    Ok(url)
}

/// Decodes a pool frontend's SSE body into engine output.
///
/// The stream ends with the body, at the first error, or when the client
/// goes away. Ending early drops the response body, which closes the
/// connection and so cancels the request at the pool.
///
/// `SseLineCodec` consumes `data: [DONE]` without emitting it, so a pool that
/// closes its body cleanly before `[DONE]` looks like a finished stream.
fn pool_events<T: DeserializeOwned + Send + 'static>(
    response: reqwest::Response,
    ctx: Arc<dyn AsyncEngineContext>,
) -> impl Stream<Item = Annotated<T>> + Send {
    let body = StreamReader::new(response.bytes_stream().map_err(io::Error::other));
    let mut messages = FramedRead::new(body, SseLineCodec::new());
    async_stream::stream! {
        loop {
            let message = tokio::select! {
                biased;
                _ = ctx.stopped() => break,
                message = messages.next() => message,
            };
            let message = match message {
                Some(Ok(message)) => message,
                Some(Err(error)) => {
                    yield Annotated::from_error(format!("reading the pool response failed: {error}"));
                    break;
                }
                None => break,
            };
            // Keep-alive comments. The Global Frontend sends its own.
            if message.data.is_none() && message.event.is_none() {
                continue;
            }
            match decode(message) {
                Ok(event) => yield event,
                Err(error) => {
                    yield Annotated::from_error(error);
                    break;
                }
            }
        }
    }
}

fn decode<T: DeserializeOwned>(message: Message) -> Result<Annotated<T>, String> {
    if message.event.as_deref() == Some("error") {
        return Err(message
            .comments
            .map(|comments| comments.join("\n"))
            .unwrap_or_else(|| "the pool reported a stream error".to_owned()));
    }
    let data = match message.data.as_deref() {
        None => None,
        Some(data) => Some(serde_json::from_str(data).map_err(|error| {
            error_frame_message(data).unwrap_or_else(|| {
                format!("the pool sent an event this frontend cannot decode: {error}")
            })
        })?),
    };
    Ok(Annotated {
        data,
        id: message.id,
        event: message.event,
        comment: message.comments,
        error: None,
    })
}

/// The message of a `{"error": {"message": ...}}` frame, which a Dynamo
/// frontend sends when a stream fails after it started.
fn error_frame_message(data: &str) -> Option<String> {
    let frame: serde_json::Value = serde_json::from_str(data).ok()?;
    let error = frame.get("error")?;
    Some(
        error
            .get("message")
            .and_then(serde_json::Value::as_str)
            .map_or_else(|| error.to_string(), str::to_owned),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn error_body_reads_are_bounded() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app =
            axum::Router::new().route("/", axum::routing::get(|| async { "x".repeat(64 * 1024) }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let response = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        let body = read_error_body(response).await;
        server.abort();
        assert_eq!(body, format!("{} [truncated]", "x".repeat(16 * 1024)));
    }

    #[test]
    fn route_url_keeps_the_frontend_path() {
        for (frontend, expected) in [
            ("http://pool:8000", "http://pool:8000/v1/chat/completions"),
            ("http://pool:8000/", "http://pool:8000/v1/chat/completions"),
            (
                "http://gateway:8080/ohio/",
                "http://gateway:8080/ohio/v1/chat/completions",
            ),
        ] {
            let url = route_url(
                &frontend.parse().unwrap(),
                NvCreateChatCompletionRequest::ROUTE,
            );
            assert_eq!(url.unwrap().as_str(), expected, "{frontend}");
        }
        assert!(route_url(&"mailto:ops@example.com".parse().unwrap(), &["v1"]).is_err());
    }

    #[test]
    fn non_streaming_requests_ask_the_pool_to_stream_usage() {
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap();
        request.request_stream();
        assert_eq!(request.inner.stream, Some(true));
        assert!(request.inner.stream_options.unwrap().include_usage);

        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            }))
            .unwrap();
        request.request_stream();
        assert_eq!(
            request.inner.stream_options, None,
            "a streaming client's own options must pass through"
        );
    }

    #[test]
    fn error_frames_surface_the_pool_message() {
        let message = Message {
            id: None,
            event: None,
            data: Some(r#"{"error":{"message":"worker died","type":"internal_error"}}"#.to_owned()),
            comments: None,
        };
        let error = decode::<NvCreateChatCompletionStreamResponse>(message).unwrap_err();
        assert_eq!(error, "worker died");
    }
}
