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

use std::{pin::Pin, sync::Arc, time::Duration};

use anyhow::{Context, ensure};
use futures_core::Stream;
use tokio::{
    sync::Semaphore,
    time::{Instant, timeout_at},
};
use tonic::{Request, Response, Status, Streaming};

use crate::{
    preprocessor::{MAX_BODY, headers},
    proto::envoy::{
        config::core::v3::{HeaderValue, HeaderValueOption},
        service::ext_proc::v3::{
            self as ext,
            external_processor_server::{ExternalProcessor, ExternalProcessorServer},
            processing_request::Request as Input,
            processing_response::Response as Output,
        },
    },
    router::{MODEL_HEADER, Router},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);
const CHUNK_SIZE: usize = 64 * 1024;

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    pub status_code: u16,
    pub message: String,
}

impl Error {
    pub fn new(status_code: u16, message: impl Into<String>) -> Self {
        Self {
            status_code,
            message: message.into(),
        }
    }
}

pub struct Server {
    router: Arc<Router>,
    requests: Arc<Semaphore>,
    streams: Arc<Semaphore>,
}

struct Prepared {
    body: Vec<u8>,
    headers: ext::HeaderMutation,
    trailers: bool,
}

impl Server {
    pub fn new(router: Arc<Router>) -> Self {
        Self {
            router,
            requests: Arc::new(Semaphore::new(8)),
            streams: Arc::new(Semaphore::new(16)),
        }
    }

    pub fn into_service(self) -> ExternalProcessorServer<Self> {
        ExternalProcessorServer::new(self)
    }

    async fn prepare(
        router: &Router,
        input: &mut Streaming<ext::ProcessingRequest>,
    ) -> anyhow::Result<Prepared> {
        let first = input.message().await?.context("missing request headers")?;
        let Some(Input::RequestHeaders(first)) = first.request else {
            anyhow::bail!("request headers must arrive first");
        };
        ensure!(
            !first.end_of_stream,
            "chat completion requires a request body"
        );
        let values = first
            .headers
            .unwrap_or_default()
            .headers
            .into_iter()
            .map(|h| {
                let value = if h.raw_value.is_empty() {
                    h.value
                } else {
                    String::from_utf8(h.raw_value)?
                };
                Ok((h.key, value))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let (headers, mut remove_headers) = headers(&values)?;
        let mut body = Vec::new();
        let mut trailers = false;
        loop {
            let next = input.message().await?.context("incomplete request body")?;
            match next.request {
                Some(Input::RequestBody(chunk)) => {
                    ensure!(
                        !trailers || (chunk.body.is_empty() && chunk.end_of_stream),
                        "expected body end after request trailers"
                    );
                    if chunk.body.len() > MAX_BODY - body.len() {
                        return Err(Error::new(413, "request body exceeds 2 MiB").into());
                    }
                    body.extend_from_slice(&chunk.body);
                    if chunk.end_of_stream {
                        break;
                    }
                }
                // Stock agentgateway 1.0 sends an empty body EOS after trailers.
                Some(Input::RequestTrailers(_)) if !trailers => trailers = true,
                _ => anyhow::bail!("expected request body"),
            }
        }
        let (body, model) = router.decide(&body, &headers).await?;
        if body.len() > MAX_BODY {
            return Err(Error::new(413, "rewritten request body exceeds 2 MiB").into());
        }
        remove_headers.push("content-length".into());
        Ok(Prepared {
            body,
            headers: ext::HeaderMutation {
                set_headers: vec![set_header(MODEL_HEADER, &model)],
                remove_headers,
            },
            trailers,
        })
    }
}

#[tonic::async_trait]
impl ExternalProcessor for Server {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ext::ProcessingResponse, Status>> + Send>>;

    async fn process(
        &self,
        request: Request<Streaming<ext::ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let stream_permit = self.streams.clone().try_acquire_owned();
        let request_permit = stream_permit
            .as_ref()
            .ok()
            .and_then(|_| self.requests.clone().try_acquire_owned().ok());
        let router = self.router.clone();
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        let mut input = request.into_inner();
        let output = async_stream::stream! {
            // These permits belong to the output stream, including while it is backpressured.
            // Dropping that stream also cancels input reads and the routing future.
            let Ok(_stream_permit) = stream_permit else {
                yield Ok(reject(Error::new(503, "processor stream limit reached")));
                return;
            };
            let Some(request_permit) = request_permit else {
                yield Ok(reject(Error::new(503, "preprocessor concurrency limit reached")));
                return;
            };
            let result = timeout_at(deadline, Self::prepare(&router, &mut input)).await;
            drop(request_permit);
            let prepared = match result {
                Ok(Ok(prepared)) => prepared,
                Ok(Err(error)) => {
                    tracing::warn!(%error, "preprocessing rejected request");
                    yield Ok(reject(error.downcast::<Error>()
                        .unwrap_or_else(|e| Error::new(400, e.to_string()))));
                    return;
                }
                Err(_) => {
                    yield Ok(reject(Error::new(504, "preprocessing deadline exceeded")));
                    return;
                }
            };
            let response_deadline = Instant::now() + RESPONSE_TIMEOUT;
            // v1.0 selects the route when this header reply arrives, before reading the body.
            yield Ok(reply(Output::RequestHeaders(ext::HeadersResponse {
                response: Some(ext::CommonResponse {
                    header_mutation: Some(prepared.headers),
                    ..Default::default()
                }),
            })));
            if prepared.trailers {
                yield Ok(reply(Output::RequestTrailers(ext::TrailersResponse::default())));
            }
            for (i, chunk) in prepared.body.chunks(CHUNK_SIZE).enumerate() {
                if Instant::now() >= response_deadline {
                    yield Err(Status::deadline_exceeded("request output timed out"));
                    return;
                }
                yield Ok(reply(Output::RequestBody(body_reply(
                    chunk.to_vec(), (i + 1) * CHUNK_SIZE >= prepared.body.len(),
                ))));
            }
            let mut headers_seen = false;
            let mut trailers_seen = false;
            loop {
                let next = match timeout_at(response_deadline, input.message()).await {
                    Ok(Ok(Some(next))) => next,
                    Ok(Ok(None)) => {
                        yield Err(Status::failed_precondition("incomplete response stream"));
                        return;
                    }
                    Ok(Err(error)) => { yield Err(error); return; }
                    Err(_) => {
                        yield Err(Status::deadline_exceeded("response processing timed out"));
                        return;
                    }
                };
                let (output, done) = match next.request {
                    Some(Input::ResponseHeaders(h)) if !headers_seen => {
                        headers_seen = true;
                        (Output::ResponseHeaders(ext::HeadersResponse {
                            response: Some(ext::CommonResponse::default()),
                        }), h.end_of_stream)
                    }
                    Some(Input::ResponseBody(b)) if headers_seen
                        && (!trailers_seen || (b.body.is_empty() && b.end_of_stream)) => {
                        let done = b.end_of_stream;
                        (Output::ResponseBody(body_reply(b.body, done)), done)
                    }
                    Some(Input::ResponseTrailers(_)) if headers_seen && !trailers_seen => {
                        trailers_seen = true;
                        // ACK trailers, then consume agentgateway's synthetic body EOS.
                        (Output::ResponseTrailers(ext::TrailersResponse::default()), false)
                    }
                    _ => {
                        yield Err(Status::invalid_argument("unexpected response processing phase"));
                        return;
                    }
                };
                yield Ok(reply(output));
                if done { return; }
            }
        };
        Ok(Response::new(Box::pin(output)))
    }
}

fn reply(response: Output) -> ext::ProcessingResponse {
    ext::ProcessingResponse {
        response: Some(response),
        ..Default::default()
    }
}

fn body_reply(body: Vec<u8>, end_of_stream: bool) -> ext::BodyResponse {
    ext::BodyResponse {
        response: Some(ext::CommonResponse {
            body_mutation: Some(ext::BodyMutation {
                mutation: Some(ext::body_mutation::Mutation::StreamedResponse(
                    ext::StreamedBodyResponse {
                        body,
                        end_of_stream,
                    },
                )),
            }),
            ..Default::default()
        }),
    }
}

fn set_header(key: &str, value: &str) -> HeaderValueOption {
    HeaderValueOption {
        header: Some(HeaderValue {
            key: key.into(),
            value: String::new(),
            raw_value: value.as_bytes().to_vec(),
        }),
        append_action: 2,
        keep_empty_value: false,
    }
}

fn reject(error: Error) -> ext::ProcessingResponse {
    reply(Output::ImmediateResponse(ext::ImmediateResponse {
        status: Some(crate::proto::envoy::r#type::v3::HttpStatus {
            code: i32::from(error.status_code),
        }),
        headers: Some(ext::HeaderMutation {
            set_headers: vec![set_header("content-type", "application/json")],
            remove_headers: vec![],
        }),
        body: serde_json::to_vec(&serde_json::json!({"error": {
            "message": error.message, "type": "switchyard_preproc_error"
        }}))
        .expect("JSON error object serializes"),
        details: "switchyard_preproc".into(),
        ..Default::default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};

    type Client =
        ext::external_processor_client::ExternalProcessorClient<tonic::transport::Channel>;

    async fn start() -> (Client, Arc<Semaphore>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let runner = switchyard_runner::Runner::from_toml(include_str!("../config/routes.toml"));
        let server = Server::new(Arc::new(Router::new(runner.unwrap())));
        let requests = server.requests.clone();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(server.into_service())
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        (
            Client::connect(format!("http://{address}")).await.unwrap(),
            requests,
        )
    }

    fn input(request: Input) -> ext::ProcessingRequest {
        ext::ProcessingRequest {
            request: Some(request),
            ..Default::default()
        }
    }

    fn body(bytes: &[u8], end_of_stream: bool) -> ext::HttpBody {
        ext::HttpBody {
            body: bytes.to_vec(),
            end_of_stream,
        }
    }

    fn headers() -> Input {
        let headers = [
            (":method", "POST"),
            (":path", "/v1/chat/completions"),
            ("content-type", "application/json"),
            (MODEL_HEADER, "forged"),
        ]
        .into_iter()
        .map(|(k, v)| set_header(k, v).header.unwrap())
        .collect();
        Input::RequestHeaders(ext::HttpHeaders {
            headers: Some(crate::proto::envoy::config::core::v3::HeaderMap { headers }),
            end_of_stream: false,
        })
    }

    async fn send(tx: &mpsc::Sender<ext::ProcessingRequest>, request: Input) {
        tx.send(input(request)).await.unwrap();
    }

    async fn next(rx: &mut Streaming<ext::ProcessingResponse>) -> Output {
        rx.message().await.unwrap().unwrap().response.unwrap()
    }

    fn streamed(reply: ext::BodyResponse) -> ext::StreamedBodyResponse {
        let Some(ext::body_mutation::Mutation::StreamedResponse(body)) =
            reply.response.unwrap().body_mutation.unwrap().mutation
        else {
            panic!()
        };
        body
    }

    #[tokio::test]
    async fn routes_upload_and_passes_response_through_trailers_to_eos() {
        let (mut client, requests) = start().await;
        let (tx, receive) = mpsc::channel(8);
        send(&tx, headers()).await;
        let receive = ReceiverStream::new(receive);
        let mut rx = client.process(receive).await.unwrap().into_inner();
        let json = br#"{"model":"auto","messages":[{"role":"user","content":"hello"}]}"#;
        for chunk in json.chunks(40) {
            send(&tx, Input::RequestBody(body(chunk, false))).await;
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(30), rx.message())
                .await
                .is_err()
        );
        send(&tx, Input::RequestTrailers(ext::HttpTrailers::default())).await;
        send(&tx, Input::RequestBody(body(b"", true))).await;
        let Output::RequestHeaders(h) = next(&mut rx).await else {
            panic!()
        };
        let h = h.response.unwrap().header_mutation.unwrap();
        let selected = h.set_headers[0].header.as_ref().unwrap();
        assert_eq!(selected.raw_value, b"Qwen/Qwen3-0.6B");
        assert!(h.remove_headers.contains(&"content-length".to_owned()));
        assert!(matches!(next(&mut rx).await, Output::RequestTrailers(_)));
        let Output::RequestBody(b) = next(&mut rx).await else {
            panic!()
        };
        let b = streamed(b);
        assert!(b.end_of_stream);
        let value: serde_json::Value = serde_json::from_slice(&b.body).unwrap();
        assert_eq!(value["model"], "Qwen/Qwen3-0.6B");
        assert_eq!(requests.available_permits(), 8);
        send(&tx, Input::ResponseHeaders(ext::HttpHeaders::default())).await;
        assert!(matches!(next(&mut rx).await, Output::ResponseHeaders(_)));
        for (bytes, done) in [
            (b"data: hello\n\n".as_slice(), false),
            (b"".as_slice(), true),
        ] {
            send(&tx, Input::ResponseBody(body(bytes, done))).await;
            let Output::ResponseBody(b) = next(&mut rx).await else {
                panic!()
            };
            let b = streamed(b);
            assert_eq!((b.body.as_slice(), b.end_of_stream), (bytes, done));
            if !done {
                send(&tx, Input::ResponseTrailers(ext::HttpTrailers::default())).await;
                assert!(matches!(next(&mut rx).await, Output::ResponseTrailers(_)));
            }
        }
        assert!(rx.message().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rejects_cumulative_body_limit_before_routing_headers() {
        let (mut client, _) = start().await;
        let messages = tokio_stream::iter([
            input(headers()),
            input(Input::RequestBody(body(&vec![b' '; MAX_BODY], false))),
            input(Input::RequestBody(body(b" ", true))),
        ]);
        let mut rx = client.process(messages).await.unwrap().into_inner();
        let Output::ImmediateResponse(error) = next(&mut rx).await else {
            panic!()
        };
        assert_eq!(error.status.unwrap().code, 413);
        assert!(rx.message().await.unwrap().is_none());
    }
}
