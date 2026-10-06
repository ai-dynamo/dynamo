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

use std::{
    pin::Pin,
    sync::{Arc, atomic::Ordering},
    task::{Context, Poll},
    time::Duration,
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_stream::{Stream, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

use crate::{
    proto::envoy::{
        config::core::v3::{HeaderValue, HeaderValueOption},
        extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode,
        service::ext_proc::v3::{
            self as ext, external_processor_server::ExternalProcessor, processing_request,
            processing_response,
        },
    },
    router::{Router, TARGET_HEADER},
};

pub const MAX_BODY: usize = 2 * 1024 * 1024;
pub const MAX_CONCURRENT: usize = 8;
pub const MAX_IN_FLIGHT: usize = 16;
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);
const STREAM_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub struct Preproc {
    pub router: Arc<Router>,
    pub capacity: Arc<Semaphore>,
    pub streams: Arc<Semaphore>,
}

struct ResponseStream {
    receiver: ReceiverStream<Result<ext::ProcessingResponse, Status>>,
    _permit: OwnedSemaphorePermit,
}

impl Stream for ResponseStream {
    type Item = Result<ext::ProcessingResponse, Status>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().receiver).poll_next(context)
    }
}

fn response(kind: processing_response::Response) -> ext::ProcessingResponse {
    ext::ProcessingResponse {
        response: Some(kind),
        ..Default::default()
    }
}

fn error_response(code: i32, message: &str) -> ext::ProcessingResponse {
    response(processing_response::Response::ImmediateResponse(ext::ImmediateResponse {
        status: Some(crate::proto::envoy::r#type::v3::HttpStatus { code }),
        headers: Some(ext::HeaderMutation { set_headers: vec![set_header("content-type", "application/json")], remove_headers: vec![] }),
        body: serde_json::to_vec(&serde_json::json!({"error": {"message": message, "type": "switchyard_preproc_error"}})).unwrap_or_default(),
        details: "switchyard_preproc".to_owned(), ..Default::default()
    }))
}

fn set_header(key: &str, value: &str) -> HeaderValueOption {
    HeaderValueOption {
        header: Some(HeaderValue {
            key: key.to_owned(),
            value: String::new(),
            raw_value: value.as_bytes().to_vec(),
        }),
        append_action: 2,
        keep_empty_value: false,
    }
}

fn headers(input: ext::HttpHeaders) -> anyhow::Result<(http::HeaderMap, Vec<String>)> {
    anyhow::ensure!(
        !input.end_of_stream,
        "chat completion requires a request body"
    );
    let mut map = http::HeaderMap::new();
    let mut method = None;
    let mut path = None;
    let mut remove = [
        TARGET_HEADER,
        "x-gateway-destination-endpoint",
        "x-worker-instance-id",
        "x-prefill-instance-id",
        "x-dp-rank",
        "x-data-parallel-rank",
        "x-prefill-dp-rank",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    for header in input.headers.unwrap_or_default().headers {
        let value = if header.raw_value.is_empty() {
            header.value.into_bytes()
        } else {
            header.raw_value
        };
        let value = std::str::from_utf8(&value)?;
        let key = header.key.to_ascii_lowercase();
        if key == ":method" {
            anyhow::ensure!(
                method.replace(value.to_owned()).is_none(),
                "duplicate method"
            );
            continue;
        }
        if key == ":path" {
            anyhow::ensure!(path.replace(value.to_owned()).is_none(), "duplicate path");
            continue;
        }
        if key.starts_with(':') {
            continue;
        }
        if key.starts_with("x-dynamo-") || key.starts_with("x-gateway-") {
            remove.push(key.clone());
        }
        let name = http::HeaderName::from_bytes(key.as_bytes())?;
        anyhow::ensure!(!map.contains_key(&name), "duplicate request header");
        map.insert(name, http::HeaderValue::from_str(value)?);
    }
    anyhow::ensure!(method.as_deref() == Some("POST"), "only POST is supported");
    anyhow::ensure!(
        path.as_deref().and_then(|p| p.split('?').next()) == Some("/v1/chat/completions"),
        "only /v1/chat/completions is supported"
    );
    anyhow::ensure!(
        map.get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v
                .split(';')
                .next()
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))),
        "content-type must be application/json"
    );
    if map.contains_key("content-encoding") {
        return Err(
            crate::error::Reject::new(415, "compressed request bodies are unsupported").into(),
        );
    }
    if let Some(length) = map.get("content-length")
        && length.to_str()?.parse::<usize>()? > MAX_BODY
    {
        return Err(crate::error::Reject::new(413, "request body exceeds 2 MiB").into());
    }
    remove.sort();
    remove.dedup();
    Ok((map, remove))
}

impl Preproc {
    async fn process_request(
        &self,
        input: &mut Streaming<ext::ProcessingRequest>,
        output: &mpsc::Sender<Result<ext::ProcessingResponse, Status>>,
    ) -> anyhow::Result<bool> {
        let first = input
            .message()
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing request headers"))?;
        let buffered = first
            .protocol_config
            .as_ref()
            .is_some_and(|p| p.request_body_mode == BodySendMode::Buffered as i32);
        let process_response = first
            .protocol_config
            .as_ref()
            .is_none_or(|p| p.response_body_mode != 0);
        anyhow::ensure!(
            first.protocol_config.as_ref().is_none_or(|p| buffered
                || (p.request_body_mode == BodySendMode::FullDuplexStreamed as i32
                    && p.send_body_without_waiting_for_header_response)),
            "Buffered or FullDuplexStreamed request processing is required"
        );
        anyhow::ensure!(
            first
                .protocol_config
                .as_ref()
                .is_none_or(|p| p.response_body_mode == 0
                    || p.response_body_mode == BodySendMode::FullDuplexStreamed as i32),
            "response processing must be disabled or FullDuplexStreamed"
        );
        let Some(processing_request::Request::RequestHeaders(first)) = first.request else {
            anyhow::bail!("request headers must arrive first");
        };
        let (headers, remove) = headers(first)?;
        if buffered {
            output
                .send(Ok(response(processing_response::Response::RequestHeaders(
                    ext::HeadersResponse {
                        response: Some(ext::CommonResponse {
                            header_mutation: Some(ext::HeaderMutation {
                                set_headers: vec![],
                                remove_headers: remove.clone(),
                            }),
                            ..Default::default()
                        }),
                    },
                ))))
                .await?;
        }
        let mut body = Vec::new();
        loop {
            let request = input
                .message()
                .await?
                .ok_or_else(|| anyhow::anyhow!("missing complete request body"))?;
            let Some(processing_request::Request::RequestBody(chunk)) = request.request else {
                anyhow::bail!("expected request body");
            };
            if chunk.body.len() > MAX_BODY - body.len() {
                return Err(crate::error::Reject::new(413, "request body exceeds 2 MiB").into());
            }
            body.extend_from_slice(&chunk.body);
            if chunk.end_of_stream {
                break;
            }
            anyhow::ensure!(!buffered, "complete Buffered body is required");
        }
        let (body, target) = self.router.decide(&body, &headers).await?;
        if body.len() > MAX_BODY {
            return Err(
                crate::error::Reject::new(413, "rewritten request body exceeds 2 MiB").into(),
            );
        }
        let mut mutation = ext::HeaderMutation {
            set_headers: vec![
                set_header(TARGET_HEADER, &target),
                set_header("content-length", &body.len().to_string()),
            ],
            remove_headers: vec![],
        };
        let body_mutation = if buffered {
            ext::body_mutation::Mutation::Body(body)
        } else {
            // v1.0 routes as soon as headers are acknowledged. Publish the decision first.
            mutation.remove_headers = remove;
            output
                .send(Ok(response(processing_response::Response::RequestHeaders(
                    ext::HeadersResponse {
                        response: Some(ext::CommonResponse {
                            header_mutation: Some(mutation.clone()),
                            ..Default::default()
                        }),
                    },
                ))))
                .await?;
            ext::body_mutation::Mutation::StreamedResponse(ext::StreamedBodyResponse {
                body,
                end_of_stream: true,
            })
        };
        output
            .send(Ok(response(processing_response::Response::RequestBody(
                ext::BodyResponse {
                    response: Some(ext::CommonResponse {
                        header_mutation: buffered.then_some(mutation),
                        body_mutation: Some(ext::BodyMutation {
                            mutation: Some(body_mutation),
                        }),
                        ..Default::default()
                    }),
                },
            ))))
            .await?;
        Ok(process_response)
    }

    async fn pass_response(
        input: &mut Streaming<ext::ProcessingRequest>,
        output: &mpsc::Sender<Result<ext::ProcessingResponse, Status>>,
    ) -> anyhow::Result<()> {
        let mut headers_seen = false;
        while let Some(request) = input.message().await? {
            let (reply, done) = match request.request {
                Some(processing_request::Request::ResponseHeaders(headers)) if !headers_seen => {
                    headers_seen = true;
                    (
                        processing_response::Response::ResponseHeaders(ext::HeadersResponse {
                            response: Some(ext::CommonResponse::default()),
                        }),
                        headers.end_of_stream,
                    )
                }
                Some(processing_request::Request::ResponseBody(body)) if headers_seen => {
                    let done = body.end_of_stream;
                    (
                        processing_response::Response::ResponseBody(ext::BodyResponse {
                            response: Some(ext::CommonResponse {
                                body_mutation: Some(ext::BodyMutation {
                                    mutation: Some(ext::body_mutation::Mutation::StreamedResponse(
                                        ext::StreamedBodyResponse {
                                            body: body.body,
                                            end_of_stream: done,
                                        },
                                    )),
                                }),
                                ..Default::default()
                            }),
                        }),
                        done,
                    )
                }
                _ => anyhow::bail!("expected response headers or body"),
            };
            output.send(Ok(response(reply))).await?;
            if done {
                return Ok(());
            }
        }
        anyhow::bail!("response stream ended before completion")
    }
}

#[tonic::async_trait]
impl ExternalProcessor for Preproc {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ext::ProcessingResponse, Status>> + Send>>;

    async fn process(
        &self,
        request: Request<Streaming<ext::ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let Ok(stream) = self.streams.clone().try_acquire_owned() else {
            self.router.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(Response::new(Box::pin(tokio_stream::iter([Ok(
                error_response(503, "processor stream limit reached"),
            )]))));
        };
        let Ok(permit) = self.capacity.clone().try_acquire_owned() else {
            self.router.errors.fetch_add(1, Ordering::Relaxed);
            return Ok(Response::new(Box::pin(tokio_stream::iter([Ok(
                error_response(503, "preprocessor concurrency limit reached"),
            )]))));
        };
        let mut input = request.into_inner();
        let (sender, receiver) = mpsc::channel(2);
        let this = self.clone();
        tokio::spawn(async move {
            let result =
                tokio::time::timeout(STREAM_TIMEOUT, this.process_request(&mut input, &sender))
                    .await;
            drop(permit);
            let failure = match result {
                Ok(Ok(process_response)) => {
                    if process_response {
                        tokio::select! {
                            _ = sender.closed() => {},
                            result = tokio::time::timeout(RESPONSE_TIMEOUT, Self::pass_response(&mut input, &sender)) => {
                                if !matches!(result, Ok(Ok(()))) {
                                    tracing::warn!(?result, "response passthrough failed");
                                    let _ = tokio::time::timeout(Duration::from_millis(100), sender.send(Err(Status::unavailable("response passthrough failed")))).await;
                                }
                            }
                        }
                    }
                    None
                }
                Ok(Err(error)) => {
                    tracing::warn!(error = %error, "preprocessing rejected request");
                    Some(error_response(
                        error
                            .downcast_ref::<crate::error::Reject>()
                            .map_or(400, |reject| reject.code),
                        &error.to_string(),
                    ))
                }
                Err(_) => Some(error_response(504, "preprocessing deadline exceeded")),
            };
            if let Some(failure) = failure {
                this.router.errors.fetch_add(1, Ordering::Relaxed);
                let _ = tokio::time::timeout(Duration::from_millis(100), sender.send(Ok(failure)))
                    .await;
            }
        });
        // Bound buffered output until the gRPC response stream drains or is dropped.
        let output = ResponseStream {
            receiver: ReceiverStream::new(receiver),
            _permit: stream,
        };
        Ok(Response::new(Box::pin(output)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(extra: Vec<HeaderValue>) -> ext::HttpHeaders {
        let mut values = vec![
            (":method", "POST"),
            (":path", "/v1/chat/completions"),
            ("content-type", "application/json"),
        ]
        .into_iter()
        .map(|(key, value)| HeaderValue {
            key: key.into(),
            value: value.into(),
            raw_value: vec![],
        })
        .collect::<Vec<_>>();
        values.extend(extra);
        ext::HttpHeaders {
            headers: Some(crate::proto::envoy::config::core::v3::HeaderMap { headers: values }),
            end_of_stream: false,
        }
    }
    #[test]
    fn clears_spoofed_target_and_worker_controls() {
        let (_, remove) = headers(request(vec![HeaderValue {
            key: "x-dynamo-worker-id".into(),
            value: "forged".into(),
            raw_value: vec![],
        }]))
        .unwrap();
        assert!(remove.contains(&TARGET_HEADER.to_owned()));
        assert!(remove.contains(&"x-gateway-destination-endpoint".to_owned()));
        assert!(remove.contains(&"x-dynamo-worker-id".to_owned()));
        for alias in [
            "x-worker-instance-id",
            "x-prefill-instance-id",
            "x-dp-rank",
            "x-data-parallel-rank",
            "x-prefill-dp-rank",
        ] {
            assert!(remove.contains(&alias.to_owned()));
        }
    }
    #[test]
    fn rejects_duplicate_session_ids() {
        let value = HeaderValue {
            key: "x-switchyard-session-id".into(),
            value: "one".into(),
            raw_value: vec![],
        };
        assert!(headers(request(vec![value.clone(), value])).is_err());
    }
    async fn start() -> (
        ext::external_processor_client::ExternalProcessorClient<tonic::transport::Channel>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let service = Preproc {
            router: Arc::new(Router::new(
                crate::config::Config::from_toml(
                    include_str!("../config/routes.toml"),
                    include_str!("../config/pool-bindings.toml"),
                )
                .unwrap(),
            )),
            capacity: Arc::new(Semaphore::new(1)),
            streams: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
        };
        let (shutdown, stop) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(ext::external_processor_server::ExternalProcessorServer::new(service))
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    async {
                        let _ = stop.await;
                    },
                )
                .await
                .unwrap();
        });
        let client = ext::external_processor_client::ExternalProcessorClient::connect(format!(
            "http://{address}"
        ))
        .await
        .unwrap();
        (client, shutdown)
    }

    #[tokio::test]
    async fn grpc_acknowledges_headers_before_body_and_mutates_route_and_length() {
        let (mut client, shutdown) = start().await;
        let (send, receive) = mpsc::channel(2);
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestHeaders(request(vec![]))),
            protocol_config: Some(ext::ProtocolConfiguration {
                request_body_mode: BodySendMode::Buffered as i32,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();
        let mut output = client
            .process(ReceiverStream::new(receive))
            .await
            .unwrap()
            .into_inner();
        let header = tokio::time::timeout(Duration::from_secs(1), output.message())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(
            header.response,
            Some(processing_response::Response::RequestHeaders(_))
        ));
        let body = serde_json::to_vec(&serde_json::json!({"model":"auto", "messages":[{"role":"user", "content":"hello"}], "unknown":{"keep":true}})).unwrap();
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                body,
                end_of_stream: true,
            })),
            ..Default::default()
        })
        .await
        .unwrap();
        let body = output.message().await.unwrap().unwrap();
        let Some(processing_response::Response::RequestBody(body)) = body.response else {
            panic!("expected body mutation")
        };
        let common = body.response.unwrap();
        let mutation = common.header_mutation.unwrap();
        let values = mutation
            .set_headers
            .into_iter()
            .map(|h| {
                let h = h.header.unwrap();
                (h.key, String::from_utf8(h.raw_value).unwrap())
            })
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(values[TARGET_HEADER], "qwen-small");
        let Some(ext::body_mutation::Mutation::Body(bytes)) =
            common.body_mutation.unwrap().mutation
        else {
            panic!("expected replacement body")
        };
        assert_eq!(values["content-length"], bytes.len().to_string());
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["model"], "Qwen/Qwen3-0.6B");
        assert_eq!(json["unknown"], serde_json::json!({"keep":true}));
        assert!(output.message().await.unwrap().is_none());
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn grpc_rejects_bad_json_and_bounds_stream_wait_and_concurrency() {
        let (mut client, shutdown) = start().await;
        let (send, receive) = mpsc::channel(2);
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestHeaders(request(vec![]))),
            protocol_config: Some(ext::ProtocolConfiguration {
                request_body_mode: BodySendMode::Buffered as i32,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();
        let mut output = client
            .process(ReceiverStream::new(receive))
            .await
            .unwrap()
            .into_inner();
        output.message().await.unwrap().unwrap();
        let (_send2, receive2) = mpsc::channel::<ext::ProcessingRequest>(1);
        let mut rejected = client
            .process(ReceiverStream::new(receive2))
            .await
            .unwrap()
            .into_inner();
        let Some(processing_response::Response::ImmediateResponse(rejected)) =
            rejected.message().await.unwrap().unwrap().response
        else {
            panic!("expected capacity rejection")
        };
        assert_eq!(rejected.status.unwrap().code, 503);
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                body: b"invalid".to_vec(),
                end_of_stream: true,
            })),
            ..Default::default()
        })
        .await
        .unwrap();
        let error = output.message().await.unwrap().unwrap();
        let Some(processing_response::Response::ImmediateResponse(error)) = error.response else {
            panic!("expected error")
        };
        assert_eq!(error.status.unwrap().code, 400);
        assert!(output.message().await.unwrap().is_none());
        let (send, receive) = mpsc::channel(2);
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestHeaders(request(vec![]))),
            protocol_config: Some(ext::ProtocolConfiguration {
                request_body_mode: BodySendMode::Buffered as i32,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();
        let mut output = client
            .process(ReceiverStream::new(receive))
            .await
            .unwrap()
            .into_inner();
        output.message().await.unwrap().unwrap();
        let error = tokio::time::timeout(Duration::from_secs(6), output.message())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let Some(processing_response::Response::ImmediateResponse(error)) = error.response else {
            panic!("expected timeout")
        };
        assert_eq!(error.status.unwrap().code, 504);
        let _ = shutdown.send(());
    }
    #[tokio::test]
    async fn grpc_v1_buffers_before_route_and_passes_long_responses_without_holding_capacity() {
        let (mut client, shutdown) = start().await;
        let (send, receive) = mpsc::channel(2);
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestHeaders(request(vec![]))),
            ..Default::default()
        })
        .await
        .unwrap();
        let mut output = client
            .process(ReceiverStream::new(receive))
            .await
            .unwrap()
            .into_inner();
        let body = br#"{"model":"auto","messages":[{"role":"user","content":"hello"}]}"#;
        for chunk in body.chunks(11) {
            send.send(ext::ProcessingRequest {
                request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                    body: chunk.to_vec(),
                    end_of_stream: false,
                })),
                ..Default::default()
            })
            .await
            .unwrap();
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), output.message())
                .await
                .is_err()
        );
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                body: vec![],
                end_of_stream: true,
            })),
            ..Default::default()
        })
        .await
        .unwrap();
        let Some(processing_response::Response::RequestHeaders(header)) =
            output.message().await.unwrap().unwrap().response
        else {
            panic!("expected routing headers before body")
        };
        assert!(
            header
                .response
                .unwrap()
                .header_mutation
                .unwrap()
                .set_headers
                .into_iter()
                .any(|h| h
                    .header
                    .is_some_and(|h| h.key == TARGET_HEADER && h.raw_value == b"qwen-small"))
        );
        let Some(processing_response::Response::RequestBody(reply)) =
            output.message().await.unwrap().unwrap().response
        else {
            panic!("expected streamed request replacement")
        };
        let Some(ext::body_mutation::Mutation::StreamedResponse(body)) =
            reply.response.unwrap().body_mutation.unwrap().mutation
        else {
            panic!("expected streamed mutation")
        };
        assert!(body.end_of_stream);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body.body).unwrap()["model"],
            "Qwen/Qwen3-0.6B"
        );
        send.send(ext::ProcessingRequest {
            request: Some(processing_request::Request::ResponseHeaders(
                ext::HttpHeaders::default(),
            )),
            ..Default::default()
        })
        .await
        .unwrap();
        assert!(matches!(
            output.message().await.unwrap().unwrap().response,
            Some(processing_response::Response::ResponseHeaders(_))
        ));
        // A response in flight must not consume the sole routing permit.
        let mut second = client
            .process(tokio_stream::iter([
                ext::ProcessingRequest {
                    request: Some(processing_request::Request::RequestHeaders(request(vec![]))),
                    protocol_config: Some(ext::ProtocolConfiguration {
                        request_body_mode: BodySendMode::Buffered as i32,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ext::ProcessingRequest {
                    request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                        body: b"invalid".to_vec(),
                        end_of_stream: true,
                    })),
                    ..Default::default()
                },
            ]))
            .await
            .unwrap()
            .into_inner();
        assert!(matches!(
            second.message().await.unwrap().unwrap().response,
            Some(processing_response::Response::RequestHeaders(_))
        ));
        let Some(processing_response::Response::ImmediateResponse(error)) =
            second.message().await.unwrap().unwrap().response
        else {
            panic!("expected invalid JSON rejection, not capacity rejection")
        };
        assert_eq!(error.status.unwrap().code, 400);
        tokio::time::sleep(STREAM_TIMEOUT + Duration::from_millis(100)).await;
        for (bytes, done) in [
            (b"data: hello\n\n".as_slice(), false),
            (b"data: [DONE]\n\n".as_slice(), true),
        ] {
            send.send(ext::ProcessingRequest {
                request: Some(processing_request::Request::ResponseBody(ext::HttpBody {
                    body: bytes.to_vec(),
                    end_of_stream: done,
                })),
                ..Default::default()
            })
            .await
            .unwrap();
            let Some(processing_response::Response::ResponseBody(reply)) =
                output.message().await.unwrap().unwrap().response
            else {
                panic!("expected response passthrough")
            };
            let Some(ext::body_mutation::Mutation::StreamedResponse(body)) =
                reply.response.unwrap().body_mutation.unwrap().mutation
            else {
                panic!("expected streamed response")
            };
            assert_eq!(body.body, bytes);
            assert_eq!(body.end_of_stream, done);
        }
        assert!(output.message().await.unwrap().is_none());
        let _ = shutdown.send(());
    }

    #[tokio::test]
    async fn grpc_v1_bounds_cumulative_request_chunks_before_acknowledging_headers() {
        let (mut client, shutdown) = start().await;
        let mut output = client
            .process(tokio_stream::iter([
                ext::ProcessingRequest {
                    request: Some(processing_request::Request::RequestHeaders(request(vec![]))),
                    ..Default::default()
                },
                ext::ProcessingRequest {
                    request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                        body: vec![b' '; MAX_BODY],
                        end_of_stream: false,
                    })),
                    ..Default::default()
                },
                ext::ProcessingRequest {
                    request: Some(processing_request::Request::RequestBody(ext::HttpBody {
                        body: vec![b' '],
                        end_of_stream: false,
                    })),
                    ..Default::default()
                },
            ]))
            .await
            .unwrap()
            .into_inner();
        let Some(processing_response::Response::ImmediateResponse(error)) =
            output.message().await.unwrap().unwrap().response
        else {
            panic!("expected size rejection before headers")
        };
        assert_eq!(error.status.unwrap().code, 413);
        let _ = shutdown.send(());
    }
}
