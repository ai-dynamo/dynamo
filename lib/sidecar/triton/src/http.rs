// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Triton KServe v2 HTTP/REST transport.

mod codec;

use std::collections::BTreeSet;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_sidecar_common::{
    DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcTransportConfig, HttpEndpoint, cannot_connect,
    connection_timeout, format_error_chain, invalid_argument,
};
use reqwest::{RequestBuilder, Response, StatusCode, header};
use serde::Deserialize;

use crate::client::wait_until_ready;

/// Largest request or response body, matching the gRPC message limit.
const MAX_BODY_SIZE: usize = DEFAULT_MAX_GRPC_MESSAGE_SIZE;

#[derive(Clone)]
pub(crate) struct HttpClient {
    client: reqwest::Client,
    base: url::Url,
}

#[derive(Deserialize)]
struct RepositoryIndexEntry {
    name: String,
    #[serde(default)]
    state: String,
}

impl HttpClient {
    pub(crate) async fn connect(
        endpoint: &HttpEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let base = url::Url::parse(endpoint.as_str()).map_err(|error| {
            invalid_argument(format!(
                "invalid Triton HTTP endpoint `{endpoint}`: {error}"
            ))
        })?;
        let client = reqwest::Client::builder()
            .connect_timeout(transport.connect_attempt_timeout)
            .build()
            .map_err(|error| {
                invalid_argument(format!("could not configure Triton HTTP client: {error}"))
            })?;
        let client = Self { client, base };
        client.wait_until_ready(transport).await?;
        Ok(client)
    }

    /// `segments` appended to the endpoint path, each percent-encoded.
    fn url(&self, segments: &[&str]) -> url::Url {
        let mut url = self.base.clone();
        url.path_segments_mut()
            .expect("HTTP(S) endpoints always have a hierarchical path")
            .pop_if_empty()
            .extend(segments);
        url
    }

    async fn wait_until_ready(&self, transport: GrpcTransportConfig) -> Result<(), DynamoError> {
        let url = self.url(&["v2", "health", "ready"]);
        wait_until_ready("HTTP", transport, || async {
            let response = self
                .client
                .get(url.clone())
                .send()
                .await
                .map_err(|error| format_error_chain(&error))?;
            let status = response.status();
            if status.is_success() {
                Ok(true)
            } else {
                Err(format!("HTTP {status}"))
            }
        })
        .await
    }

    pub(crate) async fn model_ready(&self, model: &str) -> Result<bool, DynamoError> {
        const OPERATION: &str = "ModelReady";
        let response = self
            .client
            .get(self.url(&["v2", "models", model, "ready"]))
            .send()
            .await
            .map_err(|error| request_error(OPERATION, &error))?;
        let status = response.status();
        // Triton answers a model that is unknown or still loading with a
        // client error rather than a body, unlike gRPC's `ready: false`.
        if status.is_success() {
            Ok(true)
        } else if status.is_client_error() && !is_auth_failure(status) {
            Ok(false)
        } else {
            Err(response_error(OPERATION, response).await)
        }
    }

    /// Names of the models Triton reports as READY, sorted and de-duplicated.
    pub(crate) async fn ready_models(&self) -> Result<Vec<String>, DynamoError> {
        const OPERATION: &str = "RepositoryIndex";
        let request = self
            .client
            .post(self.url(&["v2", "repository", "index"]))
            .json(&serde_json::json!({"ready": true}));
        let body = self.send(OPERATION, request).await?;
        let entries: Vec<RepositoryIndexEntry> =
            serde_json::from_slice(&body.bytes).map_err(|error| {
                protocol_error(format!("invalid repository index response: {error}"))
            })?;
        Ok(entries
            .into_iter()
            .filter(|entry| entry.state.eq_ignore_ascii_case("READY"))
            .map(|entry| entry.name)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    pub(crate) async fn model_config(
        &self,
        model: &str,
    ) -> Result<inference::ModelConfig, DynamoError> {
        let request = self
            .client
            .get(self.url(&["v2", "models", model, "config"]));
        let body = self.send("ModelConfig", request).await?;
        codec::model_config_from_json(&body.bytes)
    }

    pub(crate) async fn infer(
        &self,
        request: inference::ModelInferRequest,
    ) -> Result<inference::ModelInferResponse, DynamoError> {
        const OPERATION: &str = "ModelInfer";
        let url = if request.model_version.is_empty() {
            self.url(&["v2", "models", &request.model_name, "infer"])
        } else {
            self.url(&[
                "v2",
                "models",
                &request.model_name,
                "versions",
                &request.model_version,
                "infer",
            ])
        };
        let encoded = codec::encode_infer_request(&request)?;
        drop(request);
        if encoded.body.len() > MAX_BODY_SIZE {
            return Err(invalid_argument(format!(
                "{OPERATION}: request body of {} bytes exceeds the {MAX_BODY_SIZE}-byte limit",
                encoded.body.len()
            )));
        }

        let mut http = self.client.post(url);
        http = if encoded.has_binary_data() {
            http.header(header::CONTENT_TYPE, "application/octet-stream")
                .header(codec::HEADER_LENGTH, encoded.header_length)
        } else {
            http.header(header::CONTENT_TYPE, "application/json")
        };
        let body = self.send(OPERATION, http.body(encoded.body)).await?;
        codec::decode_infer_response(&body.bytes, body.header_length)
    }

    /// Send `request` and read a successful response body.
    async fn send(
        &self,
        operation: &str,
        request: RequestBuilder,
    ) -> Result<ResponseBody, DynamoError> {
        let response = request
            .send()
            .await
            .map_err(|error| request_error(operation, &error))?;
        if !response.status().is_success() {
            return Err(response_error(operation, response).await);
        }
        let header_length = response
            .headers()
            .get(codec::HEADER_LENGTH)
            .map(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .ok_or_else(|| {
                        protocol_error(format!("invalid {} header {value:?}", codec::HEADER_LENGTH))
                    })
            })
            .transpose()?;
        let bytes = read_body(operation, response).await?;
        Ok(ResponseBody {
            bytes,
            header_length,
        })
    }
}

struct ResponseBody {
    bytes: Vec<u8>,
    header_length: Option<usize>,
}

/// Read at most [`MAX_BODY_SIZE`] bytes of `response`.
async fn read_body(operation: &str, mut response: Response) -> Result<Vec<u8>, DynamoError> {
    let too_large = || {
        backend_error(
            BackendError::Unknown,
            format!("{operation}: response body exceeds the {MAX_BODY_SIZE}-byte limit"),
        )
    };
    let expected = response.content_length().unwrap_or(0);
    if expected > MAX_BODY_SIZE as u64 {
        return Err(too_large());
    }
    let mut body = Vec::with_capacity(expected as usize);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| request_error(operation, &error))?
    {
        if body.len() + chunk.len() > MAX_BODY_SIZE {
            return Err(too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn backend_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    backend_error(
        BackendError::Unknown,
        format!("invalid Triton HTTP response: {}", message.into()),
    )
}

fn request_error(operation: &str, error: &reqwest::Error) -> DynamoError {
    let message = format!("{operation}: {}", format_error_chain(error));
    if error.is_timeout() {
        connection_timeout(message)
    } else if error.is_connect() {
        cannot_connect(message)
    } else {
        backend_error(BackendError::Unknown, message)
    }
}

fn is_auth_failure(status: StatusCode) -> bool {
    matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
}

/// Map a non-success Triton response the way `status_to_dynamo` maps the
/// equivalent gRPC status codes.
async fn response_error(operation: &str, response: Response) -> DynamoError {
    let status = response.status();
    let detail = match read_body(operation, response).await {
        Ok(body) => error_detail(&body),
        Err(error) => error.to_string(),
    };
    let kind = match status.as_u16() {
        400 | 404 | 409 | 412 | 422 => BackendError::InvalidArgument,
        408 | 504 => BackendError::ConnectionTimeout,
        499 => BackendError::Cancelled,
        502 | 503 => BackendError::CannotConnect,
        _ => BackendError::Unknown,
    };
    let auth = if is_auth_failure(status) {
        "; the sidecar does not have backend authentication configured"
    } else {
        ""
    };
    backend_error(kind, format!("{operation}: {detail} (HTTP {status}){auth}"))
}

/// Triton reports failures as `{"error": "..."}`; fall back to the raw body.
fn error_detail(body: &[u8]) -> String {
    #[derive(Deserialize)]
    struct ErrorBody {
        error: String,
    }
    const MAX_DETAIL: usize = 1024;
    if let Ok(body) = serde_json::from_slice::<ErrorBody>(body) {
        return body.error;
    }
    let detail = String::from_utf8_lossy(&body[..body.len().min(MAX_DETAIL)]);
    let detail = detail.trim();
    if detail.is_empty() {
        "no error detail".to_string()
    } else {
        detail.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_backend_common::{BackendError, ErrorType};
    use dynamo_llm::grpc::service::kserve::inference;
    use dynamo_sidecar_common::{GrpcTransportConfig, HttpEndpoint};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::HttpClient;

    struct Reply {
        status: &'static str,
        headers: Vec<(&'static str, String)>,
        body: Vec<u8>,
    }

    impl Reply {
        fn json(status: &'static str, body: serde_json::Value) -> Self {
            Self {
                status,
                headers: Vec::new(),
                body: body.to_string().into_bytes(),
            }
        }
    }

    /// Serve one canned reply per connection and return each request's
    /// request line and body.
    async fn serve(
        replies: Vec<Reply>,
    ) -> (String, tokio::task::JoinHandle<Vec<(String, Vec<u8>)>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/triton", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                let body_start = loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let head = String::from_utf8_lossy(&request[..body_start]).to_string();
                let length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                while request.len() < body_start + length {
                    let read = socket.read(&mut buffer).await.unwrap();
                    request.extend_from_slice(&buffer[..read]);
                }
                requests.push((
                    head.lines().next().unwrap().to_string(),
                    request[body_start..].to_vec(),
                ));

                let mut response = format!(
                    "HTTP/1.1 {}\r\nContent-Length: {}\r\nConnection: close\r\n",
                    reply.status,
                    reply.body.len()
                );
                for (name, value) in reply.headers {
                    response.push_str(&format!("{name}: {value}\r\n"));
                }
                response.push_str("\r\n");
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.write_all(&reply.body).await.unwrap();
            }
            requests
        });
        (endpoint, task)
    }

    fn transport() -> GrpcTransportConfig {
        GrpcTransportConfig {
            retry_interval: Duration::from_millis(10),
            startup_deadline: Duration::from_secs(5),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn serves_triton_operations_over_http() {
        let mut infer_body = serde_json::to_vec(&serde_json::json!({
            "model_name": "my model",
            "outputs": [{
                "name": "OUTPUT0",
                "datatype": "INT32",
                "shape": [1],
                "parameters": {"binary_data_size": 4},
            }],
        }))
        .unwrap();
        let header_length = infer_body.len();
        infer_body.extend_from_slice(&9_i32.to_le_bytes());

        let (endpoint, server) = serve(vec![
            Reply::json("503 Service Unavailable", serde_json::json!({})),
            Reply::json("200 OK", serde_json::json!({})),
            Reply::json(
                "200 OK",
                serde_json::json!([
                    {"name": "b", "state": "READY"},
                    {"name": "a", "state": "READY"},
                    {"name": "c", "state": "UNAVAILABLE"},
                ]),
            ),
            Reply::json("400 Bad Request", serde_json::json!({"error": "not ready"})),
            Reply::json(
                "200 OK",
                serde_json::json!({"name": "my model", "max_batch_size": 4}),
            ),
            Reply {
                status: "200 OK",
                headers: vec![("Inference-Header-Content-Length", header_length.to_string())],
                body: infer_body,
            },
            Reply::json(
                "400 Bad Request",
                serde_json::json!({"error": "unexpected shape"}),
            ),
        ])
        .await;

        let endpoint = HttpEndpoint::parse(&endpoint, "--http-endpoint").unwrap();
        let client = HttpClient::connect(&endpoint, transport()).await.unwrap();
        assert_eq!(client.ready_models().await.unwrap(), ["a", "b"]);
        assert!(!client.model_ready("my model").await.unwrap());
        assert_eq!(
            client
                .model_config("my model")
                .await
                .unwrap()
                .max_batch_size,
            4
        );

        let request = inference::ModelInferRequest {
            model_name: "my model".to_string(),
            inputs: vec![inference::model_infer_request::InferInputTensor {
                name: "INPUT0".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1],
                ..Default::default()
            }],
            raw_input_contents: vec![5_i32.to_le_bytes().to_vec()],
            ..Default::default()
        };
        let response = client.infer(request.clone()).await.unwrap();
        assert_eq!(response.raw_output_contents, [9_i32.to_le_bytes().to_vec()]);

        let error = client.infer(request).await.unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(error.to_string().contains("ModelInfer: unexpected shape"));

        let requests = server.await.unwrap();
        let lines: Vec<_> = requests.iter().map(|(line, _)| line.as_str()).collect();
        assert_eq!(
            lines,
            [
                "GET /triton/v2/health/ready HTTP/1.1",
                "GET /triton/v2/health/ready HTTP/1.1",
                "POST /triton/v2/repository/index HTTP/1.1",
                "GET /triton/v2/models/my%20model/ready HTTP/1.1",
                "GET /triton/v2/models/my%20model/config HTTP/1.1",
                "POST /triton/v2/models/my%20model/infer HTTP/1.1",
                "POST /triton/v2/models/my%20model/infer HTTP/1.1",
            ]
        );
        assert_eq!(requests[2].1, br#"{"ready":true}"#);
        assert!(requests[5].1.ends_with(&5_i32.to_le_bytes()));
    }
}
