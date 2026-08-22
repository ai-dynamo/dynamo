// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response};
use axum::routing::any;
use reqwest::{Client, Url};
use tokio_util::sync::CancellationToken;

use crate::error::SidecarError;
use crate::metadata::{PrefillEndpoint, strip_epp_headers};
use crate::proxy::{cancel_on_response_drop, forward};

#[async_trait]
pub trait PdAdapter: Send + Sync + 'static {
    async fn execute(
        &self,
        request: Request<Body>,
        prefill_endpoint: PrefillEndpoint,
        cancellation: CancellationToken,
    ) -> Result<Response<Body>, SidecarError>;
}

#[derive(Debug, Default)]
pub struct UnavailablePdAdapter;

#[async_trait]
impl PdAdapter for UnavailablePdAdapter {
    async fn execute(
        &self,
        _request: Request<Body>,
        _prefill_endpoint: PrefillEndpoint,
        _cancellation: CancellationToken,
    ) -> Result<Response<Body>, SidecarError> {
        Err(SidecarError::AdapterUnavailable)
    }
}

#[derive(Clone)]
pub struct SidecarState {
    client: Client,
    decode_engine_url: Url,
    adapter: Arc<dyn PdAdapter>,
}

impl SidecarState {
    pub fn new(
        decode_engine_url: Url,
        connect_timeout: Duration,
        read_timeout: Duration,
        adapter: Arc<dyn PdAdapter>,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(connect_timeout)
                .read_timeout(read_timeout)
                .build()?,
            decode_engine_url,
            adapter,
        })
    }
}

pub fn router(state: SidecarState) -> Router {
    Router::new().fallback(any(handle)).with_state(state)
}

async fn handle(
    State(state): State<SidecarState>,
    mut request: Request<Body>,
) -> Result<Response<Body>, SidecarError> {
    // Parse before stripping so repeated values remain observable and invalid.
    let prefill_endpoint = PrefillEndpoint::parse_headers(request.headers())?;
    strip_epp_headers(request.headers_mut());

    let cancellation = CancellationToken::new();
    // If the handler future is dropped before it can return a response (for
    // example because the Gateway disconnected during adapter dispatch), the
    // guard cancels backend work. Once a response exists, ownership moves to
    // the response-body wrapper so cancellation also covers stream teardown.
    let cancellation_guard = cancellation.clone().drop_guard();
    let response = match prefill_endpoint {
        Some(prefill_endpoint) => {
            state
                .adapter
                .execute(request, prefill_endpoint, cancellation.clone())
                .await?
        }
        None => {
            forward(
                &state.client,
                &state.decode_engine_url,
                request,
                &cancellation,
            )
            .await?
        }
    };

    Ok(cancel_on_response_drop(
        response,
        cancellation_guard.disarm(),
    ))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use axum::body::{Body, Bytes, to_bytes};
    use axum::http::{HeaderValue, Method, Request, Response, StatusCode};
    use axum::routing::any;
    use futures::{StreamExt, stream};
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tower::ServiceExt;

    use super::*;
    use crate::metadata::PREFILLER_HOST_PORT;

    #[derive(Debug)]
    struct ObservedRequest {
        endpoint: String,
        method: Method,
        uri: String,
        headers: axum::http::HeaderMap,
        body: Bytes,
    }

    struct RecordingAdapter {
        observed: Arc<Mutex<Vec<ObservedRequest>>>,
        cancellation_tx: Mutex<Option<oneshot::Sender<CancellationToken>>>,
        pending_stream: bool,
    }

    struct BlockingAdapter {
        cancellation_tx: Mutex<Option<oneshot::Sender<CancellationToken>>>,
    }

    #[async_trait]
    impl PdAdapter for RecordingAdapter {
        async fn execute(
            &self,
            request: Request<Body>,
            prefill_endpoint: PrefillEndpoint,
            cancellation: CancellationToken,
        ) -> Result<Response<Body>, SidecarError> {
            let (parts, body) = request.into_parts();
            let body = to_bytes(body, usize::MAX).await.unwrap();
            self.observed.lock().unwrap().push(ObservedRequest {
                endpoint: prefill_endpoint.to_string(),
                method: parts.method,
                uri: parts.uri.to_string(),
                headers: parts.headers,
                body,
            });
            if let Some(tx) = self.cancellation_tx.lock().unwrap().take() {
                let _ = tx.send(cancellation);
            }

            let response_body = if self.pending_stream {
                Body::from_stream(
                    stream::once(async {
                        Ok::<_, Infallible>(Bytes::from_static(b"data: first\n\n"))
                    })
                    .chain(stream::pending()),
                )
            } else {
                Body::from_stream(stream::iter([
                    Ok::<_, Infallible>(Bytes::from_static(b"data: one\n\n")),
                    Ok::<_, Infallible>(Bytes::from_static(b"data: two\n\n")),
                ]))
            };
            Ok(Response::new(response_body))
        }
    }

    #[async_trait]
    impl PdAdapter for BlockingAdapter {
        async fn execute(
            &self,
            _request: Request<Body>,
            _prefill_endpoint: PrefillEndpoint,
            cancellation: CancellationToken,
        ) -> Result<Response<Body>, SidecarError> {
            if let Some(tx) = self.cancellation_tx.lock().unwrap().take() {
                let _ = tx.send(cancellation.clone());
            }
            cancellation.cancelled().await;
            Err(SidecarError::Cancelled)
        }
    }

    fn test_state(adapter: Arc<dyn PdAdapter>, decode_engine_url: reqwest::Url) -> SidecarState {
        SidecarState::new(
            decode_engine_url,
            Duration::from_secs(10),
            Duration::from_secs(300),
            adapter,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn invalid_metadata_returns_openai_bad_gateway() {
        let app = router(test_state(
            Arc::new(UnavailablePdAdapter),
            reqwest::Url::parse("http://localhost:8001").unwrap(),
        ));
        let request = Request::builder()
            .uri("/v1/chat/completions")
            .header(PREFILLER_HOST_PORT, "prefill:8001,other:8001")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "invalid_epp_metadata");
    }

    #[tokio::test]
    async fn valid_metadata_dispatches_to_adapter_without_leaking_epp_headers() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let adapter = Arc::new(RecordingAdapter {
            observed: observed.clone(),
            cancellation_tx: Mutex::new(None),
            pending_stream: false,
        });
        let app = router(test_state(
            adapter,
            reqwest::Url::parse("http://localhost:8001").unwrap(),
        ));
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions?trace=true")
            .header(PREFILLER_HOST_PORT, "[2001:db8::10]:8001")
            .header("x-gateway-destination-endpoint", "decode:8000")
            .header("x-dynamo-routing-mode", "disaggregated")
            .header("authorization", "Bearer token")
            .body(Body::from(r#"{"model":"test"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            "data: one\n\ndata: two\n\n"
        );

        let observed = observed.lock().unwrap();
        let request = &observed[0];
        assert_eq!(request.endpoint, "[2001:db8::10]:8001");
        assert_eq!(request.method, Method::POST);
        assert_eq!(request.uri, "/v1/chat/completions?trace=true");
        assert_eq!(request.body, r#"{"model":"test"}"#);
        assert_eq!(request.headers["authorization"], "Bearer token");
        assert!(!request.headers.contains_key(PREFILLER_HOST_PORT));
        assert!(
            !request
                .headers
                .contains_key("x-gateway-destination-endpoint")
        );
        assert!(!request.headers.contains_key("x-dynamo-routing-mode"));
    }

    #[tokio::test]
    async fn absent_metadata_streams_decode_passthrough_unchanged() {
        let (observed_tx, observed_rx) = oneshot::channel();
        let observed_tx = Arc::new(Mutex::new(Some(observed_tx)));
        let upstream = Router::new().fallback(any(move |request: Request<Body>| {
            let observed_tx = observed_tx.clone();
            async move {
                let (parts, body) = request.into_parts();
                let body = to_bytes(body, usize::MAX).await.unwrap();
                if let Some(tx) = observed_tx.lock().unwrap().take() {
                    let _ = tx.send((parts, body));
                }
                Response::builder()
                    .status(StatusCode::ACCEPTED)
                    .header("x-upstream", "decode")
                    .body(Body::from_stream(stream::iter([
                        Ok::<_, Infallible>(Bytes::from_static(b"chunk-1")),
                        Ok::<_, Infallible>(Bytes::from_static(b"chunk-2")),
                    ])))
                    .unwrap()
            }
        }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let app = router(test_state(
            Arc::new(UnavailablePdAdapter),
            reqwest::Url::parse(&format!("http://{address}")).unwrap(),
        ));
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/completions?stream=true")
            .header("content-type", "application/json")
            .header("x-gateway-destination-endpoint", "decode:8000")
            .header("x-custom", "preserved")
            .body(Body::from("original-body"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(response.headers()["x-upstream"], "decode");
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            "chunk-1chunk-2"
        );

        let (parts, body) = observed_rx.await.unwrap();
        assert_eq!(parts.method, Method::POST);
        assert_eq!(parts.uri.path(), "/v1/completions");
        assert_eq!(parts.uri.query(), Some("stream=true"));
        assert_eq!(parts.headers["x-custom"], "preserved");
        assert!(!parts.headers.contains_key("x-gateway-destination-endpoint"));
        assert_eq!(body, "original-body");
        server.abort();
    }

    #[tokio::test]
    async fn dropping_response_stream_propagates_cancellation() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (cancellation_tx, cancellation_rx) = oneshot::channel();
        let adapter = Arc::new(RecordingAdapter {
            observed,
            cancellation_tx: Mutex::new(Some(cancellation_tx)),
            pending_stream: true,
        });
        let app = router(test_state(
            adapter,
            reqwest::Url::parse("http://localhost:8001").unwrap(),
        ));
        let request = Request::builder()
            .uri("/v1/chat/completions")
            .header(PREFILLER_HOST_PORT, "prefill:8001")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let cancellation = cancellation_rx.await.unwrap();
        let mut stream = response.into_body().into_data_stream();
        assert_eq!(stream.next().await.unwrap().unwrap(), "data: first\n\n");
        assert!(!cancellation.is_cancelled());
        drop(stream);

        tokio::time::timeout(Duration::from_secs(1), cancellation.cancelled())
            .await
            .expect("dropping the response stream must cancel adapter work");
    }

    #[tokio::test]
    async fn dropping_request_during_dispatch_propagates_cancellation() {
        let (cancellation_tx, cancellation_rx) = oneshot::channel();
        let adapter = Arc::new(BlockingAdapter {
            cancellation_tx: Mutex::new(Some(cancellation_tx)),
        });
        let app = router(test_state(
            adapter,
            reqwest::Url::parse("http://localhost:8001").unwrap(),
        ));
        let request = Request::builder()
            .uri("/v1/chat/completions")
            .header(PREFILLER_HOST_PORT, "prefill:8001")
            .body(Body::empty())
            .unwrap();

        let task = tokio::spawn(app.oneshot(request));
        let cancellation = cancellation_rx.await.unwrap();
        assert!(!cancellation.is_cancelled());
        task.abort();

        tokio::time::timeout(Duration::from_secs(1), cancellation.cancelled())
            .await
            .expect("dropping the handler must cancel adapter work");
    }

    #[tokio::test]
    async fn repeated_metadata_is_rejected() {
        let app = router(test_state(
            Arc::new(UnavailablePdAdapter),
            reqwest::Url::parse("http://localhost:8001").unwrap(),
        ));
        let mut request = Request::builder()
            .uri("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();
        request.headers_mut().append(
            PREFILLER_HOST_PORT,
            HeaderValue::from_static("prefill-a:8001"),
        );
        request.headers_mut().append(
            PREFILLER_HOST_PORT,
            HeaderValue::from_static("prefill-b:8001"),
        );

        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::BAD_GATEWAY
        );
    }
}
