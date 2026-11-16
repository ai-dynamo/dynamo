// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{HeaderMap, Request, Response, header};
use futures::{Stream, StreamExt};
use reqwest::{Client, Url};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

use crate::error::SidecarError;

const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

pub async fn forward(
    client: &Client,
    base_url: &Url,
    request: Request<Body>,
    cancellation: &CancellationToken,
) -> Result<Response<Body>, SidecarError> {
    let (parts, body) = request.into_parts();
    let target = target_url(base_url, &parts.uri);

    let mut headers = parts.headers;
    strip_proxy_headers(&mut headers);
    let upstream_request = client
        .request(parts.method, target)
        .headers(headers)
        .body(reqwest::Body::wrap_stream(body.into_data_stream()));

    let upstream_response = tokio::select! {
        response = upstream_request.send() => response.map_err(SidecarError::DecodeUpstream)?,
        () = cancellation.cancelled() => return Err(SidecarError::Cancelled),
    };

    let status = upstream_response.status();
    let mut response_headers = upstream_response.headers().clone();
    strip_proxy_headers(&mut response_headers);
    let body = Body::from_stream(upstream_response.bytes_stream());
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    Ok(response)
}

fn target_url(base_url: &Url, request_uri: &axum::http::Uri) -> Url {
    let mut target = base_url.clone();
    let base_path = base_url.path().trim_end_matches('/');
    let request_path = request_uri.path();
    target.set_path(&format!("{base_path}{request_path}"));
    target.set_query(request_uri.query());
    target.set_fragment(None);
    target
}

fn strip_proxy_headers(headers: &mut HeaderMap) {
    headers.remove(header::HOST);
    headers.remove(header::CONTENT_LENGTH);

    let connection_tokens = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|token| token.trim().parse::<axum::http::HeaderName>().ok())
        .collect::<Vec<_>>();
    for name in connection_tokens {
        headers.remove(name);
    }
    for name in HOP_BY_HOP_HEADERS {
        headers.remove(*name);
    }
}

pub struct CancelOnDropStream<S> {
    inner: S,
    cancellation: CancellationToken,
    cancelled: std::pin::Pin<Box<WaitForCancellationFutureOwned>>,
}

impl<S> CancelOnDropStream<S> {
    pub fn new(inner: S, cancellation: CancellationToken) -> Self {
        let cancelled = Box::pin(cancellation.clone().cancelled_owned());
        Self {
            inner,
            cancellation,
            cancelled,
        }
    }
}

impl<S: Stream + Unpin> Stream for CancelOnDropStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if std::future::Future::poll(self.cancelled.as_mut(), context).is_ready() {
            return std::task::Poll::Ready(None);
        }
        self.inner.poll_next_unpin(context)
    }
}

impl<S> Drop for CancelOnDropStream<S> {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

pub fn cancel_on_response_drop(
    response: Response<Body>,
    cancellation: CancellationToken,
) -> Response<Body> {
    let (parts, body) = response.into_parts();
    let stream = CancelOnDropStream::new(body.into_data_stream(), cancellation);
    Response::from_parts(parts, Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    #[test]
    fn target_url_preserves_base_path_and_request_query() {
        let base_url = Url::parse("http://decode:8001/engine").unwrap();
        let request_uri = "/v1/chat/completions?stream=true".parse().unwrap();

        assert_eq!(
            target_url(&base_url, &request_uri).as_str(),
            "http://decode:8001/engine/v1/chat/completions?stream=true"
        );
    }

    #[test]
    fn strips_standard_and_connection_nominated_hop_by_hop_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONNECTION, HeaderValue::from_static("x-remove"));
        headers.insert("x-remove", HeaderValue::from_static("value"));
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        headers.insert("proxy-connection", HeaderValue::from_static("keep-alive"));
        headers.insert("x-preserved", HeaderValue::from_static("value"));

        strip_proxy_headers(&mut headers);

        assert!(!headers.contains_key(header::CONNECTION));
        assert!(!headers.contains_key("x-remove"));
        assert!(!headers.contains_key("keep-alive"));
        assert!(!headers.contains_key("proxy-connection"));
        assert_eq!(headers["x-preserved"], "value");
    }
}
