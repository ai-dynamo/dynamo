// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::{HeaderMap, Request, Response, header};
use futures::{Stream, StreamExt};
use reqwest::{Client, Url};
use tokio_util::sync::CancellationToken;

use crate::error::SidecarError;

const HOP_BY_HOP_HEADERS: &[axum::http::HeaderName] = &[
    header::CONNECTION,
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

pub async fn forward(
    client: &Client,
    base_url: &Url,
    request: Request<Body>,
    cancellation: &CancellationToken,
) -> Result<Response<Body>, SidecarError> {
    let (parts, body) = request.into_parts();
    let mut target = base_url.clone();
    target.set_path(parts.uri.path());
    target.set_query(parts.uri.query());

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
        headers.remove(name);
    }
}

pub struct CancelOnDropStream<S> {
    inner: S,
    cancellation: CancellationToken,
}

impl<S> CancelOnDropStream<S> {
    pub fn new(inner: S, cancellation: CancellationToken) -> Self {
        Self {
            inner,
            cancellation,
        }
    }
}

impl<S: Stream + Unpin> Stream for CancelOnDropStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
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
