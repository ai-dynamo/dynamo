// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requests the Global Frontend sends as deltas against the last request of
//! their conversation (see [`dynamo_global_common::delta`]). With
//! `DYN_HTTP_DELTA_STATE_MB` set, the frontend keeps that last request, up to
//! that much memory, rebuilds each delta into the request the client sent
//! before any handler reads it, and accepts zstd-compressed request bodies.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use dynamo_global_common::delta::{self, Delta, Digest, MISSING, STATE_HEADER};
use dynamo_runtime::config::environment_names::llm as env_llm;
use moka::future::Cache;

/// How long a conversation's last request is kept without a next one.
const IDLE: Duration = Duration::from_secs(30 * 60);

/// The last request of each conversation, by its digest.
#[derive(Clone, Debug)]
pub struct DeltaStore {
    states: Cache<Digest, Arc<delta::State>>,
}

impl DeltaStore {
    /// A store of up to `DYN_HTTP_DELTA_STATE_MB` MiB; `None` unless it is
    /// set above zero.
    pub fn from_env() -> Option<Self> {
        let mib = std::env::var(env_llm::DYN_HTTP_DELTA_STATE_MB)
            .ok()?
            .parse::<u64>()
            .ok()
            .filter(|mib| *mib > 0)?;
        Some(Self::new(mib * 1024 * 1024))
    }

    /// A store of up to roughly `bytes` of requests. The least used go first.
    pub fn new(bytes: u64) -> Self {
        let states = Cache::builder()
            .max_capacity(bytes)
            .weigher(|_, state: &Arc<delta::State>| {
                u32::try_from(state.weight()).unwrap_or(u32::MAX)
            })
            .time_to_idle(IDLE)
            .build();
        Self { states }
    }
}

/// Keeps each request that asks to be kept, and rebuilds each delta into the
/// request it stands for, then keeps that in place of its parent. The
/// response names what was kept. A delta whose parent is not kept, or does
/// not match, gets 409 before anything else reads it, and its sender sends
/// the whole request instead.
pub async fn rebuild_deltas(
    State(store): State<DeltaStore>,
    request: Request,
    next: Next,
) -> Response {
    let delta = match request.headers().get(delta::DELTA_HEADER) {
        Some(value) => match value.to_str().ok().and_then(|value| value.parse().ok()) {
            Some(delta) => Some(delta),
            None => return rejection(StatusCode::BAD_REQUEST, "malformed x-dynamo-delta"),
        },
        None if request.headers().contains_key(delta::STORE_HEADER) => None,
        None => return next.run(request).await,
    };
    let (mut parts, body) = request.into_parts();
    let Ok(body) = axum::body::to_bytes(body, super::openai::get_body_limit()).await else {
        return rejection(StatusCode::PAYLOAD_TOO_LARGE, "request body is too large");
    };
    let authorization = parts.headers.get(header::AUTHORIZATION);
    let authorization = authorization.map(HeaderValue::as_bytes);
    let (body, kept) = match delta {
        // A request without a list is passed on, unkept.
        None => {
            let kept = delta::Request::parse(&body)
                .map(|request| delta::State::new(&request, authorization));
            (body, kept)
        }
        Some(delta) => match rebuild(&store, &delta, &body, authorization).await {
            Some(kept) => (Bytes::from(kept.body()), Some(kept)),
            None => {
                let mut response = rejection(StatusCode::CONFLICT, "unknown delta parent");
                let missing = HeaderValue::from_static(MISSING);
                response.headers_mut().insert(STATE_HEADER, missing);
                return response;
            }
        },
    };
    let kept = match kept {
        Some(kept) => {
            let id = kept.id;
            store.states.insert(id, Arc::new(kept)).await;
            if let Some(delta) = delta {
                store.states.invalidate(&delta.parent).await;
            }
            Some(id)
        }
        None => None,
    };
    parts.headers.remove(delta::DELTA_HEADER);
    parts.headers.remove(delta::STORE_HEADER);
    parts
        .headers
        .insert(header::CONTENT_LENGTH, body.len().into());
    let mut response = next.run(Request::from_parts(parts, Body::from(body))).await;
    if let Some(id) = kept
        && let Ok(id) = HeaderValue::from_str(&id.to_string())
    {
        response.headers_mut().insert(STATE_HEADER, id);
    }
    response
}

/// The request `delta` with `body` makes of its kept parent; `None` if the
/// parent is not kept or `delta` does not extend it.
async fn rebuild(
    store: &DeltaStore,
    delta: &Delta,
    body: &[u8],
    authorization: Option<&[u8]>,
) -> Option<delta::State> {
    let parent = store.states.get(&delta.parent).await?;
    parent.apply(delta, &delta::Request::parse(body)?, authorization)
}

fn rejection(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}});
    (status, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::routing::post;
    use dynamo_global_common::delta::{Parent, STORE_HEADER};
    use tower::ServiceExt;
    use tower_http::decompression::RequestDecompressionLayer;

    const FIRST: &str = r#"{"model":"m","messages":[{"role":"user","content":"a"}]}"#;
    const SECOND: &str =
        r#"{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}"#;

    /// A frontend whose only route answers with the body it was given.
    fn frontend(store: DeltaStore) -> Router {
        Router::new()
            .route(
                "/v1/chat/completions",
                post(|body: Bytes| async move { body }),
            )
            .layer(axum::middleware::from_fn_with_state(store, rebuild_deltas))
            .layer(RequestDecompressionLayer::new())
    }

    async fn send(frontend: &Router, headers: &[(&str, &str)], body: Vec<u8>) -> Response {
        let mut request = Request::post("/v1/chat/completions");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let request = request.body(Body::from(body)).unwrap();
        frontend.clone().oneshot(request).await.unwrap()
    }

    async fn text(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        String::from_utf8(bytes.unwrap().to_vec()).unwrap()
    }

    fn json(text: &str) -> serde_json::Value {
        serde_json::from_str(text).unwrap()
    }

    /// The delta header and body that send `SECOND` against `FIRST`.
    fn second_as_delta() -> (Delta, Vec<u8>) {
        let first = delta::Request::parse(FIRST.as_bytes()).unwrap();
        let parent = Parent::new(first.state(None), &first);
        parent
            .delta(&delta::Request::parse(SECOND.as_bytes()).unwrap())
            .unwrap()
    }

    #[tokio::test]
    async fn a_delta_reaches_the_handler_as_the_whole_request() {
        let frontend = frontend(DeltaStore::new(1 << 20));
        let stored = send(&frontend, &[(STORE_HEADER, "1")], FIRST.into()).await;
        let first = delta::Request::parse(FIRST.as_bytes()).unwrap();
        let kept = first.state(None).to_string();
        assert_eq!(stored.headers()[STATE_HEADER], kept.as_str());
        assert_eq!(text(stored).await, FIRST);

        let (header, body) = second_as_delta();
        let compressed = zstd::bulk::compress(&body, 1).unwrap();
        let headers = [
            (delta::DELTA_HEADER, header.to_string()),
            ("content-encoding", "zstd".to_owned()),
        ];
        let headers = headers
            .each_ref()
            .map(|(name, value)| (*name, value.as_str()));
        let rebuilt = send(&frontend, &headers, compressed).await;
        assert_eq!(rebuilt.status(), StatusCode::OK);
        let second = delta::Request::parse(SECOND.as_bytes()).unwrap();
        let next = second.state(None).to_string();
        assert_eq!(rebuilt.headers()[STATE_HEADER], next.as_str());
        assert_eq!(json(&text(rebuilt).await), json(SECOND));

        // The parent made way for its child.
        let header = header.to_string();
        let again = send(&frontend, &[(delta::DELTA_HEADER, &header)], body).await;
        assert_eq!(again.status(), StatusCode::CONFLICT);
        assert_eq!(again.headers()[STATE_HEADER], MISSING);
    }

    #[tokio::test]
    async fn a_delta_without_its_parent_is_refused() {
        let frontend = frontend(DeltaStore::new(1 << 20));
        let (header, body) = second_as_delta();
        let header = header.to_string();
        let refused = send(&frontend, &[(delta::DELTA_HEADER, &header)], body.clone()).await;
        assert_eq!(refused.status(), StatusCode::CONFLICT);
        assert_eq!(refused.headers()[STATE_HEADER], MISSING);

        // Other credentials do not reach the parent either.
        send(&frontend, &[(STORE_HEADER, "1")], FIRST.into()).await;
        let headers = [
            (delta::DELTA_HEADER, header.as_str()),
            ("authorization", "Bearer x"),
        ];
        assert_eq!(
            send(&frontend, &headers, body).await.status(),
            StatusCode::CONFLICT
        );
        let malformed = [(delta::DELTA_HEADER, "parent=1")];
        let malformed = send(&frontend, &malformed, Vec::new()).await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn other_requests_pass_through_unkept() {
        let frontend = frontend(DeltaStore::new(1 << 20));
        let plain = send(&frontend, &[], FIRST.into()).await;
        assert!(!plain.headers().contains_key(STATE_HEADER));
        assert_eq!(text(plain).await, FIRST);
        let prompt = r#"{"model":"m","prompt":"a"}"#;
        let unlisted = send(&frontend, &[(STORE_HEADER, "1")], prompt.into()).await;
        assert!(!unlisted.headers().contains_key(STATE_HEADER));
        assert_eq!(text(unlisted).await, prompt);
    }
}
