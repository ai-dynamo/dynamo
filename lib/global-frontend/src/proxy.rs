// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend's HTTP server. Each `POST /v1/...` request goes, as
//! received, to the same route on the pool frontend its selector picks, and
//! the pool's response comes back as sent. Only the request's `model` is
//! read; the pool frontend validates, preprocesses, and formats everything
//! else exactly as for a request sent to it directly.

use std::future::IntoFuture;
use std::io;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::{Json, Router};
use dynamo_global_router::{InFlightTracker, PoolSelector, RoutingRequest};
use dynamo_llm::protocols::agents::HEADER_DYNAMO_SESSION_ID;
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_BODY_LIMIT_MB;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::PoolSnapshot;

/// The request body limit when `DYN_HTTP_BODY_LIMIT_MB` is unset, as for
/// Dynamo's HTTP service: enough for 500k-token prompts.
const DEFAULT_BODY_LIMIT_MB: usize = 45;
/// How long open requests may run after shutdown starts.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(30);

struct Proxy {
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    selector: Arc<dyn PoolSelector>,
    in_flight: Arc<InFlightTracker>,
    http: reqwest::Client,
}

/// The Global Frontend's routes over the pools `pools` publishes.
///
/// Each request counts in `in_flight` from its pool choice until its
/// response ends; selectors and signals that weigh this process's own
/// requests must read the same tracker. `http` must not set a total request
/// timeout, because a response streams for as long as the pool generates.
pub fn app(
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    selector: Arc<dyn PoolSelector>,
    in_flight: Arc<InFlightTracker>,
    http: reqwest::Client,
) -> Router {
    let proxy = Arc::new(Proxy {
        pools,
        selector,
        in_flight,
        http,
    });
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/{*route}", post(forward))
        .route("/health", get(health))
        .route("/live", get(|| async { Json(json!({"status": "live"})) }))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "no such route".to_owned()) })
        .layer(DefaultBodyLimit::max(body_limit()))
        .with_state(proxy)
}

/// Serves `app` on `listener` until `shutdown` fires, then lets open
/// requests finish for a grace period.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    shutdown: CancellationToken,
) -> io::Result<()> {
    // Streamed tokens are small writes, which Nagle's algorithm would hold
    // for the client's delayed ACK.
    let listener = listener.tap_io(|tcp| {
        if let Err(error) = tcp.set_nodelay(true) {
            tracing::debug!(%error, "failed to set TCP_NODELAY");
        }
    });
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.clone().cancelled_owned())
        .into_future();
    tokio::select! {
        result = server => result,
        _ = async {
            shutdown.cancelled().await;
            tokio::time::sleep(SHUTDOWN_GRACE).await;
        } => Ok(()),
    }
}

fn body_limit() -> usize {
    std::env::var(DYN_HTTP_BODY_LIMIT_MB)
        .ok()
        .and_then(|mb| mb.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BODY_LIMIT_MB)
        * 1024
        * 1024
}

/// The one field read from a request.
#[derive(Deserialize)]
struct Model {
    model: String,
}

async fn forward(
    State(proxy): State<Arc<Proxy>>,
    method: Method,
    uri: Uri,
    mut headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Ok(Model { model }) = serde_json::from_slice(&body) else {
        let message = "the request body must be a JSON object with a string `model`";
        return error(StatusCode::BAD_REQUEST, message.to_owned());
    };
    let snapshot = proxy.pools.borrow().clone();
    let candidates = snapshot.pools_for(&model).collect::<Vec<_>>();
    if candidates.is_empty() {
        return error(
            StatusCode::NOT_FOUND,
            format!("model {model} is not served"),
        );
    }
    remove_hop_headers(&mut headers);
    headers.remove(header::HOST);
    headers.insert(header::CONTENT_LENGTH, body.len().into());
    let session = headers
        .get(HEADER_DYNAMO_SESSION_ID)
        .and_then(|session| session.to_str().ok());
    let request = RoutingRequest::new(&model).with_session(session);
    let Some(target) = proxy.selector.select(&request, &candidates) else {
        let message = format!("no pool can serve model {model}");
        return error(StatusCode::SERVICE_UNAVAILABLE, message);
    };
    let pool = &target.pool.id;
    let in_flight = proxy.in_flight.start(pool);
    let Some(url) = route_url(target.frontend, &uri) else {
        return error(StatusCode::BAD_REQUEST, "invalid route".to_owned());
    };
    tracing::debug!(%pool, %url, "forwarding request");
    // A stream body cannot be copied, so reqwest keeps none for redirects,
    // and the bytes are freed once written to the pool.
    let upload = reqwest::Body::wrap_stream(futures::stream::once(futures::future::ready(Ok::<
        _,
        io::Error,
    >(
        body
    ))));
    // If the client leaves while the pool queues the request, this handler
    // is dropped, which drops the send and closes the pool connection,
    // cancelling the request there.
    let upstream = match proxy
        .http
        .request(method, url.clone())
        .headers(headers)
        .body(upload)
        .send()
        .await
    {
        Ok(upstream) => upstream,
        Err(failure) => {
            // reqwest names only the outermost failure ("error sending
            // request"); the cause, such as a refused or reset connection,
            // is in its sources.
            let failure = anyhow::Error::new(failure);
            tracing::warn!(
                %pool,
                %url,
                error = format!("{failure:#}"),
                "pool frontend failed"
            );
            let message = format!("the pool serving model {model} could not be reached");
            return error(StatusCode::SERVICE_UNAVAILABLE, message);
        }
    };

    let status = upstream.status();
    let mut headers = upstream.headers().clone();
    remove_hop_headers(&mut headers);
    // The stream keeps the request counted in flight until it ends or the
    // client drops it; dropping it closes the pool connection.
    let body = upstream.bytes_stream().map(move |chunk| {
        let _in_flight = &in_flight;
        chunk
    });
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// `uri`'s path and query below `frontend`, keeping any path the frontend
/// URL has. `None` if the path would leave it, for example through `..`.
fn route_url(frontend: &Url, uri: &Uri) -> Option<Url> {
    let path = format!("{}{}", frontend.path().trim_end_matches('/'), uri.path());
    let mut url = frontend.clone();
    url.set_path(&path);
    url.set_query(uri.query());
    (url.path() == path).then_some(url)
}

async fn models(State(proxy): State<Arc<Proxy>>) -> Json<serde_json::Value> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let snapshot = proxy.pools.borrow().clone();
    let data = snapshot
        .models()
        .into_iter()
        .map(|model| json!({"id": model, "object": "model", "created": created, "owned_by": "nvidia"}))
        .collect::<Vec<_>>();
    Json(json!({"object": "list", "data": data}))
}

/// Healthy while some pool has a frontend to send to.
async fn health(State(proxy): State<Arc<Proxy>>) -> Response {
    let reachable = proxy
        .pools
        .borrow()
        .pools()
        .iter()
        .any(|pool| !pool.frontends.is_empty());
    if reachable {
        Json(json!({"status": "healthy"})).into_response()
    } else {
        let body = Json(json!({"status": "unhealthy", "message": "no pool has a frontend"}));
        (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
    }
}

/// An error in the OpenAI shape.
fn error(status: StatusCode, message: String) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let body = json!({"error": {"message": message, "type": kind, "code": status.as_u16()}});
    (status, Json(body)).into_response()
}

/// Removes hop-by-hop headers, which apply to one connection only: the fixed
/// ones, and any the `Connection` header names (RFC 9110).
fn remove_hop_headers(headers: &mut HeaderMap) {
    let named = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect::<Vec<_>>();
    for name in named {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_by_hop_headers_stay_on_their_connection() {
        let mut headers = HeaderMap::new();
        for (name, value) in [
            ("connection", "keep-alive, x-private"),
            ("x-private", "1"),
            ("transfer-encoding", "chunked"),
            ("keep-alive", "timeout=5"),
            ("x-dynamo-session-id", "s"),
            ("content-type", "application/json"),
        ] {
            headers.insert(HeaderName::from_static(name), value.parse().unwrap());
        }
        remove_hop_headers(&mut headers);
        let mut left = headers.keys().map(HeaderName::as_str).collect::<Vec<_>>();
        left.sort();
        assert_eq!(left, ["content-type", "x-dynamo-session-id"]);
    }

    #[test]
    fn routes_stay_below_the_frontend_url() {
        let route = |frontend: &str, uri: &str| {
            route_url(&frontend.parse().unwrap(), &uri.parse().unwrap()).map(String::from)
        };
        assert_eq!(
            route("http://pool:8000", "/v1/chat/completions?x=1").as_deref(),
            Some("http://pool:8000/v1/chat/completions?x=1")
        );
        assert_eq!(
            route("http://gateway:8080/ohio/", "/v1/responses").as_deref(),
            Some("http://gateway:8080/ohio/v1/responses")
        );
        assert_eq!(route("http://pool:8000", "/v1/../metrics"), None);
    }

    #[test]
    fn only_the_model_is_read() {
        let model = |body: &str| {
            serde_json::from_slice::<Model>(body.as_bytes())
                .ok()
                .map(|Model { model }| model)
        };
        assert_eq!(
            model(r#"{"messages":[{"role":"user","content":"hi"}],"model":"qwen","stream":true}"#)
                .as_deref(),
            Some("qwen")
        );
        for body in [r#"{"messages":[]}"#, r#"{"model":7}"#, "[]", "not json"] {
            assert_eq!(model(body), None, "{body}");
        }
    }
}
