// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend's HTTP server. Each `POST /v1/...` request goes, as
//! received or as a delta the pool frontend rebuilds it from (see
//! [`crate::Deltas`]), to the same route on the pool frontend its selector
//! picks, and the pool's response comes back as sent. Only the request's
//! `model`, and for a request whose headers name no session its input, are
//! read; the pool frontend validates, preprocesses, and formats everything
//! exactly as for a request sent to it directly.

use std::io;
use std::pin::pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dynamo_global_common::delta::{DELTA_HEADER, MISSING, STATE_HEADER, STORE_HEADER};
use dynamo_global_router::{InFlightTracker, Pool, PoolSelector, RequestInput, RoutingRequest};
use dynamo_llm::protocols::common::extensions::session_affinity_from_headers;
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_BODY_LIMIT_MB;
use dynamo_runtime::metrics::prometheus_names::frontend_service::error_type;
use futures::StreamExt;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use serde::Deserialize;
use serde_json::json;
use serde_json::value::RawValue;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use url::Url;

use crate::PoolSnapshot;
use crate::delta::{Conversation, Deltas, Upload};
use crate::metrics::Metrics;

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
    metrics: Metrics,
    deltas: Option<Arc<Deltas>>,
}

/// The Global Frontend's routes over the pools `pools` publishes.
///
/// Each request counts in `in_flight` from its pool choice until its
/// response ends; selectors and signals that weigh this process's own
/// requests must read the same tracker. `http` must not set a total request
/// timeout, because a response streams for as long as the pool generates.
/// `/metrics` serves `metrics`. With `deltas`, requests that continue a
/// conversation go to pool frontends that keep requests as deltas.
pub fn app(
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    selector: Arc<dyn PoolSelector>,
    in_flight: Arc<InFlightTracker>,
    http: reqwest::Client,
    metrics: Metrics,
    deltas: Option<Arc<Deltas>>,
) -> Router {
    let proxy = Arc::new(Proxy {
        pools,
        selector,
        in_flight,
        http,
        metrics,
        deltas,
    });
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/{*route}", post(forward))
        .route("/metrics", get(scrape))
        .route("/health", get(health))
        .route("/live", get(|| async { Json(json!({"status": "live"})) }))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "no such route".to_owned()) })
        .layer(DefaultBodyLimit::max(body_limit()))
        .with_state(proxy)
}

/// Serves `app` on `listener` until `shutdown` fires. Then it stops accepting
/// connections and lets open ones finish their requests; any still open after
/// a grace period are closed, dropping their response bodies.
pub async fn serve(listener: TcpListener, app: Router, shutdown: CancellationToken) {
    // A connection task is polled on every wakeup, i.e. for every forwarded
    // response chunk, and each poll also polls its shutdown signal. Each
    // connection waits on its own child token, whose lock no other connection
    // touches; `axum::serve` has every connection poll one shared `Notify`,
    // which locks its waiter list on every poll of a pending waiter. This
    // mirrors lib/llm/src/http/service/server.rs.
    //
    // Build each route's handler once, as `Router::into_make_service` does,
    // not per request.
    let app = app.with_state(());
    let graceful = CancellationToken::new();
    let hard = CancellationToken::new();
    let _close_connections = hard.clone().drop_guard();
    let connections = TaskTracker::new();
    let mut shutdown = pin!(shutdown.cancelled());
    loop {
        let stream = tokio::select! {
            biased;
            _ = &mut shutdown => break,
            stream = accept(&listener) => stream,
        };
        let connection = serve_connection(stream, app.clone(), graceful.child_token());
        connections.spawn(hard.child_token().run_until_cancelled_owned(connection));
    }

    // Refuse new connections at once instead of leaving them in the backlog.
    drop(listener);
    graceful.cancel();
    connections.close();
    let _ = tokio::time::timeout(SHUTDOWN_GRACE, connections.wait()).await;
}

async fn accept(listener: &TcpListener) -> TcpStream {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                // Streamed tokens are small writes, which Nagle's algorithm
                // would hold for the client's delayed ACK.
                if let Err(error) = stream.set_nodelay(true) {
                    tracing::debug!(%error, "failed to set TCP_NODELAY");
                }
                return stream;
            }
            // Errors such as EMFILE are transient: back off and retry.
            Err(error) => {
                tracing::debug!(%error, "failed to accept a connection");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

async fn serve_connection(stream: TcpStream, app: Router, graceful: CancellationToken) {
    let builder = Builder::new(TokioExecutor::new());
    let service = TowerToHyperService::new(app);
    let mut connection = pin!(builder.serve_connection(TokioIo::new(stream), service));
    if graceful
        .run_until_cancelled(connection.as_mut())
        .await
        .is_none()
    {
        connection.as_mut().graceful_shutdown();
        let _ = connection.await;
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

/// The fields read from a request: its model, and the fields that hold its
/// input (see [`RequestInput`]), borrowed from the body.
#[derive(Deserialize)]
struct Request<'a> {
    model: String,
    #[serde(borrow)]
    system: Option<&'a RawValue>,
    #[serde(borrow)]
    messages: Option<&'a RawValue>,
    #[serde(borrow)]
    input: Option<&'a RawValue>,
}

impl Request<'_> {
    fn input(&self) -> RequestInput<'_> {
        RequestInput {
            system: self.system,
            messages: self.messages,
            input: self.input,
        }
    }
}

async fn forward(
    State(proxy): State<Arc<Proxy>>,
    method: Method,
    uri: Uri,
    mut headers: HeaderMap,
    body: Bytes,
) -> Response {
    let received = Instant::now();
    let metrics = &proxy.metrics;
    let Ok(request) = serde_json::from_slice::<Request>(&body) else {
        metrics.rejected(error_type::VALIDATION);
        let message = "the request body must be a JSON object with a string `model`";
        return error(StatusCode::BAD_REQUEST, message.to_owned());
    };
    let model = &request.model;
    let snapshot = proxy.pools.borrow().clone();
    let candidates = snapshot.pools_for(model).collect::<Vec<_>>();
    if candidates.is_empty() {
        metrics.rejected(error_type::NOT_FOUND);
        return error(
            StatusCode::NOT_FOUND,
            format!("model {model} is not served"),
        );
    }
    remove_hop_headers(&mut headers);
    headers.remove(header::HOST);
    // Deltas are between this process and the pools, never the client's.
    headers.remove(DELTA_HEADER);
    headers.remove(STORE_HEADER);
    let session = session_affinity_from_headers(&headers);
    // Without a named session, the input shows which request this one
    // continues.
    let prefix = match session {
        None if proxy.selector.uses_sessions() => request.input().prefix_keys(model),
        _ => Vec::new(),
    };
    let routing = RoutingRequest::new(model)
        .with_session(session.as_ref().map(|session| session.as_str()))
        .with_prefix(&prefix);
    let Some(target) = proxy.selector.select(&routing, &candidates) else {
        metrics.rejected(error_type::UNAVAILABLE);
        let message = format!("no pool can serve model {model}");
        return error(StatusCode::SERVICE_UNAVAILABLE, message);
    };
    let pool = target.pool;
    let in_flight = proxy.in_flight.start(&pool.id);
    let session = session.as_ref().map(|session| session.as_str());
    let conversation = Conversation::new(session, &prefix);
    let mut upload = match (&proxy.deltas, conversation) {
        (Some(deltas), Some(conversation)) => {
            let authorization = headers.get(header::AUTHORIZATION);
            let authorization = authorization.map(HeaderValue::as_bytes);
            let frontend = target.frontend;
            deltas
                .upload(conversation, body, authorization, pool, frontend)
                .await
        }
        _ => Upload::whole(target.frontend, body),
    };
    let Some(url) = route_url(upload.frontend, &uri) else {
        metrics.rejected(error_type::VALIDATION);
        return error(StatusCode::BAD_REQUEST, "invalid route".to_owned());
    };
    tracing::debug!(pool = %pool.id, %url, kind = upload.kind, "forwarding request");
    // If the client leaves while the pool queues the request, this handler
    // is dropped, which drops the send and closes the pool connection,
    // cancelling the request there, and counts it as cancelled.
    let sending = metrics.sending(pool);
    // A delta's headers are kept to send the whole request if the pool
    // lacks its parent.
    let resend_headers = upload.is_delta().then(|| headers.clone());
    let mut sent = send(proxy.as_ref(), &method, &url, headers, &mut upload, pool).await;
    if let (Some(deltas), Some(headers)) = (&proxy.deltas, resend_headers)
        && sent.as_ref().is_ok_and(lacks_parent)
        && deltas.resend(&mut upload).await
    {
        metrics.delta_missed(pool);
        sent = send(proxy.as_ref(), &method, &url, headers, &mut upload, pool).await;
    }
    let upstream = match sent {
        Ok(upstream) => upstream,
        Err(failure) => {
            sending.answered(None);
            // reqwest names only the outermost failure ("error sending
            // request"); the cause, such as a refused or reset connection,
            // is in its sources.
            let failure = anyhow::Error::new(failure);
            tracing::warn!(
                pool = %pool.id,
                %url,
                error = format!("{failure:#}"),
                "pool frontend failed"
            );
            let message = format!("pool frontend {url} failed: {failure}");
            return error(StatusCode::SERVICE_UNAVAILABLE, message);
        }
    };

    if let Some(deltas) = &proxy.deltas {
        deltas.answered(upload, upstream.headers()).await;
    }
    let status = upstream.status();
    sending.answered(Some(status));
    let mut timing = status
        .is_success()
        .then(|| metrics.response(pool, received));
    let mut headers = upstream.headers().clone();
    remove_hop_headers(&mut headers);
    headers.remove(STATE_HEADER);
    // The stream keeps the request counted in flight, and times it, until it
    // ends or the client drops it; dropping it closes the pool connection.
    let body = upstream.bytes_stream().map(move |chunk| {
        let _in_flight = &in_flight;
        if let Some(timing) = &mut timing {
            timing.chunk(&chunk);
        }
        chunk
    });
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// Sends `upload`'s body to `url` with `headers` and the upload's own. A
/// stream body cannot be copied, so reqwest keeps none for redirects, and
/// the bytes are freed once written to the pool.
async fn send(
    proxy: &Proxy,
    method: &Method,
    url: &Url,
    mut headers: HeaderMap,
    upload: &mut Upload<'_>,
    pool: &Pool,
) -> reqwest::Result<reqwest::Response> {
    let body = std::mem::take(&mut upload.body);
    proxy.metrics.uploaded(pool, upload.kind, body.len());
    headers.insert(header::CONTENT_LENGTH, body.len().into());
    for (name, value) in &upload.headers {
        headers.insert(name, value.clone());
    }
    let body = futures::stream::once(futures::future::ready(Ok::<_, io::Error>(body)));
    proxy
        .http
        .request(method.clone(), url.clone())
        .headers(headers)
        .body(reqwest::Body::wrap_stream(body))
        .send()
        .await
}

/// Whether `response` refuses a delta whose parent the pool does not keep.
fn lacks_parent(response: &reqwest::Response) -> bool {
    response.status() == StatusCode::CONFLICT
        && response
            .headers()
            .get(STATE_HEADER)
            .is_some_and(|state| state == MISSING)
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

async fn scrape(State(proxy): State<Arc<Proxy>>) -> Response {
    let pools = proxy.pools.borrow().clone();
    match proxy.metrics.render(&pools) {
        Ok(text) => ([(header::CONTENT_TYPE, prometheus::TEXT_FORMAT)], text).into_response(),
        Err(failure) => {
            tracing::warn!(error = %failure, "failed to encode metrics");
            let message = "failed to encode metrics".to_owned();
            error(StatusCode::INTERNAL_SERVER_ERROR, message)
        }
    }
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

    #[tokio::test]
    async fn accepted_connections_turn_nagle_off() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let _client = TcpStream::connect(address).await.unwrap();
        let stream = accept(&listener).await;
        assert!(stream.nodelay().unwrap());
    }

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
    fn only_the_model_and_input_are_read() {
        let read = |body: &'static str| serde_json::from_slice::<Request>(body.as_bytes()).ok();
        let request =
            read(r#"{"messages":[{"role":"user","content":"hi"}],"model":"qwen","stream":true}"#)
                .unwrap();
        assert_eq!(request.model, "qwen");
        assert_eq!(request.input().prefix_keys("qwen").len(), 1);
        // The input is not checked here: a request without usable input is
        // still forwarded.
        let request = read(r#"{"model":"qwen","messages":{"role":"user"},"input":7}"#).unwrap();
        assert!(request.input().prefix_keys("qwen").is_empty());
        for body in [r#"{"messages":[]}"#, r#"{"model":7}"#, "[]", "not json"] {
            assert!(read(body).is_none(), "{body}");
        }
    }
}
