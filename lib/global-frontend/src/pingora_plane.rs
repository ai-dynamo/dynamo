// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend's data plane on Pingora (`--data-plane pingora`, a
//! prototype). Its routes, errors and metrics are those of [`crate::app`].
//!
//! Pingora buffers each `POST /v1/...` body before choosing an upstream, so
//! the request's `model` picks the pool; the body then goes to the pool as
//! received. One task per request drives both the client's connection and the
//! pool's, so a streamed response chunk is not handed between tasks.

use std::net::ToSocketAddrs;
use std::os::fd::IntoRawFd;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axum::body::Bytes;
use axum::http::{Method, StatusCode, Uri, header};
use dynamo_global_router::{InFlightGuard, InFlightTracker, Pool, PoolSelector, RoutingRequest};
use dynamo_llm::protocols::agents::HEADER_DYNAMO_SESSION_ID;
use dynamo_runtime::metrics::prometheus_names::frontend_service::error_type;
use pingora_core::server::Fds;
use pingora_core::server::configuration::ServerConf;
use pingora_core::services::Service as _;
use pingora_core::upstreams::peer::HttpPeer;
use pingora_core::{Error, ErrorSource, ErrorType, Result};
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{FailToProxy, ProxyHttp, Session};
use serde_json::json;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::{Position, Url};

use crate::PoolSnapshot;
use crate::metrics::{Metrics, ResponseMetrics};
use crate::proxy::{self, Model};

/// Idle pool connections kept for reuse: about the concurrency served.
const UPSTREAM_POOL_SIZE: usize = 16_384;
/// How long a client may take to send a request body.
const BODY_TIMEOUT: Duration = Duration::from_secs(60);
/// How often shutdown checks whether requests in flight have ended.
const DRAIN_POLL: Duration = Duration::from_millis(50);

/// The Global Frontend's request handling, as Pingora's `ProxyHttp`.
pub struct PingoraProxy {
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    selector: Arc<dyn PoolSelector>,
    in_flight: Arc<InFlightTracker>,
    metrics: Metrics,
    connect_timeout: Duration,
}

impl PingoraProxy {
    /// Routes over the pools `pools` publishes, as [`crate::app`] does.
    /// `connect_timeout` bounds connecting to a pool frontend.
    pub fn new(
        pools: watch::Receiver<Arc<PoolSnapshot>>,
        selector: Arc<dyn PoolSelector>,
        in_flight: Arc<InFlightTracker>,
        metrics: Metrics,
        connect_timeout: Duration,
    ) -> Self {
        Self {
            pools,
            selector,
            in_flight,
            metrics,
            connect_timeout,
        }
    }

    /// Picks the pool for a buffered `POST /v1/...` request, or answers it.
    /// `Ok(true)` means the request was answered here.
    async fn route(&self, session: &mut Session, ctx: &mut Ctx) -> Result<bool> {
        let received = Instant::now();
        let model = session
            .get_buffered_body()
            .and_then(|body| serde_json::from_slice::<Model>(body).ok());
        let Some(Model { model }) = model else {
            self.metrics.rejected(error_type::VALIDATION);
            let message = "the request body must be a JSON object with a string `model`";
            return respond_error(session, StatusCode::BAD_REQUEST, message).await;
        };
        let snapshot = self.pools.borrow().clone();
        let candidates = snapshot.pools_for(&model).collect::<Vec<_>>();
        if candidates.is_empty() {
            self.metrics.rejected(error_type::NOT_FOUND);
            let message = format!("model {model} is not served");
            return respond_error(session, StatusCode::NOT_FOUND, &message).await;
        }
        let request = session.req_header();
        let session_id = request
            .headers
            .get(HEADER_DYNAMO_SESSION_ID)
            .and_then(|session| session.to_str().ok());
        let routing = RoutingRequest::new(&model).with_session(session_id);
        let Some(target) = self.selector.select(&routing, &candidates) else {
            self.metrics.rejected(error_type::UNAVAILABLE);
            let message = format!("no pool can serve model {model}");
            return respond_error(session, StatusCode::SERVICE_UNAVAILABLE, &message).await;
        };
        let in_flight = self.in_flight.start(&target.pool.id);
        let pool = snapshot
            .pools()
            .iter()
            .position(|pool| std::ptr::eq(pool, target.pool))
            .expect("the selector picks one of the candidates");
        let Some(url) = proxy::route_url(target.frontend, &request.uri) else {
            self.metrics.rejected(error_type::VALIDATION);
            return respond_error(session, StatusCode::BAD_REQUEST, "invalid route").await;
        };
        let path = url[Position::BeforePath..Position::AfterQuery]
            .parse::<Uri>()
            .expect("a URL's path and query form a URI");
        let host = url[Position::BeforeHost..Position::AfterPort].to_owned();
        ctx.route = Some(Route {
            peer: peer(&url, self.connect_timeout),
            url,
            path,
            host,
            snapshot,
            pool,
            received,
            _in_flight: in_flight,
            answered: false,
            timing: None,
        });
        Ok(false)
    }
}

/// One request's state across Pingora's hooks.
#[derive(Default)]
pub struct Ctx {
    route: Option<Route>,
}

/// A request on its way to a pool.
struct Route {
    /// `None` if the frontend's address does not resolve; connecting fails.
    peer: Option<HttpPeer>,
    url: Url,
    path: Uri,
    host: String,
    snapshot: Arc<PoolSnapshot>,
    pool: usize,
    received: Instant,
    /// Counts the request in flight until the route drops, when the request ends.
    _in_flight: InFlightGuard,
    /// Whether the pool's response headers arrived.
    answered: bool,
    /// A successful response's timing; dropping it records the duration.
    timing: Option<ResponseMetrics>,
}

impl Route {
    fn pool(&self) -> &Pool {
        &self.snapshot.pools()[self.pool]
    }
}

#[async_trait]
impl ProxyHttp for PingoraProxy {
    type CTX = Ctx;

    fn new_ctx(&self) -> Ctx {
        Ctx::default()
    }

    fn early_request_body_buffer_limit(&self, session: &Session, _ctx: &Ctx) -> Option<usize> {
        let request = session.req_header();
        (request.method == Method::POST && request.uri.path().starts_with("/v1/"))
            .then(proxy::body_limit)
    }

    fn early_request_body_buffer_timeout(
        &self,
        _session: &Session,
        _ctx: &Ctx,
    ) -> Option<Duration> {
        Some(BODY_TIMEOUT)
    }

    async fn request_filter(&self, session: &mut Session, ctx: &mut Ctx) -> Result<bool> {
        let request = session.req_header();
        let method = request.method.clone();
        let path = request.uri.path().to_owned();
        match (method, path.as_str()) {
            (Method::GET, "/v1/models") => {
                let body = proxy::models_json(&self.pools.borrow());
                respond_json(session, StatusCode::OK, &body).await
            }
            (Method::GET, "/health") => {
                let (status, body) = proxy::health_json(&self.pools.borrow());
                respond_json(session, status, &body).await
            }
            (Method::GET, "/live") => {
                respond_json(session, StatusCode::OK, &json!({"status": "live"})).await
            }
            (Method::GET, "/metrics") => {
                let pools = self.pools.borrow().clone();
                match self.metrics.render(&pools) {
                    Ok(text) => {
                        let body = Bytes::from(text);
                        respond(session, StatusCode::OK, prometheus::TEXT_FORMAT, body).await
                    }
                    Err(failure) => {
                        tracing::warn!(error = %failure, "failed to encode metrics");
                        let status = StatusCode::INTERNAL_SERVER_ERROR;
                        respond_error(session, status, "failed to encode metrics").await
                    }
                }
            }
            (Method::POST, path) if path.starts_with("/v1/") => self.route(session, ctx).await,
            _ => respond_error(session, StatusCode::NOT_FOUND, "no such route").await,
        }
    }

    async fn upstream_peer(&self, _session: &mut Session, ctx: &mut Ctx) -> Result<Box<HttpPeer>> {
        let route = ctx
            .route
            .as_ref()
            .expect("request_filter routed the request");
        match &route.peer {
            Some(peer) => Ok(Box::new(peer.clone())),
            None => Error::e_explain(
                ErrorType::ConnectNoRoute,
                format!("{} does not resolve", route.host),
            ),
        }
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream: &mut RequestHeader,
        ctx: &mut Ctx,
    ) -> Result<()> {
        let route = ctx
            .route
            .as_ref()
            .expect("request_filter routed the request");
        for name in proxy::hop_headers(&upstream.headers) {
            upstream.remove_header(&name);
        }
        upstream.insert_header(header::HOST, route.host.as_str())?;
        upstream.set_uri(route.path.clone());
        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        response: &mut ResponseHeader,
        ctx: &mut Ctx,
    ) -> Result<()> {
        if let Some(route) = ctx.route.as_mut() {
            let status = response.status;
            self.metrics.answered(route.pool(), Some(status));
            route.answered = true;
            if status.is_success() {
                let timing = self.metrics.response(route.pool(), route.received);
                route.timing = Some(timing);
            }
        }
        Ok(())
    }

    async fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
        ctx: &mut Ctx,
    ) -> Result<Option<Duration>> {
        let timing = ctx.route.as_mut().and_then(|route| route.timing.as_mut());
        if let (Some(timing), Some(chunk)) = (timing, body.as_ref())
            && !chunk.is_empty()
        {
            timing.chunk(&Ok::<(), ()>(()));
        }
        Ok(None)
    }

    async fn logging(&self, _session: &mut Session, error: Option<&Error>, ctx: &mut Ctx) {
        // Dropping the route ends the request: it leaves the in-flight count
        // and a successful response records its duration.
        let Some(mut route) = ctx.route.take() else {
            return;
        };
        let source = error.map(|error| error.esource());
        if !route.answered {
            if source == Some(&ErrorSource::Downstream) {
                self.metrics.cancelled(route.pool());
            } else {
                self.metrics.answered(route.pool(), None);
            }
        } else if source == Some(&ErrorSource::Upstream)
            && let Some(timing) = route.timing.as_mut()
        {
            timing.chunk(&Err::<(), ()>(()));
        }
    }

    fn error_while_proxy(
        &self,
        peer: &HttpPeer,
        _session: &mut Session,
        error: Box<Error>,
        _ctx: &mut Ctx,
        client_reused: bool,
    ) -> Box<Error> {
        // The whole body is buffered, so a request that failed on a reused
        // pool connection, which the pool may have closed while it was idle,
        // can go again on a new one.
        let mut error = error.more_context(format!("Peer: {peer}"));
        error.retry.decide_reuse(client_reused);
        error
    }

    async fn fail_to_proxy(
        &self,
        session: &mut Session,
        error: &Error,
        ctx: &mut Ctx,
    ) -> FailToProxy {
        let client_gone = *error.esource() == ErrorSource::Downstream
            && matches!(
                error.etype(),
                ErrorType::ConnectionClosed | ErrorType::ReadError | ErrorType::WriteError
            );
        if client_gone || session.response_written().is_some() {
            return FailToProxy {
                error_code: 0,
                can_reuse_downstream: false,
            };
        }
        let (status, message) = match (ctx.route.as_ref(), error.esource()) {
            (Some(route), ErrorSource::Upstream) => {
                tracing::warn!(
                    pool = %route.pool().id,
                    url = %route.url,
                    error = %error,
                    "pool frontend failed"
                );
                let message = format!("pool frontend {} failed: {error}", route.url);
                (StatusCode::SERVICE_UNAVAILABLE, message)
            }
            _ => {
                let status = match error.etype() {
                    ErrorType::HTTPStatus(code) => {
                        StatusCode::from_u16(*code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                    _ if *error.esource() == ErrorSource::Downstream => StatusCode::BAD_REQUEST,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, error.to_string())
            }
        };
        let _ = respond_error(session, status, &message).await;
        FailToProxy {
            error_code: status.as_u16(),
            can_reuse_downstream: false,
        }
    }
}

/// A plain HTTP peer for the frontend `url`, or `None` if its address does
/// not resolve. A host name resolves here, on the request's task.
fn peer(url: &Url, connect_timeout: Duration) -> Option<HttpPeer> {
    let host = url.host_str()?;
    let address = (host, url.port_or_known_default()?)
        .to_socket_addrs()
        .ok()?
        .next()?;
    let mut peer = HttpPeer::new(address, false, host.to_owned());
    peer.options.connection_timeout = Some(connect_timeout);
    Some(peer)
}

async fn respond(
    session: &mut Session,
    status: StatusCode,
    content_type: &'static str,
    body: Bytes,
) -> Result<bool> {
    let mut response = ResponseHeader::build(status, Some(2))?;
    response.insert_header(header::CONTENT_TYPE, content_type)?;
    response.insert_header(header::CONTENT_LENGTH, body.len().to_string())?;
    session
        .write_response_header(Box::new(response), false)
        .await?;
    session.write_response_body(Some(body), true).await?;
    Ok(true)
}

async fn respond_json(
    session: &mut Session,
    status: StatusCode,
    body: &serde_json::Value,
) -> Result<bool> {
    let body = Bytes::from(serde_json::to_vec(body).expect("JSON values serialize"));
    respond(session, status, "application/json", body).await
}

async fn respond_error(session: &mut Session, status: StatusCode, message: &str) -> Result<bool> {
    respond_json(session, status, &proxy::error_json(status, message)).await
}

/// Serves `proxy` on `listener` until `shutdown` fires. Then it stops
/// accepting connections and waits, up to a grace period, for requests in
/// flight to end.
pub async fn serve_pingora(
    listener: std::net::TcpListener,
    proxy: PingoraProxy,
    shutdown: CancellationToken,
) {
    let pools = proxy.pools.clone();
    let in_flight = proxy.in_flight.clone();
    let conf = ServerConf {
        upstream_keepalive_pool_size: UPSTREAM_POOL_SIZE,
        ..ServerConf::default()
    };
    let mut service = pingora_proxy::http_proxy_service(&Arc::new(conf), proxy);
    let address = listener
        .local_addr()
        .expect("a bound listener has an address")
        .to_string();
    listener
        .set_nonblocking(true)
        .expect("a TCP listener can be made non-blocking");
    let mut fds = Fds::new();
    fds.add(address.clone(), listener.into_raw_fd());
    service.add_tcp(&address);
    let (stop, stopped) = watch::channel(false);
    let serving = tokio::spawn(async move {
        service
            .start_service(Some(Arc::new(fds.into())), stopped, 1)
            .await;
    });

    shutdown.cancelled().await;
    let _ = stop.send(true);
    let deadline = Instant::now() + proxy::SHUTDOWN_GRACE;
    let open = || {
        let pools = pools.borrow();
        pools
            .pools()
            .iter()
            .map(|pool| in_flight.get(&pool.id))
            .sum::<u64>()
    };
    while open() > 0 && Instant::now() < deadline {
        tokio::time::sleep(DRAIN_POLL).await;
    }
    serving.abort();
}
