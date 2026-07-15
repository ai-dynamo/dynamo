// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process extension routes for the frontend HTTP service.
//!
//! The types in this module deliberately contain no Python concepts. Language
//! bindings provide callbacks, while the HTTP service owns routing, limits,
//! request context, and error sanitization.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{OriginalUri, Path, Request};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodFilter, MethodRouter};
use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::logging::{DistributedTraceContext, get_distributed_tracing_context};
use dynamo_runtime::pipeline::context::Controller;

use super::RouteDoc;
use super::metadata::extract_metadata_from_http;
use super::openai::{get_body_limit, get_or_create_request_id};

pub type CustomHttpRouteFuture =
    Pin<Box<dyn Future<Output = Result<CustomHttpResponse, CustomHttpError>> + Send + 'static>>;

pub type CustomHttpRouteCallback =
    Arc<dyn Fn(CustomHttpRequest) -> CustomHttpRouteFuture + Send + Sync + 'static>;

#[derive(Clone)]
pub struct CustomHttpRoute {
    pub method: Method,
    pub path: String,
    pub source: String,
    pub callback: CustomHttpRouteCallback,
}

impl std::fmt::Debug for CustomHttpRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomHttpRoute")
            .field("method", &self.method)
            .field("path", &self.path)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub struct CustomHttpRequest {
    pub method: Method,
    pub path: String,
    pub path_params: HashMap<String, String>,
    pub query_string: String,
    pub query_params: BTreeMap<String, Vec<String>>,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub context: Arc<dyn AsyncEngineContext>,
    pub trace_context: Option<DistributedTraceContext>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct CustomHttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl CustomHttpResponse {
    pub fn new(
        status: u16,
        headers: impl IntoIterator<Item = (String, Vec<String>)>,
        body: impl Into<Bytes>,
    ) -> anyhow::Result<Self> {
        let status = StatusCode::from_u16(status)
            .map_err(|_| anyhow::anyhow!("invalid HTTP response status {status}"))?;
        let mut header_map = HeaderMap::new();
        for (name, values) in headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|err| anyhow::anyhow!("invalid HTTP response header {name:?}: {err}"))?;
            for value in values {
                let value = HeaderValue::from_str(&value).map_err(|err| {
                    anyhow::anyhow!("invalid value for HTTP response header {name}: {err}")
                })?;
                header_map.append(name.clone(), value);
            }
        }
        Ok(Self {
            status,
            headers: header_map,
            body: body.into(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CustomHttpError {
    #[error("HTTP {status}: {message}")]
    Http { status: u16, message: String },
    #[error("{0}")]
    Internal(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Segment {
    Static(String),
    Parameter,
    CatchAll,
}

fn parse_pattern(path: &str, allow_catch_all: bool) -> anyhow::Result<Vec<Segment>> {
    if !path.starts_with('/') {
        anyhow::bail!("route path must be absolute: {path:?}");
    }
    if path.contains('?') || path.contains('#') {
        anyhow::bail!("route path must not contain a query string or fragment: {path:?}");
    }
    if path == "/" {
        return Ok(Vec::new());
    }

    let raw_segments = path[1..].split('/').collect::<Vec<_>>();
    if raw_segments.iter().any(|segment| segment.is_empty()) {
        anyhow::bail!("route path contains an empty segment: {path:?}");
    }

    let mut parameters = HashSet::new();
    let mut segments = Vec::with_capacity(raw_segments.len());
    for (index, segment) in raw_segments.iter().enumerate() {
        if segment.starts_with('{') || segment.ends_with('}') {
            if !(segment.starts_with('{') && segment.ends_with('}')) {
                anyhow::bail!("malformed route parameter in {path:?}");
            }
            let name = &segment[1..segment.len() - 1];
            if let Some(name) = name.strip_prefix('*') {
                if !allow_catch_all || name.is_empty() || index + 1 != raw_segments.len() {
                    anyhow::bail!("catch-all routes are not supported: {path:?}");
                }
                segments.push(Segment::CatchAll);
                continue;
            }
            let mut chars = name.chars();
            let valid = chars
                .next()
                .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
                && chars.all(|c| c == '_' || c.is_ascii_alphanumeric());
            if !valid {
                anyhow::bail!("invalid route parameter name {name:?} in {path:?}");
            }
            if !parameters.insert(name) {
                anyhow::bail!("duplicate route parameter {name:?} in {path:?}");
            }
            segments.push(Segment::Parameter);
        } else {
            if segment.contains('{') || segment.contains('}') {
                anyhow::bail!("malformed route parameter in {path:?}");
            }
            segments.push(Segment::Static((*segment).to_string()));
        }
    }
    Ok(segments)
}

fn segments_compatible(left: &Segment, right: &Segment) -> bool {
    !matches!((left, right), (Segment::Static(a), Segment::Static(b)) if a != b)
}

fn patterns_overlap(left: &[Segment], right: &[Segment]) -> bool {
    let left_catch_all = left
        .iter()
        .position(|segment| *segment == Segment::CatchAll);
    let right_catch_all = right
        .iter()
        .position(|segment| *segment == Segment::CatchAll);
    let shared_prefix = left_catch_all
        .unwrap_or(left.len())
        .min(right_catch_all.unwrap_or(right.len()));

    if !(0..shared_prefix).all(|index| segments_compatible(&left[index], &right[index])) {
        return false;
    }

    match (left_catch_all, right_catch_all) {
        (Some(index), _) => right.len() > index,
        (_, Some(index)) => left.len() > index,
        (None, None) => left.len() == right.len(),
    }
}

fn canonical_pattern(segments: &[Segment]) -> String {
    if segments.is_empty() {
        return "/".to_string();
    }
    segments
        .iter()
        .map(|segment| match segment {
            Segment::Static(value) => format!("/{value}"),
            Segment::Parameter => "/{}".to_string(),
            Segment::CatchAll => "/{*}".to_string(),
        })
        .collect()
}

fn method_filter(method: &Method) -> anyhow::Result<MethodFilter> {
    match *method {
        Method::GET => Ok(MethodFilter::GET),
        Method::POST => Ok(MethodFilter::POST),
        Method::PUT => Ok(MethodFilter::PUT),
        Method::PATCH => Ok(MethodFilter::PATCH),
        Method::DELETE => Ok(MethodFilter::DELETE),
        Method::HEAD => Ok(MethodFilter::HEAD),
        Method::OPTIONS => Ok(MethodFilter::OPTIONS),
        _ => anyhow::bail!("unsupported custom route method {method}"),
    }
}

fn validate_routes(routes: &[CustomHttpRoute], built_in_docs: &[RouteDoc]) -> anyhow::Result<()> {
    let built_in_patterns = built_in_docs
        .iter()
        .map(|doc| parse_pattern(doc.path(), true).map(|pattern| (doc.path(), pattern)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut registered = HashMap::<(Method, String), &str>::new();
    let mut pattern_paths = HashMap::<String, &str>::new();

    for route in routes {
        method_filter(&route.method)
            .map_err(|err| anyhow::anyhow!("custom routes from {:?}: {err}", route.source))?;
        let pattern = parse_pattern(&route.path, false)
            .map_err(|err| anyhow::anyhow!("custom routes from {:?}: {err}", route.source))?;
        let canonical = canonical_pattern(&pattern);
        if let Some(first_path) = pattern_paths.insert(canonical.clone(), &route.path)
            && first_path != route.path
        {
            anyhow::bail!(
                "custom routes from {:?}: route pattern {} conflicts with previously registered {}",
                route.source,
                route.path,
                first_path
            );
        }
        if let Some(first_source) =
            registered.insert((route.method.clone(), canonical), route.source.as_str())
        {
            anyhow::bail!(
                "custom routes from {:?}: duplicate {} {} (first registered by {:?})",
                route.source,
                route.method,
                route.path,
                first_source
            );
        }
        if let Some((built_in_path, _)) = built_in_patterns
            .iter()
            .find(|(_, built_in)| patterns_overlap(&pattern, built_in))
        {
            anyhow::bail!(
                "custom routes from {:?}: route {} {} can match built-in path {}",
                route.source,
                route.method,
                route.path,
                built_in_path
            );
        }
    }
    Ok(())
}

struct KillOnDrop(Option<Arc<dyn AsyncEngineContext>>);

impl KillOnDrop {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if let Some(context) = self.0.take() {
            context.kill();
        }
    }
}

async fn handle(
    callback: CustomHttpRouteCallback,
    source: String,
    Path(path_params): Path<HashMap<String, String>>,
    OriginalUri(uri): OriginalUri,
    request: Request,
) -> Response {
    handle_with_body_limit(
        callback,
        source,
        Path(path_params),
        OriginalUri(uri),
        request,
        get_body_limit(),
    )
    .await
}

async fn handle_with_body_limit(
    callback: CustomHttpRouteCallback,
    source: String,
    Path(path_params): Path<HashMap<String, String>>,
    OriginalUri(uri): OriginalUri,
    request: Request,
    body_limit: usize,
) -> Response {
    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, body_limit).await {
        Ok(body) => body,
        Err(err) => {
            tracing::warn!(source, error = %err, "custom route request body rejected");
            return (StatusCode::PAYLOAD_TOO_LARGE, "Request body too large").into_response();
        }
    };
    let metadata = match extract_metadata_from_http(&parts.headers) {
        Ok(metadata) => metadata,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };
    let request_id = get_or_create_request_id(&parts.headers);
    let context: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new(request_id));
    let mut cancellation = KillOnDrop(Some(context.clone()));
    let query_string = uri.query().unwrap_or_default().to_string();
    let mut query_params = BTreeMap::<String, Vec<String>>::new();
    for (name, value) in url::form_urlencoded::parse(query_string.as_bytes()) {
        query_params
            .entry(name.into_owned())
            .or_default()
            .push(value.into_owned());
    }
    let custom_request = CustomHttpRequest {
        method: parts.method,
        path: uri.path().to_string(),
        path_params,
        query_string,
        query_params,
        headers: parts.headers,
        body,
        context,
        trace_context: get_distributed_tracing_context(),
        metadata,
    };

    let result = callback(custom_request).await;
    cancellation.disarm();
    match result {
        Ok(response) => {
            let mut builder = Response::builder().status(response.status);
            for (name, value) in response.headers.iter() {
                builder = builder.header(name, value);
            }
            builder
                .body(Body::from(response.body))
                .unwrap_or_else(|err| {
                    tracing::error!(source, error = %err, "failed to build custom route response");
                    (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
                })
        }
        Err(CustomHttpError::Http { status, message }) => {
            match StatusCode::from_u16(status).ok().filter(|_| status <= 599) {
                Some(status) => (status, message).into_response(),
                None => {
                    tracing::error!(
                        source,
                        status,
                        "custom route handler returned an invalid HTTP error status"
                    );
                    (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
                }
            }
        }
        Err(CustomHttpError::Internal(err)) => {
            tracing::error!(source, error = %err, "custom route handler failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
        }
    }
}

pub fn router(
    routes: &[CustomHttpRoute],
    built_in_docs: &[RouteDoc],
) -> anyhow::Result<(Vec<RouteDoc>, Router)> {
    if routes.is_empty() {
        return Ok((Vec::new(), Router::new()));
    }

    let mut reserved = built_in_docs.to_vec();
    reserved.push(RouteDoc::new(Method::GET, "/openapi.json"));
    reserved.push(RouteDoc::new(Method::GET, "/docs"));
    reserved.push(RouteDoc::new(Method::GET, "/docs/{*asset}"));
    validate_routes(routes, &reserved)?;

    let mut grouped = BTreeMap::<String, Vec<&CustomHttpRoute>>::new();
    for route in routes {
        grouped.entry(route.path.clone()).or_default().push(route);
    }

    let mut router = Router::new();
    let mut docs = Vec::with_capacity(routes.len());
    for (path, path_routes) in grouped {
        let mut methods = MethodRouter::new();
        for route in path_routes {
            let callback = route.callback.clone();
            let source = route.source.clone();
            methods = methods.on(method_filter(&route.method)?, move |path, uri, request| {
                handle(callback.clone(), source.clone(), path, uri, request)
            });
            docs.push(RouteDoc::new(route.method.clone(), route.path.clone()));
        }
        router = router.route(&path, methods);
    }
    Ok((docs, router))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt as _;

    #[test]
    fn validates_patterns_and_collisions_without_panicking() {
        let callback: CustomHttpRouteCallback = Arc::new(|_| {
            Box::pin(async {
                CustomHttpResponse::new(200, [], Bytes::new())
                    .map_err(|e| CustomHttpError::Internal(e.to_string()))
            })
        });
        let built_ins = vec![RouteDoc::new(Method::GET, "/v1/models/{*model_id}")];
        let route = |path: &str| CustomHttpRoute {
            method: Method::GET,
            path: path.to_string(),
            source: "test.py".to_string(),
            callback: callback.clone(),
        };

        assert!(router(&[route("/custom/{tenant}")], &built_ins).is_ok());
        assert!(router(&[route("/v1/models/acme")], &built_ins).is_err());
        assert!(router(&[route("/custom/{*rest}")], &built_ins).is_err());
        assert!(router(&[route("custom")], &built_ins).is_err());
    }

    #[test]
    fn response_validation_rejects_invalid_values() {
        assert!(CustomHttpResponse::new(99, [], Bytes::new()).is_err());
        assert!(
            CustomHttpResponse::new(
                200,
                [("bad header".to_string(), vec!["value".to_string()])],
                Bytes::new()
            )
            .is_err()
        );
    }

    fn ok_callback() -> CustomHttpRouteCallback {
        Arc::new(|request| {
            Box::pin(async move {
                assert_eq!(
                    request.path_params.get("tenant").map(String::as_str),
                    Some("acme")
                );
                assert_eq!(request.query_string, "tag=a&tag=b");
                assert_eq!(request.query_params["tag"], ["a", "b"]);
                assert_eq!(request.headers["x-test"], "request");
                assert_eq!(request.body, Bytes::from_static(b"payload"));
                CustomHttpResponse::new(
                    201,
                    [
                        (
                            "content-type".to_string(),
                            vec!["application/octet-stream".to_string()],
                        ),
                        (
                            "x-result".to_string(),
                            vec!["a".to_string(), "b".to_string()],
                        ),
                    ],
                    Bytes::from_static(b"response"),
                )
                .map_err(|err| CustomHttpError::Internal(err.to_string()))
            })
        })
    }

    #[tokio::test]
    async fn dispatches_parameterized_routes_and_transports_request_data() {
        let routes = [CustomHttpRoute {
            method: Method::POST,
            path: "/custom/{tenant}".to_string(),
            source: "test.py".to_string(),
            callback: ok_callback(),
        }];
        let (_, router) = router(&routes, &[]).unwrap();
        let response = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/custom/acme?tag=a&tag=b")
                    .header("x-test", "request")
                    .body(Body::from("payload"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(
            response
                .headers()
                .get_all("x-result")
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            Bytes::from_static(b"response")
        );
    }

    #[tokio::test]
    async fn preserves_multiple_methods_on_one_path() {
        let callback: CustomHttpRouteCallback = Arc::new(|request| {
            Box::pin(async move {
                CustomHttpResponse::new(200, [], request.method.to_string())
                    .map_err(|err| CustomHttpError::Internal(err.to_string()))
            })
        });
        let routes = [Method::GET, Method::PUT].map(|method| CustomHttpRoute {
            method,
            path: "/custom".to_string(),
            source: "test.py".to_string(),
            callback: callback.clone(),
        });
        let (_, router) = router(&routes, &[]).unwrap();

        for method in [Method::GET, Method::PUT] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method.clone())
                        .uri("/custom")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                to_bytes(response.into_body(), usize::MAX).await.unwrap(),
                method.to_string()
            );
        }

        let response = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/custom")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn enforces_configured_body_limit() {
        let response = handle_with_body_limit(
            ok_callback(),
            "test.py".to_string(),
            Path(HashMap::from([("tenant".to_string(), "acme".to_string())])),
            OriginalUri("/custom/acme?tag=a&tag=b".parse().unwrap()),
            Request::builder()
                .method(Method::POST)
                .uri("/custom/acme?tag=a&tag=b")
                .header("x-test", "request")
                .body(Body::from("x"))
                .unwrap(),
            0,
        )
        .await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn maps_http_errors_and_sanitizes_unexpected_errors() {
        for (error, expected_status, expected_body) in [
            (
                CustomHttpError::Http {
                    status: 418,
                    message: "teapot".to_string(),
                },
                StatusCode::IM_A_TEAPOT,
                "teapot",
            ),
            (
                CustomHttpError::Internal("secret traceback".to_string()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
            ),
            (
                CustomHttpError::Http {
                    status: 0,
                    message: "secret invalid status".to_string(),
                },
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
            ),
        ] {
            let error = Arc::new(std::sync::Mutex::new(Some(error)));
            let callback: CustomHttpRouteCallback = Arc::new(move |_| {
                let error = error.lock().unwrap().take().unwrap();
                Box::pin(async move { Err(error) })
            });
            let routes = [CustomHttpRoute {
                method: Method::GET,
                path: "/custom".to_string(),
                source: "test.py".to_string(),
                callback,
            }];
            let (_, router) = router(&routes, &[]).unwrap();
            let response = router
                .oneshot(
                    Request::builder()
                        .uri("/custom")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status);
            assert_eq!(
                to_bytes(response.into_body(), usize::MAX).await.unwrap(),
                expected_body
            );
        }
    }

    #[tokio::test]
    async fn kills_context_when_handler_is_dropped() {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = Arc::new(std::sync::Mutex::new(Some(tx)));
        let callback: CustomHttpRouteCallback = Arc::new(move |request| {
            tx.lock()
                .unwrap()
                .take()
                .unwrap()
                .send(request.context)
                .unwrap();
            Box::pin(std::future::pending())
        });
        let task = tokio::spawn(handle(
            callback,
            "test.py".to_string(),
            Path(HashMap::new()),
            OriginalUri("/custom".parse().unwrap()),
            Request::builder()
                .uri("/custom")
                .body(Body::empty())
                .unwrap(),
        ));
        let context = rx.await.unwrap();

        task.abort();
        let _ = task.await;
        assert!(context.is_killed());
    }
}
