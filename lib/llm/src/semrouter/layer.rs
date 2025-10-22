// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tower layer for semantic routing middleware

use std::{future::Future, pin::Pin, sync::Arc, task::{Context as TaskContext, Poll}};

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use tracing::{debug, warn};

use super::{
    classifier::Classifier,
    config::{ClassRoute, SemRouterConfig},
    ctx::{RequestCtx, RouteMode},
    decision::{action_to_decision, choose_action_for_scored_labels, Decision},
};

#[derive(Clone)]
pub struct SemRouterHandle {
    cfg: Arc<SemRouterConfig>,
    clf: Arc<dyn Classifier>,
}

impl SemRouterHandle {
    pub fn new(cfg: SemRouterConfig, clf: Arc<dyn Classifier>) -> Self {
        Self {
            cfg: Arc::new(cfg),
            clf,
        }
    }

    /// Core decision function (multi-class).
    async fn decide(&self, ctx: &RequestCtx, body_json: &serde_json::Value, text: &str) -> Decision {
        if !self.cfg.enabled {
            return Decision::PassThrough;
        }
        let mode = self.cfg.mode;
        if matches!(mode, super::Mode::Off) || matches!(ctx.route_mode, RouteMode::Off) {
            return Decision::PassThrough;
        }

        let classification = match self.clf.predict(text).await {
            Ok(c) => c,
            Err(e) => {
                warn!(error = ?e, "classifier error; falling back to passthrough");
                return Decision::PassThrough;
            }
        };

        // Map label -> (action, threshold) from config for quick lookup.
        let mut table: std::collections::HashMap<&str, (&super::RouteAction, f32)> = Default::default();
        for ClassRoute {
            label,
            threshold,
            action,
        } in &self.cfg.classes
        {
            table.insert(label.as_str(), (action, *threshold));
        }

        // Sorted list from classifier:
        let ordered = classification
            .labels
            .iter()
            .map(|ls| (ls.label.as_str(), ls.score));
        let mut decision = choose_action_for_scored_labels(ordered, |label| table.get(label).copied());

        // If nothing triggered, use fallback (if any).
        if matches!(decision, Decision::PassThrough) && self.cfg.fallback.is_some() {
            decision = action_to_decision(self.cfg.fallback.as_ref().unwrap());
        }

        // Policy check: if user has an explicit model and policy says "never", block overrides.
        // Allow when (1) model equals routing alias, or (2) header says force/auto and policy allows, or (3) global mode is Force.
        let model_in = body_json.get("model").and_then(|v| v.as_str());
        let alias_ok = self
            .cfg
            .model_alias
            .as_deref()
            .is_some_and(|a| Some(a) == model_in);
        let force_like = matches!(ctx.route_mode, RouteMode::Force) || matches!(mode, super::Mode::Force);

        use super::OverridePolicy::*;
        if let Decision::Override { .. } = decision {
            match self.cfg.default_policy {
                NeverWhenExplicit => {
                    let has_explicit = model_in.is_some() && !alias_ok;
                    if has_explicit && !force_like {
                        decision = Decision::PassThrough;
                    }
                }
                AllowWhenOptIn => {
                    let opted = alias_ok || matches!(ctx.route_mode, RouteMode::Auto | RouteMode::Force);
                    if !opted && !force_like {
                        decision = Decision::PassThrough;
                    }
                }
                Always => { /* do nothing */ }
            }
        }

        decision
    }

    /// Fire-and-forget shadowing (stub; wire up an internal client in step 2).
    #[allow(unused)]
    pub fn shadow(&self, _orig: &RequestCtx, _req: &Request<Body>, _route_to: &str) {
        // TODO: Send to a configured alternate backend without blocking.
        // Keep bounded queue; drop oldest under pressure.
        debug!("Shadow routing requested but no sender is configured in step 1; skipping.");
    }
}

#[derive(Clone)]
pub struct SemRouterLayer {
    handle: SemRouterHandle,
}

impl SemRouterLayer {
    pub fn new(handle: SemRouterHandle) -> Self {
        Self { handle }
    }
}

impl<S> tower::Layer<S> for SemRouterLayer {
    type Service = SemRouterService<S>;
    fn layer(&self, inner: S) -> Self::Service {
        SemRouterService {
            inner,
            handle: self.handle.clone(),
        }
    }
}

#[derive(Clone)]
pub struct SemRouterService<S> {
    inner: S,
    handle: SemRouterHandle,
}

impl<S> tower::Service<Request<Body>> for SemRouterService<S>
where
    S: tower::Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut TaskContext<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let mut next = self.inner.clone();
        let handle = self.handle.clone();

        Box::pin(async move {
            use axum::body::to_bytes;

            // Read and buffer the body ONCE; we'll rebuild it after classification.
            let (mut parts, body) = req.into_parts();
            let body_bytes = match to_bytes(body, usize::MAX).await {
                Ok(b) => b,
                Err(_) => {
                    let resp = Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Body::from("invalid request body"))
                        .unwrap();
                    return Ok(resp);
                }
            };

            // Parse JSON body.
            let mut json_val: serde_json::Value = match serde_json::from_slice(&body_bytes) {
                Ok(v) => v,
                Err(_) => {
                    // If not JSON, just pass through.
                    parts
                        .extensions
                        .insert::<&'static str>("semrouter-skipped-non-json");
                    let req2 = Request::from_parts(parts, Body::from(body_bytes.to_vec()));
                    return next.call(req2).await;
                }
            };

            let is_stream = json_val
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let model_in = json_val.get("model").and_then(|v| v.as_str());
            let ctx = super::RequestCtx::from_headers_and_model(&parts.headers, model_in, is_stream);

            // Extract text for classification:
            let text = super::ctx::extract_text_for_classification(&json_val);

            // Compute decision:
            let decision = handle.decide(&ctx, &json_val, &text).await;

            // Apply decision:
            match decision {
                Decision::PassThrough => {
                    let req2 = Request::from_parts(parts, Body::from(body_bytes.to_vec()));
                    next.call(req2).await
                }
                Decision::Override { model_id } => {
                    // Rewrite JSON body field `model` and add an informational header.
                    json_val["model"] = serde_json::Value::String(model_id.clone());
                    let new_bytes = match serde_json::to_vec(&json_val) {
                        Ok(b) => b,
                        Err(_) => body_bytes.to_vec(),
                    };
                    parts.headers.insert(
                        axum::http::header::HeaderName::from_static("x-dynamo-routed-model"),
                        axum::http::HeaderValue::from_str(&model_id)
                            .unwrap_or(axum::http::HeaderValue::from_static("invalid")),
                    );
                    let req2 = Request::from_parts(parts, Body::from(new_bytes));
                    next.call(req2).await
                }
                Decision::Reject { reason } => {
                    let resp = Response::builder()
                        .status(StatusCode::FORBIDDEN)
                        .body(Body::from(reason))
                        .unwrap();
                    Ok(resp)
                }
                Decision::Shadow { route_to } => {
                    // Pass through immediately and optionally tee (not implemented in step 1).
                    handle.shadow(
                        &ctx,
                        &Request::from_parts(parts.clone(), Body::from(body_bytes.to_vec())),
                        &route_to,
                    );
                    let req2 = Request::from_parts(parts, Body::from(body_bytes.to_vec()));
                    next.call(req2).await
                }
            }
        })
    }
}
