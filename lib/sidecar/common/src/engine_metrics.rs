// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fetch an engine's Prometheus exposition for re-export.
//!
//! Sidecars drive the engine over gRPC, but engine metrics live on the
//! engine's HTTP server. This module owns that one HTTP call: a bounded `GET`
//! issued from the runtime's collection callbacks (see
//! `dynamo_backend_common::exposition`), so the scraper's interval is the only
//! cadence and nothing is cached between scrapes. Each sidecar crate decides
//! where the URL comes from.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use clap::Args;
use dynamo_backend_common::exposition::{
    Exposition, ExpositionFilter, ExpositionOutputs, ExpositionSource, ExpositionSourceConfig,
};
use dynamo_backend_common::{DynamoError, EngineMetrics};
use dynamo_runtime::metrics::otlp_export::ExportConfig;
use dynamo_runtime::metrics::prom_text::{TextFormat, parse_text};
use futures::StreamExt;
use reqwest::header::{ACCEPT, CONTENT_TYPE};

use crate::{HttpEndpoint, invalid_argument};

/// Upper bound on one fetch, from request to the last body byte. Equal to the
/// runtime's engine health-probe timeout and well under Prometheus's default
/// 10 s scrape timeout, so a hung engine delays a scrape rather than failing it.
pub const ENGINE_METRICS_FETCH_TIMEOUT: Duration = Duration::from_secs(3);

/// Connect bound. A refused loopback port returns at once; this caps the cost
/// of a wrong host or port guess per scrape.
pub const ENGINE_METRICS_CONNECT_TIMEOUT: Duration = Duration::from_secs(1);

/// Body cap, matching the KV publisher's payload cap and roughly 180 times a
/// single-rank vLLM exposition under load.
pub const ENGINE_METRICS_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Prefer classic text from engines that negotiate; parse either dialect.
const ACCEPT_VALUE: &str =
    "text/plain;version=0.0.4;q=1, application/openmetrics-text;version=1.0.0;q=0.5, */*;q=0.1";

/// CLI surface shared by the sidecars that re-export engine metrics.
#[derive(Args, Clone, Debug)]
pub struct EngineMetricsArgs {
    /// Re-export the engine's Prometheus metrics, with Dynamo identity labels,
    /// on the sidecar `/metrics` endpoint. Set to false when Prometheus
    /// scrapes the engine directly, so the families are not collected twice.
    /// OTLP export of the same families follows the OTEL_* variables instead.
    #[arg(
        long = "engine-metrics-reexport",
        env = "DYN_SIDECAR_ENGINE_METRICS_REEXPORT",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub reexport: bool,
}

impl EngineMetricsArgs {
    /// Surfaces to feed: `/metrics` per the flag, OTLP when the runtime's
    /// exporter is configured.
    pub fn outputs(&self) -> ExpositionOutputs {
        ExpositionOutputs {
            prometheus: self.reexport,
            otlp: ExportConfig::from_env().ok().flatten().is_some(),
        }
    }
}

/// Time and size bounds for one fetch. The defaults are the module constants;
/// tests shrink them.
#[derive(Clone, Copy, Debug)]
pub struct FetchBounds {
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub max_body_bytes: usize,
}

impl Default for FetchBounds {
    fn default() -> Self {
        Self {
            timeout: ENGINE_METRICS_FETCH_TIMEOUT,
            connect_timeout: ENGINE_METRICS_CONNECT_TIMEOUT,
            max_body_bytes: ENGINE_METRICS_MAX_BODY_BYTES,
        }
    }
}

/// The engine's `/metrics` URL and the client that fetches it.
pub struct EngineMetricsEndpoint {
    url: url::Url,
    client: reqwest::Client,
    max_body_bytes: usize,
}

impl EngineMetricsEndpoint {
    /// `/metrics` under `base`, fetched with the default bounds.
    pub fn new(base: &HttpEndpoint) -> Result<Self, DynamoError> {
        Self::with_bounds(base, FetchBounds::default())
    }

    pub fn with_bounds(base: &HttpEndpoint, bounds: FetchBounds) -> Result<Self, DynamoError> {
        let url = base.join("metrics")?;
        let client = reqwest::Client::builder()
            .timeout(bounds.timeout)
            .connect_timeout(bounds.connect_timeout)
            .no_proxy()
            .build()
            .map_err(|error| {
                invalid_argument(format!(
                    "could not configure the engine metrics client: {error}"
                ))
            })?;
        Ok(Self {
            url,
            client,
            max_body_bytes: bounds.max_body_bytes,
        })
    }

    pub fn url(&self) -> &url::Url {
        &self.url
    }

    /// One bounded fetch. Non-2xx status, an oversized body, or a timeout is
    /// an error; the caller decides what a missing exposition means.
    pub async fn fetch(&self) -> anyhow::Result<Exposition> {
        let response = self
            .client
            .get(self.url.clone())
            .header(ACCEPT, ACCEPT_VALUE)
            .send()
            .await
            .with_context(|| format!("GET {}", self.url))?;
        let status = response.status();
        if !status.is_success() {
            bail!("GET {} returned HTTP {status}", self.url);
        }
        if let Some(length) = response.content_length()
            && length > self.max_body_bytes as u64
        {
            bail!(
                "GET {} declares {length} bytes, over the {} byte limit",
                self.url,
                self.max_body_bytes
            );
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("reading GET {}", self.url))?;
            if body.len() + chunk.len() > self.max_body_bytes {
                bail!(
                    "GET {} body exceeds the {} byte limit",
                    self.url,
                    self.max_body_bytes
                );
            }
            body.extend_from_slice(&chunk);
        }
        let text = String::from_utf8(body)
            .with_context(|| format!("GET {} body is not UTF-8", self.url))?;
        let format = TextFormat::detect(content_type.as_deref(), &text);
        Ok(Exposition { text, format })
    }

    /// Adapt this endpoint to the runtime's collection callbacks.
    pub fn source(self: &Arc<Self>) -> ExpositionSource {
        let endpoint = self.clone();
        Arc::new(move || {
            let endpoint = endpoint.clone();
            Box::pin(async move { endpoint.fetch().await })
        })
    }
}

/// Probe the endpoint once for the startup log, then register it on the
/// selected surfaces. The probe does not gate registration: the callbacks
/// retry on every scrape, so an engine that is not serving metrics yet is
/// picked up when it is. `hint` names the operator's fix when the probe fails.
pub async fn register_engine_metrics(
    metrics: &EngineMetrics,
    engine: &str,
    endpoint: Arc<EngineMetricsEndpoint>,
    prefixes: &[&str],
    outputs: ExpositionOutputs,
    hint: &str,
) -> Result<(), DynamoError> {
    let url = endpoint.url().to_string();
    if !outputs.any() {
        tracing::info!(
            engine,
            url,
            "engine metrics collection is off: re-export is disabled and no OTLP metrics exporter is configured"
        );
        return Ok(());
    }
    let filter = ExpositionFilter::include(prefixes);
    match endpoint.fetch().await {
        Ok(exposition) => match parse_text(&exposition.text, exposition.format) {
            Ok(families) => {
                let matched = families
                    .iter()
                    .filter(|family| filter.keeps(&family.name))
                    .count();
                if matched == 0 {
                    tracing::warn!(
                        engine,
                        url,
                        total = families.len(),
                        ?prefixes,
                        "engine metrics endpoint serves no families with the expected prefixes"
                    );
                } else {
                    tracing::info!(
                        engine,
                        url,
                        families = matched,
                        format = ?exposition.format,
                        reexport = outputs.prometheus,
                        otlp = outputs.otlp,
                        "engine metrics endpoint reachable"
                    );
                }
            }
            Err(error) => tracing::warn!(
                engine,
                url,
                error = %format!("{error:#}"),
                "engine metrics endpoint returned text this sidecar cannot parse"
            ),
        },
        Err(error) => tracing::warn!(
            engine,
            url,
            error = %format!("{error:#}"),
            hint,
            "engine metrics endpoint unreachable at startup; retrying on every scrape"
        ),
    }
    metrics.register_exposition_source(ExpositionSourceConfig {
        name: format!("{engine} {url}"),
        source: endpoint.source(),
        filter,
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::http::header;
    use axum::routing::get;
    use clap::Parser;

    #[derive(Parser)]
    struct TestArgs {
        #[command(flatten)]
        engine_metrics: EngineMetricsArgs,
    }

    #[test]
    fn reexport_flag_defaults_on_and_accepts_an_explicit_value() {
        assert!(
            TestArgs::try_parse_from(["test"])
                .unwrap()
                .engine_metrics
                .reexport
        );
        assert!(
            !TestArgs::try_parse_from(["test", "--engine-metrics-reexport", "false"])
                .unwrap()
                .engine_metrics
                .reexport
        );
        assert!(
            TestArgs::try_parse_from(["test", "--engine-metrics-reexport=true"])
                .unwrap()
                .engine_metrics
                .reexport
        );
    }

    async fn serve(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{address}")
    }

    fn endpoint(base: &str, bounds: FetchBounds) -> Arc<EngineMetricsEndpoint> {
        let base = HttpEndpoint::parse(base, "--test").unwrap();
        Arc::new(EngineMetricsEndpoint::with_bounds(&base, bounds).unwrap())
    }

    #[tokio::test]
    async fn fetch_detects_the_dialect_and_keeps_a_path_prefix() {
        let router = Router::new()
            .route(
                "/metrics",
                get(|| async {
                    (
                        [(
                            header::CONTENT_TYPE,
                            "text/plain; version=0.0.4; charset=utf-8",
                        )],
                        "# TYPE m gauge\nm 1\n",
                    )
                }),
            )
            .route(
                "/admin/v1/metrics",
                get(|| async {
                    (
                        [(
                            header::CONTENT_TYPE,
                            "application/openmetrics-text; version=1.0.0; charset=utf-8",
                        )],
                        "# TYPE m gauge\nm 1\n# EOF\n",
                    )
                }),
            );
        let base = serve(router).await;

        let plain = endpoint(&base, FetchBounds::default())
            .fetch()
            .await
            .unwrap();
        assert_eq!(plain.format, TextFormat::Classic);
        assert_eq!(plain.text, "# TYPE m gauge\nm 1\n");

        let prefixed = endpoint(&format!("{base}/admin/v1"), FetchBounds::default());
        assert_eq!(prefixed.url().as_str(), format!("{base}/admin/v1/metrics"));
        assert_eq!(
            prefixed.fetch().await.unwrap().format,
            TextFormat::OpenMetrics
        );
    }

    #[tokio::test]
    async fn fetch_rejects_errors_oversized_bodies_and_slow_engines() {
        let router = Router::new()
            .route("/big/metrics", get(|| async { "x".repeat(2 * 1024) }))
            .route(
                "/slow/metrics",
                get(|| async {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    "m 1\n"
                }),
            );
        let base = serve(router).await;

        let missing = endpoint(&base, FetchBounds::default())
            .fetch()
            .await
            .unwrap_err();
        assert!(format!("{missing:#}").contains("HTTP 404"), "{missing:#}");

        let small = FetchBounds {
            max_body_bytes: 1024,
            ..FetchBounds::default()
        };
        let big = endpoint(&format!("{base}/big"), small)
            .fetch()
            .await
            .unwrap_err();
        assert!(format!("{big:#}").contains("1024 byte limit"), "{big:#}");

        let quick = FetchBounds {
            timeout: Duration::from_millis(200),
            ..FetchBounds::default()
        };
        let started = std::time::Instant::now();
        let slow = endpoint(&format!("{base}/slow"), quick)
            .fetch()
            .await
            .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1), "{slow:#}");

        let refused = endpoint("http://127.0.0.1:9", FetchBounds::default())
            .fetch()
            .await
            .unwrap_err();
        assert!(format!("{refused:#}").contains("GET http://127.0.0.1:9/metrics"));
    }
}
