// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics scraper source.
//!
//! Periodically fetches metrics from a Prometheus-compatible endpoint
//! and parses Dynamo frontend metrics (TTFT, TPOT, throughput, queue depth).

use anyhow::{Context, Result};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing as log;

use super::{AppEvent, Source};
use crate::model::PrometheusMetrics;

/// Configuration for the metrics source.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub url: String,
    pub scrape_interval: Duration,
}

/// Prometheus metrics scraper source.
pub struct MetricsSource {
    config: MetricsConfig,
}

impl MetricsSource {
    pub fn new(config: MetricsConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Source for MetricsSource {
    async fn run(
        self: Box<Self>,
        tx: tokio::sync::mpsc::Sender<AppEvent>,
        cancel: CancellationToken,
    ) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();

        let mut interval = tokio::time::interval(self.config.scrape_interval);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = interval.tick() => {
                    match scrape_metrics(&client, &self.config.url).await {
                        Ok(metrics) => {
                            let _ = tx.send(AppEvent::MetricsUpdate(metrics)).await;
                        }
                        Err(e) => {
                            log::debug!("Metrics scrape failed: {e:#}");
                            let _ = tx.send(AppEvent::MetricsUpdate(PrometheusMetrics {
                                available: false,
                                ..Default::default()
                            })).await;
                        }
                    }
                }
            }
        }
    }
}

async fn scrape_metrics(client: &reqwest::Client, url: &str) -> Result<PrometheusMetrics> {
    let body = client
        .get(url)
        .send()
        .await
        .context("HTTP request failed")?
        .text()
        .await
        .context("Failed to read response body")?;

    Ok(parse_prometheus_text(&body))
}

/// Parse Prometheus text exposition format and extract Dynamo frontend metrics.
///
/// This is a simple parser that looks for specific metric names. It handles
/// both `dynamo_frontend_*` and custom-prefixed metrics.
pub fn parse_prometheus_text(text: &str) -> PrometheusMetrics {
    let mut metrics = PrometheusMetrics {
        available: true,
        ..Default::default()
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse TTFT histogram quantiles
        if line.contains("time_to_first_token") || line.contains("ttft") {
            if let Some(val) = extract_quantile(line, "0.5") {
                metrics.ttft_p50_ms = Some(val * 1000.0); // seconds to ms
            }
            if let Some(val) = extract_quantile(line, "0.99") {
                metrics.ttft_p99_ms = Some(val * 1000.0);
            }
        }

        // Parse TPOT histogram quantiles
        if line.contains("time_per_output_token") || line.contains("tpot") {
            if let Some(val) = extract_quantile(line, "0.5") {
                metrics.tpot_p50_ms = Some(val * 1000.0);
            }
            if let Some(val) = extract_quantile(line, "0.99") {
                metrics.tpot_p99_ms = Some(val * 1000.0);
            }
        }

        // Parse gauge metrics
        if line.contains("requests_inflight")
            && let Some(val) = extract_gauge_value(line)
        {
            metrics.requests_inflight = Some(val as u64);
        }
        if line.contains("requests_queued")
            && let Some(val) = extract_gauge_value(line)
        {
            metrics.requests_queued = Some(val as u64);
        }
        if (line.contains("tokens_per_second") || line.contains("token_throughput"))
            && let Some(val) = extract_gauge_value(line)
        {
            metrics.tokens_per_sec = Some(val);
        }
    }

    metrics
}

/// Extract a quantile value from a Prometheus histogram line.
/// Example: `metric_name{quantile="0.5"} 0.042`
fn extract_quantile(line: &str, quantile: &str) -> Option<f64> {
    let pattern = format!("quantile=\"{}\"", quantile);
    if !line.contains(&pattern) {
        return None;
    }
    extract_gauge_value(line)
}

/// Extract the numeric value from a Prometheus metric line.
/// Example: `metric_name{labels...} 42.5` → `Some(42.5)`
fn extract_gauge_value(line: &str) -> Option<f64> {
    // Find the last whitespace-separated token as the value
    let value_str = line.rsplit_once(|c: char| c.is_whitespace())?.1;
    value_str.parse::<f64>().ok()
}

/// Mock metrics source for testing.
#[cfg(test)]
pub mod mock {
    use super::*;

    #[allow(dead_code)]
    pub struct MockMetricsSource {
        pub metrics: PrometheusMetrics,
    }

    #[async_trait::async_trait]
    impl Source for MockMetricsSource {
        async fn run(
            self: Box<Self>,
            tx: tokio::sync::mpsc::Sender<AppEvent>,
            cancel: CancellationToken,
        ) {
            let _ = tx.send(AppEvent::MetricsUpdate(self.metrics.clone())).await;
            cancel.cancelled().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prometheus_text_histogram() {
        let text = r#"
# HELP dynamo_frontend_time_to_first_token_seconds TTFT histogram
# TYPE dynamo_frontend_time_to_first_token_seconds histogram
dynamo_frontend_time_to_first_token_seconds{quantile="0.5"} 0.042
dynamo_frontend_time_to_first_token_seconds{quantile="0.99"} 0.180
dynamo_frontend_time_to_first_token_seconds_count 1500
dynamo_frontend_time_to_first_token_seconds_sum 63.0
"#;
        let metrics = parse_prometheus_text(text);
        assert!(metrics.available);
        assert!((metrics.ttft_p50_ms.unwrap() - 42.0).abs() < 0.1);
        assert!((metrics.ttft_p99_ms.unwrap() - 180.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_prometheus_text_gauges() {
        let text = r#"
dynamo_frontend_requests_inflight 12
dynamo_frontend_requests_queued 3
dynamo_frontend_tokens_per_second 450.5
"#;
        let metrics = parse_prometheus_text(text);
        assert_eq!(metrics.requests_inflight, Some(12));
        assert_eq!(metrics.requests_queued, Some(3));
        assert!((metrics.tokens_per_sec.unwrap() - 450.5).abs() < 0.1);
    }

    #[test]
    fn test_parse_prometheus_text_empty() {
        let metrics = parse_prometheus_text("");
        assert!(metrics.available);
        assert!(metrics.ttft_p50_ms.is_none());
        assert!(metrics.requests_inflight.is_none());
    }

    #[test]
    fn test_parse_prometheus_text_comments_only() {
        let text = "# HELP some_metric\n# TYPE some_metric gauge\n";
        let metrics = parse_prometheus_text(text);
        assert!(metrics.available);
    }

    #[test]
    fn test_extract_gauge_value() {
        assert_eq!(extract_gauge_value("metric_name 42.5"), Some(42.5));
        assert_eq!(
            extract_gauge_value("metric_name{label=\"foo\"} 100"),
            Some(100.0)
        );
        assert_eq!(extract_gauge_value("no_value"), None);
    }

    #[test]
    fn test_extract_quantile() {
        let line = r#"metric{quantile="0.5"} 0.042"#;
        assert!((extract_quantile(line, "0.5").unwrap() - 0.042).abs() < 0.001);
        assert!(extract_quantile(line, "0.99").is_none());
    }

    #[test]
    fn test_parse_tpot_metrics() {
        let text = r#"
dynamo_frontend_time_per_output_token_seconds{quantile="0.5"} 0.008
dynamo_frontend_time_per_output_token_seconds{quantile="0.99"} 0.025
"#;
        let metrics = parse_prometheus_text(text);
        assert!((metrics.tpot_p50_ms.unwrap() - 8.0).abs() < 0.1);
        assert!((metrics.tpot_p99_ms.unwrap() - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_prometheus_nan_and_inf() {
        let text = r#"
dynamo_frontend_time_to_first_token_seconds{quantile="0.5"} NaN
dynamo_frontend_requests_inflight +Inf
"#;
        let metrics = parse_prometheus_text(text);
        // NaN should parse as f64::NAN
        assert!(metrics.ttft_p50_ms.is_some());
        assert!(metrics.ttft_p50_ms.unwrap().is_nan());
    }

    #[test]
    fn test_parse_prometheus_mixed_full_payload() {
        let text = r#"
# HELP dynamo_frontend_time_to_first_token_seconds Time to first token
# TYPE dynamo_frontend_time_to_first_token_seconds summary
dynamo_frontend_time_to_first_token_seconds{quantile="0.5"} 0.042
dynamo_frontend_time_to_first_token_seconds{quantile="0.99"} 0.180
dynamo_frontend_time_to_first_token_seconds_sum 63.0
dynamo_frontend_time_to_first_token_seconds_count 1500
# HELP dynamo_frontend_time_per_output_token_seconds TPOT
# TYPE dynamo_frontend_time_per_output_token_seconds summary
dynamo_frontend_time_per_output_token_seconds{quantile="0.5"} 0.008
dynamo_frontend_time_per_output_token_seconds{quantile="0.99"} 0.025
# HELP dynamo_frontend_requests_inflight Current inflight requests
# TYPE dynamo_frontend_requests_inflight gauge
dynamo_frontend_requests_inflight 15
# HELP dynamo_frontend_requests_queued Queued requests
# TYPE dynamo_frontend_requests_queued gauge
dynamo_frontend_requests_queued 4
# HELP dynamo_frontend_tokens_per_second Token throughput
# TYPE dynamo_frontend_tokens_per_second gauge
dynamo_frontend_tokens_per_second 523.7
"#;
        let metrics = parse_prometheus_text(text);
        assert!(metrics.available);
        assert!((metrics.ttft_p50_ms.unwrap() - 42.0).abs() < 0.1);
        assert!((metrics.ttft_p99_ms.unwrap() - 180.0).abs() < 0.1);
        assert!((metrics.tpot_p50_ms.unwrap() - 8.0).abs() < 0.1);
        assert!((metrics.tpot_p99_ms.unwrap() - 25.0).abs() < 0.1);
        assert_eq!(metrics.requests_inflight, Some(15));
        assert_eq!(metrics.requests_queued, Some(4));
        assert!((metrics.tokens_per_sec.unwrap() - 523.7).abs() < 0.1);
    }

    #[test]
    fn test_parse_prometheus_custom_prefix() {
        let text = r#"
my_custom_prefix_ttft_seconds{quantile="0.5"} 0.050
my_custom_prefix_requests_inflight 8
my_custom_prefix_token_throughput 300.0
"#;
        let metrics = parse_prometheus_text(text);
        assert!((metrics.ttft_p50_ms.unwrap() - 50.0).abs() < 0.1);
        assert_eq!(metrics.requests_inflight, Some(8));
        assert!((metrics.tokens_per_sec.unwrap() - 300.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_prometheus_whitespace_variations() {
        let text = "  dynamo_frontend_requests_inflight   7  \n";
        let metrics = parse_prometheus_text(text);
        assert_eq!(metrics.requests_inflight, Some(7));
    }

    #[test]
    fn test_extract_gauge_value_scientific_notation() {
        assert_eq!(extract_gauge_value("metric 1.5e2"), Some(150.0));
        assert_eq!(extract_gauge_value("metric 1e3"), Some(1000.0));
    }

    #[test]
    fn test_extract_gauge_value_negative() {
        assert_eq!(extract_gauge_value("metric -1.5"), Some(-1.5));
    }

    #[test]
    fn test_extract_gauge_value_zero() {
        assert_eq!(extract_gauge_value("metric 0"), Some(0.0));
        assert_eq!(extract_gauge_value("metric 0.0"), Some(0.0));
    }

    #[test]
    fn test_parse_prometheus_only_ttft_no_tpot() {
        let text = r#"
dynamo_frontend_time_to_first_token_seconds{quantile="0.5"} 0.042
"#;
        let metrics = parse_prometheus_text(text);
        assert!(metrics.ttft_p50_ms.is_some());
        assert!(metrics.tpot_p50_ms.is_none());
        assert!(metrics.requests_inflight.is_none());
    }
}
