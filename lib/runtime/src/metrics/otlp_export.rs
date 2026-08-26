// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Map Prometheus metric families to OTLP.
//!
//! The OTel Rust SDK's metric data model is read-only -- no public
//! constructors, no external producer hook, and no `Summary` variant -- so this
//! builds `opentelemetry-proto` messages directly instead of going through
//! `SdkMeterProvider`.
//!
//! Deliberately free of any runtime types: the same mapping serves an
//! out-of-process scraper.

use crate::config::environment_names::logging::otlp as env_otlp;
use crate::metrics::MetricsRegistry;
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::metrics::v1::metrics_service_client::MetricsServiceClient;
use opentelemetry_proto::tonic::common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value};
use opentelemetry_proto::tonic::metrics::v1::{
    AggregationTemporality, Gauge, Histogram, HistogramDataPoint, Metric, NumberDataPoint,
    ResourceMetrics, ScopeMetrics, Sum, Summary, SummaryDataPoint, metric, number_data_point,
    summary_data_point::ValueAtQuantile,
};
use opentelemetry_proto::tonic::resource::v1::Resource;
use prometheus::proto::{MetricFamily, MetricType};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio_util::sync::CancellationToken;

const SCOPE_NAME: &str = "dynamo.runtime.metrics";

/// Build an OTLP payload from Prometheus families.
///
/// `start_time` anchors cumulative sums; it must stay fixed for the process
/// lifetime or consumers will read every export as a counter reset.
pub fn to_resource_metrics(
    families: &[MetricFamily],
    resource_attrs: Vec<KeyValue>,
    start_time: SystemTime,
) -> ResourceMetrics {
    let now = unix_nanos(SystemTime::now());
    let start = unix_nanos(start_time);

    let metrics = families
        .iter()
        .filter_map(|family| to_metric(family, start, now))
        .collect();

    ResourceMetrics {
        resource: Some(Resource {
            attributes: resource_attrs,
            dropped_attributes_count: 0,
            entity_refs: Vec::new(),
        }),
        scope_metrics: vec![ScopeMetrics {
            scope: Some(InstrumentationScope {
                name: SCOPE_NAME.to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                attributes: Vec::new(),
                dropped_attributes_count: 0,
            }),
            metrics,
            schema_url: String::new(),
        }],
        schema_url: String::new(),
    }
}

fn to_metric(family: &MetricFamily, start: u64, now: u64) -> Option<Metric> {
    let data = match family.get_field_type() {
        MetricType::COUNTER => metric::Data::Sum(Sum {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| number_point(m, m.get_counter().value(), start, now))
                .collect(),
            // Prometheus is always cumulative; converting to delta would need
            // reset detection we have no basis for.
            aggregation_temporality: AggregationTemporality::Cumulative as i32,
            is_monotonic: true,
        }),
        MetricType::GAUGE => metric::Data::Gauge(Gauge {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| number_point(m, m.get_gauge().value(), start, now))
                .collect(),
        }),
        MetricType::HISTOGRAM => metric::Data::Histogram(Histogram {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| histogram_point(m, start, now))
                .collect(),
            aggregation_temporality: AggregationTemporality::Cumulative as i32,
        }),
        MetricType::SUMMARY => metric::Data::Summary(Summary {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| summary_point(m, start, now))
                .collect(),
        }),
        // UNTYPED is normalised to a gauge upstream; anything else is a
        // Prometheus type we do not emit.
        _ => return None,
    };

    Some(Metric {
        name: family.name().to_string(),
        description: family.help().to_string(),
        unit: String::new(),
        metadata: Vec::new(),
        data: Some(data),
    })
}

fn number_point(
    metric: &prometheus::proto::Metric,
    value: f64,
    start: u64,
    now: u64,
) -> NumberDataPoint {
    NumberDataPoint {
        attributes: attributes(metric),
        start_time_unix_nano: start,
        time_unix_nano: now,
        exemplars: Vec::new(),
        flags: 0,
        value: Some(number_data_point::Value::AsDouble(value)),
    }
}

fn histogram_point(metric: &prometheus::proto::Metric, start: u64, now: u64) -> HistogramDataPoint {
    let h = metric.get_histogram();

    // Prometheus buckets are cumulative and include a final `+Inf`; OTLP wants
    // per-bucket counts and omits the implicit overflow bound.
    let mut bounds = Vec::new();
    let mut counts = Vec::new();
    let mut previous = 0u64;
    for bucket in h.get_bucket() {
        let cumulative = bucket.cumulative_count();
        counts.push(cumulative.saturating_sub(previous));
        previous = cumulative;
        if bucket.upper_bound().is_finite() {
            bounds.push(bucket.upper_bound());
        }
    }
    // OTLP requires exactly one more count than bounds.
    if counts.len() == bounds.len() {
        counts.push(h.sample_count().saturating_sub(previous));
    }

    HistogramDataPoint {
        attributes: attributes(metric),
        start_time_unix_nano: start,
        time_unix_nano: now,
        count: h.sample_count(),
        sum: Some(h.sample_sum()),
        bucket_counts: counts,
        explicit_bounds: bounds,
        exemplars: Vec::new(),
        flags: 0,
        min: None,
        max: None,
    }
}

fn summary_point(metric: &prometheus::proto::Metric, start: u64, now: u64) -> SummaryDataPoint {
    let s = metric.get_summary();
    SummaryDataPoint {
        attributes: attributes(metric),
        start_time_unix_nano: start,
        time_unix_nano: now,
        count: s.sample_count(),
        sum: s.sample_sum(),
        quantile_values: s
            .get_quantile()
            .iter()
            .map(|q| ValueAtQuantile {
                quantile: q.quantile(),
                value: q.value(),
            })
            .collect(),
        flags: 0,
    }
}

fn attributes(metric: &prometheus::proto::Metric) -> Vec<KeyValue> {
    metric
        .get_label()
        .iter()
        .map(|label| KeyValue {
            key: label.name().to_string(),
            value: Some(AnyValue {
                value: Some(any_value::Value::StringValue(label.value().to_string())),
            }),
        })
        .collect()
}

fn unix_nanos(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or_default()
}

/// Resolved OTLP metrics export settings.
pub struct ExportConfig {
    pub endpoint: String,
    pub interval: Duration,
    pub service_name: String,
}

impl ExportConfig {
    /// `None` unless `OTEL_METRICS_EXPORTER=otlp`. Prometheus scraping is
    /// unaffected either way.
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Ok(exporter) = std::env::var(env_otlp::OTEL_METRICS_EXPORTER) else {
            return Ok(None);
        };
        if !exporter.trim().eq_ignore_ascii_case("otlp") {
            return Ok(None);
        }

        let protocol = crate::logging::resolve_signal_otlp_protocol(
            crate::logging::otlp_protocol_from_env(),
            std::env::var(env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL)
                .ok()
                .as_deref(),
            env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
        );
        // Only gRPC is implemented. Failing loudly beats silently exporting
        // over the wrong transport, or silently not exporting at all.
        if protocol != crate::logging::OtlpProtocol::Grpc {
            anyhow::bail!(
                "{}=http/protobuf is not supported for metrics; use grpc",
                env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL
            );
        }

        Ok(Some(Self {
            endpoint: crate::logging::resolve_otlp_endpoint(
                protocol,
                std::env::var(env_otlp::OTEL_EXPORTER_OTLP_METRICS_ENDPOINT).ok(),
                std::env::var(env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT).ok(),
                "/v1/metrics",
            ),
            interval: export_interval(),
            service_name: crate::logging::get_service_name(),
        }))
    }
}

fn export_interval() -> Duration {
    const DEFAULT_MS: u64 = 60_000;
    let millis = std::env::var(env_otlp::OTEL_METRIC_EXPORT_INTERVAL)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .unwrap_or(DEFAULT_MS);
    Duration::from_millis(millis)
}

/// Collect and export until `cancel` fires.
///
/// Collection reads through a TTL cache because expfmt callbacks take the
/// Python GIL and `/metrics` is scraped independently. Export failures are
/// logged and retried on the next tick: metrics are resendable, so a
/// transient collector outage should not tear down the task.
pub async fn run(registry: MetricsRegistry, config: ExportConfig, cancel: CancellationToken) {
    let mut client = match MetricsServiceClient::connect(config.endpoint.clone()).await {
        Ok(client) => client,
        Err(error) => {
            tracing::error!(%error, endpoint = %config.endpoint, "OTLP metrics exporter failed to connect");
            return;
        }
    };

    let attrs = vec![KeyValue {
        key: "service.name".to_string(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(config.service_name)),
        }),
    }];
    // Fixed for the process lifetime; a moving start time reads as a counter
    // reset on every export.
    let start_time = SystemTime::now();
    // Half the interval keeps a near-simultaneous scrape from forcing a second
    // GIL-taking collection, without serving stale data to the exporter.
    let ttl = config.interval / 2;

    let mut ticker = tokio::time::interval(config.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = ticker.tick() => {}
        }

        let families = match registry.metric_families_cached(ttl) {
            Ok(families) => families,
            Err(error) => {
                tracing::warn!(%error, "OTLP metrics collection failed");
                continue;
            }
        };

        let request = ExportMetricsServiceRequest {
            resource_metrics: vec![to_resource_metrics(&families, attrs.clone(), start_time)],
        };
        if let Err(error) = client.export(request).await {
            tracing::warn!(%error, endpoint = %config.endpoint, "OTLP metrics export failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::prom_text::parse_exposition;

    fn export(text: &str) -> Vec<Metric> {
        let families = parse_exposition(text).expect("parse");
        let rm = to_resource_metrics(&families, Vec::new(), UNIX_EPOCH);
        rm.scope_metrics.into_iter().next().expect("scope").metrics
    }

    /// Prometheus buckets are cumulative and carry `+Inf`; OTLP wants per-bucket
    /// counts with the overflow bound implicit and one more count than bounds.
    #[test]
    fn histogram_buckets_are_de_cumulated() {
        let metrics = export(
            r#"# TYPE d_seconds histogram
d_seconds_bucket{le="0.1"} 2
d_seconds_bucket{le="0.5"} 5
d_seconds_bucket{le="+Inf"} 9
d_seconds_sum 1.5
d_seconds_count 9
"#,
        );

        let Some(metric::Data::Histogram(h)) = &metrics[0].data else {
            panic!("expected histogram, got {:?}", metrics[0].data);
        };
        let point = &h.data_points[0];
        assert_eq!(point.explicit_bounds, vec![0.1, 0.5]);
        assert_eq!(point.bucket_counts, vec![2, 3, 4]);
        assert_eq!(point.bucket_counts.len(), point.explicit_bounds.len() + 1);
        assert_eq!(point.count, 9);
        assert_eq!(point.sum, Some(1.5));
        assert_eq!(
            h.aggregation_temporality,
            AggregationTemporality::Cumulative as i32
        );
    }

    /// Counters must be monotonic cumulative sums, and HELP must reach
    /// `description` -- the fidelity a scraping sidecar cannot preserve.
    #[test]
    fn counter_maps_to_monotonic_cumulative_sum_with_help() {
        let metrics = export(
            r#"# HELP d_requests_total Total requests
# TYPE d_requests_total counter
d_requests_total{model="a"} 17
"#,
        );

        assert_eq!(metrics[0].name, "d_requests_total");
        assert_eq!(metrics[0].description, "Total requests");

        let Some(metric::Data::Sum(sum)) = &metrics[0].data else {
            panic!("expected sum, got {:?}", metrics[0].data);
        };
        assert!(sum.is_monotonic);
        assert_eq!(
            sum.aggregation_temporality,
            AggregationTemporality::Cumulative as i32
        );
        assert_eq!(
            sum.data_points[0].value,
            Some(number_data_point::Value::AsDouble(17.0))
        );
        assert_eq!(sum.data_points[0].attributes[0].key, "model");
    }

    /// Export is opt-in, and an unsupported protocol must fail loudly rather
    /// than silently not export.
    #[test]
    fn config_is_opt_in_and_rejects_unsupported_protocol() {
        temp_env::with_vars([(env_otlp::OTEL_METRICS_EXPORTER, None::<&str>)], || {
            assert!(ExportConfig::from_env().expect("disabled").is_none())
        });

        temp_env::with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
                    Some("http://collector:4317"),
                ),
            ],
            || {
                let config = ExportConfig::from_env()
                    .expect("enabled")
                    .expect("some config");
                assert_eq!(config.endpoint, "http://collector:4317");
                assert_eq!(config.interval, Duration::from_millis(60_000));
            },
        );

        temp_env::with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
                    Some("http/protobuf"),
                ),
            ],
            || assert!(ExportConfig::from_env().is_err()),
        );
    }

    /// Summaries only survive because we bypass the SDK, whose data model has
    /// no `Summary` variant.
    #[test]
    fn summary_quantiles_survive() {
        let metrics = export(
            r#"# TYPE d_pause_seconds summary
d_pause_seconds{quantile="0.99"} 0.2
d_pause_seconds_sum 12.5
d_pause_seconds_count 300
"#,
        );

        let Some(metric::Data::Summary(s)) = &metrics[0].data else {
            panic!("expected summary, got {:?}", metrics[0].data);
        };
        let point = &s.data_points[0];
        assert_eq!(point.count, 300);
        assert_eq!(point.sum, 12.5);
        assert_eq!(point.quantile_values[0].quantile, 0.99);
        assert_eq!(point.quantile_values[0].value, 0.2);
    }
}
