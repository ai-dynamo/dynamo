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

use opentelemetry_proto::tonic::common::v1::{AnyValue, InstrumentationScope, KeyValue, any_value};
use opentelemetry_proto::tonic::metrics::v1::{
    AggregationTemporality, Gauge, Histogram, HistogramDataPoint, Metric, NumberDataPoint,
    ResourceMetrics, ScopeMetrics, Sum, Summary, SummaryDataPoint, metric, number_data_point,
    summary_data_point::ValueAtQuantile,
};
use opentelemetry_proto::tonic::resource::v1::Resource;
use prometheus::proto::{MetricFamily, MetricType};
use std::time::{SystemTime, UNIX_EPOCH};

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
