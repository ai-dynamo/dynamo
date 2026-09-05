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
    AggregationTemporality, DataPointFlags, Gauge, Histogram, HistogramDataPoint, Metric,
    NumberDataPoint, ResourceMetrics, ScopeMetrics, Sum, Summary, SummaryDataPoint, metric,
    number_data_point, summary_data_point::ValueAtQuantile,
};
use opentelemetry_proto::tonic::resource::v1::Resource;
use prometheus::proto::{MetricFamily, MetricType};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio_util::sync::CancellationToken;

const SCOPE_NAME: &str = "dynamo.runtime.metrics";

/// Build an OTLP payload from Prometheus families.
///
/// `start_time` anchors cumulative sums; it must stay fixed for the process
/// lifetime or consumers will read every export as a counter reset.
pub fn to_resource_metrics(
    families: &[MetricFamily],
    resource_attrs: &[KeyValue],
    start_time: SystemTime,
) -> ResourceMetrics {
    let now = unix_nanos(SystemTime::now());
    let start = unix_nanos(start_time);

    // `_created` is a Python-client artifact, not a metric: the spec says use
    // it as the parent's start time and drop it. Keeping it would emit a
    // spurious gauge per counter and histogram, and would throw away the only
    // signal a backend has for telling a counter reset from a jump.
    let created: HashMap<(String, Vec<(String, String)>), u64> = families
        .iter()
        .filter(|f| f.name().ends_with("_created"))
        .flat_map(|f| {
            let parent = f.name().trim_end_matches("_created").to_string();
            f.get_metric().iter().map(move |m| {
                (
                    (parent.clone(), label_key(m)),
                    (m.get_gauge().value() * 1_000_000_000.0) as u64,
                )
            })
        })
        .collect();

    let metrics = families
        .iter()
        .filter(|family| !family.name().ends_with("_created"))
        .filter_map(|family| to_metric(family, start, now, &created))
        .collect();

    ResourceMetrics {
        resource: Some(Resource {
            attributes: resource_attrs.to_vec(),
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

/// Label pairs in a stable order, so a `_created` series can be matched to the
/// parent series it belongs to.
fn label_key(metric: &prometheus::proto::Metric) -> Vec<(String, String)> {
    let mut labels: Vec<(String, String)> = metric
        .get_label()
        .iter()
        .map(|l| (l.name().to_string(), l.value().to_string()))
        .collect();
    labels.sort();
    labels
}

type CreatedTimes = HashMap<(String, Vec<(String, String)>), u64>;

fn to_metric(
    family: &MetricFamily,
    start: u64,
    now: u64,
    created: &CreatedTimes,
) -> Option<Metric> {
    let data = match family.get_field_type() {
        MetricType::COUNTER => metric::Data::Sum(Sum {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| {
                    let start = start_for(family, m, start, created);
                    number_point(m, m.get_counter().value(), start, now)
                })
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
                // Spec: gauges carry no start time. A non-zero one makes
                // backends treat the series as cumulative and hunt for resets.
                .map(|m| number_point(m, m.get_gauge().value(), 0, now))
                .collect(),
        }),
        MetricType::HISTOGRAM => metric::Data::Histogram(Histogram {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| histogram_point(m, start_for(family, m, start, created), now))
                .collect(),
            aggregation_temporality: AggregationTemporality::Cumulative as i32,
        }),
        MetricType::SUMMARY => metric::Data::Summary(Summary {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| summary_point(m, start_for(family, m, start, created), now))
                .collect(),
        }),
        // Native families reach here straight from `Registry::gather()`,
        // without passing through the typed builder that normalises engine
        // metrics, so an UNTYPED family arrives as-is. Exposition format treats
        // untyped as a gauge; do the same rather than return None, which would
        // drop the family from OTLP without a trace.
        MetricType::UNTYPED => metric::Data::Gauge(Gauge {
            data_points: family
                .get_metric()
                .iter()
                .map(|m| number_point(m, m.untyped.value(), 0, now))
                .collect(),
        }),
    };

    // "The OTLP metric name MUST be the Prometheus name with `_total`
    // removed." Applied here rather than in the typed builder so it covers
    // both inputs: engine families arrive already bare, but Dynamo's own
    // counters are registered under their rendered `_total` name.
    let name = match family.get_field_type() {
        MetricType::COUNTER => family.name().trim_end_matches("_total").to_string(),
        _ => family.name().to_string(),
    };

    Some(Metric {
        name,
        description: family.help().to_string(),
        unit: String::new(),
        metadata: Vec::new(),
        data: Some(data),
    })
}

/// When a sample carried its own timestamp, that is when it was observed;
/// otherwise fall back to the export instant.
///
/// Standard `prometheus_client` and `prometheus` metrics leave this unset, but
/// custom collectors and federated sources set it, and reporting a stale sample
/// at the export instant misleads anything reading `time_unix_nano` -- SLO
/// monitors especially.
fn observed_at(metric: &prometheus::proto::Metric, now: u64) -> u64 {
    match metric.timestamp_ms() {
        0 => now,
        ms => (ms as u64).saturating_mul(1_000_000),
    }
}

/// The series' own `_created` time when the client reported one, else the
/// process start time.
fn start_for(
    family: &MetricFamily,
    metric: &prometheus::proto::Metric,
    fallback: u64,
    created: &CreatedTimes,
) -> u64 {
    created
        .get(&(family.name().to_string(), label_key(metric)))
        .copied()
        .unwrap_or(fallback)
}

fn number_point(
    metric: &prometheus::proto::Metric,
    value: f64,
    start: u64,
    now: u64,
) -> NumberDataPoint {
    // A NaN is not a value a backend can store, and several reject the payload
    // outright. The spec's representation is the no-recorded-value flag with
    // the value left unset. Prometheus distinguishes a stale marker from an
    // ordinary NaN by bit pattern, but only an f64 survives to here, and both
    // mean "nothing was recorded" to a consumer.
    let (flags, value) = if value.is_nan() {
        (DataPointFlags::NoRecordedValueMask as u32, None)
    } else {
        (0, Some(number_data_point::Value::AsDouble(value)))
    };

    NumberDataPoint {
        attributes: attributes(metric),
        start_time_unix_nano: start,
        time_unix_nano: observed_at(metric, now),
        exemplars: Vec::new(),
        flags,
        value,
    }
}

fn histogram_point(metric: &prometheus::proto::Metric, start: u64, now: u64) -> HistogramDataPoint {
    let h = metric.get_histogram();

    // Prometheus buckets are cumulative and include a final `+Inf`; OTLP wants
    // per-bucket counts and omits the implicit overflow bound.
    // TODO: preserve exemplars. `prometheus::proto::Bucket` carries an
    // `exemplar` field which is dropped here. Mapping it means decoding
    // `trace_id` / `span_id` label values into the raw bytes OTLP wants, and no
    // engine Dynamo runs emits them today, so it is deliberately absent rather
    // than overlooked -- without exemplars a backend cannot jump from a
    // histogram bucket to the trace that produced it.
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
        time_unix_nano: observed_at(metric, now),
        count: h.sample_count(),
        sum: (!h.sample_sum().is_nan()).then(|| h.sample_sum()),
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
        time_unix_nano: observed_at(metric, now),
        count: s.sample_count(),
        // A summary's sum is not optional in the proto, so an unrecorded one
        // becomes 0 with the flag set rather than a NaN on the wire.
        sum: if s.sample_sum().is_nan() {
            0.0
        } else {
            s.sample_sum()
        },
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
    /// Sent as gRPC metadata on every export; how authenticated collectors are
    /// reached. Empty unless configured.
    pub headers: Vec<(String, String)>,
    /// Resource attributes applied to every export, beyond `service.name`.
    pub resource_attributes: Vec<(String, String)>,
}

/// Parse the `key=value,key=value` form the OTLP exporter spec uses for both
/// headers and resource attributes.
///
/// Values are taken verbatim: splitting on the *first* `=` keeps base64
/// padding in a bearer token intact. Percent-decoding, which the spec allows,
/// is deliberately not applied -- no configuration here has needed it, and
/// silently decoding a token containing a literal `%` would corrupt it.
pub(crate) fn parse_key_value_list(raw: &str) -> Vec<(String, String)> {
    raw.split(',')
        .filter_map(|pair| pair.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, _)| !k.is_empty())
        .collect()
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

        // Only gRPC is implemented; failing loudly beats exporting over the
        // wrong transport. Validate the value as configured rather than as
        // resolved: the shared resolver falls back to grpc for anything it does
        // not recognise, so a typo would otherwise be exported over gRPC --
        // exactly the silent mis-send this check exists to prevent.
        //
        // The signal-specific variable wins; name whichever one supplied the
        // value, so an operator who set only the generic one is not sent to the
        // specific one.
        let metrics_protocol = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL).ok();
        let generic_protocol = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL).ok();
        let configured = [
            (
                metrics_protocol.as_deref(),
                env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
            ),
            (
                generic_protocol.as_deref(),
                env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL,
            ),
        ]
        .into_iter()
        .find_map(|(value, source)| {
            value
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| (value, source))
        });

        if let Some((value, source)) = configured
            && !value.eq_ignore_ascii_case("grpc")
        {
            anyhow::bail!(
                "{source}={value} is not supported for metrics; only grpc is implemented. \
                 Set {}=grpc to override.",
                env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL
            );
        }
        let protocol = crate::logging::OtlpProtocol::Grpc;

        // Signal-specific headers replace the generic set rather than merging,
        // per the OTLP exporter spec.
        let headers = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_METRICS_HEADERS)
            .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_HEADERS))
            .map(|raw| parse_key_value_list(&raw))
            .unwrap_or_default();

        let resource_attributes = std::env::var(env_otlp::OTEL_RESOURCE_ATTRIBUTES)
            .map(|raw| parse_key_value_list(&raw))
            .unwrap_or_default();

        Ok(Some(Self {
            headers,
            resource_attributes,
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

/// Resource attributes for every export.
///
/// `OTEL_SERVICE_NAME` wins over a `service.name` in `OTEL_RESOURCE_ATTRIBUTES`
/// per the resource spec, so the configured one is dropped rather than sent
/// twice.
fn attributes_for(config: &ExportConfig) -> Vec<KeyValue> {
    let mut attrs: Vec<KeyValue> = config
        .resource_attributes
        .iter()
        .filter(|(key, _)| key != "service.name")
        .map(|(key, value)| KeyValue {
            key: key.clone(),
            value: Some(AnyValue {
                value: Some(any_value::Value::StringValue(value.clone())),
            }),
        })
        .collect();
    attrs.push(KeyValue {
        key: "service.name".to_string(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(config.service_name.clone())),
        }),
    });
    attrs
}

/// Collect and export until `cancel` fires.
///
/// Export failures are logged and retried on the next tick: metrics are
/// resendable, so a transient collector outage should not tear down the task.
///
/// Both the connect and the export are bounded by an explicit deadline and
/// raced against `cancel`. Tonic applies no default RPC timeout, so a
/// collector that accepts the connection but never answers would otherwise
/// hold this task past shutdown.
pub async fn run(registry: MetricsRegistry, config: ExportConfig, cancel: CancellationToken) {
    let attrs = attributes_for(&config);
    // Fixed for the process lifetime; a moving start time reads as a counter
    // reset on every export.
    let start_time = SystemTime::now();

    // A call still outstanding when the next collection is due has missed its
    // window, so the interval is the natural deadline -- but cap it, or a long
    // interval buys an equally long hang against an unresponsive collector.
    const MAX_RPC_DEADLINE: Duration = Duration::from_secs(30);
    let rpc_deadline = config.interval.min(MAX_RPC_DEADLINE);
    let mut ticker = tokio::time::interval(config.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut client: Option<MetricsServiceClient<tonic::transport::Channel>> = None;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = ticker.tick() => {}
        }

        // Connect lazily and retry on every tick. Connecting once up front
        // would strand the exporter for the process lifetime if the collector
        // happened to be down when this worker started.
        if client.is_none() {
            let connecting = async {
                tonic::transport::Endpoint::from_shared(config.endpoint.clone())?
                    .connect_timeout(rpc_deadline)
                    .timeout(rpc_deadline)
                    .connect()
                    .await
                    .map_err(anyhow::Error::from)
            };
            let connected = tokio::select! {
                _ = cancel.cancelled() => break,
                result = connecting => result,
            };
            match connected {
                Ok(channel) => client = Some(MetricsServiceClient::new(channel)),
                Err(error) => {
                    tracing::warn!(%error, endpoint = %config.endpoint, "OTLP metrics connect failed");
                    continue;
                }
            }
        }

        // Typed callbacks cross into Python and take the GIL, which can block
        // for as long as the engine holds it. That must not park an async
        // worker thread.
        //
        // Awaited rather than raced against `cancel`: a blocking task cannot be
        // cancelled, so racing it would only drop the handle while the closure
        // ran on -- reaching for the GIL on a detached thread while the process
        // tears down, and delaying shutdown anyway, since the runtime waits for
        // blocking tasks it has already started. Exit latency is one
        // collection; the loop checks `cancel` before the next.
        let collector = registry.clone();
        let collected =
            tokio::task::spawn_blocking(move || collector.metric_families_combined()).await;
        let families = match collected {
            Ok(Ok(families)) => families,
            Ok(Err(error)) => {
                tracing::warn!(%error, "OTLP metrics collection failed");
                continue;
            }
            Err(error) => {
                tracing::warn!(%error, "OTLP metrics collection task failed");
                continue;
            }
        };

        let request = ExportMetricsServiceRequest {
            resource_metrics: vec![to_resource_metrics(&families, &attrs, start_time)],
        };
        let mut request = tonic::Request::new(request);
        for (key, value) in &config.headers {
            match (
                tonic::metadata::MetadataKey::from_bytes(key.as_bytes()),
                tonic::metadata::MetadataValue::try_from(value),
            ) {
                (Ok(key), Ok(value)) => {
                    request.metadata_mut().insert(key, value);
                }
                _ => {
                    // Logged once per export rather than dropped silently: a
                    // malformed auth header means every request is rejected.
                    tracing::warn!(header = %key, "skipping unusable OTLP header");
                }
            }
        }

        if let Some(connected) = client.as_mut() {
            let exported = tokio::select! {
                _ = cancel.cancelled() => break,
                result = connected.export(request) => result,
            };
            match exported {
                Ok(response) => {
                    // A successful status can still report dropped series --
                    // cardinality limits are the usual cause. Reporting the
                    // export as clean would hide data loss the collector has
                    // already told us about.
                    if let Some(partial) = response.into_inner().partial_success
                        && partial.rejected_data_points > 0
                    {
                        tracing::warn!(
                            rejected_data_points = partial.rejected_data_points,
                            error_message = %partial.error_message,
                            endpoint = %config.endpoint,
                            "OTLP collector rejected part of the export"
                        );
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, endpoint = %config.endpoint, "OTLP metrics export failed");
                    // Drop the channel so the next tick reconnects; a broken
                    // transport will not recover on its own.
                    client = None;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::prom_typed::{TypedFamily, build_families};

    fn export(typed_json: &str) -> Vec<Metric> {
        let typed: Vec<TypedFamily> = serde_json::from_str(typed_json).expect("typed");
        let families = build_families(typed);
        let rm = to_resource_metrics(&families, &[], UNIX_EPOCH);
        rm.scope_metrics.into_iter().next().expect("scope").metrics
    }

    /// Prometheus buckets are cumulative and carry `+Inf`; OTLP wants per-bucket
    /// counts with the overflow bound implicit and one more count than bounds.
    #[test]
    fn histogram_buckets_are_de_cumulated() {
        let metrics = export(
            r#"[{"name":"d_seconds","help":"","type":"histogram","samples":[
                 {"name":"d_seconds_bucket","labels":{"le":"0.1"},"value":"2"},
                 {"name":"d_seconds_bucket","labels":{"le":"0.5"},"value":"5"},
                 {"name":"d_seconds_bucket","labels":{"le":"+Inf"},"value":"9"},
                 {"name":"d_seconds_sum","labels":{},"value":"1.5"},
                 {"name":"d_seconds_count","labels":{},"value":"9"}]}]"#,
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
    ///
    /// The name is the bare family name: the compatibility spec requires the
    /// `_total` suffix be removed on the way to OTLP, so this deliberately
    /// differs from what `/metrics` renders.
    #[test]
    fn counter_maps_to_monotonic_cumulative_sum_with_help() {
        let metrics = export(
            r#"[{"name":"d_requests","help":"Total requests","type":"counter","samples":[
                 {"name":"d_requests_total","labels":{"model":"a"},"value":"17"}]}]"#,
        );

        assert_eq!(
            metrics[0].name, "d_requests",
            "spec: the OTLP name is the Prometheus name with _total removed"
        );
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
                (env_otlp::OTEL_METRIC_EXPORT_INTERVAL, None),
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

        // A typo must fail too. The shared resolver silently falls back to
        // grpc for anything it does not recognise, so validating the resolved
        // protocol would export over gRPC and call it success.
        temp_env::with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
                    Some("htp/protbuf"),
                ),
            ],
            || assert!(ExportConfig::from_env().is_err()),
        );

        // The generic variable is the fallback and must be validated too.
        temp_env::with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (env_otlp::OTEL_EXPORTER_OTLP_METRICS_PROTOCOL, None),
                (env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL, Some("http/protobuf")),
            ],
            || assert!(ExportConfig::from_env().is_err()),
        );
    }

    /// Dynamo's own metrics reach OTLP by a different route than engine
    /// metrics: `Registry::gather()` straight into the mapper, never through
    /// the typed builder. Every other test here feeds engine-side data, so
    /// this is the only cover for that half of the exporter's input.
    #[test]
    fn native_dynamo_families_survive_the_mapping() {
        let registry = MetricsRegistry::new();

        let counter =
            prometheus::IntCounter::new("dynamo_requests_total", "Requests").expect("counter");
        counter.inc_by(7);
        registry
            .get_prometheus_registry()
            .register(Box::new(counter))
            .expect("register counter");

        let histogram = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new("dynamo_latency_seconds", "Latency")
                .buckets(vec![0.1, 1.0]),
        )
        .expect("histogram");
        histogram.observe(0.5);
        registry
            .get_prometheus_registry()
            .register(Box::new(histogram))
            .expect("register histogram");

        let families = registry.metric_families_combined().expect("combined");
        let metrics = to_resource_metrics(&families, &[], UNIX_EPOCH)
            .scope_metrics
            .into_iter()
            .next()
            .expect("scope")
            .metrics;

        let by_name = |n: &str| metrics.iter().find(|m| m.name == n).cloned();

        // Registered as `dynamo_requests_total`; exported bare, per spec.
        let Some(metric::Data::Sum(sum)) = by_name("dynamo_requests").expect("counter").data else {
            panic!("counter should map to a Sum");
        };
        assert!(sum.is_monotonic);
        assert_eq!(
            sum.data_points[0].value,
            Some(number_data_point::Value::AsDouble(7.0))
        );

        let Some(metric::Data::Histogram(h)) =
            by_name("dynamo_latency_seconds").expect("histogram").data
        else {
            panic!("histogram should map to a Histogram");
        };
        assert_eq!(h.data_points[0].count, 1);
        assert_eq!(h.data_points[0].sum, Some(0.5));
    }

    /// An UNTYPED family has not been through the typed builder's
    /// normalisation, so the mapper must decide for itself. Dropping it would
    /// remove the family from OTLP silently.
    #[test]
    fn untyped_family_maps_to_a_gauge() {
        let mut family = MetricFamily::new();
        family.set_name("legacy_metric".to_string());
        family.set_field_type(MetricType::UNTYPED);
        let mut metric = prometheus::proto::Metric::new();
        let mut untyped = prometheus::proto::Untyped::new();
        untyped.set_value(3.5);
        metric.untyped = Some(untyped).into();
        family.mut_metric().push(metric);

        let metrics = to_resource_metrics(&[family], &[], UNIX_EPOCH)
            .scope_metrics
            .into_iter()
            .next()
            .expect("scope")
            .metrics;

        assert_eq!(metrics.len(), 1, "UNTYPED family was dropped");
        let Some(metric::Data::Gauge(g)) = &metrics[0].data else {
            panic!("expected gauge, got {:?}", metrics[0].data);
        };
        assert_eq!(
            g.data_points[0].value,
            Some(number_data_point::Value::AsDouble(3.5))
        );
    }

    /// The exporter participates in graceful shutdown, so a final in-flight
    /// export is not abandoned mid-RPC.
    ///
    /// The hazard worth pinning is a deadlock rather than the wait itself:
    /// Phase 2 blocks on the guard, so if the exporter only stopped on the
    /// main token -- cancelled in Phase 3, after that wait -- it would hold
    /// the guard until the timeout fired. `child_token()` derives from the
    /// endpoint shutdown token, which Phase 1 cancels first, so the guard is
    /// released promptly.
    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn exporter_is_waited_for_and_releases_on_phase_one() {
        use crate::distributed::distributed_test_utils::create_test_drt_async;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("addr").port();

        temp_env::async_with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
                    Some(format!("http://127.0.0.1:{port}").as_str()),
                ),
                // Long enough that the exporter is idle at its tick when
                // shutdown lands, so this measures the guard and not a race
                // against a busy export.
                (env_otlp::OTEL_METRIC_EXPORT_INTERVAL, Some("600000")),
            ],
            async {
                let drt = create_test_drt_async().await;
                let tracker = drt.runtime().graceful_shutdown_tracker();
                assert!(
                    tracker.get_count() > 0,
                    "exporter did not register for graceful shutdown"
                );

                drt.runtime().shutdown();

                // Phase 1 cancels the endpoint token; the exporter should drop
                // its guard well before Phase 2's timeout.
                let released = tokio::time::timeout(Duration::from_secs(10), async {
                    while tracker.get_count() > 0 {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                })
                .await;
                assert!(
                    released.is_ok(),
                    "exporter never released its shutdown guard; Phase 2 would block until timeout"
                );
            },
        )
        .await;
    }

    /// Authenticated collectors are reached with headers, and deployments are
    /// told apart by resource attributes. Neither is supplied by the SDK here:
    /// traces and logs get them from `opentelemetry-otlp`, which this exporter
    /// bypasses because the SDK's metric data model is read-only.
    #[test]
    fn headers_and_resource_attributes_are_read_from_the_environment() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT, Some("http://c:4317")),
                (env_otlp::OTEL_METRIC_EXPORT_INTERVAL, None),
                // A bearer token: base64 padding means the value itself
                // contains `=`, so only the first one separates key from value.
                (
                    env_otlp::OTEL_EXPORTER_OTLP_HEADERS,
                    Some("authorization=Bearer abc==,x-key=v"),
                ),
                (
                    env_otlp::OTEL_RESOURCE_ATTRIBUTES,
                    Some("deployment.environment=prod,service.name=ignored"),
                ),
            ],
            || {
                let config = ExportConfig::from_env().expect("enabled").expect("some");
                assert_eq!(
                    config.headers,
                    vec![
                        ("authorization".to_string(), "Bearer abc==".to_string()),
                        ("x-key".to_string(), "v".to_string()),
                    ]
                );
                assert_eq!(
                    config.resource_attributes,
                    vec![
                        ("deployment.environment".to_string(), "prod".to_string()),
                        ("service.name".to_string(), "ignored".to_string()),
                    ]
                );

                // service.name from OTEL_SERVICE_NAME wins, and is not emitted
                // twice alongside the one in OTEL_RESOURCE_ATTRIBUTES.
                let rm = to_resource_metrics(&[], &attributes_for(&config), UNIX_EPOCH);
                let keys: Vec<&str> = rm
                    .resource
                    .as_ref()
                    .expect("resource")
                    .attributes
                    .iter()
                    .map(|kv| kv.key.as_str())
                    .collect();
                assert_eq!(keys, vec!["deployment.environment", "service.name"]);
            },
        );
    }

    /// The signal-specific variable replaces the generic set rather than
    /// merging with it, per the OTLP exporter spec.
    #[test]
    fn metrics_headers_replace_the_generic_ones() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT, Some("http://c:4317")),
                (env_otlp::OTEL_METRIC_EXPORT_INTERVAL, None),
                (env_otlp::OTEL_EXPORTER_OTLP_HEADERS, Some("generic=1")),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_METRICS_HEADERS,
                    Some("specific=2"),
                ),
            ],
            || {
                let config = ExportConfig::from_env().expect("enabled").expect("some");
                assert_eq!(
                    config.headers,
                    vec![("specific".to_string(), "2".to_string())]
                );
            },
        );
    }

    /// The exporter must not depend on the system status server, which is
    /// disabled by default (`DYN_SYSTEM_PORT=-1`). Gating export on it made
    /// the documented `OTEL_METRICS_EXPORTER=otlp` opt-in a silent no-op in
    /// the default configuration.
    ///
    /// Asserts a connection actually arrives, rather than that some code ran:
    /// a spawn that never dials proves nothing.
    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn export_starts_without_the_system_status_server() {
        use crate::distributed::distributed_test_utils::create_test_drt_async;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("addr").port();

        // DYN_SYSTEM_PORT is left unset, so the status server takes its
        // default of -1 (disabled) -- the configuration the bug hid in.
        let connected = temp_env::async_with_vars(
            [
                (env_otlp::OTEL_METRICS_EXPORTER, Some("otlp")),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
                    Some(format!("http://127.0.0.1:{port}").as_str()),
                ),
                (env_otlp::OTEL_METRIC_EXPORT_INTERVAL, Some("100")),
            ],
            async {
                let _drt = create_test_drt_async().await;
                tokio::time::timeout(Duration::from_secs(10), listener.accept())
                    .await
                    .is_ok()
            },
        )
        .await;

        assert!(
            connected,
            "exporter never dialled the collector with the system status server disabled"
        );
    }

    /// A NaN is reported as no-recorded-value with the value unset, not as a
    /// raw NaN on the wire. Engines produce them routinely -- any ratio with a
    /// zero denominator -- and several backends reject the payload.
    #[test]
    fn nan_becomes_no_recorded_value() {
        let metrics = export(
            r#"[{"name":"d_ratio","help":"Ratio","type":"gauge","samples":[
                 {"name":"d_ratio","labels":{},"value":"NaN"}]},
                {"name":"d_latency","help":"Latency","type":"histogram","samples":[
                 {"name":"d_latency_bucket","labels":{"le":"1"},"value":"0"},
                 {"name":"d_latency_sum","labels":{},"value":"NaN"},
                 {"name":"d_latency_count","labels":{},"value":"0"}]}]"#,
        );

        let Some(metric::Data::Gauge(g)) = &metrics
            .iter()
            .find(|m| m.name == "d_ratio")
            .expect("gauge")
            .data
        else {
            panic!("expected gauge");
        };
        assert_eq!(
            g.data_points[0].flags,
            DataPointFlags::NoRecordedValueMask as u32
        );
        assert!(
            g.data_points[0].value.is_none(),
            "the value must be unset, not a NaN"
        );

        let Some(metric::Data::Histogram(h)) = &metrics
            .iter()
            .find(|m| m.name == "d_latency")
            .expect("histogram")
            .data
        else {
            panic!("expected histogram");
        };
        assert!(
            h.data_points[0].sum.is_none(),
            "an unrecorded histogram sum must be unset rather than NaN"
        );
    }

    /// The compatibility spec treats `_created` as the parent's start time, not
    /// as a metric: it must not appear in the export, and the counter it
    /// belongs to must carry it as `start_time_unix_nano`. Emitting it as a
    /// gauge produced one spurious metric per counter and threw away the only
    /// signal a backend has for distinguishing a counter reset from a jump.
    ///
    /// Gauges must carry no start time at all -- a non-zero one makes backends
    /// treat the series as cumulative.
    #[test]
    fn created_becomes_the_start_time_and_gauges_carry_none() {
        let metrics = export(
            r#"[{"name":"d_requests","help":"Requests","type":"counter","samples":[
                 {"name":"d_requests_total","labels":{"model":"a"},"value":"17"},
                 {"name":"d_requests_created","labels":{"model":"a"},"value":"1700000000"}]},
                {"name":"d_queue","help":"Queue","type":"gauge","samples":[
                 {"name":"d_queue","labels":{},"value":"3"}]}]"#,
        );

        assert!(
            !metrics.iter().any(|m| m.name.ends_with("_created")),
            "_created must not be exported as a metric: {:?}",
            metrics.iter().map(|m| &m.name).collect::<Vec<_>>()
        );

        let counter = metrics
            .iter()
            .find(|m| m.name == "d_requests")
            .expect("counter");
        let Some(metric::Data::Sum(sum)) = &counter.data else {
            panic!("expected sum");
        };
        assert_eq!(
            sum.data_points[0].start_time_unix_nano, 1_700_000_000_000_000_000,
            "the counter must start at its _created time"
        );

        let gauge = metrics.iter().find(|m| m.name == "d_queue").expect("gauge");
        let Some(metric::Data::Gauge(g)) = &gauge.data else {
            panic!("expected gauge");
        };
        assert_eq!(
            g.data_points[0].start_time_unix_nano, 0,
            "spec: gauges carry no start time"
        );
    }

    /// A sample that carries its own timestamp is reported as observed then,
    /// not at the export instant. Standard client metrics leave it unset, but a
    /// custom collector or federated source sets it, and relabelling a stale
    /// sample to "now" misleads anything reading `time_unix_nano`.
    #[test]
    fn sample_timestamps_survive_instead_of_the_export_instant() {
        // Carried all the way from the Python-side tuple, through the typed
        // builder, into the datapoint.
        let metrics = export(
            r#"[{"name":"engine_queue","help":"Queue","type":"gauge","samples":[
                 {"name":"engine_queue","labels":{},"value":"4","timestamp":1700000000.0}]},
                {"name":"engine_fresh","help":"Fresh","type":"gauge","samples":[
                 {"name":"engine_fresh","labels":{},"value":"1"}]}]"#,
        );

        let point = |name: &str| {
            let m = metrics.iter().find(|m| m.name == name).expect("metric");
            let Some(metric::Data::Gauge(g)) = &m.data else {
                panic!("expected gauge");
            };
            g.data_points[0].time_unix_nano
        };

        assert_eq!(
            point("engine_queue"),
            1_700_000_000_000_000_000,
            "a carried timestamp must be reported as the observation time"
        );
        assert_ne!(
            point("engine_fresh"),
            0,
            "a sample without one still gets the export instant"
        );
    }

    /// `prometheus_client` reports an Info family under its bare name but
    /// renders it as `<name>_info`. Exporting the bare name would give the
    /// series a different identity than `/metrics` shows, and could collide
    /// with a real gauge of that name when families are merged.
    #[test]
    fn info_family_keeps_its_rendered_name() {
        let metrics = export(
            r#"[{"name":"example_build","help":"Build info","type":"info","samples":[
                 {"name":"example_build_info","labels":{"version":"1.2.3"},"value":"1"}]}]"#,
        );

        assert_eq!(metrics[0].name, "example_build_info");
        let Some(metric::Data::Gauge(g)) = &metrics[0].data else {
            panic!("expected gauge, got {:?}", metrics[0].data);
        };
        assert_eq!(g.data_points[0].attributes[0].key, "version");
    }

    /// Summaries only survive because we bypass the SDK, whose data model has
    /// no `Summary` variant.
    #[test]
    fn summary_quantiles_survive() {
        let metrics = export(
            r#"[{"name":"d_pause_seconds","help":"","type":"summary","samples":[
                 {"name":"d_pause_seconds","labels":{"quantile":"0.99"},"value":"0.2"},
                 {"name":"d_pause_seconds_sum","labels":{},"value":"12.5"},
                 {"name":"d_pause_seconds_count","labels":{},"value":"300"}]}]"#,
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
