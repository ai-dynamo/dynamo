// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build `MetricFamily` from a typed engine-metrics structure.
//!
//! `prometheus_client` already holds metrics typed; today Dynamo flattens them
//! to exposition text (`generate_latest`) and Rust reconstructs the structure
//! by parsing. This builds from the structure directly instead, which is where
//! the type, help and family name arrive authoritative rather than re-inferred.
//!
//! Samples remain suffix-encoded even in the typed model -- `_bucket` with
//! `le`, `_sum`, `_count`, `_total`, `_created` -- so grouping is still by
//! convention, but scoped to a family of known type rather than guessed
//! globally.

use prometheus::proto::{
    Bucket, Counter, Gauge, Histogram, LabelPair, Metric, MetricFamily, MetricType, Quantile,
    Summary,
};
use std::collections::BTreeMap;

#[cfg(test)]
use serde::Deserialize;

/// Accept a float as either a JSON number or a string, parsing strings with
/// `str::parse`, which is correctly rounded.
#[cfg(test)]
fn de_f64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    use serde::Deserialize as _;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Text(String),
        Num(f64),
    }
    match Repr::deserialize(d)? {
        Repr::Text(t) => t.parse().map_err(serde::de::Error::custom),
        Repr::Num(v) => Ok(v),
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(Deserialize))]
pub struct TypedSample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    /// Fixtures carry this as a string: JSON float parsing can land a ULP off
    /// the correctly-rounded value, and a fixture must not introduce error the
    /// real path -- a native f64 across PyO3 -- never has.
    #[cfg_attr(test, serde(deserialize_with = "de_f64"))]
    pub value: f64,
    /// Seconds since the epoch, when the source recorded one. Standard
    /// `prometheus_client` metrics leave this unset; custom collectors and
    /// federated sources can populate it.
    #[cfg_attr(test, serde(default))]
    pub timestamp: Option<f64>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(Deserialize))]
pub struct TypedFamily {
    pub name: String,
    pub help: String,
    #[cfg_attr(test, serde(rename = "type"))]
    pub kind: String,
    pub samples: Vec<TypedSample>,
}

/// Convert typed families into `MetricFamily`, sorted by name.
///
/// `_created` samples are promoted to standalone gauge families, which is how
/// `generate_latest` renders them and therefore what the text path already
/// exports. Dropping them here would silently stop exporting series that ship
/// today.
pub fn build_families(typed: Vec<TypedFamily>) -> Vec<MetricFamily> {
    let mut out = Vec::new();
    for mut family in typed {
        let (created, rest) = family
            .samples
            .drain(..)
            .partition(|s| s.name.ends_with("_created"));
        family.samples = rest;
        out.extend(promote_created(&family.help, created));
        if let Some(built) = build_one(family) {
            out.push(built);
        }
    }
    out.sort_by(|a, b| a.name().cmp(b.name()));
    out
}

/// `_created` carries the construction timestamp, not a measurement, and the
/// typed model nests it inside its parent while the text model gives it its own
/// family. Emit one gauge family per `_created` sample so both agree.
fn promote_created(help: &str, samples: Vec<TypedSample>) -> Vec<MetricFamily> {
    let mut by_name: BTreeMap<String, MetricFamily> = BTreeMap::new();
    for sample in samples {
        let entry = by_name.entry(sample.name.clone()).or_insert_with(|| {
            let mut f = MetricFamily::new();
            f.set_name(sample.name.clone());
            f.set_help(help.to_string());
            f.set_field_type(MetricType::GAUGE);
            f
        });
        let mut metric = Metric::new();
        metric.set_label(label_pairs(sample.labels.into_iter().collect()));
        let mut gauge = Gauge::new();
        gauge.set_value(sample.value);
        metric.set_gauge(gauge);
        entry.mut_metric().push(metric);
    }
    by_name.into_values().collect()
}

fn build_one(family: TypedFamily) -> Option<MetricFamily> {
    // The OTLP name is the Prometheus *family* name, which `collect()` already
    // reports bare -- so a counter exports as `foo`, not `foo_total`, as the
    // compatibility spec requires. Info is the exception: its `_info` suffix is
    // part of the name rather than a rendering artifact.
    let (metric_type, rendered_suffix) = match family.kind.as_str() {
        "counter" => (MetricType::COUNTER, None),
        "gauge" => (MetricType::GAUGE, None),
        "histogram" => (MetricType::HISTOGRAM, None),
        "summary" => (MetricType::SUMMARY, None),
        // Info has no proto counterpart; it renders as a gauge whose value is
        // always 1 and whose labels carry the payload.
        "info" => (MetricType::GAUGE, Some("_info")),
        // "unknown" is the exposition format's untyped, which is a gauge.
        // Anything else is a type this build does not know: treat it as a
        // gauge so the metric still ships, but say so.
        "unknown" => (MetricType::GAUGE, None),
        other => {
            tracing::debug!(
                metric_name = %family.name,
                kind = %other,
                "unrecognised metric type; treating as gauge"
            );
            (MetricType::GAUGE, None)
        }
    };

    let samples = family.samples;
    if samples.is_empty() {
        return None;
    }

    let name = rendered_suffix
        .and_then(|suffix| samples.iter().find(|s| s.name.ends_with(suffix)))
        .map_or_else(|| family.name.clone(), |s| s.name.clone());

    let mut out = MetricFamily::new();
    out.set_name(name);
    out.set_help(family.help);
    out.set_field_type(metric_type);

    match metric_type {
        MetricType::HISTOGRAM | MetricType::SUMMARY => {
            let point_label = if metric_type == MetricType::HISTOGRAM {
                "le"
            } else {
                "quantile"
            };
            // Group by the labels that identify a series; the bucket/quantile
            // label identifies a point within one.
            struct Series {
                labels: Vec<(String, String)>,
                points: Vec<(f64, f64)>,
                sum: f64,
                count: u64,
                /// A histogram is one proto `Metric` built from several
                /// samples, so keep the first timestamp any of them carried.
                timestamp: Option<f64>,
            }
            let mut series: Vec<Series> = Vec::new();
            for sample in samples {
                let key: Vec<(String, String)> = sample
                    .labels
                    .iter()
                    .filter(|(k, _)| k.as_str() != point_label)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let entry = match series.iter_mut().position(|s| s.labels == key) {
                    Some(i) => &mut series[i],
                    None => {
                        series.push(Series {
                            labels: key,
                            points: Vec::new(),
                            sum: 0.0,
                            count: 0,
                            timestamp: None,
                        });
                        series.last_mut()?
                    }
                };
                if entry.timestamp.is_none() {
                    entry.timestamp = sample.timestamp;
                }
                match sample.labels.get(point_label) {
                    Some(point) => {
                        let bound = parse_bound(point);
                        // +Inf is implicit: TextEncoder synthesises it from
                        // sample_count and would render a stored one as `inf`.
                        if bound.is_finite() {
                            entry.points.push((bound, sample.value));
                        }
                    }
                    None if sample.name.ends_with("_sum") => entry.sum = sample.value,
                    None if sample.name.ends_with("_count") => entry.count = sample.value as u64,
                    None => {}
                }
            }

            for Series {
                labels,
                mut points,
                sum,
                count,
                timestamp,
            } in series
            {
                points.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut metric = Metric::new();
                metric.set_label(label_pairs(labels));
                if metric_type == MetricType::HISTOGRAM {
                    let mut h = Histogram::new();
                    h.set_bucket(
                        points
                            .into_iter()
                            .map(|(bound, cumulative)| {
                                let mut bucket = Bucket::new();
                                bucket.set_upper_bound(bound);
                                bucket.set_cumulative_count(cumulative as u64);
                                bucket
                            })
                            .collect(),
                    );
                    h.set_sample_sum(sum);
                    h.set_sample_count(count);
                    metric.set_histogram(h);
                } else {
                    let mut s = Summary::new();
                    s.set_quantile(
                        points
                            .into_iter()
                            .map(|(q, value)| {
                                let mut quantile = Quantile::new();
                                quantile.set_quantile(q);
                                quantile.set_value(value);
                                quantile
                            })
                            .collect(),
                    );
                    s.set_sample_sum(sum);
                    s.set_sample_count(count);
                    metric.set_summary(s);
                }
                if let Some(seconds) = timestamp {
                    metric.set_timestamp_ms((seconds * 1_000.0) as i64);
                }
                out.mut_metric().push(metric);
            }
        }
        _ => {
            for sample in samples {
                let mut metric = Metric::new();
                metric.set_label(label_pairs(sample.labels.into_iter().collect::<Vec<_>>()));
                if metric_type == MetricType::COUNTER {
                    let mut c = Counter::new();
                    c.set_value(sample.value);
                    metric.set_counter(c);
                } else {
                    let mut g = Gauge::new();
                    g.set_value(sample.value);
                    metric.set_gauge(g);
                }
                if let Some(seconds) = sample.timestamp {
                    metric.set_timestamp_ms((seconds * 1_000.0) as i64);
                }
                out.mut_metric().push(metric);
            }
        }
    }

    (!out.get_metric().is_empty()).then_some(out)
}

fn parse_bound(text: &str) -> f64 {
    match text {
        "+Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        other => other.parse().unwrap_or(f64::NAN),
    }
}

fn label_pairs(mut labels: Vec<(String, String)>) -> Vec<LabelPair> {
    labels.sort();
    labels
        .into_iter()
        .map(|(k, v)| {
            let mut pair = LabelPair::new();
            pair.set_name(k);
            pair.set_value(v);
            pair
        })
        .collect()
}
