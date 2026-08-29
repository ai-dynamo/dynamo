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
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct TypedSample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
    /// Present in the exposition format and in `prometheus_client`'s model.
    /// Nothing populates it today, but the boundary should not narrow the
    /// contract on its own.
    #[serde(default)]
    pub timestamp: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct TypedFamily {
    pub name: String,
    pub help: String,
    #[serde(rename = "type")]
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
    for family in typed {
        out.extend(promote_created(&family));
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
fn promote_created(family: &TypedFamily) -> Vec<MetricFamily> {
    let mut by_name: BTreeMap<&str, MetricFamily> = BTreeMap::new();
    for sample in family
        .samples
        .iter()
        .filter(|s| s.name.ends_with("_created"))
    {
        let entry = by_name.entry(sample.name.as_str()).or_insert_with(|| {
            let mut f = MetricFamily::new();
            f.set_name(sample.name.clone());
            f.set_help(family.help.clone());
            f.set_field_type(MetricType::GAUGE);
            f
        });
        let mut metric = Metric::new();
        metric.set_label(label_pairs(
            sample
                .labels
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ));
        let mut gauge = Gauge::new();
        gauge.set_value(sample.value);
        metric.set_gauge(gauge);
        entry.mut_metric().push(metric);
    }
    by_name.into_values().collect()
}

fn build_one(family: TypedFamily) -> Option<MetricFamily> {
    let metric_type = match family.kind.as_str() {
        "counter" => MetricType::COUNTER,
        "gauge" => MetricType::GAUGE,
        "histogram" => MetricType::HISTOGRAM,
        "summary" => MetricType::SUMMARY,
        _ => MetricType::GAUGE,
    };

    // `_created` carries no measurement and is opt-out upstream via
    // PROMETHEUS_DISABLE_CREATED_SERIES. generate_latest promotes it to its own
    // gauge family; dropping it keeps one representation of a metric.
    let samples: Vec<TypedSample> = family
        .samples
        .into_iter()
        .filter(|s| !s.name.ends_with("_created"))
        .collect();
    if samples.is_empty() {
        return None;
    }

    // A counter's family is reported as `foo` but rendered as `foo_total`.
    // Keep the rendered name so the metric surface is unchanged.
    let name = match metric_type {
        MetricType::COUNTER => samples
            .iter()
            .find(|s| s.name.ends_with("_total"))
            .map_or_else(|| family.name.clone(), |s| s.name.clone()),
        _ => family.name.clone(),
    };

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
                        });
                        series.last_mut()?
                    }
                };
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
