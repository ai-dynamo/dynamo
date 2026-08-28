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
pub fn build_families(typed: Vec<TypedFamily>) -> Vec<MetricFamily> {
    let mut out: Vec<MetricFamily> = typed.into_iter().filter_map(build_one).collect();
    out.sort_by(|a, b| a.name().cmp(b.name()));
    out
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
            let mut series: Vec<(Vec<(String, String)>, Vec<(f64, f64)>, f64, u64)> = Vec::new();
            for sample in samples {
                let key: Vec<(String, String)> = sample
                    .labels
                    .iter()
                    .filter(|(k, _)| k.as_str() != point_label)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let entry = match series.iter_mut().position(|(k, _, _, _)| *k == key) {
                    Some(i) => &mut series[i],
                    None => {
                        series.push((key, Vec::new(), 0.0, 0));
                        series.last_mut()?
                    }
                };
                match sample.labels.get(point_label) {
                    Some(point) => {
                        let bound = parse_bound(point);
                        // +Inf is implicit: TextEncoder synthesises it from
                        // sample_count and would render a stored one as `inf`.
                        if bound.is_finite() {
                            entry.1.push((bound, sample.value));
                        }
                    }
                    None if sample.name.ends_with("_sum") => entry.2 = sample.value,
                    None if sample.name.ends_with("_count") => entry.3 = sample.value as u64,
                    None => {}
                }
            }

            for (labels, mut points, sum, count) in series {
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
