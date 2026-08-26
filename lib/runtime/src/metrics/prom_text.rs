// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse Prometheus text exposition into `prometheus::proto::MetricFamily`.
//!
//! Engine metrics (vLLM / SGLang / TensorRT-LLM) never enter the Rust
//! `prometheus::Registry`. They arrive as an exposition *string* produced by a
//! Python callback (see `MetricsRegistry::add_expfmt_callback`) which
//! `prometheus_expfmt_combined` appends verbatim to `/metrics`.
//!
//! `gather()` therefore cannot see them, and any OTLP bridge built on `gather()`
//! alone would silently omit every engine metric. This module parses that text
//! back into the same typed representation `gather()` returns, so both sources
//! can be merged and exported through one path.
//!
//! `prometheus-parse` is an implementation detail deliberately hidden behind
//! [`parse_exposition`]: the exposition format is frozen, our parse surface is
//! narrow, and the upstream crate is dormant, so swapping in an in-tree parser
//! must stay a single-file change.

use prometheus::proto::{
    Bucket, Counter, Gauge, Histogram, LabelPair, Metric, MetricFamily, MetricType, Quantile,
    Summary,
};
use std::collections::HashMap;

/// Parse Prometheus text exposition into metric families.
///
/// Families are returned sorted by name so callers see a deterministic order.
/// Malformed input is reported as an error rather than silently dropped: a
/// broken engine callback should be visible, not quietly halve the metric set.
pub fn parse_exposition(text: &str) -> anyhow::Result<Vec<MetricFamily>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let lines = text.lines().map(|line| Ok(line.to_owned()));
    let scrape = prometheus_parse::Scrape::parse(lines)
        .map_err(|e| anyhow::anyhow!("parsing prometheus exposition text: {e}"))?;

    // The text format emits a histogram's or summary's sum and count as
    // separate sample lines, which `prometheus-parse` surfaces as standalone
    // untyped samples. Index them up front so they can be folded into their
    // parent, and so they are not also emitted as bogus gauge families.
    let siblings = Siblings::index(&scrape);

    let mut families: HashMap<String, MetricFamily> = HashMap::new();

    for sample in &scrape.samples {
        if siblings.is_absorbed(&sample.metric) {
            continue;
        }

        let name = sample.metric.clone();
        let metric_type = sample_metric_type(&sample.value);

        let family = families.entry(name.clone()).or_insert_with(|| {
            let mut f = MetricFamily::new();
            f.set_name(name.clone());
            // `docs` is keyed by family name; absent HELP yields an empty
            // string, matching what the Prometheus encoder emits.
            f.set_help(scrape.docs.get(&name).cloned().unwrap_or_default());
            f.set_field_type(metric_type);
            f
        });

        if family.get_field_type() != metric_type {
            anyhow::bail!(
                "metric family '{}' has inconsistent types within one exposition payload",
                name
            );
        }

        family.mut_metric().push(build_metric(sample, &siblings));
    }

    let mut out: Vec<MetricFamily> = families.into_values().collect();
    out.sort_by(|a, b| a.name().cmp(b.name()));
    Ok(out)
}

/// `_sum` / `_count` sample values belonging to a histogram or summary parent,
/// keyed by `(parent name, label key)`.
struct Siblings {
    sums: HashMap<(String, String), f64>,
    counts: HashMap<(String, String), f64>,
    /// Sample names that were folded into a parent and must not become families.
    absorbed: std::collections::HashSet<String>,
}

impl Siblings {
    fn index(scrape: &prometheus_parse::Scrape) -> Self {
        // Only fold when a parent of the right type actually exists, so a
        // legitimate standalone gauge ending in `_sum` is left intact.
        let parents: std::collections::HashSet<&str> = scrape
            .samples
            .iter()
            .filter(|s| {
                matches!(
                    s.value,
                    prometheus_parse::Value::Histogram(_) | prometheus_parse::Value::Summary(_)
                )
            })
            .map(|s| s.metric.as_str())
            .collect();

        let mut sums = HashMap::new();
        let mut counts = HashMap::new();
        let mut absorbed = std::collections::HashSet::new();

        for sample in &scrape.samples {
            let value = match sample.value {
                prometheus_parse::Value::Untyped(v)
                | prometheus_parse::Value::Gauge(v)
                | prometheus_parse::Value::Counter(v) => v,
                _ => continue,
            };

            for (suffix, target) in [("_sum", &mut sums), ("_count", &mut counts)] {
                let Some(parent) = sample.metric.strip_suffix(suffix) else {
                    continue;
                };
                if !parents.contains(parent) {
                    continue;
                }
                target.insert((parent.to_string(), label_key_from(&sample.labels)), value);
                absorbed.insert(sample.metric.clone());
            }
        }

        Self {
            sums,
            counts,
            absorbed,
        }
    }

    fn is_absorbed(&self, name: &str) -> bool {
        self.absorbed.contains(name)
    }

    fn sum(&self, name: &str, labels: &str) -> Option<f64> {
        self.sums
            .get(&(name.to_string(), labels.to_string()))
            .copied()
    }

    fn count(&self, name: &str, labels: &str) -> Option<u64> {
        self.counts
            .get(&(name.to_string(), labels.to_string()))
            .map(|v| *v as u64)
    }
}

fn sample_metric_type(value: &prometheus_parse::Value) -> MetricType {
    match value {
        prometheus_parse::Value::Counter(_) => MetricType::COUNTER,
        prometheus_parse::Value::Gauge(_) => MetricType::GAUGE,
        prometheus_parse::Value::Histogram(_) => MetricType::HISTOGRAM,
        prometheus_parse::Value::Summary(_) => MetricType::SUMMARY,
        // Untyped has no OTLP-meaningful aggregation; a gauge is the
        // conservative reading (point-in-time, non-monotonic).
        prometheus_parse::Value::Untyped(_) => MetricType::GAUGE,
    }
}

fn build_metric(sample: &prometheus_parse::Sample, siblings: &Siblings) -> Metric {
    let mut metric = Metric::new();
    metric.set_label(build_labels(&sample.labels));
    let key = label_key_from(&sample.labels);

    match &sample.value {
        prometheus_parse::Value::Counter(v) => {
            let mut counter = Counter::new();
            counter.set_value(*v);
            metric.set_counter(counter);
        }
        prometheus_parse::Value::Gauge(v) | prometheus_parse::Value::Untyped(v) => {
            let mut gauge = Gauge::new();
            gauge.set_value(*v);
            metric.set_gauge(gauge);
        }
        prometheus_parse::Value::Histogram(counts) => {
            let mut histogram = Histogram::new();
            // Exposition buckets are already cumulative and `prometheus-parse`
            // preserves that, so counts carry through unchanged.
            let buckets = counts
                .iter()
                .map(|hc| {
                    let mut bucket = Bucket::new();
                    bucket.set_upper_bound(hc.less_than);
                    bucket.set_cumulative_count(hc.count as u64);
                    bucket
                })
                .collect::<Vec<_>>();
            histogram.set_bucket(buckets);
            histogram.set_sample_sum(siblings.sum(&sample.metric, &key).unwrap_or_default());
            histogram.set_sample_count(siblings.count(&sample.metric, &key).unwrap_or_default());
            metric.set_histogram(histogram);
        }
        prometheus_parse::Value::Summary(counts) => {
            let mut summary = Summary::new();
            let quantiles = counts
                .iter()
                .map(|sc| {
                    let mut q = Quantile::new();
                    q.set_quantile(sc.quantile);
                    q.set_value(sc.count);
                    q
                })
                .collect::<Vec<_>>();
            summary.set_quantile(quantiles);
            summary.set_sample_sum(siblings.sum(&sample.metric, &key).unwrap_or_default());
            summary.set_sample_count(siblings.count(&sample.metric, &key).unwrap_or_default());
            metric.set_summary(summary);
        }
    }

    metric
}

fn build_labels(labels: &prometheus_parse::Labels) -> Vec<LabelPair> {
    let mut pairs: Vec<LabelPair> = labels
        .iter()
        .map(|(k, v)| {
            let mut pair = LabelPair::new();
            pair.set_name(k.to_string());
            pair.set_value(v.to_string());
            pair
        })
        .collect();
    // Sorted for stable series identity when merging and de-duplicating.
    pairs.sort_by(|a, b| a.name().cmp(b.name()));
    pairs
}

/// Canonical identity for a label set, used to pair a histogram with its
/// `_sum` / `_count` siblings.
fn label_key_from(labels: &prometheus_parse::Labels) -> String {
    let mut pairs: Vec<(String, String)> = labels
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(fs: &[MetricFamily]) -> Vec<&str> {
        fs.iter().map(|f| f.name()).collect()
    }

    /// `_sum` and `_count` arrive as separate samples. Unfolded, the parent's
    /// sum stays 0 and OTLP gains two gauge series `/metrics` never shows.
    #[test]
    fn histogram_folds_sum_and_count_into_parent() {
        let fs = parse_exposition(
            r#"# HELP d_seconds Request duration
# TYPE d_seconds histogram
d_seconds_bucket{model="a",le="0.1"} 1
d_seconds_bucket{model="a",le="+Inf"} 5
d_seconds_sum{model="a"} 2.75
d_seconds_count{model="a"} 5
"#,
        )
        .expect("parse");

        assert_eq!(names(&fs), vec!["d_seconds"]);
        assert_eq!(fs[0].get_field_type(), MetricType::HISTOGRAM);
        assert_eq!(fs[0].help(), "Request duration");

        let h = fs[0].get_metric()[0].get_histogram();
        assert_eq!(h.get_bucket().len(), 2);
        assert_eq!(h.sample_sum(), 2.75);
        assert_eq!(h.sample_count(), 5);
    }

    #[test]
    fn summary_preserves_quantiles_and_folds_siblings() {
        let fs = parse_exposition(
            r#"# TYPE d_pause_seconds summary
d_pause_seconds{quantile="0.5"} 0.01
d_pause_seconds_sum 12.5
d_pause_seconds_count 300
"#,
        )
        .expect("parse");

        assert_eq!(names(&fs), vec!["d_pause_seconds"]);
        let sm = fs[0].get_metric()[0].get_summary();
        assert_eq!(sm.get_quantile()[0].quantile(), 0.5);
        assert_eq!(sm.sample_sum(), 12.5);
        assert_eq!(sm.sample_count(), 300);
    }

    /// Folding keys off a real parent, so a plain gauge ending in `_sum`
    /// must survive.
    #[test]
    fn standalone_sum_suffixed_gauge_is_not_absorbed() {
        let fs = parse_exposition("# TYPE d_batch_sum gauge\nd_batch_sum 42\n").expect("parse");
        assert_eq!(names(&fs), vec!["d_batch_sum"]);
        assert_eq!(fs[0].get_metric()[0].get_gauge().value(), 42.0);
    }
}
