// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse Prometheus text exposition into `prometheus::proto::MetricFamily`.
//!
//! Engine metrics (vLLM / SGLang / TensorRT-LLM) never enter the Rust
//! `prometheus::Registry`. They arrive as an exposition string from a Python
//! callback (see `MetricsRegistry::add_expfmt_callback`) which
//! `prometheus_expfmt_combined` appends verbatim to `/metrics`, so `gather()`
//! cannot see them. Parsing that text back into families lets one code path
//! export both sources.
//!
//! Hand-rolled rather than using `prometheus-parse`: that crate silently drops
//! every sample whose name contains a colon, and vLLM names all of its metrics
//! `vllm:*`. The exposition grammar is frozen and the subset we need is small.

use prometheus::proto::{
    Bucket, Counter, Gauge, Histogram, LabelPair, Metric, MetricFamily, MetricType, Quantile,
    Summary,
};
use std::collections::HashMap;

/// One parsed sample line.
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    value: f64,
}

impl Sample {
    /// Labels excluding `key`, which carries bucket/quantile position rather
    /// than series identity.
    fn labels_without(&self, key: &str) -> Vec<(String, String)> {
        self.labels
            .iter()
            .filter(|(k, _)| k != key)
            .cloned()
            .collect()
    }

    fn label(&self, key: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Parse Prometheus text exposition into metric families, sorted by name.
pub fn parse_exposition(text: &str) -> anyhow::Result<Vec<MetricFamily>> {
    let mut docs: HashMap<&str, &str> = HashMap::new();
    let mut types: HashMap<&str, MetricType> = HashMap::new();
    let mut samples: Vec<Sample> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            parse_comment(rest.trim(), &mut docs, &mut types);
        } else if let Some(sample) = parse_sample(line) {
            samples.push(sample);
        }
    }

    // A declared TYPE owns its suffixed samples, so route each sample to the
    // family that claims it before building anything.
    let mut families: Vec<MetricFamily> = Vec::new();
    let mut claimed: Vec<bool> = vec![false; samples.len()];

    for (name, metric_type) in &types {
        let family = match metric_type {
            MetricType::HISTOGRAM => build_histogram(name, &samples, &mut claimed),
            MetricType::SUMMARY => build_summary(name, &samples, &mut claimed),
            other => build_simple(name, *other, &samples, &mut claimed),
        };
        if let Some(mut family) = family {
            family.set_help(docs.get(name).copied().unwrap_or_default().to_string());
            families.push(family);
        }
    }

    // Samples with no TYPE line are untyped; a gauge is the conservative
    // reading since untyped carries no aggregation semantics.
    //
    // A declared name is skipped even when its sample went unclaimed -- e.g. a
    // histogram's bare `<name>` line with no `le`. Emitting it here would
    // produce a second family with a declared family's name, and the merger
    // rejects that pair for inconsistent type, costing the whole export tick.
    let mut untyped: HashMap<&str, MetricFamily> = HashMap::new();
    for (idx, sample) in samples.iter().enumerate() {
        if claimed[idx] || types.contains_key(sample.name.as_str()) {
            continue;
        }
        let family = untyped.entry(&sample.name).or_insert_with(|| {
            let mut f = MetricFamily::new();
            f.set_name(sample.name.clone());
            f.set_help(
                docs.get(sample.name.as_str())
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
            );
            f.set_field_type(MetricType::GAUGE);
            f
        });
        let mut metric = Metric::new();
        metric.set_label(label_pairs(&sample.labels));
        let mut gauge = Gauge::new();
        gauge.set_value(sample.value);
        metric.set_gauge(gauge);
        family.mut_metric().push(metric);
    }
    families.extend(untyped.into_values());

    families.sort_by(|a, b| a.name().cmp(b.name()));
    Ok(families)
}

fn parse_comment<'a>(
    rest: &'a str,
    docs: &mut HashMap<&'a str, &'a str>,
    types: &mut HashMap<&'a str, MetricType>,
) {
    if let Some(body) = rest.strip_prefix("HELP ") {
        let (name, help) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
        docs.insert(name, help.trim());
    } else if let Some(body) = rest.strip_prefix("TYPE ") {
        let Some((name, kind)) = body.split_once(char::is_whitespace) else {
            return;
        };
        let metric_type = match kind.trim() {
            "counter" => MetricType::COUNTER,
            "gauge" => MetricType::GAUGE,
            "histogram" => MetricType::HISTOGRAM,
            "summary" => MetricType::SUMMARY,
            _ => MetricType::GAUGE,
        };
        types.insert(name, metric_type);
    }
}

/// `name{labels} value [timestamp]`. Returns `None` for unparseable lines
/// rather than failing the whole payload: one malformed engine line should not
/// cost us every other metric.
fn parse_sample(line: &str) -> Option<Sample> {
    let (name, remainder) = match line.find('{') {
        Some(open) => {
            let close = line.rfind('}')?;
            let labels = &line[open + 1..close];
            (
                line[..open].trim().to_string(),
                (parse_labels(labels), line[close + 1..].trim()),
            )
        }
        None => {
            let (name, rest) = line.split_once(char::is_whitespace)?;
            (name.trim().to_string(), (Vec::new(), rest.trim()))
        }
    };
    let (labels, rest) = remainder;

    // A trailing timestamp is permitted and ignored; we stamp at export.
    let value = rest.split_whitespace().next()?;
    Some(Sample {
        name,
        labels,
        value: parse_value(value)?,
    })
}

fn parse_value(text: &str) -> Option<f64> {
    match text {
        "+Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        "NaN" => Some(f64::NAN),
        other => other.parse().ok(),
    }
}

fn parse_labels(text: &str) -> Vec<(String, String)> {
    let mut labels = Vec::new();
    let mut chars = text.chars().peekable();

    while chars.peek().is_some() {
        let key: String = chars
            .by_ref()
            .take_while(|c| *c != '=')
            .filter(|c| !c.is_whitespace() && *c != ',')
            .collect();
        if key.is_empty() {
            break;
        }
        // Skip to the opening quote of the value.
        if chars.next_if_eq(&'"').is_none() {
            break;
        }
        let mut value = String::new();
        let mut escaped = false;
        for c in chars.by_ref() {
            if escaped {
                value.push(match c {
                    'n' => '\n',
                    other => other,
                });
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                break;
            } else {
                value.push(c);
            }
        }
        labels.push((key, value));
        chars.next_if_eq(&',');
    }
    labels
}

fn label_pairs(labels: &[(String, String)]) -> Vec<LabelPair> {
    let mut pairs: Vec<LabelPair> = labels
        .iter()
        .map(|(k, v)| {
            let mut pair = LabelPair::new();
            pair.set_name(k.clone());
            pair.set_value(v.clone());
            pair
        })
        .collect();
    // Sorted so series identity is stable when merging.
    pairs.sort_by(|a, b| a.name().cmp(b.name()));
    pairs
}

/// Claim every sample named `name`, one metric each.
fn build_simple(
    name: &str,
    metric_type: MetricType,
    samples: &[Sample],
    claimed: &mut [bool],
) -> Option<MetricFamily> {
    let mut family = new_family(name, metric_type);
    for (idx, sample) in samples.iter().enumerate() {
        if claimed[idx] || sample.name != name {
            continue;
        }
        claimed[idx] = true;
        let mut metric = Metric::new();
        metric.set_label(label_pairs(&sample.labels));
        if metric_type == MetricType::COUNTER {
            let mut counter = Counter::new();
            counter.set_value(sample.value);
            metric.set_counter(counter);
        } else {
            let mut gauge = Gauge::new();
            gauge.set_value(sample.value);
            metric.set_gauge(gauge);
        }
        family.mut_metric().push(metric);
    }
    (!family.get_metric().is_empty()).then_some(family)
}

/// A histogram is spread across `<name>_bucket` (with `le`), `<name>_sum` and
/// `<name>_count`, grouped by the remaining labels.
fn build_histogram(name: &str, samples: &[Sample], claimed: &mut [bool]) -> Option<MetricFamily> {
    #[derive(Default)]
    struct Series {
        buckets: Vec<Bucket>,
        sum: f64,
        count: u64,
    }

    let mut family = new_family(name, MetricType::HISTOGRAM);
    let mut series: Vec<(Vec<(String, String)>, Series)> = Vec::new();

    for (idx, sample) in samples.iter().enumerate() {
        if claimed[idx] {
            continue;
        }
        let (key, part) = if sample.name == format!("{name}_bucket") {
            (sample.labels_without("le"), Part::Bucket)
        } else if sample.name == format!("{name}_sum") {
            (sample.labels.clone(), Part::Sum)
        } else if sample.name == format!("{name}_count") {
            (sample.labels.clone(), Part::Count)
        } else {
            continue;
        };
        claimed[idx] = true;

        let entry = match series.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => &mut entry.1,
            None => {
                series.push((key, Series::default()));
                &mut series.last_mut()?.1
            }
        };

        match part {
            Part::Bucket => {
                let Some(bound) = sample.label("le").and_then(parse_value) else {
                    continue;
                };
                let mut bucket = Bucket::new();
                bucket.set_upper_bound(bound);
                bucket.set_cumulative_count(sample.value as u64);
                entry.buckets.push(bucket);
            }
            Part::Sum => entry.sum = sample.value,
            Part::Count => entry.count = sample.value as u64,
        }
    }

    for (labels, mut parts) in series {
        parts
            .buckets
            .sort_by(|a, b| a.upper_bound().total_cmp(&b.upper_bound()));
        let mut histogram = Histogram::new();
        histogram.set_bucket(parts.buckets);
        histogram.set_sample_sum(parts.sum);
        histogram.set_sample_count(parts.count);
        let mut metric = Metric::new();
        metric.set_label(label_pairs(&labels));
        metric.set_histogram(histogram);
        family.mut_metric().push(metric);
    }
    (!family.get_metric().is_empty()).then_some(family)
}

/// A summary is `<name>` with a `quantile` label plus `<name>_sum` and
/// `<name>_count`.
fn build_summary(name: &str, samples: &[Sample], claimed: &mut [bool]) -> Option<MetricFamily> {
    #[derive(Default)]
    struct Series {
        quantiles: Vec<Quantile>,
        sum: f64,
        count: u64,
    }

    let mut family = new_family(name, MetricType::SUMMARY);
    let mut series: Vec<(Vec<(String, String)>, Series)> = Vec::new();

    for (idx, sample) in samples.iter().enumerate() {
        if claimed[idx] {
            continue;
        }
        let (key, part) = if sample.name == name && sample.label("quantile").is_some() {
            (sample.labels_without("quantile"), Part::Bucket)
        } else if sample.name == format!("{name}_sum") {
            (sample.labels.clone(), Part::Sum)
        } else if sample.name == format!("{name}_count") {
            (sample.labels.clone(), Part::Count)
        } else {
            continue;
        };
        claimed[idx] = true;

        let entry = match series.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => &mut entry.1,
            None => {
                series.push((key, Series::default()));
                &mut series.last_mut()?.1
            }
        };

        match part {
            Part::Bucket => {
                let Some(q) = sample.label("quantile").and_then(parse_value) else {
                    continue;
                };
                let mut quantile = Quantile::new();
                quantile.set_quantile(q);
                quantile.set_value(sample.value);
                entry.quantiles.push(quantile);
            }
            Part::Sum => entry.sum = sample.value,
            Part::Count => entry.count = sample.value as u64,
        }
    }

    for (labels, parts) in series {
        let mut summary = Summary::new();
        summary.set_quantile(parts.quantiles);
        summary.set_sample_sum(parts.sum);
        summary.set_sample_count(parts.count);
        let mut metric = Metric::new();
        metric.set_label(label_pairs(&labels));
        metric.set_summary(summary);
        family.mut_metric().push(metric);
    }
    (!family.get_metric().is_empty()).then_some(family)
}

enum Part {
    Bucket,
    Sum,
    Count,
}

fn new_family(name: &str, metric_type: MetricType) -> MetricFamily {
    let mut family = MetricFamily::new();
    family.set_name(name.to_string());
    family.set_field_type(metric_type);
    family
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(fs: &[MetricFamily]) -> Vec<&str> {
        fs.iter().map(|f| f.name()).collect()
    }

    /// Real vLLM exposition: every name carries a colon, and the histogram is
    /// spread across `_bucket` / `_sum` / `_count`.
    #[test]
    fn parses_vllm_shaped_exposition() {
        let fs = parse_exposition(
            r#"# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="llama"} 3
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",model_name="llama",le="0.1"} 1
vllm:time_to_first_token_seconds_bucket{engine="0",model_name="llama",le="+Inf"} 5
vllm:time_to_first_token_seconds_sum{engine="0",model_name="llama"} 2.75
vllm:time_to_first_token_seconds_count{engine="0",model_name="llama"} 5
"#,
        )
        .expect("parse");

        assert_eq!(
            names(&fs),
            vec![
                "vllm:num_requests_running",
                "vllm:time_to_first_token_seconds"
            ]
        );

        let gauge = &fs[0];
        assert_eq!(gauge.get_field_type(), MetricType::GAUGE);
        assert_eq!(gauge.help(), "Number of requests currently running on GPU.");
        assert_eq!(gauge.get_metric()[0].get_gauge().value(), 3.0);
        assert_eq!(gauge.get_metric()[0].get_label()[0].name(), "engine");

        let hist = &fs[1];
        assert_eq!(hist.get_field_type(), MetricType::HISTOGRAM);
        let h = hist.get_metric()[0].get_histogram();
        assert_eq!(h.get_bucket().len(), 2);
        assert_eq!(h.get_bucket()[0].upper_bound(), 0.1);
        assert!(h.get_bucket()[1].upper_bound().is_infinite());
        assert_eq!(h.sample_sum(), 2.75);
        assert_eq!(h.sample_count(), 5);
        // `le` identifies the bucket, not the series.
        assert_eq!(hist.get_metric()[0].get_label().len(), 2);
    }

    #[test]
    fn summary_quantiles_and_siblings_group_into_one_family() {
        let fs = parse_exposition(
            r#"# TYPE d_pause_seconds summary
d_pause_seconds{quantile="0.5"} 0.01
d_pause_seconds{quantile="0.99"} 0.2
d_pause_seconds_sum 12.5
d_pause_seconds_count 300
"#,
        )
        .expect("parse");

        assert_eq!(names(&fs), vec!["d_pause_seconds"]);
        let sm = fs[0].get_metric()[0].get_summary();
        assert_eq!(sm.get_quantile().len(), 2);
        assert_eq!(sm.get_quantile()[0].quantile(), 0.5);
        assert_eq!(sm.sample_sum(), 12.5);
        assert_eq!(sm.sample_count(), 300);
    }

    /// Suffixes are only claimed by a declared parent, so a plain gauge named
    /// `*_sum` survives, and untyped samples fall back to gauge. A declared
    /// name never yields a second family: a bare `<name>` line under a
    /// histogram TYPE must not become a gauge alongside it, or the merger
    /// rejects the pair and the whole export tick is lost.
    #[test]
    fn untyped_fallback_never_shadows_a_declared_family() {
        let fs = parse_exposition(
            "# TYPE d_batch_sum gauge\nd_batch_sum 42\nd_stray_metric{a=\"b\"} 7\n\
             # TYPE d_lat histogram\nd_lat_bucket{le=\"1\"} 2\nd_lat 99\n",
        )
        .expect("parse");

        assert_eq!(names(&fs), vec!["d_batch_sum", "d_lat", "d_stray_metric"]);
        assert_eq!(fs[0].get_metric()[0].get_gauge().value(), 42.0);
        assert_eq!(fs[1].get_field_type(), MetricType::HISTOGRAM);
        assert_eq!(fs[2].get_field_type(), MetricType::GAUGE);
    }
}
