// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fidelity of engine-metric parsing, against exposition captured from a real
//! engine rather than hand-written samples.
//!
//! Hand-written fixtures miss what nobody thought to type: an earlier parser
//! silently dropped every `vllm:*` metric because it mishandled colons, and six
//! synthetic tests passed throughout. These fixtures are `generate_latest`
//! output from vLLM 0.18.0 (`PrometheusStatLogger`), captured with and without
//! `PROMETHEUS_DISABLE_CREATED_SERIES`.

use dynamo_runtime::metrics::prom_text::parse_exposition;
use prometheus::proto::{MetricFamily, MetricType};
use std::collections::BTreeMap;

const WITH_CREATED: &str = include_str!("data/vllm-0.18.0-metrics.txt");
const NO_CREATED: &str = include_str!("data/vllm-0.18.0-metrics-no-created.txt");

/// Series identity as Prometheus sees it: sample name plus sorted labels.
type SeriesKey = (String, Vec<(String, String)>);

/// Every series the exposition text declares, read independently of our parser.
fn series_in_text(text: &str) -> BTreeMap<SeriesKey, String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (head, value) = match line.rfind(' ') {
            Some(i) => (&line[..i], line[i + 1..].to_string()),
            None => continue,
        };
        let (name, labels) = match head.find('{') {
            Some(open) => {
                let close = head.rfind('}').unwrap_or(head.len());
                let mut pairs: Vec<(String, String)> = head[open + 1..close]
                    .split(',')
                    .filter(|p| !p.trim().is_empty())
                    .filter_map(|p| p.split_once('='))
                    .map(|(k, v)| (k.trim().to_string(), v.trim().trim_matches('"').to_string()))
                    .collect();
                pairs.sort();
                (head[..open].trim().to_string(), pairs)
            }
            None => (head.trim().to_string(), Vec::new()),
        };
        out.insert((name, labels), value);
    }
    out
}

/// Every series our parsed families represent, re-expanded to sample level.
fn series_in_families(families: &[MetricFamily]) -> BTreeMap<SeriesKey, String> {
    let mut out = BTreeMap::new();
    for family in families {
        let name = family.name().to_string();
        for metric in family.get_metric() {
            let mut labels: Vec<(String, String)> = metric
                .get_label()
                .iter()
                .map(|l| (l.name().to_string(), l.value().to_string()))
                .collect();
            labels.sort();

            match family.get_field_type() {
                MetricType::HISTOGRAM => {
                    let h = metric.get_histogram();
                    for bucket in h.get_bucket() {
                        let mut with_le = labels.clone();
                        with_le.push(("le".into(), canonical(bucket.upper_bound())));
                        with_le.sort();
                        out.insert(
                            (format!("{name}_bucket"), with_le),
                            (bucket.cumulative_count() as f64).to_string(),
                        );
                    }
                    // The +Inf bucket is implicit in the family, exactly as
                    // `TextEncoder` treats it; reconstruct it from sample_count
                    // so this compares meaning rather than storage.
                    let mut inf = labels.clone();
                    inf.push(("le".into(), "+Inf".into()));
                    inf.sort();
                    out.insert(
                        (format!("{name}_bucket"), inf),
                        (h.sample_count() as f64).to_string(),
                    );
                    out.insert(
                        (format!("{name}_sum"), labels.clone()),
                        h.sample_sum().to_string(),
                    );
                    out.insert(
                        (format!("{name}_count"), labels.clone()),
                        (h.sample_count() as f64).to_string(),
                    );
                }
                MetricType::SUMMARY => {
                    let s = metric.get_summary();
                    for q in s.get_quantile() {
                        let mut with_q = labels.clone();
                        with_q.push(("quantile".into(), canonical(q.quantile())));
                        with_q.sort();
                        out.insert((name.clone(), with_q), q.value().to_string());
                    }
                    out.insert(
                        (format!("{name}_sum"), labels.clone()),
                        s.sample_sum().to_string(),
                    );
                    out.insert(
                        (format!("{name}_count"), labels.clone()),
                        (s.sample_count() as f64).to_string(),
                    );
                }
                MetricType::COUNTER => {
                    out.insert(
                        (name.clone(), labels),
                        metric.get_counter().value().to_string(),
                    );
                }
                _ => {
                    out.insert(
                        (name.clone(), labels),
                        metric.get_gauge().value().to_string(),
                    );
                }
            }
        }
    }
    out
}

/// OpenMetrics canonical number, so `le="1"` and `le="1.0"` compare equal.
fn canonical(v: f64) -> String {
    if v.is_infinite() {
        return if v > 0.0 {
            "+Inf".into()
        } else {
            "-Inf".into()
        };
    }
    format!("{v:?}")
}

fn canonical_str(v: &str) -> String {
    match v {
        "+Inf" | "-Inf" | "NaN" => v.to_string(),
        other => other
            .parse::<f64>()
            .map(canonical)
            .unwrap_or_else(|_| other.to_string()),
    }
}

fn assert_no_series_lost(text: &str, label: &str) {
    let families = parse_exposition(text).expect("parse real engine exposition");
    let expected = series_in_text(text);
    let actual = series_in_families(&families);

    let norm = |m: &BTreeMap<SeriesKey, String>| -> BTreeMap<SeriesKey, String> {
        m.iter()
            .map(|((n, l), v)| {
                let labels = l
                    .iter()
                    .map(|(k, lv)| {
                        if k == "le" || k == "quantile" {
                            (k.clone(), canonical_str(lv))
                        } else {
                            (k.clone(), lv.clone())
                        }
                    })
                    .collect();
                ((n.clone(), labels), canonical_str(v))
            })
            .collect()
    };
    let (expected, actual) = (norm(&expected), norm(&actual));

    let missing: Vec<_> = expected
        .keys()
        .filter(|k| !actual.contains_key(*k))
        .collect();
    let extra: Vec<_> = actual
        .keys()
        .filter(|k| !expected.contains_key(*k))
        .collect();
    assert!(
        missing.is_empty(),
        "[{label}] {} series present in engine output but lost by the parser, e.g. {:?}",
        missing.len(),
        &missing[..missing.len().min(5)]
    );
    assert!(
        extra.is_empty(),
        "[{label}] {} series invented by the parser, e.g. {:?}",
        extra.len(),
        &extra[..extra.len().min(5)]
    );

    let wrong: Vec<_> = expected
        .iter()
        .filter(|(k, v)| actual.get(*k) != Some(*v))
        .take(5)
        .collect();
    assert!(
        wrong.is_empty(),
        "[{label}] value mismatches, e.g. {wrong:?}"
    );
}

/// Default engine configuration, where prometheus_client emits `_created`.
#[test]
fn real_vllm_exposition_survives_parsing() {
    assert!(
        WITH_CREATED.contains("_created"),
        "fixture should exercise _created"
    );
    assert_no_series_lost(WITH_CREATED, "with _created");
}

/// `PROMETHEUS_DISABLE_CREATED_SERIES=True`, which deployments may set.
#[test]
fn real_vllm_exposition_without_created_survives_parsing() {
    assert!(
        !NO_CREATED.contains("_created"),
        "fixture should have no _created"
    );
    assert_no_series_lost(NO_CREATED, "no _created");
}

/// Colons are legal in metric names and vLLM uses them throughout; a parser
/// that mishandles them drops every engine metric while looking healthy.
#[test]
fn colon_named_families_are_parsed() {
    let families = parse_exposition(WITH_CREATED).expect("parse");
    let colon_named = families
        .iter()
        .filter(|f| f.name().starts_with("vllm:"))
        .count();
    assert!(
        colon_named > 50,
        "expected the vllm: families, got {colon_named}"
    );
}

/// Re-encoding parsed families must produce valid exposition format. The
/// bucket bound `+Inf` is spelled that way in the spec; `inf` is not accepted.
#[test]
fn reencoded_families_use_a_valid_infinity_bound() {
    use prometheus::Encoder;
    let families = parse_exposition(WITH_CREATED).expect("parse");
    let mut buf = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&families, &mut buf)
        .expect("encode");
    let text = String::from_utf8(buf).expect("utf8");

    assert!(
        !text.contains("le=\"inf\""),
        "invalid infinity bound in re-encoded output"
    );
    assert!(
        text.contains("le=\"+Inf\""),
        "expected synthesised +Inf buckets"
    );
}

/// Not an assertion — prints one real family before and after, for review.
#[test]
#[ignore = "diagnostic: cargo test --test engine_metrics_fidelity -- --ignored --nocapture"]
fn dump_roundtrip_of_one_real_family() {
    use prometheus::Encoder;
    let families = parse_exposition(WITH_CREATED).expect("parse");
    let target: Vec<_> = families
        .into_iter()
        .filter(|f| f.name() == "vllm:time_to_first_token_seconds")
        .collect();
    let mut buf = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&target, &mut buf)
        .expect("encode");
    println!("{}", String::from_utf8(buf).unwrap());
}
