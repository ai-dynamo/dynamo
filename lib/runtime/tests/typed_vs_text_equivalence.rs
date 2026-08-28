// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Does the typed engine-metrics contract carry the same metrics as the text
//! one? Both fixtures were captured from a single vLLM 0.18.0 registry in one
//! process, so any difference is the contract, not drift between captures.

use dynamo_runtime::metrics::{prom_text::parse_exposition, prom_typed};
use prometheus::proto::{MetricFamily, MetricType};
use std::collections::BTreeMap;

const TEXT: &str = include_str!("data/vllm-same-registry.txt");
const TYPED: &str = include_str!("data/vllm-same-registry-typed.json");

type SeriesKey = (String, Vec<(String, String)>);

/// Expand families to the series a Prometheus consumer would observe.
fn series(families: &[MetricFamily]) -> BTreeMap<SeriesKey, String> {
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
                    for b in h.get_bucket() {
                        let mut le = labels.clone();
                        le.push(("le".into(), format!("{:?}", b.upper_bound())));
                        le.sort();
                        out.insert(
                            (format!("{name}_bucket"), le),
                            (b.cumulative_count() as f64).to_string(),
                        );
                    }
                    // +Inf is implicit in the family; reconstruct as the encoder does.
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
                        let mut qk = labels.clone();
                        qk.push(("quantile".into(), format!("{:?}", q.quantile())));
                        qk.sort();
                        out.insert((name.clone(), qk), q.value().to_string());
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

/// `_created` is opt-out upstream and the two contracts represent it
/// differently; compare the measurements both agree are measurements.
///
/// `process_*` and `python_*` come from prometheus_client's default collectors
/// and are sampled live at collect time, so they legitimately differ between
/// two sequential captures (writing the text fixture moves `process_open_fds`).
/// They are not engine metrics and are out of scope here.
fn comparable(m: BTreeMap<SeriesKey, String>) -> BTreeMap<SeriesKey, String> {
    m.into_iter()
        .filter(|((n, _), _)| {
            !n.ends_with("_created") && !n.starts_with("process_") && !n.starts_with("python_")
        })
        .collect()
}

fn both() -> (BTreeMap<SeriesKey, String>, BTreeMap<SeriesKey, String>) {
    let from_text = parse_exposition(TEXT).expect("parse text");
    let typed: Vec<prom_typed::TypedFamily> = serde_json::from_str(TYPED).expect("parse typed");
    let from_typed = prom_typed::build_families(typed);
    (
        comparable(series(&from_text)),
        comparable(series(&from_typed)),
    )
}

/// The decisive question: does the typed contract lose or invent any series?
#[test]
fn typed_and_text_agree_on_which_series_exist() {
    let (text, typed) = both();
    let missing: Vec<_> = text.keys().filter(|k| !typed.contains_key(*k)).collect();
    let extra: Vec<_> = typed.keys().filter(|k| !text.contains_key(*k)).collect();

    assert!(
        missing.is_empty(),
        "{} series in the text contract are missing from typed, e.g. {:?}",
        missing.len(),
        &missing[..missing.len().min(5)]
    );
    assert!(
        extra.is_empty(),
        "{} series only in typed, e.g. {:?}",
        extra.len(),
        &extra[..extra.len().min(5)]
    );
    assert!(
        text.len() > 300,
        "fixture should be substantial, got {}",
        text.len()
    );
}

/// And do they agree on the values?
#[test]
fn typed_and_text_agree_on_values() {
    let (text, typed) = both();
    let wrong: Vec<_> = text
        .iter()
        .filter_map(|(k, v)| {
            typed
                .get(k)
                .filter(|t| {
                    v.parse::<f64>()
                        .ok()
                        .zip(t.parse::<f64>().ok())
                        .is_none_or(|(a, b)| a != b && !(a.is_nan() && b.is_nan()))
                })
                .map(|t| (k.clone(), v.clone(), t.clone()))
        })
        .take(5)
        .collect();
    assert!(wrong.is_empty(), "value mismatches: {wrong:?}");
}
