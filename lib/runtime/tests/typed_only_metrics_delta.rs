// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! If /metrics were rendered from the typed contract instead of passing the
//! engine's own text through, what would change? Measured against one real
//! vLLM 0.18.0 registry captured in both representations.

use dynamo_runtime::metrics::prom_typed;
use prometheus::Encoder;
use std::collections::{BTreeMap, BTreeSet};

const TEXT: &str = include_str!("data/vllm-same-registry.txt");
const TYPED: &str = include_str!("data/vllm-same-registry-typed.json");

/// Series as a scraper sees them: name + sorted labels -> value string.
fn series(text: &str) -> BTreeMap<(String, Vec<(String, String)>), String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(split) = line.rfind(' ') else {
            continue;
        };
        let (head, value) = (&line[..split], line[split + 1..].to_string());
        let (name, labels) = match head.find('{') {
            Some(open) => {
                let close = head.rfind('}').unwrap_or(head.len());
                let mut pairs: Vec<(String, String)> = head[open + 1..close]
                    .split(',')
                    .filter(|p| !p.trim().is_empty())
                    .filter_map(|p| p.split_once('='))
                    .map(|(k, v)| (k.trim().into(), v.trim().trim_matches('"').into()))
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

fn typed_rendered() -> String {
    let typed: Vec<prom_typed::TypedFamily> = serde_json::from_str(TYPED).expect("typed fixture");
    let families = prom_typed::build_families(typed);
    let mut buf = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&families, &mut buf)
        .expect("encode");
    String::from_utf8(buf).expect("utf8")
}

/// Quantifies the delta rather than asserting it away.
#[test]
#[ignore = "diagnostic: cargo test --test typed_only_metrics_delta -- --ignored --nocapture"]
fn report_typed_only_delta() {
    let engine = series(TEXT);
    let ours = series(&typed_rendered());

    let engine_keys: BTreeSet<_> = engine.keys().cloned().collect();
    let our_keys: BTreeSet<_> = ours.keys().cloned().collect();

    let lost: Vec<_> = engine_keys.difference(&our_keys).cloned().collect();
    let added: Vec<_> = our_keys.difference(&engine_keys).cloned().collect();

    let mut created = 0;
    let mut le_format = 0;
    let mut other_lost = Vec::new();
    for (name, labels) in &lost {
        if name.ends_with("_created") {
            created += 1;
        } else if labels.iter().any(|(k, _)| k == "le" || k == "quantile") {
            le_format += 1;
        } else {
            other_lost.push((name.clone(), labels.clone()));
        }
    }

    let value_fmt = engine
        .iter()
        .filter(|(k, v)| ours.get(*k).is_some_and(|o| o != *v))
        .filter(|(k, v)| ours.get(*k).and_then(|o| o.parse::<f64>().ok()) != v.parse::<f64>().ok())
        .count();

    println!("engine /metrics series : {}", engine.len());
    println!("typed-rendered series  : {}", ours.len());
    println!();
    println!("lost: {} total", lost.len());
    println!("  _created (opt-out upstream)      : {created}");
    println!("  le/quantile number formatting    : {le_format}");
    println!("  other (would be real loss)       : {}", other_lost.len());
    for k in other_lost.iter().take(5) {
        println!("      {k:?}");
    }
    println!("added (not in engine output)       : {}", added.len());
    for k in added.iter().take(5) {
        println!("      {k:?}");
    }
    println!("numerically different values       : {value_fmt}");
}

/// Does `_created` reach the structured path today, via the text contract?
#[test]
#[ignore = "diagnostic"]
fn report_created_handling() {
    use dynamo_runtime::metrics::prom_text::parse_exposition;
    let from_text = parse_exposition(TEXT).expect("parse");
    let text_created = from_text
        .iter()
        .filter(|f| f.name().ends_with("_created"))
        .count();

    let typed: Vec<prom_typed::TypedFamily> = serde_json::from_str(TYPED).expect("typed");
    let typed_created_samples: usize = typed
        .iter()
        .map(|f| {
            f.samples
                .iter()
                .filter(|s| s.name.ends_with("_created"))
                .count()
        })
        .sum();
    let built = prom_typed::build_families(serde_json::from_str(TYPED).expect("typed"));
    let built_created = built
        .iter()
        .filter(|f| f.name().ends_with("_created"))
        .count();

    println!("text path  -> _created FAMILIES parsed : {text_created}");
    println!("typed input-> _created SAMPLES present : {typed_created_samples}");
    println!("typed build-> _created families kept   : {built_created}");
}
