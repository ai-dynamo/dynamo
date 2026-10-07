// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The delivery rule for one arm of a load point.
//!
//! Inputs: every phantom publisher's summary (`--summary-out`) and the serving indexer's static
//! source accounting (`DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT`, read after the last publisher
//! exited). A load point is invalid when either arm is invalid. An arm is invalid when:
//!
//! - delivered write blocks are below `min_delivered_fraction` (default 0.98) of planned, over
//!   the run and, when the indexer split warm-up from timed, over each part;
//! - any gap reset occurred (`ResetDegraded`), or any other rank reset discarded indexed state;
//! - any phantom's first applied event was not event 1 (its prefix was lost), or a phantom with
//!   planned events delivered none;
//! - any warm-up ran past the timed start, a publisher hit send errors or was interrupted, the
//!   indexer accounts for a different number of static sources than the publishers host, or its
//!   snapshot predates the last publisher's finish.
//!
//! Warnings (not invalidating): high-water-mark drops at the publishers and phantoms whose
//! subscriber reconnected; both normally also surface as gap resets or a delivery shortfall.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;

pub const DEFAULT_MIN_DELIVERED_FRACTION: f64 = 0.98;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Blocks {
    pub events: u64,
    pub write_blocks: u64,
}

impl Blocks {
    fn from_json(value: &Value) -> Result<Self> {
        Ok(Self {
            events: field_u64(value, "events")?,
            write_blocks: field_u64(value, "write_blocks")?,
        })
    }

    fn add(&mut self, other: Self) {
        self.events += other.events;
        self.write_blocks += other.write_blocks;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub valid: bool,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub phantoms: u64,
    pub planned_warmup: Blocks,
    pub planned_timed: Blocks,
    pub sent: Blocks,
    pub delivered: Blocks,
    pub delivered_warmup: Option<Blocks>,
    pub delivered_timed: Option<Blocks>,
    pub delivered_fraction: f64,
    pub delivered_timed_fraction: Option<f64>,
    /// Delivered equals sent, event for event and block for block.
    pub exact: bool,
    pub endpoints_per_sub: Option<u64>,
}

fn field<'a>(value: &'a Value, path: &str) -> Result<&'a Value> {
    let pointer = format!("/{}", path.replace('.', "/"));
    value
        .pointer(&pointer)
        .with_context(|| format!("missing {path}"))
}

fn field_u64(value: &Value, path: &str) -> Result<u64> {
    field(value, path)?
        .as_u64()
        .with_context(|| format!("{path} is not an unsigned integer"))
}

fn fraction(delivered: u64, planned: u64) -> f64 {
    if planned == 0 {
        return 1.0;
    }
    delivered as f64 / planned as f64
}

/// Apply the rule to one arm: `publishers` are publisher summaries and `indexer` is the
/// accounting report the serving indexer wrote.
pub fn check(
    publishers: &[Value],
    indexer: &Value,
    min_delivered_fraction: f64,
) -> Result<Verdict> {
    ensure!(!publishers.is_empty(), "no publisher summaries");
    let mut reasons = Vec::new();
    let mut warnings = Vec::new();
    let mut phantoms = 0;
    let mut planned_warmup = Blocks::default();
    let mut planned_timed = Blocks::default();
    let mut sent = Blocks::default();
    let mut finished_unix_ms = 0;
    let (mut late_warmups, mut send_errors, mut interrupted) = (0, 0, false);
    let (mut hwm_drops, mut resubscribed) = (0, 0);
    for summary in publishers {
        ensure!(
            summary["kind"] == "summary",
            "a publisher file is not a summary"
        );
        let range = field(summary, "plan.phantoms")?;
        phantoms += range[1].as_u64().context("plan.phantoms")?
            - range[0].as_u64().context("plan.phantoms")?;
        planned_warmup.add(Blocks::from_json(field(summary, "plan.planned.warmup")?)?);
        planned_timed.add(Blocks::from_json(field(summary, "plan.planned.timed")?)?);
        sent.add(Blocks::from_json(field(summary, "sent")?)?);
        finished_unix_ms = finished_unix_ms.max(field_u64(summary, "finished_unix_ms")?);
        late_warmups += field_u64(summary, "late_warmups")?;
        send_errors += field_u64(summary, "send_errors")?;
        interrupted |= summary["interrupted"].as_bool().unwrap_or(true);
        hwm_drops += field_u64(summary, "hwm_dropped_envelopes")?;
        resubscribed += field_u64(summary, "resubscribed_phantoms")?
            + field_u64(summary, "unsubscribed_phantoms")?;
    }

    let accounting = field(indexer, "accounting")?;
    let delivered = Blocks::from_json(field(accounting, "totals")?)?;
    let blocks_or_none = |key: &str| -> Result<Option<Blocks>> {
        match &indexer[key] {
            Value::Null => Ok(None),
            value => Blocks::from_json(value).map(Some),
        }
    };
    let delivered_warmup = blocks_or_none("warmup")?;
    let delivered_timed = blocks_or_none("timed")?;

    let mut planned = planned_warmup;
    planned.add(planned_timed);
    let delivered_fraction = fraction(delivered.write_blocks, planned.write_blocks);
    let mut require = |label: &str, delivered: u64, planned: u64| {
        let fraction = fraction(delivered, planned);
        if fraction < min_delivered_fraction {
            reasons.push(format!(
                "{label}: delivered {delivered} of {planned} planned write blocks ({fraction:.4} < {min_delivered_fraction})"
            ));
        }
        fraction
    };
    require("run", delivered.write_blocks, planned.write_blocks);
    let delivered_timed_fraction = match (delivered_warmup, delivered_timed) {
        (Some(warmup), Some(timed)) => {
            require("warm-up", warmup.write_blocks, planned_warmup.write_blocks);
            Some(require(
                "timed",
                timed.write_blocks,
                planned_timed.write_blocks,
            ))
        }
        _ => {
            warnings.push(
                "the indexer did not split warm-up from timed (set DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS)"
                    .to_string(),
            );
            None
        }
    };

    for (key, what) in [
        ("gap_resets", "gap resets (ResetDegraded)"),
        ("rank_resets", "rank resets after indexed events"),
        (
            "sources_first_event_late",
            "phantoms whose first applied event was not event 1",
        ),
    ] {
        let count = field_u64(accounting, key)?;
        if count > 0 {
            reasons.push(format!("{count} {what}"));
        }
    }
    let sources_with_events = field_u64(accounting, "sources_with_events")?;
    if planned_warmup.events > 0 && sources_with_events < phantoms {
        reasons.push(format!(
            "only {sources_with_events} of {phantoms} phantoms delivered any event"
        ));
    }
    let static_sources = field_u64(indexer, "static_sources")?;
    if static_sources != phantoms {
        reasons.push(format!(
            "the indexer accounts for {static_sources} static sources but the publishers host {phantoms}"
        ));
    }
    let snapshot_unix_ms = field_u64(indexer, "t_unix_ms")?;
    if snapshot_unix_ms < finished_unix_ms {
        reasons.push(format!(
            "the indexer snapshot ({snapshot_unix_ms}) predates the last publisher finish ({finished_unix_ms}); read it after the publishers exit"
        ));
    }
    if late_warmups > 0 {
        reasons.push(format!(
            "{late_warmups} phantoms finished warm-up after the timed start"
        ));
    }
    if send_errors > 0 || interrupted {
        reasons.push(format!(
            "publisher send errors {send_errors}, interrupted {interrupted}"
        ));
    }
    if hwm_drops > 0 {
        warnings.push(format!(
            "{hwm_drops} envelopes dropped at the publishers' high-water mark"
        ));
    }
    if resubscribed > 0 {
        warnings.push(format!(
            "{resubscribed} phantom subscriptions reconnected or left after the gate"
        ));
    }
    if delivered.events > sent.events || delivered.write_blocks > sent.write_blocks {
        reasons.push(format!(
            "delivered {delivered:?} exceeds sent {sent:?}: accounting and summaries are from different runs"
        ));
    }

    Ok(Verdict {
        valid: reasons.is_empty(),
        reasons,
        warnings,
        phantoms,
        planned_warmup,
        planned_timed,
        sent,
        delivered,
        delivered_warmup,
        delivered_timed,
        delivered_fraction,
        delivered_timed_fraction,
        exact: delivered == sent,
        endpoints_per_sub: indexer["endpoints_per_sub"].as_u64(),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// One publisher hosting two phantoms that plan `warmup` + `timed` write blocks.
    fn publisher(first: u64, warmup: u64, timed: u64) -> Value {
        json!({
            "kind": "summary",
            "finished_unix_ms": 1_000,
            "plan": {
                "phantoms": [first, first + 2],
                "planned": {
                    "warmup": { "events": 10, "write_blocks": warmup },
                    "timed": { "events": 10, "write_blocks": timed },
                },
            },
            "sent": { "events": 20, "write_blocks": warmup + timed },
            "late_warmups": 0,
            "send_errors": 0,
            "interrupted": false,
            "hwm_dropped_envelopes": 0,
            "resubscribed_phantoms": 0,
            "unsubscribed_phantoms": 0,
        })
    }

    fn indexer(warmup: u64, timed: u64, gap_resets: u64) -> Value {
        json!({
            "kind": "final",
            "t_unix_ms": 2_000,
            "static_sources": 4,
            "endpoints_per_sub": 3,
            "warmup": { "events": 20, "write_blocks": warmup },
            "timed": { "events": 20, "write_blocks": timed },
            "accounting": {
                "totals": { "events": 40, "write_blocks": warmup + timed },
                "sources_with_events": 4,
                "sources_first_event_late": 0,
                "gap_resets": gap_resets,
                "rank_resets": 0,
            },
        })
    }

    #[test]
    fn full_delivery_is_valid_and_exact() {
        let publishers = [publisher(0, 100, 100), publisher(2, 100, 100)];
        let verdict = check(
            &publishers,
            &indexer(200, 200, 0),
            DEFAULT_MIN_DELIVERED_FRACTION,
        )
        .unwrap();
        assert!(verdict.valid, "{:?}", verdict.reasons);
        assert!(verdict.exact);
        assert_eq!(verdict.planned_timed.write_blocks, 200);
        assert_eq!(verdict.delivered_timed_fraction, Some(1.0));
    }

    #[test]
    fn a_timed_shortfall_hidden_by_a_large_warm_up_is_invalid() {
        // 2190 of 2200 blocks overall passes 0.98; 190 of 200 timed blocks does not.
        let publishers = [publisher(0, 1000, 100), publisher(2, 1000, 100)];
        let verdict = check(
            &publishers,
            &indexer(2000, 190, 0),
            DEFAULT_MIN_DELIVERED_FRACTION,
        )
        .unwrap();
        assert!(verdict.delivered_fraction > DEFAULT_MIN_DELIVERED_FRACTION);
        assert!(!verdict.valid);
        assert_eq!(verdict.reasons.len(), 1);
        assert!(
            verdict.reasons[0].starts_with("timed:"),
            "{:?}",
            verdict.reasons
        );
        assert!(!verdict.exact);
    }

    #[test]
    fn any_gap_reset_or_stale_snapshot_is_invalid() {
        let publishers = [publisher(0, 100, 100), publisher(2, 100, 100)];
        let verdict = check(
            &publishers,
            &indexer(200, 200, 1),
            DEFAULT_MIN_DELIVERED_FRACTION,
        )
        .unwrap();
        assert_eq!(verdict.reasons, vec!["1 gap resets (ResetDegraded)"]);

        let mut stale = indexer(200, 200, 0);
        stale["t_unix_ms"] = json!(500);
        let verdict = check(&publishers, &stale, DEFAULT_MIN_DELIVERED_FRACTION).unwrap();
        assert!(!verdict.valid);
        assert!(verdict.reasons[0].contains("predates"));
    }
}
