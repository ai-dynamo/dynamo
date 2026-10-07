// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The delivery rule for one arm of a load point.
//!
//! Inputs: every phantom publisher's summary (`--summary-out`) and the serving indexer's static
//! source accounting (`DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT`, read three report intervals
//! after the last publisher exited). A load point is invalid when either arm is. An arm is
//! invalid when any of these holds:
//!
//! - **Exact delivery per phantom.** For every phantom, the events and write blocks the indexer
//!   admitted, and its first and last admitted event IDs, must equal what the publisher planned
//!   (event IDs run contiguously from 1). Without gaps or high-water-mark drops delivered equals
//!   planned once the indexer drains, so any difference is a loss, a duplicate, or a stale file.
//! - **Timed window.** The indexer marks the timed start and end
//!   (`DYN_EXPERIMENT_STATIC_KV_TIMED_{START,END}_UNIX_MS`); both marks are required. At the
//!   start, the admitted totals must equal the planned warm-up exactly (no warm-up spilling into
//!   the window, no timed traffic before it). Between the marks, admitted write blocks must reach
//!   `min_window_fraction` of the planned timed blocks and must not exceed them. The start mark
//!   must match the publishers' start, and the end mark must fall within `max_end_grace_ms` after
//!   the publishers' stop.
//! - **Drained at both edges.** A FIFO barrier through the indexer's event queues must pass
//!   within `max_drain_ms` at each mark: admission feeds unbounded queues, so an arm that falls
//!   behind would otherwise neither drop nor push back.
//! - **Paced as planned.** Each publisher's timed lag p99 and maximum stay within bounds, and its
//!   last timed send is at most `max_send_overrun_ms` after the stop.
//! - Any gap reset (`ResetDegraded`), any other rank reset after a phantom indexed events, any
//!   late warm-up, publisher send error or interruption, a static-source count different from
//!   the phantoms hosted, an `endpoints_per_sub` other than the expected one, or an accounting
//!   file written less than two report intervals after the last publisher finished.
//!
//! Warnings (never invalidating by themselves): high-water-mark drops and resubscriptions; both
//! also break exact delivery.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;

/// Bounds of the rule; [`Rule::default`] holds the documented defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Rule {
    pub min_window_fraction: f64,
    pub max_drain_ms: f64,
    pub max_lag_p99_ms: f64,
    pub max_lag_ms: f64,
    pub max_send_overrun_ms: u64,
    pub max_end_grace_ms: u64,
    pub expect_endpoints_per_sub: Option<u64>,
}

impl Default for Rule {
    fn default() -> Self {
        Self {
            min_window_fraction: 0.99,
            max_drain_ms: 1000.0,
            max_lag_p99_ms: 50.0,
            max_lag_ms: 1000.0,
            max_send_overrun_ms: 1000,
            max_end_grace_ms: 10_000,
            expect_endpoints_per_sub: None,
        }
    }
}

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

    fn minus(self, earlier: Self) -> Self {
        Self {
            events: self.events.saturating_sub(earlier.events),
            write_blocks: self.write_blocks.saturating_sub(earlier.write_blocks),
        }
    }
}

/// One edge of the indexer's timed window.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Mark {
    pub requested_unix_ms: u64,
    pub taken_unix_ms: u64,
    pub totals: Blocks,
    pub drain_ms: Option<f64>,
}

impl Mark {
    fn from_json(value: &Value) -> Result<Option<Self>> {
        if value.is_null() {
            return Ok(None);
        }
        Ok(Some(Self {
            requested_unix_ms: field_u64(value, "requested_unix_ms")?,
            taken_unix_ms: field_u64(value, "taken_unix_ms")?,
            totals: Blocks::from_json(field(value, "totals")?)?,
            drain_ms: value["drain_ms"].as_f64(),
        }))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub valid: bool,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub rule: Rule,
    pub phantoms: u64,
    pub planned_warmup: Blocks,
    pub planned_timed: Blocks,
    pub sent: Blocks,
    pub delivered: Blocks,
    /// Phantoms whose admitted events, blocks, or first/last event IDs differ from the plan.
    pub mismatched_phantoms: u64,
    pub start: Option<Mark>,
    pub end: Option<Mark>,
    /// Admitted between the start and end marks.
    pub delivered_window: Option<Blocks>,
    pub delivered_window_fraction: Option<f64>,
    pub max_timed_lag_p99_ms: Option<f64>,
    pub max_timed_lag_ms: Option<f64>,
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

/// Rows of a `columns` + `rows` table, keyed by their first column.
fn table(value: &Value, columns_key: &str, rows_key: &str) -> Result<Vec<HashMap<String, u64>>> {
    let columns: Vec<&str> = field(value, columns_key)?
        .as_array()
        .with_context(|| format!("{columns_key} is not a list"))?
        .iter()
        .map(|column| column.as_str().context("a column name is not a string"))
        .collect::<Result<_>>()?;
    field(value, rows_key)?
        .as_array()
        .with_context(|| format!("{rows_key} is not a list"))?
        .iter()
        .map(|row| {
            let row = row.as_array().context("a row is not a list")?;
            ensure!(row.len() == columns.len(), "a row has the wrong width");
            columns
                .iter()
                .zip(row)
                .map(|(column, cell)| {
                    Ok((
                        column.to_string(),
                        cell.as_u64().context("a cell is not an unsigned integer")?,
                    ))
                })
                .collect()
        })
        .collect()
}

/// Apply `rule` to one arm: `publishers` are publisher summaries and `indexer` is the accounting
/// file the serving indexer wrote.
pub fn check(publishers: &[Value], indexer: &Value, rule: Rule) -> Result<Verdict> {
    ensure!(!publishers.is_empty(), "no publisher summaries");
    let mut reasons = Vec::new();
    let mut warnings = Vec::new();
    let mut phantoms = 0;
    let mut planned_warmup = Blocks::default();
    let mut planned_timed = Blocks::default();
    let mut sent = Blocks::default();
    let mut finished_unix_ms = 0;
    let mut start_at = None;
    let mut stop_at = None;
    let (mut late_warmups, mut send_errors, mut interrupted) = (0, 0, false);
    let (mut hwm_drops, mut resubscribed) = (0, 0);
    let (mut lag_p99_ms, mut lag_max_ms) = (None::<f64>, None::<f64>);
    let mut planned_rows = HashMap::new();
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

        let start = field_u64(summary, "plan.start_at_unix_ms")?;
        if start_at.is_some_and(|known| known != start) {
            reasons.push(format!(
                "publishers disagree on the timed start ({start_at:?} vs {start})"
            ));
        }
        start_at = Some(start);
        if let Some(stop) = summary["stop_at_unix_ms"].as_u64() {
            stop_at = Some(stop_at.map_or(stop, |known: u64| known.max(stop)));
        }
        if let Some(p99) = summary["timed_lag"]["p99_ms"].as_f64() {
            lag_p99_ms = Some(lag_p99_ms.map_or(p99, |known| known.max(p99)));
            let max = summary["timed_lag"]["max_ms"].as_f64().unwrap_or(p99);
            lag_max_ms = Some(lag_max_ms.map_or(max, |known| known.max(max)));
        }
        if let (Some(last), Some(stop)) = (
            summary["last_timed_send_unix_ms"].as_u64(),
            summary["stop_at_unix_ms"].as_u64(),
        ) && last > stop + rule.max_send_overrun_ms
        {
            reasons.push(format!(
                "a publisher's last timed send ({last}) is {} ms after its stop ({stop}), more than {} ms",
                last - stop,
                rule.max_send_overrun_ms
            ));
        }
        for row in table(summary, "per_phantom_columns", "per_phantom")? {
            planned_rows.insert(row["worker_id"], row);
        }
    }
    if let Some(p99) = lag_p99_ms
        && p99 > rule.max_lag_p99_ms
    {
        reasons.push(format!(
            "timed send lag p99 {p99:.1} ms exceeds {} ms",
            rule.max_lag_p99_ms
        ));
    }
    if let Some(max) = lag_max_ms
        && max > rule.max_lag_ms
    {
        reasons.push(format!(
            "timed send lag max {max:.1} ms exceeds {} ms",
            rule.max_lag_ms
        ));
    }
    if stop_at.is_none() {
        reasons.push("the publishers ran without --duration-s, so the window has no stop".into());
    }

    let accounting = field(indexer, "accounting")?;
    let delivered = Blocks::from_json(field(accounting, "totals")?)?;

    // Exact delivery per phantom.
    let delivered_rows: HashMap<u64, HashMap<String, u64>> =
        table(indexer, "source_columns", "sources")?
            .into_iter()
            .map(|row| (row["worker_id"], row))
            .collect();
    let mut mismatches = Vec::new();
    for (worker_id, planned) in &planned_rows {
        let got = delivered_rows.get(worker_id);
        let got_value = |key: &str| got.map_or(0, |row| row[key]);
        let planned_events = planned["planned_events"];
        let expected_first = u64::from(planned_events > 0);
        let actual = (
            got_value("events"),
            got_value("write_blocks"),
            got_value("first_event_id"),
            got_value("last_event_id"),
        );
        let expected = (
            planned_events,
            planned["planned_write_blocks"],
            expected_first,
            planned["planned_last_event_id"],
        );
        if got.is_none() || actual != expected {
            mismatches.push((*worker_id, expected, actual, planned["sent_events"]));
        }
    }
    mismatches.sort_unstable();
    if !mismatches.is_empty() {
        let examples: Vec<String> = mismatches
            .iter()
            .take(8)
            .map(|(worker, expected, actual, sent_events)| {
                format!(
                    "{worker:#x}: planned (events, blocks, first, last) {expected:?}, sent {sent_events} events, admitted {actual:?}"
                )
            })
            .collect();
        reasons.push(format!(
            "{} of {} phantoms were not delivered exactly as planned, e.g. {}",
            mismatches.len(),
            planned_rows.len(),
            examples.join("; ")
        ));
    }
    if (planned_rows.len() as u64) != phantoms {
        reasons.push(format!(
            "the publisher summaries list {} phantoms but host {phantoms}",
            planned_rows.len()
        ));
    }
    let foreign = delivered_rows
        .keys()
        .filter(|worker| !planned_rows.contains_key(worker))
        .count();
    if foreign > 0 {
        reasons.push(format!(
            "the indexer accounts for {foreign} static sources no publisher summary lists"
        ));
    }

    // The timed window.
    let start = Mark::from_json(&indexer["start"])?;
    let end = Mark::from_json(&indexer["end"])?;
    let mut delivered_window = None;
    let mut delivered_window_fraction = None;
    match (&start, &end) {
        (Some(start), Some(end)) => {
            if start_at.is_some_and(|start_at| start_at != start.requested_unix_ms) {
                reasons.push(format!(
                    "the indexer marked the timed start at {} but the publishers started at {start_at:?}",
                    start.requested_unix_ms
                ));
            }
            if let Some(stop) = stop_at
                && (end.requested_unix_ms < stop
                    || end.requested_unix_ms > stop + rule.max_end_grace_ms)
            {
                reasons.push(format!(
                    "the indexer's end mark ({}) must fall within {} ms after the publishers' stop ({stop})",
                    end.requested_unix_ms, rule.max_end_grace_ms
                ));
            }
            if start.totals != planned_warmup {
                reasons.push(format!(
                    "at the timed start the indexer had admitted {:?}, not the planned warm-up {planned_warmup:?}",
                    start.totals
                ));
            }
            let window = end.totals.minus(start.totals);
            let window_fraction = fraction(window.write_blocks, planned_timed.write_blocks);
            if window_fraction < rule.min_window_fraction {
                reasons.push(format!(
                    "window: admitted {} of {} planned timed write blocks between the marks ({window_fraction:.4} < {})",
                    window.write_blocks, planned_timed.write_blocks, rule.min_window_fraction
                ));
            }
            if window.write_blocks > planned_timed.write_blocks
                || window.events > planned_timed.events
            {
                reasons.push(format!(
                    "window: admitted {window:?} exceeds the planned timed {planned_timed:?} (spillover)"
                ));
            }
            for (label, mark) in [("start", start), ("end", end)] {
                match mark.drain_ms {
                    Some(ms) if ms <= rule.max_drain_ms => {}
                    Some(ms) => reasons.push(format!(
                        "the indexer took {ms:.0} ms to drain its event queues at the {label} mark (max {} ms)",
                        rule.max_drain_ms
                    )),
                    None => reasons.push(format!(
                        "the indexer's {label} mark has no drain measurement: {}",
                        indexer[label]["drain_error"]
                    )),
                }
            }
            delivered_window = Some(window);
            delivered_window_fraction = Some(window_fraction);
        }
        _ => reasons.push(
            "the indexer did not mark both edges of the timed window (set DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS and _END_UNIX_MS, both in the future at launch)"
                .to_string(),
        ),
    }

    for (key, what) in [
        ("gap_resets", "gap resets (ResetDegraded)"),
        ("rank_resets", "rank resets after indexed events"),
        (
            "sources_first_event_late",
            "phantoms whose first admitted event was not event 1",
        ),
    ] {
        let count = field_u64(accounting, key)?;
        if count > 0 {
            reasons.push(format!("{count} {what}"));
        }
    }
    let static_sources = field_u64(indexer, "static_sources")?;
    if static_sources != phantoms {
        reasons.push(format!(
            "the indexer accounts for {static_sources} static sources but the publishers host {phantoms}"
        ));
    }
    let endpoints_per_sub = indexer["endpoints_per_sub"].as_u64();
    if let Some(expected) = rule.expect_endpoints_per_sub
        && endpoints_per_sub != Some(expected)
    {
        reasons.push(format!(
            "the indexer ran with endpoints_per_sub {endpoints_per_sub:?}, expected {expected}"
        ));
    }
    let snapshot_unix_ms = field_u64(indexer, "t_unix_ms")?;
    let interval_ms = (indexer["report_interval_s"].as_f64().unwrap_or(10.0) * 1e3) as u64;
    if snapshot_unix_ms < finished_unix_ms + 2 * interval_ms {
        reasons.push(format!(
            "the indexer file ({snapshot_unix_ms}) is less than two report intervals after the last publisher finish ({finished_unix_ms}); read it later"
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

    Ok(Verdict {
        valid: reasons.is_empty(),
        reasons,
        warnings,
        rule,
        phantoms,
        planned_warmup,
        planned_timed,
        sent,
        delivered,
        mismatched_phantoms: mismatches.len() as u64,
        start,
        end,
        delivered_window,
        delivered_window_fraction,
        max_timed_lag_p99_ms: lag_p99_ms,
        max_timed_lag_ms: lag_max_ms,
        endpoints_per_sub,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const START: u64 = 10_000;
    const STOP: u64 = 20_000;

    /// One publisher hosting phantoms `first` and `first + 1`; each plans 5 warm-up and 5 timed
    /// events of `warmup` / `timed` blocks in total, and sent everything.
    fn publisher(first: u64, warmup: u64, timed: u64) -> Value {
        let rows: Vec<Value> = (first..first + 2)
            .map(|worker| {
                json!([
                    worker,
                    10,
                    (warmup + timed) / 2,
                    10,
                    10,
                    (warmup + timed) / 2,
                    10
                ])
            })
            .collect();
        json!({
            "kind": "summary",
            "finished_unix_ms": STOP + 1_000,
            "stop_at_unix_ms": STOP,
            "last_timed_send_unix_ms": STOP - 5,
            "timed_lag": { "p99_ms": 2.0, "max_ms": 3.0 },
            "plan": {
                "phantoms": [first, first + 2],
                "start_at_unix_ms": START,
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
            "per_phantom_columns": ["worker_id", "planned_events", "planned_write_blocks",
                "planned_last_event_id", "sent_events", "sent_write_blocks", "last_sent_event_id"],
            "per_phantom": rows,
        })
    }

    fn mark(at: u64, events: u64, write_blocks: u64) -> Value {
        json!({
            "requested_unix_ms": at,
            "taken_unix_ms": at + 3,
            "totals": { "events": events, "write_blocks": write_blocks },
            "drain_ms": 0.4,
        })
    }

    /// The indexer side of a run where every phantom was delivered exactly.
    fn indexer(warmup: u64, timed: u64) -> Value {
        let rows: Vec<Value> = (0..4)
            .map(|worker| json!([worker, 10, (warmup + timed) / 4, 1, 10, 0]))
            .collect();
        json!({
            "kind": "interval",
            "t_unix_ms": STOP + 30_000,
            "report_interval_s": 10.0,
            "static_sources": 4,
            "endpoints_per_sub": 2,
            "start": mark(START, 20, warmup),
            "end": mark(STOP + 2_000, 40, warmup + timed),
            "accounting": {
                "totals": { "events": 40, "write_blocks": warmup + timed },
                "sources_with_events": 4,
                "sources_first_event_late": 0,
                "gap_resets": 0,
                "rank_resets": 0,
            },
            "source_columns": ["worker_id", "events", "write_blocks", "first_event_id",
                "last_event_id", "gap_resets"],
            "sources": rows,
        })
    }

    fn publishers() -> [Value; 2] {
        [publisher(0, 100, 100), publisher(2, 100, 100)]
    }

    #[test]
    fn exact_delivery_inside_a_drained_window_is_valid() {
        let rule = Rule {
            expect_endpoints_per_sub: Some(2),
            ..Rule::default()
        };
        let verdict = check(&publishers(), &indexer(200, 200), rule).unwrap();
        assert!(verdict.valid, "{:?}", verdict.reasons);
        assert_eq!(verdict.mismatched_phantoms, 0);
        assert_eq!(verdict.delivered_window_fraction, Some(1.0));
    }

    #[test]
    fn a_tail_loss_within_the_old_tolerance_is_invalid() {
        // The reviewed counterexample: 2% short at the end of the timed window used to pass the
        // 98% aggregate rule. Exactness per phantom catches it.
        let mut accounting = indexer(200, 200);
        accounting["sources"][3][2] = json!(96);
        accounting["sources"][3][4] = json!(9);
        let verdict = check(&publishers(), &accounting, Rule::default()).unwrap();
        assert!(!verdict.valid);
        assert_eq!(verdict.mismatched_phantoms, 1);
        assert!(
            verdict.reasons[0].contains("not delivered exactly"),
            "{:?}",
            verdict.reasons
        );
    }

    #[test]
    fn warm_up_spilling_into_the_window_is_invalid() {
        // 196 warm-up blocks at the start mark: the last 4 were admitted inside the window.
        let mut accounting = indexer(200, 200);
        accounting["start"] = mark(START, 20, 196);
        let verdict = check(&publishers(), &accounting, Rule::default()).unwrap();
        assert!(!verdict.valid);
        assert!(
            verdict
                .reasons
                .iter()
                .any(|reason| reason.contains("not the planned warm-up")),
            "{:?}",
            verdict.reasons
        );
        assert!(
            verdict
                .reasons
                .iter()
                .any(|reason| reason.contains("spillover")),
            "{:?}",
            verdict.reasons
        );
    }

    #[test]
    fn a_short_window_or_missing_marks_are_invalid() {
        // Everything arrived, but 10 of the 200 timed blocks only after the end mark.
        let mut accounting = indexer(200, 200);
        accounting["end"] = mark(STOP + 2_000, 38, 390);
        let verdict = check(&publishers(), &accounting, Rule::default()).unwrap();
        assert!(
            verdict
                .reasons
                .iter()
                .any(|reason| reason.starts_with("window:"))
        );

        accounting["end"] = Value::Null;
        let verdict = check(&publishers(), &accounting, Rule::default()).unwrap();
        assert!(verdict.reasons[0].contains("did not mark both edges"));
    }

    #[test]
    fn an_undrained_mark_lag_or_stale_file_is_invalid() {
        let mut accounting = indexer(200, 200);
        accounting["end"]["drain_ms"] = json!(5_000.0);
        let verdict = check(&publishers(), &accounting, Rule::default()).unwrap();
        assert!(verdict.reasons[0].contains("to drain its event queues at the end mark"));

        let mut slow = publishers();
        slow[1]["timed_lag"]["p99_ms"] = json!(80.0);
        slow[1]["last_timed_send_unix_ms"] = json!(STOP + 2_000);
        let verdict = check(&slow, &indexer(200, 200), Rule::default()).unwrap();
        assert_eq!(verdict.reasons.len(), 2, "{:?}", verdict.reasons);

        let mut stale = indexer(200, 200);
        stale["t_unix_ms"] = json!(STOP + 15_000);
        let verdict = check(&publishers(), &stale, Rule::default()).unwrap();
        assert!(verdict.reasons[0].contains("two report intervals"));

        let rule = Rule {
            expect_endpoints_per_sub: Some(4),
            ..Rule::default()
        };
        let verdict = check(&publishers(), &indexer(200, 200), rule).unwrap();
        assert!(verdict.reasons[0].contains("endpoints_per_sub"));
    }

    #[test]
    fn any_gap_reset_is_invalid() {
        let mut accounting = indexer(200, 200);
        accounting["accounting"]["gap_resets"] = json!(1);
        let verdict = check(&publishers(), &accounting, Rule::default()).unwrap();
        assert_eq!(verdict.reasons, vec!["1 gap resets (ResetDegraded)"]);
    }
}
