// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `audit_diff` — offline trace-equivalence checker for `kvbm_audit` events.
//!
//! Parses two log files captured from `RUST_LOG=kvbm_audit=info` runs of
//! the disagg-bringup workload (e.g. once with `KVBM_DISAGG_LEADER=legacy`,
//! once with `=unified`), extracts the `kvbm_audit` events from each,
//! and compares the `(event, role, request_id)` signature sequences.
//!
//! Exits 0 on equivalence, 1 on divergence (with a per-position
//! L vs R diff matching the in-process test format from
//! `tests/audit_helpers/assert_event_signatures_equal`).
//!
//! ## Usage
//!
//! ```text
//! audit_diff --legacy <PATH> --unified <PATH>
//!            [--filter-prefixes <PREFIX,PREFIX,...>]
//!            [--multiset]
//! ```
//!
//! ## Input format
//!
//! Each log line is searched for the substring `kvbm_audit:` (the
//! target tag emitted by `crate::audit!` via `tracing::info!`).  The
//! tail after that marker is parsed as space-separated `key=value`
//! pairs, with `"…"`-quoted values supported.  Required fields are
//! `event`; `role` and `request_id` are optional.
//!
//! ## Filters
//!
//! `--filter-prefixes` drops events whose name starts with any of the
//! comma-separated prefixes.  Useful for whitelisting events that are
//! known to differ between leaders during transition (currently none —
//! both leaders emit the same audit shape).
//!
//! `--multiset` switches to multiset comparison: two streams are
//! equivalent if their (event, role, request_id) signatures appear
//! with the same multiplicity, regardless of ordering.  Use this if
//! the workload triggers HashSet-iteration-ordered audits (e.g. UCO
//! emissions where finished_sending/recving sets aren't sorted).
//!
//! `--normalize-request-ids` rewrites each captured `request_id` to a
//! stable counter (`req-1`, `req-2`, ...) in order of first
//! appearance, per file.  The two runs produce different vLLM-issued
//! request UUIDs but the audit shape is identical; normalizing makes
//! the signature comparison meaningful for live runs.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AuditEvent {
    event: String,
    role: Option<String>,
    request_id: Option<String>,
    fields: Vec<(String, String)>,
}

impl AuditEvent {
    fn signature(&self) -> (String, Option<String>, Option<String>) {
        (
            self.event.clone(),
            self.role.clone(),
            self.request_id.clone(),
        )
    }
}

#[derive(Debug, Default)]
struct Args {
    legacy: PathBuf,
    unified: PathBuf,
    filter_prefixes: Vec<String>,
    multiset: bool,
    normalize_request_ids: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut out = Args::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--legacy" => {
                out.legacy = args
                    .next()
                    .ok_or_else(|| "missing value for --legacy".to_string())?
                    .into();
            }
            "--unified" => {
                out.unified = args
                    .next()
                    .ok_or_else(|| "missing value for --unified".to_string())?
                    .into();
            }
            "--filter-prefixes" => {
                let val = args
                    .next()
                    .ok_or_else(|| "missing value for --filter-prefixes".to_string())?;
                out.filter_prefixes = val
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--multiset" => out.multiset = true,
            "--normalize-request-ids" => out.normalize_request_ids = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized arg: {other}")),
        }
    }
    if out.legacy.as_os_str().is_empty() {
        return Err("missing --legacy".to_string());
    }
    if out.unified.as_os_str().is_empty() {
        return Err("missing --unified".to_string());
    }
    Ok(out)
}

fn print_usage() {
    eprintln!(
        "audit_diff --legacy <PATH> --unified <PATH> \
         [--filter-prefixes <PREFIX,...>] [--multiset] \
         [--normalize-request-ids]"
    );
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(err) => {
            eprintln!("audit_diff: {err}");
            print_usage();
            return ExitCode::from(2);
        }
    };

    let legacy_events = match parse_file(&args.legacy) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "audit_diff: failed to parse {}: {err}",
                args.legacy.display()
            );
            return ExitCode::from(2);
        }
    };
    let unified_events = match parse_file(&args.unified) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "audit_diff: failed to parse {}: {err}",
                args.unified.display()
            );
            return ExitCode::from(2);
        }
    };

    let mut legacy = filter_prefixes(legacy_events, &args.filter_prefixes);
    let mut unified = filter_prefixes(unified_events, &args.filter_prefixes);
    if args.normalize_request_ids {
        normalize_request_ids(&mut legacy);
        normalize_request_ids(&mut unified);
    }

    eprintln!(
        "audit_diff: legacy={} events  unified={} events",
        legacy.len(),
        unified.len()
    );

    let equivalent = if args.multiset {
        compare_multiset(&legacy, &unified)
    } else {
        compare_sequence(&legacy, &unified)
    };

    if equivalent {
        eprintln!(
            "audit_diff: OK — sequences {} on signature triples",
            if args.multiset {
                "match as multisets"
            } else {
                "match in order"
            }
        );
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn parse_file(path: &PathBuf) -> Result<Vec<AuditEvent>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Some(event) = parse_audit_line(&line) {
            events.push(event);
        }
    }
    Ok(events)
}

/// Find the `kvbm_audit:` marker in a line and parse the trailing
/// `key=value` pairs into an `AuditEvent`.  Returns `None` for lines
/// that don't contain the marker or that lack the required `event`
/// field.  ANSI escape sequences (used by tracing-subscriber's
/// default fmt when ANSI is enabled) are stripped before parsing so
/// the same output can be diff'd whether or not the producer set
/// `NO_COLOR=1`.
fn parse_audit_line(line: &str) -> Option<AuditEvent> {
    let cleaned = strip_ansi_escapes(line);
    let idx = cleaned.find("kvbm_audit:")?;
    let body = &cleaned[idx + "kvbm_audit:".len()..];
    let pairs = parse_kv_pairs(body);
    let mut event = AuditEvent::default();
    for (k, v) in pairs {
        match k.as_str() {
            "event" => event.event = v,
            "role" => event.role = Some(v),
            "request_id" => event.request_id = Some(v),
            _ => event.fields.push((k, v)),
        }
    }
    if event.event.is_empty() {
        return None;
    }
    event.fields.sort_by(|a, b| a.0.cmp(&b.0));
    Some(event)
}

/// Strip CSI ANSI escape sequences (`ESC [ ... letter`) from a
/// string.  Tracing-subscriber emits these for color/styling when
/// ANSI is enabled (the default for fmt subscribers writing to
/// stderr, which is also true when stderr is redirected to a file
/// unless `NO_COLOR=1` is set).  We don't try to handle every ANSI
/// variant — just the CSI sequences tracing actually uses.
fn strip_ansi_escapes(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // ESC = 0x1B = '\x1b'.  Tracing's CSI form is `ESC [ ... letter`.
        if bytes[i] == 0x1B && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Skip past the final byte (range 0x40..=0x7E).
            i += 2;
            while i < bytes.len() && !(0x40..=0x7E).contains(&bytes[i]) {
                i += 1;
            }
            // Also skip the final byte itself.
            if i < bytes.len() {
                i += 1;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

/// Parse `key="value"` and `key=bare_value` pairs separated by
/// whitespace.  Quoted values may contain spaces; bare values run
/// until the next whitespace.  Backslash-escapes inside quoted values
/// are honored (`\\"`, `\\\\`).
fn parse_kv_pairs(input: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip leading whitespace and separators.
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Read key up to '='.
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            // Skip stray token without '='.
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            continue;
        }
        let key = std::str::from_utf8(&bytes[key_start..i])
            .unwrap_or("")
            .trim()
            .to_string();
        i += 1; // consume '='

        // Read value.  Quoted: read until matching '"', honoring
        // backslash escapes.  Bare: read until next whitespace.
        let value = if i < bytes.len() && bytes[i] == b'"' {
            i += 1;
            let val_start = i;
            let mut buf = Vec::new();
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    buf.push(bytes[i + 1]);
                    i += 2;
                } else {
                    buf.push(bytes[i]);
                    i += 1;
                }
            }
            if i < bytes.len() {
                i += 1; // consume closing '"'
            }
            let _ = val_start;
            String::from_utf8(buf).unwrap_or_default()
        } else {
            let val_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            std::str::from_utf8(&bytes[val_start..i])
                .unwrap_or("")
                .to_string()
        };

        if !key.is_empty() {
            out.push((key, value));
        }
    }
    out
}

/// Rewrite each `request_id` to `req-1`, `req-2`, ... in order of
/// first appearance.  Idempotent: events without a `request_id` are
/// untouched.
fn normalize_request_ids(events: &mut [AuditEvent]) {
    let mut map: HashMap<String, String> = HashMap::new();
    let mut next: usize = 1;
    for event in events.iter_mut() {
        if let Some(rid) = event.request_id.clone() {
            let normalized = map
                .entry(rid)
                .or_insert_with(|| {
                    let n = format!("req-{next}");
                    next += 1;
                    n
                })
                .clone();
            event.request_id = Some(normalized);
        }
    }
}

fn filter_prefixes(events: Vec<AuditEvent>, prefixes: &[String]) -> Vec<AuditEvent> {
    if prefixes.is_empty() {
        return events;
    }
    events
        .into_iter()
        .filter(|e| !prefixes.iter().any(|p| e.event.starts_with(p)))
        .collect()
}

fn compare_sequence(left: &[AuditEvent], right: &[AuditEvent]) -> bool {
    let l_sigs: Vec<_> = left.iter().map(|e| e.signature()).collect();
    let r_sigs: Vec<_> = right.iter().map(|e| e.signature()).collect();
    if l_sigs == r_sigs {
        return true;
    }
    eprintln!("\n=== audit-trace divergence (sequence) ===");
    eprintln!("left.len={}  right.len={}", left.len(), right.len());
    let max = left.len().max(right.len());
    for i in 0..max {
        let l = left.get(i).map(format_signature);
        let r = right.get(i).map(format_signature);
        let marker = if l == r { " " } else { "*" };
        eprintln!(
            "{marker} [{i:04}] L={:<60}  R={}",
            l.unwrap_or_else(|| "<missing>".to_string()),
            r.unwrap_or_else(|| "<missing>".to_string())
        );
    }
    false
}

fn compare_multiset(left: &[AuditEvent], right: &[AuditEvent]) -> bool {
    let l_counts = signature_counts(left);
    let r_counts = signature_counts(right);
    if l_counts == r_counts {
        return true;
    }
    eprintln!("\n=== audit-trace divergence (multiset) ===");
    let mut all_keys: Vec<_> = l_counts.keys().chain(r_counts.keys()).cloned().collect();
    all_keys.sort();
    all_keys.dedup();
    for key in all_keys {
        let l = l_counts.get(&key).copied().unwrap_or(0);
        let r = r_counts.get(&key).copied().unwrap_or(0);
        let marker = if l == r { " " } else { "*" };
        eprintln!(
            "{marker} L={l:>4}  R={r:>4}  {} role={} rid={}",
            key.0,
            key.1.as_deref().unwrap_or("-"),
            key.2.as_deref().unwrap_or("-")
        );
    }
    false
}

fn signature_counts(
    events: &[AuditEvent],
) -> HashMap<(String, Option<String>, Option<String>), usize> {
    let mut out = HashMap::new();
    for e in events {
        *out.entry(e.signature()).or_insert(0) += 1;
    }
    out
}

fn format_signature(e: &AuditEvent) -> String {
    format!(
        "{} role={} rid={}",
        e.event,
        e.role.as_deref().unwrap_or("-"),
        e.request_id.as_deref().unwrap_or("-")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_csi_escapes() {
        let raw = "\x1b[2m2026-05-01T12:34:56\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mkvbm_audit\x1b[0m\x1b[2m:\x1b[0m \x1b[3mevent\x1b[0m\x1b[2m=\x1b[0m\"gnmt_entry\"";
        let cleaned = strip_ansi_escapes(raw);
        assert!(!cleaned.contains('\x1b'), "ESC remained: {cleaned:?}");
        assert!(cleaned.contains("kvbm_audit:"));
        assert!(cleaned.contains("event=\"gnmt_entry\""));
    }

    #[test]
    fn parses_audit_line_with_ansi_codes() {
        let raw = "\x1b[2m2026-05-01T12:34:56\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mkvbm_audit\x1b[0m\x1b[2m:\x1b[0m \x1b[3mevent\x1b[0m\x1b[2m=\x1b[0m\"gnmt_entry\" \x1b[3mrole\x1b[0m\x1b[2m=\x1b[0m\"decode\" \x1b[3mrequest_id\x1b[0m\x1b[2m=\x1b[0m\"req-1\"";
        let event = parse_audit_line(raw).expect("parsed");
        assert_eq!(event.event, "gnmt_entry");
        assert_eq!(event.role.as_deref(), Some("decode"));
        assert_eq!(event.request_id.as_deref(), Some("req-1"));
    }

    #[test]
    fn parses_basic_audit_line() {
        let line = "2026-05-01T12:34:56.789Z  INFO kvbm_audit: \
                    event=\"gnmt_entry\" role=\"decode\" request_id=\"req-1\" \
                    num_computed_tokens=0";
        let event = parse_audit_line(line).expect("parsed");
        assert_eq!(event.event, "gnmt_entry");
        assert_eq!(event.role.as_deref(), Some("decode"));
        assert_eq!(event.request_id.as_deref(), Some("req-1"));
        assert_eq!(
            event.fields,
            vec![("num_computed_tokens".into(), "0".into())]
        );
    }

    #[test]
    fn ignores_non_audit_lines() {
        let line = "2026-05-01T12:34:56.789Z  INFO some::other::module: hello world";
        assert!(parse_audit_line(line).is_none());
    }

    #[test]
    fn handles_bare_unquoted_values() {
        let line = "INFO kvbm_audit: event=test role=decode count=5 ok=true";
        let event = parse_audit_line(line).expect("parsed");
        assert_eq!(event.event, "test");
        assert_eq!(event.role.as_deref(), Some("decode"));
        let fields: HashMap<_, _> = event.fields.into_iter().collect();
        assert_eq!(fields.get("count").map(String::as_str), Some("5"));
        assert_eq!(fields.get("ok").map(String::as_str), Some("true"));
    }

    #[test]
    fn quoted_values_with_spaces_preserved() {
        let line = "INFO kvbm_audit: event=test reason=\"hello world\" k=v";
        let event = parse_audit_line(line).expect("parsed");
        let fields: HashMap<_, _> = event.fields.into_iter().collect();
        assert_eq!(
            fields.get("reason").map(String::as_str),
            Some("hello world")
        );
        assert_eq!(fields.get("k").map(String::as_str), Some("v"));
    }

    #[test]
    fn missing_event_field_returns_none() {
        let line = "INFO kvbm_audit: role=\"decode\" request_id=\"r\"";
        assert!(parse_audit_line(line).is_none());
    }

    #[test]
    fn sequence_compare_detects_missing() {
        let l = vec![event_sig("a", "decode", "r")];
        let r = vec![];
        assert!(!compare_sequence(&l, &r));
    }

    #[test]
    fn sequence_compare_matches_identical() {
        let a = event_sig("a", "decode", "r");
        let l = vec![a.clone()];
        let r = vec![a];
        assert!(compare_sequence(&l, &r));
    }

    #[test]
    fn multiset_compare_ignores_order() {
        let a = event_sig("a", "decode", "r");
        let b = event_sig("b", "prefill", "r");
        let l = vec![a.clone(), b.clone()];
        let r = vec![b, a];
        assert!(!compare_sequence(&l, &r));
        assert!(compare_multiset(&l, &r));
    }

    #[test]
    fn normalize_request_ids_assigns_stable_counters() {
        let mut events = vec![
            event_sig("a", "decode", "vllm-uuid-aaa"),
            event_sig("b", "decode", "vllm-uuid-bbb"),
            event_sig("c", "decode", "vllm-uuid-aaa"),
        ];
        normalize_request_ids(&mut events);
        assert_eq!(events[0].request_id.as_deref(), Some("req-1"));
        assert_eq!(events[1].request_id.as_deref(), Some("req-2"));
        assert_eq!(events[2].request_id.as_deref(), Some("req-1"));
    }

    #[test]
    fn normalize_request_ids_skips_events_without_rid() {
        let mut events = vec![AuditEvent {
            event: "tick".into(),
            role: Some("decode".into()),
            request_id: None,
            fields: vec![],
        }];
        normalize_request_ids(&mut events);
        assert_eq!(events[0].request_id, None);
    }

    #[test]
    fn filter_prefixes_drops_matching_events() {
        let l = vec![
            event_sig("foo_a", "decode", "r"),
            event_sig("bar", "decode", "r"),
            event_sig("foo_b", "decode", "r"),
        ];
        let filtered = filter_prefixes(l, &["foo_".to_string()]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].event, "bar");
    }

    fn event_sig(event: &str, role: &str, rid: &str) -> AuditEvent {
        AuditEvent {
            event: event.to_string(),
            role: Some(role.to_string()),
            request_id: Some(rid.to_string()),
            fields: vec![],
        }
    }
}
