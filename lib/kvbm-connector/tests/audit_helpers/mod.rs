// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Audit-event capture helpers for trace-equivalence tests.
//!
//! Installs a single process-wide `tracing` subscriber whose only layer
//! is an [`AuditCollector`] that filters on `target == "kvbm_audit"` and
//! drains matching events into an in-memory ring.  Tests use
//! [`install_collector`] to obtain a [`AuditCaptureHandle`] (idempotent
//! across multiple tests in the same binary thanks to `OnceLock`),
//! and [`AuditCaptureHandle::drain`] to take the events captured since
//! the last drain.
//!
//! Equality of two captured runs is asserted via
//! [`assert_event_signatures_equal`], which compares the `(event, role,
//! request_id)` tuple in order.  Full field maps are kept on
//! [`AuditEvent`] for diff context on failure.

#![allow(dead_code)]

use std::sync::{Arc, Mutex, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

/// One captured `kvbm_audit` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEvent {
    /// The audit event name (e.g. `"gnmt_entry"`).
    pub event: String,
    /// The `role` field if present (e.g. `"decode"` / `"prefill"`).
    pub role: Option<String>,
    /// The `request_id` field if present.
    pub request_id: Option<String>,
    /// All other fields, sorted by name, formatted via Debug.
    pub fields: Vec<(String, String)>,
}

impl AuditEvent {
    /// Tuple used for sequence-equivalence assertions.  Fields beyond
    /// the signature are inspected only on failure (via the `fields`
    /// vector) since they often carry test-run-unique values like
    /// session UUIDs that intentionally differ between baseline and
    /// unified runs.
    pub fn signature(&self) -> (String, Option<String>, Option<String>) {
        (
            self.event.clone(),
            self.role.clone(),
            self.request_id.clone(),
        )
    }
}

/// Handle held by tests to drain captured events.
#[derive(Clone)]
pub struct AuditCaptureHandle {
    events: Arc<Mutex<Vec<AuditEvent>>>,
}

impl AuditCaptureHandle {
    /// Take all events captured since the last drain.
    pub fn drain(&self) -> Vec<AuditEvent> {
        let mut guard = self.events.lock().unwrap();
        std::mem::take(&mut *guard)
    }

    /// Snapshot without draining (rarely needed; `drain` is the
    /// canonical workflow because tests run baseline → unified
    /// back-to-back and want clean separation).
    pub fn snapshot(&self) -> Vec<AuditEvent> {
        self.events.lock().unwrap().clone()
    }
}

/// Process-wide capture handle.  Installed exactly once per test
/// binary by [`install_collector`].
static GLOBAL_HANDLE: OnceLock<AuditCaptureHandle> = OnceLock::new();

/// Process-wide serialization mutex for tests that compare captured
/// audit streams.  Cargo runs tests in parallel by default, but the
/// capture handle is a single shared buffer — concurrent tests would
/// see each other's events.  Each test acquires this mutex at the
/// start of its body.  Returned guard's `Drop` releases the mutex
/// even on panic (parking_lot does not poison).
static AUDIT_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Acquire the audit-capture serialization lock.  Hold the returned
/// guard for the duration of the test body.  The guard is dropped
/// either at end of scope or on panic, releasing the lock.
pub fn audit_test_lock() -> parking_lot::MutexGuard<'static, ()> {
    AUDIT_TEST_LOCK.lock()
}

/// Install (idempotently) a process-wide subscriber that captures
/// `kvbm_audit` events.  Returns a clone of the global handle.
///
/// The first call sets `tracing`'s global default subscriber.
/// Subsequent calls return the same handle without re-installing.
/// Multiple tests in the same binary share one capture buffer; tests
/// must drain between phases to keep their assertions clean.
pub fn install_collector() -> AuditCaptureHandle {
    GLOBAL_HANDLE
        .get_or_init(|| {
            let events: Arc<Mutex<Vec<AuditEvent>>> = Arc::new(Mutex::new(Vec::new()));
            let layer = AuditCollector {
                events: Arc::clone(&events),
            };
            tracing_subscriber::registry().with(layer).init();
            AuditCaptureHandle { events }
        })
        .clone()
}

struct AuditCollector {
    events: Arc<Mutex<Vec<AuditEvent>>>,
}

impl<S: Subscriber> Layer<S> for AuditCollector {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != "kvbm_audit" {
            return;
        }
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        if let Some(captured) = visitor.into_audit_event() {
            self.events.lock().unwrap().push(captured);
        }
    }
}

#[derive(Default)]
struct FieldVisitor {
    event: Option<String>,
    role: Option<String>,
    request_id: Option<String>,
    others: Vec<(String, String)>,
}

impl FieldVisitor {
    fn into_audit_event(mut self) -> Option<AuditEvent> {
        let event = self.event.take()?;
        self.others.sort_by(|a, b| a.0.cmp(&b.0));
        Some(AuditEvent {
            event,
            role: self.role,
            request_id: self.request_id,
            fields: self.others,
        })
    }

    fn record(&mut self, name: &str, value: String) {
        match name {
            "event" => self.event = Some(value),
            "role" => self.role = Some(value),
            "request_id" => self.request_id = Some(value),
            other => self.others.push((other.to_string(), value)),
        }
    }
}

impl Visit for FieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record(field.name(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record(field.name(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record(field.name(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record(field.name(), value.to_string());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record(field.name(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // Strip the surrounding quotes that tracing's `%`/`?` debug
        // formatting tends to add for string-valued fields, so
        // "decode" matches against the role string equivalently
        // whether emitted via `record_str` or `record_debug`.
        let raw = format!("{value:?}");
        let cleaned = strip_outer_quotes(&raw);
        self.record(field.name(), cleaned.to_string());
    }
}

fn strip_outer_quotes(s: &str) -> &str {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Assert that two captured event sequences match on the
/// `(event, role, request_id)` signature, in order.
///
/// On failure prints both sequences with a `*` marker on the first
/// divergence so you can locate it quickly in the test output.
pub fn assert_event_signatures_equal(left: &[AuditEvent], right: &[AuditEvent], context: &str) {
    let l_sigs: Vec<_> = left.iter().map(|e| e.signature()).collect();
    let r_sigs: Vec<_> = right.iter().map(|e| e.signature()).collect();

    if l_sigs == r_sigs {
        return;
    }

    let mut msg = format!("\n=== audit-trace divergence: {context} ===\n");
    msg.push_str(&format!(
        "left.len={}  right.len={}\n",
        left.len(),
        right.len()
    ));
    let max = left.len().max(right.len());
    for i in 0..max {
        let l = left.get(i).map(|e| format_signature(e));
        let r = right.get(i).map(|e| format_signature(e));
        let marker = if l == r { " " } else { "*" };
        msg.push_str(&format!(
            "{marker} [{i:03}] L={:<60}  R={}\n",
            l.unwrap_or_else(|| "<missing>".to_string()),
            r.unwrap_or_else(|| "<missing>".to_string())
        ));
    }
    panic!("{msg}");
}

fn format_signature(e: &AuditEvent) -> String {
    format!(
        "{} role={} rid={}",
        e.event,
        e.role.as_deref().unwrap_or("-"),
        e.request_id.as_deref().unwrap_or("-")
    )
}

/// Filter helper: drop events whose `event` name matches any of the
/// provided prefixes.  Used to whitelist unified-only audit events
/// when comparing against bare-leader baselines.
pub fn filter_out_event_prefixes(events: Vec<AuditEvent>, prefixes: &[&str]) -> Vec<AuditEvent> {
    events
        .into_iter()
        .filter(|e| !prefixes.iter().any(|p| e.event.starts_with(p)))
        .collect()
}
