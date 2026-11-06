// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-level reconciliation of native backend facts with shared workload records.

use std::collections::{BTreeMap, BTreeSet};

use aisim_aiperf_workload_runner::participant_adapter::HostedTypedReport;
use aisimulate_core::replay::{ReplayReport, ReplayTerminalStatus};
use anyhow::{Context, Result, ensure};

/// Populations actually reconciled, retained with full-scenario artifacts.
#[derive(Debug, serde::Serialize)]
pub(crate) struct Reconciliation {
    pub(crate) backend_requests: usize,
    pub(crate) native_records: usize,
    pub(crate) dispatched_requests: usize,
    pub(crate) time_comparisons: usize,
    pub(crate) output_count_comparisons: usize,
}

/// Backend milliseconds and workload integer nanoseconds have different
/// representations. Require agreement within one nanosecond, the adapter's
/// documented ceiling conversion, rather than inventing float-bit equality.
fn same_time(ns: i64, ms: f64, uuid: &str, field: &str) -> Result<()> {
    ensure!(ms.is_finite(), "{uuid}: backend {field} is not finite");
    let difference = (ns as f64 - ms * 1_000_000.0).abs();
    ensure!(
        difference <= 1.0,
        "{uuid}: {field} differs: native={ns}ns backend={ms}ms difference={difference}ns"
    );
    Ok(())
}

/// Every physical occurrence must have one shared-workload metric record and
/// one dispatch identity. Compare delivered token population, terminal class,
/// and client-visible timestamps without filtering to successful requests.
pub(crate) fn reconcile(
    backend: &ReplayReport,
    typed: &HostedTypedReport,
) -> Result<Reconciliation> {
    ensure!(
        !backend.per_request.is_empty(),
        "backend request evidence is empty"
    );
    let mut physical = BTreeMap::new();
    for row in &backend.per_request {
        ensure!(
            physical.insert(row.uuid.as_str(), row).is_none(),
            "duplicate backend UUID {}",
            row.uuid
        );
    }
    let mut dispatch = BTreeSet::new();
    for row in &typed.dispatch.requests {
        let uuid = row["request"]["uuid"]
            .as_str()
            .context("dispatch UUID missing")?;
        ensure!(dispatch.insert(uuid), "duplicate dispatch UUID {uuid}");
    }
    ensure!(
        dispatch.len() == typed.dispatch.submitted_requests,
        "dispatch/submission population mismatch"
    );
    ensure!(
        dispatch.len() == physical.len(),
        "dispatch/backend population mismatch: {} vs {}",
        dispatch.len(),
        physical.len()
    );
    for uuid in &dispatch {
        ensure!(
            physical.contains_key(uuid),
            "dispatch UUID {uuid} absent from backend"
        );
    }

    let mut native_ids = BTreeSet::new();
    let mut time_comparisons = 0;
    let mut output_count_comparisons = 0;
    for phase in &typed.records {
        for (id, record) in &phase.records {
            let uuid = id.to_string();
            ensure!(
                native_ids.insert(uuid.clone()),
                "duplicate native UUID {uuid}"
            );
            let row = physical
                .get(uuid.as_str())
                .with_context(|| format!("native UUID {uuid} absent from backend"))?;
            same_time(record.start_ns, row.arrival_time_ms, &uuid, "start")?;
            same_time(record.end_ns, row.terminal_time_ms, &uuid, "terminal")?;
            time_comparisons += 2;
            for (label, native, modeled) in [
                ("first token", record.first_token_ns, row.first_token_ms),
                ("last token", record.last_token_ns, row.last_token_ms),
            ] {
                ensure!(
                    native.is_some() == modeled.is_some(),
                    "{uuid}: {label} presence mismatch"
                );
                if let (Some(native), Some(modeled)) = (native, modeled) {
                    same_time(native, modeled, &uuid, label)?;
                    time_comparisons += 1;
                }
            }
            ensure!(
                usize::try_from(record.token_frame_count)? == row.output_length,
                "{uuid}: delivered frame/token population mismatch: {} vs {}",
                record.token_frame_count,
                row.output_length
            );
            output_count_comparisons += 1;
            if let Some(output) = record.tokens.output_sequence_length() {
                ensure!(
                    usize::try_from(output)? == row.output_length,
                    "{uuid}: native/backend output counts differ: {output} vs {}",
                    row.output_length
                );
                output_count_comparisons += 1;
            }
            match row.terminal_status {
                ReplayTerminalStatus::Completed => ensure!(
                    !record.errored && !record.canceled,
                    "{uuid}: completed backend has native error/cancellation"
                ),
                ReplayTerminalStatus::Canceled => ensure!(
                    record.canceled,
                    "{uuid}: backend cancellation absent from native record"
                ),
                ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => ensure!(
                    record.errored,
                    "{uuid}: backend failure absent from native record"
                ),
            }
        }
    }
    ensure!(
        native_ids.len() == physical.len(),
        "native/backend population mismatch: {} vs {}",
        native_ids.len(),
        physical.len()
    );
    Ok(Reconciliation {
        backend_requests: physical.len(),
        native_records: native_ids.len(),
        dispatched_requests: dispatch.len(),
        time_comparisons,
        output_count_comparisons,
    })
}
