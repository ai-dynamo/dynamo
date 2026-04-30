// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbm_audit` tracing target — request-level audit events.
//!
//! Stage A of the CD audit work: cheap, throwaway-friendly instrumentation
//! that we can post-process into a two-column ordered timeline showing
//! exactly which connector-API hooks fire on each side of a CD request,
//! and what decisions the wrappers take at each one.
//!
//! Conventions:
//! - Every event uses `target: "kvbm_audit"` so it can be filtered at
//!   subscribe time (`RUST_LOG=kvbm_audit=info`) and grepped out of
//!   mixed logs.
//! - Every event carries `event` (stable name), `request_id`, and
//!   `role` ("prefill" | "decode" | "both"). Additional fields are
//!   per-event.
//! - Use the [`audit!`] macro for terseness and to keep field shape
//!   consistent.
//!
//! Once Stage A finishes the diagnostic work, this can either evolve
//! into a hub-side feature (request-id-keyed ring buffer pushed via
//! velo) or be deleted. Keep it self-contained so removal is easy.

/// Emit a `kvbm_audit` tracing event.
///
/// ```ignore
/// audit!("gnmt_entry", role = "decode", request_id, num_computed_tokens);
/// audit!("gnmt_exit", role = "decode", request_id, decision = "remote", n = full_block_external_tokens);
/// ```
#[macro_export]
macro_rules! audit {
    ($event:expr, $($field:tt)*) => {
        tracing::info!(
            target: "kvbm_audit",
            event = $event,
            $($field)*
        )
    };
}

/// Emit a `build_meta_entry` audit event with detailed scheduler
/// state. Throttled to every 20th iteration when no requests are
/// scheduled, since vLLM polls this hook on every scheduler tick.
///
/// Captures, in addition to the iteration / new+cached counts:
/// - total scheduled tokens this iteration
/// - per-request scheduled tokens (`req_id -> count` summary)
/// - per-request num_computed_tokens (from new + cached request data)
pub fn audit_build_meta(role: &'static str, output: &crate::common::SchedulerOutput) {
    let n_new = output.scheduled_new_reqs.len();
    let n_cached = output.scheduled_cached_reqs.len();
    let n_total = n_new + n_cached;
    if n_total == 0 && output.iteration % 20 != 0 {
        return;
    }

    // Compact JSON-ish summary: rid → scheduled tokens.
    let scheduled = if output.num_scheduled_tokens.is_empty() {
        String::from("{}")
    } else {
        let mut entries: Vec<(&String, &usize)> = output.num_scheduled_tokens.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let body: Vec<String> = entries
            .into_iter()
            .map(|(rid, n)| format!("{rid}:{n}"))
            .collect();
        format!("{{{}}}", body.join(","))
    };

    // num_computed_tokens per request, pulled from new and cached
    // entries the scheduler is dispatching this iteration.
    let mut per_req_computed: Vec<String> = Vec::new();
    for new_req in &output.scheduled_new_reqs {
        per_req_computed.push(format!(
            "{}:new:{}",
            new_req.req_id, new_req.num_computed_tokens
        ));
    }
    for c in &output.scheduled_cached_reqs {
        per_req_computed.push(format!("{}:cached:{}", c.req_id, c.num_computed_tokens));
    }
    let computed_summary = if per_req_computed.is_empty() {
        String::from("{}")
    } else {
        format!("{{{}}}", per_req_computed.join(","))
    };

    crate::audit!(
        "build_meta_entry",
        role,
        iteration = output.iteration,
        num_new_requests = n_new,
        num_cached_requests = n_cached,
        total_scheduled_tokens = output.total_num_scheduled_tokens,
        scheduled = %scheduled,
        per_req_computed = %computed_summary
    );
}
