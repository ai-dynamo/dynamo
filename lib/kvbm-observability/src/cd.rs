// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the conditional-disaggregation (CD) path.
//!
//! These are the durable, always-on companions to the `kvbm_audit` tracing
//! events on the CD path. The audit channel is smoke-test instrumentation
//! (verbose, log-level dependent, ~hundreds of MB/run) and is meant to be
//! quieted for long benchmarks; these counters/histograms answer the same
//! "how many local vs remote, how big, how much computed" questions from the
//! /metrics endpoint regardless of log level.
//!
//! TOKEN-QUANTITY CORRECTNESS (the USAA-1 trap): the CD path has four
//! confusable token quantities. The metrics below read the EXACT variable:
//! - local prefill tokens  = `num_prefill_tokens()` = total − num_computed − local_match
//! - remote prefill tokens = `split.remote_blocks() * block_size` (NOT
//!   `full_block_external_tokens`, which over-counts by the local-match; NOT
//!   the wire `num_prefill_tokens` field, which includes the prefix-cache hit).
//! - prefill-side computed = `expected_outputs.len() * block_size`, which equals
//!   `remote_blocks()` by construction ⇒ the decode `remote_prefill_tokens` and
//!   the prefill `prefill_computed_tokens` reconcile 1:1.

use prometheus::{Histogram, HistogramOpts, IntCounter, IntCounterVec, Opts, Registry};

/// Token-count histogram buckets (prompt-scale; Qwen3 native ctx ~40960).
fn token_buckets() -> Vec<f64> {
    vec![
        256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0,
    ]
}

/// Conditional-disaggregation metrics. Constructed once per engine process
/// (decode AND prefill share the struct); each process only increments the
/// fields relevant to its role, so the non-applicable fields stay at zero.
#[derive(Clone)]
pub struct CdMetrics {
    // --- decode side (Q1/Q2/Q3/Q4) ---
    /// CD prefill routing decisions, by outcome. label `decision` ∈
    /// {local, remote, remote_downgraded_zero_block, remote_downgraded_overload,
    /// remote_rejected_budget}. Q1 = decision="local"; Q2 = decision="remote";
    /// `remote_downgraded_overload` = a Remote decision the decode downgraded to
    /// a local prefill because the inflight budget (hub-prefill-pressure proxy)
    /// was exhausted (Approach B-GNMT).
    pub prefill_decisions_total: IntCounterVec,
    /// Q3: tokens the decode will itself prefill on a Local decision
    /// (= num_prefill_tokens() = total − num_computed − local_match).
    pub local_prefill_tokens_total: IntCounter,
    /// Q4: tokens the decode expects the REMOTE to compute
    /// (= split.remote_blocks() * block_size), summed.
    pub remote_prefill_tokens_total: IntCounter,
    /// Q2 distribution: per-request remote-compute remainder size.
    pub remote_prefill_tokens: Histogram,
    /// Interesting: the external WINDOW decode reserves/ships
    /// (= full_block_external_tokens = remote + local-match, block-floored).
    pub remote_prefill_window_tokens_total: IntCounter,
    /// Interesting: vLLM prefix-cache hit size on CD requests (base_offset),
    /// surfaced explicitly so it is never folded into the remote count.
    pub prefix_cache_hit_tokens: Histogram,
    /// Interesting: Remote decisions that did not enqueue, by reason
    /// {zero_block, budget_exhausted}. (Both were 0 in job 2180509.)
    pub remote_prefill_declined_total: IntCounterVec,

    // --- prefill side (Q6 + reconciliation) ---
    /// Q6: tokens the prefill worker is actually asked to compute
    /// (= expected_outputs.len() * block_size), summed.
    pub prefill_computed_tokens_total: IntCounter,
    /// Q6 distribution: per-request prefill forward-pass size.
    pub prefill_computed_tokens: Histogram,
    /// Interesting: tokens the prefill worker PULLED from decode (the onboarded
    /// prefix it did NOT compute) — the deliberate complement of Q6.
    pub prefill_pulled_tokens_total: IntCounter,
    /// Reconciliation: per-request finalize outcome of the expected_outputs set,
    /// by `outcome` ∈ {drained, undrained}. `undrained` = prefill produced fewer
    /// net-new blocks than decode expected — the within-prefill divergence signal
    /// that would have surfaced the USAA-1 class as a metric, not a crash.
    pub prefill_output_residual_total: IntCounterVec,
}

impl CdMetrics {
    pub fn new() -> Self {
        Self {
            prefill_decisions_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_cd_prefill_decisions_total",
                    "CD prefill routing decisions by outcome (local|remote|remote_downgraded_zero_block|remote_downgraded_overload|remote_rejected_budget)",
                ),
                &["decision"],
            )
            .expect("valid metric"),
            local_prefill_tokens_total: IntCounter::with_opts(Opts::new(
                "kvbm_cd_local_prefill_tokens_total",
                "Tokens the decode prefills locally (uncached, unmatched) on Local CD decisions",
            ))
            .expect("valid metric"),
            remote_prefill_tokens_total: IntCounter::with_opts(Opts::new(
                "kvbm_cd_remote_prefill_tokens_total",
                "Tokens the decode expects the remote prefill to compute (remote_blocks*block_size)",
            ))
            .expect("valid metric"),
            remote_prefill_tokens: Histogram::with_opts(
                HistogramOpts::new(
                    "kvbm_cd_remote_prefill_tokens",
                    "Distribution of per-request remote-prefill compute size (tokens)",
                )
                .buckets(token_buckets()),
            )
            .expect("valid metric"),
            remote_prefill_window_tokens_total: IntCounter::with_opts(Opts::new(
                "kvbm_cd_remote_prefill_window_tokens_total",
                "External window the decode reserves/ships for remote prefill (full_block_external_tokens)",
            ))
            .expect("valid metric"),
            prefix_cache_hit_tokens: Histogram::with_opts(
                HistogramOpts::new(
                    "kvbm_cd_prefix_cache_hit_tokens",
                    "Distribution of vLLM prefix-cache hit size (base_offset) on CD requests",
                )
                .buckets(token_buckets()),
            )
            .expect("valid metric"),
            remote_prefill_declined_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_cd_remote_prefill_declined_total",
                    "Remote CD decisions that did not enqueue, by reason (zero_block|budget_exhausted)",
                ),
                &["reason"],
            )
            .expect("valid metric"),
            prefill_computed_tokens_total: IntCounter::with_opts(Opts::new(
                "kvbm_cd_prefill_computed_tokens_total",
                "Tokens the prefill worker is asked to compute (expected_outputs*block_size)",
            ))
            .expect("valid metric"),
            prefill_computed_tokens: Histogram::with_opts(
                HistogramOpts::new(
                    "kvbm_cd_prefill_computed_tokens",
                    "Distribution of per-request prefill forward-pass size (tokens)",
                )
                .buckets(token_buckets()),
            )
            .expect("valid metric"),
            prefill_pulled_tokens_total: IntCounter::with_opts(Opts::new(
                "kvbm_cd_prefill_pulled_tokens_total",
                "Tokens the prefill worker pulled from decode (onboarded prefix, not computed)",
            ))
            .expect("valid metric"),
            prefill_output_residual_total: IntCounterVec::new(
                Opts::new(
                    "kvbm_cd_prefill_output_residual_total",
                    "Prefill finalize outcome of the expected-outputs set, by outcome (drained|undrained)",
                ),
                &["outcome"],
            )
            .expect("valid metric"),
        }
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.prefill_decisions_total.clone()))?;
        registry.register(Box::new(self.local_prefill_tokens_total.clone()))?;
        registry.register(Box::new(self.remote_prefill_tokens_total.clone()))?;
        registry.register(Box::new(self.remote_prefill_tokens.clone()))?;
        registry.register(Box::new(self.remote_prefill_window_tokens_total.clone()))?;
        registry.register(Box::new(self.prefix_cache_hit_tokens.clone()))?;
        registry.register(Box::new(self.remote_prefill_declined_total.clone()))?;
        registry.register(Box::new(self.prefill_computed_tokens_total.clone()))?;
        registry.register(Box::new(self.prefill_computed_tokens.clone()))?;
        registry.register(Box::new(self.prefill_pulled_tokens_total.clone()))?;
        registry.register(Box::new(self.prefill_output_residual_total.clone()))?;
        Ok(())
    }

    // --- decode-side recorders ------------------------------------------------

    /// Record one CD decision. `decision` ∈ {local, remote,
    /// remote_downgraded_zero_block, remote_downgraded_overload,
    /// remote_rejected_budget}.
    pub fn record_decision(&self, decision: &'static str) {
        self.prefill_decisions_total
            .with_label_values(&[decision])
            .inc();
    }

    /// Q3: tokens the decode will prefill locally (Local decision).
    pub fn record_local_prefill_tokens(&self, tokens: u64) {
        self.local_prefill_tokens_total.inc_by(tokens);
    }

    /// Q4 + Q2-dist: tokens the decode expects the remote to compute
    /// (`remote_blocks * block_size`). Increments the sum AND observes the
    /// distribution. Call exactly once per remote request (guard recompute).
    pub fn record_remote_prefill_tokens(&self, tokens: u64) {
        self.remote_prefill_tokens_total.inc_by(tokens);
        self.remote_prefill_tokens.observe(tokens as f64);
    }

    /// Interesting: the reserved/shipped external window.
    pub fn record_remote_prefill_window(&self, tokens: u64) {
        self.remote_prefill_window_tokens_total.inc_by(tokens);
    }

    /// Interesting: prefix-cache hit size (base_offset) on a CD request.
    pub fn observe_prefix_cache_hit(&self, tokens: u64) {
        self.prefix_cache_hit_tokens.observe(tokens as f64);
    }

    /// Interesting: a Remote decision that did not enqueue. `reason` ∈
    /// {zero_block, budget_exhausted}.
    pub fn record_remote_declined(&self, reason: &'static str) {
        self.remote_prefill_declined_total
            .with_label_values(&[reason])
            .inc();
    }

    // --- prefill-side recorders -----------------------------------------------

    /// Q6 + dist: tokens the prefill worker is asked to compute. Call once per
    /// remote-prefill request the worker accepts.
    pub fn record_prefill_computed_tokens(&self, tokens: u64) {
        self.prefill_computed_tokens_total.inc_by(tokens);
        self.prefill_computed_tokens.observe(tokens as f64);
    }

    /// Interesting: tokens pulled from decode (onboarded prefix, not computed).
    pub fn record_prefill_pulled_tokens(&self, tokens: u64) {
        self.prefill_pulled_tokens_total.inc_by(tokens);
    }

    /// Reconciliation: prefill finalize residual outcome. `outcome` ∈
    /// {drained, undrained}.
    pub fn record_prefill_output_residual(&self, outcome: &'static str) {
        self.prefill_output_residual_total
            .with_label_values(&[outcome])
            .inc();
    }
}

impl Default for CdMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cd_metric_names_registered() {
        let cd = CdMetrics::new();
        let registry = Registry::new();
        cd.register(&registry).unwrap();

        // Labelled *Vec metrics only appear in gather() once a label set is used,
        // so touch each (mirrors real usage where they are always recorded).
        cd.record_decision("local");
        cd.record_remote_declined("zero_block");
        cd.record_prefill_output_residual("drained");

        let names: Vec<_> = registry
            .gather()
            .iter()
            .map(|mf| mf.name().to_string())
            .collect();

        for expected in [
            "kvbm_cd_prefill_decisions_total",
            "kvbm_cd_local_prefill_tokens_total",
            "kvbm_cd_remote_prefill_tokens_total",
            "kvbm_cd_remote_prefill_tokens",
            "kvbm_cd_prefill_computed_tokens_total",
            "kvbm_cd_prefill_output_residual_total",
        ] {
            assert!(names.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn test_decision_counter_labels_independent() {
        let cd = CdMetrics::new();
        cd.record_decision("local");
        cd.record_decision("remote");
        cd.record_decision("remote");
        cd.record_decision("remote_downgraded_overload");
        assert_eq!(cd.prefill_decisions_total.with_label_values(&["local"]).get(), 1);
        assert_eq!(cd.prefill_decisions_total.with_label_values(&["remote"]).get(), 2);
        assert_eq!(
            cd.prefill_decisions_total
                .with_label_values(&["remote_downgraded_overload"])
                .get(),
            1
        );
    }

    #[test]
    fn test_remote_tokens_sum_and_histogram_move_together() {
        let cd = CdMetrics::new();
        cd.record_remote_prefill_tokens(512);
        cd.record_remote_prefill_tokens(1024);
        assert_eq!(cd.remote_prefill_tokens_total.get(), 1536);
        assert_eq!(cd.remote_prefill_tokens.get_sample_count(), 2);
        assert_eq!(cd.remote_prefill_tokens.get_sample_sum() as u64, 1536);
    }

    #[test]
    fn test_decode_q4_reconciles_with_prefill_q6() {
        // remote_blocks()*bs (decode Q4) == expected_outputs*bs (prefill Q6) by
        // construction; the two counter sums must match for a request set.
        let cd = CdMetrics::new();
        for n in [512u64, 1024, 5376] {
            cd.record_remote_prefill_tokens(n); // decode side
            cd.record_prefill_computed_tokens(n); // prefill side
        }
        assert_eq!(
            cd.remote_prefill_tokens_total.get(),
            cd.prefill_computed_tokens_total.get()
        );
    }
}
