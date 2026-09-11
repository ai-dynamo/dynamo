// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shutdown timing: the knobs that bound each stage of [`Worker`]'s
//! graceful-shutdown sequence, and the deadline they compose into.
//!
//! Extracted from `worker.rs` so the whole timing surface is legible in one
//! place — the stage bounds only make sense relative to one another, and
//! reading them interleaved with lifecycle code hid the fact that they can
//! sum past the deadline they nominally live inside (see
//! [`shutdown_deadline`]).
//!
//! Every knob resolves through [`env_secs`], so an operator gets the same
//! parse behaviour and the same warning on every variable. What differs per
//! knob is only the *floor*, which each resolver states explicitly:
//! a zero grace period or drain budget means "skip that stage", while a zero
//! cleanup budget would cancel `engine.cleanup()` on its first poll and is
//! therefore rejected in favour of the default.
//!
//! # One budget, not a sum
//!
//! Every stage draws from the same deadline, measured once from SIGTERM. A
//! stage receives `min(its cap, remaining)`. No reserve is withheld from
//! earlier stages — withholding one zeroed the in-flight barrier under the
//! debug defaults, where grace and reserve together consumed the whole total.
//! Cleanup is funded by its own floor in `cleanup_once`, and the force-exit
//! watchdog is extended by that floor so the two cannot race.
//!
//! This is deliberately not a set of independent timers: the earlier design
//! summed to roughly 95s worst case from a knob documented as 30, which could
//! exceed a pod's `terminationGracePeriodSeconds` and get the worker SIGKILLed
//! mid-teardown.

use std::time::{Duration, Instant};

/// Default grace-period in seconds between discovery unregister and engine drain.
pub(crate) const DEFAULT_GRACE_PERIOD_SECS: f64 = 5.0;

/// Environment variable name for overriding the grace-period.
pub(crate) const GRACE_PERIOD_ENV: &str = "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS";

/// Default drain budget: max time spent polling `is_quiescent` before cleanup.
/// Capped at `graceful_shutdown_timeout - CLEANUP_RESERVE_S`.
pub(crate) const DEFAULT_DRAIN_TIMEOUT_S: f64 = 30.0;
pub(crate) const DRAIN_TIMEOUT_ENV: &str = "DYN_PREFILL_DRAIN_TIMEOUT_S";

/// Cadence at which a long-running drain emits a progress log.
pub(crate) const DRAIN_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Floor granted to `engine.cleanup()` when the total budget is already spent.
///
/// Not withheld from earlier stages — see [`ShutdownBudget::allowance`].
/// Cleanup is the one stage that must run even out of budget, because
/// abandoning it leaks GPU memory; the hard-exit deadline bounds the overrun.
pub(crate) const CLEANUP_RESERVE_S: f64 = 5.0;

/// Default cap on waiting for request-plane in-flight requests to finish.
///
/// `f64::INFINITY` means "no cap of its own" — the stage is bounded by the
/// remaining total, like [`Stage::Unregister`]. It used to default to 900s to
/// match `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`, which was a knob that
/// could never bind: `total_budget` is composed from the *worker* timeout plus
/// the grace period (35s on release defaults), and `allowance` is
/// `min(cap, remaining)`, so the documented 900s was 25x unreachable and
/// raising it changed nothing. The total is the authority; this caps the stage
/// *within* it.
pub(crate) const DEFAULT_INFLIGHT_TIMEOUT_S: f64 = f64::INFINITY;
pub(crate) const INFLIGHT_TIMEOUT_ENV: &str =
    dynamo_runtime::config::environment_names::worker::DYN_WORKER_SHUTDOWN_INFLIGHT_TIMEOUT_SECS;

/// Override for the `engine.cleanup()` bound.
pub(crate) const CLEANUP_TIMEOUT_ENV: &str =
    dynamo_runtime::config::environment_names::worker::DYN_WORKER_SHUTDOWN_CLEANUP_TIMEOUT_SECS;

/// Upper bound on any configured duration, in seconds (~10 years).
///
/// `Duration::from_secs_f64` panics on a value it cannot represent, and every
/// knob here feeds one. `is_finite()` alone is not enough: `1e30` is finite
/// and still panics. A shutdown knob larger than this is a configuration
/// error, and the one moment we must not panic is while shutting down.
const MAX_CONFIGURED_SECS: f64 = 315_360_000.0;

/// Convert a configured seconds value into a `Duration`, clamping rather than
/// panicking. Non-finite and negative values become zero.
pub fn duration_from_secs(secs: f64) -> Duration {
    if !secs.is_finite() || secs <= 0.0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(secs.min(MAX_CONFIGURED_SECS))
}

/// `true` when `secs` is a usable configured duration.
pub fn is_valid_configured_secs(secs: f64) -> bool {
    secs.is_finite() && (0.0..=MAX_CONFIGURED_SECS).contains(&secs)
}

/// Read `env` as a seconds value.
///
/// `None` means "unset, empty, or unusable" — the caller substitutes its own
/// default. Non-finite values are rejected alongside parse failures because
/// `Duration::from_secs_f64` panics on them. Callers apply their own floor;
/// this deliberately does not clamp, so that a knob for which zero is
/// meaningful and a knob for which zero is harmful can share one parser.
fn env_secs(env: &str) -> Option<f64> {
    let raw = std::env::var(env).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    // Trimmed: the empty check above already trims, and Rust's `f64` grammar
    // admits no surrounding whitespace, so a trailing newline from a ConfigMap
    // or a Helm block scalar silently fell through to the default. Python's
    // `float()` strips, so the same variable resolved differently in the Rust
    // and Python workers of one deployment.
    match raw.trim().parse::<f64>() {
        // Negatives are in range here on purpose: each knob applies its own
        // floor, and "negative means skip this stage" is meaningful for some.
        Ok(v) if v.is_finite() && v <= MAX_CONFIGURED_SECS => Some(v),
        _ => {
            tracing::warn!(
                env = env,
                value = raw,
                max_secs = MAX_CONFIGURED_SECS,
                "invalid or out-of-range duration; using default"
            );
            None
        }
    }
}

/// Grace period between discovery unregister and drain. Negative clamps to
/// zero: skipping the sleep is a coherent request.
pub(crate) fn grace_period_secs() -> f64 {
    env_secs(GRACE_PERIOD_ENV)
        .unwrap_or(DEFAULT_GRACE_PERIOD_SECS)
        .max(0.0)
}

/// Prefill KV-transfer drain budget. Negative clamps to zero: skipping the
/// drain is a coherent request.
pub(crate) fn drain_timeout_secs() -> f64 {
    env_secs(DRAIN_TIMEOUT_ENV)
        .unwrap_or(DEFAULT_DRAIN_TIMEOUT_S)
        .max(0.0)
}

/// Bound for `engine.cleanup()`, defaulting to the post-signal deadline.
///
/// Unlike the stage budgets above, a non-positive value is rejected rather
/// than clamped: `timeout(ZERO, fut)` polls once and cancels, which would
/// abandon every cleanup immediately — strictly worse than the unbounded
/// behaviour this replaced.
pub(crate) fn cleanup_timeout() -> Duration {
    let default = graceful_shutdown_timeout();
    match env_secs(CLEANUP_TIMEOUT_ENV) {
        Some(v) if v > 0.0 => duration_from_secs(v),
        Some(v) => {
            tracing::warn!(
                "Non-positive {}={}; using {}s (a zero budget would cancel cleanup immediately)",
                CLEANUP_TIMEOUT_ENV,
                v,
                default.as_secs()
            );
            default
        }
        None => default,
    }
}

/// Post-signal shutdown deadline. Delegates to `dynamo_runtime::worker` so
/// this crate and `Worker::execute` — which `run.rs` deliberately bypasses —
/// can never disagree about what the operator configured.
pub(crate) fn graceful_shutdown_timeout() -> Duration {
    dynamo_runtime::worker::graceful_shutdown_timeout()
}

/// What to do when a prefill engine cannot report KV-transfer state.
///
/// The point of naming this is that "engine has no introspection" and "engine
/// says it is still busy" must not look the same from the outside. An
/// undeclared engine still waits — that is the safe default — but says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvTransferFallback {
    /// The engine has not declared a policy. Treated as
    /// [`WaitFullBudget`](Self::WaitFullBudget), with a warning naming the
    /// stage, so a silent fixed delay cannot be mistaken for a real drain.
    Undeclared,
    /// Wait the whole KV stage budget before releasing GPU memory. Correct for
    /// an engine whose transfers a decode peer may still be pulling.
    WaitFullBudget,
    /// Skip the stage. Only for engines that genuinely hold no KV a peer could
    /// still read.
    Skip,
}

impl KvTransferFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            KvTransferFallback::Undeclared => "undeclared",
            KvTransferFallback::WaitFullBudget => "wait",
            KvTransferFallback::Skip => "skip",
        }
    }
}

pub(crate) const KV_FALLBACK_ENV: &str =
    dynamo_runtime::config::environment_names::worker::DYN_WORKER_SHUTDOWN_KV_TRANSFER_FALLBACK;

/// Operator override for the fallback, or `None` when unset/invalid.
///
/// Takes precedence over the engine's declaration: an operator who knows their
/// deployment can overrule a conservative engine default.
pub(crate) fn kv_transfer_fallback_override() -> Option<KvTransferFallback> {
    let raw = std::env::var(KV_FALLBACK_ENV).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => None,
        "wait" => Some(KvTransferFallback::WaitFullBudget),
        "skip" => Some(KvTransferFallback::Skip),
        other => {
            tracing::warn!(
                env = KV_FALLBACK_ENV,
                value = other,
                "invalid KV-transfer fallback; expected 'wait' or 'skip'. Using the engine's policy."
            );
            None
        }
    }
}

/// Request-plane in-flight drain budget. Negative clamps to zero: skipping the
/// wait is a coherent request.
pub(crate) fn inflight_timeout_secs() -> f64 {
    env_secs(INFLIGHT_TIMEOUT_ENV)
        .unwrap_or(DEFAULT_INFLIGHT_TIMEOUT_S)
        .max(0.0)
}

/// Programmatic overrides for the shutdown knobs.
///
/// Nested on `WorkerConfig` rather than flattened, following `RuntimeConfig`:
/// a flat field per knob would have to be respelled in the Rust struct, the
/// PyO3 signature and the Python dataclass, kept in sync by nothing.
///
/// Every field is `None` by default, meaning "use the environment, then the
/// built-in default". A set field wins over the environment: a caller
/// embedding a worker should not have to mutate process-global state to
/// configure it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShutdownConfig {
    /// Total SIGTERM-to-exit budget. Overrides
    /// `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` + router grace.
    pub total_secs: Option<f64>,
    /// Time to keep serving after the discovery unregister.
    pub router_grace_secs: Option<f64>,
    /// Cap on waiting for admitted requests to finish.
    pub inflight_timeout_secs: Option<f64>,
    /// Cap on waiting for prefill KV transfers.
    pub kv_transfer_timeout_secs: Option<f64>,
    /// Cap on `engine.cleanup()`.
    pub cleanup_timeout_secs: Option<f64>,
    /// Policy when a prefill engine cannot report KV-transfer state.
    pub kv_transfer_fallback: Option<KvTransferFallback>,
}

/// Per-stage caps, resolved once when the budget is armed so a stage cannot
/// observe a different value than the one the deadline was computed against.
#[derive(Clone, Copy, Debug)]
struct StageMaxima {
    router_grace: Duration,
    inflight: Duration,
    kv_transfer: Duration,
    cleanup: Duration,
}

impl StageMaxima {
    fn resolve(config: &ShutdownConfig) -> Self {
        let secs = |explicit: Option<f64>, from_env: fn() -> f64| {
            duration_from_secs(explicit.unwrap_or_else(from_env))
        };
        Self {
            router_grace: secs(config.router_grace_secs, grace_period_secs),
            // An uncapped stage is `Duration::MAX`, not zero: `duration_from_secs`
            // maps a non-finite value to `ZERO`, which would skip the barrier
            // outright rather than let the remaining total bound it.
            inflight: match config.inflight_timeout_secs.unwrap_or_else(inflight_timeout_secs) {
                v if v.is_infinite() => Duration::MAX,
                v => duration_from_secs(v),
            },
            kv_transfer: secs(config.kv_transfer_timeout_secs, drain_timeout_secs),
            cleanup: config
                .cleanup_timeout_secs
                .filter(|v| *v > 0.0)
                .map(duration_from_secs)
                .unwrap_or_else(cleanup_timeout),
        }
    }

    fn get(&self, stage: Stage) -> Duration {
        match stage {
            // No cap of their own — one discovery RPC and one atomic flip — so
            // `allowance` reduces to the remaining total. That is deliberately
            // not "unbounded": the discovery RPC can block on an unreachable
            // API server, and the remaining total is what stops it wedging
            // shutdown before cleanup.
            Stage::Unregister | Stage::StopAdmission => Duration::MAX,
            Stage::RouterGrace => self.router_grace,
            Stage::Inflight => self.inflight,
            Stage::KvTransfer => self.kv_transfer,
            Stage::Cleanup => self.cleanup,
        }
    }
}

/// One stage of the graceful-shutdown sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Remove the worker from discovery so routers stop selecting it.
    Unregister,
    /// Keep serving while routers observe the discovery unregister.
    RouterGrace,
    /// Close admission to new requests.
    StopAdmission,
    /// Wait for admitted requests to finish.
    Inflight,
    /// Wait for prefill KV transfers that outlive the request stream.
    KvTransfer,
    /// Release engine resources.
    Cleanup,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Unregister => "unregister",
            Stage::RouterGrace => "router_grace",
            Stage::StopAdmission => "stop_admission",
            Stage::Inflight => "inflight",
            Stage::KvTransfer => "kv_transfer",
            Stage::Cleanup => "cleanup",
        }
    }
}

/// Why a stage stopped. Reported on every stage so a shutdown can be read from
/// logs without inferring intent from timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageReason {
    /// The stage's completion condition was met.
    Completed,
    /// The stage ran out of budget.
    TimedOut,
    /// The stage did not apply, or had zero budget.
    Skipped,
    /// The drain was cancelled (Admin `resume`, or a superseding shutdown).
    Cancelled,
    /// The engine cannot report this condition; the declared fallback applied.
    Unsupported,
}

impl StageReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StageReason::Completed => "completed",
            StageReason::TimedOut => "timed_out",
            StageReason::Skipped => "skipped",
            StageReason::Cancelled => "cancelled",
            StageReason::Unsupported => "unsupported",
        }
    }
}

/// What one stage did. Both shutdown entry points return these so stage
/// logging and metrics have a single implementation.
#[derive(Clone, Debug)]
pub struct StageOutcome {
    pub stage: Stage,
    pub reason: StageReason,
    pub elapsed: Duration,
    /// Budget left when the stage finished; `None` when unbounded.
    pub remaining_total: Option<Duration>,
    /// Why the stage could not complete, when the cause is worth surfacing to
    /// an operator — e.g. the last error an engine's predicate raised. Fed to
    /// the Admin API's `last_error` so a stalled drain is explainable.
    pub detail: Option<String>,
}

impl StageOutcome {
    pub(crate) fn new(
        stage: Stage,
        reason: StageReason,
        elapsed: Duration,
        budget: &ShutdownBudget,
    ) -> Self {
        Self {
            stage,
            reason,
            elapsed,
            remaining_total: budget.remaining(),
            detail: None,
        }
    }

    pub(crate) fn with_detail(mut self, detail: Option<String>) -> Self {
        self.detail = detail;
        self
    }

    /// Emit the stage record. `TimedOut` is a warning because it means the
    /// stage's completion condition was not met.
    pub(crate) fn log(&self) {
        let remaining = self
            .remaining_total
            .map(|d| d.as_secs_f64())
            .unwrap_or(f64::INFINITY);
        if self.reason == StageReason::TimedOut {
            tracing::warn!(
                stage = self.stage.name(),
                reason = self.reason.as_str(),
                elapsed_s = self.elapsed.as_secs_f64(),
                remaining_total_s = remaining,
                detail = self.detail.as_deref().unwrap_or(""),
                "shutdown stage timed out"
            );
        } else {
            tracing::info!(
                stage = self.stage.name(),
                reason = self.reason.as_str(),
                elapsed_s = self.elapsed.as_secs_f64(),
                remaining_total_s = remaining,
                "shutdown stage finished"
            );
        }
    }
}

/// The single SIGTERM-to-exit budget every stage draws from.
///
/// Stage maxima are caps, not additive deadlines: a stage gets
/// `min(stage_max, remaining_total)`. No reserve is withheld here — see
/// [`ShutdownBudget::allowance`] for why withholding one zeroed the in-flight
/// barrier under the debug defaults. Cleanup is funded instead by
/// `cleanup_once`'s own floor, with [`force_exit_deadline`] extending the
/// watchdog to cover it.
#[derive(Clone, Copy, Debug)]
pub struct ShutdownBudget {
    /// `None` for the reversible Admin drain, which has no deadline.
    deadline: Option<Instant>,
    maxima: StageMaxima,
    /// Programmatic fallback override; takes precedence over the environment,
    /// which in turn takes precedence over the engine's declaration.
    kv_fallback: Option<KvTransferFallback>,
}

impl ShutdownBudget {
    /// A drain with no deadline — the reversible Admin path, which ends when it
    /// reaches `drained` or is cancelled by `resume`.
    pub fn unbounded() -> Self {
        Self {
            deadline: None,
            maxima: StageMaxima::resolve(&ShutdownConfig::default()),
            kv_fallback: None,
        }
    }

    /// Arm the total budget now. Callers arm this once, at the instant shutdown
    /// begins, and never reset it.
    pub fn starting_now(total: Duration) -> Self {
        Self {
            // A pathological env value could overflow the deadline; saturate
            // rather than panic inside `Instant`'s addition.
            deadline: Instant::now().checked_add(total),
            maxima: StageMaxima::resolve(&ShutdownConfig::default()),
            kv_fallback: None,
        }
    }

    /// Arm the total budget from an explicit config, falling back to the
    /// environment for anything the caller left unset.
    pub fn from_config(config: &ShutdownConfig) -> Self {
        Self::from_config_starting_at(config, Instant::now())
    }

    /// Arm the total budget from an explicit config, measured from `origin`
    /// rather than from now.
    ///
    /// The force-exit watchdog starts its clock the instant the shutdown token
    /// is cancelled, but the stages are armed later — `begin_engine_route_shutdown`
    /// and the RL endpoint teardown both run first, and both are unbounded. When
    /// the budget measured from *its* start instead, that skew came straight out
    /// of the cleanup floor `force_exit_deadline` adds, so the watchdog fired
    /// during `engine.cleanup()` — the exact failure the floor exists to prevent.
    /// Sharing one origin is what makes the floor real.
    pub fn from_config_starting_at(config: &ShutdownConfig, origin: Instant) -> Self {
        let maxima = StageMaxima::resolve(config);
        // Via `total_budget` so this and the hard-exit timer cannot diverge —
        // and so an unrepresentable `total_secs` is clamped rather than
        // panicking here, which a raw `from_secs_f64` did.
        let total = total_budget(config);
        Self {
            // A pathological env value could overflow the deadline; saturate
            // rather than panic inside `Instant`'s addition.
            deadline: origin.checked_add(total),
            maxima,
            kv_fallback: config.kv_transfer_fallback,
        }
    }

    /// Resolved cap for `stage`, before the remaining total is applied.
    pub fn stage_max(&self, stage: Stage) -> Duration {
        self.maxima.get(stage)
    }

    /// Programmatic KV fallback override, if the caller set one.
    pub fn kv_fallback_override(&self) -> Option<KvTransferFallback> {
        self.kv_fallback
    }

    /// Arm the total budget from the configured post-signal deadline.
    pub fn from_env() -> Self {
        Self::from_config(&ShutdownConfig::default())
    }

    /// Budget left, or `None` when unbounded. Zero once the deadline passes.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// `true` once a bounded budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining().is_some_and(|left| left.is_zero())
    }

    /// How long `stage` may run: `min(stage_max, remaining)`, or `None` when
    /// unbounded.
    ///
    /// No reserve is withheld for cleanup here, deliberately. Subtracting one
    /// zeroed the in-flight barrier under the debug defaults — grace (5s) plus
    /// the reserve (5s) consumed the whole 10s total — so shutdown released
    /// engine memory while a request was still executing, which is the exact
    /// failure this sequence exists to prevent. `cleanup_once` already
    /// substitutes [`CLEANUP_RESERVE_S`] as a floor when the budget is spent,
    /// so cleanup was never actually relying on the withholding, and the
    /// hard-exit deadline remains the real backstop.
    ///
    /// A zero allowance does not mean the same thing to every stage:
    ///
    /// * A read-only predicate stage (in-flight, KV quiescence) still gets one
    ///   poll, because `timeout(ZERO, fut)` polls once before cancelling and
    ///   an already-satisfied condition should complete rather than be
    ///   reported as unfinished.
    /// * A stage with side effects must **not** run under a zero timeout.
    ///   `cleanup` would start teardown and abandon it on the first poll, so
    ///   `cleanup_once` substitutes the reserve floor instead.
    /// * A fixed sleep (`router_grace`) is simply skipped.
    pub fn allowance(&self, stage: Stage) -> Option<Duration> {
        let remaining = self.remaining()?;
        Some(self.maxima.get(stage).min(remaining))
    }
}

/// The single total shutdown budget for `config`.
///
/// Shared by [`ShutdownBudget::from_config`] and by the hard-exit timer in
/// `Worker::run`, so the deadline the stages spend against and the deadline
/// that force-exits the process can never disagree — previously the timer
/// read the environment directly and ignored a programmatic `total_secs`
/// entirely.
pub(crate) fn total_budget(config: &ShutdownConfig) -> Duration {
    match config.total_secs {
        Some(secs) => duration_from_secs(secs),
        // The grace sleep is a fixed wait, so it sits on top of the
        // drain+cleanup timeout rather than eating into it.
        None => shutdown_deadline(
            graceful_shutdown_timeout(),
            config.router_grace_secs.unwrap_or_else(grace_period_secs),
        ),
    }
}

/// The instant the process force-exits, which is the stage budget plus the
/// floor `cleanup_once` is guaranteed to grant.
///
/// The two must not be the same instant. A stage is allowed to spend the whole
/// remaining budget — the KV wait does exactly that whenever its cap exceeds
/// what is left, which is the default on both build profiles — and
/// `cleanup_once` then falls back to [`CLEANUP_RESERVE_S`]. If the watchdog
/// fired at the end of the stage budget it would kill the process during that
/// floor, so a healthy worker whose engine simply cannot report KV quiescence
/// exited 70 with its engine never cleaned up. Extending the watchdog by the
/// floor does not lengthen shutdown in the normal case; it stops the two
/// racing.
pub(crate) fn force_exit_deadline(config: &ShutdownConfig) -> Duration {
    total_budget(config).saturating_add(Duration::from_secs_f64(CLEANUP_RESERVE_S))
}

/// Compose the post-signal shutdown deadline from the drain+cleanup
/// timeout and the grace-period sleep that precedes them.
///
/// The grace sleep is a fixed wait (not a hang risk), so reserving its
/// duration on top of `timeout` ensures the drain loop and
/// `engine.cleanup()` always get the full timeout budget regardless of
/// how the operator configures the grace period. Without this reserve,
/// a grace period equal to the timeout (the debug default — both 5s)
/// consumes the whole budget and the deadline expires before drain or
/// cleanup get scheduled.
pub(crate) fn shutdown_deadline(timeout: Duration, grace_secs: f64) -> Duration {
    let grace = duration_from_secs(grace_secs);
    timeout.saturating_add(grace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// These tests mutate process-global environment state; serialize them so
    /// cargo's parallel test threads don't race.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    struct EnvGuard(&'static str);

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            unsafe { std::env::set_var(key, value) };
            Self(key)
        }
        fn unset(key: &'static str) -> Self {
            unsafe { std::env::remove_var(key) };
            Self(key)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe { std::env::remove_var(self.0) };
        }
    }

    #[test]
    fn env_secs_treats_unset_empty_and_invalid_alike() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = EnvGuard::unset(GRACE_PERIOD_ENV);
        assert_eq!(env_secs(GRACE_PERIOD_ENV), None);

        for bad in ["", "   ", "abc", "NaN", "inf", "-inf"] {
            let _g = EnvGuard::set(GRACE_PERIOD_ENV, bad);
            assert_eq!(env_secs(GRACE_PERIOD_ENV), None, "{bad:?} must be rejected");
        }
    }

    /// Regression: the empty check trimmed but the parse did not, so a value
    /// carrying a trailing newline — what a Helm block scalar or a ConfigMap
    /// key routinely produces — was rejected and silently replaced by the
    /// default. Python's `float()` strips, so the same variable resolved to two
    /// different numbers in one deployment.
    #[test]
    fn env_secs_accepts_values_with_surrounding_whitespace() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for padded in [" 30", "30 ", "30\n", "  30.5  "] {
            let _g = EnvGuard::set(GRACE_PERIOD_ENV, padded);
            let parsed = env_secs(GRACE_PERIOD_ENV);
            assert!(
                parsed.is_some_and(|v| v == 30.0 || v == 30.5),
                "{padded:?} must parse, got {parsed:?}"
            );
        }
    }

    #[test]
    fn grace_period_resolves_default_value_and_floor() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            let _g = EnvGuard::unset(GRACE_PERIOD_ENV);
            assert_eq!(grace_period_secs(), DEFAULT_GRACE_PERIOD_SECS);
        }
        {
            let _g = EnvGuard::set(GRACE_PERIOD_ENV, "2.5");
            assert_eq!(grace_period_secs(), 2.5);
        }
        {
            // Zero is meaningful here: skip the sleep.
            let _g = EnvGuard::set(GRACE_PERIOD_ENV, "-1");
            assert_eq!(grace_period_secs(), 0.0);
        }
    }

    #[test]
    fn drain_timeout_resolves_default_value_and_floor() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            let _g = EnvGuard::unset(DRAIN_TIMEOUT_ENV);
            assert_eq!(drain_timeout_secs(), DEFAULT_DRAIN_TIMEOUT_S);
        }
        {
            let _g = EnvGuard::set(DRAIN_TIMEOUT_ENV, "12");
            assert_eq!(drain_timeout_secs(), 12.0);
        }
        {
            let _g = EnvGuard::set(DRAIN_TIMEOUT_ENV, "-1");
            assert_eq!(drain_timeout_secs(), 0.0);
        }
    }

    #[test]
    fn cleanup_timeout_defaults_and_parses() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            let _g = EnvGuard::unset(CLEANUP_TIMEOUT_ENV);
            assert_eq!(cleanup_timeout(), graceful_shutdown_timeout());
        }
        {
            let _g = EnvGuard::set(CLEANUP_TIMEOUT_ENV, "12.5");
            assert_eq!(cleanup_timeout(), Duration::from_secs_f64(12.5));
        }
    }

    /// The one knob where zero is NOT a coherent request — it would cancel
    /// cleanup on its first poll.
    #[test]
    fn cleanup_timeout_rejects_non_positive_rather_than_clamping() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for bad in ["0", "-1", "abc"] {
            let _g = EnvGuard::set(CLEANUP_TIMEOUT_ENV, bad);
            assert_eq!(
                cleanup_timeout(),
                graceful_shutdown_timeout(),
                "{bad:?} must fall back to the default, not produce a zero budget"
            );
        }
    }

    #[test]
    fn shutdown_deadline_adds_grace_to_timeout() {
        assert_eq!(
            shutdown_deadline(Duration::from_secs(30), 5.0),
            Duration::from_secs(35)
        );
    }

    #[test]
    fn shutdown_deadline_ignores_non_positive_grace() {
        assert_eq!(
            shutdown_deadline(Duration::from_secs(30), 0.0),
            Duration::from_secs(30)
        );
        assert_eq!(
            shutdown_deadline(Duration::from_secs(30), -1.0),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn shutdown_deadline_saturates_instead_of_overflowing() {
        assert_eq!(shutdown_deadline(Duration::MAX, 5.0), Duration::MAX);
    }

    // -------------------------------------------------------------------
    // ShutdownBudget
    // -------------------------------------------------------------------

    /// The Admin drain has no deadline: stages wait until they complete or are
    /// cancelled, never because time ran out.
    #[test]
    fn unbounded_budget_has_no_remaining_and_no_allowance() {
        let budget = ShutdownBudget::unbounded();
        assert_eq!(budget.remaining(), None);
        assert!(!budget.is_exhausted());
        for stage in [
            Stage::RouterGrace,
            Stage::Inflight,
            Stage::KvTransfer,
            Stage::Cleanup,
        ] {
            assert_eq!(budget.allowance(stage), None, "{stage:?}");
        }
    }

    /// Stage maxima are caps, not additive deadlines.
    #[test]
    fn allowance_is_capped_by_the_stage_maximum() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _grace = EnvGuard::set(GRACE_PERIOD_ENV, "5");
        let _cleanup = EnvGuard::set(CLEANUP_TIMEOUT_ENV, "10");

        // Plenty of total budget: the stage max is what binds.
        let budget = ShutdownBudget::starting_now(Duration::from_secs(3600));
        assert_eq!(
            budget.allowance(Stage::RouterGrace),
            Some(Duration::from_secs(5))
        );
    }

    /// With little total left, the remaining budget binds instead of the cap.
    #[test]
    fn allowance_is_capped_by_the_remaining_total() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _grace = EnvGuard::set(GRACE_PERIOD_ENV, "30");

        let budget = ShutdownBudget::starting_now(Duration::from_secs(20));
        let allowance = budget
            .allowance(Stage::RouterGrace)
            .expect("bounded budget yields an allowance");
        assert!(
            allowance <= Duration::from_secs(20) && allowance > Duration::from_secs(19),
            "expected ~20s (the remaining total, under the 30s cap), got {allowance:?}"
        );
    }

    /// The defect that made the whole sequence pointless under the debug
    /// defaults: withholding a cleanup reserve left the in-flight barrier with
    /// a zero budget, so it "timed out" in ~1ms and `engine.cleanup()` ran
    /// while a request was still executing.
    ///
    /// Debug totals are `timeout (5s) + grace (5s)`, so after the grace there
    /// is exactly 5s left — all of which the old formula reserved.
    #[test]
    fn inflight_barrier_still_has_budget_after_the_grace_at_debug_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _grace = EnvGuard::unset(GRACE_PERIOD_ENV);
        let _inflight = EnvGuard::unset(INFLIGHT_TIMEOUT_ENV);
        let _cleanup = EnvGuard::unset(CLEANUP_TIMEOUT_ENV);

        // The state the worker is in once the router grace has been served.
        let after_grace = ShutdownBudget::starting_now(graceful_shutdown_timeout());
        let allowance = after_grace
            .allowance(Stage::Inflight)
            .expect("bounded budget yields an allowance");
        assert!(
            !allowance.is_zero(),
            "the in-flight barrier must get a real budget after the grace; \
             a zero budget lets cleanup run during an active request"
        );
    }

    /// Cleanup is funded even when the stages before it spent everything —
    /// `cleanup_once` substitutes the floor, which is why `allowance` does not
    /// need to withhold one.
    #[test]
    fn cleanup_reserve_is_a_floor_not_a_withholding() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _cleanup = EnvGuard::set(CLEANUP_TIMEOUT_ENV, "10");

        // Budget fully spent.
        let budget = ShutdownBudget::starting_now(Duration::ZERO);
        assert_eq!(budget.allowance(Stage::Cleanup), Some(Duration::ZERO));
        assert!(budget.is_exhausted());
        // The floor itself is applied by `Worker::cleanup_once`, which is
        // covered by `cleanup_once_is_bounded_when_engine_hangs`; the budget
        // simply reports the truth rather than pre-emptively withholding.
    }

    /// An expired budget reports exhausted and hands out zero, which callers
    /// treat as "skip this stage" rather than `timeout(ZERO, ..)`.
    #[test]
    fn expired_budget_is_exhausted_and_allows_nothing() {
        let budget = ShutdownBudget::starting_now(Duration::ZERO);
        assert!(budget.is_exhausted());
        assert_eq!(budget.remaining(), Some(Duration::ZERO));
        assert_eq!(budget.allowance(Stage::Inflight), Some(Duration::ZERO));
    }

    /// A pathological env value must not panic in `Instant`'s addition.
    /// An absurd total must not panic — but "does not panic" was the whole
    /// assertion here, and `remaining`/`allowance` contain no panicking
    /// operations, so the test could only have failed if `checked_add` itself
    /// did. Pin the semantics that actually matter: the deadline saturates to
    /// unbounded rather than wrapping into an already-expired budget, which
    /// would skip every stage.
    #[test]
    fn absurd_total_saturates_to_unbounded_rather_than_expiring() {
        let budget = ShutdownBudget::starting_now(Duration::MAX);
        match budget.remaining() {
            // Saturated: `is_exhausted` must stay false, or every stage is
            // skipped at the instant shutdown begins.
            None => assert!(!budget.is_exhausted()),
            Some(remaining) => assert!(
                !remaining.is_zero(),
                "a saturating deadline must not present as already spent"
            ),
        }
        assert!(!budget.is_exhausted());
    }

    /// Regression: reserving `cleanup_timeout()` (which defaults to the whole
    /// deadline) rather than `CLEANUP_RESERVE_S` starved every earlier stage to
    /// zero under default configuration, silently skipping the KV drain.
    #[test]
    fn default_configuration_leaves_the_kv_stage_a_usable_budget() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _grace = EnvGuard::unset(GRACE_PERIOD_ENV);
        let _cleanup = EnvGuard::unset(CLEANUP_TIMEOUT_ENV);
        let _drain = EnvGuard::unset(DRAIN_TIMEOUT_ENV);

        let budget = ShutdownBudget::from_env();
        let kv = budget
            .allowance(Stage::KvTransfer)
            .expect("bounded budget yields an allowance");

        // Not merely non-zero. Reserving `cleanup_timeout()` instead of
        // `CLEANUP_RESERVE_S` still leaves a non-zero remainder under both
        // build profiles, so `!kv.is_zero()` passed under the very mutation
        // this test names. Assert the stage keeps essentially the whole
        // post-grace remainder, which the mutation does not.
        let total = total_budget(&ShutdownConfig::default());
        let grace = duration_from_secs(grace_period_secs());
        let post_grace = total.saturating_sub(grace);
        let floor = post_grace.mul_f64(0.9).min(
            Duration::from_secs_f64(drain_timeout_secs()),
        );
        assert!(
            kv >= floor,
            "the KV stage must keep the post-grace remainder on defaults; \
             got {kv:?}, expected at least {floor:?} of {post_grace:?}"
        );
    }

    #[test]
    fn kv_fallback_override_parses_both_policies_case_insensitively() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (raw, expected) in [
            ("wait", KvTransferFallback::WaitFullBudget),
            ("WAIT", KvTransferFallback::WaitFullBudget),
            ("  skip  ", KvTransferFallback::Skip),
            ("Skip", KvTransferFallback::Skip),
        ] {
            let _g = EnvGuard::set(KV_FALLBACK_ENV, raw);
            assert_eq!(kv_transfer_fallback_override(), Some(expected), "{raw:?}");
        }
    }

    /// An unset or nonsense value must defer to the engine's declaration
    /// rather than silently picking a policy.
    #[test]
    fn kv_fallback_override_defers_to_the_engine_when_unset_or_invalid() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            let _g = EnvGuard::unset(KV_FALLBACK_ENV);
            assert_eq!(kv_transfer_fallback_override(), None);
        }
        for bad in ["", "   ", "yes", "true", "0"] {
            let _g = EnvGuard::set(KV_FALLBACK_ENV, bad);
            assert_eq!(kv_transfer_fallback_override(), None, "{bad:?}");
        }
    }

    #[test]
    fn kv_fallback_names_are_stable_for_logs() {
        assert_eq!(KvTransferFallback::Undeclared.as_str(), "undeclared");
        assert_eq!(KvTransferFallback::WaitFullBudget.as_str(), "wait");
        assert_eq!(KvTransferFallback::Skip.as_str(), "skip");
    }

    /// Regression: `Duration::from_secs_f64` panics on a value it cannot
    /// represent, and every knob feeds one. A panic during shutdown aborts the
    /// drain, so out-of-range values must be neutralised at the boundary.
    #[test]
    fn duration_conversion_never_panics_on_hostile_values() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e30, -1.0, -0.0] {
            let d = duration_from_secs(bad);
            assert!(d <= Duration::from_secs_f64(MAX_CONFIGURED_SECS), "{bad}");
        }
        assert_eq!(duration_from_secs(f64::NAN), Duration::ZERO);
        assert_eq!(duration_from_secs(-1.0), Duration::ZERO);
        // A merely large value clamps rather than panicking.
        assert_eq!(
            duration_from_secs(1e30),
            Duration::from_secs_f64(MAX_CONFIGURED_SECS)
        );
    }

    /// `is_finite()` alone is not enough — `1e30` is finite and still panics.
    #[test]
    fn configured_secs_validation_rejects_out_of_range_magnitudes() {
        assert!(is_valid_configured_secs(0.0));
        assert!(is_valid_configured_secs(45.0));
        assert!(is_valid_configured_secs(MAX_CONFIGURED_SECS));
        for bad in [f64::INFINITY, f64::NAN, 1e30, -1.0] {
            assert!(!is_valid_configured_secs(bad), "{bad} must be rejected");
        }
    }

    #[test]
    fn env_secs_rejects_out_of_range_magnitudes() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for bad in ["1e30", "inf", "-inf", "NaN"] {
            let _g = EnvGuard::set(GRACE_PERIOD_ENV, bad);
            assert_eq!(env_secs(GRACE_PERIOD_ENV), None, "{bad:?}");
        }
    }

    /// The hard-exit timer and the stage budget must come from one number.
    /// Previously the timer read the environment directly, so a programmatic
    /// `total_secs` was silently ignored and the process could force-exit on a
    /// deadline the stages knew nothing about.
    #[test]
    fn total_budget_honours_an_explicit_total() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = EnvGuard::unset(GRACE_PERIOD_ENV);
        let config = ShutdownConfig {
            total_secs: Some(45.0),
            ..ShutdownConfig::default()
        };
        assert_eq!(total_budget(&config), Duration::from_secs(45));

        let budget = ShutdownBudget::from_config(&config);
        let remaining = budget.remaining().expect("bounded");
        assert!(
            remaining > Duration::from_secs(44) && remaining <= Duration::from_secs(45),
            "the budget must start from the same total, got {remaining:?}"
        );
    }

    #[test]
    fn total_budget_falls_back_to_timeout_plus_grace() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _g = EnvGuard::set(GRACE_PERIOD_ENV, "7");
        let config = ShutdownConfig::default();
        assert_eq!(
            total_budget(&config),
            shutdown_deadline(graceful_shutdown_timeout(), 7.0)
        );
    }

    /// Regression: a stage may legitimately spend the entire remaining budget
    /// — `allowance` is `min(cap, remaining)` and the KV cap exceeds what is
    /// left on both build profiles — after which `cleanup_once` is still owed
    /// its floor. If the watchdog fired at the end of the stage budget it
    /// killed the process during that floor, so a healthy prefill worker whose
    /// engine cannot report quiescence exited 70 with its engine never cleaned
    /// up.
    #[test]
    fn force_exit_deadline_leaves_room_for_the_cleanup_floor() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _grace = EnvGuard::unset(GRACE_PERIOD_ENV);
        let config = ShutdownConfig::default();

        let stages = total_budget(&config);
        let watchdog = force_exit_deadline(&config);
        assert!(
            watchdog >= stages + Duration::from_secs_f64(CLEANUP_RESERVE_S),
            "watchdog {watchdog:?} must outlast the stage budget {stages:?} by the cleanup floor"
        );

        // The precondition that triggered it, asserted without reading the
        // clock twice: the KV cap meets or exceeds what is left after the
        // grace, so `min(cap, remaining)` is the whole remainder.
        assert!(
            Duration::from_secs_f64(drain_timeout_secs()) >= graceful_shutdown_timeout(),
            "the KV cap is expected to allow the stage to consume the entire remainder"
        );
    }

    /// Regression: the watchdog and the stage budget used to read the clock at
    /// two different moments — the watchdog when the shutdown token was
    /// cancelled, the budget when the orchestrator was finally reached, with
    /// the unbounded engine-route and RL-endpoint teardown in between. The
    /// identity asserted above still held, but the cleanup floor it buys was
    /// consumed by that skew, so the watchdog fired during `engine.cleanup()`.
    ///
    /// Asserting the identity is not enough; this pins the origin itself.
    #[test]
    fn a_budget_armed_from_an_earlier_origin_spends_the_skew() {
        let config = ShutdownConfig {
            total_secs: Some(30.0),
            ..Default::default()
        };
        let skew = Duration::from_secs(10);
        let origin = Instant::now() - skew;

        let budget = ShutdownBudget::from_config_starting_at(&config, origin);
        let remaining = budget.remaining().expect("a bounded budget");

        // ~20s, not 30s. Armed from `now` instead, the stages would outlive the
        // watchdog — which counts from `origin` — by exactly `skew`.
        assert!(
            remaining <= Duration::from_secs(21) && remaining >= Duration::from_secs(19),
            "expected the 10s skew to come out of the 30s total; got {remaining:?}"
        );
    }

    #[test]
    fn stage_names_are_stable_for_logs_and_metrics() {
        assert_eq!(Stage::RouterGrace.name(), "router_grace");
        assert_eq!(Stage::Inflight.name(), "inflight");
        assert_eq!(Stage::KvTransfer.name(), "kv_transfer");
        assert_eq!(Stage::Cleanup.name(), "cleanup");
        assert_eq!(StageReason::TimedOut.as_str(), "timed_out");
        assert_eq!(StageReason::Unsupported.as_str(), "unsupported");
    }

    #[test]
    fn inflight_timeout_resolves_default_value_and_floor() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            let _g = EnvGuard::unset(INFLIGHT_TIMEOUT_ENV);
            assert_eq!(inflight_timeout_secs(), DEFAULT_INFLIGHT_TIMEOUT_S);
        }
        {
            let _g = EnvGuard::set(INFLIGHT_TIMEOUT_ENV, "20");
            assert_eq!(inflight_timeout_secs(), 20.0);
        }
        {
            let _g = EnvGuard::set(INFLIGHT_TIMEOUT_ENV, "-1");
            assert_eq!(inflight_timeout_secs(), 0.0);
        }
    }
}
