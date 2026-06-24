// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The planner-replay seam.
//!
//! A [`PlannerHook`] is invoked once per [`PlannerTick`](super::events::SimulationEventKind::PlannerTick)
//! event inside the unified `run()` loop: it receives the metrics drained at the tick
//! (per-pass FPM snapshots accumulated since the last tick, the traffic window, and worker
//! counts) and returns a scaling decision plus the time of the next tick. This replaces the
//! old Python-driven `advance_to`/`apply_scaling` stepping loop — the planner's decision logic
//! still lives in Python (via a PyO3 wrapper that implements this trait), but the simulation
//! now owns the drive loop and the tick cadence is just another event.
//!
//! [`NoopPlannerHook`] is the inert stand-in for tests and the no-planner path.

use super::components::TrafficStats;
use crate::common::protocols::ForwardPassSnapshot;

/// Metrics handed to the planner at one tick. The runtime has already advanced the clock to
/// `now_ms` and settled all same-timestamp work before this is built, so the planner observes a
/// consistent post-settlement snapshot (matching the old advance-then-tick ordering).
#[derive(Debug)]
pub struct PlannerTickMetrics {
    /// Simulated clock at this tick (equals the scheduled tick time).
    pub now_ms: f64,
    /// Every prefill FPM snapshot accumulated since the previous tick (empty in agg mode,
    /// which routes all passes through `decode_fpm`).
    pub prefill_fpm: Vec<(usize, ForwardPassSnapshot)>,
    /// Every decode (agg: aggregated) FPM snapshot accumulated since the previous tick.
    pub decode_fpm: Vec<(usize, ForwardPassSnapshot)>,
    /// Traffic stats over `[previous tick, now]`. Drained every tick; the Python side merges
    /// partial windows across ticks that don't consume traffic (mirrors how it already
    /// accumulates FPM snapshots).
    pub traffic: TrafficStats,
    /// Active (ready, non-draining) worker counts. Agg reports decode only (`active_prefill = 0`).
    pub active_prefill: usize,
    pub active_decode: usize,
    /// Total workers including pending startup + pending removal.
    pub total_prefill: usize,
    pub total_decode: usize,
}

/// The planner's decision for one tick. A `None` target leaves that count unchanged;
/// `next_tick_ms = None` stops the recurring tick (the sim then runs to natural completion).
#[derive(Debug, Default, Clone)]
pub struct PlannerTickDecision {
    /// Target prefill replica count (agg ignores this).
    pub target_prefill: Option<usize>,
    /// Target decode (agg: total) replica count.
    pub target_decode: Option<usize>,
    /// Absolute simulated time (ms) of the next tick. `None` => do not re-arm.
    pub next_tick_ms: Option<f64>,
}

/// Implemented by the (Python-backed) planner. One call per `PlannerTick`; ticks are seconds
/// apart in sim-time, so the per-call cost (a GIL acquire + a Python method) is negligible.
///
/// Intentionally NOT `Send`: the simulation runs single-threaded and the PyO3 implementation
/// holds Python state, so the runtime loop holds the GIL rather than crossing threads.
pub trait PlannerHook {
    /// First tick time (ms) the planner wants. Seeds the first `PlannerTick`. A non-finite
    /// value means "no tick" (the runtime skips seeding) — used by [`NoopPlannerHook`].
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64>;

    /// Drive one tick: decide scaling targets and the next tick time.
    fn on_tick(&mut self, metrics: PlannerTickMetrics) -> anyhow::Result<PlannerTickDecision>;
}

/// A planner that never scales and never re-arms — used in tests and as an inert stand-in.
pub struct NoopPlannerHook;

impl PlannerHook for NoopPlannerHook {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(f64::INFINITY) // non-finite => the runtime does not seed a tick
    }

    fn on_tick(&mut self, _metrics: PlannerTickMetrics) -> anyhow::Result<PlannerTickDecision> {
        Ok(PlannerTickDecision::default())
    }
}
