// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sticky-session`: route every later request of a session to the worker that served its last
//! request, falling back to the default cost function.
//!
//! This mirrors the intent of the live router's session affinity in policy state, because
//! offline replay never sets a host affinity target.
//!
//! - A request without a session ID, or the first request of a session, takes the default cost
//!   function's choice: the default scorer with the host's configured weights, then the lowest
//!   cost, or the default's range-normalized softmax when `router_temperature` is positive.
//! - `mode: hard`: a later request goes to its session's bound worker whenever that worker is
//!   eligible; otherwise it takes the default choice and the session rebinds to it.
//! - `mode: bounded`: as `hard`, except that the request leaves its bound worker when that worker's
//!   active requests exceed `load_factor` times the candidates' mean active requests.
//!
//! Sessions live in a bounded least-recently-used map of `max_sessions` entries. Ties and samples
//! use a per-instance RNG seeded by `seed` that visits workers in canonical order. Host pins and
//! eligibility always win; a pinned request rebinds its session to the pinned worker.

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{
    SessionContext, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerType};

use crate::choice::{Chooser, TieBreak, row_of};
use crate::session_map::{DEFAULT_MAX_SESSIONS, SessionMap};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "sticky-session";

const DEFAULT_LOAD_FACTOR: f64 = 1.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Hard,
    Bounded,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    mode: Mode,
    /// Bounded mode only. Absent means the default of 1.25.
    #[serde(default)]
    load_factor: Option<f64>,
    #[serde(default = "default_max_sessions")]
    max_sessions: usize,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    tie_break: TieBreak,
}

fn default_max_sessions() -> usize {
    DEFAULT_MAX_SESSIONS
}

#[derive(Clone, Copy, Debug)]
struct Settings {
    mode: Mode,
    load_factor: f64,
    max_sessions: usize,
    seed: Option<u64>,
    tie_break: TieBreak,
}

impl Parameters {
    fn validate(self) -> Result<Settings, WorkerSelectionPolicyProviderError> {
        let load_factor = match (self.mode, self.load_factor) {
            (Mode::Hard, Some(_)) => {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "load_factor applies only to mode bounded",
                ));
            }
            // Hard mode never compares load.
            (Mode::Hard, None) => f64::INFINITY,
            (Mode::Bounded, load_factor) => {
                let load_factor = load_factor.unwrap_or(DEFAULT_LOAD_FACTOR);
                if !(load_factor.is_finite() && load_factor >= 1.0) {
                    return Err(WorkerSelectionPolicyProviderError::new(
                        "load_factor must be a finite number of at least 1.0",
                    ));
                }
                load_factor
            }
        };
        if self.max_sessions == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "max_sessions must be positive",
            ));
        }
        Ok(Settings {
            mode: self.mode,
            load_factor,
            max_sessions: self.max_sessions,
            seed: self.seed,
            tie_break: self.tie_break,
        })
    }
}

struct StickySessionPicker {
    settings: Settings,
    temperature: f64,
    chooser: Chooser,
    sessions: SessionMap,
}

impl StickySessionPicker {
    fn new(settings: Settings, temperature: f64) -> Self {
        Self {
            chooser: Chooser::new(settings.seed, settings.tie_break),
            sessions: SessionMap::new(settings.max_sessions),
            settings,
            temperature,
        }
    }

    /// The session's bound row, if its worker is a candidate and within the load bound.
    fn bound_row(
        &self,
        input: WorkerInputView<'_>,
        session: &str,
    ) -> Result<Option<usize>, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let Some(row) = self
            .sessions
            .get(session)
            .and_then(|worker| row_of(candidates, worker))
        else {
            return Ok(None);
        };
        if self.settings.mode == Mode::Hard {
            return Ok(Some(row));
        }
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let requests = |row: usize| load.get(row).map_or(0, |load| load.active_requests());
        let mean =
            (0..candidates.len()).map(requests).sum::<usize>() as f64 / candidates.len() as f64;
        Ok((requests(row) as f64 <= self.settings.load_factor * mean).then_some(row))
    }

    fn default_choice(&mut self, input: WorkerInputView<'_>) -> usize {
        let candidates = input.candidates();
        let cost = |row: usize| candidates[row].cost();
        if self.temperature == 0.0 {
            self.chooser.lowest(cost)
        } else {
            self.chooser.sample_default(cost, self.temperature)
        }
    }
}

impl WorkerPicker for StickySessionPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        self.chooser.begin(candidates);
        let session = context.session_context().map(SessionContext::session_id);
        let row = if let Some(pinned) = context.pinned_worker() {
            row_of(candidates, pinned).unwrap_or(0)
        } else if let Some(row) = session
            .map(|session| self.bound_row(input, session))
            .transpose()?
            .flatten()
        {
            row
        } else {
            self.default_choice(input)
        };
        if let Some(session) = session {
            self.sessions.bind(session, candidates[row].worker());
        }
        Ok(row)
    }
}

fn policy(config: &KvRouterConfig, role: WorkerType, settings: Settings) -> WorkerSelectionPolicy {
    // The fallback is the default cost function with the host's configured weights.
    let scorer = crate::default::cost_scorer(config, config, role);
    let temperature = crate::default::router_temperature(config);
    WorkerSelectionPolicy::new(
        config.clone(),
        role.default_selector_label(),
        vec![scorer],
        Box::new(StickySessionPicker::new(settings, temperature)),
    )
    .with_exclusive_affinity(true)
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let settings = parameters.deserialize::<Parameters>()?.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, role, _partition| policy(config, role, settings),
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use dynamo_kv_router::protocols::WorkerWithDpRank;

    use super::*;
    use crate::test_support::{
        Worker, in_session, populate_replay, request, resolve_policy, select_populated,
        select_replay,
    };

    fn sticky(parameters: &str) -> WorkerSelectionPolicy {
        resolve_policy(POLICY_TYPE, parameters).unwrap()
    }

    /// Without sessions, `sticky-session` is the default cost function: with the reservoir
    /// discipline it matches the seeded `dynamo-default-cost-fn` draw for draw, ties included.
    #[test]
    fn requests_without_a_session_take_the_default_choice() {
        for mode in ["hard", "bounded"] {
            let policy = sticky(&format!("{{mode: {mode}, seed: 4, tie_break: reservoir}}"));
            let default = resolve_policy("dynamo-default-cost-fn", "{seed: 4}").unwrap();
            let mut rng = fastrand::Rng::with_seed(8);
            for _ in 0..300 {
                let workers: Vec<Worker> = (0..rng.u64(2..=8))
                    .map(|id| {
                        Worker::new(id)
                            .cached(rng.usize(0..3))
                            .requests(rng.usize(0..4))
                            .prefill(16 * rng.usize(0..3))
                    })
                    .collect();
                let mut request = request(4, 1);
                let configs = populate_replay(&mut request, &workers);
                assert_eq!(
                    select_populated(&policy, &request, &configs),
                    select_populated(&default, &request, &configs),
                    "{mode}"
                );
            }
        }
    }

    #[test]
    fn hard_mode_keeps_a_session_on_its_worker() {
        let policy = sticky("{mode: hard, seed: 1}");
        let idle = [Worker::new(0).prefill(64), Worker::new(1)];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &idle),
            1
        );
        // Worker 1 is now far costlier, but the session stays; a new session takes worker 0.
        let busy = [Worker::new(0), Worker::new(1).prefill(50_000).requests(40)];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &busy),
            1
        );
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "t"), &busy),
            0
        );
        assert_eq!(select_replay(&policy, request(4, 1), &busy), 0);
    }

    #[test]
    fn hard_mode_rebinds_when_its_worker_is_ineligible() {
        let policy = sticky("{mode: hard, seed: 1}");
        let workers = [
            Worker::new(0),
            Worker::new(1).prefill(64),
            Worker::new(2).prefill(128),
        ];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &workers),
            0
        );
        let mut excluded = in_session(request(4, 1), "s");
        excluded.allowed_worker_ids = Some(HashSet::from([1, 2]));
        assert_eq!(select_replay(&policy, excluded, &workers), 1);
        // The session now follows worker 1 even though worker 0 is eligible and cheaper.
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &workers),
            1
        );
    }

    #[test]
    fn bounded_mode_leaves_a_worker_above_the_load_bound() {
        let policy = sticky("{mode: bounded, load_factor: 1.5, seed: 1}");
        let first = [Worker::new(0), Worker::new(1).prefill(64)];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &first),
            0
        );
        // Mean active requests 4: worker 0 at 6 sits exactly on 1.5 × 4 and stays.
        let at_bound = [
            Worker::new(0).requests(6).prefill(640),
            Worker::new(1).requests(2),
        ];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &at_bound),
            0
        );
        // At 7 it exceeds 1.5 × 4.5 = 6.75, so the session takes the default choice and rebinds.
        let above = [
            Worker::new(0).requests(7).prefill(640),
            Worker::new(1).requests(2),
        ];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &above),
            1
        );
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &first),
            1
        );
        // Hard mode ignores load entirely.
        let hard = sticky("{mode: hard, seed: 1}");
        assert_eq!(
            select_replay(&hard, in_session(request(4, 1), "s"), &first),
            0
        );
        assert_eq!(
            select_replay(&hard, in_session(request(4, 1), "s"), &above),
            0
        );
    }

    #[test]
    fn sessions_beyond_max_sessions_are_forgotten() {
        let policy = sticky("{mode: hard, max_sessions: 1, seed: 1}");
        let first = [Worker::new(0), Worker::new(1).prefill(64)];
        let flipped = [Worker::new(0).prefill(64), Worker::new(1)];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "a"), &first),
            0
        );
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "b"), &first),
            0
        );
        // `a` was evicted by `b`, so it takes the default choice again.
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "a"), &flipped),
            1
        );
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "a"), &first),
            1
        );
    }

    #[test]
    fn a_host_pin_wins_and_rebinds_the_session() {
        let policy = sticky("{mode: hard, seed: 1}");
        let workers = [Worker::new(0), Worker::new(1).prefill(64)];
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &workers),
            0
        );
        let mut pinned = in_session(request(4, 1), "s");
        pinned.pinned_worker = Some(WorkerWithDpRank::from_worker_id(1));
        assert_eq!(select_replay(&policy, pinned, &workers), 1);
        assert_eq!(
            select_replay(&policy, in_session(request(4, 1), "s"), &workers),
            1
        );
    }

    #[test]
    fn parameter_validation_is_strict() {
        for (parameters, expected) in [
            ("{}", "missing field `mode`"),
            ("{mode: soft}", "unknown variant `soft`"),
            (
                "{mode: hard, load_factor: 1.25}",
                "load_factor applies only to mode bounded",
            ),
            (
                "{mode: bounded, load_factor: 0.9}",
                "load_factor must be a finite number of at least 1.0",
            ),
            (
                "{mode: bounded, load_factor: .inf}",
                "load_factor must be a finite number of at least 1.0",
            ),
            (
                "{mode: bounded, load_factor: .nan}",
                "load_factor must be a finite number of at least 1.0",
            ),
            (
                "{mode: hard, max_sessions: 0}",
                "max_sessions must be positive",
            ),
            ("{mode: hard, affinity: 1}", "unknown field `affinity`"),
            ("{mode: hard, tie_break: coin}", "unknown variant `coin`"),
        ] {
            let Err(error) = resolve_policy(POLICY_TYPE, parameters) else {
                panic!("{parameters} must fail");
            };
            assert!(error.contains(expected), "{parameters}: {error}");
        }
        for valid in [
            "{mode: hard}",
            "{mode: bounded}",
            "{mode: bounded, load_factor: 2.0, max_sessions: 8, seed: 3, tie_break: one_draw}",
        ] {
            resolve_policy(POLICY_TYPE, valid).unwrap_or_else(|error| panic!("{valid}: {error}"));
        }
    }
}
