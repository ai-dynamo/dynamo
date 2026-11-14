// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive imbalance policy: rewrites the router weights from pool-level load imbalance.

mod controller;
mod math;
mod picker;
mod scorer;
mod state;
mod ticker;

use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::plugins::prelude::*;

/// YAML `parameters:` for one policy instance. The cost reproduces the default policy's device-cache
/// and load terms; with these defaults the weights stay at baseline under even load, so picks match the
/// default policy's whenever the terms it omits (lower-tier and shared cache hits, credit decay,
/// per-request weight overrides) are inactive. Weights only move once the smoothed imbalance passes
/// `enter_threshold` or, with `slot_capacity` set, a worker is full.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Parameters {
    /// Control loop period in milliseconds; must be positive.
    #[serde(default = "default_interval_ms")]
    pub(crate) interval_ms: u64,
    /// Weight of the previous smoothed imbalance in the moving average, in `0..=1`.
    /// Higher values react more slowly to a change in load.
    #[serde(default = "default_smoothing")]
    pub(crate) smoothing: f64,
    /// Smoothed imbalance at which the split is 0.5, in `0..=1`.
    #[serde(default = "default_sigmoid_midpoint")]
    pub(crate) sigmoid_midpoint: f64,
    /// Steepness of the sigmoid that maps imbalance to split; must be positive.
    #[serde(default = "default_sigmoid_slope")]
    pub(crate) sigmoid_slope: f64,
    /// How far both load weights rise at full split (`1 + gain`); finite and non-negative.
    #[serde(default = "default_gain")]
    pub(crate) gain: f64,
    /// Active-request weight added at full split, in KV blocks per in-flight request; finite and
    /// non-negative. Scaling by `gain` keeps a baseline of 0 (the router default) at 0, and only this
    /// term counts in-flight requests (sequence slots) rather than KV blocks. 0 restores pure scaling.
    #[serde(default = "default_spread_active_request_weight")]
    pub(crate) spread_active_request_weight: f64,
    /// Minimum cycles in the current state before engaging or disengaging again; 0 disables the dwell.
    #[serde(default = "default_dwell_cycles")]
    pub(crate) dwell_cycles: u32,
    /// Age in milliseconds after which the loop stops reading the last load sample and holds its
    /// state until the scorer publishes a new one; must be at least `interval_ms`. A gap between
    /// requests does not drain the weights: the next request's sample does, if the pool is even.
    #[serde(default = "default_stale_after_ms")]
    pub(crate) stale_after_ms: u64,
    /// Smoothed imbalance above which the loop engages, in `0..=1`.
    #[serde(default = "default_enter_threshold")]
    pub(crate) enter_threshold: f64,
    /// Smoothed imbalance below which the loop disengages, in `0..=1` and below `enter_threshold`.
    #[serde(default = "default_exit_threshold")]
    pub(crate) exit_threshold: f64,
    /// Sequence slots per worker (the engine's `max_num_seqs`); absent disables the saturation rule.
    /// The router does not expose this value to the policy, so it is configured here.
    #[serde(default)]
    pub(crate) slot_capacity: Option<u32>,
    /// Fraction of `slot_capacity` at which the busiest worker counts as full, in `0..=1` and above 0.
    /// A full worker engages the loop, and while a worker is full the loop does not disengage and
    /// keeps at least its engage-level split: a pool that is full but even has a low imbalance, and
    /// the affinity weights then send the next wave to a worker without a free slot.
    #[serde(default = "default_saturation_hold")]
    pub(crate) saturation_hold: f64,
    /// Baseline cache credit; absent falls back to the router configuration.
    #[serde(default)]
    pub(crate) overlap_score_credit: Option<f64>,
    /// Baseline prefill load scale; absent falls back to the router configuration.
    #[serde(default)]
    pub(crate) prefill_load_scale: Option<f64>,
    /// Baseline active-request weight; absent falls back to the router configuration.
    #[serde(default)]
    pub(crate) decode_active_request_weight: Option<f64>,
    /// Baseline picker temperature; absent falls back to the router configuration.
    #[serde(default)]
    pub(crate) router_temperature: Option<f64>,
}

fn default_interval_ms() -> u64 {
    250
}
fn default_smoothing() -> f64 {
    0.3
}
fn default_sigmoid_midpoint() -> f64 {
    0.5
}
fn default_sigmoid_slope() -> f64 {
    10.0
}
fn default_gain() -> f64 {
    4.0
}
fn default_spread_active_request_weight() -> f64 {
    64.0
}
fn default_dwell_cycles() -> u32 {
    4
}
fn default_stale_after_ms() -> u64 {
    5_000
}
fn default_enter_threshold() -> f64 {
    0.6
}
fn default_exit_threshold() -> f64 {
    0.4
}
fn default_saturation_hold() -> f64 {
    0.85
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            interval_ms: default_interval_ms(),
            smoothing: default_smoothing(),
            sigmoid_midpoint: default_sigmoid_midpoint(),
            sigmoid_slope: default_sigmoid_slope(),
            gain: default_gain(),
            spread_active_request_weight: default_spread_active_request_weight(),
            dwell_cycles: default_dwell_cycles(),
            stale_after_ms: default_stale_after_ms(),
            enter_threshold: default_enter_threshold(),
            exit_threshold: default_exit_threshold(),
            slot_capacity: None,
            saturation_hold: default_saturation_hold(),
            overlap_score_credit: None,
            prefill_load_scale: None,
            decode_active_request_weight: None,
            router_temperature: None,
        }
    }
}

impl Parameters {
    pub(crate) fn validate(&self) -> Result<(), String> {
        fn unit(name: &str, v: f64) -> Result<(), String> {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                Ok(())
            } else {
                Err(format!("{name} must be within 0..=1, got {v}"))
            }
        }
        if self.interval_ms == 0 {
            return Err("interval_ms must be positive".into());
        }
        if self.stale_after_ms < self.interval_ms {
            return Err(format!(
                "stale_after_ms must be at least interval_ms ({}), got {}",
                self.interval_ms, self.stale_after_ms
            ));
        }
        unit("smoothing", self.smoothing)?;
        unit("sigmoid_midpoint", self.sigmoid_midpoint)?;
        if !(self.sigmoid_slope.is_finite() && self.sigmoid_slope > 0.0) {
            return Err("sigmoid_slope must be positive".into());
        }
        if !(self.gain.is_finite() && self.gain >= 0.0) {
            return Err("gain must be a finite non-negative number".into());
        }
        if !(self.spread_active_request_weight.is_finite()
            && self.spread_active_request_weight >= 0.0)
        {
            return Err(format!(
                "spread_active_request_weight must be a finite non-negative number, got {}",
                self.spread_active_request_weight
            ));
        }
        unit("enter_threshold", self.enter_threshold)?;
        unit("exit_threshold", self.exit_threshold)?;
        if self.exit_threshold >= self.enter_threshold {
            return Err("exit_threshold must be lower than enter_threshold".into());
        }
        if self.slot_capacity == Some(0) {
            return Err("slot_capacity must be positive".into());
        }
        unit("saturation_hold", self.saturation_hold)?;
        if self.saturation_hold == 0.0 {
            return Err("saturation_hold must be above 0".into());
        }
        for (name, v) in [
            ("overlap_score_credit", self.overlap_score_credit),
            ("prefill_load_scale", self.prefill_load_scale),
            (
                "decode_active_request_weight",
                self.decode_active_request_weight,
            ),
            ("router_temperature", self.router_temperature),
        ] {
            if let Some(v) = v
                && !(v.is_finite() && v >= 0.0)
            {
                return Err(format!(
                    "{name} must be a finite non-negative number, got {v}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn interval(&self) -> Duration {
        Duration::from_millis(self.interval_ms)
    }

    pub(crate) fn stale_after(&self) -> Duration {
        Duration::from_millis(self.stale_after_ms)
    }

    /// In-flight requests at which a worker counts as full, or `None` when the hold is off.
    pub(crate) fn saturation_requests(&self) -> Option<f64> {
        self.slot_capacity
            .map(|slots| f64::from(slots) * self.saturation_hold)
    }

    /// The router configuration with any baseline overrides from the YAML applied.
    pub(crate) fn effective_config(&self, config: &SelectionConfig) -> SelectionConfig {
        let mut c = config.clone();
        if let Some(v) = self.overlap_score_credit {
            c.overlap_score_credit = v;
        }
        if let Some(v) = self.prefill_load_scale {
            c.prefill_load_scale = v;
        }
        if let Some(v) = self.decode_active_request_weight {
            c.decode_active_request_weight = v;
        }
        if let Some(v) = self.router_temperature {
            c.router_temperature = v;
        }
        c
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let params: Parameters = parameters.deserialize()?;
    params
        .validate()
        .map_err(WorkerSelectionPolicyProviderError::new)?;

    Ok(Arc::new(
        move |config: &SelectionConfig, worker_type, _partition| {
            let config = &params.effective_config(config);
            let shared = Arc::new(state::SharedState::new(state::Weights::baseline(config)));
            let controller = controller::Controller::new(&params, state::Weights::baseline(config));
            // No need to keep the handle: the task stops on its own once this policy is dropped.
            let _ = ticker::start(Arc::clone(&shared), controller, params.interval());
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                vec![Box::new(scorer::AdaptiveScorer::new(
                    Arc::clone(&shared),
                    params.interval(),
                ))],
                Box::new(picker::AdaptivePicker::new(config.router_temperature)),
            )
        },
    ))
}

/// Register the `adaptive-imbalance` policy type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection("adaptive-imbalance", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use dynamo_kv_router::plugins::prelude::*;
    use dynamo_scheduler::RouterConfigOverride;
    use dynamo_scheduler::host::{WorkerSelectionInput, WorkerSelector};

    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(Parameters::default().validate().is_ok());
    }

    #[test]
    fn rejects_bad_parameters() {
        let d = Parameters::default;
        let cases = [
            (
                "zero sigmoid_slope",
                Parameters {
                    sigmoid_slope: 0.0,
                    ..d()
                },
            ),
            (
                "exit_threshold above enter_threshold",
                Parameters {
                    exit_threshold: 0.7,
                    ..d()
                },
            ),
            ("negative gain", Parameters { gain: -1.0, ..d() }),
            (
                "negative spread_active_request_weight",
                Parameters {
                    spread_active_request_weight: -1.0,
                    ..d()
                },
            ),
            (
                "stale_after_ms below interval_ms",
                Parameters {
                    interval_ms: 250,
                    stale_after_ms: 100,
                    ..Parameters::default()
                },
            ),
            (
                "zero interval_ms",
                Parameters {
                    interval_ms: 0,
                    ..d()
                },
            ),
            (
                "NaN smoothing",
                Parameters {
                    smoothing: f64::NAN,
                    ..d()
                },
            ),
            (
                "smoothing above 1",
                Parameters {
                    smoothing: 1.5,
                    ..d()
                },
            ),
            (
                "negative smoothing",
                Parameters {
                    smoothing: -0.1,
                    ..d()
                },
            ),
            (
                "zero slot_capacity",
                Parameters {
                    slot_capacity: Some(0),
                    ..d()
                },
            ),
            (
                "zero saturation_hold",
                Parameters {
                    saturation_hold: 0.0,
                    ..d()
                },
            ),
            (
                "saturation_hold above 1",
                Parameters {
                    saturation_hold: 1.5,
                    ..d()
                },
            ),
        ];
        for (name, params) in cases {
            assert!(params.validate().is_err(), "{name} must be rejected");
        }
    }

    fn parse(yaml: &str) -> Result<Parameters, String> {
        let params: Parameters = serde_yaml::from_str(yaml).map_err(|e| e.to_string())?;
        params.validate()?;
        Ok(params)
    }

    #[test]
    fn yaml_parameters_accept_defaults_and_reject_bad_values() {
        assert!(parse("{}").is_ok(), "empty mapping uses defaults");
        assert!(parse("sigmoid_slope: -1").is_err());
        assert!(
            parse("gane: 3").is_err(),
            "deny_unknown_fields must reject typos"
        );

        let with_baseline = parse("overlap_score_credit: 0.5\nrouter_temperature: 0.2")
            .expect("baseline overrides are optional parameters");
        let config = with_baseline.effective_config(&SelectionConfig::default());
        assert_eq!(config.overlap_score_credit, 0.5);
        assert_eq!(config.router_temperature, 0.2);
        assert_eq!(
            config.prefill_load_scale,
            SelectionConfig::default().prefill_load_scale,
            "absent overrides inherit the router configuration"
        );

        assert!(
            parse("prefill_load_scale: -1").is_err(),
            "baseline overrides must be finite and non-negative"
        );
    }

    #[test]
    fn yaml_spread_active_request_weight_has_a_default_and_an_override() {
        assert_eq!(parse("{}").unwrap().spread_active_request_weight, 64.0);
        assert_eq!(
            parse("spread_active_request_weight: 128")
                .unwrap()
                .spread_active_request_weight,
            128.0
        );
        assert!(parse("spread_active_request_weight: .nan").is_err());
    }

    #[test]
    fn yaml_slot_hold_is_off_by_default_and_set_from_slot_capacity() {
        let defaults = parse("{}").unwrap();
        assert_eq!(defaults.slot_capacity, None);
        assert_eq!(defaults.saturation_requests(), None);

        let held = parse("slot_capacity: 32").unwrap();
        assert_eq!(held.saturation_requests(), Some(32.0 * 0.85));
        let custom = parse("slot_capacity: 32\nsaturation_hold: 0.5").unwrap();
        assert_eq!(custom.saturation_requests(), Some(16.0));
        assert!(parse("slot_capacity: -1").is_err());
    }

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1024)
        }
    }

    /// Fixed picker seed, so sampled picks are the same on every run.
    const PICKER_SEED: u64 = 42;

    /// The adaptive scorer and picker wired into a policy, plus the state they share.
    fn adaptive_policy(
        config: SelectionConfig,
        interval: Duration,
    ) -> (Arc<state::SharedState>, WorkerSelectionPolicy) {
        let shared = Arc::new(state::SharedState::new(state::Weights::baseline(&config)));
        let temperature = config.router_temperature;
        let policy = WorkerSelectionPolicy::new(
            config,
            "test",
            vec![Box::new(scorer::AdaptiveScorer::new(
                Arc::clone(&shared),
                interval,
            ))],
            Box::new(picker::AdaptivePicker::with_seed(temperature, PICKER_SEED)),
        );
        (shared, policy)
    }

    /// A query-only request with a load projection for each listed worker and none for the rest.
    fn request_with_loads(loads: &[(WorkerWithDpRank, usize)]) -> SchedulingRequest {
        let mut request = SchedulingRequest {
            accounting_attribution: Default::default(),
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for &(worker, active_requests) in loads {
            request.worker_loads.insert(
                worker,
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        request
    }

    #[test]
    fn selects_the_less_loaded_worker_and_publishes_load() {
        let (shared, policy) = adaptive_policy(
            SelectionConfig {
                decode_active_request_weight: 1.0,
                router_temperature: 0.0,
                ..SelectionConfig::default()
            },
            Parameters::default().interval(),
        );

        let workers = HashMap::from([(41, TestWorker), (29, TestWorker)]);
        let idle = WorkerWithDpRank::from_worker_id(29);
        let busy = WorkerWithDpRank::from_worker_id(41);
        let request = request_with_loads(&[(idle, 0), (busy, 2)]);

        let selected = policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker, idle);

        let mut observed = shared.load_snapshot().active_requests.clone();
        observed.sort_by(f64::total_cmp);
        assert_eq!(
            observed,
            vec![0.0, 2.0],
            "the scorer feeds the loop's load sample"
        );
    }

    struct TaintedWorker(HashSet<String>);

    impl WorkerConfigLike for TaintedWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1024)
        }

        fn taints(&self) -> &HashSet<String> {
            &self.0
        }
    }

    #[test]
    fn preferred_taint_scales_the_cost_as_in_the_default_policy() {
        let (_, policy) = adaptive_policy(
            SelectionConfig {
                decode_active_request_weight: 1.0,
                router_temperature: 0.0,
                ..SelectionConfig::default()
            },
            Parameters::default().interval(),
        );

        let workers = HashMap::from([
            (41, TaintedWorker(HashSet::new())),
            (29, TaintedWorker(HashSet::from(["fast".to_string()]))),
        ]);
        let idle = WorkerWithDpRank::from_worker_id(41);
        let preferred = WorkerWithDpRank::from_worker_id(29);
        // Costs 1 (prefill) for the idle worker and 1 + 1 (prefill + one active request) for the
        // preferred one; the preference multiplies the latter by exp(-tanh(1)) ≈ 0.47, so it wins.
        let mut request = request_with_loads(&[(idle, 0), (preferred, 1)]);
        request
            .routing_constraints
            .preferred_taints
            .insert("fast".into(), 1.0);

        let selected = policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(selected.worker, preferred);
    }

    #[test]
    fn workers_without_a_load_projection_stay_out_of_the_load_sample() {
        let (shared, policy) = adaptive_policy(
            SelectionConfig {
                decode_active_request_weight: 1.0,
                router_temperature: 0.0,
                ..SelectionConfig::default()
            },
            Parameters::default().interval(),
        );

        let workers = HashMap::from([(7, TestWorker), (8, TestWorker), (9, TestWorker)]);
        let request = request_with_loads(&[
            (WorkerWithDpRank::from_worker_id(7), 3),
            (WorkerWithDpRank::from_worker_id(8), 1),
        ]);

        policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();

        let mut observed = shared.load_snapshot().active_requests.clone();
        observed.sort_by(f64::total_cmp);
        assert_eq!(
            observed,
            vec![1.0, 3.0],
            "worker 9 has no load observation, so it must not count as an idle worker"
        );
    }

    /// Selects on `workers` with one load projection per entry of `loads`.
    fn select(
        policy: &WorkerSelectionPolicy,
        workers: &HashMap<u64, TestWorker>,
        loads: &[(WorkerWithDpRank, usize)],
    ) -> WorkerWithDpRank {
        select_with_override(policy, workers, loads, None)
    }

    fn select_with_override(
        policy: &WorkerSelectionPolicy,
        workers: &HashMap<u64, TestWorker>,
        loads: &[(WorkerWithDpRank, usize)],
        router_config_override: Option<RouterConfigOverride>,
    ) -> WorkerWithDpRank {
        let mut request = request_with_loads(loads);
        request.router_config_override = router_config_override;
        policy
            .select_worker(WorkerSelectionInput::configured(
                workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap()
            .worker
    }

    #[test]
    fn load_sample_is_published_at_most_once_per_interval() {
        let (shared, policy) = adaptive_policy(
            SelectionConfig {
                decode_active_request_weight: 1.0,
                router_temperature: 0.0,
                ..SelectionConfig::default()
            },
            Duration::from_secs(3600),
        );
        let workers = HashMap::from([(1, TestWorker), (2, TestWorker)]);
        let (a, b) = (
            WorkerWithDpRank::from_worker_id(1),
            WorkerWithDpRank::from_worker_id(2),
        );

        select(&policy, &workers, &[(a, 1), (b, 2)]);
        let first = shared.load_snapshot();
        select(&policy, &workers, &[(a, 7), (b, 9)]);
        let second = shared.load_snapshot();

        assert!(
            Arc::ptr_eq(&first, &second),
            "the second request lands inside the interval and must not publish"
        );
        let mut observed = second.active_requests.clone();
        observed.sort_by(f64::total_cmp);
        assert_eq!(
            observed,
            vec![1.0, 2.0],
            "the first request publishes immediately"
        );
    }

    #[test]
    fn single_worker_request_does_not_replace_a_pool_sample() {
        let (shared, policy) = adaptive_policy(
            SelectionConfig {
                decode_active_request_weight: 1.0,
                router_temperature: 0.0,
                ..SelectionConfig::default()
            },
            // Zero interval: any request with a valid sample would publish.
            Duration::ZERO,
        );
        let (a, b) = (
            WorkerWithDpRank::from_worker_id(1),
            WorkerWithDpRank::from_worker_id(2),
        );
        let pool = HashMap::from([(1, TestWorker), (2, TestWorker)]);
        select(&policy, &pool, &[(a, 1), (b, 9)]);
        let pool_sample = shared.load_snapshot();

        let only_a = HashMap::from([(1, TestWorker)]);
        assert_eq!(select(&policy, &only_a, &[(a, 4)]), a);
        assert!(
            Arc::ptr_eq(&pool_sample, &shared.load_snapshot()),
            "a one-worker candidate set must not overwrite the two-worker sample"
        );

        select(&policy, &pool, &[(a, 3), (b, 5)]);
        let mut observed = shared.load_snapshot().active_requests.clone();
        observed.sort_by(f64::total_cmp);
        assert_eq!(
            observed,
            vec![3.0, 5.0],
            "the next two-worker request still publishes"
        );
    }

    #[test]
    fn per_request_temperature_override_wins_over_the_baseline_temperature() {
        // Baseline temperature 5 on normalized gaps 0, 0.5, 1 gives pick odds of roughly 37/33/30 %.
        // With PICKER_SEED the 20 sampled picks are fixed and include busier workers.
        let (_shared, policy) = adaptive_policy(
            SelectionConfig {
                decode_active_request_weight: 1.0,
                router_temperature: 5.0,
                ..SelectionConfig::default()
            },
            Parameters::default().interval(),
        );
        let workers = HashMap::from([(1, TestWorker), (2, TestWorker), (3, TestWorker)]);
        let idle = WorkerWithDpRank::from_worker_id(1);
        let loads = [
            (idle, 0),
            (WorkerWithDpRank::from_worker_id(2), 1),
            (WorkerWithDpRank::from_worker_id(3), 2),
        ];

        let sampled: Vec<_> = (0..20).map(|_| select(&policy, &workers, &loads)).collect();
        assert!(
            sampled.iter().any(|w| *w != idle),
            "without an override the baseline temperature samples: {sampled:?}"
        );

        let zero = RouterConfigOverride {
            router_temperature: Some(0.0),
            ..RouterConfigOverride::default()
        };
        for _ in 0..20 {
            assert_eq!(
                select_with_override(&policy, &workers, &loads, Some(zero.clone())),
                idle,
                "override 0 must be a deterministic argmin"
            );
        }
    }

    /// The host rejects an empty candidate set before calling the picker, so the picker's own
    /// empty check is a fallback that this harness cannot reach.
    #[test]
    fn no_eligible_worker_is_an_error() {
        let (_shared, policy) =
            adaptive_policy(SelectionConfig::default(), Parameters::default().interval());
        let workers: HashMap<u64, TestWorker> = HashMap::new();
        let request = request_with_loads(&[]);
        let result = policy.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ));
        assert!(
            matches!(result, Err(KvSchedulerError::NoEndpoints)),
            "{:?}",
            result.err()
        );
    }
}
