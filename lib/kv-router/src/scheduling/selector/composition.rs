// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::DefaultWorkerSelector;
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::scheduling::{RoutingEligibility, SchedulingRequest};

#[derive(Default)]
struct CandidateMask {
    rows: Vec<bool>,
}

impl CandidateMask {
    fn reset(&mut self, row_count: usize) {
        self.rows.clear();
        self.rows.resize(row_count, true);
    }

    fn remove(&mut self, row: usize) {
        self.rows[row] = false;
    }

    fn is_active(&self, row: usize) -> bool {
        self.rows.get(row).is_some_and(|active| *active)
    }

    fn active_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter_map(|(row, active)| (*active).then_some(row))
    }

    fn active_count(&self) -> usize {
        self.rows.iter().filter(|active| **active).count()
    }
}

#[derive(Default)]
struct SelectionScratch {
    // Compact identity column; policy stages address workers only by row index.
    worker_rows: Vec<WorkerWithDpRank>,
    active: CandidateMask,
    costs: Vec<f64>,
    probabilities: Vec<f64>,
}

impl SelectionScratch {
    fn reset<C: WorkerConfigLike>(
        &mut self,
        workers: &HashMap<WorkerId, C>,
        eligibility: RoutingEligibility<'_>,
    ) {
        self.worker_rows.clear();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            self.worker_rows.push(worker);
        });
        self.active.reset(self.worker_rows.len());
        self.costs.clear();
        self.costs.resize(self.worker_rows.len(), 0.0);
        self.probabilities.clear();
        self.probabilities.resize(self.worker_rows.len(), 0.0);
    }
}

struct SelectionInputs<'a, C> {
    workers: &'a HashMap<WorkerId, C>,
    request: &'a SchedulingRequest,
    rows: &'a [WorkerWithDpRank],
    block_size: u32,
}

impl<C> SelectionInputs<'_, C> {
    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn worker(&self, row: usize) -> WorkerWithDpRank {
        self.rows[row]
    }
}

trait WorkerFilter<C> {
    fn keep(&mut self, input: &SelectionInputs<'_, C>, row: usize) -> bool;
}

trait WorkerScorer<C> {
    fn prepare(&mut self, _input: &SelectionInputs<'_, C>, _active: &CandidateMask) {}

    fn cost(&mut self, input: &SelectionInputs<'_, C>, row: usize) -> f64;
}

trait WorkerPicker<C> {
    fn pick(
        &mut self,
        input: &SelectionInputs<'_, C>,
        active: &CandidateMask,
        costs: &[f64],
        probabilities: &mut [f64],
    ) -> Option<usize>;
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum PolicyError {
    #[error("scorers produced a non-finite cost for candidate row {row}")]
    NonFiniteCost { row: usize },

    #[error("picker returned row {row}, but only {row_count} rows exist")]
    PickOutOfRange { row: usize, row_count: usize },

    #[error("picker returned inactive candidate row {row}")]
    PickedInactive { row: usize },
}

struct PolicyPipeline<C> {
    filters: Vec<Box<dyn WorkerFilter<C>>>,
    scorers: Vec<Box<dyn WorkerScorer<C>>>,
    picker: Box<dyn WorkerPicker<C>>,
    scratch: SelectionScratch,
}

impl<C: WorkerConfigLike> PolicyPipeline<C> {
    fn new(
        filters: Vec<Box<dyn WorkerFilter<C>>>,
        scorers: Vec<Box<dyn WorkerScorer<C>>>,
        picker: Box<dyn WorkerPicker<C>>,
    ) -> Self {
        Self {
            filters,
            scorers,
            picker,
            scratch: SelectionScratch::default(),
        }
    }

    fn select(
        &mut self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<Option<(WorkerWithDpRank, f64)>, PolicyError> {
        self.scratch.reset(workers, eligibility);
        if self.scratch.worker_rows.is_empty() {
            return Ok(None);
        }

        let input = SelectionInputs {
            workers,
            request,
            rows: &self.scratch.worker_rows,
            block_size,
        };

        for filter in &mut self.filters {
            for row in 0..input.row_count() {
                if self.scratch.active.is_active(row) && !filter.keep(&input, row) {
                    self.scratch.active.remove(row);
                }
            }
        }
        if self.scratch.active.active_count() == 0 {
            return Ok(None);
        }

        for scorer in &mut self.scorers {
            scorer.prepare(&input, &self.scratch.active);
            for row in self.scratch.active.active_rows() {
                let cost = self.scratch.costs[row] + scorer.cost(&input, row);
                if !cost.is_finite() {
                    return Err(PolicyError::NonFiniteCost { row });
                }
                self.scratch.costs[row] = cost;
            }
        }

        let Some(row) = self.picker.pick(
            &input,
            &self.scratch.active,
            &self.scratch.costs,
            &mut self.scratch.probabilities,
        ) else {
            return Ok(None);
        };
        if row >= input.row_count() {
            return Err(PolicyError::PickOutOfRange {
                row,
                row_count: input.row_count(),
            });
        }
        if !self.scratch.active.is_active(row) {
            return Err(PolicyError::PickedInactive { row });
        }

        Ok(Some((input.worker(row), self.scratch.costs[row])))
    }
}

struct DefaultCostScorer {
    selector: DefaultWorkerSelector,
    weights: Option<super::LogitWeights>,
    min_active_prefill_tokens: usize,
}

impl DefaultCostScorer {
    fn new(selector: DefaultWorkerSelector) -> Self {
        Self {
            selector,
            weights: None,
            min_active_prefill_tokens: 0,
        }
    }
}

impl<C: WorkerConfigLike> WorkerScorer<C> for DefaultCostScorer {
    fn prepare(&mut self, input: &SelectionInputs<'_, C>, active: &CandidateMask) {
        let weights = self.selector.selection_weights(input.request);
        self.min_active_prefill_tokens =
            if input.request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                active
                    .active_rows()
                    .map(|row| {
                        input
                            .request
                            .worker_load_for(input.worker(row))
                            .active_prefill_tokens
                    })
                    .min()
                    .expect("active candidate rows non-empty")
            } else {
                0
            };
        self.weights = Some(weights);
    }

    fn cost(&mut self, input: &SelectionInputs<'_, C>, row: usize) -> f64 {
        self.selector.score_worker(
            input.workers,
            input.request,
            input.worker(row),
            input.block_size,
            self.min_active_prefill_tokens,
            self.weights.expect("scorer prepared before use"),
        )
    }
}

struct DefaultPicker {
    default_temperature: f64,
    rng: Option<fastrand::Rng>,
}

impl DefaultPicker {
    fn new(default_temperature: f64) -> Self {
        Self {
            default_temperature,
            rng: None,
        }
    }

    #[cfg(test)]
    fn new_seeded(default_temperature: f64, seed: u64) -> Self {
        Self {
            default_temperature,
            rng: Some(fastrand::Rng::with_seed(seed)),
        }
    }

    fn sample(&mut self) -> f64 {
        self.rng
            .as_mut()
            .map_or_else(fastrand::f64, fastrand::Rng::f64)
    }

    fn choose_tie(&mut self, tie_count: usize) -> bool {
        self.rng.as_mut().map_or_else(
            || fastrand::usize(0..tie_count) == 0,
            |rng| rng.usize(0..tie_count) == 0,
        )
    }
}

impl<C> WorkerPicker<C> for DefaultPicker {
    fn pick(
        &mut self,
        input: &SelectionInputs<'_, C>,
        active: &CandidateMask,
        costs: &[f64],
        probabilities: &mut [f64],
    ) -> Option<usize> {
        let temperature = input
            .request
            .router_config_override
            .as_ref()
            .and_then(|config| config.router_temperature)
            .unwrap_or(self.default_temperature);

        if temperature == 0.0 {
            let mut best_row = None;
            let mut best_cost = f64::INFINITY;
            let mut tie_count = 0;
            for row in active.active_rows() {
                if costs[row] < best_cost {
                    best_row = Some(row);
                    best_cost = costs[row];
                    tie_count = 1;
                    continue;
                }
                if costs[row] == best_cost {
                    tie_count += 1;
                    if self.choose_tie(tie_count) {
                        best_row = Some(row);
                    }
                }
            }
            return best_row;
        }

        let (min_cost, max_cost) = active.active_rows().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), row| (minimum.min(costs[row]), maximum.max(costs[row])),
        );
        if min_cost == max_cost {
            let probability = 1.0 / active.active_count() as f64;
            for row in active.active_rows() {
                probabilities[row] = probability;
            }
        } else {
            let scale = -1.0 / ((max_cost - min_cost) * temperature);
            let max_scaled = min_cost * scale;
            for row in active.active_rows() {
                probabilities[row] = (costs[row] * scale - max_scaled).exp();
            }
        }

        let sum: f64 = active.active_rows().map(|row| probabilities[row]).sum();
        for row in active.active_rows() {
            probabilities[row] /= sum;
        }

        let sample = self.sample();
        let mut cumulative = 0.0;
        let mut last_row = None;
        for row in active.active_rows() {
            last_row = Some(row);
            cumulative += probabilities[row];
            if sample <= cumulative {
                return Some(row);
            }
        }
        last_row
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::KvRouterConfig;
    use crate::protocols::RoutingConstraints;
    use crate::scheduling::selector::WorkerSelector;
    use crate::scheduling::{OverlapSignals, ScheduleMode};
    use crate::sequences::WorkerLoadProjection;

    #[derive(Default)]
    struct TestWorkerConfig {
        dp_size: u32,
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for TestWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            self.dp_size.max(1)
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1_024)
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    fn request() -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 128,
            lora_name: None,
            expected_output_tokens: Some(32),
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            overlap: OverlapSignals::default(),
            shared_cache_hits: None,
            worker_loads: FxHashMap::default(),
            resp_tx: None,
        }
    }

    struct RemoveWorker(WorkerId);

    impl WorkerFilter<TestWorkerConfig> for RemoveWorker {
        fn keep(&mut self, input: &SelectionInputs<'_, TestWorkerConfig>, row: usize) -> bool {
            input.worker(row).worker_id != self.0
        }
    }

    struct WorkerIdCost;

    impl WorkerScorer<TestWorkerConfig> for WorkerIdCost {
        fn cost(&mut self, input: &SelectionInputs<'_, TestWorkerConfig>, row: usize) -> f64 {
            input.worker(row).worker_id as f64
        }
    }

    struct WorkerBonus {
        worker_id: WorkerId,
        cost: f64,
    }

    impl WorkerScorer<TestWorkerConfig> for WorkerBonus {
        fn cost(&mut self, input: &SelectionInputs<'_, TestWorkerConfig>, row: usize) -> f64 {
            if input.worker(row).worker_id == self.worker_id {
                self.cost
            } else {
                0.0
            }
        }
    }

    struct HighestWorkerIdPicker;

    impl WorkerPicker<TestWorkerConfig> for HighestWorkerIdPicker {
        fn pick(
            &mut self,
            input: &SelectionInputs<'_, TestWorkerConfig>,
            active: &CandidateMask,
            _costs: &[f64],
            _probabilities: &mut [f64],
        ) -> Option<usize> {
            active
                .active_rows()
                .max_by_key(|row| input.worker(*row).worker_id)
        }
    }

    #[test]
    fn composes_two_filters_additive_costs_and_custom_picker() {
        let workers = (1..=4)
            .map(|worker_id| (worker_id, TestWorkerConfig::default()))
            .collect();
        let request = request();
        let mut pipeline = PolicyPipeline::new(
            vec![Box::new(RemoveWorker(1)), Box::new(RemoveWorker(4))],
            vec![
                Box::new(WorkerIdCost),
                Box::new(WorkerBonus {
                    worker_id: 3,
                    cost: -10.0,
                }),
            ],
            Box::new(HighestWorkerIdPicker),
        );

        let (worker, cost) = pipeline
            .select(&workers, &request, request.eligibility(), 16)
            .unwrap()
            .unwrap();

        assert_eq!(worker.worker_id, 3);
        assert_eq!(cost, -7.0);
    }

    struct RejectAll;

    impl WorkerFilter<TestWorkerConfig> for RejectAll {
        fn keep(&mut self, _input: &SelectionInputs<'_, TestWorkerConfig>, _row: usize) -> bool {
            false
        }
    }

    struct PanicPicker;

    impl WorkerPicker<TestWorkerConfig> for PanicPicker {
        fn pick(
            &mut self,
            _input: &SelectionInputs<'_, TestWorkerConfig>,
            _active: &CandidateMask,
            _costs: &[f64],
            _probabilities: &mut [f64],
        ) -> Option<usize> {
            panic!("picker must not run with an empty mask")
        }
    }

    #[test]
    fn empty_filter_result_skips_scoring_and_picking() {
        let workers = HashMap::from([(1, TestWorkerConfig::default())]);
        let request = request();
        let mut pipeline =
            PolicyPipeline::new(vec![Box::new(RejectAll)], Vec::new(), Box::new(PanicPicker));

        assert_eq!(
            pipeline
                .select(&workers, &request, request.eligibility(), 16)
                .unwrap(),
            None
        );
    }

    struct FixedPicker(usize);

    impl WorkerPicker<TestWorkerConfig> for FixedPicker {
        fn pick(
            &mut self,
            _input: &SelectionInputs<'_, TestWorkerConfig>,
            _active: &CandidateMask,
            _costs: &[f64],
            _probabilities: &mut [f64],
        ) -> Option<usize> {
            Some(self.0)
        }
    }

    struct RemovedWorkerPicker(WorkerId);

    impl WorkerPicker<TestWorkerConfig> for RemovedWorkerPicker {
        fn pick(
            &mut self,
            input: &SelectionInputs<'_, TestWorkerConfig>,
            _active: &CandidateMask,
            _costs: &[f64],
            _probabilities: &mut [f64],
        ) -> Option<usize> {
            input
                .rows
                .iter()
                .position(|worker| worker.worker_id == self.0)
        }
    }

    #[test]
    fn validates_picker_row_before_mapping_it_to_a_worker() {
        let workers = HashMap::from([
            (1, TestWorkerConfig::default()),
            (2, TestWorkerConfig::default()),
        ]);
        let request = request();
        let mut out_of_range =
            PolicyPipeline::new(Vec::new(), Vec::new(), Box::new(FixedPicker(usize::MAX)));
        assert_eq!(
            out_of_range
                .select(&workers, &request, request.eligibility(), 16)
                .unwrap_err(),
            PolicyError::PickOutOfRange {
                row: usize::MAX,
                row_count: 2,
            }
        );

        let mut inactive = PolicyPipeline::new(
            vec![Box::new(RemoveWorker(1))],
            Vec::new(),
            Box::new(RemovedWorkerPicker(1)),
        );
        let error = inactive
            .select(&workers, &request, request.eligibility(), 16)
            .unwrap_err();
        assert!(matches!(error, PolicyError::PickedInactive { .. }));
    }

    struct NonFiniteScorer;

    impl WorkerScorer<TestWorkerConfig> for NonFiniteScorer {
        fn cost(&mut self, _input: &SelectionInputs<'_, TestWorkerConfig>, _row: usize) -> f64 {
            f64::NAN
        }
    }

    #[test]
    fn rejects_non_finite_additive_costs() {
        let workers = HashMap::from([(1, TestWorkerConfig::default())]);
        let request = request();
        let mut pipeline = PolicyPipeline::new(
            Vec::new(),
            vec![Box::new(NonFiniteScorer)],
            Box::new(FixedPicker(0)),
        );

        assert!(matches!(
            pipeline
                .select(&workers, &request, request.eligibility(), 16)
                .unwrap_err(),
            PolicyError::NonFiniteCost { row: 0 }
        ));
    }

    #[test]
    fn default_composition_matches_seeded_selector() {
        for temperature in [0.0, 0.7] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..Default::default()
            };
            let selector = DefaultWorkerSelector::new_seeded(Some(config.clone()), "test", 42);
            let mut pipeline = PolicyPipeline::new(
                Vec::new(),
                vec![Box::new(DefaultCostScorer::new(
                    DefaultWorkerSelector::new(Some(config.clone()), "test"),
                ))],
                Box::new(DefaultPicker::new_seeded(temperature, 42)),
            );
            let workers = HashMap::from([(
                10,
                TestWorkerConfig {
                    dp_size: 3,
                    ..Default::default()
                },
            )]);
            let mut request = request();
            request.worker_loads = (0..3)
                .map(|dp_rank| {
                    (
                        WorkerWithDpRank::new(10, dp_rank),
                        WorkerLoadProjection {
                            active_decode_blocks: dp_rank as usize * 3,
                            active_requests: dp_rank as usize,
                            ..Default::default()
                        },
                    )
                })
                .collect();

            for _ in 0..64 {
                let expected = selector
                    .select_worker(&workers, &request, request.eligibility(), 16)
                    .unwrap()
                    .worker;
                let actual = pipeline
                    .select(&workers, &request, request.eligibility(), 16)
                    .unwrap()
                    .unwrap()
                    .0;
                assert_eq!(actual, expected, "temperature={temperature}");
            }
        }
    }

    #[test]
    fn reuses_worker_mask_cost_and_probability_storage() {
        let workers = (0..32)
            .map(|worker_id| (worker_id, TestWorkerConfig::default()))
            .collect();
        let request = request();
        let mut pipeline = PolicyPipeline::new(
            Vec::new(),
            vec![Box::new(WorkerIdCost)],
            Box::new(DefaultPicker::new(0.0)),
        );
        pipeline
            .select(&workers, &request, request.eligibility(), 16)
            .unwrap();
        let capacities = (
            pipeline.scratch.worker_rows.capacity(),
            pipeline.scratch.active.rows.capacity(),
            pipeline.scratch.costs.capacity(),
            pipeline.scratch.probabilities.capacity(),
        );

        pipeline
            .select(&workers, &request, request.eligibility(), 16)
            .unwrap();
        assert_eq!(
            capacities,
            (
                pipeline.scratch.worker_rows.capacity(),
                pipeline.scratch.active.rows.capacity(),
                pipeline.scratch.costs.capacity(),
                pipeline.scratch.probabilities.capacity(),
            )
        );
    }
}
