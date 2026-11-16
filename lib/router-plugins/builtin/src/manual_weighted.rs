// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{DefaultHasher, Hasher};
use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

const POLICY_TYPE: &str = "dynamo-manual-weighted";

#[derive(Default)]
struct WeightedPicker {
    weights: WorkerWeights,
}

impl WorkerPicker for WeightedPicker {
    fn worker_weights(&mut self) -> Option<&mut WorkerWeights> {
        Some(&mut self.weights)
    }

    fn pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let seed = fastrand::u64(..);
        let mut selected = None;
        let mut best = f64::INFINITY;
        let mut ties = 0;
        for (row, candidate) in input.candidates().iter().enumerate() {
            let worker = candidate.worker().worker_id;
            let weight = self.weights.weight(worker);
            if weight == 0.0 {
                continue;
            }
            // Every rank of a worker shares one exponential-race draw. Extra eligible
            // ranks therefore do not multiply its share. Log space avoids weight overflow.
            let mut hash = DefaultHasher::new();
            hash.write_u64(seed);
            hash.write_u64(worker);
            let uniform = ((hash.finish() >> 12) as f64 + 0.5) / 4_503_599_627_370_496.0;
            let score = (-uniform.ln()).ln() - weight.ln();
            if score < best {
                best = score;
                selected = Some(row);
                ties = 1;
            } else if score == best {
                ties += 1;
                if fastrand::usize(..ties) == 0 {
                    selected = Some(row);
                }
            }
        }
        selected.ok_or(WorkerSelectionPolicyError::NoEligibleWorkers)
    }
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RouterPluginRegistryError> {
    registry.register_worker_selection(
        POLICY_TYPE,
        Arc::new(|parameters| {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct NoParameters {}
            let _: NoParameters = parameters.deserialize()?;
            Ok(Arc::new(|config, role, _partition| {
                WorkerSelectionPolicy::new(
                    config.clone(),
                    role.as_str(),
                    Vec::new(),
                    Box::<WeightedPicker>::default(),
                )
            }))
        }),
    )?;
    Ok(())
}
