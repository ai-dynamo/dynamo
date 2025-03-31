// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TODO(#400):
// Instead of passing in a block_size, we should get this data from the backend component's config.
// What changes need to be made:
// 1. Take as an argument the name of the backend component.
// 2. Update the backend component to produce a config in a standard location.
// 3. Update the KvRouter to read the config from the backend component.

use std::collections::HashMap;

use clap::Parser;

use dynamo_llm::kv_router::{
    protocols::WorkerSelectionResult,
    scheduler::{KvSchedulerError, SchedulingRequest},
    scoring::ProcessedEndpoints,
    KvRouter, WorkerSelector,
};
use dynamo_runtime::{
    logging, pipeline::network::Ingress, DistributedRuntime, Result, Runtime, Worker,
};
use rand::Rng;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Namespace for the distributed component
    #[arg(long)]
    namespace: String,

    /// Component name for the service
    #[arg(long, default_value = "kv_aware_router")]
    component: String,

    /// Block size for the router
    #[arg(long)]
    block_size: usize,
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let args = Args::parse();
    let runtime = DistributedRuntime::from_settings(runtime).await?;

    let component = runtime
        .namespace(&args.namespace)?
        .component(&args.component)?;

    let selector = Box::new(CustomWorkerSelector);

    let router = KvRouter::new(component.clone(), args.block_size, Some(selector)).await?;
    let router = Ingress::for_engine(router)?;

    component
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        .handler(router)
        .start()
        .await
}

pub struct CustomWorkerSelector;

impl WorkerSelector for CustomWorkerSelector {
    fn select_worker(
        &self,
        workers: &ProcessedEndpoints,
        request: &SchedulingRequest,
        block_size: usize,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let mut worker_scores = HashMap::new();
        let mut max_active = 0.0;

        // Calculate worker scores and find max waiting requests
        for (worker_id, ep) in workers.endpoints.iter() {
            // Calculate score similar to Python version
            if let Some(score) = request.overlap.scores.get(&worker_id) {
                let score = *score as f64 * block_size as f64 / request.isl_tokens as f64;
                worker_scores.insert(worker_id, score);
            }

            // Track max waiting requests
            max_active = f64::max(max_active, ep.data.request_active_slots as f64);
        }

        // Calculate logits for each worker
        let mut best_logit = f64::NEG_INFINITY;
        let mut best_workers = Vec::new();

        for (worker_id, ep) in workers.endpoints.iter() {
            let worker_id = *worker_id;

            // Get score or default to 0.0
            let score = worker_scores.get(&worker_id).copied().unwrap_or(0.0);

            // Calculate normalized metrics
            let gpu_cache_usage = ep.data.kv_active_blocks as f64 / ep.data.kv_total_blocks as f64;
            let normalized_active = if max_active > 0.0 {
                ep.data.request_active_slots as f64 / max_active
            } else {
                0.0
            };

            // Calculate logit using same formula as Python
            let logit = 2.0 * score - gpu_cache_usage - normalized_active;

            tracing::info!(
                "Formula for {}: {:.3} = 2.0 * {:.3} - {:.3} - {:.3}",
                worker_id,
                logit,
                score,
                gpu_cache_usage,
                normalized_active
            );

            // Track best workers
            match logit.partial_cmp(&best_logit) {
                Some(std::cmp::Ordering::Greater) => {
                    best_logit = logit;
                    best_workers.clear();
                    best_workers.push(worker_id);
                }
                Some(std::cmp::Ordering::Equal) => {
                    best_workers.push(worker_id);
                }
                _ => {}
            }
        }

        // Return early if no valid workers found
        if best_workers.is_empty() || best_logit == 0.0 {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let worker_id = if best_workers.len() == 1 {
            best_workers[0]
        } else {
            // Randomly select from best workers
            let mut rng = rand::rng();
            best_workers[rng.random_range(0..best_workers.len())]
        };

        // Log selection metrics
        tracing::info!("Selected worker: {}, logit: {:.3}", worker_id, best_logit);

        let total_blocks = std::cmp::min(request.isl_tokens / block_size, 1) as u64;
        let overlap_blocks = request.overlap.scores.get(&worker_id).copied().unwrap_or(0) as usize;

        Ok(WorkerSelectionResult {
            worker_id,
            required_blocks: total_blocks,
            overlap_blocks,
        })
    }
}
