// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static heterogeneous topology for interactive aggregated replay.
//!
//! Public worker IDs are authored identities. The replay engine assigns its
//! own dense indices after sorting pools and workers; callers must never infer
//! pool membership from either ID space.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::common::protocols::{EngineType, G1Backend, MockEngineArgs, WorkerType};
use crate::replay::ReplayWorkerLifecycleStatus;

pub const DEFAULT_REPLAY_POOL_ID: &str = "default";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolRouter {
    RoundRobin,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerTarget {
    pub pool_id: String,
    pub worker_id: usize,
    pub dp_rank: usize,
}

impl WorkerTarget {
    pub fn new(pool_id: impl Into<String>, worker_id: usize, dp_rank: usize) -> Self {
        Self {
            pool_id: pool_id.into(),
            worker_id,
            dp_rank,
        }
    }

    pub fn default_pool(worker_id: usize, dp_rank: usize) -> Self {
        Self::new(DEFAULT_REPLAY_POOL_ID, worker_id, dp_rank)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    #[serde(default)]
    pub max_num_seqs: Option<usize>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub taints: Vec<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default = "default_true")]
    /// Serving eligibility at session start. `false` with `draining=false`
    /// means static-inactive: provisioned and billed, but never starts and is
    /// never eligible for placement.
    pub active: bool,
    #[serde(default)]
    pub draining: bool,
}

const fn default_true() -> bool {
    true
}

impl WorkerSpec {
    pub fn active(worker_id: usize) -> Self {
        Self {
            worker_id,
            max_num_seqs: None,
            tags: Vec::new(),
            taints: Vec::new(),
            capabilities: Vec::new(),
            active: true,
            draining: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSpec {
    pub pool_id: String,
    pub engine_args: MockEngineArgs,
    pub workers: Vec<WorkerSpec>,
    #[serde(default = "default_pool_router")]
    pub router: PoolRouter,
}

const fn default_pool_router() -> PoolRouter {
    PoolRouter::RoundRobin
}

#[derive(Debug, Clone)]
pub(in crate::replay::offline) struct ResolvedPoolWorker {
    pub target: WorkerTarget,
    pub engine_args: MockEngineArgs,
    pub tags: BTreeSet<String>,
    pub taints: BTreeSet<String>,
    pub capabilities: BTreeSet<String>,
    pub active: bool,
    pub draining: bool,
}

impl ResolvedPoolWorker {
    pub fn lifecycle_status(&self) -> ReplayWorkerLifecycleStatus {
        if self.draining {
            ReplayWorkerLifecycleStatus::Draining
        } else if self.active {
            ReplayWorkerLifecycleStatus::Active
        } else {
            ReplayWorkerLifecycleStatus::StaticInactive
        }
    }
}

#[derive(Debug, Clone)]
pub(in crate::replay::offline) struct ResolvedPoolTopology {
    pub pool_routers: Vec<(String, PoolRouter)>,
    pub workers: Vec<ResolvedPoolWorker>,
}

impl ResolvedPoolTopology {
    pub fn resolve(mut pools: Vec<PoolSpec>, trace_block_size: usize) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("interactive replay trace_block_size must be greater than zero");
        }
        if pools.is_empty() {
            bail!("interactive replay topology requires at least one pool");
        }

        pools.sort_by(|left, right| left.pool_id.cmp(&right.pool_id));
        let mut pool_ids = BTreeSet::new();
        let mut worker_ids = BTreeSet::new();
        let mut pool_routers = Vec::with_capacity(pools.len());
        let mut workers = Vec::new();

        for mut pool in pools {
            if pool.pool_id.trim().is_empty() {
                bail!("interactive replay pool_id must not be empty");
            }
            if !pool_ids.insert(pool.pool_id.clone()) {
                bail!("interactive replay duplicate pool_id {:?}", pool.pool_id);
            }
            if pool.workers.is_empty() {
                bail!(
                    "interactive replay pool {:?} requires at least one worker",
                    pool.pool_id
                );
            }

            let args = pool.engine_args.normalized()?;
            validate_pool_args(&pool.pool_id, &args, trace_block_size)?;
            if !args.worker_max_num_seqs.is_empty() || !args.worker_taints.is_empty() {
                bail!(
                    "interactive replay pool {:?} must configure per-worker capacity and taints through WorkerSpec",
                    pool.pool_id
                );
            }

            pool.workers.sort_by_key(|worker| worker.worker_id);
            for worker in pool.workers {
                if !worker_ids.insert((pool.pool_id.clone(), worker.worker_id)) {
                    bail!(
                        "interactive replay pool {:?} duplicates worker_id {}",
                        pool.pool_id,
                        worker.worker_id,
                    );
                }
                if worker.max_num_seqs == Some(0) {
                    bail!(
                        "interactive replay pool {:?} worker {} max_num_seqs must be positive",
                        pool.pool_id,
                        worker.worker_id
                    );
                }

                let tags = validate_labels(&pool.pool_id, worker.worker_id, "tag", worker.tags)?;
                let taints =
                    validate_labels(&pool.pool_id, worker.worker_id, "taint", worker.taints)?;
                let capabilities = validate_labels(
                    &pool.pool_id,
                    worker.worker_id,
                    "capability",
                    worker.capabilities,
                )?;
                let mut worker_args = args.clone();
                if let Some(max_num_seqs) = worker.max_num_seqs {
                    worker_args.max_num_seqs = Some(max_num_seqs);
                }
                workers.push(ResolvedPoolWorker {
                    target: WorkerTarget::new(pool.pool_id.clone(), worker.worker_id, 0),
                    engine_args: worker_args,
                    tags,
                    taints,
                    capabilities,
                    active: worker.active && !worker.draining,
                    draining: worker.draining,
                });
            }
            pool_routers.push((pool.pool_id, pool.router));
        }

        if !workers.iter().any(|worker| worker.active) {
            bail!("interactive replay topology requires at least one active, non-draining worker");
        }

        Ok(Self {
            pool_routers,
            workers,
        })
    }
}

fn validate_pool_args(pool_id: &str, args: &MockEngineArgs, trace_block_size: usize) -> Result<()> {
    if args.engine_type != EngineType::Vllm {
        bail!("interactive replay pool {pool_id:?} supports only the vLLM mock engine");
    }
    if args.worker_type != WorkerType::Aggregated {
        bail!("interactive replay pool {pool_id:?} supports only aggregated workers");
    }
    if args.dp_size != 1 {
        bail!(
            "interactive replay pool {pool_id:?} requires dp_size=1, got {}",
            args.dp_size
        );
    }
    if args.resolved_g1_backend() != G1Backend::Native {
        bail!("interactive replay pool {pool_id:?} does not support KVBM/offload");
    }
    if args.block_size != trace_block_size {
        bail!(
            "authoritative interactive replay pool {pool_id:?} block_size {} does not match trace_block_size {trace_block_size}",
            args.block_size
        );
    }
    if args.startup_time.is_some_and(|seconds| seconds != 0.0) {
        bail!(
            "interactive replay P0 topology is static; pool {pool_id:?} must not configure startup_time"
        );
    }
    Ok(())
}

fn validate_labels(
    pool_id: &str,
    worker_id: usize,
    kind: &str,
    values: Vec<String>,
) -> Result<BTreeSet<String>> {
    let mut labels = BTreeSet::new();
    for value in values {
        if value.is_empty() || value.trim() != value {
            bail!(
                "interactive replay pool {pool_id:?} worker {worker_id} has an empty or untrimmed {kind}"
            );
        }
        if !labels.insert(value.clone()) {
            bail!(
                "interactive replay pool {pool_id:?} worker {worker_id} duplicates {kind} {value:?}"
            );
        }
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(speedup_ratio: f64, blocks: usize) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(blocks)
            .speedup_ratio(speedup_ratio)
            .build()
            .unwrap()
    }

    #[test]
    fn topology_is_canonical_and_membership_is_explicit() -> Result<()> {
        let resolved = ResolvedPoolTopology::resolve(
            vec![
                PoolSpec {
                    pool_id: "slow".to_string(),
                    engine_args: args(0.0, 32),
                    workers: vec![WorkerSpec::active(20)],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "fast".to_string(),
                    engine_args: args(2.0, 64),
                    workers: vec![WorkerSpec::active(99), WorkerSpec::active(10)],
                    router: PoolRouter::RoundRobin,
                },
            ],
            4,
        )?;
        assert_eq!(
            resolved
                .workers
                .iter()
                .map(|worker| (worker.target.pool_id.as_str(), worker.target.worker_id))
                .collect::<Vec<_>>(),
            [("fast", 10), ("fast", 99), ("slow", 20)]
        );
        assert_eq!(resolved.workers[0].engine_args.num_gpu_blocks, 64);
        assert_eq!(resolved.workers[2].engine_args.num_gpu_blocks, 32);
        Ok(())
    }

    #[test]
    fn topology_rejects_duplicate_workers_and_represents_ineligible_workers() {
        let duplicate = ResolvedPoolTopology::resolve(
            vec![PoolSpec {
                pool_id: "a".to_string(),
                engine_args: args(0.0, 32),
                workers: vec![WorkerSpec::active(7), WorkerSpec::active(7)],
                router: PoolRouter::RoundRobin,
            }],
            4,
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("duplicates worker_id 7"));

        let mut inactive = WorkerSpec::active(3);
        inactive.active = false;
        let resolved = ResolvedPoolTopology::resolve(
            vec![
                PoolSpec {
                    pool_id: "a".to_string(),
                    engine_args: args(0.0, 32),
                    workers: vec![inactive, WorkerSpec::active(4)],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "b".to_string(),
                    engine_args: args(1.0, 64),
                    workers: vec![WorkerSpec::active(3)],
                    router: PoolRouter::RoundRobin,
                },
            ],
            4,
        )
        .unwrap();
        assert!(!resolved.workers[0].active);
        assert!(!resolved.workers[0].draining);
        assert_eq!(
            resolved.workers[0].lifecycle_status(),
            ReplayWorkerLifecycleStatus::StaticInactive
        );
        assert!(resolved.workers[2].active);
        assert_eq!(
            resolved.workers[2].lifecycle_status(),
            ReplayWorkerLifecycleStatus::Active
        );
    }
}
