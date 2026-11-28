// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An opt-in correction for the extra prefill steps in an observed suffix.
//! A background thread builds and publishes per-rank profiles. Selection reads immutable data.

use super::{PolicyParameters, picker, role_parameters, scorer};
use dynamo_kv_router::plugins::prelude::*;
use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

const NAME: &str = "dynamo-prefill-chunk-table";

#[derive(Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Profiles {
    profiles: Vec<RankProfile>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RankProfile {
    worker_id: u64,
    dp_rank: u32,
    /// Measured fixed cost per additional step, in equivalent prefill blocks.
    step_cost_blocks: f64,
    /// Each path lists cumulative token counts through the end of a cold prefill.
    paths: Vec<Vec<usize>>,
}

type Table = HashMap<WorkerWithDpRank, HashMap<usize, PathCost>>;

struct PathCost {
    ends: Vec<usize>,
    step_cost_blocks: f64,
}

impl PathCost {
    fn extra_cost(&self, prefix: usize) -> f64 {
        let first = if prefix == 0 {
            0
        } else if let Ok(end) = self.ends.binary_search(&prefix) {
            end + 1
        } else {
            return 0.0;
        };
        self.ends.len().saturating_sub(first + 1) as f64 * self.step_cost_blocks
    }
}

impl Profiles {
    fn compile(&self) -> Result<Table, String> {
        let mut ranks = Table::new();
        for profile in &self.profiles {
            if !profile.step_cost_blocks.is_finite()
                || profile.step_cost_blocks < 0.0
                || !(profile.step_cost_blocks * 31.0).is_finite()
            {
                return Err("invalid prefill step cost".into());
            }
            if profile.paths.len() > 64 {
                return Err("at most 64 observed paths per rank".into());
            }
            let mut paths = HashMap::new();
            for ends in &profile.paths {
                let Some(&total) = ends.last() else {
                    return Err("empty chunk path".into());
                };
                if ends.len() > 32 || ends[0] == 0 || ends.windows(2).any(|p| p[0] >= p[1]) {
                    return Err("expected 1 to 32 strictly increasing positive chunk ends".into());
                }
                let path = PathCost {
                    ends: ends.clone(),
                    step_cost_blocks: profile.step_cost_blocks,
                };
                if paths.insert(total, path).is_some() {
                    return Err("ambiguous chunk path for one prompt length".into());
                }
            }
            let rank = WorkerWithDpRank::new(profile.worker_id, profile.dp_rank);
            if ranks.insert(rank, paths).is_some() {
                return Err("duplicate worker rank".into());
            }
        }
        Ok(ranks)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    profile_file: PathBuf,
}

#[derive(Deserialize)]
struct Publication {
    model_name: String,
    routing_group: String,
    block_size: u32,
    valid_until_unix_ms: u64,
    profiles: Vec<RankProfile>,
}

struct Snapshot {
    partition: RoutingPartitionId,
    block_size: u32,
    expires: SystemTime,
    ranks: Table,
}

impl Publication {
    fn compile(self) -> Result<Snapshot, String> {
        let expires = SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_millis(self.valid_until_unix_ms))
            .ok_or("profile deadline out of range")?;
        Ok(Snapshot {
            partition: RoutingPartitionId::new(self.model_name, self.routing_group),
            block_size: self.block_size,
            expires,
            ranks: Profiles {
                profiles: self.profiles,
            }
            .compile()?,
        })
    }
}

type Published = Arc<RwLock<Option<Snapshot>>>;

fn observe(path: PathBuf) -> std::io::Result<Published> {
    let table = Arc::new(RwLock::new(None));
    let weak = Arc::downgrade(&table);
    std::thread::Builder::new()
        .name("prefill-chunk-table".into())
        .spawn(move || {
            let mut previous_error = None;
            let mut previous_bytes = None;
            while let Some(table) = weak.upgrade() {
                let bytes = std::fs::read(&path);
                if let Ok(ref bytes) = bytes
                    && previous_bytes.as_ref() == Some(bytes)
                {
                    drop(table);
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
                let next = bytes.map_err(|error| error.to_string()).and_then(|bytes| {
                    let profile = serde_json::from_slice::<Publication>(&bytes);
                    previous_bytes = Some(bytes);
                    profile.map_err(|error| error.to_string())?.compile()
                });
                match next {
                    Ok(next) => {
                        previous_error = None;
                        let old = table.write().replace(next);
                        drop(old);
                    }
                    Err(error) => {
                        // Retry after an unavailable file, even when the restored contents match.
                        previous_bytes = None;
                        if previous_error.as_ref() != Some(&error) {
                            tracing::warn!(%error, "prefill chunk profile unavailable");
                            previous_error = Some(error);
                            let old = std::mem::take(&mut *table.write());
                            drop(old);
                        }
                    }
                }
                drop(table);
                std::thread::sleep(Duration::from_secs(1));
            }
        })?;
    Ok(table)
}

struct ChunkScorer {
    native: Box<dyn WorkerScorer>,
    partition: RoutingPartitionId,
    table: Published,
    prefill_scale: f64,
}

impl WorkerScorer for ChunkScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        self.native.required_worker_inputs() | WorkerInputs::CACHE | WorkerInputs::PREFERRED_TAINT
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        self.native.score(context, candidates, costs)?;
        if !context.tracks_prefill_tokens() || self.prefill_scale == 0.0 {
            return Ok(());
        }
        let Some(table) = self.table.try_read() else {
            return Ok(());
        };
        let Some(snapshot) = table.as_ref() else {
            return Ok(());
        };
        if snapshot.partition != self.partition
            || snapshot.block_size != context.block_size()
            || SystemTime::now() >= snapshot.expires
        {
            return Ok(());
        }
        for (candidate, cost) in candidates.iter().zip(costs) {
            let Some(path) = snapshot
                .ranks
                .get(&candidate.worker())
                .and_then(|paths| paths.get(&context.isl_tokens()))
            else {
                continue;
            };
            let Some(cache) = candidate.cache() else {
                continue;
            };
            // Tier weights are placement credits. Use the physical prefix geometry
            // when the indexer supplies it, including contiguous host and disk hits.
            let prefix = if cache.has_tier_matches() {
                let blocks = cache.device_overlap_blocks()
                    + cache.host_overlap_blocks()
                    + cache.disk_overlap_blocks();
                (blocks as usize).saturating_mul(context.block_size() as usize)
            } else {
                cache.accounting_cache_estimate().1
            };
            let adjusted = *cost
                + path.extra_cost(prefix)
                    * self.prefill_scale
                    * candidate.preferred_taint_multiplier().unwrap_or(1.0);
            if adjusted.is_finite() {
                *cost = adjusted;
            }
        }
        Ok(())
    }
}

fn policy(
    config: &SelectionConfig,
    role: WorkerType,
    table: Published,
    partition: RoutingPartitionRef<'_>,
) -> WorkerSelectionPolicy {
    let parameters = role_parameters(config, role, PolicyParameters::from(config));
    let label = role.default_selector_label();
    let native = scorer::build(&parameters, label);
    let scorer = if parameters.prefill_load_scale > 0.0
        && matches!(role, WorkerType::Aggregated | WorkerType::Prefill)
    {
        Box::new(ChunkScorer {
            native,
            partition: partition.into_owned(),
            table,
            prefill_scale: parameters.prefill_load_scale,
        }) as Box<dyn WorkerScorer>
    } else {
        native
    };
    WorkerSelectionPolicy::new(
        config.clone(),
        label,
        vec![scorer],
        Box::new(picker::DefaultPicker::new(
            parameters.router_temperature,
            None,
        )),
    )
    .with_exclusive_affinity(true)
}

pub(super) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        NAME,
        Arc::new(|values| {
            let parameters: Parameters = values.deserialize()?;
            if parameters.profile_file.as_os_str().is_empty() {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "expected a profile file path",
                ));
            }
            let table = observe(parameters.profile_file)
                .map_err(|error| WorkerSelectionPolicyProviderError::new(error.to_string()))?;
            Ok(Arc::new(
                move |config: &SelectionConfig, role, partition| {
                    policy(config, role, table.clone(), partition)
                },
            ))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_requires_an_exact_boundary_and_charges_only_extra_steps() {
        let path = PathCost {
            ends: vec![2048, 4608, 6144],
            step_cost_blocks: 1.5,
        };
        assert_eq!(path.extra_cost(0), 3.0);
        assert_eq!(path.extra_cost(2048), 1.5);
        assert_eq!(path.extra_cost(2049), 0.0);
        assert_eq!(path.extra_cost(4608), 0.0);
        assert_eq!(path.extra_cost(6144), 0.0);
    }

    #[test]
    fn reject_ambiguous_and_invalid_paths() {
        for paths in [
            vec![vec![4, 8], vec![8]],
            vec![vec![2, 2]],
            vec![vec![0]],
            vec![vec![]],
            vec![(1..=33).collect()],
            (1..=65).map(|n| vec![n]).collect(),
        ] {
            let params = Profiles {
                profiles: vec![RankProfile {
                    worker_id: 1,
                    dp_rank: 0,
                    step_cost_blocks: 1.0,
                    paths,
                }],
            };
            assert!(params.compile().is_err());
        }
    }
}
