// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory for one logical mock engine.

use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::generalized::{EngineIdentity, GeneralizedMockerEngine, RankIdentity};
use crate::engine::scheduler::{SchedulerRank, engine_seed_offset};
use crate::engine::{EngineConfig, TimingModel};

/// A single-rank or attention-DP engine.
pub type Engine = GeneralizedMockerEngine<SchedulerRank>;

/// Constructs engines with one process-local timing provider.
///
/// The serializable engine configuration contains only a provider descriptor.
/// A Runner resolves external timing providers before constructing this
/// factory, which then shares the provider across all attention-DP ranks.
#[derive(Clone)]
pub struct EngineFactory {
    config: EngineConfig,
    timing: Arc<dyn TimingModel>,
}

impl EngineFactory {
    /// Construct a factory for a built-in timing model.
    pub fn new(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let timing = config.built_in_timing_model()?;
        Ok(Self { config, timing })
    }

    /// Construct a factory with a process-local timing provider.
    pub fn with_timing_model(config: EngineConfig, timing: Arc<dyn TimingModel>) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, timing })
    }

    /// Build one scheduler/KV/timing rank with an explicit identity.
    pub fn build_rank(&self, identity: RankIdentity) -> Result<SchedulerRank> {
        let seed_offset = engine_seed_offset(identity)?;
        SchedulerRank::new_with_timing_model(
            identity,
            &self.config,
            Arc::clone(&self.timing),
            seed_offset,
        )
    }

    /// Build a single-rank or attention-DP logical engine.
    pub fn build(&self, identity: EngineIdentity, dp_size: NonZeroU32) -> Result<Engine> {
        GeneralizedMockerEngine::new_with_rank_factory(identity, dp_size, |rank_identity| {
            self.build_rank(rank_identity)
        })
    }
}
