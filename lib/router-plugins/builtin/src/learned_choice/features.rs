// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature sets and context sources of `learned-choice`. `FEATURES.md` is the specification;
//! keep the two in step. Feature set v1 is frozen once calibration starts.
//!
//! Every value comes from the public plugin API. None reads `expected_output_tokens` (replay
//! fills it with the true output length), the modeled prefill backlog, or anything else derived
//! from a performance model. Set aggregates are summed in canonical worker order, so a feature
//! never depends on the host's unspecified row order.

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerSelectionContext, WorkerSelectionPolicyError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;

use crate::signals::rendezvous;

/// Token scale of the `_k` features: the engine's default chunked-prefill batch.
const TOKEN_SCALE: f64 = 8192.0;
/// Request-count scale of `active_requests_s`.
const REQUEST_SCALE: f64 = 32.0;
/// Token scale inside the log features, so a 1K-token prefill scores zero.
const LOG_TOKEN_SCALE: f64 = 1024.0;
/// KV capacity assumed when a worker advertises none.
pub(crate) const FALLBACK_KV_BLOCKS: u64 = 65_536;
/// Prompt tokens that key `hash_home`, as `chwbl` keys them.
const HASH_HOME_PREFIX_TOKENS: u32 = 256;

pub(crate) const DEFAULT_LOGIT_SCALED: usize = 0;
pub(crate) const OVERLAP_FRAC: usize = 1;
pub(crate) const NEW_PREFILL_TOKENS_K: usize = 2;
pub(crate) const ACTIVE_PREFILL_TOKENS_K: usize = 3;
pub(crate) const KV_LOAD_FRAC: usize = 4;
pub(crate) const ACTIVE_REQUESTS_S: usize = 5;
pub(crate) const SESSION_AFFINITY: usize = 6;
pub(crate) const ISL_X_PREFILL_LOAD: usize = 7;
pub(crate) const LOG_PTOK: usize = 8;
pub(crate) const LOG_NEW_PREFILL: usize = 9;
pub(crate) const LOG_BS: usize = 10;
pub(crate) const HASH_HOME: usize = 11;
pub(crate) const NEW_PREFILL_X_REQUESTS: usize = 21;
pub(crate) const PREFILL_ATTN: usize = 22;

/// Load features that get set-relative transforms in v2: (feature, ratio floor), in the order of
/// the v2 ratio, below-fraction and excess blocks. Each floor is about one unit of that load.
const SET_RELATIVE: [(usize, f64); 3] = [
    (KV_LOAD_FRAC, 0.01),
    (ACTIVE_PREFILL_TOKENS_K, 512.0 / TOKEN_SCALE),
    (ACTIVE_REQUESTS_S, 1.0 / REQUEST_SCALE),
];
const RATIO_BASE: usize = 12;
const BELOW_BASE: usize = 15;
const EXCESS_BASE: usize = 18;

const V1_NAMES: [&str; 8] = [
    "default_logit_scaled",
    "overlap_frac",
    "new_prefill_tokens_k",
    "active_prefill_tokens_k",
    "kv_load_frac",
    "active_requests_s",
    "session_affinity",
    "isl_x_prefill_load",
];

const V2_NAMES: [&str; 23] = [
    "default_logit_scaled",
    "overlap_frac",
    "new_prefill_tokens_k",
    "active_prefill_tokens_k",
    "kv_load_frac",
    "active_requests_s",
    "session_affinity",
    "isl_x_prefill_load",
    "log_ptok",
    "log_new_prefill",
    "log_bs",
    "hash_home",
    "kv_load_ratio",
    "active_prefill_ratio",
    "active_requests_ratio",
    "kv_load_below_frac",
    "active_prefill_below_frac",
    "active_requests_below_frac",
    "kv_load_excess",
    "active_prefill_excess",
    "active_requests_excess",
    "new_prefill_x_requests",
    "prefill_attn",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum FeatureSet {
    V1,
    V2,
}

impl FeatureSet {
    pub(crate) fn names(self) -> &'static [&'static str] {
        match self {
            Self::V1 => &V1_NAMES,
            Self::V2 => &V2_NAMES,
        }
    }

    pub(crate) fn dim(self) -> usize {
        self.names().len()
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}

/// Request-level quantities a context term can condition on (`context.sources`). Each is the
/// same whatever the number of candidates when the candidate set is duplicated, unlike sums,
/// counts, or a mean of a one-hot feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContextSource {
    MeanKvLoadFrac,
    MeanActivePrefillTokensK,
    MeanActiveRequestsS,
    MaxOverlapFrac,
    IslK,
    IsFirstTurn,
}

const SOURCES: [(&str, ContextSource); 6] = [
    ("mean_kv_load_frac", ContextSource::MeanKvLoadFrac),
    (
        "mean_active_prefill_tokens_k",
        ContextSource::MeanActivePrefillTokensK,
    ),
    ("mean_active_requests_s", ContextSource::MeanActiveRequestsS),
    ("max_overlap_frac", ContextSource::MaxOverlapFrac),
    ("isl_k", ContextSource::IslK),
    ("is_first_turn", ContextSource::IsFirstTurn),
];

impl ContextSource {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        SOURCES
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, source)| *source)
    }

    pub(crate) fn known_names() -> impl Iterator<Item = &'static str> {
        SOURCES.iter().map(|(name, _)| *name)
    }
}

/// Request-level inputs the features need besides the candidate table.
pub(crate) struct RequestInputs {
    /// The worker that served this request's session last, if the policy remembers one.
    pub(crate) previous_worker: Option<WorkerWithDpRank>,
}

/// Feature matrix of one decision: `rows × dim`, row-major, indexed by host row.
pub(crate) struct FeatureTable {
    pub(crate) set: FeatureSet,
    pub(crate) values: Vec<f64>,
    pub(crate) isl_k: f64,
    pub(crate) is_first_turn: bool,
    workers: Vec<WorkerWithDpRank>,
    sorted: Vec<f64>,
}

impl FeatureTable {
    pub(crate) fn new(set: FeatureSet) -> Self {
        Self {
            set,
            values: Vec::new(),
            isl_k: 0.0,
            is_first_turn: true,
            workers: Vec::new(),
            sorted: Vec::new(),
        }
    }

    pub(crate) fn row(&self, row: usize) -> &[f64] {
        let dim = self.set.dim();
        &self.values[row * dim..(row + 1) * dim]
    }

    /// Mean of feature `index` over the candidates, summed in canonical `order`.
    pub(crate) fn mean(&self, index: usize, order: &[usize]) -> f64 {
        let dim = self.set.dim();
        order
            .iter()
            .map(|&row| self.values[row * dim + index])
            .sum::<f64>()
            / order.len() as f64
    }

    /// Value of one context source for this decision.
    pub(crate) fn source(&self, source: ContextSource, order: &[usize]) -> f64 {
        match source {
            ContextSource::MeanKvLoadFrac => self.mean(KV_LOAD_FRAC, order),
            ContextSource::MeanActivePrefillTokensK => self.mean(ACTIVE_PREFILL_TOKENS_K, order),
            ContextSource::MeanActiveRequestsS => self.mean(ACTIVE_REQUESTS_S, order),
            ContextSource::MaxOverlapFrac => order
                .iter()
                .map(|&row| self.row(row)[OVERLAP_FRAC])
                .fold(f64::NEG_INFINITY, f64::max),
            ContextSource::IslK => self.isl_k,
            ContextSource::IsFirstTurn => f64::from(u8::from(self.is_first_turn)),
        }
    }

    /// Fill the table for one decision. `order` lists the rows in canonical worker order.
    pub(crate) fn compute(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
        order: &[usize],
        request: &RequestInputs,
    ) -> Result<(), WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if cache.len() != candidates.len() || load.len() != candidates.len() {
            return Err(WorkerSelectionPolicyError::failed(
                "worker inputs do not match the candidates",
            ));
        }
        let dim = self.set.dim();
        let v2 = self.set == FeatureSet::V2;
        self.values.clear();
        self.values.resize(candidates.len() * dim, 0.0);
        self.workers.clear();
        self.workers
            .extend(candidates.iter().map(|candidate| candidate.worker()));

        let request_blocks = context.request_blocks().max(1) as f64;
        let prompt_tokens = context.prompt_tokens();
        self.isl_k = prompt_tokens as f64 / TOKEN_SCALE;
        self.is_first_turn = request.previous_worker.is_none();

        for (row, ((candidate, cache), load)) in candidates
            .iter()
            .zip(cache.iter())
            .zip(load.iter())
            .enumerate()
        {
            let (overlap_blocks, cached_tokens) = cache.accounting_cache_estimate();
            let new_prefill = prompt_tokens.saturating_sub(cached_tokens);
            let capacity = context
                .worker_capacity(candidate.worker())
                .and_then(|capacity| capacity.total_kv_blocks())
                .unwrap_or(FALLBACK_KV_BLOCKS);
            let x = &mut self.values[row * dim..(row + 1) * dim];
            x[DEFAULT_LOGIT_SCALED] = candidate.cost() / request_blocks;
            x[OVERLAP_FRAC] = overlap_blocks / request_blocks;
            x[NEW_PREFILL_TOKENS_K] = new_prefill as f64 / TOKEN_SCALE;
            x[ACTIVE_PREFILL_TOKENS_K] = load.active_prefill_tokens() as f64 / TOKEN_SCALE;
            x[KV_LOAD_FRAC] = load.decode_cost_blocks() / capacity as f64;
            x[ACTIVE_REQUESTS_S] = load.active_requests() as f64 / REQUEST_SCALE;
            x[SESSION_AFFINITY] = f64::from(u8::from(
                request.previous_worker == Some(candidate.worker()),
            ));
            x[ISL_X_PREFILL_LOAD] = self.isl_k * x[ACTIVE_PREFILL_TOKENS_K];
            if !v2 {
                continue;
            }
            let prefill_tokens = load.active_prefill_tokens().saturating_add(new_prefill);
            x[LOG_PTOK] = (prefill_tokens.max(1) as f64 / LOG_TOKEN_SCALE).ln();
            x[LOG_NEW_PREFILL] = (new_prefill.max(1) as f64 / LOG_TOKEN_SCALE).ln();
            x[LOG_BS] = (load.active_requests() as f64).ln_1p();
            x[NEW_PREFILL_X_REQUESTS] = x[NEW_PREFILL_TOKENS_K] * x[ACTIVE_REQUESTS_S];
            // Attention work of prefilling the last `n` of `L` prompt tokens: Σ_{t=L−n}^{L} t.
            let (n, length) = (new_prefill as f64, prompt_tokens as f64);
            x[PREFILL_ATTN] = n * (length - n / 2.0) / (TOKEN_SCALE * TOKEN_SCALE);
        }
        if v2 {
            self.set_relative(order);
            self.hash_home(context, order);
        }
        Ok(())
    }

    fn set_relative(&mut self, order: &[usize]) {
        let dim = self.set.dim();
        let count = order.len() as f64;
        for (block, (feature, floor)) in SET_RELATIVE.into_iter().enumerate() {
            let mean = self.mean(feature, order);
            self.sorted.clear();
            self.sorted
                .extend(order.iter().map(|&row| self.values[row * dim + feature]));
            self.sorted.sort_unstable_by(f64::total_cmp);
            for &row in order {
                let value = self.values[row * dim + feature];
                let below = self.sorted.partition_point(|&other| other < value);
                let x = &mut self.values[row * dim..(row + 1) * dim];
                x[RATIO_BASE + block] = value / (floor + mean);
                x[BELOW_BASE + block] = below as f64 / count;
                x[EXCESS_BASE + block] = (value - mean).max(0.0);
            }
        }
    }

    /// Mark the two highest rendezvous weights for the request's key as the key's home.
    fn hash_home(&mut self, context: &WorkerSelectionContext<'_>, order: &[usize]) {
        let dim = self.set.dim();
        let Some(key) = hash_home_key(context) else {
            return;
        };
        let weight = |row: usize| rendezvous(key, self.workers[row], 0);
        let (mut first, mut second) = (None::<u64>, None::<u64>);
        for &row in order {
            let value = weight(row);
            if first.is_none_or(|first| value > first) {
                second = first;
                first = Some(value);
            } else if second.is_none_or(|second| value > second) {
                second = Some(value);
            }
        }
        // With two or fewer candidates every worker is home.
        let threshold = if order.len() <= 2 {
            0
        } else {
            second.unwrap_or(0)
        };
        for &row in order {
            self.values[row * dim + HASH_HOME] = f64::from(u8::from(weight(row) >= threshold));
        }
    }
}

/// The prefix hash of the prompt's first `HASH_HOME_PREFIX_TOKENS` tokens (whole blocks, at least
/// one), or None without prefix hashes. Never the session ID: offline replay synthesizes a
/// single-use session ID per request for closed-loop flat traces, which a live router would not
/// see, so a session key would make the feature depend on the load mode (build audit F1).
fn hash_home_key(context: &WorkerSelectionContext<'_>) -> Option<u64> {
    let hashes = context.prefix_hashes()?;
    let blocks = (HASH_HOME_PREFIX_TOKENS / context.block_size().max(1)).max(1) as usize;
    hashes
        .get(blocks.min(hashes.len()).checked_sub(1)?)
        .copied()
}
