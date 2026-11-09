// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fs;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer};
use thiserror::Error;
use tokio::time::Instant;

use super::KvSchedulerError;
use super::config::RouterQueuePolicy;
use crate::plugins::request_classifier::RawRequestClassifierConfig;
use crate::plugins::request_classifier::RequestClassifierConfig;
use crate::plugins::worker_selection::{RawWorkerSelectionConfig, WorkerSelectionConfig};
use crate::plugins::worker_selection::{WorkerCapacity, WorkerLoadInput};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::WorkerLoadProjection;

const SYNTHETIC_POLICY_CLASS: &str = "default";

const FAMILY_BUCKET_DEPRECATION: &str = "Policy-family/cache-bucket configuration is deprecated, with removal planned for 1.8.0. Migrate to flat policy classes, which default to EDF. Set queue_policy explicitly to retain your desired ordering.";

#[derive(Debug, Error)]
pub enum RouterPolicyConfigError {
    #[error("failed to read router policy config {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse router policy config {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("invalid router policy config: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyClassConfig {
    pub name: String,
    pub queue_policy: RouterQueuePolicy,
    pub quantum: usize,
    /// Time-to-first-token budget in whole milliseconds, at most `u32::MAX` (about 49.7 days),
    /// or `None` when the class sets none. An EDF class orders a request without an explicit
    /// deadline by its router ingress plus this budget. Other orders neither sort nor expire
    /// by it.
    pub slo_ttft_ms: Option<NonZeroU32>,
    /// Controlled delay, EDF classes only: the online router rejects a request still waiting
    /// for admission when its due time passes. That is its explicit deadline, which wins even
    /// when later, else router ingress plus `slo_ttft_ms`. Off by default, when the class
    /// budget only orders the class. Explicit deadlines expire either way.
    pub controlled_delay: bool,
    /// The limits this class waits on.
    pub busy_thresholds: BusyThresholds,
    pub request_queue_limit_per_worker: Option<usize>,
    pub raw_isl_token_queue_limit_per_worker: Option<usize>,
    pub cached_token_queue_limit_per_worker: Option<usize>,
}

impl PolicyClassConfig {
    pub fn queueing_enabled(&self) -> bool {
        self.busy_thresholds.is_configured()
    }

    /// Whether an online queue expires this class's budget: controlled delay in an EDF class,
    /// the only order in which every lane root, blocked lanes included, carries the earliest
    /// due time of its lane.
    pub(crate) fn expires_class_budget(&self) -> bool {
        self.controlled_delay && self.queue_policy == RouterQueuePolicy::Edf
    }

    /// Any zero per-worker limit means `queue_rejection` rejects every queued
    /// entry (`current >= limit` holds at zero usage), so the class can never
    /// hold a queued request and must keep the direct admission path.
    pub fn never_queues(&self) -> bool {
        self.request_queue_limit_per_worker == Some(0)
            || self.raw_isl_token_queue_limit_per_worker == Some(0)
            || self.cached_token_queue_limit_per_worker == Some(0)
    }

    /// When this class's TTFT budget runs out for a request that entered the router at
    /// `ingress_at`, or `None` when the class sets no budget.
    pub fn slo_ttft_due_at(&self, ingress_at: Instant) -> Option<Instant> {
        // The `u32` millisecond budget keeps the sum within the monotonic clock's range.
        self.slo_ttft_ms
            .map(|slo_ttft_ms| ingress_at + Duration::from_millis(u64::from(slo_ttft_ms.get())))
    }
}

/// Supplies the host's live reject thresholds. Selection skips a worker whose every rank is
/// past them, and rejects the request when no eligible worker remains.
pub type RejectThresholdsProvider = Arc<dyn Fn() -> BusyThresholds + Send + Sync + 'static>;

/// Per-rank load limits. A rank is full once its booked load passes any configured limit.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BusyThresholds {
    /// Booked prefill tokens above which a rank is full.
    // TODO(v1.7): Remove the absolute threshold; use `prefill_busy_threshold_frac`.
    pub prefill_busy_threshold: Option<usize>,
    /// Fraction of the rank's `max_num_batched_tokens` above which it is full. When the
    /// rank does not report it, the fraction is taken of
    /// [`DEFAULT_MAX_BATCHED_TOKENS`](crate::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS).
    pub prefill_busy_threshold_frac: Option<f64>,
    /// Fraction of the rank's `total_kv_blocks` above which the KV blocks its active
    /// requests hold make it full. A rank that does not report `total_kv_blocks` skips it.
    pub decode_busy_threshold_frac: Option<f64>,
}

impl BusyThresholds {
    /// Whether any limit is set.
    pub fn is_configured(&self) -> bool {
        self.prefill_busy_threshold.is_some()
            || self.prefill_busy_threshold_frac.is_some()
            || self.decode_busy_threshold_frac.is_some()
    }

    /// Whether a rank with this `LOAD` row and capacity is within every configured limit.
    pub fn has_room(&self, load: &WorkerLoadInput, capacity: WorkerCapacity) -> bool {
        let tokens = load.active_prefill_tokens();
        let prefill_budget = capacity.prefill_budget();
        let absolute_full = self
            .prefill_busy_threshold
            .is_some_and(|threshold| tokens > threshold);
        let fractional_full = self
            .prefill_busy_threshold_frac
            .is_some_and(|threshold| (tokens as f64) > threshold * (prefill_budget as f64));
        let decode_full = self
            .decode_busy_threshold_frac
            .zip(capacity.total_kv_blocks())
            .is_some_and(|(threshold, total_kv_blocks)| {
                (load.active_decode_blocks() as f64) > threshold * (total_kv_blocks as f64)
            });
        !(absolute_full || fractional_full || decode_full)
    }

    /// Whether this rank's `LOAD` row exceeds any configured limit. A rank without a
    /// projection counts as idle.
    pub fn rank_is_full(
        &self,
        worker: WorkerWithDpRank,
        capacity: WorkerCapacity,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool {
        let load = WorkerLoadInput::for_worker(loads, worker, WorkerLoadProjection::default);
        !self.has_room(&load, capacity)
    }

    /// Whether no configured rank of `worker_id` has room, judged on the `LOAD` rows in `loads`.
    pub fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool {
        let capacity = WorkerCapacity::of(config);
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        (start..end).all(|dp_rank| {
            self.rank_is_full(WorkerWithDpRank::new(worker_id, dp_rank), capacity, loads)
        })
    }
}

/// Ranks past a set of busy thresholds, counted per worker while a request's load is
/// projected. Only a worker with at least one full rank has an entry, so the map stays empty
/// while no worker is near its limits and selection pays nothing to consult it.
#[derive(Debug, Default)]
pub(crate) struct FullRankCounts {
    limits: BusyThresholds,
    full_ranks: FxHashMap<WorkerId, u32>,
}

impl FullRankCounts {
    /// Start a pass judged against `limits`, forgetting the previous pass.
    pub(crate) fn reset(&mut self, limits: BusyThresholds) {
        self.limits = limits;
        self.full_ranks.clear();
    }

    /// Forget what the current pass recorded; it starts over from the first rank.
    pub(crate) fn clear(&mut self) {
        self.full_ranks.clear();
    }

    /// Judge one rank's projected load against the limits and count it when it has no room.
    #[inline]
    pub(crate) fn record(
        &mut self,
        worker: WorkerWithDpRank,
        capacity: WorkerCapacity,
        projection: &WorkerLoadProjection,
    ) {
        if !self
            .limits
            .has_room(&WorkerLoadInput::from_projection(*projection), capacity)
        {
            *self.full_ranks.entry(worker.worker_id).or_insert(0) += 1;
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.full_ranks.is_empty()
    }

    /// Whether every one of the worker's `dp_size` ranks was counted full.
    #[inline]
    pub(crate) fn worker_is_full(&self, worker_id: WorkerId, dp_size: u32) -> bool {
        self.full_ranks
            .get(&worker_id)
            .is_some_and(|&full| full >= dp_size)
    }
}

/// One set of policy classes and how a request selects one of them.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyProfile {
    classes: Vec<PolicyClassConfig>,
    lookup: ClassLookup,
}

#[derive(Debug, Clone, PartialEq)]
enum ClassLookup {
    /// Flat classes and the synthetic class: each class is selected by its own name.
    ByName {
        class_indices: HashMap<String, usize>,
        default_class_index: usize,
    },
    /// The deprecated policy-family/cache-bucket format.
    FamilyBucket(FamilyBucketClassifier),
}

#[derive(Debug, Clone, PartialEq)]
struct FamilyBucketClassifier {
    default_family_index: usize,
    family_indices: HashMap<String, usize>,
    explicit_class_indices: HashMap<String, usize>,
    buckets: Vec<UncachedIslBucket>,
    class_by_family_bucket: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UncachedIslBucket {
    min_tokens: usize,
}

impl FamilyBucketClassifier {
    /// Returns only selections that do not require a cache snapshot.
    fn direct_class_index(&self, requested: Option<&str>) -> Option<usize> {
        requested.and_then(|name| self.explicit_class_indices.get(name).copied())
    }

    /// Combines a recognized family (or the default) with the observed bucket.
    fn class_index(&self, requested: Option<&str>, uncached_tokens: usize) -> usize {
        if let Some(class_index) = self.direct_class_index(requested) {
            return class_index;
        }

        let family_index = requested
            .and_then(|name| self.family_indices.get(name).copied())
            .unwrap_or(self.default_family_index);
        self.class_index_for_family(family_index, uncached_tokens)
    }

    fn strict_class_index(&self, requested: &str, uncached_tokens: usize) -> Option<usize> {
        self.explicit_class_indices
            .get(requested)
            .copied()
            .or_else(|| {
                self.family_indices
                    .get(requested)
                    .map(|family_index| self.class_index_for_family(*family_index, uncached_tokens))
            })
    }

    fn class_index_for_family(&self, family_index: usize, uncached_tokens: usize) -> usize {
        let bucket_index = self
            .buckets
            .partition_point(|bucket| bucket.min_tokens <= uncached_tokens)
            .saturating_sub(1);
        self.class_by_family_bucket[family_index * self.buckets.len() + bucket_index]
    }
}

impl PolicyProfile {
    pub fn synthetic(
        router_queue_threshold: Option<f64>,
        router_queue_policy: RouterQueuePolicy,
    ) -> Self {
        let class = PolicyClassConfig {
            name: SYNTHETIC_POLICY_CLASS.to_string(),
            queue_policy: router_queue_policy,
            quantum: 1,
            slo_ttft_ms: None,
            controlled_delay: false,
            busy_thresholds: BusyThresholds {
                prefill_busy_threshold: None,
                prefill_busy_threshold_frac: router_queue_threshold,
                decode_busy_threshold_frac: None,
            },
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
        };
        Self {
            lookup: ClassLookup::ByName {
                class_indices: HashMap::from([(class.name.clone(), 0)]),
                default_class_index: 0,
            },
            classes: vec![class],
        }
    }

    pub fn classes(&self) -> &[PolicyClassConfig] {
        &self.classes
    }

    /// The class an omitted name selects. In the deprecated policy-family format, that is
    /// the default family's class for zero uncached tokens.
    pub fn default_class(&self) -> &PolicyClassConfig {
        let index = match &self.lookup {
            ClassLookup::ByName {
                default_class_index,
                ..
            } => *default_class_index,
            ClassLookup::FamilyBucket(classifier) => classifier.class_index(None, 0),
        };
        &self.classes[index]
    }

    /// Resolves a requested class without cache state. `Ok(None)` means only
    /// [`resolve_class_index_for_uncached_tokens`](Self::resolve_class_index_for_uncached_tokens)
    /// can resolve it: in the deprecated policy-family format, every name but an explicit
    /// class's resolves by the request's uncached tokens.
    pub fn direct_class_index(
        &self,
        requested: Option<&str>,
    ) -> Result<Option<usize>, KvSchedulerError> {
        match &self.lookup {
            ClassLookup::ByName { .. } => self.resolve_class_index(requested).map(Some),
            ClassLookup::FamilyBucket(classifier) => Ok(classifier.direct_class_index(requested)),
        }
    }

    /// Resolves a requested class name to its queue. An omitted name selects the default
    /// class; a name that matches no configured class is an error. A profile in the
    /// deprecated policy-family format resolves as for zero uncached tokens, as
    /// [`default_class`](Self::default_class) does; to select by a request's cache state, use
    /// [`resolve_class_index_for_uncached_tokens`](Self::resolve_class_index_for_uncached_tokens).
    pub fn resolve_class_index(&self, requested: Option<&str>) -> Result<usize, KvSchedulerError> {
        self.resolve_class_index_for_uncached_tokens(requested, 0)
    }

    /// Resolves a requested class name to its queue for a request with `uncached_tokens`
    /// uncached input tokens. A flat profile ignores them and resolves as
    /// [`resolve_class_index`](Self::resolve_class_index) does.
    ///
    /// In the deprecated policy-family format, an explicit class's name selects that class.
    /// Any other name, or none, never fails: a family name selects that family and every
    /// other name the default family, and the family's class for `uncached_tokens` is the
    /// queue.
    pub fn resolve_class_index_for_uncached_tokens(
        &self,
        requested: Option<&str>,
        uncached_tokens: usize,
    ) -> Result<usize, KvSchedulerError> {
        match (&self.lookup, requested) {
            (
                ClassLookup::ByName {
                    default_class_index,
                    ..
                },
                None,
            ) => Ok(*default_class_index),
            (ClassLookup::ByName { .. }, Some(name)) => self
                .resolve_class_index_strict(name, uncached_tokens)
                .ok_or_else(|| KvSchedulerError::UnknownPolicyClass(name.to_string())),
            (ClassLookup::FamilyBucket(classifier), requested) => {
                Ok(classifier.class_index(requested, uncached_tokens))
            }
        }
    }

    pub fn class(&self, index: usize) -> &PolicyClassConfig {
        &self.classes[index]
    }

    /// Resolves a configured class name, or `None` for any other name. In the deprecated
    /// policy-family format, it resolves only an explicit class's name, or a family name to
    /// that family's class for `uncached_tokens`.
    pub(crate) fn resolve_class_index_strict(
        &self,
        requested: &str,
        uncached_tokens: usize,
    ) -> Option<usize> {
        match &self.lookup {
            ClassLookup::ByName { class_indices, .. } => class_indices.get(requested).copied(),
            ClassLookup::FamilyBucket(classifier) => {
                classifier.strict_class_index(requested, uncached_tokens)
            }
        }
    }

    /// Whether this profile uses the deprecated policy-family/cache-bucket format.
    fn uses_family_buckets(&self) -> bool {
        matches!(self.lookup, ClassLookup::FamilyBucket(_))
    }
}

/// Process-wide cache and tracking settings, separate from policy parameters.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(transparent)]
pub(super) struct RouterSettings(HashMap<String, serde_json::Value>);

impl RouterSettings {
    pub(super) fn apply(&self, config: &mut super::config::KvRouterConfig) -> Result<(), String> {
        // Deserialize into each field's existing type, including nullable fields.
        // Do not round-trip KvRouterConfig: it has process-local, non-wire fields.
        macro_rules! apply_fields {
            ($($field:ident),* $(,)?) => {
                for (name, value) in &self.0 {
                    match name.as_str() {
                        $(stringify!($field) => {
                            config.$field = serde_json::from_value(value.clone())
                                .map_err(|error| format!("router.{name}: {error}"))?;
                        })*
                        _ => return Err(format!("unknown router setting: {name}")),
                    }
                }
            };
        }
        apply_fields!(
            host_cache_hit_weight,
            disk_cache_hit_weight,
            use_kv_events,
            router_replica_sync,
            router_track_active_blocks,
            router_track_output_blocks,
            router_assume_kv_reuse,
            router_track_prefill_tokens,
            router_tracking_hash,
            router_tracking_key_file,
            router_tracking_key_id,
            router_prefill_load_model,
            router_ttl_secs,
            router_approximate_cache_policy,
            router_event_threads,
            use_remote_indexer,
            serve_indexer,
            enable_session_prefix_index,
            shared_cache_type,
            router_predicted_ttl_secs,
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterPolicyConfig {
    router: Option<RouterSettings>,
    root: Option<PolicyProfile>,
    models: HashMap<String, PolicyProfile>,
    worker_selection: Option<WorkerSelectionConfig>,
    request_classifier: Option<RequestClassifierConfig>,
}

impl RouterPolicyConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, RouterPolicyConfigError> {
        let path = path.as_ref();
        let contents =
            fs::read_to_string(path).map_err(|source| RouterPolicyConfigError::Read {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_yaml(&contents).map_err(|error| match error {
            RouterPolicyConfigError::Parse { source, .. } => RouterPolicyConfigError::Parse {
                path: path.display().to_string(),
                source,
            },
            other => other,
        })
    }

    pub fn from_yaml(contents: &str) -> Result<Self, RouterPolicyConfigError> {
        let raw: RawRouterPolicyConfig =
            serde_yaml::from_str(contents).map_err(|source| RouterPolicyConfigError::Parse {
                path: "<inline>".to_string(),
                source,
            })?;
        raw.resolve()
    }

    pub fn resolve_profile(
        &self,
        model_name: Option<&str>,
        fallback_threshold: Option<f64>,
        fallback_policy: RouterQueuePolicy,
    ) -> PolicyProfile {
        // Model profiles replace the root wholesale; the synthetic profile is
        // constructed only when neither configured profile applies.
        model_name
            .and_then(|name| self.models.get(name))
            .or(self.root.as_ref())
            .cloned()
            .unwrap_or_else(|| PolicyProfile::synthetic(fallback_threshold, fallback_policy))
    }

    pub(super) fn router(&self) -> Option<&RouterSettings> {
        self.router.as_ref()
    }

    /// Returns the process-wide worker-selection policy configuration, if present.
    pub fn worker_selection(&self) -> Option<&WorkerSelectionConfig> {
        self.worker_selection.as_ref()
    }

    /// Returns the process-wide request-classifier plugin configuration, if present.
    pub fn request_classifier(&self) -> Option<&RequestClassifierConfig> {
        self.request_classifier.as_ref()
    }

    /// Whether this document configures queue policy profiles.
    pub fn has_routing_profiles(&self) -> bool {
        self.root.is_some() || !self.models.is_empty()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRouterPolicyConfig {
    router: Option<RouterSettings>,
    default_policy_class: Option<String>,
    #[serde(default, deserialize_with = "present")]
    default_policy_family: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    uncached_isl_buckets: Option<Option<Vec<RawUncachedIslBucket>>>,
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    worker_selection: Option<RawWorkerSelectionConfig>,
    request_classifier: Option<RawRequestClassifierConfig>,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
        if let Some(router) = &self.router {
            router
                .apply(&mut super::config::KvRouterConfig::default())
                .map_err(RouterPolicyConfigError::Validation)?;
        }
        let root = match self.policy_classes {
            Some(policy_classes) => Some(resolve_profile(
                RawPolicyProfile {
                    default_policy_class: self.default_policy_class,
                    default_policy_family: self.default_policy_family,
                    uncached_isl_buckets: self.uncached_isl_buckets,
                    policy_classes,
                },
                "root",
            )?),
            // Without classes, the root configures no profile, and a null family or bucket
            // key there is unset, so only a set one is incomplete.
            None if matches!(self.default_policy_family, Some(Some(_)))
                || matches!(self.uncached_isl_buckets, Some(Some(_))) =>
            {
                return Err(incomplete_family_bucket_profile("root"));
            }
            None if self.default_policy_class.is_some() => {
                return Err(RouterPolicyConfigError::Validation(
                    "root default_policy_class requires policy_classes".to_string(),
                ));
            }
            None => None,
        };

        let mut models = HashMap::with_capacity(self.models.len());
        for (model_name, profile) in self.models {
            if model_name.is_empty() {
                return Err(RouterPolicyConfigError::Validation(
                    "model profile name must not be empty".to_string(),
                ));
            }
            let resolved = resolve_profile(profile, &format!("model {model_name:?}"))?;
            models.insert(model_name, resolved);
        }

        let worker_selection = match self.worker_selection {
            Some(config) => Some(config.resolve()?),
            None => None,
        };

        let request_classifier = self
            .request_classifier
            .map(|config| config.resolve())
            .transpose()?;
        if self.router.is_none()
            && root.is_none()
            && models.is_empty()
            && worker_selection.is_none()
            && request_classifier.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "router policy config must define router settings, a root profile, at least one model profile, worker_selection, or request_classifier".to_string(),
            ));
        }

        if root
            .iter()
            .chain(models.values())
            .any(PolicyProfile::uses_family_buckets)
        {
            tracing::warn!("{FAMILY_BUCKET_DEPRECATION}");
        }
        Ok(RouterPolicyConfig {
            router: self.router,
            root,
            models,
            worker_selection,
            request_classifier,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyProfile {
    default_policy_class: Option<String>,
    /// Either key, even null, puts the profile in the deprecated policy-family format.
    #[serde(default, deserialize_with = "present")]
    default_policy_family: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    uncached_isl_buckets: Option<Option<Vec<RawUncachedIslBucket>>>,
    policy_classes: Vec<RawPolicyClassConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawUncachedIslBucket {
    min_tokens: usize,
    bucket: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyClassConfig {
    name: String,
    /// Deprecated policy-family/cache-bucket format only. Either key, even null, puts its
    /// profile in that format, where null leaves the class unbound.
    #[serde(default, deserialize_with = "present")]
    policy_family: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    cache_bucket: Option<Option<String>>,
    /// Omitted, a flat class is EDF and a policy-family class FCFS; the synthetic class
    /// keeps the router-wide policy.
    #[serde(default, deserialize_with = "present")]
    queue_policy: Option<RouterQueuePolicy>,
    quantum: usize,
    #[serde(default, deserialize_with = "present")]
    slo_ttft_ms: Option<Option<u32>>,
    #[serde(default, deserialize_with = "present")]
    controlled_delay: Option<bool>,
    prefill_busy_threshold: Option<usize>,
    prefill_busy_threshold_frac: Option<f64>,
    decode_busy_threshold_frac: Option<f64>,
    request_queue_limit_per_worker: Option<usize>,
    raw_isl_token_queue_limit_per_worker: Option<usize>,
    cached_token_queue_limit_per_worker: Option<usize>,
}

/// Deserializes a present key as `Some`, so a YAML null reaches the value's own type: an
/// error for a plain value, `Some(None)` for an `Option`. With `#[serde(default)]`, an absent
/// key stays `None`.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// How a profile's requests select their classes.
#[derive(Debug, Clone, Copy)]
enum ProfileFormat {
    /// By class name.
    Flat,
    /// By policy family and uncached-ISL bucket, deprecated.
    FamilyBucket,
}

fn resolve_profile(
    profile: RawPolicyProfile,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    let RawPolicyProfile {
        default_policy_class,
        default_policy_family,
        uncached_isl_buckets,
        policy_classes,
    } = profile;
    // Any family or bucket key, even null or empty, selects the deprecated format, so a
    // malformed profile in that format fails instead of loading as flat classes.
    let family_bucket = default_policy_family.is_some()
        || uncached_isl_buckets.is_some()
        || policy_classes
            .iter()
            .any(|class| class.policy_family.is_some() || class.cache_bucket.is_some());
    if !family_bucket {
        return resolve_flat_profile(default_policy_class, policy_classes, location);
    }
    if default_policy_class.is_some() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} default_policy_class cannot be combined with policy families or cache buckets"
        )));
    }
    let (Some(Some(default_policy_family)), Some(Some(uncached_isl_buckets))) =
        (default_policy_family, uncached_isl_buckets)
    else {
        return Err(incomplete_family_bucket_profile(location));
    };
    resolve_family_bucket_profile(
        default_policy_family,
        uncached_isl_buckets,
        policy_classes,
        location,
    )
}

fn incomplete_family_bucket_profile(location: &str) -> RouterPolicyConfigError {
    RouterPolicyConfigError::Validation(format!(
        "{location} profile must specify default_policy_family, uncached_isl_buckets, and policy_classes when any policy-family or cache-bucket field is present"
    ))
}

fn resolve_flat_profile(
    default_policy_class: Option<String>,
    raw_classes: Vec<RawPolicyClassConfig>,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    if raw_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let mut classes = Vec::with_capacity(raw_classes.len());
    let mut class_indices = HashMap::with_capacity(raw_classes.len());
    for raw in raw_classes {
        let class = resolve_policy_class(raw, ProfileFormat::Flat, location)?;
        if class_indices
            .insert(class.name.clone(), classes.len())
            .is_some()
        {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                class.name
            )));
        }
        classes.push(class);
    }

    let default_class_index = match default_policy_class {
        Some(name) => class_indices.get(&name).copied().ok_or_else(|| {
            RouterPolicyConfigError::Validation(format!(
                "{location} default_policy_class {name:?} does not name a configured policy class"
            ))
        })?,
        None if classes.len() == 1 => 0,
        None => {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} default_policy_class is required with more than one policy class"
            )));
        }
    };

    Ok(PolicyProfile {
        classes,
        lookup: ClassLookup::ByName {
            class_indices,
            default_class_index,
        },
    })
}

fn resolve_family_bucket_profile(
    default_policy_family: String,
    raw_buckets: Vec<RawUncachedIslBucket>,
    raw_classes: Vec<RawPolicyClassConfig>,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    validate_identifier(&default_policy_family, "policy family", location)?;
    if raw_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let resolved_buckets = resolve_uncached_isl_buckets(raw_buckets, location)?;
    let mut names = HashSet::with_capacity(raw_classes.len());
    let mut classes = Vec::with_capacity(raw_classes.len());
    let mut bindings = Vec::with_capacity(raw_classes.len());
    for mut raw in raw_classes {
        let policy_family = raw.policy_family.take().flatten();
        let cache_bucket = raw.cache_bucket.take().flatten();
        let class = resolve_policy_class(raw, ProfileFormat::FamilyBucket, location)?;
        let binding = resolve_class_binding(
            &class.name,
            policy_family,
            cache_bucket,
            &resolved_buckets.indices,
            location,
        )?;
        if !names.insert(class.name.clone()) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate policy class {:?}",
                class.name
            )));
        }
        classes.push(class);
        bindings.push(binding);
    }

    let mut family_names = Vec::new();
    let mut family_indices = HashMap::new();
    for binding in &bindings {
        let ClassBinding::FamilyBucket { policy_family, .. } = binding else {
            continue;
        };
        if !family_indices.contains_key(policy_family) {
            let family_index = family_names.len();
            family_names.push(policy_family.clone());
            family_indices.insert(policy_family.clone(), family_index);
        }
    }

    let Some(default_family_index) = family_indices.get(&default_policy_family).copied() else {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} default_policy_family {default_policy_family:?} does not name a configured family"
        )));
    };

    let mut explicit_class_indices = HashMap::new();
    let mut class_by_family_bucket = vec![
        None;
        family_names
            .len()
            .saturating_mul(resolved_buckets.buckets.len())
    ];
    for (class_index, binding) in bindings.into_iter().enumerate() {
        match binding {
            ClassBinding::Explicit => {
                let class_name = &classes[class_index].name;
                if family_indices.contains_key(class_name) {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} explicit policy class {class_name:?} collides with a policy family"
                    )));
                }
                explicit_class_indices.insert(class_name.clone(), class_index);
            }
            ClassBinding::FamilyBucket {
                policy_family,
                bucket_index,
            } => {
                let family_index = family_indices[&policy_family];
                let table_index = family_index * resolved_buckets.buckets.len() + bucket_index;
                if class_by_family_bucket[table_index]
                    .replace(class_index)
                    .is_some()
                {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} contains duplicate policy classes for family {policy_family:?} and bucket {:?}",
                        resolved_buckets.names[bucket_index]
                    )));
                }
            }
        }
    }

    for (family_index, family_name) in family_names.iter().enumerate() {
        for (bucket_index, bucket_name) in resolved_buckets.names.iter().enumerate() {
            if class_by_family_bucket[family_index * resolved_buckets.buckets.len() + bucket_index]
                .is_none()
            {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} is missing a policy class for family {family_name:?} and bucket {bucket_name:?}"
                )));
            }
        }
    }

    Ok(PolicyProfile {
        classes,
        lookup: ClassLookup::FamilyBucket(FamilyBucketClassifier {
            default_family_index,
            family_indices,
            explicit_class_indices,
            buckets: resolved_buckets.buckets,
            class_by_family_bucket: class_by_family_bucket
                .into_iter()
                .map(|class_index| class_index.expect("validated complete policy matrix"))
                .collect(),
        }),
    })
}

enum ClassBinding {
    Explicit,
    FamilyBucket {
        policy_family: String,
        bucket_index: usize,
    },
}

fn resolve_class_binding(
    class_name: &str,
    policy_family: Option<String>,
    cache_bucket: Option<String>,
    bucket_indices: &HashMap<String, usize>,
    location: &str,
) -> Result<ClassBinding, RouterPolicyConfigError> {
    match (policy_family, cache_bucket) {
        (None, None) => Ok(ClassBinding::Explicit),
        (Some(policy_family), Some(cache_bucket)) => {
            validate_identifier(&policy_family, "policy family", location)?;
            validate_identifier(&cache_bucket, "cache bucket", location)?;
            let Some(bucket_index) = bucket_indices.get(&cache_bucket).copied() else {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} policy class {class_name:?} references unknown cache bucket {cache_bucket:?}"
                )));
            };
            Ok(ClassBinding::FamilyBucket {
                policy_family,
                bucket_index,
            })
        }
        _ => Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {class_name:?} must specify both policy_family and cache_bucket or neither for an explicit class"
        ))),
    }
}

struct ResolvedBuckets {
    buckets: Vec<UncachedIslBucket>,
    names: Vec<String>,
    indices: HashMap<String, usize>,
}

fn resolve_uncached_isl_buckets(
    raw_buckets: Vec<RawUncachedIslBucket>,
    location: &str,
) -> Result<ResolvedBuckets, RouterPolicyConfigError> {
    if raw_buckets.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} uncached_isl_buckets must not be empty"
        )));
    }
    if raw_buckets[0].min_tokens != 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} uncached_isl_buckets must start at min_tokens 0"
        )));
    }
    for window in raw_buckets.windows(2) {
        if window[1].min_tokens <= window[0].min_tokens {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} uncached_isl_buckets min_tokens must be strictly increasing"
            )));
        }
    }

    let mut bucket_names = Vec::with_capacity(raw_buckets.len());
    let mut bucket_indices = HashMap::with_capacity(raw_buckets.len());
    let mut buckets = Vec::with_capacity(raw_buckets.len());
    for raw in raw_buckets {
        validate_identifier(&raw.bucket, "cache bucket", location)?;
        let bucket_index = bucket_names.len();
        if bucket_indices
            .insert(raw.bucket.clone(), bucket_index)
            .is_some()
        {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} contains duplicate cache bucket {:?}",
                raw.bucket
            )));
        }
        bucket_names.push(raw.bucket);
        buckets.push(UncachedIslBucket {
            min_tokens: raw.min_tokens,
        });
    }

    Ok(ResolvedBuckets {
        buckets,
        names: bucket_names,
        indices: bucket_indices,
    })
}

fn resolve_policy_class(
    raw: RawPolicyClassConfig,
    format: ProfileFormat,
    location: &str,
) -> Result<PolicyClassConfig, RouterPolicyConfigError> {
    validate_identifier(&raw.name, "policy class", location)?;
    if raw.quantum == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} quantum must be greater than zero",
            raw.name
        )));
    }
    let queue_policy = match format {
        ProfileFormat::Flat => {
            let queue_policy = raw.queue_policy.unwrap_or(RouterQueuePolicy::Edf);
            if queue_policy == RouterQueuePolicy::Lcfs {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} policy class {:?} queue_policy must be edf, fcfs, or wspt",
                    raw.name
                )));
            }
            queue_policy
        }
        // The deprecated format keeps its own orders and FCFS default. EDF and its settings
        // need flat classes.
        ProfileFormat::FamilyBucket => {
            let queue_policy = raw.queue_policy.unwrap_or(RouterQueuePolicy::Fcfs);
            if !matches!(
                queue_policy,
                RouterQueuePolicy::Fcfs | RouterQueuePolicy::Wspt
            ) {
                return Err(RouterPolicyConfigError::Validation(format!(
                    "{location} policy class {:?} queue_policy must be fcfs or wspt in a policy-family profile",
                    raw.name
                )));
            }
            for (field, set) in [
                ("slo_ttft_ms", raw.slo_ttft_ms.is_some()),
                ("controlled_delay", raw.controlled_delay.is_some()),
            ] {
                if set {
                    return Err(RouterPolicyConfigError::Validation(format!(
                        "{location} policy class {:?} {field} requires flat policy classes",
                        raw.name
                    )));
                }
            }
            queue_policy
        }
    };
    let controlled_delay = raw.controlled_delay.unwrap_or(false);
    if controlled_delay && queue_policy != RouterQueuePolicy::Edf {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} controlled_delay requires queue_policy edf",
            raw.name
        )));
    }
    for (field, value) in [
        (
            "prefill_busy_threshold_frac",
            raw.prefill_busy_threshold_frac,
        ),
        ("decode_busy_threshold_frac", raw.decode_busy_threshold_frac),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} {field} must be finite and non-negative",
                raw.name
            )));
        }
    }
    let slo_ttft_ms = match raw.slo_ttft_ms.flatten() {
        None => None,
        Some(slo_ttft_ms) => Some(NonZeroU32::new(slo_ttft_ms).ok_or_else(|| {
            RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} slo_ttft_ms must be greater than zero",
                raw.name
            ))
        })?),
    };

    if raw.prefill_busy_threshold.is_some() {
        tracing::warn!(
            policy_class = %raw.name,
            "prefill_busy_threshold is deprecated and will be removed in v1.7; use prefill_busy_threshold_frac"
        );
    }
    Ok(PolicyClassConfig {
        name: raw.name,
        queue_policy,
        quantum: raw.quantum,
        slo_ttft_ms,
        controlled_delay,
        busy_thresholds: BusyThresholds {
            prefill_busy_threshold: raw.prefill_busy_threshold,
            prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
            decode_busy_threshold_frac: raw.decode_busy_threshold_frac,
        },
        request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
        raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
        cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
    })
}

pub(crate) fn validate_identifier(
    name: &str,
    kind: &str,
    location: &str,
) -> Result<(), RouterPolicyConfigError> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Ok(());
    }

    Err(RouterPolicyConfigError::Validation(format!(
        "{location} {kind} name {name:?} must match [A-Za-z0-9_.-]+"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequences::WorkerLoadProjection;

    fn prefill_load(active_prefill_tokens: usize) -> WorkerLoadInput {
        WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_prefill_tokens,
            ..Default::default()
        })
    }

    #[test]
    fn worker_selection_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  aggregated: example
  instances:
    - name: example
      type: example-policy
      parameters:
        score_weight: 1.0
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated_instance(), Some("example"));
        let instance = selection.instance("example").unwrap();
        assert_eq!(instance.policy_type(), "example-policy");
        assert!(matches!(
            instance.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
        assert_eq!(
            config
                .resolve_profile(None, Some(2.0), RouterQueuePolicy::Wspt)
                .default_class()
                .queue_policy,
            RouterQueuePolicy::Wspt
        );
    }

    #[test]
    fn request_classifier_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
request_classifier:
  type: thunderagent
  parameters:
    pause_threshold: 0.9
"#,
        )
        .unwrap();

        let classifier = config.request_classifier().unwrap();
        assert_eq!(classifier.classifier_type(), "thunderagent");
        assert!(matches!(
            classifier.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
    }

    #[test]
    fn rejects_invalid_request_classifier_config() {
        for yaml in [
            r#"
request_classifier:
  type: default
"#,
            r#"
request_classifier:
  type: thunderagent
  parameters: 1
"#,
            r#"
request_classifier:
  type: ""
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn worker_selection_accepts_every_worker_type() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  prefill: prefill-policy
  decode: default
  encode: encode-policy
  instances:
    - name: prefill-policy
      type: cache-aware
      parameters: {}
    - name: encode-policy
      type: media-aware
      parameters: {}
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated_instance(), None);
        assert_eq!(selection.prefill_instance(), Some("prefill-policy"));
        assert_eq!(selection.decode_instance(), Some("default"));
        assert_eq!(selection.encode_instance(), Some("encode-policy"));
    }

    #[test]
    fn rejects_invalid_worker_selection_config() {
        for yaml in [
            r#"
worker_selection: {}
"#,
            r#"
worker_selection:
  aggregated: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  prefill: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  decode: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  encode: missing
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  default: present
  instances:
    - name: present
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: default
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: alpha
      type: alpha
      parameters: 1
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: root-default
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_class: model-direct
    policy_classes:
      - name: model-queued
        quantum: 4
        prefill_busy_threshold_frac: 0.0
      - name: model-direct
        quantum: 2
        request_queue_limit_per_worker: 0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-direct");
        assert_eq!(
            exact
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
        assert!(!exact.default_class().queueing_enabled());
        assert!(
            exact
                .class(exact.resolve_class_index(Some("model-queued")).unwrap())
                .queueing_enabled()
        );
        // A configured class without queue_policy is EDF, whatever the router-wide policy.
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Edf);
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        assert_eq!(
            exact.class(exact.resolve_class_index(None).unwrap()).name,
            "model-direct"
        );
        for requested in ["unknown", "root-default"] {
            assert!(
                exact.resolve_class_index(Some(requested)).is_err(),
                "{requested:?} must not resolve in the model profile"
            );
        }

        let unmatched = config.resolve_profile(Some("other"), Some(3.0), RouterQueuePolicy::Fcfs);
        assert_eq!(unmatched.default_class().name, "root-default");
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold,
            Some(100)
        );
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
    }

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root_deprecated() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: root-default
    policy_family: standard
    cache_bucket: all
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_family: latency
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: cached
      - min_tokens: 32
        bucket: uncached
    policy_classes:
      - name: model-cached
        policy_family: latency
        cache_bucket: cached
        quantum: 2
        request_queue_limit_per_worker: 0
      - name: model-uncached
        policy_family: latency
        cache_bucket: uncached
        quantum: 4
        prefill_busy_threshold_frac: 0.0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-cached");
        assert_eq!(
            exact
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
        assert!(!exact.default_class().queueing_enabled());
        assert!(
            exact
                .class(
                    exact
                        .resolve_class_index_for_uncached_tokens(None, usize::MAX)
                        .unwrap()
                )
                .queueing_enabled()
        );
        // A deprecated class without queue_policy is FCFS, whatever the router-wide policy.
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Fcfs);
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        assert_eq!(
            exact
                .class(
                    exact
                        .resolve_class_index_for_uncached_tokens(Some("unknown"), usize::MAX)
                        .unwrap()
                )
                .name,
            "model-uncached",
            "unknown policies must use the model's default family and observed bucket"
        );

        let unmatched = config.resolve_profile(Some("other"), Some(3.0), RouterQueuePolicy::Fcfs);
        assert_eq!(unmatched.default_class().name, "root-default");
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold,
            Some(100)
        );
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
    }

    #[test]
    fn has_room_without_limits_or_reported_budget() {
        let mut limits = BusyThresholds::default();
        assert!(!limits.is_configured());
        assert!(limits.has_room(&prefill_load(usize::MAX), WorkerCapacity::default()));

        // An unreported budget is taken as DEFAULT_MAX_BATCHED_TOKENS.
        limits.prefill_busy_threshold_frac = Some(0.5);
        let half_default =
            (crate::plugins::worker_selection::inputs::DEFAULT_MAX_BATCHED_TOKENS / 2) as usize;
        assert!(limits.has_room(&prefill_load(half_default), WorkerCapacity::default()));
        assert!(!limits.has_room(&prefill_load(half_default + 1), WorkerCapacity::default()));
    }

    #[test]
    fn decode_limit_bounds_held_kv_blocks() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: decode-bound
    quantum: 4
    decode_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let class = profile.default_class();
        assert!(class.queueing_enabled());

        let limits = class.busy_thresholds;
        let held = |active_decode_blocks| {
            WorkerLoadInput::from_projection(WorkerLoadProjection {
                active_decode_blocks,
                // The incoming request's blocks do not count toward the limit.
                additional_active_blocks: 1_000,
                ..Default::default()
            })
        };
        let kv = |total_kv_blocks| WorkerCapacity::new(None, total_kv_blocks);
        assert!(limits.has_room(&held(50), kv(Some(100))));
        assert!(!limits.has_room(&held(51), kv(Some(100))));
        // An unreported KV capacity skips the decode limit.
        assert!(limits.has_room(&held(1_000_000), kv(None)));

        let invalid = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: decode-bound
    quantum: 4
    decode_busy_threshold_frac: -0.1
"#,
        )
        .unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("decode_busy_threshold_frac must be finite and non-negative")
        );
    }

    #[test]
    fn decode_limit_bounds_held_kv_blocks_deprecated() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
    quantum: 4
    decode_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let class = profile.default_class();
        assert!(class.queueing_enabled());

        let limits = class.busy_thresholds;
        let held = |active_decode_blocks| {
            WorkerLoadInput::from_projection(WorkerLoadProjection {
                active_decode_blocks,
                // The incoming request's blocks do not count toward the limit.
                additional_active_blocks: 1_000,
                ..Default::default()
            })
        };
        let kv = |total_kv_blocks| WorkerCapacity::new(None, total_kv_blocks);
        assert!(limits.has_room(&held(50), kv(Some(100))));
        assert!(!limits.has_room(&held(51), kv(Some(100))));
        // An unreported KV capacity skips the decode limit.
        assert!(limits.has_room(&held(1_000_000), kv(None)));

        let invalid = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
    quantum: 4
    decode_busy_threshold_frac: -0.1
"#,
        )
        .unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("decode_busy_threshold_frac must be finite and non-negative")
        );
    }

    #[test]
    fn rootless_model_config_falls_back_for_unmatched_model() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
models:
  exact-model:
    policy_classes:
      - name: absolute
        quantum: 4
        prefill_busy_threshold: 10
        prefill_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(7.0), RouterQueuePolicy::Wspt);
        let limits = exact.default_class().busy_thresholds;
        let budget = |tokens| WorkerCapacity::new(Some(tokens), None);
        assert!(!limits.has_room(&prefill_load(11), budget(10_000_000)));
        assert!(!limits.has_room(&prefill_load(6), budget(10)));
        assert!(limits.has_room(&prefill_load(5), budget(10)));

        let fallback = config.resolve_profile(Some("other"), Some(7.0), RouterQueuePolicy::Wspt);
        assert_eq!(fallback.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            fallback
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(7.0)
        );
        assert_eq!(
            fallback.default_class().queue_policy,
            RouterQueuePolicy::Wspt
        );
        // Only an omitted name or the synthetic class's own name selects it.
        for requested in [None, Some(SYNTHETIC_POLICY_CLASS)] {
            assert_eq!(fallback.resolve_class_index(requested).ok(), Some(0));
        }
        assert!(fallback.resolve_class_index(Some("absolute")).is_err());
        assert_eq!(
            fallback.resolve_class_index_strict(SYNTHETIC_POLICY_CLASS, 0),
            Some(0)
        );
        assert_eq!(fallback.resolve_class_index_strict("absolute", 0), None);
    }

    #[test]
    fn rootless_model_config_falls_back_for_unmatched_model_deprecated() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
models:
  exact-model:
    default_policy_family: standard
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: absolute
        policy_family: standard
        cache_bucket: all
        quantum: 4
        prefill_busy_threshold: 10
        prefill_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(7.0), RouterQueuePolicy::Wspt);
        let limits = exact.default_class().busy_thresholds;
        let budget = |tokens| WorkerCapacity::new(Some(tokens), None);
        assert!(!limits.has_room(&prefill_load(11), budget(10_000_000)));
        assert!(!limits.has_room(&prefill_load(6), budget(10)));
        assert!(limits.has_room(&prefill_load(5), budget(10)));

        let fallback = config.resolve_profile(Some("other"), Some(7.0), RouterQueuePolicy::Wspt);
        assert_eq!(fallback.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            fallback
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(7.0)
        );
        assert_eq!(
            fallback.default_class().queue_policy,
            RouterQueuePolicy::Wspt
        );

        // The deprecated model profile sends an unknown name to its default family, but the
        // synthetic fallback still selects only an omitted name or its own class name.
        assert_eq!(
            exact
                .class(
                    exact
                        .resolve_class_index_for_uncached_tokens(Some("unknown"), 0)
                        .unwrap()
                )
                .name,
            "absolute"
        );
        for requested in [None, Some(SYNTHETIC_POLICY_CLASS)] {
            assert_eq!(fallback.resolve_class_index(requested).ok(), Some(0));
        }
        for requested in ["absolute", "standard", "unknown"] {
            assert!(
                matches!(
                    fallback.resolve_class_index_for_uncached_tokens(Some(requested), 0),
                    Err(KvSchedulerError::UnknownPolicyClass(name)) if name == requested
                ),
                "{requested:?}"
            );
            assert_eq!(fallback.resolve_class_index_strict(requested, 0), None);
        }
    }

    #[test]
    fn classes_resolve_by_name_and_only_omitted_names_use_the_default() {
        let profile = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 800
  - name: standard
    quantum: 200
  - name: batch
    quantum: 100
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs);

        // Every configured class is its own queue, selected by its exact name.
        for (index, name) in ["priority", "standard", "batch"].into_iter().enumerate() {
            assert_eq!(profile.class(index).name, name);
            assert_eq!(profile.resolve_class_index(Some(name)).ok(), Some(index));
            assert_eq!(profile.resolve_class_index_strict(name, 0), Some(index));
        }
        // An omitted name uses the configured default, not the first class. Any other name
        // must match a configured class exactly.
        assert_eq!(profile.resolve_class_index(None).ok(), Some(1));
        for requested in ["", "unknown", "Priority"] {
            assert!(
                matches!(
                    profile.resolve_class_index(Some(requested)),
                    Err(KvSchedulerError::UnknownPolicyClass(name)) if name == requested
                ),
                "{requested:?}"
            );
            assert_eq!(profile.resolve_class_index_strict(requested, 0), None);
        }

        // One class is the default without default_policy_class, and may also name itself.
        for default in ["", "default_policy_class: only\n"] {
            let single = RouterPolicyConfig::from_yaml(&format!(
                "{default}policy_classes:\n  - name: only\n    quantum: 1\n"
            ))
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs);
            assert_eq!(single.default_class().name, "only");
            assert_eq!(single.resolve_class_index(Some("only")).ok(), Some(0));
            assert!(single.resolve_class_index(Some("unknown")).is_err());
        }
    }

    #[test]
    fn rejects_invalid_profiles() {
        for (yaml, expected) in [
            (
                "policy_classes: []\n",
                "root policy_classes must not be empty",
            ),
            (
                "policy_classes:\n  - name: first\n    quantum: 1\n  - name: second\n    quantum: 2\n",
                "root default_policy_class is required with more than one policy class",
            ),
            (
                "default_policy_class: missing\npolicy_classes:\n  - name: only\n    quantum: 1\n",
                "root default_policy_class \"missing\" does not name a configured policy class",
            ),
            (
                "default_policy_class: only\n",
                "root default_policy_class requires policy_classes",
            ),
            (
                "default_policy_class: same\npolicy_classes:\n  - name: same\n    quantum: 1\n  - name: same\n    quantum: 2\n",
                "root contains duplicate policy class \"same\"",
            ),
            (
                "policy_classes:\n  - name: invalid/name\n    quantum: 1\n",
                "root policy class name \"invalid/name\" must match [A-Za-z0-9_.-]+",
            ),
            (
                "policy_classes:\n  - name: zero\n    quantum: 0\n",
                "root policy class \"zero\" quantum must be greater than zero",
            ),
            (
                "policy_classes:\n  - name: lcfs\n    queue_policy: lcfs\n    quantum: 1\n",
                "root policy class \"lcfs\" queue_policy must be edf, fcfs, or wspt",
            ),
            (
                "models:\n  m:\n    policy_classes:\n      - name: a\n        quantum: 1\n      - name: b\n        quantum: 1\n",
                "model \"m\" default_policy_class is required with more than one policy class",
            ),
            (
                "models:\n  m:\n    default_policy_class: absent\n    policy_classes:\n      - name: a\n        quantum: 1\n",
                "model \"m\" default_policy_class \"absent\" does not name a configured policy class",
            ),
        ] {
            let error = RouterPolicyConfig::from_yaml(yaml).unwrap_err().to_string();
            assert!(error.contains(expected), "{yaml}: {error}");
        }
    }

    /// The deprecated sample policy, as `examples/router/policy-class-queues.yaml` shipped it
    /// before flat classes.
    const FAMILY_BUCKET_SAMPLE: &str = r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 3072
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    queue_policy: wspt
    quantum: 2048
    prefill_busy_threshold_frac: 16.0
    request_queue_limit_per_worker: 1024
    cached_token_queue_limit_per_worker: 2097152
  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    queue_policy: fcfs
    quantum: 512
    prefill_busy_threshold_frac: 16.0
    request_queue_limit_per_worker: 512
    raw_isl_token_queue_limit_per_worker: 1048576
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    queue_policy: wspt
    quantum: 4096
    prefill_busy_threshold_frac: 16.0
    request_queue_limit_per_worker: 256
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    queue_policy: wspt
    quantum: 1024
    prefill_busy_threshold_frac: 16.0
    request_queue_limit_per_worker: 128
    raw_isl_token_queue_limit_per_worker: 524288
  - name: custom_priority
    queue_policy: wspt
    quantum: 4096
    prefill_busy_threshold_frac: 16.0
    request_queue_limit_per_worker: 128

models:
  example/large-model:
    default_policy_family: latency
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: cached
      - min_tokens: 4096
        bucket: uncached
    policy_classes:
      - name: latency_cached
        policy_family: latency
        cache_bucket: cached
        queue_policy: wspt
        quantum: 4096
        prefill_busy_threshold_frac: 16.0
        request_queue_limit_per_worker: 256
      - name: latency_uncached
        policy_family: latency
        cache_bucket: uncached
        queue_policy: wspt
        quantum: 1024
        prefill_busy_threshold_frac: 16.0
        request_queue_limit_per_worker: 128
      - name: batch_cached
        policy_family: batch
        cache_bucket: cached
        queue_policy: fcfs
        quantum: 2048
        prefill_busy_threshold_frac: 16.0
        cached_token_queue_limit_per_worker: 1048576
      - name: batch_uncached
        policy_family: batch
        cache_bucket: uncached
        queue_policy: fcfs
        quantum: 1024
        prefill_busy_threshold_frac: 16.0
        raw_isl_token_queue_limit_per_worker: 524288
"#;

    /// Runs `load` on this thread and counts the family/bucket deprecation warnings it logs.
    fn family_bucket_warnings<T>(load: impl FnOnce() -> T) -> (T, usize) {
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counter(Arc<AtomicUsize>);
        impl tracing::Subscriber for Counter {
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                struct Message(Option<String>);
                impl tracing::field::Visit for Message {
                    fn record_debug(
                        &mut self,
                        field: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        if field.name() == "message" {
                            self.0 = Some(format!("{value:?}"));
                        }
                    }
                }
                let mut message = Message(None);
                event.record(&mut message);
                if *event.metadata().level() == tracing::Level::WARN
                    && message.0.as_deref() == Some(FAMILY_BUCKET_DEPRECATION)
                {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }

        // A permanently registered dispatcher makes tracing compute callsite interest over
        // every live dispatcher, so a test thread without a subscriber cannot cache the
        // warning's callsite as disabled for this one.
        static IDLE: OnceLock<tracing::Dispatch> = OnceLock::new();
        IDLE.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()));
        let count = Arc::new(AtomicUsize::new(0));
        let result = tracing::subscriber::with_default(Counter(Arc::clone(&count)), load);
        (result, count.load(Ordering::Relaxed))
    }

    #[test]
    fn family_bucket_keys_never_load_as_flat_classes() {
        // Any deprecated key selects that format even when null or empty, so the profile then
        // needs every key of that format; it fails instead of loading as flat classes.
        let assert_rejected = |yaml: String| {
            let error = RouterPolicyConfig::from_yaml(&yaml).unwrap_err();
            assert!(
                matches!(error, RouterPolicyConfigError::Parse { .. })
                    || error.to_string().contains(
                        "profile must specify default_policy_family, uncached_isl_buckets, and policy_classes"
                    ),
                "{yaml}: {error}"
            );
        };
        for value in ["", "null", "\"\"", "[]", "standard"] {
            for key in ["default_policy_family", "uncached_isl_buckets"] {
                assert_rejected(format!(
                    "{key}: {value}\npolicy_classes:\n  - name: c\n    quantum: 1\n"
                ));
                assert_rejected(format!(
                    "models:\n  m:\n    {key}: {value}\n    policy_classes:\n      - name: c\n        quantum: 1\n"
                ));
            }
            for key in ["policy_family", "cache_bucket"] {
                assert_rejected(format!(
                    "policy_classes:\n  - name: c\n    quantum: 1\n    {key}: {value}\n"
                ));
                assert_rejected(format!(
                    "models:\n  m:\n    policy_classes:\n      - name: c\n        quantum: 1\n        {key}: {value}\n"
                ));
            }
        }

        // One profile cannot select classes both ways.
        let family_bucket = "default_policy_family: f\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: all\npolicy_classes:\n  - name: c\n    policy_family: f\n    cache_bucket: all\n    quantum: 1\n";
        for (yaml, expected) in [
            (
                format!("default_policy_class: c\n{family_bucket}"),
                "root default_policy_class cannot be combined with policy families or cache buckets",
            ),
            (
                format!(
                    "models:\n  m:\n    default_policy_class: c\n{}",
                    family_bucket
                        .lines()
                        .map(|line| format!("    {line}\n"))
                        .collect::<String>()
                ),
                "model \"m\" default_policy_class cannot be combined with policy families or cache buckets",
            ),
        ] {
            let error = RouterPolicyConfig::from_yaml(&yaml)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{yaml}: {error}");
        }
    }

    #[test]
    fn family_bucket_profiles_select_classes_by_family_and_uncached_tokens() {
        let (config, warnings) =
            family_bucket_warnings(|| RouterPolicyConfig::from_yaml(FAMILY_BUCKET_SAMPLE).unwrap());
        assert_eq!(warnings, 1);

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Edf);
        assert!(root.uses_family_buckets());
        let classes = root
            .classes()
            .iter()
            .map(|class| (class.name.as_str(), class.queue_policy, class.quantum))
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            [
                ("cached", RouterQueuePolicy::Wspt, 2048),
                ("uncached", RouterQueuePolicy::Fcfs, 512),
                ("latency_cached", RouterQueuePolicy::Wspt, 4096),
                ("latency_uncached", RouterQueuePolicy::Wspt, 1024),
                ("custom_priority", RouterQueuePolicy::Wspt, 4096),
            ]
        );
        assert!(root.classes().iter().all(|class| {
            class.queueing_enabled() && class.slo_ttft_ms.is_none() && !class.controlled_delay
        }));
        assert_eq!(root.default_class().name, "cached");
        assert_eq!(
            root.default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(16.0)
        );

        // A family, else the default family for any other name or none, picks the class for
        // the request's uncached tokens. Names never fail.
        let class = |requested, uncached_tokens| {
            root.class(
                root.resolve_class_index_for_uncached_tokens(requested, uncached_tokens)
                    .unwrap(),
            )
            .name
            .as_str()
        };
        for (requested, uncached_tokens, expected) in [
            (Some("latency"), 0, "latency_cached"),
            (Some("latency"), 3071, "latency_cached"),
            (Some("latency"), 3072, "latency_uncached"),
            (Some("latency"), usize::MAX, "latency_uncached"),
            (Some("standard"), 3072, "uncached"),
            (None, 0, "cached"),
            (None, 3071, "cached"),
            (None, 3072, "uncached"),
            (None, usize::MAX, "uncached"),
            (Some("unknown"), 0, "cached"),
            (Some("unknown"), 3072, "uncached"),
            (Some(""), 3072, "uncached"),
            // A family-bound class's own name is not a selector: the bucket still decides.
            (Some("cached"), usize::MAX, "uncached"),
            (Some("latency_cached"), usize::MAX, "uncached"),
            (Some("uncached"), 0, "cached"),
            // An explicit class bypasses cache classification.
            (Some("custom_priority"), 0, "custom_priority"),
            (Some("custom_priority"), usize::MAX, "custom_priority"),
        ] {
            assert_eq!(
                class(requested, uncached_tokens),
                expected,
                "{requested:?} with {uncached_tokens} uncached tokens"
            );
        }

        // Without cache state, a name resolves as for zero uncached tokens, the bucket the
        // default class comes from.
        for (requested, expected) in [
            (None, "cached"),
            (Some("unknown"), "cached"),
            (Some("latency"), "latency_cached"),
            (Some("custom_priority"), "custom_priority"),
        ] {
            assert_eq!(
                root.class(root.resolve_class_index(requested).unwrap())
                    .name,
                expected,
                "{requested:?}"
            );
        }
        // Only an explicit class resolves without cache state.
        assert_eq!(
            root.direct_class_index(Some("custom_priority")).unwrap(),
            Some(4)
        );
        for requested in [None, Some("latency"), Some("cached"), Some("unknown")] {
            assert_eq!(
                root.direct_class_index(requested).unwrap(),
                None,
                "{requested:?}"
            );
        }
        // A classifier override must name a family or an explicit class.
        assert_eq!(root.resolve_class_index_strict("latency", 0), Some(2));
        assert_eq!(root.resolve_class_index_strict("latency", 3072), Some(3));
        assert_eq!(
            root.resolve_class_index_strict("custom_priority", 3072),
            Some(4)
        );
        for requested in ["cached", "latency_uncached", "unknown", ""] {
            assert_eq!(
                root.resolve_class_index_strict(requested, 0),
                None,
                "{requested:?}"
            );
        }

        // A model profile replaces the root, with its own default family and boundaries.
        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Fcfs,
        );
        assert!(model.uses_family_buckets());
        assert_eq!(model.classes().len(), 4);
        assert_eq!(model.default_class().name, "latency_cached");
        let class = |requested, uncached_tokens| {
            model
                .class(
                    model
                        .resolve_class_index_for_uncached_tokens(requested, uncached_tokens)
                        .unwrap(),
                )
                .name
                .as_str()
        };
        assert_eq!(class(Some("unknown"), 4095), "latency_cached");
        assert_eq!(class(Some("unknown"), 4096), "latency_uncached");
        assert_eq!(class(Some("batch"), 0), "batch_cached");
        assert_eq!(class(Some("batch"), 4096), "batch_uncached");
        assert_eq!(class(Some("standard"), 0), "latency_cached");
        assert_eq!(class(Some("custom_priority"), 0), "latency_cached");
        assert_eq!(model.resolve_class_index_strict("custom_priority", 0), None);
        assert_eq!(
            config
                .resolve_profile(Some("other"), None, RouterQueuePolicy::Fcfs)
                .default_class()
                .name,
            "cached",
            "an unmatched model uses the root profile"
        );
    }

    #[test]
    fn family_bucket_classes_keep_fcfs_and_reject_flat_only_settings() {
        let class = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "default_policy_family: f\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: all\npolicy_classes:\n  - name: c\n    policy_family: f\n    cache_bucket: all\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Wspt)
                    .default_class()
                    .clone()
            })
            .map_err(|error| error.to_string())
        };

        // Omitted, the order is FCFS, the format's own default, whatever the router-wide policy.
        assert_eq!(class("").unwrap().queue_policy, RouterQueuePolicy::Fcfs);
        for policy in [RouterQueuePolicy::Fcfs, RouterQueuePolicy::Wspt] {
            assert_eq!(
                class(&format!("    queue_policy: {policy}\n"))
                    .unwrap()
                    .queue_policy,
                policy
            );
        }
        for policy in [RouterQueuePolicy::Edf, RouterQueuePolicy::Lcfs] {
            let error = class(&format!("    queue_policy: {policy}\n")).unwrap_err();
            assert!(
                error.contains(
                    "root policy class \"c\" queue_policy must be fcfs or wspt in a policy-family profile"
                ),
                "{error}"
            );
        }
        // EDF settings are rejected, even when they would be off.
        for (setting, field) in [
            ("    slo_ttft_ms: 1000\n", "slo_ttft_ms"),
            ("    slo_ttft_ms: null\n", "slo_ttft_ms"),
            ("    controlled_delay: true\n", "controlled_delay"),
            ("    controlled_delay: false\n", "controlled_delay"),
        ] {
            let error = class(setting).unwrap_err();
            assert!(
                error.contains(&format!(
                    "root policy class \"c\" {field} requires flat policy classes"
                )),
                "{setting}: {error}"
            );
        }

        // Each profile keeps its own format and default order; the synthetic fallback keeps
        // the router-wide policy.
        let config = RouterPolicyConfig::from_yaml(
            r#"
models:
  family-model:
    default_policy_family: f
    uncached_isl_buckets:
      - min_tokens: 0
        bucket: all
    policy_classes:
      - name: bound
        policy_family: f
        cache_bucket: all
        quantum: 1
      - name: unbound
        policy_family: null
        cache_bucket: null
        quantum: 1
  flat-model:
    policy_classes:
      - name: flat
        quantum: 1
"#,
        )
        .unwrap();
        let profile = |model| config.resolve_profile(Some(model), None, RouterQueuePolicy::Lcfs);
        let family = profile("family-model");
        assert!(family.uses_family_buckets());
        assert!(
            family
                .classes()
                .iter()
                .all(|class| class.queue_policy == RouterQueuePolicy::Fcfs)
        );
        // A null family and bucket leave a class unbound, so its name selects it directly.
        assert_eq!(family.direct_class_index(Some("unbound")).unwrap(), Some(1));
        let flat = profile("flat-model");
        assert!(!flat.uses_family_buckets());
        assert_eq!(flat.default_class().queue_policy, RouterQueuePolicy::Edf);
        let synthetic = profile("other");
        assert_eq!(synthetic.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            synthetic.default_class().queue_policy,
            RouterQueuePolicy::Lcfs
        );
    }

    #[test]
    fn null_family_bucket_root_keys_without_classes_configure_no_root() {
        // A null root key is unset when the root sets no classes, so such a file loads without
        // a root profile; only its model profiles can be deprecated.
        let model = "models:\n  m:\n    default_policy_family: f\n    uncached_isl_buckets:\n      - min_tokens: 0\n        bucket: all\n    policy_classes:\n      - name: c\n        policy_family: f\n        cache_bucket: all\n        quantum: 1\n";
        for (yaml, expected_warnings) in [
            (
                format!(
                    "default_policy_family: null\nuncached_isl_buckets: null\npolicy_classes: null\n{model}"
                ),
                1,
            ),
            (
                "default_policy_family: ~\nrouter:\n  router_ttl_secs: 30\n".to_string(),
                0,
            ),
            (
                "uncached_isl_buckets: null\nworker_selection:\n  aggregated: example\n  instances:\n    - name: example\n      type: example-policy\n".to_string(),
                0,
            ),
        ] {
            let (config, warnings) =
                family_bucket_warnings(|| RouterPolicyConfig::from_yaml(&yaml));
            let config = config.unwrap();
            assert_eq!(warnings, expected_warnings, "{yaml}");
            assert!(config.root.is_none(), "{yaml}");
            assert_eq!(
                config
                    .resolve_profile(Some("other"), None, RouterQueuePolicy::Fcfs)
                    .default_class()
                    .name,
                SYNTHETIC_POLICY_CLASS,
                "{yaml}"
            );
        }

        // With root classes, a null key still makes an incomplete deprecated root, never a
        // flat one.
        for key in ["default_policy_family", "uncached_isl_buckets"] {
            let yaml = format!("{key}: null\npolicy_classes:\n  - name: c\n    quantum: 1\n");
            let error = RouterPolicyConfig::from_yaml(&yaml)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(
                    "root profile must specify default_policy_family, uncached_isl_buckets, and policy_classes"
                ),
                "{yaml}: {error}"
            );
        }
    }

    #[test]
    fn rejects_invalid_family_bucket_profiles() {
        let profile = |buckets: &str, classes: &str| {
            format!(
                "default_policy_family: standard\nuncached_isl_buckets:\n{buckets}policy_classes:\n{classes}"
            )
        };
        let cached = "  - min_tokens: 0\n    bucket: cached\n";
        let cached_uncached =
            "  - min_tokens: 0\n    bucket: cached\n  - min_tokens: 32\n    bucket: uncached\n";
        let standard_cached = "  - name: cached\n    policy_family: standard\n    cache_bucket: cached\n    quantum: 1\n";
        for (yaml, expected) in [
            (
                profile(cached_uncached, standard_cached),
                "root is missing a policy class for family \"standard\" and bucket \"uncached\"",
            ),
            (
                profile(
                    cached,
                    "  - name: first\n    policy_family: standard\n    cache_bucket: cached\n    quantum: 1\n  - name: second\n    policy_family: standard\n    cache_bucket: cached\n    quantum: 2\n",
                ),
                "root contains duplicate policy classes for family \"standard\" and bucket \"cached\"",
            ),
            (
                profile(
                    cached,
                    "  - name: same\n    policy_family: standard\n    cache_bucket: cached\n    quantum: 1\n  - name: same\n    quantum: 1\n",
                ),
                "root contains duplicate policy class \"same\"",
            ),
            (
                profile(
                    cached,
                    "  - name: invalid-family\n    policy_family: invalid/family\n    cache_bucket: cached\n    quantum: 1\n",
                ),
                "root policy family name \"invalid/family\" must match [A-Za-z0-9_.-]+",
            ),
            (
                profile(
                    cached,
                    "  - name: missing-bucket\n    policy_family: standard\n    cache_bucket: absent\n    quantum: 1\n",
                ),
                "root policy class \"missing-bucket\" references unknown cache bucket \"absent\"",
            ),
            (
                profile(
                    cached,
                    "  - name: partial\n    policy_family: standard\n    quantum: 1\n",
                ),
                "root policy class \"partial\" must specify both policy_family and cache_bucket or neither for an explicit class",
            ),
            (
                "default_policy_family: priority\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: cached\npolicy_classes:\n  - name: priority\n    quantum: 1\n  - name: paired\n    policy_family: priority\n    cache_bucket: cached\n    quantum: 1\n".to_string(),
                "root explicit policy class \"priority\" collides with a policy family",
            ),
            (
                profile(cached, "  - name: explicit\n    quantum: 1\n"),
                "root default_policy_family \"standard\" does not name a configured family",
            ),
            (
                profile("  - min_tokens: 1\n    bucket: cached\n", standard_cached),
                "root uncached_isl_buckets must start at min_tokens 0",
            ),
            (
                profile(
                    "  - min_tokens: 0\n    bucket: cached\n  - min_tokens: 32\n    bucket: cached\n",
                    standard_cached,
                ),
                "root contains duplicate cache bucket \"cached\"",
            ),
            (
                profile(
                    "  - min_tokens: 0\n    bucket: cached\n  - min_tokens: 64\n    bucket: uncached\n  - min_tokens: 32\n    bucket: large\n",
                    standard_cached,
                ),
                "root uncached_isl_buckets min_tokens must be strictly increasing",
            ),
            (
                profile("", standard_cached).replace("uncached_isl_buckets:\n", "uncached_isl_buckets: []\n"),
                "root uncached_isl_buckets must not be empty",
            ),
            (
                profile(
                    cached,
                    "  - name: zero\n    policy_family: standard\n    cache_bucket: cached\n    quantum: 0\n",
                ),
                "root policy class \"zero\" quantum must be greater than zero",
            ),
            (
                profile(cached, "").replace("policy_classes:\n", "policy_classes: []\n"),
                "root policy_classes must not be empty",
            ),
            (
                "default_policy_family: standard\nuncached_isl_buckets:\n  - min_tokens: 0\n    bucket: cached\n".to_string(),
                "root profile must specify default_policy_family, uncached_isl_buckets, and policy_classes",
            ),
            (
                "models:\n  m:\n    default_policy_family: standard\n    policy_classes:\n      - name: c\n        quantum: 1\n".to_string(),
                "model \"m\" profile must specify default_policy_family, uncached_isl_buckets, and policy_classes",
            ),
        ] {
            let error = RouterPolicyConfig::from_yaml(&yaml).unwrap_err().to_string();
            assert!(error.contains(expected), "{yaml}: {error}");
        }
    }

    #[test]
    fn family_bucket_deprecation_warns_once_per_successful_load() {
        let load = |yaml: &str| family_bucket_warnings(|| RouterPolicyConfig::from_yaml(yaml));
        let flat = "default_policy_class: b\npolicy_classes:\n  - name: a\n    quantum: 1\n  - name: b\n    quantum: 1\n    queue_policy: fcfs\n";
        let family_model = "    default_policy_family: f\n    uncached_isl_buckets:\n      - min_tokens: 0\n        bucket: small\n      - min_tokens: 8\n        bucket: large\n    policy_classes:\n      - name: f_small\n        policy_family: f\n        cache_bucket: small\n        queue_policy: wspt\n        quantum: 1\n        prefill_busy_threshold: 4\n      - name: f_large\n        policy_family: f\n        cache_bucket: large\n        queue_policy: fcfs\n        quantum: 1\n      - name: explicit\n        quantum: 1\n";

        // One warning per load, however many deprecated profiles, classes, and explicit
        // settings it has.
        for yaml in [
            FAMILY_BUCKET_SAMPLE.to_string(),
            format!("{flat}models:\n  m1:\n{family_model}  m2:\n{family_model}"),
            format!(
                "models:\n  m:\n{family_model}  flat:\n    policy_classes:\n      - name: c\n        quantum: 1\n"
            ),
        ] {
            let (config, warnings) = load(&yaml);
            config.unwrap();
            assert_eq!(warnings, 1, "{yaml}");
            assert_eq!(load(&yaml).1, 1, "a second load warns again: {yaml}");
        }

        // Flat, synthetic-only, and failed loads do not warn.
        for yaml in [
            flat.to_string(),
            include_str!("../../../../examples/router/policy-class-queues.yaml").to_string(),
            "worker_selection:\n  aggregated: example\n  instances:\n    - name: example\n      type: example-policy\n".to_string(),
            "router:\n  router_ttl_secs: 30\n".to_string(),
        ] {
            let (config, warnings) = load(&yaml);
            config.unwrap();
            assert_eq!(warnings, 0, "{yaml}");
        }
        for yaml in [
            FAMILY_BUCKET_SAMPLE.replace("quantum: 1024", "quantum: 0"),
            format!("{FAMILY_BUCKET_SAMPLE}request_classifier:\n  type: default\n"),
        ] {
            let (config, warnings) = load(&yaml);
            config.unwrap_err();
            assert_eq!(warnings, 0, "{yaml}");
        }
    }

    #[test]
    fn migration_guide_example_keeps_class_selection_and_order() {
        // The migration guide's example profile before and after migration, verbatim.
        const BEFORE: &str = "default_policy_family: regular
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: premium
    policy_family: premium
    cache_bucket: all
    quantum: 512
  - name: regular
    policy_family: regular
    cache_bucket: all
    quantum: 128
";
        const AFTER: &str = "default_policy_class: regular
policy_classes:
  - name: premium
    queue_policy: fcfs
    quantum: 512
  - name: regular
    queue_policy: fcfs
    quantum: 128
";
        let load = |yaml: &str| {
            let (config, warnings) =
                family_bucket_warnings(|| RouterPolicyConfig::from_yaml(yaml).unwrap());
            (
                config.resolve_profile(None, None, RouterQueuePolicy::Wspt),
                warnings,
            )
        };
        let (before, before_warnings) = load(BEFORE);
        let (after, after_warnings) = load(AFTER);
        assert_eq!((before_warnings, after_warnings), (1, 0));
        let summary = |profile: &PolicyProfile| {
            profile
                .classes()
                .iter()
                .map(|class| (class.name.clone(), class.queue_policy, class.quantum))
                .collect::<Vec<_>>()
        };
        assert_eq!(summary(&before), summary(&after));
        assert!(
            after
                .classes()
                .iter()
                .all(|class| class.queue_policy == RouterQueuePolicy::Fcfs)
        );

        // Clients keep sending the same values, whatever their cache state.
        for (requested, expected) in [
            (None, "regular"),
            (Some("premium"), "premium"),
            (Some("regular"), "regular"),
        ] {
            for profile in [&before, &after] {
                for uncached_tokens in [0, usize::MAX] {
                    let index = profile
                        .resolve_class_index_for_uncached_tokens(requested, uncached_tokens)
                        .unwrap();
                    assert_eq!(profile.class(index).name, expected, "{requested:?}");
                }
            }
        }
        // An unknown name used the default family; a flat profile rejects it.
        assert_eq!(
            before
                .class(before.resolve_class_index(Some("unknown")).unwrap())
                .name,
            "regular"
        );
        assert!(matches!(
            after.resolve_class_index(Some("unknown")),
            Err(KvSchedulerError::UnknownPolicyClass(name)) if name == "unknown"
        ));
        // Without queue_policy, the migrated classes would order by EDF instead.
        let (edf, _) = load(&AFTER.replace("    queue_policy: fcfs\n", ""));
        assert!(
            edf.classes()
                .iter()
                .all(|class| class.queue_policy == RouterQueuePolicy::Edf)
        );
    }

    #[test]
    fn controlled_delay_is_a_per_class_edf_opt_in() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: omitted
policy_classes:
  - name: enabled
    quantum: 1
    slo_ttft_ms: 100
    controlled_delay: true
  - name: omitted
    quantum: 1
    slo_ttft_ms: 100
  - name: disabled
    quantum: 1
    controlled_delay: false
  - name: ordered
    queue_policy: wspt
    quantum: 1
    controlled_delay: false
models:
  m:
    policy_classes:
      - name: enabled
        quantum: 1
        slo_ttft_ms: 100
"#,
        )
        .unwrap();
        let enabled = |model| -> Vec<(String, bool)> {
            config
                .resolve_profile(model, None, RouterQueuePolicy::Fcfs)
                .classes()
                .iter()
                .map(|class| (class.name.clone(), class.controlled_delay))
                .collect()
        };
        let expected = |pairs: &[(&str, bool)]| -> Vec<(String, bool)> {
            pairs
                .iter()
                .map(|&(name, on)| (name.to_string(), on))
                .collect()
        };
        // Each class opts in alone; omitted is off, and any order may say false.
        let root = expected(&[
            ("enabled", true),
            ("omitted", false),
            ("disabled", false),
            ("ordered", false),
        ]);
        assert_eq!(enabled(None), root);
        assert_eq!(enabled(Some("other")), root);
        // A model profile replaces the root wholesale, settings included.
        assert_eq!(enabled(Some("m")), expected(&[("enabled", false)]));
        // Without a configured profile, the synthetic class is off.
        assert!(
            !PolicyProfile::synthetic(Some(0.5), RouterQueuePolicy::Edf)
                .default_class()
                .controlled_delay
        );

        // The setting is a YAML boolean, and only EDF can enforce it.
        let class = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|_| ())
            .map_err(|error| error.to_string())
        };
        for setting in [
            "    controlled_delay: \"true\"\n",
            "    controlled_delay: 1\n",
            "    controlled_delay: null\n",
        ] {
            assert!(class(setting).is_err(), "{setting}");
        }
        for policy in ["fcfs", "wspt"] {
            let error = class(&format!(
                "    queue_policy: {policy}\n    controlled_delay: true\n"
            ))
            .unwrap_err();
            assert!(
                error
                    .contains("root policy class \"c\" controlled_delay requires queue_policy edf"),
                "{error}"
            );
        }
        class("    queue_policy: edf\n    controlled_delay: true\n").unwrap();

        // A queue expires a class budget only where the class opts in and orders by EDF.
        let base = PolicyProfile::synthetic(None, RouterQueuePolicy::Edf)
            .default_class()
            .clone();
        for (queue_policy, controlled_delay, expires) in [
            (RouterQueuePolicy::Edf, true, true),
            (RouterQueuePolicy::Edf, false, false),
            (RouterQueuePolicy::Wspt, true, false),
            (RouterQueuePolicy::Fcfs, true, false),
        ] {
            let class = PolicyClassConfig {
                queue_policy,
                controlled_delay,
                ..base.clone()
            };
            assert_eq!(
                class.expires_class_budget(),
                expires,
                "{queue_policy:?} {controlled_delay}"
            );
        }
    }

    #[test]
    fn slo_ttft_ms_is_an_optional_positive_millisecond_budget() {
        let slo_ttft_ms = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
                    .default_class()
                    .slo_ttft_ms
            })
        };

        assert_eq!(slo_ttft_ms("").unwrap(), None);
        assert_eq!(slo_ttft_ms("    slo_ttft_ms: null\n").unwrap(), None);
        assert_eq!(
            slo_ttft_ms("    slo_ttft_ms: 1\n").unwrap(),
            NonZeroU32::new(1)
        );
        // The maximum is u32::MAX milliseconds; one more is an ordinary parse error.
        assert_eq!(
            slo_ttft_ms(&format!("    slo_ttft_ms: {}\n", u32::MAX)).unwrap(),
            NonZeroU32::new(u32::MAX)
        );
        assert!(matches!(
            slo_ttft_ms(&format!("    slo_ttft_ms: {}\n", u64::from(u32::MAX) + 1)),
            Err(RouterPolicyConfigError::Parse { .. })
        ));

        let error = slo_ttft_ms("    slo_ttft_ms: 0\n").unwrap_err().to_string();
        assert!(
            error.contains("root policy class \"c\" slo_ttft_ms must be greater than zero"),
            "{error}"
        );
        for malformed in [
            "-1",
            "1.5",
            "\"3000\"",
            "3000ms",
            "18446744073709551616",
            "[]",
        ] {
            assert!(
                matches!(
                    slo_ttft_ms(&format!("    slo_ttft_ms: {malformed}\n")),
                    Err(RouterPolicyConfigError::Parse { .. })
                ),
                "unexpectedly accepted slo_ttft_ms {malformed}"
            );
        }
    }

    #[test]
    fn slo_ttft_due_at_counts_the_budget_from_ingress() {
        let class = |slo_ttft_ms| PolicyClassConfig {
            slo_ttft_ms: NonZeroU32::new(slo_ttft_ms),
            ..PolicyProfile::synthetic(None, RouterQueuePolicy::Edf)
                .default_class()
                .clone()
        };
        let ingress_at = Instant::now();
        assert_eq!(class(0).slo_ttft_due_at(ingress_at), None);
        assert_eq!(
            class(1500).slo_ttft_due_at(ingress_at),
            Some(ingress_at + Duration::from_millis(1500))
        );
        assert_eq!(
            class(u32::MAX).slo_ttft_due_at(ingress_at),
            Some(ingress_at + Duration::from_millis(u64::from(u32::MAX)))
        );
    }

    #[test]
    fn omitted_queue_policy_selects_edf_only_for_configured_classes() {
        let queue_policy = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Wspt)
                    .default_class()
                    .queue_policy
            })
        };

        assert_eq!(queue_policy("").unwrap(), RouterQueuePolicy::Edf);
        for policy in [
            RouterQueuePolicy::Edf,
            RouterQueuePolicy::Fcfs,
            RouterQueuePolicy::Wspt,
        ] {
            assert_eq!(
                queue_policy(&format!("    queue_policy: {policy}\n")).unwrap(),
                policy
            );
        }
        assert!(queue_policy("    queue_policy: lcfs\n").is_err());

        // The synthetic class keeps the router-wide policy: FCFS by default, LCFS included.
        assert_eq!(RouterQueuePolicy::default(), RouterQueuePolicy::Fcfs);
        for policy in [
            RouterQueuePolicy::default(),
            RouterQueuePolicy::Edf,
            RouterQueuePolicy::Lcfs,
        ] {
            assert_eq!(
                PolicyProfile::synthetic(None, policy)
                    .default_class()
                    .queue_policy,
                policy
            );
        }
    }

    #[test]
    fn documented_sample_exercises_root_and_model_profiles() {
        let config = RouterPolicyConfig::from_yaml(include_str!(
            "../../../../examples/router/policy-class-queues.yaml"
        ))
        .unwrap();

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let classes = root
            .classes()
            .iter()
            .map(|class| (class.name.as_str(), class.quantum, class.slo_ttft_ms))
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            [
                ("priority", 800, NonZeroU32::new(1000)),
                ("standard", 200, NonZeroU32::new(3000)),
            ]
        );
        assert!(root.classes().iter().all(|class| {
            class.queue_policy == RouterQueuePolicy::Edf && class.queueing_enabled()
        }));
        assert_eq!(
            root.class(root.resolve_class_index(Some("priority")).unwrap())
                .quantum,
            800
        );
        assert_eq!(
            root.class(root.resolve_class_index(None).unwrap()).name,
            "standard"
        );
        assert!(root.resolve_class_index(Some("unknown")).is_err());

        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Wspt,
        );
        assert_eq!(model.classes().len(), 1);
        assert_eq!(model.default_class().name, "standard");
        assert_eq!(model.default_class().quantum, 1000);
        assert!(
            model.resolve_class_index(Some("priority")).is_err(),
            "model profiles must completely replace root classes"
        );
    }

    #[test]
    fn prefill_budget_defaults_when_unreported() {
        use crate::plugins::worker_selection::inputs::DEFAULT_MAX_BATCHED_TOKENS;
        assert_eq!(
            WorkerCapacity::default().prefill_budget(),
            DEFAULT_MAX_BATCHED_TOKENS
        );
        assert_eq!(WorkerCapacity::new(Some(64), None).prefill_budget(), 64);
        assert_eq!(WorkerCapacity::default().max_num_batched_tokens(), None);
        // Every reported value round-trips; there is no sentinel.
        let max = WorkerCapacity::new(Some(u64::MAX), Some(u64::MAX));
        assert_eq!(max.max_num_batched_tokens(), Some(u64::MAX));
        assert_eq!(max.total_kv_blocks(), Some(u64::MAX));
    }

    #[test]
    fn unobserved_rank_row_is_unavailable_and_idle() {
        let seen = WorkerWithDpRank::new(1, 0);
        let unseen = WorkerWithDpRank::new(1, 1);
        let mut loads = FxHashMap::default();
        loads.insert(
            seen,
            WorkerLoadProjection {
                active_prefill_tokens: 7,
                ..Default::default()
            },
        );
        let unobserved = || WorkerLoadProjection {
            additional_active_blocks: 3,
            ..Default::default()
        };

        let row = WorkerLoadInput::for_worker(&loads, seen, unobserved);
        assert!(row.is_available());
        assert_eq!(row.active_prefill_tokens(), 7);

        let row = WorkerLoadInput::for_worker(&loads, unseen, unobserved);
        assert!(!row.is_available());
        assert_eq!(row.active_prefill_tokens(), 0);
        assert_eq!(row.decode_cost_blocks(), 3.0);
    }

    #[test]
    fn worker_is_full_only_when_every_rank_is_full() {
        let limits = BusyThresholds {
            prefill_busy_threshold: Some(10),
            ..Default::default()
        };
        let config = crate::test_utils::SimpleWorkerConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            ..Default::default()
        };
        let capacity = WorkerCapacity::of(&config);
        let busy = WorkerLoadProjection {
            active_prefill_tokens: 11,
            ..Default::default()
        };
        let mut loads = FxHashMap::default();

        loads.insert(WorkerWithDpRank::new(5, 2), busy);
        // Rank 3 has no projection and counts as idle.
        assert!(limits.rank_is_full(WorkerWithDpRank::new(5, 2), capacity, &loads));
        assert!(!limits.rank_is_full(WorkerWithDpRank::new(5, 3), capacity, &loads));
        assert!(!limits.worker_is_full(5, &config, &loads));

        loads.insert(WorkerWithDpRank::new(5, 3), busy);
        assert!(limits.worker_is_full(5, &config, &loads));

        // A rank outside the worker's range does not stand in for one inside it.
        loads.remove(&WorkerWithDpRank::new(5, 3));
        loads.insert(WorkerWithDpRank::new(5, 0), busy);
        assert!(!limits.worker_is_full(5, &config, &loads));
    }

    #[test]
    fn zero_capacity_is_unreported() {
        let zero = WorkerCapacity::new(Some(0), Some(0));
        assert_eq!(zero.max_num_batched_tokens(), None);
        assert_eq!(zero.total_kv_blocks(), None);
        assert_eq!(zero, WorkerCapacity::default());

        // vLLM under the Ray data-parallel backend registers `total_kv_blocks = 0` as a
        // placeholder; the decode limit skips such a rank instead of calling it full.
        let limits = BusyThresholds {
            decode_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        let held = WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_decode_blocks: 1_000,
            ..Default::default()
        });
        assert!(limits.has_room(&held, WorkerCapacity::new(None, Some(0))));
        assert!(!limits.has_room(&held, WorkerCapacity::new(None, Some(1_000))));

        // A zero prefill budget falls back to the default budget instead of being full at once.
        let limits = BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        assert!(limits.has_room(&prefill_load(1), WorkerCapacity::new(Some(0), None)));
    }
}
