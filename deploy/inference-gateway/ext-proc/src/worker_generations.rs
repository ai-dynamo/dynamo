// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tracks worker generations across DGD rolling updates.
//!
//! Each generation runs under its own Dynamo namespace (`<namespace>-<8 hex>`).
//! Like the Frontend's per-namespace `WorkerSet`s, this builds one generation
//! when a namespace registers its first decode-capable model card and retires
//! it when the last one is removed. Generic over the generation type so it can
//! be tested without a KV router.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryStream, ModelCardInstanceId,
};
use dynamo_runtime::namespace::{NamespaceFilter, NamespacePrefixMode};
use futures::{FutureExt, StreamExt};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

/// Endpoint the EPP routes to on its worker component.
pub(crate) const GENERATE_ENDPOINT: &str = "generate";

const INITIAL_RESUBSCRIBE_BACKOFF: Duration = Duration::from_millis(100);
const MAX_RESUBSCRIBE_BACKOFF: Duration = Duration::from_secs(5);
/// Delay before rebuilding a generation whose build panicked, or restarting
/// a watcher that panicked.
const PANIC_RETRY_DELAY: Duration = if cfg!(test) {
    Duration::from_millis(10)
} else {
    Duration::from_secs(5)
};

/// Live generations, ordered by namespace.
pub(crate) type GenerationSet<G> = Arc<[Arc<G>]>;

/// The worker cards this EPP routes to: its component's `generate` endpoint,
/// in namespaces the filter matches.
#[derive(Clone, Debug)]
pub(crate) struct GenerationScope {
    pub(crate) namespace_filter: NamespaceFilter,
    pub(crate) prefix_mode: NamespacePrefixMode,
    pub(crate) component: String,
}

impl GenerationScope {
    /// Whether the scope holds one DGD's generations, so another model in it
    /// can only be that DGD renamed.
    fn single_deployment(&self) -> bool {
        matches!(self.namespace_filter, NamespaceFilter::Prefix(_))
            && self.prefix_mode == NamespacePrefixMode::WorkerGeneration
    }

    /// `instance`'s card, if it is a base-model card the EPP can route to:
    /// a decode or aggregated worker serving chat or completions.
    fn admit(
        &self,
        instance: &DiscoveryInstance,
    ) -> Option<(ModelCardInstanceId, ModelDeploymentCard)> {
        let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            model_suffix,
            ..
        } = instance
        else {
            return None;
        };
        if *component != self.component
            || endpoint != GENERATE_ENDPOINT
            || model_suffix.is_some()
            || !self
                .namespace_filter
                .matches_with_prefix_mode(namespace, self.prefix_mode)
        {
            return None;
        }
        let card = match instance.deserialize_model::<ModelDeploymentCard>() {
            Ok(card) => card,
            Err(error) => {
                tracing::debug!(%error, namespace, "Failed to deserialize model card, skipping");
                return None;
            }
        };
        let routable = matches!(
            card.effective_worker_type(),
            WorkerType::Decode | WorkerType::Aggregated
        ) && (card.model_type.supports_chat()
            || card.model_type.supports_completions());
        if !routable {
            return None;
        }
        let DiscoveryInstanceId::Model(id) = instance.id() else {
            return None;
        };
        Some((id, card))
    }
}

/// A namespace started or stopped serving the pinned model.
#[derive(Debug)]
pub(crate) enum GenerationChange {
    Appeared {
        namespace: String,
        card: Arc<ModelDeploymentCard>,
    },
    Vanished {
        namespace: String,
    },
}

/// One namespace's in-scope cards.
#[derive(Default)]
struct TrackedNamespace {
    /// Model served by each card.
    cards: HashMap<ModelCardInstanceId, String>,
    /// Card count per model, with one card to build from.
    models: HashMap<String, (usize, Arc<ModelDeploymentCard>)>,
}

impl TrackedNamespace {
    /// Returns the card's model if the card is new.
    fn insert(&mut self, id: ModelCardInstanceId, card: ModelDeploymentCard) -> Option<String> {
        if self.cards.contains_key(&id) {
            return None;
        }
        let model = card.display_name.clone();
        self.cards.insert(id, model.clone());
        self.models
            .entry(model.clone())
            .or_insert_with(|| (0, Arc::new(card)))
            .0 += 1;
        Some(model)
    }

    /// Returns the removed card's model.
    fn remove(&mut self, id: &ModelCardInstanceId) -> Option<String> {
        let model = self.cards.remove(id)?;
        decrement(&mut self.models, &model, |(count, _)| count);
        Some(model)
    }
}

/// Decrement `key`'s count, removing the entry at zero.
fn decrement<V>(map: &mut HashMap<String, V>, key: &str, count: impl Fn(&mut V) -> &mut usize) {
    if let Some(value) = map.get_mut(key) {
        let count = count(value);
        *count -= 1;
        if *count == 0 {
            map.remove(key);
        }
    }
}

/// Routable cards per matching namespace, and the one model this EPP serves.
///
/// The model is pinned when first seen so a broad scope never mixes models.
/// It moves only when none of its cards remain in a single-DGD scope, where
/// another model can only be the same DGD renamed.
pub(crate) struct GenerationTracker {
    scope: GenerationScope,
    namespaces: HashMap<String, TrackedNamespace>,
    /// Card count per model across all namespaces.
    model_counts: HashMap<String, usize>,
    model: Option<String>,
    /// Namespaces last reported as `Appeared`.
    serving: HashSet<String>,
}

impl GenerationTracker {
    pub(crate) fn new(scope: GenerationScope) -> Self {
        Self {
            scope,
            namespaces: HashMap::new(),
            model_counts: HashMap::new(),
            model: None,
            serving: HashSet::new(),
        }
    }

    pub(crate) fn on_added(&mut self, instance: &DiscoveryInstance) -> Vec<GenerationChange> {
        let Some((id, card)) = self.scope.admit(instance) else {
            return Vec::new();
        };
        let namespace = id.namespace.clone();
        let Some(model) = self
            .namespaces
            .entry(namespace.clone())
            .or_default()
            .insert(id, card)
        else {
            return Vec::new();
        };
        *self.model_counts.entry(model).or_default() += 1;
        self.settle(&namespace)
    }

    pub(crate) fn on_removed(&mut self, id: &DiscoveryInstanceId) -> Vec<GenerationChange> {
        let DiscoveryInstanceId::Model(id) = id else {
            return Vec::new();
        };
        let Some(tracked) = self.namespaces.get_mut(&id.namespace) else {
            return Vec::new();
        };
        let Some(model) = tracked.remove(id) else {
            return Vec::new();
        };
        if tracked.cards.is_empty() {
            self.namespaces.remove(&id.namespace);
        }
        decrement(&mut self.model_counts, &model, |count| count);
        self.settle(&id.namespace)
    }

    /// Replace the tracked cards with a full snapshot; report namespace changes.
    pub(crate) fn on_resync(&mut self, instances: &[DiscoveryInstance]) -> Vec<GenerationChange> {
        self.namespaces.clear();
        self.model_counts.clear();
        for instance in instances {
            if let Some((id, card)) = self.scope.admit(instance) {
                let namespace = id.namespace.clone();
                if let Some(model) = self
                    .namespaces
                    .entry(namespace)
                    .or_default()
                    .insert(id, card)
                {
                    *self.model_counts.entry(model).or_default() += 1;
                }
            }
        }
        self.repin();
        self.reconcile_all()
    }

    /// The card to build `namespace`'s generation from, if it serves the
    /// pinned model.
    fn card_for(&self, namespace: &str) -> Option<Arc<ModelDeploymentCard>> {
        let model = self.model.as_ref()?;
        let (_, card) = self.namespaces.get(namespace)?.models.get(model)?;
        Some(card.clone())
    }

    /// Re-evaluate `namespace`, or every namespace if the pinned model moved.
    fn settle(&mut self, namespace: &str) -> Vec<GenerationChange> {
        if self.repin() {
            return self.reconcile_all();
        }
        let mut changes = Vec::new();
        self.reconcile(namespace, &mut changes);
        changes
    }

    /// Keep the pinned model while any of its cards remain, and for good in a
    /// broad scope; otherwise pin the smallest remaining model name. Returns
    /// whether the pin moved.
    fn repin(&mut self) -> bool {
        if let Some(model) = &self.model
            && (self.model_counts.contains_key(model) || !self.scope.single_deployment())
        {
            return false;
        }
        let next = self.model_counts.keys().min().cloned();
        if next == self.model {
            return false;
        }
        let others: Vec<&String> = self
            .model_counts
            .keys()
            .filter(|model| Some(*model) != next.as_ref())
            .collect();
        tracing::info!(from = ?self.model, to = ?next, ignored = ?others, "Serving model pinned");
        self.model = next;
        true
    }

    fn reconcile_all(&mut self) -> Vec<GenerationChange> {
        let namespaces: HashSet<String> = self
            .namespaces
            .keys()
            .chain(self.serving.iter())
            .cloned()
            .collect();
        let mut changes = Vec::new();
        for namespace in namespaces {
            self.reconcile(&namespace, &mut changes);
        }
        changes
    }

    fn reconcile(&mut self, namespace: &str, changes: &mut Vec<GenerationChange>) {
        match (self.card_for(namespace), self.serving.contains(namespace)) {
            (Some(card), false) => {
                self.serving.insert(namespace.to_string());
                changes.push(GenerationChange::Appeared {
                    namespace: namespace.to_string(),
                    card,
                });
            }
            (None, true) => {
                self.serving.remove(namespace);
                changes.push(GenerationChange::Vanished {
                    namespace: namespace.to_string(),
                });
            }
            _ => {}
        }
    }
}

type BuildResult<G> = (String, u64, std::thread::Result<Option<G>>);

/// A build in flight. `seq` fences a stale result after a vanish/reappear.
struct PendingBuild {
    seq: u64,
    cancel: CancellationToken,
    card: Arc<ModelDeploymentCard>,
}

/// Published generations and in-flight builds; owned by the watcher task only.
struct GenerationState<'a, G, B> {
    published: BTreeMap<String, Arc<G>>,
    /// Published generations whose namespace vanished while no other
    /// generation was ready. Kept so a restarting worker does not lose its
    /// generation; retired once a replacement is published.
    dormant: HashSet<String>,
    pending: HashMap<String, PendingBuild>,
    next_seq: u64,
    builds: JoinSet<BuildResult<G>>,
    build: &'a B,
    publish: &'a watch::Sender<GenerationSet<G>>,
    cancel: &'a CancellationToken,
}

impl<G, B> Drop for GenerationState<'_, G, B> {
    fn drop(&mut self) {
        // Stops abandoned builds too, including when the watcher panics.
        for pending in self.pending.values() {
            pending.cancel.cancel();
        }
    }
}

impl<'a, G, B, BF> GenerationState<'a, G, B>
where
    G: Send + Sync + 'static,
    B: Fn(String, ModelDeploymentCard, CancellationToken) -> BF,
    BF: Future<Output = Option<G>> + Send + 'static,
{
    fn new(
        build: &'a B,
        publish: &'a watch::Sender<GenerationSet<G>>,
        cancel: &'a CancellationToken,
    ) -> Self {
        Self {
            published: BTreeMap::new(),
            dormant: HashSet::new(),
            pending: HashMap::new(),
            next_seq: 0,
            builds: JoinSet::new(),
            build,
            publish,
            cancel,
        }
    }

    fn publish(&self) {
        let set: GenerationSet<G> = self.published.values().cloned().collect();
        self.publish.send_replace(set);
    }

    fn apply(&mut self, change: GenerationChange) {
        match change {
            GenerationChange::Appeared { namespace, card } => {
                if self.dormant.remove(&namespace) {
                    tracing::info!(namespace, "Worker generation is serving again");
                } else {
                    tracing::info!(namespace, "Worker generation discovered");
                    self.spawn_build(namespace, card, Duration::ZERO);
                }
            }
            GenerationChange::Vanished { namespace } => self.vanish(&namespace),
        }
    }

    fn spawn_build(&mut self, namespace: String, card: Arc<ModelDeploymentCard>, delay: Duration) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let cancel = self.cancel.child_token();
        let built = (self.build)(namespace.clone(), (*card).clone(), cancel.clone());
        self.pending.insert(
            namespace.clone(),
            PendingBuild {
                seq,
                cancel: cancel.clone(),
                card,
            },
        );
        self.builds.spawn(async move {
            if !delay.is_zero() && !sleep_or_cancel(delay, &cancel).await {
                return (namespace, seq, Ok(None));
            }
            (namespace, seq, AssertUnwindSafe(built).catch_unwind().await)
        });
    }

    fn vanish(&mut self, namespace: &str) {
        if let Some(pending) = self.pending.remove(namespace) {
            pending.cancel.cancel();
        }
        if !self.published.contains_key(namespace) {
            return;
        }
        let replaced = self
            .published
            .keys()
            .any(|other| other != namespace && !self.dormant.contains(other));
        if replaced {
            self.retire(namespace);
        } else {
            tracing::info!(
                namespace,
                "Worker generation has no workers; keeping it until a replacement is ready"
            );
            self.dormant.insert(namespace.to_string());
        }
    }

    fn retire(&mut self, namespace: &str) {
        self.dormant.remove(namespace);
        // In-flight requests hold their own `Arc`, so the generation drains.
        if self.published.remove(namespace).is_some() {
            tracing::info!(namespace, "Worker generation retired");
        }
    }

    fn on_built(&mut self, (namespace, seq, built): BuildResult<G>) {
        let (namespace, pending) = match self.pending.entry(namespace) {
            Entry::Occupied(entry) if entry.get().seq == seq => entry.remove_entry(),
            // Drop results from retired or superseded builds.
            _ => return,
        };
        match built {
            Ok(Some(generation)) => {
                tracing::info!(namespace, "Worker generation ready for routing");
                self.published.insert(namespace, Arc::new(generation));
                // A live generation replaces every dormant one.
                for dormant in std::mem::take(&mut self.dormant) {
                    self.retire(&dormant);
                }
                self.publish();
            }
            Ok(None) => {}
            Err(_) => {
                tracing::error!(namespace, "Worker generation build panicked; retrying");
                pending.cancel.cancel();
                self.spawn_build(namespace, pending.card, PANIC_RETRY_DELAY);
            }
        }
    }
}

/// Publish the live generations until `cancel` fires, restarting the watch if
/// it panics.
///
/// `subscribe` is re-called when the watch fails or ends. `build` runs in its
/// own task and should return `None` once its token is cancelled.
pub(crate) async fn run_generation_watcher<G, S, SF, B, BF>(
    mut subscribe: S,
    scope: GenerationScope,
    build: B,
    publish: watch::Sender<GenerationSet<G>>,
    cancel: CancellationToken,
) where
    G: Send + Sync + 'static,
    S: FnMut(CancellationToken) -> SF,
    SF: Future<Output = anyhow::Result<DiscoveryStream>> + Send + 'static,
    B: Fn(String, ModelDeploymentCard, CancellationToken) -> BF,
    BF: Future<Output = Option<G>> + Send + 'static,
{
    loop {
        let tracker = GenerationTracker::new(scope.clone());
        let watched = watch_generations(&mut subscribe, tracker, &build, &publish, &cancel);
        if AssertUnwindSafe(watched).catch_unwind().await.is_ok() {
            return;
        }
        // The published snapshot keeps serving until the restarted watch
        // publishes again.
        tracing::error!("Worker generation watcher panicked; restarting");
        if !sleep_or_cancel(PANIC_RETRY_DELAY, &cancel).await {
            return;
        }
    }
}

type OpenWatch = Pin<Box<dyn Future<Output = anyhow::Result<DiscoveryStream>> + Send>>;

async fn watch_generations<G, S, SF, B, BF>(
    subscribe: &mut S,
    mut tracker: GenerationTracker,
    build: &B,
    publish: &watch::Sender<GenerationSet<G>>,
    cancel: &CancellationToken,
) where
    G: Send + Sync + 'static,
    S: FnMut(CancellationToken) -> SF,
    SF: Future<Output = anyhow::Result<DiscoveryStream>> + Send + 'static,
    B: Fn(String, ModelDeploymentCard, CancellationToken) -> BF,
    BF: Future<Output = Option<G>> + Send + 'static,
{
    let mut state = GenerationState::new(build, publish, cancel);
    let mut backoff = INITIAL_RESUBSCRIBE_BACKOFF;
    let mut watch_cancel = cancel.child_token();
    let mut opening: Option<OpenWatch> = Some(Box::pin(subscribe(watch_cancel.clone())));
    let mut events: Option<DiscoveryStream> = None;

    loop {
        // Builds are joined in every state, including while (re)subscribing.
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            Some(joined) = state.builds.join_next(), if !state.builds.is_empty() => {
                if let Ok(result) = joined {
                    state.on_built(result);
                }
            }
            opened = async { opening.as_mut().expect("guarded").await }, if opening.is_some() => {
                opening = None;
                match opened {
                    Ok(stream) => events = Some(stream),
                    Err(error) => {
                        tracing::warn!(%error, "Failed to watch model discovery for worker generations");
                        opening = Some(resubscribe(subscribe, &mut watch_cancel, cancel, &mut backoff));
                    }
                }
            }
            event = async { events.as_mut().expect("guarded").next().await }, if events.is_some() => {
                let changes = match event {
                    None => {
                        tracing::warn!("Model discovery watch ended; resubscribing");
                        events = None;
                        opening = Some(resubscribe(subscribe, &mut watch_cancel, cancel, &mut backoff));
                        continue;
                    }
                    Some(Err(error)) => {
                        tracing::warn!(%error, "Model discovery watch error");
                        continue;
                    }
                    Some(Ok(DiscoveryEvent::Added(instance))) => tracker.on_added(&instance),
                    Some(Ok(DiscoveryEvent::Removed(id))) => tracker.on_removed(&id),
                    Some(Ok(DiscoveryEvent::Resync(instances))) => {
                        // A watch that delivered its snapshot is healthy.
                        backoff = INITIAL_RESUBSCRIBE_BACKOFF;
                        tracker.on_resync(&instances)
                    }
                    Some(Ok(DiscoveryEvent::ModelTaintsUpdated(_))) => Vec::new(),
                };
                let published_before = state.published.len();
                for change in changes {
                    state.apply(change);
                }
                if state.published.len() != published_before {
                    state.publish();
                }
            }
        }
    }

    watch_cancel.cancel();
    state.builds.shutdown().await;
}

/// Cancel the current watch and open a new one after a jittered backoff.
fn resubscribe<S, SF>(
    subscribe: &mut S,
    watch_cancel: &mut CancellationToken,
    cancel: &CancellationToken,
    backoff: &mut Duration,
) -> OpenWatch
where
    S: FnMut(CancellationToken) -> SF,
    SF: Future<Output = anyhow::Result<DiscoveryStream>> + Send + 'static,
{
    watch_cancel.cancel();
    *watch_cancel = cancel.child_token();
    let delay = backoff.mul_f64(rand::random_range(0.5..1.0));
    *backoff = (*backoff * 2).min(MAX_RESUBSCRIBE_BACKOFF);
    let opened = subscribe(watch_cancel.clone());
    let watch_cancel = watch_cancel.clone();
    Box::pin(async move {
        if !sleep_or_cancel(delay, &watch_cancel).await {
            return Err(anyhow::anyhow!("watch cancelled"));
        }
        opened.await
    })
}

/// Sleep for `delay`, returning `false` if `cancel` fired first.
async fn sleep_or_cancel(delay: Duration, cancel: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancel.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

/// Choose a generation, weighted by `workers`.
///
/// While any generation's prefill router is active, generations whose router
/// is not yet active are skipped: a new generation's prefill router activates
/// after its decode workers register, and until then it would serve a
/// disaggregated deployment aggregated. With no workers anywhere, the first
/// generation is chosen so its router reports why. `roll(total)` returns a
/// value in `0..total`.
pub(crate) fn choose_generation<T>(
    generations: &[T],
    workers: impl Fn(&T) -> usize,
    prefill_active: impl Fn(&T) -> bool,
    roll: impl Fn(usize) -> usize,
) -> Option<&T> {
    if generations.len() <= 1 {
        return generations.first();
    }
    let disaggregated = generations.iter().any(&prefill_active);
    let disaggregated_workers = |generation: &T| {
        if prefill_active(generation) {
            workers(generation)
        } else {
            0
        }
    };
    disaggregated
        .then(|| pick_weighted(generations, disaggregated_workers, &roll))
        .flatten()
        .or_else(|| pick_weighted(generations, &workers, &roll))
        .or_else(|| generations.first())
}

/// Pick an item with probability proportional to its weight; `roll(total)`
/// returns a value in `0..total`. Returns `None` when every weight is zero.
fn pick_weighted<T>(
    items: &[T],
    weight: impl Fn(&T) -> usize,
    roll: impl FnOnce(usize) -> usize,
) -> Option<&T> {
    let weights: Vec<usize> = items.iter().map(&weight).collect();
    let total: usize = weights.iter().sum();
    if total == 0 {
        return None;
    }
    let mut point = roll(total);
    for (item, weight) in items.iter().zip(weights) {
        if point < weight {
            return Some(item);
        }
        point -= weight;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::*;
    use dynamo_llm::model_type::ModelType;
    use tokio::sync::{mpsc, oneshot};

    const MODEL: &str = "tiny";

    fn card(
        model: &str,
        worker_type: Option<WorkerType>,
        model_type: ModelType,
    ) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(model);
        card.worker_type = worker_type;
        card.model_type = model_type;
        card
    }

    fn chat() -> ModelType {
        ModelType::Chat | ModelType::Completions
    }

    /// A card registered on `namespace/component/endpoint`.
    fn registration(
        namespace: &str,
        component: &str,
        endpoint: &str,
        instance_id: u64,
        card: &ModelDeploymentCard,
    ) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            card_json: serde_json::to_value(card).unwrap(),
            model_suffix: None,
        }
    }

    /// A decode worker's card for `model` on the EPP's `backend/generate`.
    fn decode(namespace: &str, instance_id: u64, model: &str) -> DiscoveryInstance {
        let card = card(model, Some(WorkerType::Decode), chat());
        registration(namespace, "backend", GENERATE_ENDPOINT, instance_id, &card)
    }

    fn scope(
        namespace_filter: NamespaceFilter,
        prefix_mode: NamespacePrefixMode,
    ) -> GenerationScope {
        GenerationScope {
            namespace_filter,
            prefix_mode,
            component: "backend".to_string(),
        }
    }

    /// The operator's scope for a DGD named `dgd`.
    fn dgd() -> GenerationScope {
        scope(
            NamespaceFilter::Prefix("dgd".into()),
            NamespacePrefixMode::WorkerGeneration,
        )
    }

    fn summarize(changes: &[GenerationChange]) -> Vec<String> {
        let mut out: Vec<String> = changes
            .iter()
            .map(|change| match change {
                GenerationChange::Appeared { namespace, .. } => format!("+{namespace}"),
                GenerationChange::Vanished { namespace } => format!("-{namespace}"),
            })
            .collect();
        out.sort();
        out
    }

    /// Only the decode or aggregated cards on the EPP's own endpoint start a
    /// generation; anything else could hand a generation the wrong worker role,
    /// tokenizer, or liveness.
    #[test]
    fn tracker_admits_only_cards_the_epp_routes_to() {
        let ns = "dgd-aaaaaaaa";
        let decode_card = card(MODEL, Some(WorkerType::Decode), chat());
        let mut lora = decode(ns, 1, MODEL);
        if let DiscoveryInstance::Model { model_suffix, .. } = &mut lora {
            *model_suffix = Some("adapter".into());
        }
        let ignored = [
            (
                "another component",
                registration(ns, "prefill", GENERATE_ENDPOINT, 2, &decode_card),
            ),
            (
                "another endpoint",
                registration(ns, "backend", "clear_kv_blocks", 3, &decode_card),
            ),
            (
                "legacy prefill",
                registration(
                    ns,
                    "backend",
                    GENERATE_ENDPOINT,
                    4,
                    &card(MODEL, None, ModelType::Prefill),
                ),
            ),
            (
                "prefill",
                registration(
                    ns,
                    "backend",
                    GENERATE_ENDPOINT,
                    5,
                    &card(MODEL, Some(WorkerType::Prefill), ModelType::empty()),
                ),
            ),
            (
                "encode",
                registration(
                    ns,
                    "backend",
                    GENERATE_ENDPOINT,
                    6,
                    &card(MODEL, Some(WorkerType::Encode), chat()),
                ),
            ),
            (
                "embedding",
                registration(
                    ns,
                    "backend",
                    GENERATE_ENDPOINT,
                    7,
                    &card(MODEL, Some(WorkerType::Aggregated), ModelType::Embedding),
                ),
            ),
            ("LoRA", lora),
        ];
        let mut tracker = GenerationTracker::new(dgd());
        for (what, instance) in &ignored {
            assert!(
                tracker.on_added(instance).is_empty(),
                "{what} must be ignored"
            );
        }

        let changes = tracker.on_added(&decode(ns, 8, MODEL));
        let [GenerationChange::Appeared { card, .. }] = changes.as_slice() else {
            panic!("the decode card must start the generation: {changes:?}");
        };
        assert_eq!(card.worker_type, Some(WorkerType::Decode));
        let aggregated = card_on_backend("dgd-bbbbbbbb", 9, WorkerType::Aggregated);
        assert_eq!(summarize(&tracker.on_added(&aggregated)), ["+dgd-bbbbbbbb"]);
    }

    fn card_on_backend(
        namespace: &str,
        instance_id: u64,
        worker_type: WorkerType,
    ) -> DiscoveryInstance {
        let card = card(MODEL, Some(worker_type), ModelType::Chat);
        registration(namespace, "backend", GENERATE_ENDPOINT, instance_id, &card)
    }

    /// A sibling DGD sharing the prefix is excluded under worker-generation
    /// matching, and an exact scope excludes generations.
    #[test]
    fn tracker_scopes_generations_to_this_deployment() {
        let instances = [
            decode("default-foo-bar", 1, MODEL),
            decode("default-foo-1a2b3c4d", 2, MODEL),
        ];
        let resync = |filter: NamespaceFilter, mode: NamespacePrefixMode| {
            summarize(&GenerationTracker::new(scope(filter, mode)).on_resync(&instances))
        };
        let prefix = || NamespaceFilter::Prefix("default-foo".into());
        assert_eq!(
            resync(prefix(), NamespacePrefixMode::WorkerGeneration),
            ["+default-foo-1a2b3c4d"]
        );
        assert_eq!(
            resync(prefix(), NamespacePrefixMode::Literal),
            ["+default-foo-1a2b3c4d", "+default-foo-bar"]
        );
        assert!(
            resync(
                NamespaceFilter::Exact("default-foo".into()),
                NamespacePrefixMode::WorkerGeneration
            )
            .is_empty()
        );
    }

    /// The rolling update from #16085: `a` is live, `b` appears beside it,
    /// then `a` drains. `b` is reported on its first card, `a` on its last.
    #[test]
    fn tracker_follows_a_rolling_update() {
        let (a1, a2, b) = (
            decode("dgd-aaaaaaaa", 1, MODEL),
            decode("dgd-aaaaaaaa", 2, MODEL),
            decode("dgd-bbbbbbbb", 3, MODEL),
        );
        let mut tracker = GenerationTracker::new(dgd());
        assert_eq!(summarize(&tracker.on_added(&a1)), ["+dgd-aaaaaaaa"]);
        assert!(tracker.on_added(&a2).is_empty(), "one change per namespace");
        assert_eq!(summarize(&tracker.on_added(&b)), ["+dgd-bbbbbbbb"]);

        assert!(tracker.on_removed(&a1.id()).is_empty());
        assert!(
            tracker.on_removed(&a1.id()).is_empty(),
            "a repeated removal is not a second card leaving"
        );
        assert_eq!(summarize(&tracker.on_removed(&a2.id())), ["-dgd-aaaaaaaa"]);

        let all = [a1.clone(), a2.clone(), b.clone()];
        assert_eq!(summarize(&tracker.on_resync(&all)), ["+dgd-aaaaaaaa"]);
        assert_eq!(summarize(&tracker.on_resync(&[b])), ["-dgd-aaaaaaaa"]);
    }

    /// Within one DGD, a model left with no workers gives way to its rename.
    /// In a broad scope another model may be a different deployment, so the
    /// pin holds until the pinned model returns.
    #[test]
    fn tracker_moves_the_pin_only_within_one_deployment() {
        let old = decode("dgd-aaaaaaaa", 1, "model-a");
        let renamed = decode("dgd-bbbbbbbb", 2, "model-b");

        let mut tracker = GenerationTracker::new(dgd());
        assert_eq!(summarize(&tracker.on_added(&old)), ["+dgd-aaaaaaaa"]);
        assert!(tracker.on_added(&renamed).is_empty(), "one model at a time");
        assert_eq!(
            summarize(&tracker.on_removed(&old.id())),
            ["+dgd-bbbbbbbb", "-dgd-aaaaaaaa"]
        );

        let mut tracker = GenerationTracker::new(scope(
            NamespaceFilter::Global,
            NamespacePrefixMode::WorkerGeneration,
        ));
        assert_eq!(summarize(&tracker.on_added(&old)), ["+dgd-aaaaaaaa"]);
        assert!(tracker.on_added(&renamed).is_empty());
        assert_eq!(summarize(&tracker.on_removed(&old.id())), ["-dgd-aaaaaaaa"]);
        assert_eq!(summarize(&tracker.on_added(&old)), ["+dgd-aaaaaaaa"]);
    }

    /// `(name, workers, prefill_active)`, chosen with `roll` capped to the
    /// weight being rolled over.
    fn choose(generations: &[(&'static str, usize, bool)], roll: usize) -> Option<&'static str> {
        choose_generation(generations, |g| g.1, |g| g.2, |total| roll.min(total - 1)).map(|g| g.0)
    }

    #[test]
    fn choose_generation_follows_capacity() {
        let both = [("old", 3, false), ("new", 1, false)];
        assert_eq!(choose(&both, 0), Some("old"));
        assert_eq!(choose(&both, 2), Some("old"));
        assert_eq!(choose(&both, 3), Some("new"));
        assert_eq!(
            choose(&[("old", 0, false), ("new", 4, false)], 0),
            Some("new"),
            "a drained generation gets no traffic"
        );
        assert_eq!(
            choose(&[("old", 0, false), ("new", 0, false)], 0),
            Some("old"),
            "with no workers anywhere the first generation reports why"
        );
        assert_eq!(choose(&[("only", 0, false)], 0), Some("only"));
        assert_eq!(choose(&[], 0), None);
    }

    /// A disaggregated rollout keeps traffic off a generation until its
    /// prefill router is active, unless nothing else has workers.
    #[test]
    fn choose_generation_waits_for_the_prefill_router() {
        assert_eq!(
            choose(&[("old", 2, true), ("new", 6, false)], 7),
            Some("old")
        );
        assert_eq!(
            choose(&[("old", 2, true), ("new", 6, true)], 7),
            Some("new")
        );
        assert_eq!(
            choose(&[("old", 0, true), ("new", 6, false)], 5),
            Some("new"),
            "aggregated beats no route at all"
        );
    }

    type TestBuild = fn(
        String,
        ModelDeploymentCard,
        CancellationToken,
    ) -> futures::future::Pending<Option<String>>;

    /// A build the state-machine tests finish by hand.
    fn manual_build(
        _: String,
        _: ModelDeploymentCard,
        _: CancellationToken,
    ) -> futures::future::Pending<Option<String>> {
        futures::future::pending()
    }

    fn appeared(namespace: &str) -> GenerationChange {
        GenerationChange::Appeared {
            namespace: namespace.to_string(),
            card: Arc::new(card(MODEL, Some(WorkerType::Decode), chat())),
        }
    }

    fn vanished(namespace: &str) -> GenerationChange {
        GenerationChange::Vanished {
            namespace: namespace.to_string(),
        }
    }

    /// Finish `namespace`'s pending build as the generation `value`.
    fn finish(state: &mut GenerationState<'_, String, TestBuild>, namespace: &str, value: &str) {
        let seq = state.pending[namespace].seq;
        state.on_built((namespace.to_string(), seq, Ok(Some(value.to_string()))));
    }

    fn live(state: &GenerationState<'_, String, TestBuild>) -> Vec<String> {
        state.published.values().map(|g| (**g).clone()).collect()
    }

    fn channel() -> watch::Sender<GenerationSet<String>> {
        watch::channel::<GenerationSet<String>>(Arc::from([])).0
    }

    #[tokio::test]
    async fn superseded_build_never_publishes() {
        let (build, publish, cancel): (TestBuild, _, _) =
            (manual_build, channel(), CancellationToken::new());
        let mut state = GenerationState::new(&build, &publish, &cancel);
        state.apply(appeared("dgd-bbbbbbbb"));
        let stale = state.pending["dgd-bbbbbbbb"].seq;
        state.apply(vanished("dgd-bbbbbbbb"));
        state.apply(appeared("dgd-bbbbbbbb"));

        state.on_built(("dgd-bbbbbbbb".into(), stale, Ok(Some("stale".into()))));
        assert!(live(&state).is_empty());
        finish(&mut state, "dgd-bbbbbbbb", "current");
        assert_eq!(live(&state), ["current"]);
        assert_eq!(publish.borrow().len(), 1);
    }

    /// The only generation losing its workers (a restart) is kept and reused
    /// when they return, not rebuilt; a new generation replaces it.
    #[tokio::test]
    async fn only_generation_is_kept_until_replaced() {
        let (build, publish, cancel): (TestBuild, _, _) =
            (manual_build, channel(), CancellationToken::new());
        let mut state = GenerationState::new(&build, &publish, &cancel);
        state.apply(appeared("dgd-aaaaaaaa"));
        finish(&mut state, "dgd-aaaaaaaa", "a");

        state.apply(vanished("dgd-aaaaaaaa"));
        assert_eq!(live(&state), ["a"], "the only generation is kept");
        state.apply(appeared("dgd-aaaaaaaa"));
        assert!(state.pending.is_empty(), "a returning generation is reused");

        state.apply(vanished("dgd-aaaaaaaa"));
        state.apply(appeared("dgd-bbbbbbbb"));
        finish(&mut state, "dgd-bbbbbbbb", "b");
        assert_eq!(live(&state), ["b"]);
    }

    #[tokio::test]
    async fn vanished_generation_is_retired_when_another_serves() {
        let (build, publish, cancel): (TestBuild, _, _) =
            (manual_build, channel(), CancellationToken::new());
        let mut state = GenerationState::new(&build, &publish, &cancel);
        for (namespace, value) in [("dgd-aaaaaaaa", "a"), ("dgd-bbbbbbbb", "b")] {
            state.apply(appeared(namespace));
            finish(&mut state, namespace, value);
        }
        state.apply(vanished("dgd-aaaaaaaa"));
        assert_eq!(live(&state), ["b"]);
    }

    #[tokio::test]
    async fn panicked_build_is_cancelled_and_retried() {
        let (build, publish, cancel): (TestBuild, _, _) =
            (manual_build, channel(), CancellationToken::new());
        let mut state = GenerationState::new(&build, &publish, &cancel);
        state.apply(appeared("dgd-aaaaaaaa"));
        let (seq, token) = {
            let pending = &state.pending["dgd-aaaaaaaa"];
            (pending.seq, pending.cancel.clone())
        };
        state.on_built(("dgd-aaaaaaaa".into(), seq, Err(Box::new("boom"))));
        assert!(token.is_cancelled(), "the panicked build's work is stopped");
        assert_ne!(
            state.pending["dgd-aaaaaaaa"].seq, seq,
            "a new build is pending"
        );
    }

    #[tokio::test]
    async fn dropping_the_state_cancels_pending_builds() {
        let (build, publish, cancel): (TestBuild, _, _) =
            (manual_build, channel(), CancellationToken::new());
        let mut state = GenerationState::new(&build, &publish, &cancel);
        state.apply(appeared("dgd-aaaaaaaa"));
        let token = state.pending["dgd-aaaaaaaa"].cancel.clone();
        drop(state);
        assert!(token.is_cancelled());
    }

    type Events = mpsc::UnboundedSender<anyhow::Result<DiscoveryEvent>>;

    /// What each `subscribe` call does, in order. Calls past the end never
    /// open.
    enum Subscription {
        Open(mpsc::UnboundedReceiver<anyhow::Result<DiscoveryEvent>>),
        Panic,
    }

    fn open() -> (Events, Subscription) {
        let (tx, rx) = mpsc::unbounded_channel();
        (tx, Subscription::Open(rx))
    }

    /// Runs the watcher over scripted subscriptions. Every build hands the
    /// test a gate that finishes it, and every `subscribe` call is reported.
    struct Harness {
        gates: mpsc::UnboundedReceiver<(String, oneshot::Sender<()>)>,
        subscribed: mpsc::UnboundedReceiver<()>,
        published: watch::Receiver<GenerationSet<String>>,
        cancel: CancellationToken,
        task: tokio::task::JoinHandle<()>,
    }

    impl Harness {
        fn start(subscriptions: Vec<Subscription>) -> Self {
            let subscriptions = Arc::new(Mutex::new(VecDeque::from(subscriptions)));
            let (subscribed_tx, subscribed) = mpsc::unbounded_channel();
            let subscribe = move |_cancel: CancellationToken| {
                subscribed_tx.send(()).unwrap();
                let next = subscriptions.lock().unwrap().pop_front();
                let stream = match next {
                    Some(Subscription::Panic) => panic!("scripted subscribe panic"),
                    Some(Subscription::Open(rx)) => Some(Box::pin(
                        tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
                    ) as DiscoveryStream),
                    None => None,
                };
                async move {
                    match stream {
                        Some(stream) => Ok(stream),
                        None => futures::future::pending().await,
                    }
                }
            };
            let (gate_tx, gates) = mpsc::unbounded_channel();
            let build =
                move |namespace: String, _card: ModelDeploymentCard, _: CancellationToken| {
                    let gate_tx = gate_tx.clone();
                    async move {
                        let (open, opened) = oneshot::channel();
                        gate_tx.send((namespace.clone(), open)).ok()?;
                        opened.await.ok()?;
                        Some(namespace)
                    }
                };
            let (publish, published) = watch::channel::<GenerationSet<String>>(Arc::from([]));
            let cancel = CancellationToken::new();
            let task = tokio::spawn(run_generation_watcher(
                subscribe,
                dgd(),
                build,
                publish,
                cancel.clone(),
            ));
            Self {
                gates,
                subscribed,
                published,
                cancel,
                task,
            }
        }

        /// Finish the next build, checking which namespace it builds.
        async fn finish_build(&mut self, namespace: &str) {
            let (built, gate) = tokio::time::timeout(Duration::from_secs(5), self.gates.recv())
                .await
                .expect("a build must start")
                .unwrap();
            assert_eq!(built, namespace);
            gate.send(()).unwrap();
        }

        async fn next_subscribe(&mut self) {
            tokio::time::timeout(Duration::from_secs(5), self.subscribed.recv())
                .await
                .expect("the watcher must subscribe")
                .unwrap();
        }

        async fn next_published(&mut self) -> Vec<String> {
            tokio::time::timeout(Duration::from_secs(5), self.published.changed())
                .await
                .expect("the watcher must publish")
                .unwrap();
            self.published
                .borrow_and_update()
                .iter()
                .map(|g| (**g).clone())
                .collect()
        }

        async fn stop(self) {
            self.cancel.cancel();
            tokio::time::timeout(Duration::from_secs(5), self.task)
                .await
                .expect("the watcher must stop on cancellation")
                .unwrap();
        }
    }

    /// A rollout: `b` joins `a`, then `a` drains and is retired at once.
    #[tokio::test]
    async fn watcher_follows_a_rolling_update() {
        let (a, b) = (
            decode("dgd-aaaaaaaa", 1, MODEL),
            decode("dgd-bbbbbbbb", 2, MODEL),
        );
        let (events, subscription) = open();
        let mut h = Harness::start(vec![subscription]);

        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        h.finish_build("dgd-aaaaaaaa").await;
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);
        events.send(Ok(DiscoveryEvent::Added(b))).unwrap();
        h.finish_build("dgd-bbbbbbbb").await;
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa", "dgd-bbbbbbbb"]);
        events.send(Ok(DiscoveryEvent::Removed(a.id()))).unwrap();
        assert_eq!(h.next_published().await, ["dgd-bbbbbbbb"]);
        h.stop().await;
    }

    /// A build that finishes while the watch is being reopened still publishes.
    #[tokio::test]
    async fn watcher_publishes_builds_while_resubscribing() {
        let (events, subscription) = open();
        let mut h = Harness::start(vec![subscription]);
        h.next_subscribe().await;

        events
            .send(Ok(DiscoveryEvent::Added(decode("dgd-aaaaaaaa", 1, MODEL))))
            .unwrap();
        let (_, gate) = h.gates.recv().await.unwrap();
        drop(events); // The watch ends; the next subscription never opens.
        h.next_subscribe().await;
        gate.send(()).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);
        h.stop().await;
    }

    #[tokio::test]
    async fn watcher_restarts_after_a_panic() {
        let (events, subscription) = open();
        let mut h = Harness::start(vec![Subscription::Panic, subscription]);

        events
            .send(Ok(DiscoveryEvent::Added(decode("dgd-aaaaaaaa", 1, MODEL))))
            .unwrap();
        h.finish_build("dgd-aaaaaaaa").await;
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);
        h.stop().await;
    }
}
