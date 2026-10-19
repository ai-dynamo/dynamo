// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tracks worker generations across DGD rolling updates.
//!
//! Each generation runs under its own Dynamo namespace (`<namespace>-<8 hex>`).
//! Like the Frontend's per-namespace `WorkerSet`s, this builds one generation
//! when a namespace registers its first decode-capable model card and retires
//! it when the last one is removed. Generic over the generation type so it can
//! be tested without a KV router.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryStream, ModelCardInstanceId,
};
use dynamo_runtime::namespace::{NamespaceFilter, NamespacePrefixMode};
use futures::{FutureExt, StreamExt};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

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

/// Decode-capable base-model cards per matching namespace, and the one
/// model this EPP serves.
///
/// The model is pinned to the first one seen and moves only when none of its
/// cards remain (a rollout that renames it), so a broad namespace scope never
/// mixes models. Prefill-only and LoRA cards are skipped.
pub(crate) struct GenerationTracker {
    namespace_filter: NamespaceFilter,
    prefix_mode: NamespacePrefixMode,
    namespaces: HashMap<String, TrackedNamespace>,
    /// Card count per model across all namespaces.
    model_counts: HashMap<String, usize>,
    model: Option<String>,
    /// Namespaces last reported as `Appeared`.
    serving: HashSet<String>,
}

impl GenerationTracker {
    pub(crate) fn new(namespace_filter: NamespaceFilter, prefix_mode: NamespacePrefixMode) -> Self {
        Self {
            namespace_filter,
            prefix_mode,
            namespaces: HashMap::new(),
            model_counts: HashMap::new(),
            model: None,
            serving: HashSet::new(),
        }
    }

    /// `instance`'s card, if it is an in-scope, decode-capable base-model card.
    fn admit(
        &self,
        instance: &DiscoveryInstance,
    ) -> Option<(ModelCardInstanceId, ModelDeploymentCard)> {
        let DiscoveryInstance::Model { namespace, .. } = instance else {
            return None;
        };
        if !self
            .namespace_filter
            .matches_with_prefix_mode(namespace, self.prefix_mode)
        {
            return None;
        }
        let DiscoveryInstanceId::Model(id) = instance.id() else {
            return None;
        };
        if id.model_suffix.is_some() {
            return None;
        }
        let card = match instance.deserialize_model::<ModelDeploymentCard>() {
            Ok(card) => card,
            Err(error) => {
                tracing::debug!(%error, namespace, "Failed to deserialize model card, skipping");
                return None;
            }
        };
        if card.model_type.supports_prefill()
            && !card.model_type.supports_chat()
            && !card.model_type.supports_completions()
        {
            return None;
        }
        Some((id, card))
    }

    pub(crate) fn on_added(&mut self, instance: &DiscoveryInstance) -> Vec<GenerationChange> {
        let Some((id, card)) = self.admit(instance) else {
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
            if let Some((id, card)) = self.admit(instance) {
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
    pub(crate) fn card_for(&self, namespace: &str) -> Option<Arc<ModelDeploymentCard>> {
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

    /// Keep the pinned model while any of its cards remain; otherwise pin the
    /// smallest remaining model name. Returns whether the pin moved.
    fn repin(&mut self) -> bool {
        if self
            .model
            .as_ref()
            .is_some_and(|model| self.model_counts.contains_key(model))
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

/// Published generations and in-flight builds; owned by the watcher task only.
struct GenerationState<'a, G, B> {
    published: BTreeMap<String, Arc<G>>,
    /// Published generations whose namespace vanished while no other
    /// generation was ready. Kept so a restarting worker does not lose its
    /// generation; retired once a replacement is published.
    dormant: HashSet<String>,
    /// In-flight builds; `seq` fences a stale result after a vanish/reappear.
    pending: HashMap<String, (u64, CancellationToken)>,
    next_seq: u64,
    builds: JoinSet<BuildResult<G>>,
    build: &'a B,
    publish: &'a watch::Sender<GenerationSet<G>>,
    cancel: &'a CancellationToken,
}

impl<G, B, BF> GenerationState<'_, G, B>
where
    G: Send + Sync + 'static,
    B: Fn(String, ModelDeploymentCard, CancellationToken) -> BF,
    BF: Future<Output = Option<G>> + Send + 'static,
{
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
        let build_cancel = self.cancel.child_token();
        self.pending
            .insert(namespace.clone(), (seq, build_cancel.clone()));
        let built = (self.build)(namespace.clone(), (*card).clone(), build_cancel.clone());
        self.builds.spawn(async move {
            if !delay.is_zero() && !sleep_or_cancel(delay, &build_cancel).await {
                return (namespace, seq, Ok(None));
            }
            (namespace, seq, AssertUnwindSafe(built).catch_unwind().await)
        });
    }

    fn vanish(&mut self, namespace: &str) {
        if let Some((_, cancel)) = self.pending.remove(namespace) {
            cancel.cancel();
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

    fn on_built(&mut self, (namespace, seq, built): BuildResult<G>, tracker: &GenerationTracker) {
        // Drop results from retired or superseded builds.
        if self.pending.get(&namespace).map(|(s, _)| *s) != Some(seq) {
            return;
        }
        self.pending.remove(&namespace);
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
                if let Some(card) = tracker.card_for(&namespace) {
                    self.spawn_build(namespace, card, PANIC_RETRY_DELAY);
                }
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
    namespace_filter: NamespaceFilter,
    prefix_mode: NamespacePrefixMode,
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
        let tracker = GenerationTracker::new(namespace_filter.clone(), prefix_mode);
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
    let mut state = GenerationState {
        published: BTreeMap::new(),
        dormant: HashSet::new(),
        pending: HashMap::new(),
        next_seq: 0,
        builds: JoinSet::new(),
        build,
        publish,
        cancel,
    };
    let mut backoff = INITIAL_RESUBSCRIBE_BACKOFF;
    let mut watch_cancel = cancel.child_token();
    let mut opening: Option<OpenWatch> = Some(Box::pin(subscribe(watch_cancel.clone())));
    let mut events: Option<DiscoveryStream> = None;

    loop {
        // Builds are joined in every state, including while (re)subscribing.
        tokio::select! {
            _ = cancel.cancelled() => break,
            Some(joined) = state.builds.join_next(), if !state.builds.is_empty() => {
                if let Ok(result) = joined {
                    state.on_built(result, &tracker);
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
    for (_, cancel) in state.pending.values() {
        cancel.cancel();
    }
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

/// Pick an item with probability proportional to its weight; `roll(total)`
/// returns a value in `0..total`. All-zero weights pick the first item.
pub(crate) fn pick_weighted<T>(
    items: &[T],
    weight: impl Fn(&T) -> usize,
    roll: impl FnOnce(usize) -> usize,
) -> Option<&T> {
    if items.len() <= 1 {
        return items.first();
    }
    let weights: Vec<usize> = items.iter().map(&weight).collect();
    let total: usize = weights.iter().sum();
    if total == 0 {
        return items.first();
    }
    let mut point = roll(total);
    for (item, weight) in items.iter().zip(weights) {
        if point < weight {
            return Some(item);
        }
        point -= weight;
    }
    items.last()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use super::*;
    use dynamo_runtime::discovery::DiscoveryQuery;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::sync::{mpsc, oneshot};

    const MODEL: &str = "tiny";

    /// Register a card for `model` under `namespace/component/generate` in
    /// process-local discovery.
    async fn register(drt: &DistributedRuntime, namespace: &str, component: &str, model: &str) {
        let endpoint = drt
            .namespace(namespace)
            .unwrap()
            .component(component)
            .unwrap()
            .endpoint("generate");
        endpoint.register_endpoint_instance().await.unwrap();
        let mut card = ModelDeploymentCard::load_from_disk(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../lib/llm/tests/data/sample-models/TinyLlama_v1.1"
            ),
            None,
        )
        .unwrap();
        card.display_name = model.to_string();
        dynamo_llm::local_model::register_model_card(&endpoint, &card)
            .await
            .unwrap();
    }

    /// Register one card per `(namespace, component, model)` and return the
    /// listed instances.
    async fn discover(cards: &[(&str, &str, &str)]) -> (Runtime, Vec<DiscoveryInstance>) {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        for (namespace, component, model) in cards {
            register(&drt, namespace, component, model).await;
        }
        let instances = drt
            .discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap();
        (runtime, instances)
    }

    fn cards_in<'a>(instances: &'a [DiscoveryInstance], ns: &str) -> Vec<&'a DiscoveryInstance> {
        instances
            .iter()
            .filter(|i| matches!(i, DiscoveryInstance::Model { namespace, .. } if namespace == ns))
            .collect()
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

    fn tracker(filter: NamespaceFilter) -> GenerationTracker {
        GenerationTracker::new(filter, NamespacePrefixMode::WorkerGeneration)
    }

    fn dgd() -> NamespaceFilter {
        NamespaceFilter::Prefix("dgd".into())
    }

    /// Worker generations of this DGD are admitted; a sibling DGD sharing the
    /// prefix is excluded under worker-generation matching, and an exact
    /// scope excludes generations.
    #[tokio::test]
    async fn tracker_scopes_generations_to_this_deployment() {
        let (runtime, instances) = discover(&[
            ("default-foo-bar", "backend", MODEL),
            ("default-foo-1a2b3c4d", "backend", MODEL),
        ])
        .await;
        let resync = |filter: NamespaceFilter, mode: NamespacePrefixMode| {
            summarize(&GenerationTracker::new(filter, mode).on_resync(&instances))
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
        runtime.shutdown();
    }

    /// The rolling update from #16085: `a` is live, `b` appears beside it,
    /// then `a` drains. `b` is reported on its first card, `a` on its last.
    #[tokio::test]
    async fn tracker_follows_a_rolling_update() {
        let (runtime, instances) = discover(&[
            ("dgd-aaaaaaaa", "decode", MODEL),
            ("dgd-aaaaaaaa", "backend", MODEL),
            ("dgd-bbbbbbbb", "backend", MODEL),
        ])
        .await;
        let (gen_a, gen_b) = (
            cards_in(&instances, "dgd-aaaaaaaa"),
            cards_in(&instances, "dgd-bbbbbbbb"),
        );
        assert_eq!((gen_a.len(), gen_b.len()), (2, 1));

        let mut tracker = tracker(dgd());
        let added: Vec<_> = gen_a.iter().flat_map(|i| tracker.on_added(i)).collect();
        assert_eq!(
            summarize(&added),
            ["+dgd-aaaaaaaa"],
            "one change per namespace"
        );
        assert_eq!(summarize(&tracker.on_added(gen_b[0])), ["+dgd-bbbbbbbb"]);

        assert!(tracker.on_removed(&gen_a[0].id()).is_empty());
        assert!(
            tracker.on_removed(&gen_a[0].id()).is_empty(),
            "a repeated removal is not a second card leaving"
        );
        assert_eq!(
            summarize(&tracker.on_removed(&gen_a[1].id())),
            ["-dgd-aaaaaaaa"]
        );

        assert_eq!(summarize(&tracker.on_resync(&instances)), ["+dgd-aaaaaaaa"]);
        let only_b: Vec<DiscoveryInstance> = gen_b.into_iter().cloned().collect();
        assert_eq!(summarize(&tracker.on_resync(&only_b)), ["-dgd-aaaaaaaa"]);
        runtime.shutdown();
    }

    /// A broad scope that sees two models serves one, and moves to the other
    /// only once the first has no cards left.
    #[tokio::test]
    async fn tracker_pins_one_model() {
        let (runtime, instances) = discover(&[
            ("team-a", "backend", "model-a"),
            ("team-b", "backend", "model-b"),
        ])
        .await;
        let mut tracker = tracker(NamespaceFilter::Global);
        assert_eq!(summarize(&tracker.on_resync(&instances)), ["+team-a"]);
        let team_a = cards_in(&instances, "team-a")[0];
        assert_eq!(
            summarize(&tracker.on_removed(&team_a.id())),
            ["+team-b", "-team-a"]
        );
        assert!(
            tracker.on_added(team_a).is_empty(),
            "the old model returning does not displace the pinned one"
        );
        runtime.shutdown();
    }

    #[test]
    fn pick_weighted_follows_capacity() {
        let items = ["old", "new"];
        let weights = |w: [usize; 2]| move |item: &&str| if *item == "old" { w[0] } else { w[1] };
        assert_eq!(pick_weighted(&items, weights([3, 1]), |_| 0), Some(&"old"));
        assert_eq!(pick_weighted(&items, weights([3, 1]), |_| 2), Some(&"old"));
        assert_eq!(pick_weighted(&items, weights([3, 1]), |_| 3), Some(&"new"));
        assert_eq!(
            pick_weighted(&items, weights([0, 4]), |_| 0),
            Some(&"new"),
            "a drained generation gets no traffic"
        );
        assert_eq!(
            pick_weighted(&items, weights([0, 0]), |_| unreachable!()),
            Some(&"old")
        );
        assert_eq!(
            pick_weighted(&["only"], |_| unreachable!(), |_| unreachable!()),
            Some(&"only"),
            "a single generation is chosen without computing weights"
        );
        assert_eq!(pick_weighted::<&str>(&[], |_| 0, |_| 0), None);
    }

    type Events = mpsc::UnboundedSender<anyhow::Result<DiscoveryEvent>>;

    /// What each `subscribe` call does, in order. Calls past the end never
    /// open.
    enum Subscription {
        Open(mpsc::UnboundedReceiver<anyhow::Result<DiscoveryEvent>>),
        Panic,
    }

    /// Runs the watcher over scripted subscriptions. Every build hands the
    /// test a gate: `true` finishes it, `false` makes it panic.
    struct Harness {
        gates: mpsc::UnboundedReceiver<(String, oneshot::Sender<bool>)>,
        published: watch::Receiver<GenerationSet<String>>,
        cancel: CancellationToken,
        task: tokio::task::JoinHandle<()>,
    }

    impl Harness {
        fn start(subscriptions: Vec<Subscription>) -> Self {
            let subscriptions = Arc::new(Mutex::new(VecDeque::from(subscriptions)));
            let subscribe = move |_cancel: CancellationToken| {
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
                        assert!(opened.await.ok()?, "scripted build panic");
                        Some(namespace)
                    }
                };
            let (publish, published) = watch::channel::<GenerationSet<String>>(Arc::from([]));
            let cancel = CancellationToken::new();
            let task = tokio::spawn(run_generation_watcher(
                subscribe,
                dgd(),
                NamespacePrefixMode::WorkerGeneration,
                build,
                publish,
                cancel.clone(),
            ));
            Self {
                gates,
                published,
                cancel,
                task,
            }
        }

        /// Take the next build's gate, checking which namespace it builds.
        async fn gate(&mut self, namespace: &str) -> oneshot::Sender<bool> {
            let (built, gate) = tokio::time::timeout(Duration::from_secs(5), self.gates.recv())
                .await
                .expect("a build must start")
                .unwrap();
            assert_eq!(built, namespace);
            gate
        }

        async fn next_published(&mut self) -> Vec<String> {
            tokio::time::timeout(Duration::from_secs(5), self.published.changed())
                .await
                .expect("the watcher must publish")
                .unwrap();
            self.current()
        }

        fn current(&mut self) -> Vec<String> {
            self.published
                .borrow_and_update()
                .iter()
                .map(|g| (**g).clone())
                .collect()
        }

        /// Let the watcher drain its queue, then confirm nothing was published.
        async fn assert_unchanged(&mut self, why: &str) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(!self.published.has_changed().unwrap(), "{why}");
            assert!(self.gates.try_recv().is_err(), "{why}: no build may start");
        }

        async fn stop(self) {
            self.cancel.cancel();
            tokio::time::timeout(Duration::from_secs(5), self.task)
                .await
                .expect("the watcher must stop on cancellation")
                .unwrap();
        }
    }

    fn open() -> (Events, Subscription) {
        let (tx, rx) = mpsc::unbounded_channel();
        (tx, Subscription::Open(rx))
    }

    async fn two_generations() -> (Runtime, DiscoveryInstance, DiscoveryInstance) {
        let (runtime, instances) = discover(&[
            ("dgd-aaaaaaaa", "backend", MODEL),
            ("dgd-bbbbbbbb", "backend", MODEL),
        ])
        .await;
        let a = cards_in(&instances, "dgd-aaaaaaaa")[0].clone();
        let b = cards_in(&instances, "dgd-bbbbbbbb")[0].clone();
        (runtime, a, b)
    }

    /// A rollout: `b` joins `a`, then `a` drains and is retired at once. A
    /// build superseded by a vanish and reappear never publishes.
    #[tokio::test]
    async fn watcher_publishes_and_retires_generations() {
        let (runtime, a, b) = two_generations().await;
        let (events, subscription) = open();
        let mut h = Harness::start(vec![subscription]);

        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        h.gate("dgd-aaaaaaaa").await.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);

        events.send(Ok(DiscoveryEvent::Added(b.clone()))).unwrap();
        let stale = h.gate("dgd-bbbbbbbb").await;
        events.send(Ok(DiscoveryEvent::Removed(b.id()))).unwrap();
        events.send(Ok(DiscoveryEvent::Added(b.clone()))).unwrap();
        let current = h.gate("dgd-bbbbbbbb").await;
        stale.send(true).unwrap();
        h.assert_unchanged("a superseded build must not publish")
            .await;
        current.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa", "dgd-bbbbbbbb"]);

        events.send(Ok(DiscoveryEvent::Removed(a.id()))).unwrap();
        assert_eq!(h.next_published().await, ["dgd-bbbbbbbb"]);
        h.stop().await;
        runtime.shutdown();
    }

    /// The only generation losing its workers (a restart) is kept and reused
    /// when they return, not rebuilt; a new generation replaces it.
    #[tokio::test]
    async fn watcher_keeps_the_only_generation_until_replaced() {
        let (runtime, a, b) = two_generations().await;
        let (events, subscription) = open();
        let mut h = Harness::start(vec![subscription]);

        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        h.gate("dgd-aaaaaaaa").await.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);

        events.send(Ok(DiscoveryEvent::Removed(a.id()))).unwrap();
        h.assert_unchanged("the only generation is kept").await;
        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        h.assert_unchanged("a returning generation is reused").await;

        events.send(Ok(DiscoveryEvent::Removed(a.id()))).unwrap();
        events.send(Ok(DiscoveryEvent::Added(b.clone()))).unwrap();
        h.gate("dgd-bbbbbbbb").await.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-bbbbbbbb"]);
        h.stop().await;
        runtime.shutdown();
    }

    /// A build that panics is retried rather than leaving its namespace stuck.
    #[tokio::test]
    async fn watcher_retries_a_panicked_build() {
        let (runtime, a, _) = two_generations().await;
        let (events, subscription) = open();
        let mut h = Harness::start(vec![subscription]);

        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        h.gate("dgd-aaaaaaaa").await.send(false).unwrap();
        h.gate("dgd-aaaaaaaa").await.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);
        h.stop().await;
        runtime.shutdown();
    }

    /// A build that finishes while the watch is down still publishes.
    #[tokio::test]
    async fn watcher_publishes_builds_while_resubscribing() {
        let (runtime, a, _) = two_generations().await;
        let (events, subscription) = open();
        let mut h = Harness::start(vec![subscription]);

        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        let gate = h.gate("dgd-aaaaaaaa").await;
        drop(events); // The watch ends; the next subscribe never opens.
        tokio::time::sleep(Duration::from_millis(20)).await;
        gate.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);
        h.stop().await;
        runtime.shutdown();
    }

    /// A watcher that panics restarts with a fresh watch.
    #[tokio::test]
    async fn watcher_restarts_after_a_panic() {
        let (runtime, a, _) = two_generations().await;
        let (events, subscription) = open();
        let mut h = Harness::start(vec![Subscription::Panic, subscription]);

        events.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        h.gate("dgd-aaaaaaaa").await.send(true).unwrap();
        assert_eq!(h.next_published().await, ["dgd-aaaaaaaa"]);
        h.stop().await;
        runtime.shutdown();
    }
}
