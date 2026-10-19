// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Follows the worker generations a DGD rolling update creates.
//!
//! The operator starts each worker generation under its own Dynamo namespace,
//! the DGD namespace plus a generation suffix (`<namespace>-<8 hex>`), and runs
//! the old and new generations side by side until the old one drains. Each
//! generation registers its own event-plane channels, so a router bound to one
//! generation never sees the workers of the next.
//!
//! This module keeps one routing generation per live worker namespace, the
//! way the Frontend keeps one `WorkerSet` per namespace
//! (`lib/llm/src/discovery/watcher.rs`): it watches model-card discovery,
//! builds a generation when a matching namespace first registers a
//! decode-capable model card, and retires it when that namespace's last such
//! card is removed. Requests then choose among the live generations, weighted
//! by worker count (see [`pick_weighted`]).
//!
//! The generation type is generic so the watcher's bookkeeping can be tested
//! without a KV router; `epp.rs` supplies the real one.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryStream, ModelCardInstanceId,
};
use dynamo_runtime::namespace::{NamespaceFilter, NamespacePrefixMode};
use futures::StreamExt;
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

const INITIAL_RESUBSCRIBE_BACKOFF: Duration = Duration::from_millis(100);
const MAX_RESUBSCRIBE_BACKOFF: Duration = Duration::from_secs(5);

/// The live generations, ordered by namespace. Published as one immutable
/// snapshot so a request clones a single `Arc` instead of holding a lock while
/// it routes.
pub(crate) type GenerationSet<G> = Arc<[Arc<G>]>;

/// A namespace gained its first, or lost its last, decode-capable model card.
#[derive(Debug)]
pub(crate) enum GenerationChange {
    Appeared {
        namespace: String,
        card: Box<ModelDeploymentCard>,
    },
    Vanished {
        namespace: String,
    },
}

/// Decode-capable model cards per matching worker namespace, maintained from
/// the discovery watch.
///
/// Prefill-only cards are not tracked: a generation's prefill router finds its
/// own prefill workers, and a namespace that serves only prefill has nothing
/// to route a request to.
pub(crate) struct GenerationTracker {
    namespace_filter: NamespaceFilter,
    prefix_mode: NamespacePrefixMode,
    cards: HashMap<String, HashSet<ModelCardInstanceId>>,
}

impl GenerationTracker {
    pub(crate) fn new(namespace_filter: NamespaceFilter, prefix_mode: NamespacePrefixMode) -> Self {
        Self {
            namespace_filter,
            prefix_mode,
            cards: HashMap::new(),
        }
    }

    /// The model card `instance` registers, when it is a decode-capable card
    /// in a namespace this EPP serves.
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
        let DiscoveryInstanceId::Model(id) = instance.id() else {
            return None;
        };
        Some((id, card))
    }

    pub(crate) fn on_added(&mut self, instance: &DiscoveryInstance) -> Option<GenerationChange> {
        let (id, card) = self.admit(instance)?;
        let namespace = id.namespace.clone();
        let cards = self.cards.entry(namespace.clone()).or_default();
        let first = cards.is_empty();
        cards.insert(id);
        first.then(|| GenerationChange::Appeared {
            namespace,
            card: Box::new(card),
        })
    }

    pub(crate) fn on_removed(&mut self, id: &DiscoveryInstanceId) -> Option<GenerationChange> {
        let DiscoveryInstanceId::Model(id) = id else {
            return None;
        };
        let cards = self.cards.get_mut(&id.namespace)?;
        if !cards.remove(id) || !cards.is_empty() {
            return None;
        }
        self.cards.remove(&id.namespace);
        Some(GenerationChange::Vanished {
            namespace: id.namespace.clone(),
        })
    }

    /// Replace the tracked cards with a full discovery snapshot, reporting
    /// the namespaces that appeared or vanished relative to the old state.
    pub(crate) fn on_resync(&mut self, instances: &[DiscoveryInstance]) -> Vec<GenerationChange> {
        let mut cards: HashMap<String, HashSet<ModelCardInstanceId>> = HashMap::new();
        let mut first_cards: HashMap<String, ModelDeploymentCard> = HashMap::new();
        for instance in instances {
            if let Some((id, card)) = self.admit(instance) {
                first_cards.entry(id.namespace.clone()).or_insert(card);
                cards.entry(id.namespace.clone()).or_default().insert(id);
            }
        }
        let mut changes: Vec<GenerationChange> = self
            .cards
            .keys()
            .filter(|namespace| !cards.contains_key(*namespace))
            .map(|namespace| GenerationChange::Vanished {
                namespace: namespace.clone(),
            })
            .collect();
        changes.extend(
            first_cards
                .into_iter()
                .filter(|(namespace, _)| !self.cards.contains_key(namespace))
                .map(|(namespace, card)| GenerationChange::Appeared {
                    namespace,
                    card: Box::new(card),
                }),
        );
        self.cards = cards;
        changes
    }
}

/// Owns the published generations and the builds still in flight. Only the
/// watcher task holds it, so every change to the published set is serialized.
struct GenerationState<G> {
    published: BTreeMap<String, Arc<G>>,
    /// Builds in flight, by namespace, with the sequence number that fences a
    /// stale result after the namespace vanished and reappeared.
    pending: HashMap<String, (u64, CancellationToken)>,
    next_seq: u64,
    publish: watch::Sender<GenerationSet<G>>,
}

impl<G> GenerationState<G> {
    fn publish(&self) {
        let set: GenerationSet<G> = self.published.values().cloned().collect();
        self.publish.send_replace(set);
    }

    fn retire(&mut self, namespace: &str) {
        if let Some((_, cancel)) = self.pending.remove(namespace) {
            cancel.cancel();
        }
        // Dropping the published `Arc` does not cancel requests still routed
        // to the generation: they hold their own reference until their
        // bookkeeping completes, so the old generation drains normally.
        if self.published.remove(namespace).is_some() {
            tracing::info!(namespace, "Worker generation retired");
            self.publish();
        }
    }
}

/// Watch discovery for worker generations and publish the live set to
/// `publish` until `cancel` fires.
///
/// `subscribe` opens a discovery watch on all model cards; it is called again
/// when a watch fails or ends, and the new watch's initial snapshot reconciles
/// whatever changed in between. `build` constructs a generation for a newly
/// seen namespace from its first model card. It runs in its own task, so a
/// slow build never delays retiring another generation, and it should return
/// `None` once its token is cancelled.
pub(crate) async fn run_generation_watcher<G, S, SF, B, BF>(
    mut subscribe: S,
    mut tracker: GenerationTracker,
    build: B,
    publish: watch::Sender<GenerationSet<G>>,
    cancel: CancellationToken,
) where
    G: Send + Sync + 'static,
    S: FnMut(CancellationToken) -> SF,
    SF: Future<Output = anyhow::Result<DiscoveryStream>>,
    B: Fn(String, ModelDeploymentCard, CancellationToken) -> BF,
    BF: Future<Output = Option<G>> + Send + 'static,
{
    let mut state = GenerationState {
        published: BTreeMap::new(),
        pending: HashMap::new(),
        next_seq: 0,
        publish,
    };
    let mut builds: JoinSet<(String, u64, Option<G>)> = JoinSet::new();
    let mut backoff = INITIAL_RESUBSCRIBE_BACKOFF;

    'subscribe: loop {
        let watch_cancel = cancel.child_token();
        let opened = tokio::select! {
            _ = cancel.cancelled() => break,
            opened = subscribe(watch_cancel.clone()) => opened,
        };
        let mut events = match opened {
            Ok(events) => {
                backoff = INITIAL_RESUBSCRIBE_BACKOFF;
                events
            }
            Err(error) => {
                tracing::warn!(%error, "Failed to watch model discovery for worker generations");
                if !sleep_or_cancel(backoff, &cancel).await {
                    break;
                }
                backoff = (backoff * 2).min(MAX_RESUBSCRIBE_BACKOFF);
                continue;
            }
        };

        loop {
            tokio::select! {
                _ = cancel.cancelled() => break 'subscribe,
                Some(joined) = builds.join_next(), if !builds.is_empty() => {
                    let Ok((namespace, seq, generation)) = joined else {
                        continue;
                    };
                    // Accept the result only from the build still on record
                    // for this namespace; a retired or superseded build's
                    // result is dropped here.
                    if state.pending.get(&namespace).map(|(s, _)| *s) != Some(seq) {
                        continue;
                    }
                    state.pending.remove(&namespace);
                    if let Some(generation) = generation {
                        tracing::info!(namespace, "Worker generation ready for routing");
                        state.published.insert(namespace, Arc::new(generation));
                        state.publish();
                    }
                }
                event = events.next() => {
                    let changes = match event {
                        None => {
                            tracing::warn!("Model discovery watch ended; resubscribing");
                            break;
                        }
                        Some(Err(error)) => {
                            tracing::warn!(%error, "Model discovery watch error");
                            continue;
                        }
                        Some(Ok(DiscoveryEvent::Added(instance))) => {
                            tracker.on_added(&instance).into_iter().collect()
                        }
                        Some(Ok(DiscoveryEvent::Removed(id))) => {
                            tracker.on_removed(&id).into_iter().collect()
                        }
                        Some(Ok(DiscoveryEvent::Resync(instances))) => tracker.on_resync(&instances),
                        Some(Ok(DiscoveryEvent::ModelTaintsUpdated(_))) => Vec::new(),
                    };
                    for change in changes {
                        match change {
                            GenerationChange::Appeared { namespace, card } => {
                                tracing::info!(namespace, "Worker generation discovered");
                                let seq = state.next_seq;
                                state.next_seq += 1;
                                let build_cancel = cancel.child_token();
                                state
                                    .pending
                                    .insert(namespace.clone(), (seq, build_cancel.clone()));
                                let built = build(namespace.clone(), *card, build_cancel);
                                builds.spawn(async move { (namespace, seq, built.await) });
                            }
                            GenerationChange::Vanished { namespace } => state.retire(&namespace),
                        }
                    }
                }
            }
        }
        watch_cancel.cancel();
        if !sleep_or_cancel(backoff, &cancel).await {
            break;
        }
        backoff = (backoff * 2).min(MAX_RESUBSCRIBE_BACKOFF);
    }

    for (_, cancel) in state.pending.values() {
        cancel.cancel();
    }
    builds.shutdown().await;
}

/// Sleep for `delay`, returning `false` if `cancel` fired first.
async fn sleep_or_cancel(delay: Duration, cancel: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancel.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

/// Pick one item with probability proportional to its weight, the way the
/// Frontend spreads requests across its `WorkerSet`s, so traffic follows
/// capacity as a rolling update moves workers from one generation to the
/// next.
///
/// `roll` receives the total weight and returns a value in `0..total`. When
/// every weight is zero the first item is returned, so the caller's router
/// reports why nothing is routable instead of this function hiding it.
pub(crate) fn pick_weighted<T>(
    items: &[T],
    weight: impl Fn(&T) -> usize,
    roll: impl FnOnce(usize) -> usize,
) -> Option<&T> {
    if items.len() <= 1 {
        return items.first();
    }
    // Generations number one or two outside a rollout, so collecting the
    // weights costs less than computing each twice.
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
    use super::*;
    use dynamo_runtime::discovery::DiscoveryQuery;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio::sync::{mpsc, oneshot};

    fn sample_card() -> ModelDeploymentCard {
        ModelDeploymentCard::load_from_disk(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../lib/llm/tests/data/sample-models/TinyLlama_v1.1"
            ),
            None,
        )
        .unwrap()
    }

    /// Register `card` under `namespace/component/generate` in process-local
    /// discovery.
    async fn register(drt: &DistributedRuntime, namespace: &str, component: &str) {
        let endpoint = drt
            .namespace(namespace)
            .unwrap()
            .component(component)
            .unwrap()
            .endpoint("generate");
        endpoint.register_endpoint_instance().await.unwrap();
        dynamo_llm::local_model::register_model_card(&endpoint, &sample_card())
            .await
            .unwrap();
    }

    async fn process_local_drt() -> (Runtime, DistributedRuntime) {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        (runtime, drt)
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

    fn dgd_tracker() -> GenerationTracker {
        GenerationTracker::new(
            NamespaceFilter::Prefix("dgd".into()),
            NamespacePrefixMode::WorkerGeneration,
        )
    }

    /// The tracker admits worker generations of this DGD and nothing else: a
    /// sibling DGD sharing the name prefix is excluded under worker-generation
    /// matching, and an exact scope excludes generations.
    #[tokio::test]
    async fn tracker_scopes_generations_to_this_deployment() {
        let (runtime, drt) = process_local_drt().await;
        register(&drt, "default-foo-bar", "backend").await;
        register(&drt, "default-foo-1a2b3c4d", "backend").await;
        let instances = drt
            .discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap();

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
        assert_eq!(
            resync(
                NamespaceFilter::Global,
                NamespacePrefixMode::WorkerGeneration
            ),
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

    /// The rolling update from #16085: generation `a` is live, `b` appears
    /// beside it, then `a` drains. `b` is reported as soon as it registers, and
    /// `a` only once its last card is gone.
    #[tokio::test]
    async fn tracker_follows_a_rolling_update() {
        let (runtime, drt) = process_local_drt().await;
        register(&drt, "dgd-aaaaaaaa", "decode").await;
        register(&drt, "dgd-aaaaaaaa", "backend").await;
        register(&drt, "dgd-bbbbbbbb", "backend").await;
        let instances = drt
            .discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap();
        let (gen_a, gen_b) = (
            cards_in(&instances, "dgd-aaaaaaaa"),
            cards_in(&instances, "dgd-bbbbbbbb"),
        );
        assert_eq!((gen_a.len(), gen_b.len()), (2, 1));

        let mut tracker = dgd_tracker();
        let added: Vec<_> = gen_a.iter().filter_map(|i| tracker.on_added(i)).collect();
        assert_eq!(
            summarize(&added),
            ["+dgd-aaaaaaaa"],
            "one change per namespace"
        );
        let added: Vec<_> = tracker.on_added(gen_b[0]).into_iter().collect();
        assert_eq!(summarize(&added), ["+dgd-bbbbbbbb"]);

        assert!(tracker.on_removed(&gen_a[0].id()).is_none());
        assert!(
            tracker.on_removed(&gen_a[0].id()).is_none(),
            "a repeated removal is not a second card leaving"
        );
        let removed: Vec<_> = tracker.on_removed(&gen_a[1].id()).into_iter().collect();
        assert_eq!(summarize(&removed), ["-dgd-aaaaaaaa"]);

        // A resync that still lists `b` and brings back `a` reports only `a`.
        assert_eq!(summarize(&tracker.on_resync(&instances)), ["+dgd-aaaaaaaa"]);
        // A resync without `a` retires it again and leaves `b` alone.
        let only_b: Vec<DiscoveryInstance> = gen_b.into_iter().cloned().collect();
        assert_eq!(summarize(&tracker.on_resync(&only_b)), ["-dgd-aaaaaaaa"]);
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

    async fn next_published(published: &mut watch::Receiver<GenerationSet<String>>) -> Vec<String> {
        tokio::time::timeout(Duration::from_secs(5), published.changed())
            .await
            .expect("the watcher must publish")
            .unwrap();
        published
            .borrow_and_update()
            .iter()
            .map(|g| (**g).clone())
            .collect()
    }

    /// Drives the watcher loop over a scripted discovery stream with a fake
    /// builder and checks what it publishes as generations come and go,
    /// including a build that finishes after its generation was retired.
    #[tokio::test]
    async fn watcher_publishes_and_retires_generations() {
        let (runtime, drt) = process_local_drt().await;
        register(&drt, "dgd-aaaaaaaa", "backend").await;
        register(&drt, "dgd-bbbbbbbb", "backend").await;
        let instances = drt
            .discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap();
        let a = cards_in(&instances, "dgd-aaaaaaaa")[0].clone();
        let b = cards_in(&instances, "dgd-bbbbbbbb")[0].clone();

        let (event_tx, event_rx) = mpsc::unbounded_channel::<anyhow::Result<DiscoveryEvent>>();
        let mut event_rx = Some(event_rx);
        let subscribe = move |_cancel: CancellationToken| {
            let stream = event_rx.take().map(|rx| {
                Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
                    as DiscoveryStream
            });
            async move {
                match stream {
                    Some(stream) => Ok(stream),
                    None => futures::future::pending().await,
                }
            }
        };
        // Each build of `dgd-bbbbbbbb` hands the test a gate and waits for it,
        // ignoring cancellation, so the test controls when each one finishes.
        let (gate_tx, mut gate_rx) = mpsc::unbounded_channel::<oneshot::Sender<()>>();
        let build =
            move |namespace: String, _card: ModelDeploymentCard, _cancel: CancellationToken| {
                let gate_tx = gate_tx.clone();
                async move {
                    if namespace == "dgd-bbbbbbbb" {
                        let (open, opened) = oneshot::channel();
                        gate_tx.send(open).ok()?;
                        opened.await.ok()?;
                    }
                    Some(namespace)
                }
            };
        let (publish, mut published) = watch::channel::<GenerationSet<String>>(Arc::from([]));
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_generation_watcher(
            subscribe,
            dgd_tracker(),
            build,
            publish,
            cancel.clone(),
        ));

        event_tx.send(Ok(DiscoveryEvent::Added(a.clone()))).unwrap();
        assert_eq!(next_published(&mut published).await, ["dgd-aaaaaaaa"]);

        // `b` appears, vanishes, and appears again while its first build is
        // still running. The first build's late result must not publish.
        event_tx.send(Ok(DiscoveryEvent::Added(b.clone()))).unwrap();
        let stale = gate_rx.recv().await.unwrap();
        event_tx.send(Ok(DiscoveryEvent::Removed(b.id()))).unwrap();
        event_tx.send(Ok(DiscoveryEvent::Added(b.clone()))).unwrap();
        let current = gate_rx.recv().await.unwrap();
        stale.send(()).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !published.has_changed().unwrap(),
            "a superseded build must not publish"
        );
        current.send(()).unwrap();
        assert_eq!(
            next_published(&mut published).await,
            ["dgd-aaaaaaaa", "dgd-bbbbbbbb"]
        );

        // `a` drains: only `b` remains.
        event_tx.send(Ok(DiscoveryEvent::Removed(a.id()))).unwrap();
        assert_eq!(next_published(&mut published).await, ["dgd-bbbbbbbb"]);

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the watcher must stop on cancellation")
            .unwrap();
        runtime.shutdown();
    }
}
