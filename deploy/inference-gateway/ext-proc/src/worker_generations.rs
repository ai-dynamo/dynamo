// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Follows worker generations across DGD rolling updates with the Frontend's
//! discovery controller.
//!
//! Each generation runs under its own Dynamo namespace (`<namespace>-<8 hex>`).
//! The EPP runs the same [`ModelDiscoveryController`] as the Frontend, which
//! groups worker cards per namespace, endpoint, model, and role, admits each
//! group's workers by card checksum, and builds, retries, and reconciles. The
//! [`GenerationHost`] here turns committed decode and aggregated groups into
//! routing generations, judges each namespace's readiness from its committed
//! groups as the Frontend does, and publishes the generations requests choose
//! from.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use dynamo_llm::discovery::controller::{
    ControllerHost, DesiredInstance, GroupKey, GroupSpec, ModelDiscoveryController,
};
use dynamo_llm::discovery::readiness::{
    ReadinessUnit, evaluate_readiness, normalize_legacy_prefill_topology,
};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryStream, ModelCardInstanceId};
use dynamo_runtime::namespace::{NamespaceFilter, NamespacePrefixMode};
use dynamo_runtime::protocols::EndpointId;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

/// Live generations, ordered by group.
pub(crate) type GenerationSet<G> = Arc<[Arc<G>]>;

/// The namespaces whose model cards this EPP follows.
#[derive(Clone, Debug)]
pub(crate) struct GenerationScope {
    pub(crate) namespace_filter: NamespaceFilter,
    pub(crate) prefix_mode: NamespacePrefixMode,
    /// The component whose decode or aggregated workers the EPP routes to.
    pub(crate) component: String,
}

impl GenerationScope {
    /// Whether `namespace` is one of the DGD's worker generations: the
    /// operator-set prefix itself, or the prefix plus a generation suffix.
    /// Decided the same way with or without `DYN_NAMESPACE_PREFIX_STRICT`, so
    /// operators that do not set it are followed too.
    fn is_generation(&self, namespace: &str) -> bool {
        matches!(self.namespace_filter, NamespaceFilter::Prefix(_))
            && self
                .namespace_filter
                .matches_with_prefix_mode(namespace, NamespacePrefixMode::WorkerGeneration)
    }
}

/// A routing generation, built from one decode or aggregated group.
pub(crate) trait RoutingGeneration: Send + Sync + 'static {
    /// Whether the generation's topology is complete, and its namespace's
    /// prefill endpoint, if any. Called whenever the host settles.
    fn set_topology(&self, ready: bool, prefill: Option<&EndpointId>);
}

/// What one build of a decode or aggregated group gets.
pub(crate) struct BuildRequest {
    pub(crate) card: ModelDeploymentCard,
    pub(crate) endpoint: EndpointId,
    /// Whether no generation is published yet.
    pub(crate) startup: bool,
    pub(crate) cancel: CancellationToken,
}

/// A prepared group: a routing generation for decode and aggregated groups,
/// nothing for the prefill and encode groups it pairs with.
enum Prepared<G> {
    Generation(Arc<G>),
    Peer,
}

/// A committed group, as readiness and prefill pairing see it.
struct CommittedGroup {
    namespace: String,
    model: String,
    role: WorkerType,
    endpoint: EndpointId,
    unit: ReadinessUnit,
}

/// The routing generation of a committed decode or aggregated group.
struct Entry<G> {
    generation: Arc<G>,
    namespace: String,
    model: String,
    /// Whether its namespace's topology is complete.
    ready: bool,
    published: bool,
    /// Its workers are gone and no ready generation has replaced it yet.
    dormant: bool,
}

struct HostState<G> {
    /// The namespace of the first worker the EPP routes to.
    bound: Option<String>,
    groups: HashMap<GroupKey, CommittedGroup>,
    generations: BTreeMap<GroupKey, Entry<G>>,
}

/// The EPP's host for the Frontend's discovery controller.
struct GenerationHost<G, B, L> {
    scope: GenerationScope,
    build: B,
    list: L,
    publish: watch::Sender<GenerationSet<G>>,
    state: Mutex<HostState<G>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether a published generation of `model` other than `key` has workers
/// and a complete topology, so it can take over from `key`.
fn has_replacement<G>(
    generations: &BTreeMap<GroupKey, Entry<G>>,
    key: &GroupKey,
    model: &str,
) -> bool {
    generations.iter().any(|(other, entry)| {
        other != key && entry.model == model && entry.published && !entry.dormant && entry.ready
    })
}

impl<G: RoutingGeneration, B, L> GenerationHost<G, B, L> {
    fn new(
        scope: GenerationScope,
        build: B,
        list: L,
        publish: watch::Sender<GenerationSet<G>>,
    ) -> Self {
        Self {
            scope,
            build,
            list,
            publish,
            state: Mutex::new(HostState {
                bound: None,
                groups: HashMap::new(),
                generations: BTreeMap::new(),
            }),
        }
    }

    /// Whether the EPP follows `namespace`; `None` until it binds.
    fn follows(&self, namespace: &str) -> Option<bool> {
        let state = lock(&self.state);
        let bound = state.bound.as_deref()?;
        Some(if self.scope.is_generation(bound) {
            self.scope.is_generation(namespace)
        } else {
            bound == namespace
        })
    }

    /// Bind to the namespace of the first worker the EPP routes to. When that
    /// is one of the DGD's generations, follow every generation of the
    /// prefix; otherwise stay on it.
    fn bind(&self, namespace: &str) -> bool {
        lock(&self.state).bound.get_or_insert_with(|| {
            tracing::info!(
                namespace,
                follows_generations = self.scope.is_generation(namespace),
                "Bound to worker namespace"
            );
            namespace.to_string()
        });
        self.follows(namespace) == Some(true)
    }

    /// Record a committed group. Returns whether it replaced a published
    /// generation.
    fn commit(
        &self,
        state: &mut HostState<G>,
        spec: &GroupSpec,
        prepared: Prepared<G>,
        live_count: usize,
    ) -> bool {
        let card = &spec.representative.card;
        let namespace = &spec.representative.mcid.namespace;
        let model = &spec.key.model_name;
        state.groups.insert(
            spec.key.clone(),
            CommittedGroup {
                namespace: namespace.clone(),
                model: model.clone(),
                role: card.effective_worker_type(),
                endpoint: spec.representative.endpoint_id.clone(),
                unit: ReadinessUnit {
                    worker_type: card.worker_type,
                    live_count,
                    needs: card.needs.clone(),
                },
            },
        );
        let Prepared::Generation(generation) = prepared else {
            return false;
        };
        // A replacement keeps the place of the generation it replaces.
        let published = state
            .generations
            .get(&spec.key)
            .is_some_and(|entry| entry.published);
        state.generations.insert(
            spec.key.clone(),
            Entry {
                generation,
                namespace: namespace.clone(),
                model: model.clone(),
                ready: false,
                published,
                dormant: false,
            },
        );
        published
    }

    /// Bring readiness, prefill pairing, and the published set in line with
    /// the committed groups.
    fn settle(&self, state: &mut HostState<G>, mut changed: bool) {
        let mut units: HashMap<(&str, &str), Vec<ReadinessUnit>> = HashMap::new();
        for group in state.groups.values() {
            units
                .entry((&group.namespace, &group.model))
                .or_default()
                .push(group.unit.clone());
        }
        for entry in state.generations.values_mut() {
            let ready = units
                .get(&(entry.namespace.as_str(), entry.model.as_str()))
                .is_some_and(|units| evaluate_readiness(units).ready);
            if entry.ready != ready {
                tracing::info!(
                    namespace = entry.namespace,
                    ready,
                    "Worker generation topology changed"
                );
                entry.ready = ready;
            }
            let prefill = state.groups.values().find(|group| {
                group.role == WorkerType::Prefill
                    && group.namespace == entry.namespace
                    && group.model == entry.model
            });
            entry
                .generation
                .set_topology(ready, prefill.map(|group| &group.endpoint));
        }

        // A generation is published once its own topology is complete, or at
        // once while no complete generation serves, as at startup.
        let mut serving = state
            .generations
            .values()
            .any(|entry| entry.published && !entry.dormant && entry.ready);
        for entry in state.generations.values_mut() {
            if entry.published || (serving && !entry.ready) {
                continue;
            }
            tracing::info!(
                namespace = entry.namespace,
                topology_complete = entry.ready,
                "Worker generation ready for routing"
            );
            entry.published = true;
            serving |= entry.ready;
            changed = true;
        }

        // A ready generation of the same model replaces one whose workers
        // are gone.
        let retired: Vec<GroupKey> = state
            .generations
            .iter()
            .filter(|(key, entry)| {
                entry.dormant && has_replacement(&state.generations, key, &entry.model)
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in retired {
            if let Some(entry) = state.generations.remove(&key) {
                tracing::info!(namespace = entry.namespace, "Worker generation retired");
                changed = true;
            }
        }

        if changed {
            let set: GenerationSet<G> = state
                .generations
                .values()
                .filter(|entry| entry.published)
                .map(|entry| entry.generation.clone())
                .collect();
            self.publish.send_replace(set);
        }
    }
}

#[async_trait]
impl<G, B, BF, L, LF> ControllerHost for GenerationHost<G, B, L>
where
    G: RoutingGeneration,
    B: Fn(BuildRequest) -> BF + Send + Sync + 'static,
    BF: Future<Output = anyhow::Result<G>> + Send,
    L: Fn() -> LF + Send + Sync + 'static,
    LF: Future<Output = anyhow::Result<Vec<DiscoveryInstance>>> + Send,
{
    type Prepared = Prepared<G>;

    fn normalize(
        &self,
        instance: DiscoveryInstance,
        namespace_filter: &NamespaceFilter,
    ) -> anyhow::Result<Option<DesiredInstance>> {
        let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            instance_id,
            model_suffix,
            ..
        } = &instance
        else {
            return Ok(None);
        };
        // A LoRA adapter's card adds to a worker whose base card already
        // counts, and a generation is identified by its base model.
        if model_suffix.is_some()
            || !namespace_filter.matches_with_prefix_mode(namespace, self.scope.prefix_mode)
            || self.follows(namespace) == Some(false)
        {
            return Ok(None);
        }
        let mut card = instance.deserialize_model::<ModelDeploymentCard>()?;
        normalize_legacy_prefill_topology(&mut card);
        let role = card.effective_worker_type();
        // Requests are routed to the configured component's workers; prefill
        // and encode workers pair from any component.
        if matches!(role, WorkerType::Decode | WorkerType::Aggregated)
            && (*component != self.scope.component || !self.bind(namespace))
        {
            return Ok(None);
        }
        let mcid = ModelCardInstanceId {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: endpoint.clone(),
            instance_id: *instance_id,
            model_suffix: None,
        };
        let group_key = GroupKey {
            model_name: card.name().to_string(),
            worker_set_key: format!("{namespace}/{component}/{endpoint}/{}", role.as_str()),
        };
        Ok(Some(DesiredInstance {
            key: mcid.to_path(),
            endpoint_id: EndpointId {
                namespace: namespace.clone(),
                component: component.clone(),
                name: endpoint.clone(),
            },
            mcid,
            mdc_checksum: card.mdcsum().to_string(),
            card,
            group_key,
            projection_fingerprint: String::new(),
            video_contract: None,
        }))
    }

    async fn prepare(
        &self,
        spec: GroupSpec,
        _admitted_ids: watch::Receiver<Vec<u64>>,
        cancellation: CancellationToken,
    ) -> anyhow::Result<Prepared<G>> {
        let card = &spec.representative.card;
        if !matches!(
            card.effective_worker_type(),
            WorkerType::Decode | WorkerType::Aggregated
        ) {
            return Ok(Prepared::Peer);
        }
        let startup = !lock(&self.state)
            .generations
            .values()
            .any(|entry| entry.published);
        tracing::info!(
            namespace = spec.representative.mcid.namespace,
            model = spec.key.model_name,
            "Worker generation discovered"
        );
        let generation = (self.build)(BuildRequest {
            card: card.clone(),
            endpoint: spec.representative.endpoint_id.clone(),
            startup,
            cancel: cancellation,
        })
        .await?;
        Ok(Prepared::Generation(Arc::new(generation)))
    }

    fn commit_group(
        &self,
        spec: &GroupSpec,
        prepared: Prepared<G>,
        members: &[DesiredInstance],
        _adapters: &[DesiredInstance],
    ) -> anyhow::Result<()> {
        let mut state = lock(&self.state);
        let changed = self.commit(&mut state, spec, prepared, members.len());
        self.settle(&mut state, changed);
        if state
            .generations
            .get(&spec.key)
            .is_some_and(|entry| !entry.published)
        {
            tracing::info!(
                namespace = spec.representative.mcid.namespace,
                "Worker generation waiting for its topology to complete"
            );
        }
        Ok(())
    }

    fn replace_group(
        &self,
        key: &GroupKey,
        members: &[DesiredInstance],
        _adapters: &[DesiredInstance],
    ) -> anyhow::Result<()> {
        let mut state = lock(&self.state);
        if let Some(group) = state.groups.get_mut(key) {
            group.unit.live_count = members.len();
        }
        self.settle(&mut state, false);
        Ok(())
    }

    fn replace_prepared_group(
        &self,
        spec: &GroupSpec,
        prepared: Prepared<G>,
        members: &[DesiredInstance],
        adapters: &[DesiredInstance],
    ) -> anyhow::Result<()> {
        self.commit_group(spec, prepared, members, adapters)
    }

    fn remove_group(&self, key: &GroupKey) {
        let mut state = lock(&self.state);
        state.groups.remove(key);
        let keep = state.generations.get(key).is_some_and(|entry| {
            entry.published && !has_replacement(&state.generations, key, &entry.model)
        });
        let mut changed = false;
        if keep {
            if let Some(entry) = state.generations.get_mut(key) {
                // In-flight requests hold their own `Arc`; new ones fail until
                // the workers return or a replacement is ready.
                tracing::info!(
                    namespace = entry.namespace,
                    "Worker generation has no workers; keeping it until a replacement is ready"
                );
                entry.dormant = true;
            }
        } else if let Some(entry) = state.generations.remove(key)
            && entry.published
        {
            tracing::info!(namespace = entry.namespace, "Worker generation retired");
            changed = true;
        }
        self.settle(&mut state, changed);
    }

    fn discard_prepared(&self, _prepared: Prepared<G>) {}

    async fn list_instances(&self) -> anyhow::Result<Vec<DiscoveryInstance>> {
        (self.list)().await
    }
}

/// Follow worker generations with the Frontend's discovery controller until
/// `stream` ends, publishing the live generations to `publish`.
pub(crate) async fn run_generation_controller<G, B, BF, L, LF>(
    stream: DiscoveryStream,
    scope: GenerationScope,
    build: B,
    list: L,
    publish: watch::Sender<GenerationSet<G>>,
) where
    G: RoutingGeneration,
    B: Fn(BuildRequest) -> BF + Send + Sync + 'static,
    BF: Future<Output = anyhow::Result<G>> + Send,
    L: Fn() -> LF + Send + Sync + 'static,
    LF: Future<Output = anyhow::Result<Vec<DiscoveryInstance>>> + Send,
{
    let namespace_filter = scope.namespace_filter.clone();
    let host = Arc::new(GenerationHost::new(scope, build, list, publish));
    ModelDiscoveryController::new(host)
        .run(stream, namespace_filter)
        .await;
}

/// Choose a generation for a request.
///
/// Generations are isolated serving units, each judged on its own, as the
/// Frontend does. Candidates are the generations that `serve` the requested
/// model, or all of them when none does (an alias or adapter the cards do not
/// list). Among candidates, those that are `ready` (their own topology is
/// complete) are weighted by `workers`; when none is ready they serve as they
/// can, and with no workers anywhere the first candidate is chosen so its
/// router reports why. `roll(total)` returns a value in `0..total`.
pub(crate) fn choose_generation<T>(
    generations: &[T],
    serves: impl Fn(&T) -> bool,
    ready: impl Fn(&T) -> bool,
    workers: impl Fn(&T) -> usize,
    roll: impl Fn(usize) -> usize,
) -> Option<&T> {
    if generations.len() <= 1 {
        return generations.first();
    }
    let restricted = generations.iter().any(&serves);
    let candidate = |generation: &T| !restricted || serves(generation);
    let ready_workers = |generation: &T| {
        if candidate(generation) && ready(generation) {
            workers(generation)
        } else {
            0
        }
    };
    let candidate_workers = |generation: &T| {
        if candidate(generation) {
            workers(generation)
        } else {
            0
        }
    };
    pick_weighted(generations, ready_workers, &roll)
        .or_else(|| pick_weighted(generations, candidate_workers, &roll))
        .or_else(|| generations.iter().find(|generation| candidate(generation)))
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;
    use dynamo_llm::model_type::ModelType;
    use dynamo_runtime::discovery::DiscoveryEvent;
    use tokio::sync::{mpsc, oneshot};
    use tokio_stream::wrappers::UnboundedReceiverStream;

    const MODEL: &str = "tiny";
    /// Covers the controller's retry backoff without hanging a failing test.
    const WAIT: Duration = Duration::from_secs(300);

    fn card(
        worker_type: Option<WorkerType>,
        model_type: ModelType,
        needs: Vec<Vec<WorkerType>>,
    ) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(MODEL);
        card.worker_type = worker_type;
        card.model_type = model_type;
        card.needs = needs;
        card
    }

    fn chat() -> ModelType {
        ModelType::Chat | ModelType::Completions
    }

    /// A worker's card, registered by `component` in `namespace`.
    fn worker(
        namespace: &str,
        component: &str,
        instance_id: u64,
        card: &ModelDeploymentCard,
    ) -> DiscoveryInstance {
        DiscoveryInstance::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: "generate".to_string(),
            instance_id,
            card_json: serde_json::to_value(card).unwrap(),
            model_suffix: None,
        }
    }

    fn aggregated(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        serving(namespace, instance_id, MODEL)
    }

    /// An aggregated worker of `model`.
    fn serving(namespace: &str, instance_id: u64, model: &str) -> DiscoveryInstance {
        let mut card = card(Some(WorkerType::Aggregated), chat(), vec![]);
        card.display_name = model.to_string();
        worker(namespace, "backend", instance_id, &card)
    }

    fn decode(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        let card = card(
            Some(WorkerType::Decode),
            chat(),
            vec![vec![WorkerType::Prefill]],
        );
        worker(namespace, "backend", instance_id, &card)
    }

    /// A prefill worker as the mocker registers it: no legacy
    /// `ModelType::Prefill` bit.
    fn prefill(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        let card = card(
            Some(WorkerType::Prefill),
            ModelType::empty(),
            vec![vec![WorkerType::Decode]],
        );
        worker(namespace, "prefill", instance_id, &card)
    }

    /// A prefill worker from before `worker_type`: the legacy bit only.
    fn legacy_prefill(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        let card = card(None, ModelType::Prefill, vec![]);
        worker(namespace, "prefill", instance_id, &card)
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

    struct FakeGeneration {
        namespace: String,
        model: String,
        ready: AtomicBool,
        prefill: Mutex<Option<EndpointId>>,
    }

    impl FakeGeneration {
        fn ready(&self) -> bool {
            self.ready.load(Ordering::Relaxed)
        }

        fn prefill_namespace(&self) -> Option<String> {
            lock(&self.prefill)
                .as_ref()
                .map(|endpoint| endpoint.namespace.clone())
        }
    }

    impl RoutingGeneration for FakeGeneration {
        fn set_topology(&self, ready: bool, prefill: Option<&EndpointId>) {
            self.ready.store(ready, Ordering::Relaxed);
            if let Some(endpoint) = prefill {
                lock(&self.prefill).get_or_insert_with(|| endpoint.clone());
            }
        }
    }

    /// A build the test completes.
    struct Build {
        namespace: String,
        startup: bool,
        reply: oneshot::Sender<anyhow::Result<()>>,
    }

    impl Build {
        fn finish(self) {
            let _ = self.reply.send(Ok(()));
        }

        fn fail(self) {
            let _ = self.reply.send(Err(anyhow::anyhow!("router failed")));
        }
    }

    /// The Frontend's discovery controller with a [`GenerationHost`], over a
    /// discovery stream the test drives.
    struct Epp {
        events: mpsc::UnboundedSender<anyhow::Result<DiscoveryEvent>>,
        /// What the controller's periodic reconciliation lists.
        registered: Arc<Mutex<Vec<DiscoveryInstance>>>,
        builds: mpsc::UnboundedReceiver<Build>,
        published: watch::Receiver<GenerationSet<FakeGeneration>>,
        controller: tokio::task::JoinHandle<()>,
    }

    impl Epp {
        fn start(scope: GenerationScope) -> Self {
            let (events, stream) = mpsc::unbounded_channel();
            let registered = Arc::new(Mutex::new(Vec::<DiscoveryInstance>::new()));
            let (build_tx, builds) = mpsc::unbounded_channel();
            let build = move |request: BuildRequest| {
                let build_tx = build_tx.clone();
                async move {
                    let (reply, outcome) = oneshot::channel();
                    let namespace = request.endpoint.namespace;
                    let model = request.card.name().to_string();
                    let _ = build_tx.send(Build {
                        namespace: namespace.clone(),
                        startup: request.startup,
                        reply,
                    });
                    outcome.await??;
                    Ok::<_, anyhow::Error>(FakeGeneration {
                        namespace,
                        model,
                        ready: AtomicBool::new(false),
                        prefill: Mutex::new(None),
                    })
                }
            };
            let list = {
                let registered = registered.clone();
                move || {
                    let instances = lock(&registered).clone();
                    async move { Ok::<_, anyhow::Error>(instances) }
                }
            };
            let (publish, published) =
                watch::channel::<GenerationSet<FakeGeneration>>(Arc::from([]));
            let controller = tokio::spawn(run_generation_controller(
                Box::pin(UnboundedReceiverStream::new(stream)),
                scope,
                build,
                list,
                publish,
            ));
            Self {
                events,
                registered,
                builds,
                published,
                controller,
            }
        }

        fn add(&self, instance: &DiscoveryInstance) {
            lock(&self.registered).push(instance.clone());
            let added = DiscoveryEvent::Added(instance.clone());
            self.events.send(Ok(added)).unwrap();
        }

        fn remove(&self, instance: &DiscoveryInstance) {
            lock(&self.registered).retain(|registered| registered.id() != instance.id());
            let removed = DiscoveryEvent::Removed(instance.id());
            self.events.send(Ok(removed)).unwrap();
        }

        async fn next_build(&mut self) -> Build {
            tokio::time::timeout(WAIT, self.builds.recv())
                .await
                .expect("no build started")
                .unwrap()
        }

        /// Complete the next build, which must be `namespace`'s.
        async fn build(&mut self, namespace: &str) {
            let build = self.next_build().await;
            assert_eq!(build.namespace, namespace);
            build.finish();
        }

        /// Let the controller act on everything so far. The clock is paused,
        /// so it advances only once every task is idle.
        async fn settle(&self) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        fn published(&self) -> Vec<String> {
            let published = self.published.borrow();
            published
                .iter()
                .map(|generation| generation.namespace.clone())
                .collect()
        }

        /// The published generations as `model@namespace`.
        fn published_models(&self) -> Vec<String> {
            let published = self.published.borrow();
            published
                .iter()
                .map(|generation| format!("{}@{}", generation.model, generation.namespace))
                .collect()
        }

        fn generation(&self, namespace: &str) -> Arc<FakeGeneration> {
            let published = self.published.borrow();
            published
                .iter()
                .find(|generation| generation.namespace == namespace)
                .cloned()
                .expect("generation is published")
        }

        /// End the discovery stream, which stops the controller, and check
        /// that no build was left unseen.
        async fn stop(self) {
            let Self {
                events,
                mut builds,
                controller,
                ..
            } = self;
            drop(events);
            controller.await.unwrap();
            assert!(builds.try_recv().is_err(), "unexpected build");
        }
    }

    /// The rolling update from #16085: generation `a` serves, `b` starts
    /// beside it and serves once built, then `a` drains and is retired, with
    /// no EPP restart.
    #[tokio::test(start_paused = true)]
    async fn follows_a_rolling_update() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        let mut epp = Epp::start(dgd());
        let old = aggregated(a, 1);
        epp.add(&old);
        epp.build(a).await;
        epp.settle().await;
        assert_eq!(epp.published(), [a]);

        epp.add(&aggregated(b, 2));
        epp.build(b).await;
        epp.settle().await;
        assert_eq!(epp.published(), [a, b], "generations overlap");

        epp.remove(&old);
        epp.settle().await;
        assert_eq!(epp.published(), [b]);
        epp.stop().await;
    }

    /// During a rollout a disaggregated generation takes no traffic until its
    /// own prefill workers register, and each generation's prefill router gets
    /// its own namespace's prefill workers, legacy or current.
    #[tokio::test(start_paused = true)]
    async fn holds_back_a_generation_until_its_topology_is_complete() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        let mut epp = Epp::start(dgd());
        epp.add(&legacy_prefill(a, 1));
        epp.add(&decode(a, 2));
        epp.build(a).await;
        epp.settle().await;
        let old = epp.generation(a);
        assert!(old.ready());
        assert_eq!(old.prefill_namespace().as_deref(), Some(a));

        epp.add(&decode(b, 3));
        epp.build(b).await;
        epp.settle().await;
        assert_eq!(epp.published(), [a], "no prefill workers yet");

        epp.add(&prefill(b, 4));
        epp.settle().await;
        assert_eq!(epp.published(), [a, b]);
        let new = epp.generation(b);
        assert!(new.ready());
        assert_eq!(new.prefill_namespace().as_deref(), Some(b));
        assert_eq!(old.prefill_namespace().as_deref(), Some(a));
        epp.stop().await;
    }

    /// At startup the first generation serves at once: a disaggregated one
    /// serves aggregated until its prefill workers register.
    #[tokio::test(start_paused = true)]
    async fn publishes_the_first_generation_at_once() {
        let a = "dgd-aaaaaaaa";
        let mut epp = Epp::start(dgd());
        epp.add(&decode(a, 1));
        let build = epp.next_build().await;
        assert!(build.startup);
        build.finish();
        epp.settle().await;
        let first = epp.generation(a);
        assert!(!first.ready());
        assert_eq!(first.prefill_namespace(), None);

        epp.add(&prefill(a, 2));
        epp.settle().await;
        assert!(first.ready());
        assert_eq!(first.prefill_namespace().as_deref(), Some(a));

        epp.add(&decode("dgd-bbbbbbbb", 3));
        let build = epp.next_build().await;
        assert!(!build.startup);
        build.finish();
        epp.stop().await;
    }

    /// A generation whose workers all leave keeps serving until a ready
    /// replacement exists, so requests fail there rather than finding no
    /// generation; workers that return are served by a rebuilt generation.
    #[tokio::test(start_paused = true)]
    async fn keeps_the_last_generation_until_a_replacement_is_ready() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        let mut epp = Epp::start(dgd());
        let gone = aggregated(a, 1);
        epp.add(&gone);
        epp.build(a).await;
        epp.settle().await;
        let first = epp.generation(a);

        epp.remove(&gone);
        epp.settle().await;
        assert_eq!(epp.published(), [a]);

        let returned = aggregated(a, 2);
        epp.add(&returned);
        epp.build(a).await;
        epp.settle().await;
        assert_eq!(epp.published(), [a]);
        assert!(!Arc::ptr_eq(&first, &epp.generation(a)), "rebuilt in place");

        epp.remove(&returned);
        epp.settle().await;
        assert_eq!(epp.published(), [a]);
        epp.add(&aggregated(b, 3));
        epp.build(b).await;
        epp.settle().await;
        assert_eq!(epp.published(), [b]);
        epp.stop().await;
    }

    /// Once no complete generation serves, an incomplete one serves as it
    /// can, as at startup. The operator rolls each role on its own schedule,
    /// so the old decode or prefill workers may drain before the new prefill
    /// workers register.
    #[tokio::test(start_paused = true)]
    async fn serves_an_incomplete_generation_once_no_complete_one_serves() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        for drained in ["decode", "prefill"] {
            let mut epp = Epp::start(dgd());
            let old_prefill = prefill(a, 1);
            let old_decode = decode(a, 2);
            epp.add(&old_prefill);
            epp.add(&old_decode);
            epp.build(a).await;
            epp.add(&decode(b, 3));
            epp.build(b).await;
            epp.settle().await;
            assert_eq!(epp.published(), [a]);

            epp.remove(if drained == "decode" {
                &old_decode
            } else {
                &old_prefill
            });
            epp.settle().await;
            assert_eq!(epp.published(), [a, b], "old {drained} drained");
            assert!(!epp.generation(b).ready());

            epp.add(&prefill(b, 4));
            if drained == "prefill" {
                epp.remove(&old_decode);
            }
            epp.settle().await;
            assert_eq!(epp.published(), [b]);
            assert!(epp.generation(b).ready());
            epp.stop().await;
        }
    }

    /// Only a generation of the same model replaces one whose workers are
    /// gone, so one model's requests never reach another model's workers.
    #[tokio::test(start_paused = true)]
    async fn a_replacement_serves_the_same_model() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        let mut epp = Epp::start(dgd());
        let x = serving(a, 1, "model-x");
        epp.add(&x);
        epp.build(a).await;
        epp.add(&serving(a, 2, "model-y"));
        epp.build(a).await;
        epp.settle().await;
        let both = ["model-x@dgd-aaaaaaaa", "model-y@dgd-aaaaaaaa"];
        assert_eq!(epp.published_models(), both);

        epp.remove(&x);
        epp.settle().await;
        assert_eq!(epp.published_models(), both, "model-y is no replacement");

        epp.add(&serving(b, 3, "model-x"));
        epp.build(b).await;
        epp.settle().await;
        assert_eq!(
            epp.published_models(),
            ["model-x@dgd-bbbbbbbb", "model-y@dgd-aaaaaaaa"]
        );
        epp.stop().await;
    }

    /// The EPP follows only its own DGD's generations, and builds them only
    /// from the configured component's base-model cards.
    #[tokio::test(start_paused = true)]
    async fn follows_only_its_own_deployments_workers() {
        let generation = "default-foo-1a2b3c4d";
        let mut epp = Epp::start(scope(
            NamespaceFilter::Prefix("default-foo".into()),
            NamespacePrefixMode::WorkerGeneration,
        ));
        epp.add(&aggregated("default-foo-bar", 1));
        let mut adapter = aggregated(generation, 2);
        if let DiscoveryInstance::Model { model_suffix, .. } = &mut adapter {
            *model_suffix = Some("adapter".into());
        }
        epp.add(&adapter);
        let processor = card(Some(WorkerType::Aggregated), chat(), vec![]);
        epp.add(&worker(generation, "processor", 3, &processor));
        let encoder = card(Some(WorkerType::Encode), ModelType::empty(), vec![]);
        epp.add(&worker(generation, "encoder", 4, &encoder));
        epp.add(&aggregated(generation, 5));
        epp.build(generation).await;
        epp.settle().await;
        assert_eq!(epp.published(), [generation]);
        epp.stop().await;
    }

    /// Without `DYN_NAMESPACE_PREFIX_STRICT` (operators before 1.6) the EPP
    /// still follows its DGD's generations but not a sibling DGD sharing the
    /// prefix; bound to the sibling first, it stays there.
    #[tokio::test(start_paused = true)]
    async fn follows_generations_under_a_literal_prefix() {
        let literal = || {
            scope(
                NamespaceFilter::Prefix("dgd".into()),
                NamespacePrefixMode::Literal,
            )
        };
        let mut epp = Epp::start(literal());
        epp.add(&aggregated("dgd-aaaaaaaa", 1));
        epp.build("dgd-aaaaaaaa").await;
        epp.add(&aggregated("dgd-bar", 2));
        epp.add(&aggregated("dgd-bbbbbbbb", 3));
        epp.build("dgd-bbbbbbbb").await;
        epp.settle().await;
        assert_eq!(epp.published(), ["dgd-aaaaaaaa", "dgd-bbbbbbbb"]);
        epp.stop().await;

        let mut epp = Epp::start(literal());
        epp.add(&aggregated("dgd-bar", 1));
        epp.build("dgd-bar").await;
        epp.add(&aggregated("dgd-aaaaaaaa", 2));
        epp.settle().await;
        assert_eq!(epp.published(), ["dgd-bar"]);
        epp.stop().await;
    }

    /// A global or exact scope binds to one namespace for good, and only a
    /// worker the EPP routes to binds it.
    #[tokio::test(start_paused = true)]
    async fn binds_one_namespace_outside_one_deployment() {
        let mut epp = Epp::start(scope(
            NamespaceFilter::Global,
            NamespacePrefixMode::WorkerGeneration,
        ));
        epp.add(&prefill("team-b", 1));
        epp.add(&decode("team-a", 2));
        epp.build("team-a").await;
        epp.add(&aggregated("team-b", 3));
        epp.settle().await;
        assert_eq!(epp.published(), ["team-a"]);
        assert_eq!(epp.generation("team-a").prefill_namespace(), None);
        epp.stop().await;

        let mut epp = Epp::start(scope(
            NamespaceFilter::Exact("default-foo".into()),
            NamespacePrefixMode::WorkerGeneration,
        ));
        epp.add(&aggregated("default-foo-1a2b3c4d", 1));
        epp.add(&aggregated("default-foo", 2));
        epp.build("default-foo").await;
        epp.settle().await;
        assert_eq!(epp.published(), ["default-foo"]);
        epp.stop().await;
    }

    /// The controller retries a failed build.
    #[tokio::test(start_paused = true)]
    async fn retries_a_failed_build() {
        let a = "dgd-aaaaaaaa";
        let mut epp = Epp::start(dgd());
        epp.add(&aggregated(a, 1));
        epp.next_build().await.fail();
        epp.build(a).await;
        epp.settle().await;
        assert_eq!(epp.published(), [a]);
        epp.stop().await;
    }

    /// `(name, workers, ready, serves the requested model)`, chosen with
    /// `roll` capped to the weight being rolled over.
    fn choose(
        generations: &[(&'static str, usize, bool, bool)],
        roll: usize,
    ) -> Option<&'static str> {
        choose_generation(
            generations,
            |g| g.3,
            |g| g.2,
            |g| g.1,
            |total| roll.min(total - 1),
        )
        .map(|g| g.0)
    }

    #[test]
    fn choose_generation_follows_capacity() {
        let both = [("old", 3, true, false), ("new", 1, true, false)];
        assert_eq!(choose(&both, 0), Some("old"));
        assert_eq!(choose(&both, 2), Some("old"));
        assert_eq!(choose(&both, 3), Some("new"));
        assert_eq!(
            choose(&[("old", 0, true, false), ("new", 4, true, false)], 0),
            Some("new"),
            "a drained generation gets no traffic"
        );
        assert_eq!(
            choose(&[("old", 0, true, false), ("new", 0, true, false)], 0),
            Some("old"),
            "with no workers anywhere the first generation reports why"
        );
        assert_eq!(choose(&[("only", 0, false, false)], 0), Some("only"));
        assert_eq!(choose(&[], 0), None);
    }

    /// Each generation's readiness is its own: a disaggregated generation
    /// waits for its own prefill router, whatever the other generation is.
    #[test]
    fn choose_generation_judges_each_generation_on_its_own() {
        let pending_disagg = [("aggregated", 8, true, false), ("disagg", 2, false, false)];
        assert_eq!(choose(&pending_disagg, 9), Some("aggregated"));
        let ready_disagg = [("aggregated", 8, true, false), ("disagg", 2, true, false)];
        assert_eq!(
            choose(&ready_disagg, 7),
            Some("aggregated"),
            "a ready generation is weighted by capacity, not given every request"
        );
        assert_eq!(choose(&ready_disagg, 9), Some("disagg"));
        assert_eq!(
            choose(
                &[("disagg", 2, true, false), ("aggregated", 6, true, false)],
                7
            ),
            Some("aggregated"),
            "an aggregated generation is not held back by a disaggregated one"
        );
        assert_eq!(
            choose(&[("old", 0, true, false), ("new", 6, false, false)], 5),
            Some("new"),
            "serving aggregated beats no route at all"
        );
    }

    /// A request goes to a generation serving its model when one does, ready
    /// or not, and anywhere when none does.
    #[test]
    fn choose_generation_routes_by_requested_model() {
        assert_eq!(
            choose(
                &[("model-a", 4, true, false), ("model-b", 1, true, true)],
                3
            ),
            Some("model-b")
        );
        assert_eq!(
            choose(
                &[("model-a", 4, true, false), ("model-b", 1, false, true)],
                3
            ),
            Some("model-b")
        );
        assert_eq!(
            choose(
                &[("model-a", 4, true, false), ("model-b", 0, true, true)],
                3
            ),
            Some("model-b")
        );
        assert_eq!(
            choose(
                &[("model-a", 3, true, false), ("model-b", 1, true, false)],
                3
            ),
            Some("model-b")
        );
    }
}
