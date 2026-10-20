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
use dynamo_llm::discovery::{model_card_endpoint_id, model_card_instance_id, worker_set_key};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryStream};
use dynamo_runtime::namespace::{NamespaceFilter, NamespacePrefixMode};
use dynamo_runtime::protocols::EndpointId;
use rand::Rng;
use rand::seq::IndexedRandom;
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

impl<G> Entry<G> {
    /// Published, with workers and a complete topology.
    fn serving(&self) -> bool {
        self.published && !self.dormant && self.ready
    }
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

/// Lock `mutex`, recovering it if a panicking holder poisoned it.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether the EPP routes requests to a card's workers: decode or aggregated
/// workers, as opposed to the prefill and encode workers they pair with.
fn routes_requests(card: &ModelDeploymentCard) -> bool {
    matches!(
        card.effective_worker_type(),
        WorkerType::Decode | WorkerType::Aggregated
    )
}

/// Whether a published generation of `model` other than `key` has workers
/// and a complete topology, so it can take over from `key`.
fn has_replacement<G>(
    generations: &BTreeMap<GroupKey, Entry<G>>,
    key: &GroupKey,
    model: &str,
) -> bool {
    generations
        .iter()
        .any(|(other, entry)| other != key && entry.model == model && entry.serving())
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
        let mut serving = state.generations.values().any(Entry::serving);
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
        let mcid = model_card_instance_id(&instance)?;
        // A LoRA adapter's card adds to a worker whose base card already
        // counts, and a generation is identified by its base model.
        if mcid.model_suffix.is_some()
            || !namespace_filter.matches_with_prefix_mode(&mcid.namespace, self.scope.prefix_mode)
            || self.follows(&mcid.namespace) == Some(false)
        {
            return Ok(None);
        }
        let mut card = instance.deserialize_model::<ModelDeploymentCard>()?;
        normalize_legacy_prefill_topology(&mut card);
        // Requests are routed to the configured component's workers; prefill
        // and encode workers pair from any component.
        if routes_requests(&card)
            && (mcid.component != self.scope.component || !self.bind(&mcid.namespace))
        {
            return Ok(None);
        }
        let endpoint_id = model_card_endpoint_id(&mcid);
        let group_key = GroupKey {
            model_name: card.name().to_string(),
            worker_set_key: worker_set_key(&endpoint_id, card.model_type, card.worker_type),
        };
        Ok(Some(DesiredInstance {
            key: mcid.to_path(),
            mcid,
            endpoint_id,
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
        if !routes_requests(card) {
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
        match state.generations.get_mut(key) {
            // `settle` retires it once a ready replacement exists. Until
            // then, in-flight requests hold their own `Arc`, and new ones
            // fail until the workers return.
            Some(entry) if entry.published => entry.dormant = true,
            Some(_) => {
                state.generations.remove(key);
            }
            None => {}
        }
        if let Some(entry) = state.generations.get(key)
            && !has_replacement(&state.generations, key, &entry.model)
        {
            tracing::info!(
                namespace = entry.namespace,
                "Worker generation has no workers; keeping it until a replacement is ready"
            );
        }
        self.settle(&mut state, false);
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
/// router reports why.
pub(crate) fn choose_generation<'a, T>(
    generations: &'a [T],
    serves: impl Fn(&T) -> bool,
    ready: impl Fn(&T) -> bool,
    workers: impl Fn(&T) -> usize,
    rng: &mut impl Rng,
) -> Option<&'a T> {
    if generations.len() <= 1 {
        return generations.first();
    }
    let restricted = generations.iter().any(&serves);
    let candidate = |generation: &T| !restricted || serves(generation);
    let weight = |generation: &T, ready_only: bool| {
        if candidate(generation) && (!ready_only || ready(generation)) {
            workers(generation)
        } else {
            0
        }
    };
    generations
        .choose_weighted(rng, |generation| weight(generation, true))
        .or_else(|_| generations.choose_weighted(rng, |generation| weight(generation, false)))
        .ok()
        .or_else(|| generations.iter().find(|generation| candidate(generation)))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;
    use dynamo_llm::model_type::ModelType;
    use dynamo_runtime::discovery::{DiscoveryEvent, DiscoverySpec};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    const MODEL: &str = "tiny";

    fn card(model: &str, role: WorkerType, needs: Option<WorkerType>) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(model);
        card.worker_type = Some(role);
        card.needs = needs.map(|peer| vec![vec![peer]]).unwrap_or_default();
        card
    }

    /// A worker's card, registered by `component` in `namespace`, under
    /// `adapter` for a LoRA adapter's card.
    fn registration(
        namespace: &str,
        component: &str,
        instance_id: u64,
        card: &ModelDeploymentCard,
        adapter: Option<&str>,
    ) -> DiscoveryInstance {
        DiscoverySpec::from_model_with_suffix(
            namespace.to_string(),
            component.to_string(),
            "generate".to_string(),
            card,
            adapter.map(str::to_string),
        )
        .unwrap()
        .into_instance(instance_id)
    }

    /// An aggregated worker of `model`.
    fn serving(namespace: &str, instance_id: u64, model: &str) -> DiscoveryInstance {
        let card = card(model, WorkerType::Aggregated, None);
        registration(namespace, "backend", instance_id, &card, None)
    }

    fn aggregated(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        serving(namespace, instance_id, MODEL)
    }

    fn decode(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        let card = card(MODEL, WorkerType::Decode, Some(WorkerType::Prefill));
        registration(namespace, "backend", instance_id, &card, None)
    }

    /// A prefill worker as the mocker registers it: no legacy
    /// `ModelType::Prefill` bit.
    fn prefill(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        let card = card(MODEL, WorkerType::Prefill, Some(WorkerType::Decode));
        registration(namespace, "prefill", instance_id, &card, None)
    }

    /// A prefill worker from before `worker_type`: the legacy bit only.
    fn legacy_prefill(namespace: &str, instance_id: u64) -> DiscoveryInstance {
        let mut card = ModelDeploymentCard::with_name_only(MODEL);
        card.model_type = ModelType::Prefill;
        registration(namespace, "prefill", instance_id, &card, None)
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

    /// The Frontend's discovery controller with a [`GenerationHost`], over a
    /// discovery stream the test drives. Builds succeed at once.
    struct Epp {
        events: mpsc::UnboundedSender<anyhow::Result<DiscoveryEvent>>,
        /// What the controller's periodic reconciliation lists.
        registered: Arc<Mutex<Vec<DiscoveryInstance>>>,
        /// Each build's namespace, and whether it ran at startup.
        built: Arc<Mutex<Vec<(String, bool)>>>,
        published: watch::Receiver<GenerationSet<FakeGeneration>>,
        controller: tokio::task::JoinHandle<()>,
    }

    impl Epp {
        fn start(scope: GenerationScope) -> Self {
            let (events, stream) = mpsc::unbounded_channel();
            let registered = Arc::new(Mutex::new(Vec::<DiscoveryInstance>::new()));
            let built = Arc::new(Mutex::new(Vec::new()));
            let build = {
                let built = built.clone();
                move |request: BuildRequest| {
                    let namespace = request.endpoint.namespace;
                    lock(&built).push((namespace.clone(), request.startup));
                    std::future::ready(Ok::<_, anyhow::Error>(FakeGeneration {
                        namespace,
                        model: request.card.name().to_string(),
                        ready: AtomicBool::new(false),
                        prefill: Mutex::new(None),
                    }))
                }
            };
            let list = {
                let registered = registered.clone();
                move || std::future::ready(Ok::<_, anyhow::Error>(lock(&registered).clone()))
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
                built,
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

        /// Let the controller act on everything so far. The clock is paused,
        /// so it advances only once every task is idle.
        async fn settle(&self) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        /// The builds since the last call: namespace, and whether at startup.
        fn built(&self) -> Vec<(String, bool)> {
            std::mem::take(&mut *lock(&self.built))
        }

        /// The published generations as `model@namespace`, or by namespace
        /// alone for [`MODEL`].
        fn published(&self) -> Vec<String> {
            let published = self.published.borrow();
            published
                .iter()
                .map(|generation| match generation.model.as_str() {
                    MODEL => generation.namespace.clone(),
                    model => format!("{model}@{}", generation.namespace),
                })
                .collect()
        }

        fn generation(&self, model: &str, namespace: &str) -> Arc<FakeGeneration> {
            let published = self.published.borrow();
            published
                .iter()
                .find(|generation| generation.model == model && generation.namespace == namespace)
                .cloned()
                .expect("generation is published")
        }

        /// End the discovery stream, which stops the controller.
        async fn stop(self) {
            drop(self.events);
            self.controller.await.unwrap();
        }
    }

    /// What a fresh EPP publishes once `instances` register.
    async fn published_after(
        scope: GenerationScope,
        instances: &[DiscoveryInstance],
    ) -> Vec<String> {
        let epp = Epp::start(scope);
        for instance in instances {
            epp.add(instance);
        }
        epp.settle().await;
        let published = epp.published();
        epp.stop().await;
        published
    }

    /// The rolling update from #16085: generation `a` serves, `b` starts
    /// beside it and serves once built, then `a` drains and is retired.
    #[tokio::test(start_paused = true)]
    async fn follows_a_rolling_update() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        let epp = Epp::start(dgd());
        let old = aggregated(a, 1);
        epp.add(&old);
        epp.settle().await;
        epp.add(&aggregated(b, 2));
        epp.settle().await;
        assert_eq!(epp.built(), [(a.into(), true), (b.into(), false)]);
        assert_eq!(epp.published(), [a, b], "generations overlap");

        epp.remove(&old);
        epp.settle().await;
        assert_eq!(epp.published(), [b]);
        epp.stop().await;
    }

    /// During a rollout a generation takes no traffic until its own prefill
    /// workers register, unless no complete generation serves: the operator
    /// may drain the old decode or prefill workers first. Each generation's
    /// prefill router gets its own namespace's workers, legacy or current.
    #[tokio::test(start_paused = true)]
    async fn holds_back_a_generation_until_its_topology_is_complete() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        for drained in ["decode", "prefill"] {
            let epp = Epp::start(dgd());
            let (old_prefill, old_decode) = (legacy_prefill(a, 1), decode(a, 2));
            epp.add(&old_prefill);
            epp.add(&old_decode);
            epp.settle().await;
            epp.add(&decode(b, 3));
            epp.settle().await;
            assert_eq!(epp.published(), [a], "b has no prefill workers yet");
            let old = epp.generation(MODEL, a);
            assert_eq!(old.prefill_namespace().as_deref(), Some(a));

            epp.remove(if drained == "decode" {
                &old_decode
            } else {
                &old_prefill
            });
            epp.settle().await;
            assert_eq!(epp.published(), [a, b], "old {drained} drained first");

            epp.add(&prefill(b, 4));
            if drained == "prefill" {
                epp.remove(&old_decode);
            }
            epp.settle().await;
            assert_eq!(epp.published(), [b]);
            let new = epp.generation(MODEL, b);
            assert!(new.ready());
            assert_eq!(new.prefill_namespace().as_deref(), Some(b));
            epp.stop().await;
        }
    }

    /// A generation whose workers all leave keeps its place, so its requests
    /// fail there rather than reach another model's workers, until a ready
    /// generation of the same model replaces it. Workers that return are
    /// served by a rebuilt generation.
    #[tokio::test(start_paused = true)]
    async fn keeps_a_generation_until_a_replacement_of_its_model_is_ready() {
        let (a, b) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb");
        let both = ["model-x@dgd-aaaaaaaa", "model-y@dgd-aaaaaaaa"];
        let epp = Epp::start(dgd());
        let gone = serving(a, 1, "model-x");
        epp.add(&gone);
        epp.add(&serving(a, 2, "model-y"));
        epp.settle().await;
        let first = epp.generation("model-x", a);

        epp.remove(&gone);
        epp.settle().await;
        assert_eq!(epp.published(), both, "model-y is no replacement");

        let returned = serving(a, 3, "model-x");
        epp.add(&returned);
        epp.settle().await;
        assert_eq!(epp.published(), both);
        assert!(!Arc::ptr_eq(&first, &epp.generation("model-x", a)));

        epp.remove(&returned);
        epp.add(&serving(b, 4, "model-x"));
        epp.settle().await;
        assert_eq!(
            epp.published(),
            ["model-x@dgd-bbbbbbbb", "model-y@dgd-aaaaaaaa"]
        );
        epp.stop().await;
    }

    /// The EPP follows only its own DGD's generations, built from the
    /// configured component's base-model workers. Elsewhere it binds to the
    /// first namespace with a worker it routes to.
    #[tokio::test(start_paused = true)]
    async fn follows_only_its_own_deployment() {
        let (a, b, sibling) = ("dgd-aaaaaaaa", "dgd-bbbbbbbb", "dgd-bar");
        let literal = || {
            scope(
                NamespaceFilter::Prefix("dgd".into()),
                NamespacePrefixMode::Literal,
            )
        };
        let aggregated_card = card(MODEL, WorkerType::Aggregated, None);
        let encoder = card(MODEL, WorkerType::Encode, None);
        let others = [
            aggregated(sibling, 1),
            registration(a, "backend", 2, &aggregated_card, Some("adapter")),
            registration(a, "processor", 3, &aggregated_card, None),
            registration(a, "encoder", 4, &encoder, None),
            aggregated(a, 5),
        ];
        assert_eq!(published_after(dgd(), &others).await, [a]);

        // Operators before 1.6 set no strict prefix: generations are still
        // followed and a sibling DGD is not, unless the EPP bound to it first.
        let generations = [aggregated(a, 1), aggregated(sibling, 2), aggregated(b, 3)];
        assert_eq!(published_after(literal(), &generations).await, [a, b]);
        let sibling_first = [aggregated(sibling, 1), aggregated(a, 2)];
        assert_eq!(published_after(literal(), &sibling_first).await, [sibling]);

        let global = scope(
            NamespaceFilter::Global,
            NamespacePrefixMode::WorkerGeneration,
        );
        let teams = [
            prefill("team-b", 1),
            decode("team-a", 2),
            aggregated("team-b", 3),
        ];
        assert_eq!(published_after(global, &teams).await, ["team-a"]);
        let exact = scope(
            NamespaceFilter::Exact("dgd".into()),
            NamespacePrefixMode::WorkerGeneration,
        );
        let exact_first = [aggregated(a, 1), aggregated("dgd", 2)];
        assert_eq!(published_after(exact, &exact_first).await, ["dgd"]);
    }

    /// `(name, workers, ready, serves the requested model)`.
    fn choose(generations: &[(&'static str, usize, bool, bool)]) -> Option<&'static str> {
        choose_generation(generations, |g| g.3, |g| g.2, |g| g.1, &mut rand::rng()).map(|g| g.0)
    }

    /// Requests go to candidates that are ready and have workers, weighted by
    /// workers; else to any candidate with workers; else to the first
    /// candidate, whose router reports why. Candidates serve the requested
    /// model when any generation does.
    #[test]
    fn choose_generation_prefers_ready_generations_with_workers() {
        assert_eq!(choose(&[]), None);
        assert_eq!(choose(&[("only", 0, false, false)]), Some("only"));
        let overlap = [("old", 3, true, false), ("new", 1, true, false)];
        let picked: HashSet<_> = (0..1000).filter_map(|_| choose(&overlap)).collect();
        assert_eq!(picked.len(), 2, "both generations take traffic");
        for (generations, expected) in [
            (
                [("drained", 0, true, false), ("new", 4, true, false)],
                "new",
            ),
            (
                [
                    ("complete", 8, true, false),
                    ("incomplete", 2, false, false),
                ],
                "complete",
            ),
            (
                [("drained", 0, true, false), ("incomplete", 6, false, false)],
                "incomplete",
            ),
            ([("old", 0, true, false), ("new", 0, true, false)], "old"),
            (
                [("model-a", 4, true, false), ("model-b", 1, false, true)],
                "model-b",
            ),
            (
                [("model-a", 4, true, false), ("model-b", 0, true, true)],
                "model-b",
            ),
        ] {
            assert_eq!(choose(&generations), Some(expected), "{generations:?}");
        }
    }
}
