// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::affinity::SessionAffinityMode;
use super::super::input::PromptRequest;
use super::hint::transfer_hint_for_selection;
use super::reservations::{
    Reservation, ReservationClaim, ReservationIndexObserver, sweep_reservation_index,
};
use super::*;
use crate::protocols::ActiveSequenceEventData;
use crate::protocols::{RoutingConstraints, StorageTier};
use crate::services::common::replica_sync::HostReplicaChannels;
use crate::services::indexer::backend::test_util::store_event;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::sleep;
use std::time::Duration;

fn test_config(use_kv_events: bool) -> crate::config::KvRouterConfig {
    crate::config::KvRouterConfig {
        use_kv_events,
        router_queue_threshold: None,
        ..Default::default()
    }
}

fn local_core(config: crate::config::KvRouterConfig) -> SelectionCore {
    local_core_with(config, 1, CancellationToken::new())
}

fn local_core_with(
    config: crate::config::KvRouterConfig,
    indexer_threads: usize,
    cancel_token: CancellationToken,
) -> SelectionCore {
    SelectionCore::try_new_local(
        config,
        indexer_threads,
        cancel_token,
        SelectionCacheConfig::default(),
    )
    .expect("valid test config")
}

/// `new_inner` with the test defaults; `remote_indexer` swaps the primary for
/// the standalone indexer at that URL.
fn core_with(
    config: crate::config::KvRouterConfig,
    host: SelectionHost,
    policy_factory: Option<WorkerSelectionPolicyFactory>,
    worker_type: WorkerType,
    remote_indexer: Option<String>,
    affinity: Option<SessionAffinityConfig>,
) -> SelectionCore {
    let tracking_hash = Arc::new(
        TrackingHashContext::from_config(&config).expect("valid tracking hash configuration"),
    );
    let mut indexer_policy = IndexerPolicy::from_router_config(&config).expect("indexer policy");
    if let Some(base_url) = remote_indexer {
        indexer_policy = indexer_policy
            .with_remote_indexer(base_url)
            .expect("remote policy");
    }
    SelectionCore::new_inner(
        config,
        1,
        CancellationToken::new(),
        None,
        policy_factory,
        host,
        worker_type,
        true,
        SelectionCacheConfig::default(),
        tracking_hash,
        indexer_policy,
        affinity,
    )
}

fn core_with_host(host: SelectionHost) -> SelectionCore {
    core_with_host_and_policy(host, None)
}

fn core_with_host_and_policy(
    host: SelectionHost,
    policy_factory: Option<WorkerSelectionPolicyFactory>,
) -> SelectionCore {
    core_with(
        test_config(false),
        host,
        policy_factory,
        WorkerType::Aggregated,
        None,
        None,
    )
}

fn replay_reservation(selection_id: &str) -> ReservationRequest {
    ReservationRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        selection_id: selection_id.to_string(),
        worker_id: None,
        dp_rank: None,
        prompt: PromptRequest::default(),
        router_config_override: None,
        expected_output_tokens: None,
        effective_prefill_tokens: None,
        track_prefill_tokens: None,
    }
}

fn worker(worker_id: WorkerId) -> WorkerRequest {
    WorkerRequest {
        worker_id,
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        endpoint: Some(format!("http://worker-{worker_id}:8000")),
        kv_events_endpoint: None,
        kv_events_endpoints: HashMap::new(),
        replay_endpoint: None,
        block_size: Some(4),
        data_parallel_start_rank: None,
        data_parallel_size: None,
        max_num_batched_tokens: Some(1024),
        total_kv_blocks: None,
        stable_routing_id: None,
        is_eagle: None,
        taints: HashSet::new(),
        topology_domains: HashMap::new(),
        kv_transfer_domain: None,
        kv_transfer_enforcement: None,
        kv_transfer_preferred_weight: None,
        router_hint_worker_type: None,
        router_hint_source_control_endpoints: HashMap::new(),
        kv_event_source_mode: None,
    }
}

fn worker_with_kv_events(worker_id: WorkerId) -> WorkerRequest {
    WorkerRequest {
        kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
        ..worker(worker_id)
    }
}

fn prompt() -> PromptRequest {
    PromptRequest {
        token_ids: Some(vec![1, 2, 3, 4]),
        mm_routing_info: None,
        block_mm_infos: None,
        block_hashes: None,
        sequence_hashes: None,
        isl_tokens: None,
        lora_name: None,
        cache_namespace: None,
        is_eagle: None,
    }
}

fn select_request() -> SelectRequest {
    SelectRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        selection_id: None,
        prompt: prompt(),
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        advisory: false,
    }
}

fn reserve_request(selection_id: &str) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(selection_id.to_string()),
        prompt: prompt(),
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
    }
}

async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
}

async fn wait_for_pending_selection(core: &SelectionCore) {
    wait_until("pending selection", || {
        core.loads(Some("model"), Some("default"))[0].pending_count == 1
    })
    .await;
}

fn assert_shutdown_error(error: SelectionError) {
    assert!(matches!(
        error,
        SelectionError::NotReady(message)
            if message == "selection service is shutting down"
    ));
}

fn default_key() -> RoutingPartitionId {
    RoutingPartitionId::new("model", "default")
}

/// A core whose queue threshold is zero, so a second booking queues behind
/// the first.
fn saturated_core() -> Arc<SelectionCore> {
    let mut config = test_config(false);
    config.router_queue_threshold = Some(0.0);
    Arc::new(local_core(config))
}

fn lease_operation<'a>(
    prompt: PromptView<'a>,
    request_id: &str,
    track_active_blocks: bool,
) -> SelectionOperation<'a> {
    SelectionOperation {
        key: default_key(),
        prompt,
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        session: SessionBinding::None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        admission: SelectionAdmission::Lease {
            request_id: request_id.to_string(),
        },
        track_active_blocks,
        return_routing_hashes: false,
        replay_id: None,
    }
}

#[test]
fn parent_cancel_cancels_core() {
    let parent = CancellationToken::new();
    let core = local_core_with(test_config(false), 1, parent.clone());

    assert!(!core.cancel_token.is_cancelled());
    parent.cancel();
    assert!(core.cancel_token.is_cancelled());
}

#[tokio::test]
async fn selection_setup_uses_worker_type_label() {
    for (worker_type, expected_label) in [
        (WorkerType::Prefill, "prefill"),
        (WorkerType::Decode, "decode"),
        (WorkerType::Encode, "encode"),
        (WorkerType::Aggregated, "aggregated"),
    ] {
        let core = core_with(
            test_config(false),
            SelectionHost::default(),
            None,
            worker_type,
            None,
            None,
        );

        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let entry = core.entry(&default_key()).expect("selection entry");
        assert_eq!(
            entry.scheduler.worker_type(),
            expected_label,
            "{worker_type}"
        );
    }
}

async fn wait_for_overlap(
    core: &SelectionCore,
    request: impl Fn() -> SelectRequest,
) -> SelectResponse {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let response = core.select(request()).await.expect("select");
            if response.overlap.longest_matched > 0 {
                return response;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("approximate indexer never credited the booked prompt")
}

#[tokio::test]
async fn bookings_populate_the_approximate_primary_without_kv_events() {
    // use_kv_events=false: the primary is approximate and bookings feed it.
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.upsert_worker(worker(2)).await.expect("worker upsert");

    // Query-only selection records nothing.
    let first = core.select(select_request()).await.expect("select");
    assert_eq!(first.overlap.longest_matched, 0);

    // select_and_reserve records the routed prefix for the chosen worker.
    let booked = core
        .select_and_reserve(reserve_request("booked"))
        .await
        .expect("reserve");
    let credited = wait_for_overlap(&core, || {
        let mut request = select_request();
        request.allowed_worker_ids = Some(HashSet::from([booked.worker_id]));
        request
    })
    .await;
    assert_eq!(credited.worker_id, booked.worker_id);
    assert_eq!(credited.overlap.longest_matched, 4);

    // The cached replay path records too, on a different prompt.
    let prompt_b = || PromptRequest {
        token_ids: Some(vec![5, 6, 7, 8]),
        ..PromptRequest::default()
    };
    let mut request = select_request();
    request.prompt = prompt_b();
    request.selection_id = Some("cached".to_string());
    request.allowed_worker_ids = Some(HashSet::from([2]));
    core.select(request).await.expect("select");
    core.create_reservation(replay_reservation("cached"))
        .await
        .expect("cached reservation");
    let credited = wait_for_overlap(&core, || {
        let mut request = select_request();
        request.prompt = prompt_b();
        request.allowed_worker_ids = Some(HashSet::from([2]));
        request
    })
    .await;
    assert_eq!(credited.worker_id, 2);

    // And the explicit reservation form.
    let prompt_c = || PromptRequest {
        token_ids: Some(vec![9, 10, 11, 12]),
        ..PromptRequest::default()
    };
    core.create_reservation(ReservationRequest {
        worker_id: Some(1),
        prompt: prompt_c(),
        ..replay_reservation("explicit")
    })
    .await
    .expect("explicit reservation");
    let credited = wait_for_overlap(&core, || {
        let mut request = select_request();
        request.prompt = prompt_c();
        request.allowed_worker_ids = Some(HashSet::from([1]));
        request
    })
    .await;
    assert_eq!(credited.worker_id, 1);
}

#[tokio::test]
async fn unreachable_remote_indexer_is_reported_not_ready() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    drop(listener);
    let core = core_with(
        test_config(true),
        SelectionHost::default(),
        None,
        WorkerType::Aggregated,
        Some(base_url),
        None,
    );
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    assert!(matches!(
        core.select(select_request()).await,
        Err(SelectionError::Indexer(
            crate::indexer::KvRouterError::IndexerOffline
        ))
    ));
}

#[tokio::test]
async fn remote_indexer_serves_selection_without_local_kv_listeners() {
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{BlockHashOptions, StorageTier, compute_block_hash_for_seq};
    use crate::services::indexer::registry::WorkerRegistry;
    use crate::services::indexer::server::spawn_test_indexer_server;

    // The standalone indexer holds worker 2's cache for the test prompt.
    let key = default_key();
    let served = Arc::new(WorkerRegistry::new(1));
    let served_indexer = served.get_or_create_indexer(key.clone(), 4);
    let hashes: Vec<u64> =
        compute_block_hash_for_seq(&[1, 2, 3, 4], 4, BlockHashOptions::default())
            .into_iter()
            .map(|hash| hash.0)
            .collect();
    served_indexer
        .apply_event_routed(store_event(2, 0, 1, &[], &hashes, StorageTier::Device))
        .await
        .unwrap();
    if let Indexer::Single { primary, .. } = &served_indexer {
        let _ = primary.flush().await;
    }
    let (base_url, server) = spawn_test_indexer_server(served).await;

    // use_kv_events=true, but the primary is remote: workers need no
    // kv_events endpoints and no ZMQ listener is started here.
    let core = core_with(
        test_config(true),
        SelectionHost::default(),
        None,
        WorkerType::Aggregated,
        Some(base_url),
        None,
    );
    assert!(!core.listens_for_kv_events);
    for worker_id in [1, 2] {
        let record = core
            .upsert_worker(worker(worker_id))
            .await
            .expect("worker upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable, "{record:?}");
    }

    let response = core.select(select_request()).await.expect("select");
    assert_eq!(
        response.worker_id, 2,
        "remote cache credit steers selection"
    );
    assert_eq!(response.overlap.longest_matched, 4);

    core.delete_worker(2).await.expect("delete worker");
    server.abort();
}

/// Two event-driven workers; worker 1 holds every block of an 8-token
/// prompt. `configure` adjusts each registration before upsert.
async fn hint_fixture(
    configure: impl Fn(&mut WorkerRequest),
) -> (SelectionCore, Arc<SelectionEntry>, Vec<u32>) {
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{BlockHashOptions, compute_block_hash_for_seq};

    let core = local_core(test_config(true));
    for worker_id in [1, 2] {
        let mut request = worker_with_kv_events(worker_id);
        configure(&mut request);
        core.upsert_worker(request).await.expect("worker upsert");
    }
    let entry = core.entry(&default_key()).expect("entry");
    let tokens: Vec<u32> = (1..=8).collect();
    let hashes: Vec<u64> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
        .into_iter()
        .map(|hash| hash.0)
        .collect();
    entry
        .indexer
        .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
        .await
        .unwrap();
    if let Indexer::Single { primary, .. } = &entry.indexer {
        let _ = primary.flush().await;
    }
    (core, entry, tokens)
}

fn hint_capable(request: &mut WorkerRequest) {
    request.router_hint_worker_type = Some("decode".to_string());
    request.router_hint_source_control_endpoints =
        HashMap::from([(0, format!("tcp://worker-{}:9000", request.worker_id))]);
}

async fn reserve_pinned(
    core: &SelectionCore,
    selection_id: &str,
    tokens: &[u32],
    worker_id: WorkerId,
) -> SelectResponse {
    let mut request = reserve_request(selection_id);
    request.prompt = PromptRequest {
        token_ids: Some(tokens.to_vec()),
        ..PromptRequest::default()
    };
    request.pinned_worker = Some(WorkerWithDpRank::new(worker_id, 0));
    core.select_and_reserve(request).await.expect("reserve")
}

#[tokio::test]
async fn booked_selection_attaches_router_hint_from_a_better_source() {
    let (core, entry, tokens) = hint_fixture(hint_capable).await;
    assert!(entry.indexer.supports_kv_transfer_chain_retention());

    // Booking on worker 2: worker 1 is a same-role source with a longer prefix.
    let response = reserve_pinned(&core, "to-worker-2", &tokens, 2).await;
    let hint = response.kv_hint.expect("router hint for worker 2");
    assert_eq!(hint.message_id, "to-worker-2");
    assert_eq!(hint.actions[0].action_type, "kv.fetch");
    let payload: KvSourceLocationsPayload =
        serde_json::from_value(serde_json::to_value(&hint.actions[0].payload).unwrap()).unwrap();
    assert_eq!(payload.source_control_endpoint, "tcp://worker-1:9000");
    assert_eq!(payload.block_hashes.len(), 2);

    // Booking on worker 1 itself: nothing holds a longer prefix.
    let response = reserve_pinned(&core, "to-worker-1", &tokens, 1).await;
    assert!(response.kv_hint.is_none());

    // Query-only selections never carry a hint.
    let mut request = select_request();
    request.prompt = PromptRequest {
        token_ids: Some(tokens.clone()),
        ..PromptRequest::default()
    };
    request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
    let response = core.select(request).await.expect("select");
    assert!(response.kv_hint.is_none());
}

#[tokio::test]
async fn state_agent_workers_are_not_router_hint_sources() {
    let (core, _entry, tokens) = hint_fixture(|request| {
        hint_capable(request);
        if request.worker_id == 1 {
            request.kv_event_source_mode = Some("state_agent_v2".to_string());
        }
    })
    .await;
    // Worker 1 holds the prefix but reports through a state agent.
    let response = reserve_pinned(&core, "to-worker-2", &tokens, 2).await;
    assert!(response.kv_hint.is_none());
}

#[tokio::test]
async fn router_hint_needs_capable_workers() {
    let (core, _entry, tokens) = hint_fixture(|_| {}).await;
    let response = reserve_pinned(&core, "plain", &tokens, 2).await;
    assert!(response.kv_hint.is_none());
}

/// The partition's hint-capability flag follows catalog membership: set by
/// the upsert that adds a capable worker, cleared by the delete that removes
/// the last one, unaffected by other partitions. Deleted workers leave the
/// catalog instead of lingering as `Unschedulable`.
#[tokio::test]
async fn hint_capability_tracks_catalog_membership() {
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{BlockHashOptions, compute_block_hash_for_seq};

    let (core, entry, tokens) = hint_fixture(|_| {}).await;
    let hashes: Vec<u64> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
        .into_iter()
        .map(|hash| hash.0)
        .collect();
    let seed_worker_1 = || async {
        entry
            .indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        if let Indexer::Single { primary, .. } = &entry.indexer {
            let _ = primary.flush().await;
        }
    };
    let capable = |worker_id: WorkerId| {
        let mut request = worker_with_kv_events(worker_id);
        hint_capable(&mut request);
        request
    };
    let flag = || super::hint::hint_capable_partition(&entry.workers_tx.borrow());

    assert!(!flag());
    let response = reserve_pinned(&core, "before", &tokens, 2).await;
    assert!(response.kv_hint.is_none());

    // A capable worker in another partition does not flip this one.
    let mut other = capable(9);
    other.routing_group = "group-b".to_string();
    core.upsert_worker(other).await.expect("group-b upsert");
    assert!(!flag());

    // Re-registering both workers with hint capability enables hints.
    core.upsert_worker(capable(1)).await.expect("upsert 1");
    core.upsert_worker(capable(2)).await.expect("upsert 2");
    seed_worker_1().await;
    let response = reserve_pinned(&core, "enabled", &tokens, 2).await;
    assert!(
        response.kv_hint.is_some(),
        "capable partition attaches hints"
    );
    assert!(flag());

    // Deleting the capable workers clears the flag and drops their records.
    core.delete_worker(1).await.expect("delete 1");
    assert!(flag(), "worker 2 is still capable");
    assert_eq!(core.list_workers(None, None).len(), 2, "1 left the catalog");
    core.delete_worker(2).await.expect("delete 2");
    assert!(!flag());
    assert_eq!(
        core.list_workers(None, None).len(),
        1,
        "only group-b remains"
    );
    assert!(core.list_workers(Some("model"), Some("default")).is_empty());

    // Non-capable replacements keep the flag clear and hints off.
    core.upsert_worker(worker_with_kv_events(1))
        .await
        .expect("plain upsert 1");
    core.upsert_worker(worker_with_kv_events(2))
        .await
        .expect("plain upsert 2");
    seed_worker_1().await;
    assert!(!flag());
    let response = reserve_pinned(&core, "after", &tokens, 2).await;
    assert!(response.kv_hint.is_none());

    // Re-adding capable workers restores hints.
    core.upsert_worker(capable(1))
        .await
        .expect("upsert 1 again");
    core.upsert_worker(capable(2))
        .await
        .expect("upsert 2 again");
    seed_worker_1().await;
    assert!(flag());
    let response = reserve_pinned(&core, "restored", &tokens, 2).await;
    assert!(
        response.kv_hint.is_some(),
        "re-added capable worker restores hints"
    );
}

#[tokio::test]
async fn event_driven_indexer_does_not_record_bookings() {
    let core = local_core(test_config(true));
    core.upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker upsert");
    core.select_and_reserve(reserve_request("booked"))
        .await
        .expect("reserve");
    core.free_reservation("booked").await.expect("free");
    for _ in 0..3 {
        let response = core.select(select_request()).await.expect("select");
        assert_eq!(response.overlap.longest_matched, 0);
        tokio::task::yield_now().await;
    }
}

/// Picker that records what worker selection saw and always takes row 0.
struct CapturingPicker {
    observed: Arc<parking_lot::Mutex<Vec<SelectionObservation>>>,
}

#[derive(Debug, Clone)]
struct SelectionObservation {
    session_context: Option<SessionContext>,
    shared_beyond_device_blocks: Vec<u32>,
}

impl crate::scheduling::selector::WorkerPicker for CapturingPicker {
    fn required_worker_inputs(&self) -> crate::scheduling::selector::WorkerInputs {
        crate::scheduling::selector::WorkerInputs::CACHE
    }

    fn pick(
        &mut self,
        context: &crate::scheduling::selector::WorkerSelectionContext<'_>,
        input: crate::scheduling::selector::WorkerInputView<'_>,
    ) -> Result<usize, crate::scheduling::WorkerSelectionPolicyError> {
        self.observed.lock().push(SelectionObservation {
            session_context: context.session_context().cloned(),
            shared_beyond_device_blocks: input
                .cache()
                .expect("CACHE inputs requested")
                .iter()
                .map(|cache| cache.shared_beyond_device_blocks())
                .collect(),
        });
        Ok(0)
    }
}

fn capturing_policy_factory() -> (
    WorkerSelectionPolicyFactory,
    Arc<parking_lot::Mutex<Vec<SelectionObservation>>>,
) {
    let observed = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let factory_observed = Arc::clone(&observed);
    let factory: WorkerSelectionPolicyFactory = Arc::new(move |config, worker_type, _partition| {
        WorkerSelectionPolicy::new(
            config.clone(),
            worker_type.as_str(),
            Vec::new(),
            Box::new(CapturingPicker {
                observed: Arc::clone(&factory_observed),
            }),
        )
    });
    (factory, observed)
}

type SharedCacheCalls = Arc<parking_lot::Mutex<Vec<(Vec<u32>, u32, Option<String>)>>>;

/// Shared cache that reports every block as a hit and records each query.
struct RecordingSharedCache {
    calls: SharedCacheCalls,
}

#[async_trait::async_trait]
impl SharedKvCache for RecordingSharedCache {
    async fn check_blocks(
        &self,
        tokens: &[u32],
        block_size: u32,
        cache_namespace: Option<&str>,
    ) -> Result<SharedCacheHits, crate::indexer::KvRouterError> {
        self.calls.lock().push((
            tokens.to_vec(),
            block_size,
            cache_namespace.map(str::to_string),
        ));
        let blocks = (tokens.len() / block_size as usize) as u32;
        Ok(SharedCacheHits::from_hits(&vec![true; blocks as usize]))
    }
}

struct OnlyWorkerForLora {
    worker_id: WorkerId,
}

impl LoraWorkerFilter for OnlyWorkerForLora {
    fn filter_worker_ids_for_lora(
        &self,
        _lora_name: &str,
        available: &[WorkerId],
    ) -> Vec<WorkerId> {
        available
            .iter()
            .copied()
            .filter(|id| *id == self.worker_id)
            .collect()
    }
}

#[tokio::test]
async fn injected_lora_filter_narrows_candidates() {
    let core = core_with_host(SelectionHost {
        eligibility: HostEligibility {
            lora_worker_filter: Some(Arc::new(OnlyWorkerForLora { worker_id: 2 })),
        },
        ..SelectionHost::default()
    });
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.upsert_worker(worker(2)).await.expect("worker upsert");

    // LoRA request: only the filter's worker is eligible. Allow-set and
    // pinned-worker interplay is pinned by the `lora_filter` unit tests.
    for _ in 0..4 {
        let mut request = select_request();
        request.prompt.lora_name = Some("adapter-a".to_string());
        let response = core.select(request).await.expect("select");
        assert_eq!(response.worker_id, 2);
    }
}

#[tokio::test]
async fn shared_cache_hits_reach_worker_selection() {
    let calls: SharedCacheCalls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let (factory, observed) = capturing_policy_factory();
    let core = core_with_host_and_policy(
        SelectionHost {
            cache: HostCache {
                shared: Some(Arc::new(RecordingSharedCache {
                    calls: Arc::clone(&calls),
                })),
                ..HostCache::default()
            },
            ..SelectionHost::default()
        },
        Some(factory),
    );
    core.upsert_worker(worker(1)).await.expect("worker upsert");

    let mut request = select_request();
    request.prompt.cache_namespace = Some("tenant-a".to_string());
    core.select(request).await.expect("select");

    assert_eq!(
        calls.lock().as_slice(),
        &[(vec![1, 2, 3, 4], 4, Some("tenant-a".to_string()))]
    );
    let observations = observed.lock().clone();
    assert_eq!(observations.len(), 1);
    // One block in the prompt, no device overlap, so the whole prompt is a
    // shared-cache hit beyond the device prefix.
    assert_eq!(observations[0].shared_beyond_device_blocks, vec![1]);

    // Load projection does not consult the shared cache.
    core.potential_loads(PotentialLoadsRequest {
        model_name: "model".to_string(),
        routing_group: "default".to_string(),
        prompt: prompt(),
        router_config_override: None,
    })
    .await
    .expect("potential loads");
    assert_eq!(calls.lock().len(), 1);

    // Prompts without raw tokens cannot be checked against the shared cache.
    let mut request = select_request();
    request.prompt = PromptRequest {
        token_ids: None,
        block_hashes: Some(vec![11]),
        sequence_hashes: Some(vec![101]),
        isl_tokens: Some(4),
        ..PromptRequest::default()
    };
    core.select(request).await.expect("select");
    assert_eq!(calls.lock().len(), 1);
    assert_eq!(observed.lock()[1].shared_beyond_device_blocks, vec![0]);
}

#[tokio::test]
async fn session_context_reaches_worker_selection() {
    use super::super::types::SelectionSessionContext;

    let (factory, observed) = capturing_policy_factory();
    let core = core_with_host_and_policy(SelectionHost::default(), Some(factory));
    core.upsert_worker(worker(1)).await.expect("worker upsert");

    let mut request = select_request();
    request.session_id = Some("ignored-legacy".to_string());
    request.session_context = Some(SelectionSessionContext {
        session_id: "child-session".to_string(),
        parent_session_id: Some("root-session".to_string()),
        session_final: Some(true),
        input_trigger: Some(super::super::types::SelectionInputTrigger::ToolResult),
    });
    core.select(request).await.expect("select");

    let mut request = reserve_request("legacy-session-reservation");
    request.session_id = Some("legacy-only".to_string());
    core.select_and_reserve(request)
        .await
        .expect("select and reserve");

    let observations = observed.lock();
    let context = observations[0]
        .session_context
        .as_ref()
        .expect("structured session context");
    assert_eq!(context.session_id(), "child-session");
    assert_eq!(context.parent_session_id(), Some("root-session"));
    assert_eq!(context.session_final(), Some(true));
    assert_eq!(
        context.input_trigger(),
        Some(crate::scheduling::WorkerSelectionInputTrigger::ToolResult)
    );

    let legacy = observations[1]
        .session_context
        .as_ref()
        .expect("legacy session context");
    assert_eq!(legacy.session_id(), "legacy-only");
    assert_eq!(legacy.parent_session_id(), None);
}

#[tokio::test]
async fn full_affinity_table_routes_without_pinning() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let entry = core.entry(&default_key()).expect("entry");
    let table = SessionAffinity::with_config(SessionAffinityConfig {
        max_entries: 1,
        max_session_id_bytes: 256,
        ..SessionAffinityConfig::new(Duration::from_secs(60))
    })
    .expect("affinity table");
    assert!(entry.affinity.set(table).is_ok());

    for (selection_id, session_id) in [("first", "s1"), ("second", "s2")] {
        let mut request = reserve_request(selection_id);
        request.session_id = Some(session_id.to_string());
        core.select_and_reserve(request)
            .await
            .expect("a full affinity table must not fail selection");
    }
    assert_eq!(
        core.reservation_index
            .read()
            .values()
            .filter(|r| r._affinity_lease.is_some())
            .count(),
        1
    );
}

/// Both `HostLoad` providers reach the scheduler; their semantics are pinned
/// by the `scheduling::queue` unit tests.
#[tokio::test]
async fn injected_load_providers_restrict_selection() {
    let available: Arc<parking_lot::Mutex<Option<Arc<HashSet<WorkerId>>>>> =
        Arc::new(parking_lot::Mutex::new(None));
    let provider_state = Arc::clone(&available);
    let core = core_with_host(SelectionHost {
        load: HostLoad {
            available_workers: Some(Arc::new(move || provider_state.lock().clone())),
            overloaded_workers: Some(Arc::new(|| Some(HashSet::from([1])))),
            ..HostLoad::default()
        },
        ..SelectionHost::default()
    });
    for worker_id in [1, 2, 3] {
        core.upsert_worker(worker(worker_id))
            .await
            .expect("worker upsert");
    }

    // The overloaded worker is excluded while availability is unrestricted.
    for _ in 0..4 {
        let response = core.select(select_request()).await.expect("select");
        assert_ne!(response.worker_id, 1);
    }
    // The availability provider narrows the remaining candidates.
    for only in [2, 3] {
        *available.lock() = Some(Arc::new(HashSet::from([only])));
        for _ in 0..4 {
            let response = core.select(select_request()).await.expect("select");
            assert_eq!(response.worker_id, only);
        }
    }
}

#[tokio::test]
async fn shutdown_cancels_listeners_but_keeps_parent_alive() {
    let parent = CancellationToken::new();
    let core = local_core_with(test_config(true), 1, parent.clone());

    let record = core
        .upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
    assert_eq!(core.indexer_registry.listener_cancelled(1, 0), Some(false));

    core.shutdown();
    assert!(core.cancel_token.is_cancelled());
    assert!(!parent.is_cancelled());
    assert_eq!(core.indexer_registry.listener_cancelled(1, 0), Some(true));
}

#[rstest::rstest]
#[case(1)]
#[case(2)]
#[tokio::test]
async fn selection_sees_cache_after_last_worker_replacement(#[case] indexer_threads: usize) {
    let core = local_core_with(test_config(true), indexer_threads, CancellationToken::new());
    let key = default_key();
    let request = || {
        let mut request = select_request();
        request.prompt.token_ids = None;
        request.prompt.block_hashes = Some(vec![11]);
        request.prompt.sequence_hashes = Some(vec![101]);
        request.prompt.isl_tokens = Some(4);
        request
    };

    // Exercise an in-place update, then removing the last worker and adding a new one.
    // An in-place update whose listener endpoints are unchanged keeps the rank's
    // listener and index rows; only a replacement starts from an empty cache.
    for (worker_id, cached_before_store) in [(1, 0), (1, 4), (2, 0)] {
        if worker_id == 2 {
            core.delete_worker(1).await.expect("delete last worker");
        }
        core.upsert_worker(worker_with_kv_events(worker_id))
            .await
            .expect("worker upsert");
        assert_eq!(
            core.select(request()).await.unwrap().overlap.gpu,
            cached_before_store
        );

        // Write through the registry used by listeners, not the selector's saved reference.
        let indexer = core
            .indexer_registry
            .get_indexer(&key)
            .unwrap()
            .indexer
            .clone();
        indexer
            .apply_event_routed(store_event(
                worker_id,
                0,
                1,
                &[],
                &[11],
                StorageTier::Device,
            ))
            .await
            .unwrap();
        indexer.dump_events().await.expect("flush indexer");
        let selected = core.select(request()).await.expect("select cached worker");
        assert_eq!(selected.worker_id, worker_id);
        assert_eq!(selected.overlap.gpu, 4);
    }
    core.shutdown();
}

#[tokio::test]
async fn multi_rank_worker_with_replay_endpoint_is_incomplete() {
    let core = local_core(test_config(true));

    let record = core
        .upsert_worker(WorkerRequest {
            data_parallel_size: Some(2),
            kv_events_endpoints: HashMap::from([
                (0, "tcp://127.0.0.1:5557".to_string()),
                (1, "tcp://127.0.0.1:5558".to_string()),
            ]),
            replay_endpoint: Some("tcp://127.0.0.1:5600".to_string()),
            ..worker(1)
        })
        .await
        .expect("worker upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Incomplete, "{record:?}");
    assert!(!core.indexer_registry.has_worker(1));
}

#[tokio::test]
async fn reupsert_recreates_a_listener_lost_to_a_cancelled_update() {
    use crate::indexer::KvIndexerInterface;

    let core = local_core(test_config(true));
    core.upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker upsert");
    let entry = core.entry(&default_key()).expect("entry");
    let hashes = [11u64, 12];
    entry
        .indexer
        .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
        .await
        .unwrap();
    if let Indexer::Single { primary, .. } = &entry.indexer {
        let _ = primary.flush().await;
    }
    // An endpoint update cancelled right after removing the listener leaves
    // the catalog record schedulable, the listener gone, and its blocks indexed.
    core.indexer_registry.forget_listener(1, 0);

    let record = core
        .upsert_worker(worker_with_kv_events(1))
        .await
        .expect("worker re-upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable, "{record:?}");
    assert!(core.indexer_registry.has_listener(1, 0));
    // The purge is queued to the indexer thread; wait for it to land.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let overlap = entry
                .indexer
                .find_matches(hashes.iter().copied().map(LocalBlockHash).collect())
                .await
                .expect("find matches");
            if overlap.scores.is_empty() {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("stale blocks survived the re-upsert");
}

#[tokio::test]
async fn upsert_moves_global_worker_id_between_routing_groups() {
    let core = local_core(test_config(true));
    let mut group_a = worker_with_kv_events(1);
    group_a.routing_group = "group-a".to_string();
    core.upsert_worker(group_a).await.expect("group A upsert");
    let indexed = |group: &str| {
        core.indexer_registry
            .list_filtered(Some("model"), Some(group))
            .len()
    };
    assert_eq!(indexed("group-a"), 1);

    let mut group_b = worker_with_kv_events(1);
    group_b.routing_group = "group-b".to_string();
    core.upsert_worker(group_b).await.expect("group B upsert");

    assert!(core.list_workers(Some("model"), Some("group-a")).is_empty());
    assert_eq!(core.list_workers(Some("model"), Some("group-b")).len(), 1);
    assert_eq!(indexed("group-a"), 0);
    assert_eq!(indexed("group-b"), 1);

    let mut select_a = select_request();
    select_a.routing_group = "group-a".to_string();
    assert!(matches!(
        core.select(select_a).await,
        Err(SelectionError::NotReady(_))
    ));
    let mut select_b = select_request();
    select_b.routing_group = "group-b".to_string();
    assert_eq!(core.select(select_b).await.unwrap().worker_id, 1);

    core.delete_worker(1).await.expect("delete group B worker");
    assert_eq!(indexed("group-b"), 0);
}

#[tokio::test]
async fn shutdown_reports_not_ready_and_rejects_new_work() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    assert!(core.ready().ready);

    core.shutdown();

    let ready = core.ready();
    assert!(!ready.ready);
    assert_eq!(ready.schedulable_workers, 1);

    let upsert_error = core
        .upsert_worker(worker(2))
        .await
        .expect_err("upsert should fail after shutdown");
    assert_shutdown_error(upsert_error);

    let patch = serde_json::from_value(serde_json::json!({
        "endpoint": "http://worker-1:9000"
    }))
    .expect("worker patch");
    let patch_error = core
        .patch_worker(1, patch)
        .await
        .expect_err("patch should fail after shutdown");
    assert_shutdown_error(patch_error);

    let select_error = core
        .select(select_request())
        .await
        .expect_err("selection should fail after shutdown");
    assert_shutdown_error(select_error);

    let reservation_error = core
        .create_reservation(ReservationRequest {
            worker_id: Some(1),
            prompt: prompt(),
            ..replay_reservation("res-after-shutdown")
        })
        .await
        .expect_err("reservation should fail after shutdown");
    assert_shutdown_error(reservation_error);

    assert_eq!(core.list_workers(None, None).len(), 1);
    assert_eq!(core.loads(None, None).len(), 1);
    let deleted = core
        .delete_worker(1)
        .await
        .expect("delete should remain available after shutdown");
    assert_eq!(deleted.lifecycle, WorkerLifecycle::Unschedulable);
}

#[tokio::test]
async fn queued_selection_errors_on_shutdown() {
    let core = saturated_core();

    let record = core.upsert_worker(worker(1)).await.expect("worker upsert");
    assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
    core.select_and_reserve(reserve_request("res-a"))
        .await
        .expect("initial reservation");

    let queued_core = core.clone();
    let queued = tokio::spawn(async move { queued_core.select(select_request()).await });
    wait_for_pending_selection(&core).await;

    core.shutdown();
    let err = tokio::time::timeout(Duration::from_secs(1), queued)
        .await
        .expect("queued selection timed out")
        .expect("queued selection task panicked")
        .expect_err("queued selection should fail");

    assert!(matches!(
        err,
        SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown)
    ));
}

#[tokio::test]
async fn booking_is_freed_when_selected_worker_drained_while_queued() {
    let core = saturated_core();
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let entry = core.entry(&default_key()).expect("entry");
    core.select_and_reserve(reserve_request("res-a"))
        .await
        .expect("initial reservation");

    let queued_core = core.clone();
    let queued = tokio::spawn(async move {
        queued_core
            .select_and_reserve(reserve_request("queued"))
            .await
    });
    wait_for_pending_selection(&core).await;
    // First half of `delete_worker`: the worker drains while the request waits.
    core.catalog
        .set_lifecycle(1, WorkerLifecycle::Draining, Vec::new());
    core.free_reservation("res-a").await.expect("free res-a");

    let err = tokio::time::timeout(Duration::from_secs(2), queued)
        .await
        .expect("queued selection timed out")
        .expect("task panicked")
        .expect_err("drained worker is not schedulable");
    assert!(
        matches!(&err, SelectionError::Internal(m) if m.contains("no longer schedulable")),
        "{err:?}"
    );
    wait_until("booking release", || !entry.scheduler.has_request("queued")).await;
}

/// The embedded host dispatches by worker id, so a worker that drains
/// while the request queues still comes back selected: the transport
/// reports the departure and migration takes over. The endpoint is the
/// only thing missing.
#[tokio::test]
async fn lease_admission_keeps_a_selection_whose_worker_drained_while_queued() {
    let core = saturated_core();
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let entry = core.entry(&default_key()).expect("entry");
    core.select_and_reserve(reserve_request("res-a"))
        .await
        .expect("initial reservation");

    let queued_core = core.clone();
    let queued = tokio::spawn(async move {
        let req = reserve_request("queued");
        let run = queued_core
            .run_selection(lease_operation(req.prompt.view(), "queued", false))
            .await;
        match run.result {
            Ok(SelectionOutcome::Selected(selected)) => (
                selected.response.best_worker.worker_id,
                selected.endpoint,
                selected.booking.is_some(),
            ),
            Ok(SelectionOutcome::QueueRejected { .. }) => panic!("queue rejected"),
            Err(error) => panic!("lease selection failed: {error:?}"),
        }
    });
    wait_for_pending_selection(&core).await;
    core.catalog
        .set_lifecycle(1, WorkerLifecycle::Draining, Vec::new());
    core.free_reservation("res-a").await.expect("free res-a");

    let (worker_id, endpoint, leased) = tokio::time::timeout(Duration::from_secs(2), queued)
        .await
        .expect("queued selection timed out")
        .expect("task panicked");
    assert_eq!(worker_id, 1);
    assert_eq!(endpoint, None);
    assert!(leased, "lease admission returns the booking's lease");
    // The lease was dropped with the selection, so the booking is gone.
    wait_until("booking release", || !entry.scheduler.has_request("queued")).await;
}

/// A `Lease` admission hands the booking to the host: no index row, no
/// booked hashes, and nothing recorded into the approximate primary that
/// `Book` feeds.
#[tokio::test]
async fn lease_admission_installs_no_index_row_and_records_nothing() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let key = default_key();
    let entry = core.entry(&key).expect("entry");

    let req = reserve_request("leased");
    let run = core
        .run_selection(lease_operation(req.prompt.view(), "leased", true))
        .await;
    let Ok(SelectionOutcome::Selected(selected)) = run.result else {
        panic!("lease selection failed");
    };
    assert!(entry.scheduler.has_request("leased"));
    assert!(
        core.reservation_index.read().is_empty(),
        "a lease admission installs no index row"
    );
    assert!(
        selected.booking.is_some(),
        "the booking's handle goes to the host"
    );
    assert!(selected.sequence_hashes.is_none());
    assert!(selected.routing_hashes.is_none());
    // Nothing was recorded for the leased prompt: once the indexer has
    // applied everything enqueued so far, a query restricted to the booked
    // worker sees no cached blocks.
    entry.indexer.flush().await;
    let mut probe = select_request();
    probe.allowed_worker_ids = Some(HashSet::from([1]));
    let probe = core.select(probe).await.expect("select");
    assert_eq!(
        probe.overlap.longest_matched, 0,
        "a lease admission records nothing into the indexer"
    );

    drop(selected);
    wait_until("booking release", || !entry.scheduler.has_request("leased")).await;
}

#[tokio::test]
async fn dropped_selection_future_frees_its_booking() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let key = default_key();
    let entry = core.entry(&key).expect("entry");

    // Drive the selection by hand so the actor's response is delivered but
    // never consumed: poll until the booking exists, then drop the future.
    let mut selection = Box::pin(core.select_and_reserve(reserve_request("dropped")));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    wait_until("scheduler booking", || {
        if entry.scheduler.has_request("dropped") {
            return true;
        }
        assert!(
            selection.as_mut().poll(&mut context).is_pending(),
            "selection completed before the booking was observed"
        );
        false
    })
    .await;
    assert!(
        core.reservation_index
            .read()
            .get("dropped")
            .is_some_and(|reservation| reservation.booking.is_none()),
        "the id is claimed while its booking is in flight"
    );
    drop(selection);

    wait_until("booking release", || {
        !entry.scheduler.has_request("dropped")
    })
    .await;
    assert!(core.reservation_index.read().get("dropped").is_none());
}

/// The routing-decision record is an await between the booking (and the
/// session commit) and the index install; the armed lease, not lock
/// discipline, covers it. Dropping the run there frees the booking and
/// releases the id claim.
#[tokio::test]
async fn dropped_book_selection_during_routing_record_frees_booking_and_claim() {
    use crate::indexer::TieredMatchDetails;
    use crate::services::indexer::backend::RemotePrimary;
    struct PausedRecord {
        entered: AtomicBool,
        release: tokio::sync::Notify,
    }
    #[async_trait::async_trait]
    impl RemotePrimary for PausedRecord {
        async fn find_matches_by_tier(
            &self,
            _: Vec<LocalBlockHash>,
            _: bool,
        ) -> anyhow::Result<TieredMatchDetails> {
            Ok(TieredMatchDetails::default())
        }
        async fn record_routing_decision(
            &self,
            _: WorkerWithDpRank,
            _: RoutingDecisionHashes,
        ) -> anyhow::Result<()> {
            self.entered.store(true, Ordering::Release);
            self.release.notified().await;
            Ok(())
        }
        fn use_kv_events(&self) -> bool {
            false
        }
    }
    struct PausedRecordIngress(Arc<PausedRecord>);
    #[async_trait::async_trait]
    impl KvEventIngress for PausedRecordIngress {
        fn open(&self, _: &WorkerRegistry, _: &RoutingPartitionId, _: u32) -> Indexer {
            Indexer::Remote {
                primary: self.0.clone(),
                approx: None,
                primary_records_routing_decisions: true,
            }
        }
    }
    let record = Arc::new(PausedRecord {
        entered: AtomicBool::new(false),
        release: tokio::sync::Notify::new(),
    });
    let core = core_with(
        test_config(false),
        SelectionHost {
            cache: HostCache {
                index: KvIndexSource::Owned(Arc::new(PausedRecordIngress(record.clone()))),
                shared: None,
            },
            ..SelectionHost::default()
        },
        None,
        WorkerType::Aggregated,
        None,
        Some(
            SessionAffinityConfig::new(Duration::from_secs(10))
                .with_mode(SessionAffinityMode::Hard),
        ),
    );
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let key = default_key();
    let entry = core.entry(&key).expect("entry");

    // Poll by hand until the booking exists and the record is awaited.
    let mut selection = Box::pin(core.select_and_reserve(session_reservation("recording", "s")));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    wait_until("routing record entered", || {
        if record.entered.load(Ordering::Acquire) {
            return true;
        }
        assert!(
            selection.as_mut().poll(&mut context).is_pending(),
            "selection completed before the routing record was entered"
        );
        false
    })
    .await;
    assert!(entry.scheduler.has_request("recording"));
    assert!(
        core.reservation_index
            .read()
            .get("recording")
            .is_some_and(|reservation| reservation.booking.is_none()),
        "the id is still a claim while the record is in flight"
    );
    assert_eq!(
        bound_worker(&core, "s"),
        Some(1),
        "the session commits before the record"
    );
    drop(selection);

    wait_until("booking release", || {
        !entry.scheduler.has_request("recording")
    })
    .await;
    assert!(
        core.reservation_index.read().get("recording").is_none(),
        "the claim is released with the booking"
    );
}

#[tokio::test]
async fn same_selection_id_in_two_partitions_is_a_conflict() {
    let core = local_core(test_config(false));
    for (worker_id, routing_group) in [(1, "group-a"), (2, "group-b")] {
        let mut request = worker(worker_id);
        request.routing_group = routing_group.to_string();
        core.upsert_worker(request).await.expect("worker upsert");
    }
    let mut first = reserve_request("shared");
    first.routing_group = "group-a".to_string();
    core.select_and_reserve(first).await.expect("first booking");

    let mut second = reserve_request("shared");
    second.routing_group = "group-b".to_string();
    let err = core
        .select_and_reserve(second)
        .await
        .expect_err("a live id cannot be booked again");
    assert!(matches!(err, SelectionError::Conflict(_)), "{err:?}");

    let (entry, _) = core
        .indexed_booking("shared")
        .expect("first booking indexed");
    assert_eq!(entry.key.routing_group, "group-a");
    assert!(entry.scheduler.has_request("shared"));
    assert!(
        !core
            .entry(&RoutingPartitionId::new("model", "group-b"))
            .expect("entry")
            .scheduler
            .has_request("shared")
    );
}

#[tokio::test]
async fn explicit_reservation_of_a_live_id_is_a_conflict() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.select_and_reserve(reserve_request("taken"))
        .await
        .expect("booking");
    let err = core
        .create_reservation(ReservationRequest {
            worker_id: Some(1),
            prompt: prompt(),
            ..replay_reservation("taken")
        })
        .await
        .expect_err("explicit booking of a live id");
    assert!(matches!(err, SelectionError::Conflict(_)), "{err:?}");
}

#[test]
fn index_observer_replaces_stale_and_claimed_rows_of_its_partition() {
    use crate::scheduling::AttemptId;
    let index = Arc::new(RwLock::new(HashMap::new()));
    let partition = default_key();
    let observer = ReservationIndexObserver {
        index: Arc::clone(&index),
        partition: partition.clone(),
        host: None,
    };
    let booking = |attempt: u64| SchedulerBookingDescriptor {
        request_id: "shared".to_string(),
        worker: WorkerWithDpRank::new(1, 0),
        attempt_id: AttemptId::new(attempt),
    };
    let row = |partition: &RoutingPartitionId, booking| Reservation {
        partition: partition.clone(),
        booking,
        _affinity_lease: None,
    };

    // A stale row (its booking expired before the sweep ran) yields to the mirror.
    index
        .write()
        .insert("shared".to_string(), row(&partition, Some(booking(1))));
    observer.admitted(booking(2));
    assert_eq!(index.read()["shared"].booking, Some(booking(2)));

    // A completion for the replaced booking leaves the live mirror alone.
    observer.completed(&booking(1));
    assert!(index.read().contains_key("shared"));
    observer.completed(&booking(2));
    assert!(!index.read().contains_key("shared"));

    // A claim whose local booking is about to fail also yields, and dropping
    // that claim keeps the mirror.
    let claim = ReservationClaim {
        index: &index,
        selection_id: "shared".to_string(),
        armed: true,
    };
    index
        .write()
        .insert("shared".to_string(), row(&partition, None));
    observer.admitted(booking(3));
    drop(claim);
    assert_eq!(index.read()["shared"].booking, Some(booking(3)));

    // Another partition's row is never replaced.
    let other = RoutingPartitionId::new("model", "other");
    index
        .write()
        .insert("shared".to_string(), row(&other, Some(booking(4))));
    observer.admitted(booking(5));
    assert_eq!(index.read()["shared"].booking, Some(booking(4)));
}

#[tokio::test]
async fn free_of_an_in_flight_reservation_is_not_found() {
    let core = saturated_core();
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.select_and_reserve(reserve_request("res-a"))
        .await
        .expect("initial reservation");
    let queued_core = core.clone();
    let queued = tokio::spawn(async move {
        queued_core
            .select_and_reserve(reserve_request("queued"))
            .await
    });
    wait_for_pending_selection(&core).await;

    // A free racing the in-flight booking neither frees nor evicts it.
    let err = core
        .free_reservation("queued")
        .await
        .expect_err("in-flight id is not a reservation yet");
    assert!(matches!(err, SelectionError::NotFound(_)), "{err:?}");
    assert!(core.reservation_index.read().contains_key("queued"));

    core.free_reservation("res-a").await.expect("free res-a");
    tokio::time::timeout(Duration::from_secs(2), queued)
        .await
        .expect("queued selection timed out")
        .expect("task panicked")
        .expect("queued selection books");
    core.free_reservation("queued").await.expect("free queued");
    assert!(core.reservation_index.read().is_empty());
}

#[tokio::test]
async fn prefill_complete_is_idempotent_for_a_live_booking() {
    let mut config = test_config(false);
    config.router_track_prefill_tokens = true;
    let core = local_core(config);
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.select_and_reserve(reserve_request("live"))
        .await
        .expect("booking");
    core.prefill_complete("live").await.expect("first mark");
    core.prefill_complete("live")
        .await
        .expect("a repeated mark on a live booking is not an error");
    let (entry, _) = core.indexed_booking("live").expect("still indexed");
    assert!(entry.scheduler.has_request("live"));
    core.free_reservation("live").await.expect("free");
}

#[tokio::test]
async fn mirrored_replica_bookings_are_indexed_until_freed() {
    let (outbound_tx, _outbound_rx) = mpsc::channel(16);
    let (inbound_tx, inbound_rx) = mpsc::channel(16);
    let channels = parking_lot::Mutex::new(Some(HostReplicaChannels {
        outbound: Some(outbound_tx),
        inbound_tx: inbound_tx.clone(),
        inbound_rx,
        process_id: 7,
    }));
    let core = core_with_host(SelectionHost {
        replication: HostReplication {
            channels: Some(Arc::new(move |_| channels.lock().take())),
            ..HostReplication::default()
        },
        ..SelectionHost::default()
    });
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    // Replica events for a worker the scheduler has not registered yet are
    // dropped; the first local booking registers it.
    core.select_and_reserve(reserve_request("warm"))
        .await
        .expect("warm booking");
    core.free_reservation("warm").await.expect("free warm");
    let entry = core.entry(&default_key()).expect("entry");
    let peer_event = |request_id: &str, data| ActiveSequenceEvent {
        request_id: request_id.to_string(),
        worker: WorkerWithDpRank::new(1, 0),
        data,
        router_id: 99,
        lora_name: None,
    };
    let add = |request_id: &str| {
        peer_event(
            request_id,
            ActiveSequenceEventData::AddRequest {
                token_sequence: Some(vec![1, 2]),
                track_prefill_tokens: false,
                expected_output_tokens: None,
                prefill_load_hint: None,
            },
        )
    };

    inbound_tx.send(add("peer-a")).await.expect("send");
    inbound_tx.send(add("peer-b")).await.expect("send");
    wait_until("mirrored bookings indexed", || {
        core.indexed_booking("peer-a").is_some() && core.indexed_booking("peer-b").is_some()
    })
    .await;

    // A lifecycle call on a mirrored booking resolves through the index.
    core.free_reservation("peer-a")
        .await
        .expect("free mirrored");
    assert!(!entry.scheduler.has_request("peer-a"));
    assert!(core.indexed_booking("peer-a").is_none());

    // The peer freeing its own booking removes the mirror.
    inbound_tx
        .send(peer_event("peer-b", ActiveSequenceEventData::Free))
        .await
        .expect("send");
    wait_until("mirror removed", || {
        core.indexed_booking("peer-b").is_none()
    })
    .await;
    assert!(!entry.scheduler.has_request("peer-b"));
}

#[tokio::test]
async fn lifecycle_operations_resolve_through_the_index() {
    let mut config = test_config(false);
    config.router_track_prefill_tokens = true;
    let core = local_core(config);

    for (worker_id, routing_group) in [(1, "group-a"), (2, "group-b")] {
        let mut request = worker(worker_id);
        request.routing_group = routing_group.to_string();
        core.upsert_worker(request).await.expect("worker upsert");
    }

    let entries = core.initialized_entries();
    assert_eq!(entries.len(), 2);
    let target = &entries[1];
    let target_group = target.key.routing_group.clone();
    let target_worker = *target
        .workers_tx
        .borrow()
        .keys()
        .next()
        .expect("target worker");

    let mut request = reserve_request("later-entry-reservation");
    request.routing_group = target_group.clone();
    core.select_and_reserve(request)
        .await
        .expect("reserve in later entry");

    let load = || {
        core.loads(Some("model"), Some(&target_group))[0]
            .loads
            .iter()
            .find(|load| load.worker_id == target_worker)
            .expect("target load")
            .potential_prefill_tokens
    };
    assert_eq!(load(), 4);

    core.prefill_complete("later-entry-reservation")
        .await
        .expect("complete prefill in later entry");
    assert_eq!(load(), 0);

    core.free_reservation("later-entry-reservation")
        .await
        .expect("free reservation in later entry");
    assert!(matches!(
        core.add_output_block("later-entry-reservation", None),
        Err(SelectionError::NotFound(_))
    ));
}

#[tokio::test]
async fn advisory_select_reports_worker_load_and_busy_evaluation() {
    let mut config = test_config(false);
    config.conditional_disagg_prefill_busy_threshold = Some(0.5);
    config.conditional_disagg_decode_busy_threshold = Some(0.0);
    let core = local_core(config);
    let mut request = worker(1);
    request.total_kv_blocks = Some(1000);
    core.upsert_worker(request).await.expect("worker upsert");

    // Admitted (queued) select: decode evaluation comes from the catalog's
    // total_kv_blocks; no load snapshot is taken.
    let response = core.select(select_request()).await.expect("select");
    assert!(response.potential_decode_blocks > 0);
    assert_eq!(
        response.decode_busy,
        Some(true),
        "threshold 0.0 is always exceeded"
    );
    assert!(response.worker_load.is_none());

    // Advisory select: same decode evaluation plus the projected load.
    let mut request = select_request();
    request.advisory = true;
    let response = core.select(request).await.expect("advisory select");
    assert!(response.potential_decode_blocks > 0);
    assert_eq!(response.decode_busy, Some(true));
    let load = response.worker_load.expect("advisory load");
    assert_eq!(load.total_kv_blocks, Some(1000));
    assert_eq!(load.prefill_token_capacity, 1024);
    assert_eq!(load.active_prefill_tokens, 0);
    assert_eq!(load.prefill_busy, Some(false));

    // Advisory selection does not book.
    assert!(core.reservation_index.read().is_empty());
    assert_eq!(
        core.loads(Some("model"), Some("default"))[0].loads[0].potential_prefill_tokens,
        0
    );
}

#[tokio::test]
async fn busy_evaluation_is_absent_without_thresholds_or_capacity() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    let mut request = select_request();
    request.advisory = true;
    let response = core.select(request).await.expect("advisory select");
    assert_eq!(response.decode_busy, None);
    let load = response.worker_load.expect("advisory load");
    assert_eq!(load.total_kv_blocks, None);
    assert_eq!(load.prefill_busy, None);
}

#[tokio::test]
async fn reservation_index_tracks_bookings_until_freed() {
    let core = local_core(test_config(false));
    for (worker_id, routing_group) in [(1, "group-a"), (2, "group-b")] {
        let mut request = worker(worker_id);
        request.routing_group = routing_group.to_string();
        core.upsert_worker(request).await.expect("worker upsert");
    }
    let key_b = RoutingPartitionId::new("model", "group-b");

    // select_and_reserve records the booking's partition.
    let mut request = reserve_request("booked");
    request.routing_group = "group-b".to_string();
    core.select_and_reserve(request).await.expect("reserve");
    assert_eq!(
        core.reservation_index
            .read()
            .get("booked")
            .map(|r| &r.partition),
        Some(&key_b)
    );
    assert_eq!(
        core.indexed_booking("booked")
            .expect("indexed booking")
            .0
            .key,
        key_b
    );

    // The explicit reservation path records too.
    let mut request = select_request();
    request.routing_group = "group-b".to_string();
    request.selection_id = Some("cached".to_string());
    core.select(request).await.expect("select");
    assert!(core.reservation_index.read().get("cached").is_none());
    core.create_reservation(ReservationRequest {
        routing_group: "group-b".to_string(),
        ..replay_reservation("cached")
    })
    .await
    .expect("cached reservation");
    assert_eq!(
        core.reservation_index
            .read()
            .get("cached")
            .map(|r| &r.partition),
        Some(&key_b)
    );

    // Lifecycle calls still resolve, and free drops the index entry.
    core.prefill_complete("booked")
        .await
        .expect("prefill complete");
    core.free_reservation("booked").await.expect("free");
    assert!(core.reservation_index.read().get("booked").is_none());
    core.free_reservation("cached").await.expect("free");
    assert!(core.reservation_index.read().is_empty());

    // Unknown ids fall back to the full scan and stay unindexed.
    assert!(matches!(
        core.prefill_complete("never-booked").await,
        Err(SelectionError::NotFound(_))
    ));
    assert!(core.reservation_index.read().is_empty());
}

#[tokio::test]
async fn reservation_index_sweep_drops_bookings_released_out_of_band() {
    let core = local_core(test_config(false));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.select_and_reserve(reserve_request("live"))
        .await
        .expect("reserve live");
    core.select_and_reserve(reserve_request("stale"))
        .await
        .expect("reserve stale");
    assert_eq!(core.reservation_index.read().len(), 2);
    assert_eq!(
        sweep_reservation_index(&core.entries, &core.reservation_index),
        0
    );

    // Release directly through the scheduler, as force-expiry would.
    let entry = core.entry(&default_key()).expect("entry");
    entry.scheduler.free("stale").await.expect("scheduler free");

    assert_eq!(
        sweep_reservation_index(&core.entries, &core.reservation_index),
        1
    );
    let index = core.reservation_index.read();
    assert_eq!(index.len(), 1);
    assert!(index.contains_key("live"));
}

#[tokio::test(flavor = "current_thread")]
async fn lifecycle_lookup_does_not_nest_reservation_index_inside_entries() {
    // Three parties: a sweep holding `entries` and wanting `reservation_index`,
    // a lifecycle call, and a partition creation queued on `entries.write()`.
    let core = Arc::new(local_core(test_config(false)));
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.select_and_reserve(reserve_request("live"))
        .await
        .expect("reserve live");
    let deadline = Instant::now() + Duration::from_secs(5);

    let sweep_entries = core.entries.read();
    let writer = {
        let core = Arc::clone(&core);
        std::thread::spawn(move || {
            core.entries
                .write()
                .entry(RoutingPartitionId::new("other", "default"))
                .or_insert_with(|| Arc::new(OnceCell::new()));
        })
    };
    while core.entries.try_read().is_some() {
        assert!(Instant::now() < deadline, "partition writer never queued");
        std::thread::yield_now();
    }
    let lifecycle_started = Arc::new(AtomicBool::new(false));
    let lifecycle = {
        let core = Arc::clone(&core);
        let started = Arc::clone(&lifecycle_started);
        std::thread::spawn(move || {
            started.store(true, Ordering::Release);
            core.indexed_booking("live").is_some()
        })
    };
    while !lifecycle_started.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "lifecycle thread never started");
        std::thread::yield_now();
    }
    sleep(Duration::from_millis(50));

    // With the index guard held across `entries.read()` this never acquires.
    drop(
        core.reservation_index
            .try_write_for(Duration::from_secs(2))
            .expect("reservation index must not be held by a blocked lifecycle call"),
    );
    drop(sweep_entries);
    writer.join().expect("partition writer");
    assert!(lifecycle.join().expect("lifecycle lookup"));
}

#[tokio::test(flavor = "current_thread")]
async fn queued_selection_returns_refreshed_overlap_snapshot() {
    let core = saturated_core();

    for worker_id in [1, 2] {
        let mut request = worker(worker_id);
        request.max_num_batched_tokens = Some(8);
        core.upsert_worker(request).await.expect("worker upsert");
    }
    let key = default_key();
    let entry = core.entry(&key).expect("entry");
    entry
        .indexer
        .apply_event_routed(store_event(1, 0, 1, &[], &[11], StorageTier::Device))
        .await
        .unwrap();
    entry.indexer.dump_events().await.expect("flush indexer");

    for worker_id in [1, 2] {
        core.create_reservation(ReservationRequest {
            worker_id: Some(worker_id),
            dp_rank: Some(0),
            prompt: PromptRequest {
                sequence_hashes: Some(vec![1, 2]),
                isl_tokens: Some(8),
                ..PromptRequest::default()
            },
            effective_prefill_tokens: Some(8),
            ..replay_reservation(&format!("occupy-{worker_id}"))
        })
        .await
        .expect("occupy worker");
    }

    let queued_core = Arc::clone(&core);
    let queued = tokio::spawn(async move {
        queued_core
            .select_and_reserve(SelectAndReserveRequest {
                prompt: PromptRequest {
                    block_hashes: Some(vec![11, 12]),
                    sequence_hashes: Some(vec![101, 102]),
                    isl_tokens: Some(8),
                    ..PromptRequest::default()
                },
                ..reserve_request("refresh-selection")
            })
            .await
    });
    wait_for_pending_selection(&core).await;

    entry
        .indexer
        .apply_event_routed(store_event(2, 0, 1, &[], &[11, 12], StorageTier::Device))
        .await
        .unwrap();
    entry.indexer.dump_events().await.expect("flush indexer");
    // Freeze time only after async setup so background timers cannot expire the fixture.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(11)).await;
    core.free_reservation("occupy-2")
        .await
        .expect("release worker 2");

    let response = queued.await.expect("selection task").expect("selection");
    assert_eq!(response.worker_id, 2);
    assert_eq!(response.effective_prefill_tokens, 0);
    assert_eq!(response.overlap.gpu, 8);
    assert_eq!(response.overlap.cpu, 8);
    assert_eq!(response.overlap.disk, 8);
    assert_eq!(response.overlap.dp, HashMap::from([("0".to_string(), 8)]));
}

fn core_with_session_affinity_mode(mode: SessionAffinityMode) -> SelectionCore {
    core_with(
        test_config(false),
        SelectionHost::default(),
        None,
        WorkerType::Aggregated,
        None,
        Some(SessionAffinityConfig::new(Duration::from_secs(10)).with_mode(mode)),
    )
}

fn core_with_session_affinity() -> SelectionCore {
    core_with_session_affinity_mode(SessionAffinityMode::Hard)
}

/// The worker `session_id` is bound to; panics if the partition has no
/// affinity table, so `None` means unbound rather than unconfigured.
fn bound_worker(core: &SelectionCore, session_id: &str) -> Option<WorkerId> {
    let entry = core.entry(&default_key()).expect("default partition");
    let table = entry.affinity.get().expect("affinity table configured");
    table
        .query_target(session_id, None)
        .expect("query")
        .map(|target| target.worker_id)
}

fn session_reservation(selection_id: &str, session_id: &str) -> SelectAndReserveRequest {
    let mut request = reserve_request(selection_id);
    request.session_id = Some(session_id.to_string());
    request
}

/// Two workers, session `s` bound by booking `r1` (already freed).
async fn bound_session(mode: SessionAffinityMode) -> (SelectionCore, SelectResponse) {
    let core = core_with_session_affinity_mode(mode);
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.upsert_worker(worker(2)).await.expect("worker upsert");
    let first = core
        .select_and_reserve(session_reservation("r1", "s"))
        .await
        .expect("first booking");
    core.free_reservation("r1").await.expect("free");
    (core, first)
}

#[tokio::test]
async fn departed_session_worker_reinitializes_the_session() {
    let (core, first) = bound_session(SessionAffinityMode::Hard).await;
    core.delete_worker(first.worker_id).await.expect("delete");

    let second = core
        .select_and_reserve(session_reservation("r2", "s"))
        .await
        .expect("session must move off a departed worker");
    assert_ne!(second.worker_id, first.worker_id);
    assert_eq!(bound_worker(&core, "s"), Some(second.worker_id));
}

#[tokio::test]
async fn session_worker_departing_after_the_hold_reinitializes_the_session() {
    let (core, first) = bound_session(SessionAffinityMode::Hard).await;
    let key = default_key();

    // One poll takes the hold (the bound worker still passes the check)
    // and hands the request to the scheduler actor, which has not run yet.
    let mut second = Box::pin(core.select_and_reserve(session_reservation("r2", "s")));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(second.as_mut().poll(&mut context).is_pending());
    core.catalog
        .set_lifecycle(first.worker_id, WorkerLifecycle::Draining, Vec::new());
    core.publish_scheduler_config(&key);

    let second = second
        .await
        .expect("a departure after the hold is not a client fault");
    assert_ne!(second.worker_id, first.worker_id);
    assert_eq!(bound_worker(&core, "s"), Some(second.worker_id));
}

#[tokio::test]
async fn concurrent_holds_on_a_departed_worker_both_land_on_the_replacement() {
    let table = SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs(60)))
        .expect("affinity table");
    let departed = WorkerAffinityTarget::new(1, Some(0));
    let replacement = WorkerAffinityTarget::new(2, Some(0));
    let Hold::Initialize(init) = table.acquire("s", None).await.expect("acquire") else {
        panic!("fresh session must initialize");
    };
    drop(
        table
            .commit(Hold::Initialize(init), departed)
            .expect("bind"),
    );
    let (
        Hold::Bound {
            lease: mut first, ..
        },
        Hold::Bound {
            lease: mut second, ..
        },
    ) = (
        table.acquire("s", None).await.expect("first hold"),
        table.acquire("s", None).await.expect("second hold"),
    )
    else {
        panic!("both requests hold the departed binding");
    };

    // Both requests notice the departure, as `hold_session` does: the first
    // invalidation drops the binding, the second is a no-op release.
    first.invalidate();
    second.invalidate();
    let leases = [
        table.commit(
            table.acquire("s", None).await.expect("re-acquire"),
            replacement,
        ),
        table.commit(
            table.acquire("s", None).await.expect("re-acquire"),
            replacement,
        ),
    ];
    for lease in leases {
        drop(lease.expect("both requests bind to the replacement"));
    }
    assert_eq!(
        table.query_target("s", None).expect("query"),
        Some(replacement)
    );
}

/// Leases on session `s` in the default partition's table.
fn lease_count(core: &SelectionCore, session_id: &str) -> Option<usize> {
    let entry = core.entry(&default_key()).expect("default partition");
    let table = entry.affinity.get().expect("affinity table configured");
    table.lease_count(session_id)
}

/// r2 holds the departed binding when r3 re-initializes the session; r2's
/// commit then finds r3's initialization in progress and must join it, not
/// route without a lease, so the replacement binding cannot idle out while
/// r2 is still active.
#[tokio::test]
async fn failover_commit_behind_an_initializing_hold_keeps_a_lease() {
    let (core, first) = bound_session(SessionAffinityMode::Hard).await;
    let key = default_key();
    let replacement = if first.worker_id == 1 { 2 } else { 1 };
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());

    // r2 takes `Bound{w1}` and is queued in the scheduler actor.
    let mut r2 = Box::pin(core.select_and_reserve(session_reservation("r2", "s")));
    assert!(r2.as_mut().poll(&mut context).is_pending());
    core.catalog
        .set_lifecycle(first.worker_id, WorkerLifecycle::Draining, Vec::new());
    core.publish_scheduler_config(&key);

    // r3 sees w1 unschedulable, invalidates the binding, and holds
    // `Initialize` for as long as it sits in the actor unpolled.
    let mut r3 = Box::pin(core.select_and_reserve(session_reservation("r3", "s")));
    assert!(r3.as_mut().poll(&mut context).is_pending());
    assert_eq!(bound_worker(&core, "s"), None, "r3 is initializing");
    assert_eq!(lease_count(&core, "s"), Some(0));

    // r2's `Hard` commit fails, the worker departed, and `try_acquire` finds
    // r3's initialization: r2 joins it.
    let r2 = r2
        .await
        .expect("a departure after the hold is not a client fault");
    assert_eq!(r2.worker_id, replacement);
    assert_eq!(bound_worker(&core, "s"), None, "r3 has not committed yet");
    assert_eq!(
        lease_count(&core, "s"),
        Some(1),
        "r2 must hold a pending lease on r3's initialization"
    );

    let r3 = r3.await.expect("r3 binds the replacement");
    assert_eq!(r3.worker_id, replacement);
    assert_eq!(bound_worker(&core, "s"), Some(replacement));
    assert_eq!(lease_count(&core, "s"), Some(2));

    core.free_reservation("r3").await.expect("free r3");
    wait_until("r3 lease release", || lease_count(&core, "s") == Some(1)).await;
    assert_eq!(bound_worker(&core, "s"), Some(replacement));

    core.free_reservation("r2").await.expect("free r2");
    wait_until("r2 lease release", || lease_count(&core, "s") == Some(0)).await;
    // Idle, not gone: the binding lasts until the TTL.
    assert_eq!(bound_worker(&core, "s"), Some(replacement));
    let entry = core.entry(&key).expect("default partition");
    entry.affinity.get().expect("table").expire_for_test("s");
    assert_eq!(bound_worker(&core, "s"), None);
}

#[tokio::test]
async fn two_phase_reservation_binds_the_session() {
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.upsert_worker(worker(2)).await.expect("worker upsert");
    let mut request = select_request();
    request.selection_id = Some("pending".to_string());
    request.session_id = Some("s".to_string());
    let selected = core.select(request).await.expect("select");
    assert_eq!(bound_worker(&core, "s"), None, "select alone binds nothing");

    core.create_reservation(replay_reservation("pending"))
        .await
        .expect("replay booking");
    assert_eq!(bound_worker(&core, "s"), Some(selected.worker_id));
    core.free_reservation("pending").await.expect("free");

    // The binding steers the session's next selection.
    let mut request = select_request();
    request.session_id = Some("s".to_string());
    assert_eq!(
        core.select(request).await.expect("select").worker_id,
        selected.worker_id
    );
}

#[tokio::test]
async fn two_phase_replay_rejects_a_worker_the_session_left() {
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.upsert_worker(worker(2)).await.expect("worker upsert");
    let mut request = select_request();
    request.selection_id = Some("pending".to_string());
    request.session_id = Some("s".to_string());
    let cached = core.select(request).await.expect("select");
    // Another request binds the session elsewhere before the replay.
    let other = if cached.worker_id == 1 { 2 } else { 1 };
    let mut request = session_reservation("r1", "s");
    request.allowed_worker_ids = Some(HashSet::from([other]));
    core.select_and_reserve(request)
        .await
        .expect("bind elsewhere");
    core.free_reservation("r1").await.expect("free");

    let err = core
        .create_reservation(replay_reservation("pending"))
        .await
        .expect_err("hard affinity rejects the stale cached worker");
    assert!(matches!(err, SelectionError::BadRequest(_)), "{err:?}");
    let entry = core.entry(&default_key()).expect("entry");
    wait_until("rejected booking release", || {
        !entry.scheduler.has_request("pending")
    })
    .await;
    assert!(core.reservation_index.read().is_empty());
}

#[tokio::test]
async fn hard_mode_rejects_dispatch_away_from_a_live_binding() {
    let (core, first) = bound_session(SessionAffinityMode::Hard).await;
    let other = if first.worker_id == 1 { 2 } else { 1 };

    // Steering cannot reach the bound worker, so selection lands elsewhere.
    let mut request = session_reservation("r2", "s");
    request.allowed_worker_ids = Some(HashSet::from([other]));
    let err = core
        .select_and_reserve(request)
        .await
        .expect_err("hard affinity rejects a dispatch away from the binding");
    assert!(matches!(err, SelectionError::BadRequest(_)), "{err:?}");
    let entry = core.entry(&default_key()).expect("entry");
    wait_until("rejected booking release", || {
        !entry.scheduler.has_request("r2")
    })
    .await;
    assert!(core.reservation_index.read().is_empty());
    assert_eq!(bound_worker(&core, "s"), None, "stale binding is dropped");

    // The session's next request re-initializes.
    let mut request = session_reservation("r3", "s");
    request.allowed_worker_ids = Some(HashSet::from([other]));
    let third = core.select_and_reserve(request).await.expect("rebind");
    assert_eq!(third.worker_id, other);
    assert_eq!(bound_worker(&core, "s"), Some(other));
}

#[tokio::test]
async fn soft_mode_follows_the_dispatch() {
    let (core, first) = bound_session(SessionAffinityMode::Soft).await;
    let other = if first.worker_id == 1 { 2 } else { 1 };

    let mut request = session_reservation("r2", "s");
    request.allowed_worker_ids = Some(HashSet::from([other]));
    let second = core.select_and_reserve(request).await.expect("soft rebind");
    assert_eq!(second.worker_id, other);
    assert_eq!(bound_worker(&core, "s"), Some(other));
}

#[tokio::test]
async fn replicated_binding_steers_a_new_session_and_frees_with_the_booking() {
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.expect("worker upsert");
    core.upsert_worker(worker(2)).await.expect("worker upsert");

    core.dispatch_affinity_event(AffinityBindingEvent {
        partition: default_key(),
        session_id: "chat-b".to_string(),
        worker_id: 2,
        dp_rank: Some(0),
        sequence: 1,
        writer_id: 99,
    });
    let response = core
        .select_and_reserve(session_reservation("r2", "chat-b"))
        .await
        .expect("booking");
    assert_eq!(response.worker_id, 2);
    assert!(
        core.reservation_index
            .read()
            .get("r2")
            .unwrap()
            ._affinity_lease
            .is_some()
    );
    core.free_reservation("r2").await.expect("free");
    assert!(core.reservation_index.read().is_empty());
}

#[tokio::test]
async fn expired_booking_releases_affinity_lease() {
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.unwrap();
    core.select_and_reserve(session_reservation("abandoned", "session"))
        .await
        .unwrap();
    let key = default_key();
    core.entry(&key)
        .unwrap()
        .scheduler
        .free("abandoned")
        .await
        .unwrap();
    assert_eq!(
        sweep_reservation_index(&core.entries, &core.reservation_index),
        1
    );
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(bound_worker(&core, "session"), None);
}

#[tokio::test]
async fn partition_sessions_keep_independent_bindings() {
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.unwrap();
    let first = core
        .select_and_reserve(session_reservation("first", "shared-session"))
        .await
        .unwrap();
    core.upsert_worker(worker(2)).await.unwrap();
    let mut other = worker(3);
    other.routing_group = "other".to_string();
    core.upsert_worker(other).await.unwrap();
    let mut request = session_reservation("other", "shared-session");
    request.routing_group = "other".to_string();
    assert_eq!(core.select_and_reserve(request).await.unwrap().worker_id, 3);
    assert_eq!(bound_worker(&core, "shared-session"), Some(first.worker_id));
    let again = core
        .select_and_reserve(session_reservation("again", "shared-session"))
        .await
        .unwrap();
    assert_eq!(again.worker_id, first.worker_id);
}

#[tokio::test]
async fn rejoined_worker_feeds_partition_index() {
    use crate::protocols::{BlockHashOptions, compute_block_hash_for_seq};
    for update_only in [false, true] {
        let core = local_core(test_config(true));
        let request = worker_with_kv_events(1);
        core.upsert_worker(request.clone()).await.unwrap();
        let key = default_key();
        let partition = core.partition(&key).unwrap();
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes: Vec<u64> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        let indexer = core
            .indexer_registry
            .get_indexer(&key)
            .unwrap()
            .indexer
            .clone();
        indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        indexer.dump_events().await.unwrap();
        if !update_only {
            core.delete_worker(1).await.unwrap();
        }
        let mut request = request;
        request.total_kv_blocks = Some(2048);
        core.upsert_worker(request).await.unwrap();
        let retained = partition.indexer().dump_events().await.unwrap();
        if update_only {
            assert!(
                !retained.is_empty(),
                "metadata update cleared cached blocks"
            );
        } else {
            assert!(retained.is_empty(), "removed worker retained cached blocks");
            let current = core
                .indexer_registry
                .get_indexer(&key)
                .unwrap()
                .indexer
                .clone();
            current
                .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
                .await
                .unwrap();
            current.dump_events().await.unwrap();
        }
        let mut request = select_request();
        request.prompt = PromptRequest {
            token_ids: Some(tokens),
            ..PromptRequest::default()
        };
        assert_eq!(
            core.select(request).await.unwrap().overlap.longest_matched,
            8,
            "update_only={update_only}"
        );
        assert_eq!(
            partition.indexer().dump_events().await.unwrap().len(),
            indexer.dump_events().await.unwrap().len()
        );
    }
}

#[tokio::test(start_paused = true)]
async fn host_lease_manager_owns_expiry() {
    use crate::scheduling::queue::SchedulerBookingDescriptor;
    struct HostLeases;
    impl ReplicaRequestLeaseObserver for HostLeases {
        fn admitted(&self, _: SchedulerBookingDescriptor) {}
        fn progressed(&self, _: &SchedulerBookingDescriptor) {}
        fn completed(&self, _: &SchedulerBookingDescriptor) {}
    }
    for host_owned in [false, true] {
        let core = core_with_host(SelectionHost {
            replication: HostReplication {
                request_leases: host_owned
                    .then(|| Arc::new(HostLeases) as Arc<dyn ReplicaRequestLeaseObserver>),
                ..HostReplication::default()
            },
            ..SelectionHost::default()
        });
        core.upsert_worker(worker(1)).await.unwrap();
        core.select_and_reserve(reserve_request("live"))
            .await
            .unwrap();
        tokio::task::yield_now().await;
        tokio::time::advance(active_request_expiry_duration() * 3).await;
        tokio::task::yield_now().await;
        let entry = core.entry(&default_key()).unwrap();
        assert_eq!(entry.scheduler.has_request("live"), host_owned);
        if host_owned {
            core.free_reservation("live").await.unwrap();
            assert!(!entry.scheduler.has_request("live"));
        }
    }
}

#[tokio::test]
async fn catalog_updates_commit_in_order_without_exposing_candidates() {
    struct PausedUpdate {
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
        detaches: AtomicUsize,
    }
    #[async_trait::async_trait]
    impl KvEventIngress for PausedUpdate {
        fn open(
            &self,
            registry: &WorkerRegistry,
            key: &RoutingPartitionId,
            block_size: u32,
        ) -> Indexer {
            registry.get_or_create_indexer(key.clone(), block_size)
        }
        async fn reconcile(
            &self,
            _: &WorkerRegistry,
            previous: Option<&WorkerCatalogRecord>,
            _: &WorkerCatalogRecord,
        ) -> Result<(), SelectionError> {
            if previous.is_some() {
                self.entered.notify_one();
                self.release.notified().await;
            }
            Ok(())
        }
        async fn detach(&self, _: &WorkerRegistry, _: &WorkerCatalogRecord) {
            self.detaches.fetch_add(1, Ordering::SeqCst);
        }
    }
    let ingress = Arc::new(PausedUpdate {
        entered: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
        detaches: AtomicUsize::new(0),
    });
    let mut core = core_with_host(SelectionHost {
        cache: HostCache {
            index: KvIndexSource::Owned(ingress.clone()),
            shared: None,
        },
        ..SelectionHost::default()
    });
    core.listens_for_kv_events = true;
    core.upsert_worker(worker(1)).await.unwrap();
    core.select_and_reserve(reserve_request("live"))
        .await
        .unwrap();
    let mut updated = worker(1);
    updated.max_num_batched_tokens = Some(2048);
    let update = core.upsert_worker(updated);
    tokio::pin!(update);
    tokio::select! {
        _ = &mut update => panic!("update should pause before committing"),
        _ = ingress.entered.notified() => {}
    }
    let committed = core.catalog.get(1).unwrap();
    assert_eq!(committed.lifecycle, WorkerLifecycle::Schedulable);
    assert_eq!(committed.max_num_batched_tokens, Some(1024));
    let entry = core.entry(&committed.key()).unwrap();
    assert!(entry.scheduler.has_request("live"));
    let deletion = core.delete_worker(1);
    tokio::pin!(deletion);
    assert!(
        std::future::Future::poll(
            deletion.as_mut(),
            &mut std::task::Context::from_waker(std::task::Waker::noop())
        )
        .is_pending()
    );
    ingress.release.notify_one();
    assert_eq!(update.await.unwrap().max_num_batched_tokens, Some(2048));
    // A capacity update must not temporarily withdraw this worker; only the
    // pending deletion detaches it.
    assert_eq!(
        ingress.detaches.load(Ordering::SeqCst),
        0,
        "capacity update detached the worker"
    );
    assert!(entry.scheduler.has_request("live"));
    assert_eq!(
        deletion.await.unwrap().lifecycle,
        WorkerLifecycle::Unschedulable
    );
    assert_eq!(
        ingress.detaches.load(Ordering::SeqCst),
        1,
        "deletion detaches the worker"
    );
    wait_until("booking release", || !entry.scheduler.has_request("live")).await;
}

#[tokio::test]
async fn affinity_configuration_rejects_invalid_or_conflicting_config() {
    use super::super::service::SelectionServiceBuilder;
    for ttl in [
        Duration::ZERO,
        Duration::from_secs(super::super::affinity::MAX_SESSION_AFFINITY_TTL_SECS + 1),
    ] {
        let result = SelectionServiceBuilder::new(
            test_config(false),
            WorkerType::Aggregated,
            Default::default(),
        )
        .session_affinity(ttl)
        .build()
        .await;
        assert!(result.is_err());
    }
    let core = core_with_session_affinity();
    core.upsert_worker(worker(1)).await.unwrap();
    let partition = core.partition(&default_key()).unwrap();
    assert!(matches!(
        partition.session_affinity(SessionAffinityConfig::new(Duration::from_secs(20))),
        Err(SelectionError::Conflict(_))
    ));
    assert!(matches!(
        partition.session_affinity(
            SessionAffinityConfig::new(Duration::from_secs(10))
                .with_mode(SessionAffinityMode::Soft)
        ),
        Err(SelectionError::Conflict(_))
    ));
}

fn hint_config(worker_type: &str, endpoints: &[(u32, &str)]) -> SelectionWorkerConfig {
    SelectionWorkerConfig {
        endpoint: "http://worker:8000".to_string(),
        data_parallel_start_rank: 0,
        data_parallel_size: endpoints.len().max(1) as u32,
        max_num_batched_tokens: None,
        total_kv_blocks: None,
        stable_routing_id: None,
        is_eagle: None,
        taints: HashSet::new(),
        topology_domains: HashMap::new(),
        kv_transfer_domain: None,
        kv_transfer_enforcement: None,
        kv_transfer_preferred_weight: None,
        router_hint_worker_type: Some(worker_type.to_string()),
        router_hint_source_control_endpoints: endpoints
            .iter()
            .map(|(rank, endpoint)| (*rank, endpoint.to_string()))
            .collect(),
        kv_event_source_mode: None,
    }
}

fn hint_candidates(
    hashes: &[u64],
    owners: Vec<(KvTransferCandidateSource, usize)>,
) -> KvTransferCandidates {
    KvTransferCandidates {
        block_hashes: hashes
            .iter()
            .map(|h| crate::protocols::ExternalSequenceBlockHash(*h))
            .collect(),
        owner_prefix_blocks: owners,
        routing_snapshot: None,
    }
}

#[test]
fn hint_source_endpoint_follows_the_source_dp_rank() {
    let configs = HashMap::from([(
        7,
        hint_config(
            "prefill",
            &[(0, "tcp://127.0.0.1:23280"), (1, "tcp://127.0.0.1:23281")],
        ),
    )]);
    let candidates = hint_candidates(&[101, 102], vec![(WorkerWithDpRank::new(7, 1).into(), 2)]);
    let hint =
        transfer_hint_for_selection(&configs, WorkerWithDpRank::new(7, 0), 0, Some(&candidates));
    assert_eq!(
        hint.map(|payload| payload.source_control_endpoint),
        Some("tcp://127.0.0.1:23281".to_string())
    );
}

#[test]
fn hint_source_must_share_the_target_worker_type() {
    let configs = HashMap::from([
        (7, hint_config("prefill", &[(0, "tcp://127.0.0.1:23280")])),
        (8, hint_config("prefill", &[(0, "tcp://127.0.0.1:23281")])),
        (9, hint_config("decode", &[(0, "tcp://127.0.0.1:23282")])),
        (10, hint_config("decode", &[(0, "tcp://127.0.0.1:23283")])),
    ]);
    let candidates = hint_candidates(
        &[101, 102, 103],
        vec![
            (WorkerWithDpRank::new(8, 0).into(), 2),
            (WorkerWithDpRank::new(9, 0).into(), 3),
        ],
    );
    // The longer decode prefix is skipped for a prefill target.
    let prefill =
        transfer_hint_for_selection(&configs, WorkerWithDpRank::new(7, 0), 0, Some(&candidates))
            .expect("prefill hint");
    assert_eq!(prefill.source_control_endpoint, "tcp://127.0.0.1:23281");
    assert_eq!(prefill.block_hashes.len(), 2);
    let decode =
        transfer_hint_for_selection(&configs, WorkerWithDpRank::new(10, 0), 0, Some(&candidates))
            .expect("decode hint");
    assert_eq!(decode.source_control_endpoint, "tcp://127.0.0.1:23282");
    assert_eq!(decode.block_hashes.len(), 3);
}

#[test]
fn hint_resolves_a_persistent_cache_owner_over_a_state_agent_worker() {
    use crate::identity::{
        CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };
    use crate::protocols::{
        ResidencyOwner, ResidencyProjection, ResidencyRoutingSnapshot, RouterHintSourceMetadata,
    };
    let mut stale = hint_config("prefill", &[(0, "tcp://stale-worker:23280")]);
    stale.kv_event_source_mode = Some("state_agent_v2".to_string());
    let configs = HashMap::from([(7, hint_config("prefill", &[])), (8, stale)]);
    let owner = CacheOwnerId::new(
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::Explicit),
            ),
            DcId::new(3),
        ),
        StableDpSlotId::new([4; 16], IdentitySource::Explicit),
    );
    let mut candidates = hint_candidates(
        &[101, 102],
        vec![
            (
                KvTransferCandidateSource::Worker(WorkerWithDpRank::new(8, 0)),
                2,
            ),
            (
                KvTransferCandidateSource::CacheOwner(
                    ResidencyOwner::cache_owner(owner).compact_key(),
                ),
                2,
            ),
        ],
    );
    candidates.routing_snapshot = Some(Arc::new(ResidencyRoutingSnapshot::new(
        ResidencyProjection::default(),
        [(
            owner,
            RouterHintSourceMetadata {
                source_control_endpoint: "tcp://persistent-owner:23280".to_string(),
                worker_type: "prefill".to_string(),
            },
            None,
        )],
    )));
    let hint =
        transfer_hint_for_selection(&configs, WorkerWithDpRank::new(7, 0), 0, Some(&candidates))
            .expect("hint");
    assert_eq!(hint.source_control_endpoint, "tcp://persistent-owner:23280");
}
