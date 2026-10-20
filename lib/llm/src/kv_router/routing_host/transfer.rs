// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advisory, bounded decode-source lookup after prefill admission. This path
//! owns no source bookings, affinity state, buffers, or transport operations.
//!
//! CPU and GPU actions use separate inventory, capability and cost gates. A GPU
//! hint is advisory: only its GPU-only backend endpoint may acquire source
//! leases and reserve target GPU blocks. The ordinary P-to-D handoff is separate.

use std::time::Instant;

use dynamo_kv_router::{
    config::KvRouterConfig,
    kv_hints::{KvHint, KvHintAction, KvSourceLocationsPayload},
    services::selection::{PromptView, TransferSourceMemoryKind, TransferSourceQuery},
};
use uuid::Uuid;

use super::*;

/// Treat an existing same-role donor as free. This deliberately overestimates
/// the no-decode-transfer alternative, so the new path only wins when its full
/// calibrated acquisition/transfer/onboarding bound still saves time.
fn baseline_prefix_blocks(selection: &WorkerSelection, block_size: u32) -> Option<(usize, usize)> {
    let local = selection
        .selected_raw_cached_tokens
        .unwrap_or(selection.cached_tokens)
        / usize::try_from(block_size).ok()?.max(1);
    let Some(hint) = selection.kv_hint.as_ref() else {
        return Some((local, local));
    };
    if hint.protocol_version != "0.1" || hint.actions.len() != 1 {
        return None;
    }
    let action = &hint.actions[0];
    if action.action_type != "kv.fetch" || action.action_version != "1.0" {
        return None;
    }
    let payload: KvSourceLocationsPayload =
        serde_json::from_value(serde_json::to_value(&action.payload).ok()?).ok()?;
    Some((local, local.max(payload.block_hashes.len())))
}

#[cfg(test)]
fn transfer_is_profitable(
    config: &KvRouterConfig,
    block_size: u32,
    local_prefix: usize,
    baseline_prefix: usize,
    donor_prefix: usize,
    lookup_us: f64,
) -> bool {
    transfer_is_profitable_for_kind(
        config,
        TransferSourceMemoryKind::Host,
        block_size,
        local_prefix,
        baseline_prefix,
        donor_prefix,
        lookup_us,
    )
}

fn transfer_is_profitable_for_kind(
    config: &KvRouterConfig,
    memory_kind: TransferSourceMemoryKind,
    block_size: u32,
    local_prefix: usize,
    baseline_prefix: usize,
    donor_prefix: usize,
    lookup_us: f64,
) -> bool {
    let (Some(block_us), Some(token_us)) = (
        match memory_kind {
            TransferSourceMemoryKind::Host => config.transfer_source_estimated_block_us,
            TransferSourceMemoryKind::Gpu => config.transfer_source_gpu_estimated_block_us,
        },
        config.transfer_source_prefill_token_us,
    ) else {
        return false;
    };
    if !block_us.is_finite()
        || block_us <= 0.0
        || !token_us.is_finite()
        || token_us <= 0.0
        || !lookup_us.is_finite()
        || lookup_us < 0.0
        || !config.minimum_estimated_ttft_saving_ms.is_finite()
        || config.minimum_estimated_ttft_saving_ms < 0.0
        || donor_prefix <= baseline_prefix
        || donor_prefix - baseline_prefix < config.minimum_transfer_prefix_blocks
    {
        return false;
    }
    let saved_us = (donor_prefix - baseline_prefix) as f64 * block_size as f64 * token_us;
    let cost_us = donor_prefix.saturating_sub(local_prefix) as f64 * block_us + lookup_us;
    saved_us.is_finite()
        && cost_us.is_finite()
        && saved_us - cost_us > config.minimum_estimated_ttft_saving_ms * 1000.0
}

impl RoutingHost {
    pub(super) async fn enrich_prefill_transfer_hint(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        source_host: Option<&RoutingHost>,
    ) {
        // Both optional lookups share one deadline and cumulative lookup cost.
        // Unsupported/unprofitable GPU capability falls through to the existing
        // independently enabled CPU path, then ordinary local computation.
        let started = Instant::now();
        for kind in [
            TransferSourceMemoryKind::Gpu,
            TransferSourceMemoryKind::Host,
        ] {
            if let Some(hint) = self
                .prefill_transfer_hint_for_kind(request, selection, source_host, kind, started)
                .await
            {
                selection.kv_hint = Some(hint);
                return;
            }
        }
    }

    async fn prefill_transfer_hint_for_kind(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &WorkerSelection,
        source_host: Option<&RoutingHost>,
        memory_kind: TransferSourceMemoryKind,
        started: Instant,
    ) -> Option<KvHint> {
        let Some(target_router) = self.kv_router_if_enabled() else {
            return None;
        };
        let config = target_router.kv_router_config();
        let enabled = |config: &KvRouterConfig| match memory_kind {
            TransferSourceMemoryKind::Host => config.enable_decode_to_prefill_reuse,
            TransferSourceMemoryKind::Gpu => config.enable_decode_to_prefill_gpu_reuse,
        };
        if !enabled(config) {
            return None;
        }
        let Some(source_router) = source_host.and_then(RoutingHost::kv_router_if_enabled) else {
            return None;
        };
        if !enabled(source_router.kv_router_config()) {
            return None;
        }
        let source_partition = source_router.selection.partition_key();
        if target_router.selection.partition_key() != source_partition {
            return None;
        }
        let block_size = target_router.block_size();
        let Some((mut local_prefix, mut baseline_prefix)) =
            baseline_prefix_blocks(selection, block_size)
        else {
            return None;
        };
        let (target_snapshot, target_domain) = {
            let configs = target_router.workers_with_configs.borrow();
            let Some(target) = configs.get(&selection.worker.worker_id) else {
                return None;
            };
            let metadata = match memory_kind {
                TransferSourceMemoryKind::Host => {
                    target.kv_hint_transfer_metadata_for_dp_rank(selection.worker.dp_rank)
                }
                TransferSourceMemoryKind::Gpu => {
                    target.kv_hint_gpu_transfer_metadata_for_dp_rank(selection.worker.dp_rank)
                }
            };
            let Some(metadata) = metadata else {
                return None;
            };
            if metadata.worker_type != "prefill" {
                return None;
            }
            let Some(domain) = target.router_hint_transfer_domain() else {
                return None;
            };
            (target.clone(), domain.to_owned())
        };
        let routing = request.routing.as_ref();
        let parts = RoutingRequestParts::new(request);
        let timeout = Duration::from_millis(config.transfer_source_lookup_budget_ms);
        if Instant::now() >= started + timeout {
            return None;
        }
        let mut query = TransferSourceQuery {
            model_name: &source_partition.model_name,
            routing_group: &source_partition.routing_group,
            prompt: PromptView {
                token_ids: Some(parts.token_ids),
                mm_routing_info: None,
                block_mm_infos: parts.block_mm_infos,
                block_hashes: None,
                sequence_hashes: None,
                isl_tokens: None,
                lora_name: routing.and_then(|routing| routing.lora_name.as_deref()),
                cache_namespace: routing.and_then(|routing| routing.cache_namespace.as_deref()),
                is_eagle: Some(target_router.is_eagle()),
            },
            target: selection.worker,
            target_worker_type: "prefill",
            memory_kind,
            target_gpu_direct: memory_kind == TransferSourceMemoryKind::Gpu,
            target_transfer_domain: Some(&target_domain),
            target_topology_domains: target_snapshot.topology_domains(),
            target_kv_transfer_domain: target_snapshot.kv_transfer_domain(),
            target_kv_transfer_enforcement: target_snapshot.kv_transfer_enforcement(),
            target_block_size: block_size,
            target_is_eagle: target_router.is_eagle(),
            minimum_prefix_blocks: baseline_prefix,
            deadline: started + timeout,
        };
        // Cancellation cancels only this advisory query. Existing prefill guard
        // and dispatch retain sole ownership of target cleanup.
        let context = request.context();
        if memory_kind == TransferSourceMemoryKind::Gpu {
            let retained = tokio::select! {
                biased;
                _ = context.stopped() => return None,
                result = tokio::time::timeout_at((started + timeout).into(), target_router.query_local_gpu_direct_prefix(&query)) => result,
            };
            let Ok(Ok(Some(retained))) = retained else {
                return None;
            };
            // Retained GPU blocks remain usable after native Device eviction.
            // Do not credit them as saved prefill or charge them as network copy.
            local_prefix = local_prefix.max(retained);
            baseline_prefix = baseline_prefix.max(local_prefix);
            query.minimum_prefix_blocks = baseline_prefix;
        }
        let result = tokio::select! {
            biased;
            _ = context.stopped() => return None,
            result = tokio::time::timeout_at((started + timeout).into(), source_router.query_transfer_sources(query)) => result,
        };
        let result = match result {
            Ok(Ok(Some(result))) => result,
            Ok(Ok(None)) => return None,
            Ok(Err(error)) => {
                tracing::debug!(request_id = context.id(), %error, "Decode transfer source unavailable");
                return None;
            }
            Err(_) => {
                tracing::debug!(
                    request_id = context.id(),
                    "Decode transfer source lookup timed out"
                );
                return None;
            }
        };
        if result.memory_kind != memory_kind {
            return None;
        }
        let donor_prefix = result.payload.block_hashes.len();
        let lookup_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        if !transfer_is_profitable_for_kind(
            config,
            memory_kind,
            block_size,
            local_prefix,
            baseline_prefix,
            donor_prefix,
            lookup_us,
        ) {
            return None;
        }
        // Revalidate target-dependent metadata after the asynchronous query.
        // A replacement worker/card must never inherit the previous estimate.
        let target_still_compatible = target_router
            .workers_with_configs
            .borrow()
            .get(&selection.worker.worker_id)
            .is_some_and(|target| {
                target == &target_snapshot
                    && target.router_hint_transfer_domain() == Some(target_domain.as_str())
                    && match memory_kind {
                        TransferSourceMemoryKind::Host => {
                            target.kv_hint_transfer_metadata_for_dp_rank(selection.worker.dp_rank)
                        }
                        TransferSourceMemoryKind::Gpu => target
                            .kv_hint_gpu_transfer_metadata_for_dp_rank(selection.worker.dp_rank),
                    }
                    .is_some_and(|metadata| metadata.worker_type == "prefill")
            });
        if !target_still_compatible {
            return None;
        }
        tracing::debug!(
            request_id = context.id(),
            source_memory_kind = ?memory_kind,
            donor_prefix_blocks = donor_prefix,
            local_prefix_blocks = local_prefix,
            baseline_prefix_blocks = baseline_prefix,
            lookup_us,
            "Attaching decode-to-prefill transfer hint"
        );
        let action = match memory_kind {
            TransferSourceMemoryKind::Host => {
                KvHintAction::fetch(Uuid::new_v4().to_string(), result.payload)
            }
            TransferSourceMemoryKind::Gpu => {
                KvHintAction::fetch_gpu(Uuid::new_v4().to_string(), result.payload, target_domain)
            }
        };
        Some(KvHint::new(Uuid::new_v4().to_string(), vec![action]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibrated_config() -> KvRouterConfig {
        KvRouterConfig {
            enable_decode_to_prefill_reuse: true,
            transfer_source_estimated_block_us: Some(10.0),
            transfer_source_prefill_token_us: Some(2.0),
            ..Default::default()
        }
    }

    #[test]
    fn unknown_cost_and_shorter_donors_never_replace_existing_hint() {
        assert!(!transfer_is_profitable(
            &KvRouterConfig::default(),
            16,
            0,
            0,
            20,
            0.0
        ));
        assert!(!transfer_is_profitable(
            &calibrated_config(),
            16,
            0,
            20,
            20,
            0.0
        ));
        assert!(!transfer_is_profitable(
            &calibrated_config(),
            16,
            0,
            20,
            19,
            0.0
        ));
    }

    #[test]
    fn full_transfer_cost_must_beat_local_and_same_role_prefix() {
        let config = calibrated_config();
        assert!(transfer_is_profitable(&config, 16, 10, 10, 30, 5.0));
        assert!(!transfer_is_profitable(&config, 16, 0, 29, 30, 5.0));
        assert!(transfer_is_profitable(&config, 16, 28, 29, 30, 5.0));
        assert!(!transfer_is_profitable(&config, 16, 10, 10, 30, 1000.0));
    }

    #[test]
    fn saving_margin_and_nonfinite_costs_fail_closed() {
        let mut config = calibrated_config();
        config.minimum_estimated_ttft_saving_ms = 1.0;
        assert!(!transfer_is_profitable(&config, 16, 10, 10, 30, 5.0));
        config.transfer_source_estimated_block_us = Some(f64::NAN);
        assert!(!transfer_is_profitable(&config, 16, 10, 10, 30, 5.0));
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::discovery::KvSourceMembershipCoordinator;
    use crate::kv_router::SelectionPolicySource;
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use dynamo_kv_router::WorkerType;
    use dynamo_kv_router::protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
        compute_block_hash_for_seq, compute_seq_hash_for_block,
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::Instance,
        distributed::DistributedConfig,
        pipeline::{AddressedRequest, Context, ManyIn, StreamingDispatch},
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tokio::sync::watch;

    #[derive(Default)]
    struct HintDispatch(Mutex<Vec<(u64, Option<KvHint>)>>);

    #[async_trait]
    impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for HintDispatch {
        async fn generate(
            &self,
            request: SingleIn<AddressedRequest<PreprocessedRequest>>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let (addressed, context) = request.transfer(());
            let (request, _, instance) = addressed.into_parts();
            self.0
                .lock()
                .unwrap()
                .push((instance.unwrap().id(), request.kv_hint));
            Ok(ResponseStream::new(
                Box::pin(stream::once(async {
                    Annotated::from_data(LLMEngineOutput {
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    })
                })),
                context.context(),
            ))
        }
        async fn generate_bidirectional(
            &self,
            _: Instance,
            _: String,
            _: ManyIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            unreachable!()
        }
    }

    async fn host(
        distributed: &DistributedRuntime,
        component_name: &str,
        role: WorkerType,
        enabled: bool,
    ) -> (RoutingHost, u64, Arc<HintDispatch>) {
        host_with_topology(distributed, component_name, role, enabled, None).await
    }

    async fn host_with_topology(
        distributed: &DistributedRuntime,
        component_name: &str,
        role: WorkerType,
        enabled: bool,
        topology: Option<(
            &str,
            Option<&str>,
            dynamo_kv_router::protocols::KvTransferEnforcement,
        )>,
    ) -> (RoutingHost, u64, Arc<HintDispatch>) {
        host_with_gpu_options(
            distributed,
            component_name,
            role,
            enabled,
            topology,
            false,
            0.0,
        )
        .await
    }

    async fn host_with_gpu_options(
        distributed: &DistributedRuntime,
        component_name: &str,
        role: WorkerType,
        enabled: bool,
        topology: Option<(
            &str,
            Option<&str>,
            dynamo_kv_router::protocols::KvTransferEnforcement,
        )>,
        gpu: bool,
        minimum_saving_ms: f64,
    ) -> (RoutingHost, u64, Arc<HintDispatch>) {
        // Discovery instance IDs belong to a DRT, not to an endpoint. Model
        // distinct serving processes so the donor is not rejected as the target.
        let distributed = DistributedRuntime::new(
            distributed.runtime().clone(),
            DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        let endpoint = distributed
            .namespace("d-to-p-transfer-tests".to_string())
            .unwrap()
            .component(component_name.to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let mut runtime_config = ModelRuntimeConfig::default();
        if let Some((domain, island, enforcement)) = topology {
            runtime_config.kv_transfer_domain = Some(domain.to_string());
            runtime_config.kv_transfer_enforcement = Some(enforcement);
            if let Some(island) = island {
                runtime_config
                    .topology_domains
                    .insert(domain.to_string(), island.to_string());
            }
        }
        runtime_config
            .runtime_data
            .insert("router_hint".to_string(), serde_json::json!(!gpu));
        runtime_config.runtime_data.insert(
            "router_hint_worker_type".to_string(),
            serde_json::json!(role.as_str()),
        );
        runtime_config.runtime_data.insert(
            "router_hint_transfer_domain".to_string(),
            serde_json::json!("certified-test-layout"),
        );
        runtime_config.runtime_data.insert(
            "router_hint_source_control_endpoints".to_string(),
            serde_json::json!({"0": format!("tcp://{component_name}:9000")}),
        );
        if gpu {
            runtime_config
                .runtime_data
                .insert("router_hint_gpu_direct".into(), serde_json::json!(true));
            runtime_config.runtime_data.insert(
                "router_hint_gpu_source_control_endpoints".into(),
                serde_json::json!({"0": format!("tcp://{component_name}:9010")}),
            );
        }
        let (_workers_tx, workers) = watch::channel(HashMap::from([(worker_id, runtime_config)]));
        let membership = KvSourceMembershipCoordinator::start(
            endpoint.id(),
            workers.clone(),
            distributed.discovery(),
        )
        .subscribe();
        let config = KvRouterConfig {
            enable_decode_to_prefill_reuse: enabled && !gpu,
            enable_decode_to_prefill_gpu_reuse: enabled && gpu,
            transfer_source_gpu_estimated_block_us: gpu.then_some(1.0),
            minimum_estimated_ttft_saving_ms: minimum_saving_ms,
            transfer_source_lookup_budget_ms: 200,
            transfer_source_estimated_block_us: Some(1.0),
            transfer_source_prefill_token_us: Some(if gpu { 100_000.0 } else { 1_000.0 }),
            skip_initial_worker_wait: true,
            overlap_score_credit: 0.0,
            host_cache_hit_weight: 0.0,
            disk_cache_hit_weight: 0.0,
            router_track_active_blocks: false,
            router_track_prefill_tokens: false,
            ..Default::default()
        };
        let chooser = KvRouter::new_with_worker_role(
            endpoint,
            client.clone(),
            workers,
            Some(membership),
            16,
            SelectionPolicySource::Registry,
            Some(config),
            None,
            Some(role),
            role.as_str(),
            Some("test".to_string()),
            false,
            None,
            None,
        )
        .await
        .unwrap();
        let dispatch = Arc::new(HintDispatch::default());
        let inner = PushRouter::from_client_with_dispatch(
            client,
            RouterMode::KV,
            dispatch.clone() as Arc<dyn StreamingDispatch<_, _>>,
        )
        .await
        .unwrap();
        (
            RoutingHost::new(inner, Arc::new(chooser), None).unwrap(),
            worker_id,
            dispatch,
        )
    }

    fn request() -> SingleIn<PreprocessedRequest> {
        Context::new(
            PreprocessedRequest::builder()
                .model("test".to_string())
                .token_ids((1..=64).collect::<Vec<_>>())
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .unwrap(),
        )
    }

    async fn seed_host_prefix(
        host: &RoutingHost,
        worker_id: u64,
        tokens: &[u32],
        blocks: usize,
    ) -> Vec<ExternalSequenceBlockHash> {
        seed_prefix(host, worker_id, tokens, blocks, StorageTier::HostPinned).await
    }

    async fn seed_prefix(
        host: &RoutingHost,
        worker_id: u64,
        tokens: &[u32],
        blocks: usize,
        tier: StorageTier,
    ) -> Vec<ExternalSequenceBlockHash> {
        let local = compute_block_hash_for_seq(tokens, 16, BlockHashOptions::default());
        let external = compute_seq_hash_for_block(&local);
        let hashes = external[..blocks]
            .iter()
            .copied()
            .map(ExternalSequenceBlockHash)
            .collect::<Vec<_>>();
        let event = RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                dp_rank: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: local
                        .iter()
                        .zip(&hashes)
                        .map(|(local, external)| KvCacheStoredBlockData {
                            block_hash: *external,
                            tokens_hash: *local,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
            },
            tier,
        );
        host.kv_router()
            .indexer()
            .apply_event_routed(event)
            .await
            .unwrap();
        host.kv_router().indexer().dump_events().await.unwrap();
        hashes
    }

    fn same_role_hint(prefix: usize) -> KvHint {
        KvHint::new(
            "prefill-source",
            vec![KvHintAction::fetch(
                "fetch",
                KvSourceLocationsPayload {
                    source_control_endpoint: "tcp://prefill-source:9000".to_string(),
                    block_hashes: (0..prefix)
                        .map(|hash| ExternalSequenceBlockHash(hash as u64))
                        .collect(),
                },
            )],
        )
    }

    async fn selected(
        host: &RoutingHost,
        request: &SingleIn<PreprocessedRequest>,
    ) -> WorkerSelection {
        host.select_with_affinity(
            request,
            RequestPhase::Prefill,
            true,
            &CleanupBudget::default(),
        )
        .await
        .unwrap()
        .0
    }

    async fn assert_no_decode_bookings(source: &RoutingHost) {
        let loads = source
            .kv_router()
            .get_potential_loads(&[], None, None, None, None)
            .await
            .unwrap();
        assert!(
            loads
                .iter()
                .all(|load| load.active_requests == 0 && load.potential_prefill_tokens == 0),
            "donor lookup changed decode load: {loads:?}"
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn longer_decode_root_chain_reaches_final_prefill_dispatch_without_decode_booking() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let (target, target_id, dispatch) =
            Box::pin(host(&distributed, "prefill", WorkerType::Prefill, true)).await;
        let (source, source_id, _) =
            Box::pin(host(&distributed, "decode", WorkerType::Decode, true)).await;
        assert_ne!(source_id, target_id, "the donor must be a different worker");
        assert!(
            !source
                .kv_router()
                .required_worker_inputs()
                .contains(WorkerInputs::CACHE),
            "source visibility must not change decode policy inputs"
        );
        assert!(
            source
                .kv_router()
                .indexer()
                .supports_kv_transfer_chain_retention()
        );
        let request = request();
        let expected = seed_host_prefix(&source, source_id, &request.token_ids, 3).await;
        let (_, mut stream) =
            Box::pin(
                target.select_and_dispatch_prefill(request, Some(&source), |_, target| Ok(target)),
            )
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        {
            let calls = dispatch.0.lock().unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].0, target_id);
            let hint = calls[0]
                .1
                .as_ref()
                .expect("final backend request carries decode hint");
            let payload: KvSourceLocationsPayload =
                serde_json::from_value(serde_json::to_value(&hint.actions[0].payload).unwrap())
                    .unwrap();
            assert_eq!(payload.block_hashes, expected);
            assert_eq!(payload.source_control_endpoint, "tcp://decode:9000");
        }
        assert_no_decode_bookings(&source).await;
        drop(target);
        drop(source);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn disabled_missing_and_cancelled_source_lookup_preserve_same_role_hint() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let (disabled, _, _) =
            Box::pin(host(&distributed, "disabled", WorkerType::Prefill, false)).await;
        let (enabled, _, _) =
            Box::pin(host(&distributed, "enabled", WorkerType::Prefill, true)).await;
        let (source, source_id, _) =
            Box::pin(host(&distributed, "decode", WorkerType::Decode, true)).await;
        // A discovered decoder with no eligible host inventory is an advisory
        // cache miss and must preserve the already-selected same-role source.
        let input = request();
        let mut selection = selected(&enabled, &input).await;
        let expected = same_role_hint(1);
        selection.kv_hint = Some(expected.clone());
        enabled
            .enrich_prefill_transfer_hint(&input, &mut selection, Some(&source))
            .await;
        assert_eq!(selection.kv_hint, Some(expected));
        let seed_request = request();
        seed_host_prefix(&source, source_id, &seed_request.token_ids, 3).await;
        for (target, donor, cancel) in [
            (&disabled, Some(&source), false),
            (&enabled, None, false),
            (&enabled, Some(&source), true),
        ] {
            let input = request();
            let mut selection = selected(target, &input).await;
            let expected = same_role_hint(1);
            selection.kv_hint = Some(expected.clone());
            if cancel {
                input.context().stop();
            }
            target
                .enrich_prefill_transfer_hint(&input, &mut selection, donor)
                .await;
            assert_eq!(selection.kv_hint, Some(expected));
        }
        assert_no_decode_bookings(&source).await;
        drop(disabled);
        drop(enabled);
        drop(source);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn existing_prefill_prefix_and_full_transfer_cost_prevent_unprofitable_replacement() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let (target, _, _) =
            Box::pin(host(&distributed, "prefill", WorkerType::Prefill, true)).await;
        let (source, source_id, _) =
            Box::pin(host(&distributed, "decode", WorkerType::Decode, true)).await;
        let input = request();
        seed_host_prefix(&source, source_id, &input.token_ids, 3).await;
        let mut selection = selected(&target, &input).await;
        let expected = same_role_hint(3);
        selection.kv_hint = Some(expected.clone());
        target
            .enrich_prefill_transfer_hint(&input, &mut selection, Some(&source))
            .await;
        assert_eq!(selection.kv_hint, Some(expected));
        assert_no_decode_bookings(&source).await;
        drop(target);
        drop(source);
        runtime.shutdown();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn cross_role_bridge_preserves_admitted_target_topology_policy() {
        use dynamo_kv_router::protocols::KvTransferEnforcement::{Preferred, Required};
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let (source, source_id, _) = Box::pin(host_with_topology(
            &distributed,
            "topology-decode",
            WorkerType::Decode,
            true,
            Some(("zone", Some("near"), Required)),
        ))
        .await;
        let seed = request();
        seed_host_prefix(&source, source_id, &seed.token_ids, 3).await;
        for (name, island, enforcement, expected) in [
            ("required-near", Some("near"), Required, true),
            ("required-far", Some("far"), Required, false),
            ("required-missing", None, Required, false),
            ("preferred-far", Some("far"), Preferred, true),
        ] {
            let (target, _, _) = Box::pin(host_with_topology(
                &distributed,
                name,
                WorkerType::Prefill,
                true,
                Some(("zone", island, enforcement)),
            ))
            .await;
            let input = request();
            let mut selection = selected(&target, &input).await;
            target
                .enrich_prefill_transfer_hint(&input, &mut selection, Some(&source))
                .await;
            assert_eq!(selection.kv_hint.is_some(), expected, "{name}");
            drop(target);
        }
        assert_no_decode_bookings(&source).await;
        drop(source);
        runtime.shutdown();
    }
    #[test]
    fn gpu_cost_never_uses_cpu_estimate_or_nonfinite_derived_savings() {
        let mut config = calibrated_config();
        assert!(!transfer_is_profitable_for_kind(
            &config,
            TransferSourceMemoryKind::Gpu,
            16,
            1,
            1,
            3,
            0.0
        ));
        config.transfer_source_gpu_estimated_block_us = Some(1.0);
        assert!(transfer_is_profitable_for_kind(
            &config,
            TransferSourceMemoryKind::Gpu,
            16,
            1,
            1,
            3,
            0.0
        ));
        config.transfer_source_gpu_estimated_block_us = Some(1000.0);
        assert!(!transfer_is_profitable_for_kind(
            &config,
            TransferSourceMemoryKind::Gpu,
            16,
            1,
            1,
            3,
            0.0
        ));
        config.transfer_source_gpu_estimated_block_us = Some(1.0);
        config.transfer_source_prefill_token_us = Some(f64::MAX);
        assert!(!transfer_is_profitable_for_kind(
            &config,
            TransferSourceMemoryKind::Gpu,
            16,
            1,
            1,
            3,
            0.0
        ));
    }
    #[tokio::test]
    #[serial_test::serial]
    async fn gpu_retained_hint_reaches_opted_in_prefill_dispatch_without_decode_booking() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let (target, target_id, dispatch) = Box::pin(host_with_gpu_options(
            &distributed,
            "gpu-prefill",
            WorkerType::Prefill,
            true,
            None,
            true,
            0.0,
        ))
        .await;
        let (source, source_id, _) = Box::pin(host_with_gpu_options(
            &distributed,
            "gpu-decode",
            WorkerType::Decode,
            true,
            None,
            true,
            0.0,
        ))
        .await;
        let request = request();
        let expected = seed_prefix(
            &source,
            source_id,
            &request.token_ids,
            3,
            StorageTier::GpuDirect,
        )
        .await;
        let (_, mut stream) =
            Box::pin(
                target.select_and_dispatch_prefill(request, Some(&source), |_, target| Ok(target)),
            )
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        {
            let calls = dispatch.0.lock().unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].0, target_id);
            let action = &calls[0].1.as_ref().expect("explicit GPU hint").actions[0];
            assert_eq!(action.action_type, "kv.fetch_gpu");
            let payload: dynamo_kv_router::kv_hints::KvGpuSourceLocationsPayload =
                serde_json::from_value(serde_json::to_value(&action.payload).unwrap()).unwrap();
            assert_eq!(payload.block_hashes, expected);
            assert_eq!(payload.source_control_endpoint, "tcp://gpu-decode:9010");
            assert_eq!(payload.compatibility_domain, "certified-test-layout");
        }
        assert_no_decode_bookings(&source).await;
        drop(target);
        drop(source);
        runtime.shutdown();
    }
    #[tokio::test]
    #[serial_test::serial]
    async fn gpu_local_retained_baseline_prevents_false_prefill_savings_after_native_eviction() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let (target, target_id, dispatch) = Box::pin(host_with_gpu_options(
            &distributed,
            "retained-prefill",
            WorkerType::Prefill,
            true,
            None,
            true,
            3000.0,
        ))
        .await;
        let (source, source_id, _) = Box::pin(host_with_gpu_options(
            &distributed,
            "retained-decode",
            WorkerType::Decode,
            true,
            None,
            true,
            0.0,
        ))
        .await;
        let request = request();
        seed_prefix(
            &target,
            target_id,
            &request.token_ids,
            2,
            StorageTier::GpuDirect,
        )
        .await;
        seed_prefix(
            &source,
            source_id,
            &request.token_ids,
            3,
            StorageTier::GpuDirect,
        )
        .await;
        // The target has no native Device blocks but already retains two GPU
        // rows. Only one new row is saved: 1600ms < the fixture's 3000ms margin.
        // Incorrectly crediting all three donor rows would admit this hint.
        let (_, mut stream) =
            Box::pin(
                target.select_and_dispatch_prefill(request, Some(&source), |_, target| Ok(target)),
            )
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        assert!(dispatch.0.lock().unwrap()[0].1.is_none());
        assert_no_decode_bookings(&source).await;
        drop(target);
        drop(source);
        runtime.shutdown();
    }
}
