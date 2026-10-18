// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Advisory, bounded decode-source lookup after prefill admission. This path
//! owns no source bookings, affinity state, buffers, or transport operations.

use std::time::Instant;

use dynamo_kv_router::{
    config::KvRouterConfig,
    kv_hints::{KvHint, KvHintAction, KvSourceLocationsPayload},
    services::selection::{PromptView, TransferSourceQuery},
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

fn transfer_is_profitable(
    config: &KvRouterConfig,
    block_size: u32,
    local_prefix: usize,
    baseline_prefix: usize,
    donor_prefix: usize,
    lookup_us: f64,
) -> bool {
    let (Some(block_us), Some(token_us)) = (
        config.transfer_source_estimated_block_us,
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
    saved_us - cost_us > config.minimum_estimated_ttft_saving_ms * 1000.0
}

impl RoutingHost {
    pub(super) async fn enrich_prefill_transfer_hint(
        &self,
        request: &SingleIn<PreprocessedRequest>,
        selection: &mut WorkerSelection,
        source_host: Option<&RoutingHost>,
    ) {
        let Some(target_router) = self.kv_router_if_enabled() else {
            return;
        };
        let config = target_router.kv_router_config();
        if !config.enable_decode_to_prefill_reuse {
            return;
        }
        let Some(source_router) = source_host.and_then(RoutingHost::kv_router_if_enabled) else {
            return;
        };
        if !source_router
            .kv_router_config()
            .enable_decode_to_prefill_reuse
        {
            return;
        }
        let source_partition = source_router.selection.partition_key();
        if target_router.selection.partition_key() != source_partition {
            return;
        }
        let block_size = target_router.block_size();
        let Some((local_prefix, baseline_prefix)) = baseline_prefix_blocks(selection, block_size)
        else {
            return;
        };
        let (target_snapshot, target_domain) = {
            let configs = target_router.workers_with_configs.borrow();
            let Some(target) = configs.get(&selection.worker.worker_id) else {
                return;
            };
            let Some(metadata) =
                target.kv_hint_transfer_metadata_for_dp_rank(selection.worker.dp_rank)
            else {
                return;
            };
            if metadata.worker_type != "prefill" {
                return;
            }
            let Some(domain) = target.router_hint_transfer_domain() else {
                return;
            };
            (target.clone(), domain.to_owned())
        };
        let routing = request.routing.as_ref();
        let parts = RoutingRequestParts::new(request);
        let timeout = Duration::from_millis(config.transfer_source_lookup_budget_ms);
        let started = Instant::now();
        let query = TransferSourceQuery {
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
            target_transfer_domain: Some(&target_domain),
            target_block_size: block_size,
            target_is_eagle: target_router.is_eagle(),
            minimum_prefix_blocks: baseline_prefix,
            deadline: started + timeout,
        };
        // Cancellation cancels only this advisory query. Existing prefill guard
        // and dispatch retain sole ownership of target cleanup.
        let context = request.context();
        let result = tokio::select! {
            biased;
            _ = context.stopped() => return,
            result = tokio::time::timeout(timeout, source_router.query_transfer_sources(query)) => result,
        };
        let result = match result {
            Ok(Ok(Some(result))) => result,
            Ok(Ok(None)) => return,
            Ok(Err(error)) => {
                tracing::debug!(request_id = context.id(), %error, "Decode transfer source unavailable");
                return;
            }
            Err(_) => {
                tracing::debug!(
                    request_id = context.id(),
                    "Decode transfer source lookup timed out"
                );
                return;
            }
        };
        let donor_prefix = result.payload.block_hashes.len();
        let lookup_us = started.elapsed().as_secs_f64() * 1_000_000.0;
        if !transfer_is_profitable(
            config,
            block_size,
            local_prefix,
            baseline_prefix,
            donor_prefix,
            lookup_us,
        ) {
            return;
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
                    && target
                        .kv_hint_transfer_metadata_for_dp_rank(selection.worker.dp_rank)
                        .is_some_and(|metadata| metadata.worker_type == "prefill")
            });
        if !target_still_compatible {
            return;
        }
        tracing::debug!(
            request_id = context.id(),
            donor_prefix_blocks = donor_prefix,
            local_prefix_blocks = local_prefix,
            baseline_prefix_blocks = baseline_prefix,
            lookup_us,
            "Attaching decode-to-prefill transfer hint"
        );
        selection.kv_hint = Some(KvHint::new(
            Uuid::new_v4().to_string(),
            vec![KvHintAction::fetch(
                Uuid::new_v4().to_string(),
                result.payload,
            )],
        ));
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
        runtime_config
            .runtime_data
            .insert("router_hint".to_string(), serde_json::json!(true));
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
        let (_workers_tx, workers) = watch::channel(HashMap::from([(worker_id, runtime_config)]));
        let membership = KvSourceMembershipCoordinator::start(
            endpoint.id(),
            workers.clone(),
            distributed.discovery(),
        )
        .subscribe();
        let config = KvRouterConfig {
            enable_decode_to_prefill_reuse: enabled,
            transfer_source_lookup_budget_ms: 200,
            transfer_source_estimated_block_us: Some(1.0),
            transfer_source_prefill_token_us: Some(1_000.0),
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
            StorageTier::HostPinned,
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
}
