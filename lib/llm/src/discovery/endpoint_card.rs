// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{Discovery, DiscoveryEvent, DiscoveryInstanceId, DiscoveryQuery};
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_runtime::protocols::EndpointId;

use crate::model_card::ModelDeploymentCard;

/// List the model card that each instance of `endpoint` registered, keyed by
/// instance ID, as discovery reports them now.
///
/// LoRA adapter cards are left out, so each instance maps to its base model
/// card. A card that does not deserialize, for example one written by an
/// incompatible version, is logged and skipped rather than failing the list.
pub async fn list_endpoint_model_cards(
    discovery: &dyn Discovery,
    endpoint: &EndpointId,
) -> Result<HashMap<u64, ModelDeploymentCard>> {
    let instances = discovery
        .list(DiscoveryQuery::EndpointModels {
            namespace: endpoint.namespace.clone(),
            component: endpoint.component.clone(),
            endpoint: endpoint.name.clone(),
        })
        .await?;

    let mut cards = HashMap::new();
    for instance in instances {
        let DiscoveryInstanceId::Model(model_id) = instance.id() else {
            continue;
        };
        if model_id.model_suffix.is_some() {
            continue;
        }
        match instance.deserialize_model::<ModelDeploymentCard>() {
            Ok(card) if card.lora.is_none() => {
                cards.insert(model_id.instance_id, card);
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(
                    instance_id = model_id.instance_id,
                    %error,
                    "Skipping a model card that does not deserialize"
                );
            }
        }
    }
    Ok(cards)
}

/// Wait for a worker on `endpoint` to publish its `ModelDeploymentCard`.
///
/// Uses the watch-based discovery API so the wait is event-driven (no polling)
/// and returns as soon as the first card is observed. Existing registrations
/// are delivered as `Added` events at the start of the stream, so callers do
/// not need to issue a separate `list` first.
///
/// Returns `Ok(Some(card))` once a card is observed, or `Ok(None)` if `timeout`
/// elapses, the supplied `cancel_token` fires, or the discovery stream ends
/// without ever delivering a deserializable model card. When `cancel_token` is
/// `None`, the runtime's primary token is used so the wait aborts on shutdown.
pub async fn wait_for_endpoint_model_card(
    endpoint: &Endpoint,
    timeout: Duration,
    cancel_token: Option<CancellationToken>,
) -> Result<Option<ModelDeploymentCard>> {
    let cancel_token = cancel_token.unwrap_or_else(|| endpoint.drt().primary_token());
    let eid = endpoint.id();
    let query = DiscoveryQuery::EndpointModels {
        namespace: eid.namespace,
        component: eid.component,
        endpoint: eid.name,
    };

    let mut stream = endpoint
        .drt()
        .discovery()
        .list_and_watch(query, Some(cancel_token.clone()))
        .await?;

    let find_card = async {
        while let Some(event) = stream.next().await {
            match event {
                Ok(DiscoveryEvent::Added(instance)) => {
                    let card = match instance.deserialize_model::<ModelDeploymentCard>() {
                        Ok(card) => card,
                        Err(error) => {
                            tracing::warn!(
                                %error,
                                discovery_instance = ?instance.id(),
                                "Failed to deserialize model card while waiting for endpoint registration; continuing"
                            );
                            continue;
                        }
                    };
                    return Some(card);
                }
                Ok(DiscoveryEvent::ModelTaintsUpdated(_)) => {}
                Ok(DiscoveryEvent::Removed(_)) => {}
                Ok(DiscoveryEvent::Resync(_)) => {}
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "Discovery event error while waiting for endpoint model card; continuing"
                    );
                }
            }
        }
        None
    };

    Ok(tokio::select! {
        card = find_card => card,
        _ = tokio::time::sleep(timeout) => None,
        _ = cancel_token.cancelled() => None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dynamo_runtime::discovery::{
        DiscoveryInstance, DiscoverySpec, MockDiscovery, SharedMockRegistry,
    };

    use super::*;
    use crate::model_card::LoraInfo;

    fn endpoint_id(component: &str, name: &str) -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: component.to_string(),
            name: name.to_string(),
        }
    }

    /// A discovery client for one worker instance, sharing `registry` with the others.
    fn worker(instance_id: u64, registry: &SharedMockRegistry) -> Arc<dyn Discovery> {
        Arc::new(MockDiscovery::new(Some(instance_id), registry.clone()))
    }

    fn card(name: &str, runtime_value: &str) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(name);
        card.runtime_config
            .set_engine_specific("side_channel", serde_json::json!(runtime_value))
            .unwrap();
        card
    }

    async fn register_card(
        discovery: &dyn Discovery,
        endpoint: &EndpointId,
        card: &ModelDeploymentCard,
        model_suffix: Option<&str>,
    ) -> DiscoveryInstance {
        let mut spec = DiscoverySpec::from_model(
            endpoint.namespace.clone(),
            endpoint.component.clone(),
            endpoint.name.clone(),
            card,
        )
        .unwrap();
        if let DiscoverySpec::Model {
            model_suffix: suffix,
            ..
        } = &mut spec
        {
            *suffix = model_suffix.map(str::to_owned);
        }
        discovery.register(spec).await.unwrap()
    }

    fn side_channel(card: &ModelDeploymentCard) -> Option<serde_json::Value> {
        card.runtime_config
            .get_engine_specific::<serde_json::Value>("side_channel")
            .unwrap()
    }

    #[tokio::test]
    async fn lists_each_instance_card_by_instance_id() {
        let registry = SharedMockRegistry::new();
        let decode = endpoint_id("decode", "generate");
        register_card(
            worker(1, &registry).as_ref(),
            &decode,
            &card("m", "a"),
            None,
        )
        .await;
        register_card(
            worker(2, &registry).as_ref(),
            &decode,
            &card("m", "b"),
            None,
        )
        .await;

        let cards = list_endpoint_model_cards(worker(99, &registry).as_ref(), &decode)
            .await
            .unwrap();

        assert_eq!(cards.len(), 2);
        assert_eq!(side_channel(&cards[&1]), Some(serde_json::json!("a")));
        assert_eq!(side_channel(&cards[&2]), Some(serde_json::json!("b")));
    }

    #[tokio::test]
    async fn returns_empty_map_when_no_instance_registered() {
        let registry = SharedMockRegistry::new();

        let cards = list_endpoint_model_cards(
            worker(99, &registry).as_ref(),
            &endpoint_id("decode", "generate"),
        )
        .await
        .unwrap();

        assert!(cards.is_empty());
    }

    #[tokio::test]
    async fn leaves_out_cards_of_other_endpoints() {
        let registry = SharedMockRegistry::new();
        let decode = endpoint_id("decode", "generate");
        register_card(
            worker(1, &registry).as_ref(),
            &decode,
            &card("m", "a"),
            None,
        )
        .await;
        // Same component, other endpoint; and other component, same endpoint name.
        register_card(
            worker(2, &registry).as_ref(),
            &endpoint_id("decode", "other"),
            &card("m", "b"),
            None,
        )
        .await;
        register_card(
            worker(3, &registry).as_ref(),
            &endpoint_id("prefill", "generate"),
            &card("m", "c"),
            None,
        )
        .await;

        let cards = list_endpoint_model_cards(worker(99, &registry).as_ref(), &decode)
            .await
            .unwrap();

        assert_eq!(cards.keys().copied().collect::<Vec<_>>(), vec![1]);
    }

    #[tokio::test]
    async fn leaves_out_lora_adapter_cards() {
        let registry = SharedMockRegistry::new();
        let decode = endpoint_id("decode", "generate");
        let discovery = worker(1, &registry);
        register_card(discovery.as_ref(), &decode, &card("base", "a"), None).await;
        // Shaped as register_model writes a LoRA card: it points at the base
        // model and is stored under a model suffix.
        let mut lora = card("adapter", "a");
        lora.source_path = Some("base".to_string());
        lora.lora = Some(LoraInfo {
            name: "adapter".to_string(),
            max_gpu_lora_count: None,
        });
        register_card(discovery.as_ref(), &decode, &lora, Some("adapter")).await;

        let cards = list_endpoint_model_cards(worker(99, &registry).as_ref(), &decode)
            .await
            .unwrap();

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[&1].name(), "base");
    }

    #[tokio::test]
    async fn skips_a_card_that_does_not_deserialize() {
        let registry = SharedMockRegistry::new();
        let decode = endpoint_id("decode", "generate");
        register_card(
            worker(1, &registry).as_ref(),
            &decode,
            &card("m", "a"),
            None,
        )
        .await;
        worker(2, &registry)
            .register(DiscoverySpec::Model {
                namespace: decode.namespace.clone(),
                component: decode.component.clone(),
                endpoint: decode.name.clone(),
                card_json: serde_json::json!({
                    "display_name": "m",
                    "runtime_config": "not an object",
                }),
                model_suffix: None,
            })
            .await
            .unwrap();

        let cards = list_endpoint_model_cards(worker(99, &registry).as_ref(), &decode)
            .await
            .unwrap();

        assert_eq!(cards.keys().copied().collect::<Vec<_>>(), vec![1]);
    }

    #[tokio::test]
    async fn drops_an_instance_after_it_unregisters() {
        let registry = SharedMockRegistry::new();
        let decode = endpoint_id("decode", "generate");
        let leaving = worker(1, &registry);
        let instance = register_card(leaving.as_ref(), &decode, &card("m", "a"), None).await;
        register_card(
            worker(2, &registry).as_ref(),
            &decode,
            &card("m", "b"),
            None,
        )
        .await;
        let reader = worker(99, &registry);
        assert_eq!(
            list_endpoint_model_cards(reader.as_ref(), &decode)
                .await
                .unwrap()
                .len(),
            2
        );

        leaving.unregister(instance).await.unwrap();

        let cards = list_endpoint_model_cards(reader.as_ref(), &decode)
            .await
            .unwrap();
        assert_eq!(cards.keys().copied().collect::<Vec<_>>(), vec![2]);
    }
}
