// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use dynamo_backend_common::DynamoError;
use dynamo_llm::local_model::derive_lora_suffix;
use dynamo_llm::model_card::{LoraInfo, ModelDeploymentCard};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use serde_json::Value;

use crate::client;
use crate::proto as pb;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LoadLoraUpdate {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) load_inplace: bool,
}

pub(crate) fn parse_load_lora(body: &Value) -> Result<LoadLoraUpdate, DynamoError> {
    let name = body
        .get("lora_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| client::invalid_argument("lora_name must be a non-empty string"))?;
    let uri = body
        .pointer("/source/uri")
        .and_then(Value::as_str)
        .ok_or_else(|| client::invalid_argument("source.uri must be a local file URI"))?;
    let url = url::Url::parse(uri)
        .map_err(|_| client::invalid_argument("source.uri must be a local file URI"))?;
    if url.scheme() != "file"
        || url.host_str().is_some_and(|host| host != "localhost")
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(client::invalid_argument(
            "source.uri must be a local file URI",
        ));
    }
    let path = url
        .to_file_path()
        .map_err(|_| client::invalid_argument("source.uri must be a local file URI"))?;
    if !path.is_absolute() {
        return Err(client::invalid_argument(
            "source.uri must be a local file URI",
        ));
    }
    let load_inplace = match body.get("load_inplace") {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| client::invalid_argument("load_inplace must be a boolean"))?,
        None => true,
    };
    Ok(LoadLoraUpdate {
        name: name.to_string(),
        path,
        load_inplace,
    })
}

pub(crate) fn next_lora_id(loaded: &[pb::LoraAdapter], name: &str) -> Result<i64, DynamoError> {
    if let Some(adapter) = loaded.iter().find(|adapter| adapter.lora_name == name) {
        return (adapter.lora_id > 0)
            .then_some(adapter.lora_id)
            .ok_or_else(|| client::invalid_argument("loaded LoRA adapter ID is not positive"));
    }
    loaded
        .iter()
        .map(|adapter| adapter.lora_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .filter(|id| *id > 0)
        .ok_or_else(|| client::invalid_argument("no positive LoRA adapter ID is available"))
}

pub(crate) async fn publish_lora_model(
    endpoint: &Endpoint,
    adapter: &pb::LoraAdapter,
    max_loras: u32,
) -> Result<(), DynamoError> {
    let endpoint_id = endpoint.id();
    let discovery = endpoint.drt().discovery();
    publish_lora_model_to_discovery(
        discovery.as_ref(),
        &endpoint_id.namespace,
        &endpoint_id.component,
        &endpoint_id.name,
        endpoint.drt().connection_id(),
        adapter,
        max_loras,
    )
    .await
}

pub(crate) async fn publish_lora_model_to_discovery(
    discovery: &dyn Discovery,
    namespace: &str,
    component: &str,
    endpoint: &str,
    instance_id: u64,
    adapter: &pb::LoraAdapter,
    max_loras: u32,
) -> Result<(), DynamoError> {
    let models = discovery
        .list(DiscoveryQuery::EndpointModels {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
        })
        .await
        .map_err(|error| {
            client::protocol_error(format!("failed to query base model discovery: {error}"))
        })?;
    let base = models
        .iter()
        .find(|instance| {
            matches!(
                instance,
                DiscoveryInstance::Model {
                    instance_id: candidate_id,
                    model_suffix: None,
                    ..
                } if *candidate_id == instance_id
            )
        })
        .ok_or_else(|| client::protocol_error("base model is not registered in discovery"))?;
    let mut card = base
        .deserialize_model::<ModelDeploymentCard>()
        .map_err(|error| client::protocol_error(format!("invalid base model card: {error}")))?;
    if card.source_path.is_none() {
        card.source_path = Some(card.name().to_string());
    }
    card.set_name(&adapter.lora_name);
    card.aliases.clear();
    card.lora = Some(LoraInfo {
        name: adapter.lora_name.clone(),
        max_gpu_lora_count: (max_loras > 0).then_some(max_loras),
    });
    card.user_data = Some(serde_json::json!({
        "lora_adapter": true,
        "lora_id": adapter.lora_id,
    }));
    let spec = DiscoverySpec::from_model_with_suffix(
        namespace.to_string(),
        component.to_string(),
        endpoint.to_string(),
        &card,
        derive_lora_suffix(Some(&adapter.lora_name)),
    )
    .map_err(|error| client::protocol_error(format!("failed to build LoRA model card: {error}")))?;
    discovery.register(spec).await.map_err(|error| {
        client::protocol_error(format!("failed to publish LoRA model: {error}"))
    })?;
    Ok(())
}
