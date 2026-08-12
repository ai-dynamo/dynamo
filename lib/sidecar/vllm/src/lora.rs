// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use dynamo_backend_common::DynamoError;
use dynamo_llm::local_model::derive_lora_suffix;
use dynamo_llm::lora::{
    HuggingFaceLoRASource, LoRACache, LoRADownloader, LoRASource, LocalLoRASource, S3LoRASource,
};
use dynamo_llm::model_card::{LoraInfo, ModelDeploymentCard};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use serde_json::Value;

use crate::client;
use crate::proto as pb;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct LoadLoraUpdate {
    pub(crate) name: String,
    pub(crate) uri: String,
}

pub(crate) fn parse_load_lora(body: &Value) -> Result<LoadLoraUpdate, DynamoError> {
    let name = parse_lora_name(body)?;
    let uri = body
        .pointer("/source/uri")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|uri| !uri.is_empty())
        .ok_or_else(|| client::invalid_argument("source.uri must be a non-empty string"))?;
    if !["file://", "hf://", "s3://"]
        .iter()
        .any(|scheme| uri.starts_with(scheme))
    {
        return Err(client::invalid_argument(
            "source.uri must use the file://, hf://, or s3:// scheme",
        ));
    }
    if uri.starts_with("file://") && (!uri.starts_with("file:///") || uri.contains(['?', '#'])) {
        return Err(client::invalid_argument(
            "file source.uri must contain an absolute local path without a host, query, or fragment",
        ));
    }
    Ok(LoadLoraUpdate {
        name,
        uri: uri.to_string(),
    })
}

pub(crate) fn parse_lora_name(body: &Value) -> Result<String, DynamoError> {
    body.get("lora_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| client::invalid_argument("lora_name must be a non-empty string"))
}

pub(crate) fn build_downloader() -> Result<LoRADownloader, DynamoError> {
    let mut sources: Vec<Arc<dyn LoRASource>> = vec![
        Arc::new(LocalLoRASource::new()),
        Arc::new(HuggingFaceLoRASource::from_env()),
    ];
    match S3LoRASource::from_env() {
        Ok(source) => sources.push(Arc::new(source)),
        Err(error) => tracing::debug!(%error, "S3 LoRA source is not configured"),
    }
    let cache = LoRACache::from_env()
        .map_err(|error| client::invalid_argument(format!("invalid LoRA cache: {error}")))?;
    Ok(LoRADownloader::new(sources, cache))
}

pub(crate) async fn resolve_source_path(
    downloader: &LoRADownloader,
    uri: &str,
) -> Result<PathBuf, DynamoError> {
    let downloaded = downloader.download_if_needed(uri).await.map_err(|error| {
        client::invalid_argument(format!("failed to resolve LoRA source `{uri}`: {error}"))
    })?;
    let canonical = tokio::fs::canonicalize(&downloaded)
        .await
        .map_err(|error| {
            client::invalid_argument(format!(
                "failed to canonicalize LoRA directory `{}`: {error}",
                downloaded.display()
            ))
        })?;
    let valid = LoRACache::validate_path(&canonical).map_err(|error| {
        client::invalid_argument(format!(
            "failed to validate LoRA directory `{}`: {error}",
            canonical.display()
        ))
    })?;
    if !valid {
        return Err(client::invalid_argument(format!(
            "LoRA directory `{}` must contain adapter_config.json and adapter weights",
            canonical.display()
        )));
    }
    Ok(canonical)
}

pub(crate) async fn publish_lora_model(
    endpoint: &Endpoint,
    adapter: &pb::LoraAdapter,
    max_loras: u32,
) -> Result<(), DynamoError> {
    let endpoint_id = endpoint.id();
    publish_lora_model_to_discovery(
        endpoint.drt().discovery().as_ref(),
        &endpoint_id.namespace,
        &endpoint_id.component,
        &endpoint_id.name,
        endpoint.drt().connection_id(),
        adapter,
        max_loras,
    )
    .await
}

pub(crate) async fn unpublish_lora_model(
    endpoint: &Endpoint,
    lora_name: &str,
) -> Result<bool, DynamoError> {
    let endpoint_id = endpoint.id();
    unpublish_lora_model_from_discovery(
        endpoint.drt().discovery().as_ref(),
        &endpoint_id.namespace,
        &endpoint_id.component,
        &endpoint_id.name,
        endpoint.drt().connection_id(),
        lora_name,
    )
    .await
}

pub(crate) async fn unpublish_lora_model_from_discovery(
    discovery: &dyn Discovery,
    namespace: &str,
    component: &str,
    endpoint: &str,
    instance_id: u64,
    lora_name: &str,
) -> Result<bool, DynamoError> {
    let suffix = derive_lora_suffix(Some(lora_name));
    let models = discovery
        .list(DiscoveryQuery::EndpointModels {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
        })
        .await
        .map_err(|error| {
            client::protocol_error(format!("failed to query LoRA discovery: {error}"))
        })?;
    let exists = models.iter().any(|instance| {
        matches!(
            instance,
            DiscoveryInstance::Model {
                instance_id: candidate_id,
                model_suffix,
                ..
            } if *candidate_id == instance_id && *model_suffix == suffix
        )
    });
    if !exists {
        return Ok(false);
    }
    discovery
        .unregister(DiscoveryInstance::Model {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            card_json: Value::Null,
            model_suffix: suffix,
        })
        .await
        .map_err(|error| {
            client::protocol_error(format!("failed to unpublish LoRA model: {error}"))
        })?;
    Ok(true)
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
        max_gpu_lora_count: Some(max_loras),
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
