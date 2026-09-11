// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use dynamo_backend_common::DynamoError;
use dynamo_llm::local_model::derive_lora_suffix;
use dynamo_llm::lora::{
    HuggingFaceLoRASource, LoRACache, LoRADownloader, LoRASource, LocalLoRASource, S3LoRASource,
};
use dynamo_llm::model_card::{LoraInfo, ModelDeploymentCard};
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, DiscoverySpec};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use serde_json::Value;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::client;
use crate::proto as pb;

pub(crate) const LOAD_LORA: &str = "load_lora";
pub(crate) const UNLOAD_LORA: &str = "unload_lora";
pub(crate) const LIST_LORAS: &str = "list_loras";

pub(crate) fn is_lora_update(update: &str) -> bool {
    matches!(update, LOAD_LORA | UNLOAD_LORA | LIST_LORAS)
}

pub(crate) type LoraGuard = OwnedMutexGuard<()>;

#[derive(Default)]
pub(crate) struct LoraLifecycle {
    // Serialize inventory checks through native mutation and discovery publication.
    pub(crate) updates: Mutex<()>,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    published: Mutex<BTreeSet<String>>,
}

impl LoraLifecycle {
    pub(crate) async fn lock(&self, name: &str) -> LoraGuard {
        let lock = self
            .locks
            .lock()
            .await
            .entry(name.to_string())
            .or_default()
            .clone();
        lock.lock_owned().await
    }

    pub(crate) async fn mark_published(&self, name: &str) {
        self.published.lock().await.insert(name.to_string());
    }

    pub(crate) async fn forget(&self, name: &str) {
        self.published.lock().await.remove(name);
    }

    pub(crate) async fn is_published(&self, name: &str) -> bool {
        self.published.lock().await.contains(name)
    }

    pub(crate) async fn published_names(&self) -> Vec<String> {
        self.published.lock().await.iter().cloned().collect()
    }

    pub(crate) async fn replace_published(&self, fresh: BTreeSet<String>) -> Vec<String> {
        let mut published = self.published.lock().await;
        let stale = published.difference(&fresh).cloned().collect();
        *published = fresh;
        stale
    }
}

pub(crate) fn validate_adapter_name(
    name: &str,
    is_base_model_name: impl Fn(&str) -> bool,
    loaded: &[pb::LoraAdapter],
) -> Result<(), DynamoError> {
    if is_base_model_name(name) {
        return Err(client::invalid_argument(format!(
            "LoRA adapter `{name}` conflicts with the base model name or one of its aliases"
        )));
    }
    let Some(suffix) = derive_lora_suffix(Some(name)).filter(|suffix| !suffix.is_empty()) else {
        return Err(client::invalid_argument(format!(
            "LoRA adapter `{name}` does not produce a usable discovery suffix"
        )));
    };
    if let Some(existing) = loaded.iter().find(|adapter| {
        adapter.lora_name != name
            && derive_lora_suffix(Some(&adapter.lora_name)).as_deref() == Some(suffix.as_str())
    }) {
        return Err(client::invalid_argument(format!(
            "LoRA adapter `{name}` derives the discovery suffix `{suffix}`, which is already \
             used by loaded adapter `{}`; publishing it would overwrite that adapter's \
             discovery record",
            existing.lora_name
        )));
    }
    Ok(())
}

pub(crate) fn validate_inventory(
    adapters: Vec<pb::LoraAdapter>,
) -> Result<Vec<pb::LoraAdapter>, DynamoError> {
    let mut seen_names = HashSet::new();
    let mut seen_ids = HashSet::new();
    let mut seen_suffixes = HashSet::new();
    for adapter in &adapters {
        if adapter.lora_name.trim().is_empty() {
            return Err(client::protocol_error(
                "ListLoras returned an adapter with an empty name",
            ));
        }
        if adapter.lora_id <= 0 {
            return Err(client::protocol_error(format!(
                "ListLoras returned adapter `{}` with a non-positive id {}",
                adapter.lora_name, adapter.lora_id
            )));
        }
        if adapter.source_path.trim().is_empty() {
            return Err(client::protocol_error(format!(
                "ListLoras returned adapter `{}` without a source path",
                adapter.lora_name
            )));
        }
        if !seen_names.insert(adapter.lora_name.as_str()) {
            return Err(client::protocol_error(format!(
                "ListLoras returned duplicate adapter name `{}`",
                adapter.lora_name
            )));
        }
        if !seen_ids.insert(adapter.lora_id) {
            return Err(client::protocol_error(format!(
                "ListLoras returned duplicate adapter id {}",
                adapter.lora_id
            )));
        }
        let Some(suffix) =
            derive_lora_suffix(Some(&adapter.lora_name)).filter(|suffix| !suffix.is_empty())
        else {
            return Err(client::protocol_error(format!(
                "adapter `{}` does not produce a usable discovery suffix",
                adapter.lora_name
            )));
        };
        if !seen_suffixes.insert(suffix.clone()) {
            return Err(client::protocol_error(format!(
                "adapter `{}` derives the discovery suffix `{suffix}`, which another loaded \
                 adapter already uses",
                adapter.lora_name
            )));
        }
    }
    let mut adapters = adapters;
    adapters.sort_by(|left, right| left.lora_name.cmp(&right.lora_name));
    Ok(adapters)
}

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
    sources.push(Arc::new(S3LoRASource::from_env()));
    let cache = LoRACache::from_env()
        .map_err(|error| client::invalid_argument(format!("invalid LoRA cache: {error}")))?;
    Ok(LoRADownloader::new(sources, cache))
}

pub(crate) async fn resolve_source_path(
    downloader: &LoRADownloader,
    uri: &str,
) -> Result<PathBuf, DynamoError> {
    let downloaded = downloader.download_if_needed(uri).await.map_err(|error| {
        client::protocol_error(format!("failed to resolve LoRA source `{uri}`: {error}"))
    })?;
    let canonical = tokio::fs::canonicalize(&downloaded)
        .await
        .map_err(|error| {
            client::protocol_error(format!(
                "failed to canonicalize LoRA directory `{}`: {error}",
                downloaded.display()
            ))
        })?;
    let valid = LoRACache::validate_path(&canonical).map_err(|error| {
        client::protocol_error(format!(
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
    let discovery = endpoint.drt().discovery();
    let discovery = discovery.as_ref();
    let endpoint_id = endpoint.id();
    let namespace = endpoint_id.namespace.as_str();
    let component = endpoint_id.component.as_str();
    let endpoint_name = endpoint_id.name.as_str();
    let instance_id = endpoint.drt().connection_id();

    let models = discovery
        .list(DiscoveryQuery::EndpointModels {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint_name.to_string(),
        })
        .await
        .map_err(|error| {
            client::protocol_error(format!("failed to query base model discovery: {error}"))
        })?;
    let suffix = derive_lora_suffix(Some(&adapter.lora_name));
    for instance in &models {
        if matches!(instance, DiscoveryInstance::Model {
            instance_id: candidate_id, model_suffix, ..
        } if *candidate_id == instance_id && *model_suffix == suffix)
        {
            let card = instance
                .deserialize_model::<ModelDeploymentCard>()
                .map_err(|error| {
                    client::protocol_error(format!("invalid LoRA model card: {error}"))
                })?;
            if card.name() != adapter.lora_name {
                return Err(client::invalid_argument(format!(
                    "LoRA adapter `{}` collides with an existing discovery record",
                    adapter.lora_name
                )));
            }
        }
    }
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
        endpoint_name.to_string(),
        &card,
        suffix,
    )
    .map_err(|error| client::protocol_error(format!("failed to build LoRA model card: {error}")))?;
    discovery.register(spec).await.map_err(|error| {
        client::protocol_error(format!("failed to publish LoRA model: {error}"))
    })?;
    Ok(())
}

pub(crate) async fn unpublish_lora_model(
    endpoint: &Endpoint,
    lora_name: &str,
) -> Result<bool, DynamoError> {
    let discovery = endpoint.drt().discovery();
    let discovery = discovery.as_ref();
    let endpoint_id = endpoint.id();
    let namespace = endpoint_id.namespace.as_str();
    let component = endpoint_id.component.as_str();
    let endpoint_name = endpoint_id.name.as_str();
    let instance_id = endpoint.drt().connection_id();

    let suffix = derive_lora_suffix(Some(lora_name));
    let models = discovery
        .list(DiscoveryQuery::EndpointModels {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint_name.to_string(),
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
            endpoint: endpoint_name.to_string(),
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
