// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
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

/// Engine-update names that address the LoRA lifecycle surface.
pub(crate) const LOAD_LORA: &str = "load_lora";
pub(crate) const UNLOAD_LORA: &str = "unload_lora";
pub(crate) const LIST_LORAS: &str = "list_loras";

pub(crate) fn is_lora_update(update: &str) -> bool {
    matches!(update, LOAD_LORA | UNLOAD_LORA | LIST_LORAS)
}

/// Discovery reserves this suffix for the base-model sibling of a LoRA worker set.
const RESERVED_BASE_SUFFIX: &str = "_base";

/// Dynamo's view of one adapter.
///
/// vLLM owns whether the adapter is loaded and what its ID is; this record exists so
/// Dynamo can tie the server-assigned identity back to the source the caller asked for
/// and to the discovery record that makes the adapter routable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoraRecord {
    /// Public adapter name. The lifecycle key.
    pub(crate) name: String,
    /// Server-assigned, opaque. Never generated or inferred by Dynamo.
    pub(crate) id: i64,
    /// The URI the caller supplied.
    pub(crate) source_uri: String,
    /// Canonical local directory both Dynamo and vLLM resolve the adapter through.
    pub(crate) path: PathBuf,
    /// Whether Dynamo published the sibling discovery record.
    pub(crate) published: bool,
}

/// Held for the lifetime of one lifecycle operation or request admission on a
/// single adapter name.
pub(crate) type LoraGuard = OwnedMutexGuard<()>;

/// An in-flight capacity reservation, released when the load finishes.
///
/// The slot only covers the window between the capacity check and vLLM accepting
/// the adapter. Once `LoadLora` returns, the adapter appears in `ListLoras` and is
/// counted there instead, so holding the reservation any longer would double-count it.
pub(crate) struct CapacitySlot {
    name: String,
    reservations: Arc<Mutex<HashSet<String>>>,
}

impl Drop for CapacitySlot {
    fn drop(&mut self) {
        let name = std::mem::take(&mut self.name);
        let reservations = self.reservations.clone();
        // `try_lock` succeeds in the common case: the map is only ever held across
        // non-awaiting critical sections. Falling back to a task keeps `Drop` sync.
        if let Ok(mut guard) = reservations.try_lock() {
            guard.remove(&name);
            return;
        }
        tokio::spawn(async move {
            reservations.lock().await.remove(&name);
        });
    }
}

/// Dynamo-side LoRA lifecycle state.
///
/// Concurrency is deliberately split so that unrelated adapters never serialize
/// against each other:
///
/// - a keyed lock per adapter name orders load, unload, and request admission for
///   that one name;
/// - a short-lived global guard makes the capacity check and reservation atomic
///   across different names, and is never held across a download or an RPC.
#[derive(Default)]
pub(crate) struct LoraLifecycle {
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    capacity_guard: Mutex<()>,
    reservations: Arc<Mutex<HashSet<String>>>,
    published: Mutex<BTreeMap<String, LoraRecord>>,
}

impl LoraLifecycle {
    /// Acquire the per-adapter lock, serializing operations on `name` alone.
    pub(crate) async fn lock(&self, name: &str) -> LoraGuard {
        let lock = {
            let mut locks = self.locks.lock().await;
            // Stripes are intentionally retained rather than evicted on unload:
            // dropping one could separate a waiting request from a later
            // lifecycle operation on the same name.
            locks.entry(name.to_string()).or_default().clone()
        };
        lock.lock_owned().await
    }

    /// Reserve one GPU adapter slot, counting adapters vLLM already holds plus
    /// loads that are still in flight.
    ///
    /// The guard is released before the caller downloads or calls `LoadLora`, so
    /// concurrent loads of different adapters overlap.
    pub(crate) async fn reserve(
        &self,
        name: &str,
        loaded: usize,
        max_loras: u32,
    ) -> Result<CapacitySlot, DynamoError> {
        let _guard = self.capacity_guard.lock().await;
        let mut reservations = self.reservations.lock().await;
        let in_flight = reservations.iter().filter(|held| *held != name).count();
        if loaded + in_flight >= max_loras as usize {
            return Err(client::invalid_argument(format!(
                "LoRA capacity exceeded: at most {max_loras} adapter(s) may be loaded"
            )));
        }
        reservations.insert(name.to_string());
        Ok(CapacitySlot {
            name: name.to_string(),
            reservations: self.reservations.clone(),
        })
    }

    pub(crate) async fn mark_published(&self, record: LoraRecord) {
        self.published
            .lock()
            .await
            .insert(record.name.clone(), record);
    }

    pub(crate) async fn forget(&self, name: &str) -> Option<LoraRecord> {
        self.published.lock().await.remove(name)
    }

    /// Whether Dynamo currently advertises `name` as routable.
    pub(crate) async fn is_published(&self, name: &str) -> bool {
        self.published.lock().await.contains_key(name)
    }

    /// Names Dynamo believes it has published, for shutdown cleanup.
    pub(crate) async fn published_names(&self) -> Vec<String> {
        self.published.lock().await.keys().cloned().collect()
    }

    /// Replace the published set with `records`, returning names that were
    /// tracked before but are absent now (stale Dynamo-only records).
    pub(crate) async fn replace_published(&self, records: Vec<LoraRecord>) -> Vec<String> {
        let mut published = self.published.lock().await;
        let fresh: BTreeMap<String, LoraRecord> = records
            .into_iter()
            .map(|record| (record.name.clone(), record))
            .collect();
        let stale = published
            .keys()
            .filter(|name| !fresh.contains_key(*name))
            .cloned()
            .collect();
        *published = fresh;
        stale
    }
}

/// Reject adapter names that would collide with the base model or with discovery's
/// own naming, before any state is mutated.
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
    // Discovery keys on the *derived* suffix, not the raw name, so every check below
    // has to be made on the suffix. `Slug::slugify` lowercases and rewrites anything
    // outside [a-z0-9-_] to `-`, which means distinct adapter names routinely collapse
    // onto one key: `Math-R8` and `math.r8` both become `math-r8`.
    let Some(suffix) = derive_lora_suffix(Some(name)).filter(|suffix| !suffix.is_empty()) else {
        return Err(client::invalid_argument(format!(
            "LoRA adapter `{name}` does not produce a usable discovery suffix"
        )));
    };
    // Defense in depth: `Slug` trims leading underscores, so a slug of exactly `_base`
    // is not currently reachable. The sentinel is upstream's, and this keeps the
    // invariant local if that ever changes.
    if suffix == RESERVED_BASE_SUFFIX {
        return Err(client::invalid_argument(format!(
            "LoRA adapter `{name}` derives the reserved `{RESERVED_BASE_SUFFIX}` discovery \
             suffix, which identifies the base-model sibling"
        )));
    }
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

/// Validate one `ListLoras` inventory and return it sorted by name.
///
/// vLLM is authoritative for what is loaded, but Dynamo still refuses to build
/// routing state out of an inventory that cannot be keyed unambiguously.
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
        // Distinct names can still collapse onto one discovery key, which would make
        // reconciliation publish two adapters over the same sibling record.
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

/// Compare a path reported by vLLM against the directory Dynamo resolved.
///
/// Both sides must see the adapter at the same absolute path through their shared
/// mount, so a mismatch means the deployment is misconfigured or the name was
/// loaded from somewhere else.
pub(crate) fn paths_agree(reported: &str, resolved: &Path) -> bool {
    Path::new(reported) == resolved
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
    // `S3LoRASource::from_env` became infallible in #13844: it defers endpoint and
    // credential resolution to a `OnceCell` so a missing S3 configuration surfaces
    // when an `s3://` source is actually resolved, not at construction.
    sources.push(Arc::new(S3LoRASource::from_env()));
    let cache = LoRACache::from_env()
        .map_err(|error| client::invalid_argument(format!("invalid LoRA cache: {error}")))?;
    Ok(LoRADownloader::new(sources, cache))
}

pub(crate) async fn resolve_source_path(
    downloader: &LoRADownloader,
    uri: &str,
) -> Result<PathBuf, DynamoError> {
    // The URI itself was already validated by `parse_load_lora`, so a failure here is
    // a download or storage fault rather than bad caller input.
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

/// Publish the sibling record, failing if another instance already advertises the name.
///
/// `allow_existing` is set during restart reconciliation and idempotent loads, where
/// re-publishing our own record is the intended outcome.
pub(crate) async fn publish_lora_model(
    endpoint: &Endpoint,
    adapter: &pb::LoraAdapter,
    max_loras: u32,
    allow_existing: bool,
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
    if !allow_existing {
        let collision = models.iter().any(|instance| {
            matches!(
                instance,
                DiscoveryInstance::Model {
                    instance_id: candidate_id,
                    model_suffix,
                    ..
                } if *model_suffix == suffix && *candidate_id != instance_id
            )
        });
        if collision {
            return Err(client::invalid_argument(format!(
                "LoRA adapter `{}` collides with an existing discovery record",
                adapter.lora_name
            )));
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
    // Derive the sibling from the base card so the routing topology carries over
    // verbatim: model and worker type, prefill/decode `needs`, the data-parallel rank
    // range, router hints, context length and token budget, KV-event configuration,
    // tool and reasoning parsers, and any encoder dependency. Only the adapter-specific
    // fields below may differ.
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

/// Remove the sibling record for `lora_name`, reporting whether one existed.
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
