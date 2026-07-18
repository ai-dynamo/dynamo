// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex, Weak};

use async_trait::async_trait;
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::lora::{LoRACache, LoRADownloader, LoRASource, LocalLoRASource, S3LoRASource};
use dynamo_llm::model_card::LoraInfo;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::utils::lora_name_to_id;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::engine_routes::EngineRouteCallback;
use dynamo_runtime::prelude::DistributedRuntimeProvider;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{DynamoError, LLMEngine, LoraAdapter};

#[derive(Clone)]
struct ManagedLora {
    adapter: LoraAdapter,
    published: bool,
}

pub(crate) struct LoraController {
    engine: Arc<dyn LLMEngine>,
    discovery: Arc<dyn LoraDiscovery>,
    downloader: Option<LoRADownloader>,
    reserved_names: HashSet<String>,
    loaded: Mutex<HashMap<String, ManagedLora>>,
    operations: StdMutex<HashMap<String, Weak<Mutex<()>>>>,
}

#[async_trait]
trait LoraDiscovery: Send + Sync {
    async fn attach(&self, adapter: &LoraAdapter) -> anyhow::Result<()>;
    async fn detach(&self, name: &str) -> anyhow::Result<()>;
}

struct ModelDiscovery {
    endpoint: Endpoint,
    base_model: LocalModel,
    model_type: ModelType,
    worker_type: Option<WorkerType>,
    needs: Vec<Vec<WorkerType>>,
}

#[async_trait]
impl LoraDiscovery for ModelDiscovery {
    async fn attach(&self, adapter: &LoraAdapter) -> anyhow::Result<()> {
        let mut model = self.base_model.clone();
        model.set_name(&adapter.name);
        if let Err(error) = model
            .attach(
                &self.endpoint,
                self.model_type,
                ModelInput::Tokens,
                Some(LoraInfo {
                    name: adapter.name.clone(),
                    max_gpu_lora_count: self.base_model.runtime_config().max_gpu_lora_count,
                }),
                self.worker_type,
                self.needs.clone(),
            )
            .await
        {
            let _ = LocalModel::detach_from_endpoint(&self.endpoint, Some(&adapter.name)).await;
            return Err(error);
        }
        Ok(())
    }

    async fn detach(&self, name: &str) -> anyhow::Result<()> {
        LocalModel::detach_from_endpoint(&self.endpoint, Some(name)).await?;
        Ok(())
    }
}

impl LoraController {
    pub(crate) async fn new(
        engine: Arc<dyn LLMEngine>,
        endpoint: Endpoint,
        base_model: LocalModel,
        model_type: ModelType,
        worker_type: Option<WorkerType>,
        needs: Vec<Vec<WorkerType>>,
    ) -> Result<Arc<Self>, DynamoError> {
        let mut reserved_names = HashSet::from([base_model.display_name().to_string()]);
        if let Some(source_path) = base_model.card().source_path.as_ref() {
            reserved_names.insert(source_path.clone());
        }
        let discovery = Arc::new(ModelDiscovery {
            endpoint,
            base_model,
            model_type,
            worker_type,
            needs,
        });
        Self::new_with(engine, discovery, build_downloader(), reserved_names).await
    }

    async fn new_with(
        engine: Arc<dyn LLMEngine>,
        discovery: Arc<dyn LoraDiscovery>,
        downloader: Option<LoRADownloader>,
        reserved_names: HashSet<String>,
    ) -> Result<Arc<Self>, DynamoError> {
        let mut existing = engine.list_loras().await?;
        existing.sort_by(|left, right| left.name.cmp(&right.name));
        for adapter in &existing {
            validate_adapter_name(&reserved_names, &adapter.name)
                .map_err(|message| control_error(anyhow::anyhow!(message)))?;
            let expected_id = i64::from(lora_name_to_id(&adapter.name));
            if adapter.id != expected_id {
                return Err(control_error(anyhow::anyhow!(
                    "LoRA `{}` has engine id {}, expected deterministic id {expected_id}",
                    adapter.name,
                    adapter.id
                )));
            }
        }

        let mut loaded = HashMap::new();
        let mut attached: Vec<String> = Vec::new();
        for adapter in existing {
            if let Err(error) = discovery.attach(&adapter).await {
                let mut rollback_errors = Vec::new();
                for name in attached.iter().rev() {
                    if let Err(rollback) = discovery.detach(name).await {
                        rollback_errors.push(format!("{name}: {rollback}"));
                    }
                }
                let message = if rollback_errors.is_empty() {
                    format!("failed to reconcile LoRA `{}`: {error}", adapter.name)
                } else {
                    format!(
                        "failed to reconcile LoRA `{}`: {error}; rollback failed: {}",
                        adapter.name,
                        rollback_errors.join(", ")
                    )
                };
                return Err(control_error(anyhow::anyhow!(message)));
            }
            attached.push(adapter.name.clone());
            loaded.insert(
                adapter.name.clone(),
                ManagedLora {
                    adapter,
                    published: true,
                },
            );
        }
        Ok(Arc::new(Self {
            engine,
            discovery,
            downloader,
            reserved_names,
            loaded: Mutex::new(loaded),
            operations: StdMutex::new(HashMap::new()),
        }))
    }

    fn operation_lock(&self, name: &str) -> Arc<Mutex<()>> {
        let mut operations = self.operations.lock().expect("operation lock map poisoned");
        operations.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = operations.get(name).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(Mutex::new(()));
        operations.insert(name.to_string(), Arc::downgrade(&lock));
        lock
    }

    async fn load(&self, request: Value) -> Result<Value, String> {
        let name = request
            .get("lora_name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "'lora_name' is required in request".to_string())?
            .to_string();
        let uri = request
            .get("source")
            .and_then(Value::as_object)
            .and_then(|source| source.get("uri"))
            .and_then(Value::as_str)
            .filter(|uri| !uri.trim().is_empty())
            .ok_or_else(|| "'source.uri' is required in request".to_string())?;
        let load_inplace = match request.get("load_inplace") {
            None => false,
            Some(value) => value
                .as_bool()
                .ok_or_else(|| "'load_inplace' must be a boolean".to_string())?,
        };
        validate_adapter_name(&self.reserved_names, &name)?;

        let operation = self.operation_lock(&name);
        let _operation = operation.lock().await;
        let downloader = self
            .downloader
            .as_ref()
            .ok_or_else(|| "LoRA downloading is disabled; set DYN_LORA_ENABLED=true".to_string())?;
        let path = downloader
            .download_if_needed(uri)
            .await
            .map_err(|error| format!("failed to download LoRA: {error}"))?;
        let path = tokio::fs::canonicalize(&path)
            .await
            .map_err(|error| format!("failed to resolve LoRA path: {error}"))?;
        let metadata = tokio::fs::metadata(&path)
            .await
            .map_err(|error| format!("failed to inspect LoRA path: {error}"))?;
        if !metadata.is_dir() {
            return Err("LoRA source path must resolve to a directory".to_string());
        }
        let adapter = LoraAdapter {
            id: i64::from(lora_name_to_id(&name)),
            name: name.clone(),
            path: path.to_string_lossy().into_owned(),
        };
        let existing = { self.loaded.lock().await.get(&name).cloned() };
        if let Some(existing) = existing {
            if existing.adapter == adapter && !load_inplace {
                if !existing.published {
                    self.discovery
                        .attach(&existing.adapter)
                        .await
                        .map_err(|error| format!("failed to register LoRA model: {error}"))?;
                    self.loaded
                        .lock()
                        .await
                        .get_mut(&name)
                        .expect("adapter operation is serialized")
                        .published = true;
                }
                return Ok(success(&existing.adapter, "already loaded"));
            }
            if !load_inplace {
                return Err(format!(
                    "LoRA adapter '{name}' conflicts with the loaded source path"
                ));
            }
            let adapter = self
                .engine
                .load_lora_inplace(adapter)
                .await
                .map_err(|error| error.to_string())?;
            let published = if existing.published {
                true
            } else {
                match self.discovery.attach(&adapter).await {
                    Ok(()) => true,
                    Err(error) => {
                        self.loaded.lock().await.insert(
                            name,
                            ManagedLora {
                                adapter,
                                published: false,
                            },
                        );
                        return Err(format!("failed to register replaced LoRA model: {error}"));
                    }
                }
            };
            self.loaded.lock().await.insert(
                name,
                ManagedLora {
                    adapter: adapter.clone(),
                    published,
                },
            );
            return Ok(success(&adapter, "replaced successfully"));
        }

        let adapter = self
            .engine
            .load_lora(adapter)
            .await
            .map_err(|error| error.to_string())?;
        match self.discovery.attach(&adapter).await {
            Ok(()) => {}
            Err(error) => {
                let rollback = self.engine.unload_lora(&adapter.name).await;
                return Err(match rollback {
                    Ok(_) => format!("failed to register LoRA model: {error}"),
                    Err(rollback) => {
                        self.loaded.lock().await.insert(
                            name,
                            ManagedLora {
                                adapter,
                                published: false,
                            },
                        );
                        format!(
                            "failed to register LoRA model: {error}; rollback failed: {rollback}"
                        )
                    }
                });
            }
        }
        let response = success(&adapter, "loaded successfully");
        self.loaded.lock().await.insert(
            name,
            ManagedLora {
                adapter,
                published: true,
            },
        );
        Ok(response)
    }

    async fn unload(&self, request: Value) -> Result<Value, String> {
        let name = request
            .get("lora_name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "'lora_name' is required in request".to_string())?
            .to_string();

        let operation = self.operation_lock(&name);
        let _operation = operation.lock().await;
        let managed = self
            .loaded
            .lock()
            .await
            .get(&name)
            .cloned()
            .ok_or_else(|| format!("LoRA adapter '{name}' not found"))?;
        let adapter = managed.adapter.clone();
        let was_published = managed.published;

        if was_published {
            self.discovery
                .detach(&name)
                .await
                .map_err(|error| format!("failed to unregister LoRA model: {error}"))?;
            self.loaded
                .lock()
                .await
                .get_mut(&name)
                .expect("adapter operation is serialized")
                .published = false;
        }
        if let Err(error) = self.engine.unload_lora(&name).await {
            let rollback = if was_published {
                self.discovery.attach(&adapter).await
            } else {
                Ok(())
            };
            return Err(match rollback {
                Ok(()) => {
                    self.loaded
                        .lock()
                        .await
                        .get_mut(&name)
                        .expect("adapter operation is serialized")
                        .published = was_published;
                    error.to_string()
                }
                Err(rollback) => format!(
                    "failed to unload LoRA from engine: {error}; discovery rollback failed: {rollback}"
                ),
            });
        }
        self.loaded.lock().await.remove(&name);
        Ok(success(&adapter, "unloaded successfully"))
    }

    async fn list(&self) -> Result<Value, String> {
        let loaded = self.loaded.lock().await;
        let mut names = loaded.keys().cloned().collect::<Vec<_>>();
        names.sort();
        let loras = names
            .into_iter()
            .map(|name| (name.clone(), Value::from(loaded[&name].adapter.id)))
            .collect::<serde_json::Map<_, _>>();
        Ok(json!({ "status": "success", "count": loras.len(), "loras": loras }))
    }
}

fn validate_adapter_name(reserved_names: &HashSet<String>, name: &str) -> Result<(), String> {
    if reserved_names.contains(name) {
        return Err(format!(
            "LoRA adapter name '{name}' conflicts with the served base model"
        ));
    }
    Ok(())
}

fn build_downloader() -> Option<LoRADownloader> {
    if !dynamo_runtime::config::env_is_truthy("DYN_LORA_ENABLED") {
        return None;
    }
    let cache = match LoRACache::from_env() {
        Ok(cache) => cache,
        Err(error) => {
            tracing::warn!(%error, "failed to configure LoRA cache");
            return None;
        }
    };
    let mut sources: Vec<Arc<dyn LoRASource>> = vec![Arc::new(LocalLoRASource::new())];
    if let Ok(source) = S3LoRASource::from_env() {
        sources.push(Arc::new(source));
    }
    Some(LoRADownloader::new(sources, cache))
}

fn success(adapter: &LoraAdapter, action: &str) -> Value {
    json!({
        "status": "success",
        "message": format!("LoRA adapter '{}' {action}", adapter.name),
        "lora_name": adapter.name,
        "lora_id": adapter.id,
    })
}

fn control_error(error: anyhow::Error) -> DynamoError {
    DynamoError::builder()
        .error_type(crate::ErrorType::Backend(crate::BackendError::Unknown))
        .message(error.to_string())
        .build()
}

pub(crate) fn register_updates(endpoint: &Endpoint, controller: Arc<LoraController>) {
    let registry = endpoint.drt().engine_routes();

    let load: EngineRouteCallback = Arc::new({
        let controller = controller.clone();
        move |body| {
            let controller = controller.clone();
            Box::pin(async move {
                if !body.is_object() {
                    return Ok(json!({
                        "status": "error",
                        "message": "engine update request body must be a JSON object",
                    }));
                }
                Ok(control_response(controller.load(body).await))
            })
        }
    });
    registry.register("update/load_lora", load);

    let unload: EngineRouteCallback = Arc::new({
        let controller = controller.clone();
        move |body| {
            let controller = controller.clone();
            Box::pin(async move {
                if !body.is_object() {
                    return Ok(json!({
                        "status": "error",
                        "message": "engine update request body must be a JSON object",
                    }));
                }
                Ok(control_response(controller.unload(body).await))
            })
        }
    });
    registry.register("update/unload_lora", unload);

    let list: EngineRouteCallback = Arc::new(move |_body| {
        let controller = controller.clone();
        Box::pin(async move { Ok(control_response(controller.list().await)) })
    });
    registry.register("update/list_loras", list);
}

fn control_response(result: Result<Value, String>) -> Value {
    result.unwrap_or_else(|message| json!({ "status": "error", "message": message }))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures::stream::BoxStream;

    use super::*;
    use crate::engine::{EngineConfig, GenerateContext, LLMEngineOutput, PreprocessedRequest};

    struct MockEngine {
        loaded: Mutex<HashMap<String, LoraAdapter>>,
        events: Mutex<Vec<String>>,
        fail_unload: AtomicBool,
    }

    impl MockEngine {
        fn new(adapters: Vec<LoraAdapter>) -> Arc<Self> {
            Arc::new(Self {
                loaded: Mutex::new(
                    adapters
                        .into_iter()
                        .map(|adapter| (adapter.name.clone(), adapter))
                        .collect(),
                ),
                events: Mutex::new(Vec::new()),
                fail_unload: AtomicBool::new(false),
            })
        }
    }

    #[async_trait]
    impl LLMEngine for MockEngine {
        async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
            Ok(EngineConfig::default())
        }

        async fn generate(
            &self,
            _request: PreprocessedRequest,
            _ctx: GenerateContext,
        ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
            unreachable!("generation is not used by LoRA controller tests")
        }

        async fn load_lora(&self, adapter: LoraAdapter) -> Result<LoraAdapter, DynamoError> {
            self.events
                .lock()
                .await
                .push(format!("load:{}", adapter.name));
            self.loaded
                .lock()
                .await
                .insert(adapter.name.clone(), adapter.clone());
            Ok(adapter)
        }

        async fn load_lora_inplace(
            &self,
            adapter: LoraAdapter,
        ) -> Result<LoraAdapter, DynamoError> {
            self.load_lora(adapter).await
        }

        async fn unload_lora(&self, name: &str) -> Result<LoraAdapter, DynamoError> {
            self.events.lock().await.push(format!("unload:{name}"));
            if self.fail_unload.load(Ordering::SeqCst) {
                return Err(control_error(anyhow::anyhow!("unload failed")));
            }
            Ok(self
                .loaded
                .lock()
                .await
                .remove(name)
                .expect("test adapter is loaded"))
        }

        async fn list_loras(&self) -> Result<Vec<LoraAdapter>, DynamoError> {
            Ok(self.loaded.lock().await.values().cloned().collect())
        }

        async fn cleanup(&self) -> Result<(), DynamoError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct MockDiscovery {
        attached: Mutex<Vec<String>>,
        detached: Mutex<Vec<String>>,
        fail_attach: AtomicBool,
        fail_attach_name: Mutex<Option<String>>,
        fail_detach: AtomicBool,
    }

    #[async_trait]
    impl LoraDiscovery for MockDiscovery {
        async fn attach(&self, adapter: &LoraAdapter) -> anyhow::Result<()> {
            self.attached.lock().await.push(adapter.name.clone());
            if self.fail_attach.load(Ordering::SeqCst)
                || self.fail_attach_name.lock().await.as_deref() == Some(&adapter.name)
            {
                anyhow::bail!("attach failed");
            }
            Ok(())
        }

        async fn detach(&self, name: &str) -> anyhow::Result<()> {
            self.detached.lock().await.push(name.to_string());
            if self.fail_detach.load(Ordering::SeqCst) {
                anyhow::bail!("detach failed");
            }
            Ok(())
        }
    }

    fn adapter(name: &str, path: &std::path::Path) -> LoraAdapter {
        LoraAdapter {
            id: i64::from(lora_name_to_id(name)),
            name: name.to_string(),
            path: path.to_string_lossy().into_owned(),
        }
    }

    fn downloader(cache: &std::path::Path) -> LoRADownloader {
        LoRADownloader::new(
            vec![Arc::new(LocalLoRASource::new())],
            LoRACache::new(cache.to_path_buf()),
        )
    }

    #[tokio::test]
    async fn reconciles_engine_adapters_into_discovery() {
        let dir = tempfile::tempdir().unwrap();
        let existing = adapter("existing", dir.path());
        let engine = MockEngine::new(vec![existing]);
        let discovery = Arc::new(MockDiscovery::default());

        let controller = LoraController::new_with(engine, discovery.clone(), None, HashSet::new())
            .await
            .unwrap();

        assert_eq!(discovery.attached.lock().await.as_slice(), ["existing"]);
        assert_eq!(controller.list().await.unwrap()["count"], 1);
    }

    #[tokio::test]
    async fn reconciliation_rolls_back_earlier_discovery_attachments() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MockEngine::new(vec![adapter("a", dir.path()), adapter("b", dir.path())]);
        let discovery = Arc::new(MockDiscovery::default());
        *discovery.fail_attach_name.lock().await = Some("b".to_string());

        let result =
            LoraController::new_with(engine, discovery.clone(), None, HashSet::new()).await;

        assert!(result.is_err());
        assert_eq!(discovery.attached.lock().await.as_slice(), ["a", "b"]);
        assert_eq!(discovery.detached.lock().await.as_slice(), ["a"]);
    }

    #[tokio::test]
    async fn reconciliation_rejects_base_model_name_collision() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MockEngine::new(vec![adapter("base-model", dir.path())]);
        let discovery = Arc::new(MockDiscovery::default());

        let result = LoraController::new_with(
            engine,
            discovery.clone(),
            None,
            HashSet::from(["base-model".to_string()]),
        )
        .await;

        assert!(result.is_err());
        assert!(discovery.attached.lock().await.is_empty());
    }

    #[tokio::test]
    async fn reconciliation_rejects_non_deterministic_adapter_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut existing = adapter("existing", dir.path());
        existing.id += 1;
        let engine = MockEngine::new(vec![existing]);
        let discovery = Arc::new(MockDiscovery::default());

        let result =
            LoraController::new_with(engine, discovery.clone(), None, HashSet::new()).await;

        assert!(result.is_err());
        assert!(discovery.attached.lock().await.is_empty());
    }

    #[tokio::test]
    async fn reconciliation_validates_all_adapters_before_publishing_any() {
        let dir = tempfile::tempdir().unwrap();
        let valid = adapter("a-valid", dir.path());
        let mut invalid = adapter("z-invalid", dir.path());
        invalid.id += 1;
        let engine = MockEngine::new(vec![valid, invalid]);
        let discovery = Arc::new(MockDiscovery::default());

        let result =
            LoraController::new_with(engine, discovery.clone(), None, HashSet::new()).await;

        assert!(result.is_err());
        assert!(discovery.attached.lock().await.is_empty());
    }

    #[tokio::test]
    async fn loads_with_deterministic_id_and_registers_discovery() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MockEngine::new(Vec::new());
        let discovery = Arc::new(MockDiscovery::default());
        let controller = LoraController::new_with(
            engine.clone(),
            discovery.clone(),
            Some(downloader(&dir.path().join("cache"))),
            HashSet::new(),
        )
        .await
        .unwrap();

        let response = controller
            .load(json!({
                "lora_name": "adapter-a",
                "source": { "uri": format!("file://{}", dir.path().display()) }
            }))
            .await
            .unwrap();

        assert_eq!(response["lora_id"], i64::from(lora_name_to_id("adapter-a")));
        assert_eq!(discovery.attached.lock().await.as_slice(), ["adapter-a"]);
        assert_eq!(
            engine.loaded.lock().await["adapter-a"].id,
            i64::from(lora_name_to_id("adapter-a"))
        );
    }

    #[tokio::test]
    async fn load_inplace_replaces_a_stable_name_with_a_new_source() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("step-1");
        let second = root.path().join("step-2");
        std::fs::create_dir(&first).unwrap();
        std::fs::create_dir(&second).unwrap();
        let engine = MockEngine::new(Vec::new());
        let discovery = Arc::new(MockDiscovery::default());
        let controller = LoraController::new_with(
            engine.clone(),
            discovery.clone(),
            Some(downloader(&root.path().join("cache"))),
            HashSet::new(),
        )
        .await
        .unwrap();

        for (path, load_inplace) in [(&first, false), (&second, true)] {
            controller
                .load(json!({
                    "lora_name": "adapter-a",
                    "source": { "uri": format!("file://{}", path.display()) },
                    "load_inplace": load_inplace,
                }))
                .await
                .unwrap();
        }

        assert_eq!(
            engine.loaded.lock().await["adapter-a"].path,
            std::fs::canonicalize(second).unwrap().to_string_lossy()
        );
        assert_eq!(discovery.attached.lock().await.as_slice(), ["adapter-a"]);
    }

    #[tokio::test]
    async fn load_inplace_reloads_the_same_stable_source() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("stable");
        std::fs::create_dir(&source).unwrap();
        let engine = MockEngine::new(Vec::new());
        let controller = LoraController::new_with(
            engine.clone(),
            Arc::new(MockDiscovery::default()),
            Some(downloader(&root.path().join("cache"))),
            HashSet::new(),
        )
        .await
        .unwrap();

        for load_inplace in [false, true] {
            controller
                .load(json!({
                    "lora_name": "adapter-a",
                    "source": { "uri": format!("file://{}", source.display()) },
                    "load_inplace": load_inplace,
                }))
                .await
                .unwrap();
        }

        assert_eq!(
            engine.events.lock().await.as_slice(),
            ["load:adapter-a", "load:adapter-a"]
        );
    }

    #[tokio::test]
    async fn load_rejects_base_model_name_before_engine_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MockEngine::new(Vec::new());
        let controller = LoraController::new_with(
            engine.clone(),
            Arc::new(MockDiscovery::default()),
            Some(downloader(&dir.path().join("cache"))),
            HashSet::from(["base-model".to_string()]),
        )
        .await
        .unwrap();

        let error = controller
            .load(json!({
                "lora_name": "base-model",
                "source": { "uri": format!("file://{}", dir.path().display()) }
            }))
            .await
            .unwrap_err();

        assert!(error.contains("conflicts with the served base model"));
        assert!(engine.events.lock().await.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn canonicalizes_symlinked_local_source_before_engine_load() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("adapter-target");
        std::fs::create_dir(&target).unwrap();
        let alias = dir.path().join("adapter-alias");
        symlink(&target, &alias).unwrap();

        let engine = MockEngine::new(Vec::new());
        let controller = LoraController::new_with(
            engine.clone(),
            Arc::new(MockDiscovery::default()),
            Some(downloader(&dir.path().join("cache"))),
            HashSet::new(),
        )
        .await
        .unwrap();

        controller
            .load(json!({
                "lora_name": "adapter-a",
                "source": { "uri": format!("file://{}", alias.display()) }
            }))
            .await
            .unwrap();

        assert_eq!(
            engine.loaded.lock().await["adapter-a"].path,
            std::fs::canonicalize(target)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        );
    }

    #[tokio::test]
    async fn rolls_back_engine_load_when_discovery_attach_fails() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MockEngine::new(Vec::new());
        let discovery = Arc::new(MockDiscovery::default());
        discovery.fail_attach.store(true, Ordering::SeqCst);
        let controller = LoraController::new_with(
            engine.clone(),
            discovery,
            Some(downloader(&dir.path().join("cache"))),
            HashSet::new(),
        )
        .await
        .unwrap();

        let response = controller
            .load(json!({
                "lora_name": "adapter-a",
                "source": { "uri": format!("file://{}", dir.path().display()) }
            }))
            .await;

        assert!(
            response
                .unwrap_err()
                .contains("failed to register LoRA model")
        );
        assert_eq!(
            engine.events.lock().await.as_slice(),
            ["load:adapter-a", "unload:adapter-a"]
        );
    }

    #[tokio::test]
    async fn leaves_engine_loaded_when_discovery_detach_fails() {
        let dir = tempfile::tempdir().unwrap();
        let existing = adapter("adapter-a", dir.path());
        let engine = MockEngine::new(vec![existing]);
        let discovery = Arc::new(MockDiscovery::default());
        let controller =
            LoraController::new_with(engine.clone(), discovery.clone(), None, HashSet::new())
                .await
                .unwrap();
        discovery.fail_detach.store(true, Ordering::SeqCst);

        let response = controller.unload(json!({ "lora_name": "adapter-a" })).await;

        assert!(
            response
                .unwrap_err()
                .contains("failed to unregister LoRA model")
        );
        assert!(engine.events.lock().await.is_empty());
        assert_eq!(controller.list().await.unwrap()["count"], 1);
    }

    #[tokio::test]
    async fn reattaches_discovery_when_engine_unload_fails() {
        let dir = tempfile::tempdir().unwrap();
        let existing = adapter("adapter-a", dir.path());
        let engine = MockEngine::new(vec![existing]);
        engine.fail_unload.store(true, Ordering::SeqCst);
        let discovery = Arc::new(MockDiscovery::default());
        let controller =
            LoraController::new_with(engine.clone(), discovery.clone(), None, HashSet::new())
                .await
                .unwrap();

        let response = controller.unload(json!({ "lora_name": "adapter-a" })).await;

        assert!(response.unwrap_err().contains("unload failed"));
        assert_eq!(engine.events.lock().await.as_slice(), ["unload:adapter-a"]);
        assert_eq!(
            discovery.attached.lock().await.as_slice(),
            ["adapter-a", "adapter-a"]
        );
        assert_eq!(discovery.detached.lock().await.as_slice(), ["adapter-a"]);
        assert_eq!(controller.list().await.unwrap()["count"], 1);
    }

    #[tokio::test]
    async fn operation_locks_do_not_accumulate_failed_names() {
        let controller = LoraController::new_with(
            MockEngine::new(Vec::new()),
            Arc::new(MockDiscovery::default()),
            None,
            HashSet::new(),
        )
        .await
        .unwrap();

        for index in 0..100 {
            drop(controller.operation_lock(&format!("missing-{index}")));
        }
        assert!(controller.operations.lock().unwrap().len() <= 1);
    }
}
