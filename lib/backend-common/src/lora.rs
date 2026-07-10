use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::lora::{LoRACache, LoRADownloader, LoRASource, LocalLoRASource, S3LoRASource};
use dynamo_llm::model_card::LoraInfo;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::utils::lora_name_to_id;
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::component::{Component, Endpoint};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Future, stream};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{DynamoError, LLMEngine, LoraAdapter};

struct ManagedLora {
    adapter: LoraAdapter,
}

pub(crate) struct LoraController {
    engine: Arc<dyn LLMEngine>,
    discovery: Arc<dyn LoraDiscovery>,
    downloader: Option<LoRADownloader>,
    loaded: Mutex<HashMap<String, ManagedLora>>,
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
    models: Mutex<HashMap<String, LocalModel>>,
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
                    max_gpu_lora_count: None,
                }),
                self.worker_type,
                self.needs.clone(),
            )
            .await
        {
            let _ = LocalModel::detach_from_endpoint(&self.endpoint, Some(&adapter.name)).await;
            return Err(error);
        }
        self.models.lock().await.insert(adapter.name.clone(), model);
        Ok(())
    }

    async fn detach(&self, name: &str) -> anyhow::Result<()> {
        LocalModel::detach_from_endpoint(&self.endpoint, Some(name)).await?;
        self.models.lock().await.remove(name);
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
        let discovery = Arc::new(ModelDiscovery {
            endpoint,
            base_model,
            model_type,
            worker_type,
            needs,
            models: Mutex::new(HashMap::new()),
        });
        Self::new_with(engine, discovery, build_downloader()).await
    }

    async fn new_with(
        engine: Arc<dyn LLMEngine>,
        discovery: Arc<dyn LoraDiscovery>,
        downloader: Option<LoRADownloader>,
    ) -> Result<Arc<Self>, DynamoError> {
        let existing = engine.list_loras().await?;
        let controller = Arc::new(Self {
            engine,
            discovery,
            downloader,
            loaded: Mutex::new(HashMap::new()),
        });

        for adapter in existing {
            controller
                .discovery
                .attach(&adapter)
                .await
                .map_err(control_error)?;
            controller
                .loaded
                .lock()
                .await
                .insert(adapter.name.clone(), ManagedLora { adapter });
        }
        Ok(controller)
    }

    async fn handle(&self, operation: Operation, request: Value) -> Value {
        let result = match operation {
            Operation::Load => self.load(request).await,
            Operation::Unload => self.unload(request).await,
            Operation::List => self.list().await,
        };
        result.unwrap_or_else(|message| json!({ "status": "error", "message": message }))
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

        let mut loaded = self.loaded.lock().await;
        let downloader = self
            .downloader
            .as_ref()
            .ok_or_else(|| "LoRA downloading is disabled; set DYN_LORA_ENABLED=true".to_string())?;
        let path = downloader
            .download_if_needed(uri)
            .await
            .map_err(|error| format!("failed to download LoRA: {error}"))?;
        let path = std::path::absolute(&path)
            .map_err(|error| format!("failed to resolve LoRA path: {error}"))?;
        let adapter = LoraAdapter {
            id: i64::from(lora_name_to_id(&name)),
            name: name.clone(),
            path: path.to_string_lossy().into_owned(),
        };
        if let Some(existing) = loaded.get(&name) {
            if existing.adapter == adapter {
                return Ok(success(&existing.adapter, "already loaded"));
            }
            return Err(format!(
                "LoRA adapter '{name}' conflicts with the loaded source path"
            ));
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
                    Err(rollback) => format!(
                        "failed to register LoRA model: {error}; rollback failed: {rollback}"
                    ),
                });
            }
        }
        let response = success(&adapter, "loaded successfully");
        loaded.insert(name, ManagedLora { adapter });
        Ok(response)
    }

    async fn unload(&self, request: Value) -> Result<Value, String> {
        let name = request
            .get("lora_name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "'lora_name' is required in request".to_string())?
            .to_string();

        let mut loaded = self.loaded.lock().await;
        let managed = loaded
            .get(&name)
            .ok_or_else(|| format!("LoRA adapter '{name}' not found"))?;
        let adapter = managed.adapter.clone();

        self.engine
            .unload_lora(&name)
            .await
            .map_err(|error| error.to_string())?;
        if let Err(error) = self.discovery.detach(&name).await {
            let rollback = self.engine.load_lora(adapter.clone()).await;
            return Err(match rollback {
                Ok(_) => format!("failed to unregister LoRA model: {error}"),
                Err(rollback) => {
                    format!("failed to unregister LoRA model: {error}; rollback failed: {rollback}")
                }
            });
        }
        loaded.remove(&name);
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

#[derive(Clone, Copy)]
enum Operation {
    Load,
    Unload,
    List,
}

struct LoraHandler {
    controller: Arc<LoraController>,
    operation: Operation,
}

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for LoraHandler {
    async fn generate(&self, input: SingleIn<Value>) -> anyhow::Result<ManyOut<Annotated<Value>>> {
        let (request, context) = input.into_parts();
        let response = self.controller.handle(self.operation, request).await;
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { Annotated::from_data(response) })),
            context.context(),
        ))
    }
}

pub(crate) fn serve_endpoints(
    component: &Component,
    controller: Arc<LoraController>,
) -> anyhow::Result<Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>> {
    let build = |name: &'static str, operation| -> anyhow::Result<_> {
        let handler = Arc::new(LoraHandler {
            controller: controller.clone(),
            operation,
        });
        let ingress = Ingress::for_engine(handler.clone())?;
        Ok(component
            .endpoint(name)
            .endpoint_builder()
            .handler(ingress)
            .register_local_engine(handler)?
            .graceful_shutdown(true)
            .start())
    };
    let load = build("load_lora", Operation::Load)?;
    let unload = build("unload_lora", Operation::Unload)?;
    let list = build("list_loras", Operation::List)?;
    Ok(Box::pin(async move {
        tokio::try_join!(load, unload, list)?;
        Ok(())
    }))
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

        async fn unload_lora(&self, name: &str) -> Result<LoraAdapter, DynamoError> {
            self.events.lock().await.push(format!("unload:{name}"));
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
        fail_detach: AtomicBool,
    }

    #[async_trait]
    impl LoraDiscovery for MockDiscovery {
        async fn attach(&self, adapter: &LoraAdapter) -> anyhow::Result<()> {
            self.attached.lock().await.push(adapter.name.clone());
            if self.fail_attach.load(Ordering::SeqCst) {
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

        let controller = LoraController::new_with(engine, discovery.clone(), None)
            .await
            .unwrap();

        assert_eq!(discovery.attached.lock().await.as_slice(), ["existing"]);
        assert_eq!(controller.list().await.unwrap()["count"], 1);
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
    async fn rolls_back_engine_load_when_discovery_attach_fails() {
        let dir = tempfile::tempdir().unwrap();
        let engine = MockEngine::new(Vec::new());
        let discovery = Arc::new(MockDiscovery::default());
        discovery.fail_attach.store(true, Ordering::SeqCst);
        let controller = LoraController::new_with(
            engine.clone(),
            discovery,
            Some(downloader(&dir.path().join("cache"))),
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
    async fn reloads_engine_when_discovery_detach_fails() {
        let dir = tempfile::tempdir().unwrap();
        let existing = adapter("adapter-a", dir.path());
        let engine = MockEngine::new(vec![existing]);
        let discovery = Arc::new(MockDiscovery::default());
        let controller = LoraController::new_with(engine.clone(), discovery.clone(), None)
            .await
            .unwrap();
        discovery.fail_detach.store(true, Ordering::SeqCst);

        let response = controller.unload(json!({ "lora_name": "adapter-a" })).await;

        assert!(
            response
                .unwrap_err()
                .contains("failed to unregister LoRA model")
        );
        assert_eq!(
            engine.events.lock().await.as_slice(),
            ["unload:adapter-a", "load:adapter-a"]
        );
        assert_eq!(controller.list().await.unwrap()["count"], 1);
    }
}
