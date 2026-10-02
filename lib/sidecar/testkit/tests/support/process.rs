// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{DisaggregationMode, LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::discovery::{ModelManager, ModelWatcher};
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::http::service::{Metrics, service_v2::HttpService};
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::namespace::NamespaceFilter;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery};
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig};
use dynamo_runtime::pipeline::{AsyncEngine, Context, ManyOut, PushRouter, RouterMode};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::storage::kv::Selector;
use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_sidecar_testkit::{bounded, fixtures};
use futures::StreamExt;
use tempfile::TempDir;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

pub type Router = PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>;

use crate::support::ProcessFixture;

pub struct Environment {
    root: TempDir,
    next_child: AtomicUsize,
    pub model: String,
    pub namespace: String,
    pub runtime: DistributedRuntime,
}

impl Environment {
    pub async fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model");
        std::fs::create_dir(&model).unwrap();
        std::fs::write(model.join("config.json"), r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"max_position_embeddings":4096,"vocab_size":32000,"bos_token_id":1,"eos_token_id":2}"#).unwrap();
        let mut vocab = serde_json::Map::from_iter(
            (0..32_000).map(|id| (format!("token{id}"), serde_json::json!(id))),
        );
        for (id, word) in [(0, "[UNK]"), (1, "hello"), (2, "world")] {
            vocab.remove(&format!("token{id}"));
            vocab.insert(word.into(), serde_json::json!(id));
        }
        std::fs::write(model.join("tokenizer.json"), serde_json::to_vec(&serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
        })).unwrap()).unwrap();
        std::fs::write(model.join("tokenizer_config.json"), r#"{"tokenizer_class":"PreTrainedTokenizerFast","model_max_length":4096,"bos_token":"hello","eos_token":"world","chat_template":"{{ messages[0]['content'] }}"}"#).unwrap();
        let namespace = format!(
            "process-{}",
            root.path()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .replace('.', "")
        );
        let runtime = DistributedRuntime::new(
            Runtime::from_current().unwrap(),
            DistributedConfig {
                discovery_backend: DiscoveryBackend::KvStore(Selector::File(
                    root.path().join("discovery"),
                )),
                ..DistributedConfig::process_local()
            },
        )
        .await
        .unwrap();
        Self {
            root,
            next_child: AtomicUsize::new(0),
            model: model.to_string_lossy().into_owned(),
            namespace,
            runtime,
        }
    }

    pub fn endpoint(&self, component: &str) -> Endpoint {
        self.runtime
            .namespace(&self.namespace)
            .unwrap()
            .component(component)
            .unwrap()
            .endpoint("generate")
    }

    pub async fn cards(&self) -> Vec<ModelDeploymentCard> {
        self.runtime
            .discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap()
            .into_iter()
            .filter_map(|instance| match &instance {
                DiscoveryInstance::Model { namespace, .. } if namespace == &self.namespace => {
                    Some(instance.deserialize_model().unwrap())
                }
                _ => None,
            })
            .collect()
    }

    pub async fn ready(&self, component: &str) -> Arc<Router> {
        let client = self.endpoint(component).client().await.unwrap();
        bounded("sidecar endpoint registration", client.wait_for_instances())
            .await
            .unwrap();
        Arc::new(
            Router::from_client(client, RouterMode::RoundRobin)
                .await
                .unwrap(),
        )
    }

    pub async fn registrations(&self, component: &str) -> Vec<DiscoveryInstance> {
        self.runtime
            .discovery()
            .list(DiscoveryQuery::Endpoint {
                namespace: self.namespace.clone(),
                component: component.to_string(),
                endpoint: "generate".to_string(),
            })
            .await
            .unwrap()
    }

    pub async fn withdrawn(&self, component: &str, router: &Router) {
        bounded("serving endpoint withdrawal", async {
            loop {
                if self.registrations(component).await.is_empty()
                    && router.selectable_worker_ids().is_err()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        let result = router.generate(self.request("after-withdrawal", 1)).await;
        assert!(
            result.is_err(),
            "withdrawn endpoint must not accept new requests"
        );
    }

    pub fn request(&self, id: &str, max_tokens: u32) -> Context<PreprocessedRequest> {
        Context::with_id_and_metadata(
            fixtures::request(&self.model, vec![11, 22, 33, 44], max_tokens),
            id.to_string(),
            Default::default(),
        )
    }

    pub fn spawn<F: ProcessFixture>(
        &self,
        endpoint: &str,
        mode: DisaggregationMode,
        deadline: u64,
    ) -> Process {
        Process::spawn::<F>(self, endpoint, mode, deadline, false, 0, None)
    }

    pub fn spawn_with_grace<F: ProcessFixture>(
        &self,
        endpoint: &str,
        mode: DisaggregationMode,
    ) -> Process {
        Process::spawn::<F>(self, endpoint, mode, 5, false, 1, None)
    }

    pub fn spawn_with_template<F: ProcessFixture>(&self, endpoint: &str) -> Process {
        let template = self.root.path().join("custom.jinja");
        std::fs::write(&template, "world {{ messages[0]['content'] }}").unwrap();
        Process::spawn::<F>(
            self,
            endpoint,
            DisaggregationMode::Aggregated,
            5,
            false,
            0,
            Some(&template),
        )
    }

    pub fn spawn_env<F: ProcessFixture>(
        &self,
        endpoint: &str,
        mode: DisaggregationMode,
        deadline: u64,
    ) -> Process {
        Process::spawn::<F>(self, endpoint, mode, deadline, true, 0, None)
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        self.runtime.shutdown();
    }
}

pub struct Process {
    child: Child,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl Process {
    fn spawn<F: ProcessFixture>(
        env: &Environment,
        endpoint: &str,
        mode: DisaggregationMode,
        deadline: u64,
        from_env: bool,
        grace_secs: u64,
        template: Option<&Path>,
    ) -> Self {
        use std::os::unix::process::CommandExt;
        let role = match mode {
            DisaggregationMode::Aggregated => "agg",
            DisaggregationMode::Prefill => "prefill",
            DisaggregationMode::Decode => "decode",
            DisaggregationMode::Encode => "encode",
        };
        let child_id = env.next_child.fetch_add(1, Ordering::Relaxed);
        let stdout = env.root.path().join(format!("{role}-{child_id}.stdout"));
        let stderr = env.root.path().join(format!("{role}-{child_id}.stderr"));
        let mut command = F::command();
        for (key, _) in std::env::vars_os() {
            let key_text = key.to_string_lossy();
            if key_text.starts_with("DYN_")
                || key_text.starts_with("NATS_")
                || key_text.starts_with("ETCD_")
            {
                command.env_remove(key);
            }
        }
        if let Some(template) = template {
            command.arg("--custom-jinja-template").arg(template);
        }
        if from_env {
            command.env("DYN_SIDECAR_GRPC_ENDPOINT", endpoint);
        } else {
            command.args(["--grpc-endpoint", endpoint]);
        }
        #[cfg(target_os = "linux")]
        {
            let parent_pid = std::process::id() as libc::pid_t;
            // SAFETY: the post-fork callback only uses syscalls and constructs an errno value.
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != parent_pid {
                        libc::_exit(1);
                    }
                    Ok(())
                });
            }
        }
        let child = command
            .args([
                "--grpc-connections",
                "1",
                "--grpc-connect-attempt-timeout-secs",
                "1",
                "--grpc-startup-deadline-secs",
                &deadline.to_string(),
                "--namespace",
                &env.namespace,
                "--component",
                "backend",
                "--disaggregation-mode",
                role,
            ])
            .env("DYN_DISCOVERY_BACKEND", "file")
            .env("DYN_FILE_KV", env.root.path().join("discovery"))
            .env("DYN_REQUEST_PLANE", "tcp")
            .env("DYN_EVENT_PLANE", "zmq")
            .env("DYN_SYSTEM_HOST", "127.0.0.1")
            .env("DYN_SYSTEM_PORT", "0")
            .env("DYN_HEALTH_CHECK_ENABLED", "false")
            .env("DYN_LOGGING_JSONL", "1")
            .env(
                "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS",
                grace_secs.to_string(),
            )
            .env("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", "2")
            .env("DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS", "3")
            .env("DYN_PREFILL_DRAIN_TIMEOUT_S", "0")
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdout(Stdio::from(std::fs::File::create(&stdout).unwrap()))
            .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()))
            .process_group(0)
            .spawn()
            .unwrap();
        Self {
            child,
            stdout,
            stderr,
        }
    }

    pub fn signal(&self, signal: i32) {
        // The child owns this process group; never signal a shared parent group.
        assert_eq!(unsafe { libc::kill(-(self.child.id() as i32), signal) }, 0);
    }

    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().unwrap().is_none()
    }

    pub async fn exit(&mut self) -> ExitStatus {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("sidecar did not exit\n{}", self.logs()))
    }

    pub fn assert_json_logs(&self, request: Option<(&str, &str)>) {
        let mut has_request = false;
        let mut count = 0;
        for path in [&self.stdout, &self.stderr] {
            for line in std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .filter(|line| !line.is_empty())
            {
                let record: serde_json::Value =
                    serde_json::from_str(line).expect("sidecar JSONL record");
                for field in ["time", "level", "target", "message"] {
                    assert!(record[field].is_string(), "missing {field}: {record}");
                }
                let is_matching_request = match request {
                    Some((id, _)) => record["request_id"] == id,
                    None => record["span_name"] == "handle_payload",
                };
                if is_matching_request {
                    assert_eq!(record["trace_id"].as_str().unwrap().len(), 32);
                    if let Some((_, expected)) = request {
                        assert_eq!(record["trace_id"], expected);
                    }
                    assert_eq!(record["span_id"].as_str().unwrap().len(), 16);
                    has_request = true;
                }
                count += 1;
            }
        }
        assert!(
            count > 0 && has_request,
            "JSONL logs must include startup and the served request"
        );
    }

    pub fn logs(&self) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            std::fs::read_to_string(&self.stdout).unwrap_or_default(),
            std::fs::read_to_string(&self.stderr).unwrap_or_default()
        )
    }

    pub async fn shutdown(&mut self) {
        self.signal(libc::SIGTERM);
        let status = self.exit().await;
        assert!(
            status.success(),
            "sidecar shutdown: {status}\n{}",
            self.logs()
        );
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("{}", self.logs());
        }
        if self.child.try_wait().ok().flatten().is_none() {
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
        }
        let _ = self.child.wait();
    }
}

pub async fn outputs(stream: ManyOut<Annotated<LLMEngineOutput>>) -> fixtures::Outputs {
    bounded(
        "Dynamo response completion",
        stream
            .filter_map(|item| async move {
                match item.into_data() {
                    Ok(Some(data)) => Some(Ok(data)),
                    Ok(None) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .collect(),
    )
    .await
}

pub struct Gate {
    pub endpoint: String,
    accepted: watch::Receiver<bool>,
    release: watch::Sender<bool>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

impl Gate {
    pub async fn new(upstream: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let upstream = upstream.strip_prefix("http://").unwrap().to_string();
        let (accepted_tx, accepted) = watch::channel(false);
        let (release, release_rx) = watch::channel(false);
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    _ = task_cancel.cancelled() => break,
                    result = listener.accept() => {
                        let (mut downstream, _) = result.unwrap();
                        accepted_tx.send_replace(true);
                        let mut released = release_rx.clone();
                        let upstream = upstream.clone();
                        connections.spawn(async move {
                            released.wait_for(|ready| *ready).await.unwrap();
                            if let Ok(mut upstream) = TcpStream::connect(upstream).await {
                                let _ = tokio::io::copy_bidirectional(&mut downstream, &mut upstream).await;
                            }
                        });
                    }
                    _ = connections.join_next(), if !connections.is_empty() => {}
                }
            }
            connections.abort_all();
            while connections.join_next().await.is_some() {}
        });
        Self {
            endpoint,
            accepted,
            release,
            cancel,
            task,
        }
    }

    pub async fn accepted(&mut self) {
        bounded(
            "native TCP connection attempt",
            self.accepted.wait_for(|accepted| *accepted),
        )
        .await
        .unwrap();
    }
    pub fn release(&self) {
        self.release.send_replace(true);
    }
    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        bounded("TCP gate shutdown", &mut self.task).await.unwrap();
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.task.abort();
    }
}

pub struct Frontend {
    pub url: String,
    pub client: reqwest::Client,
    pub metrics: Arc<Metrics>,
    manager: Arc<ModelManager>,
    cancel: CancellationToken,
    service: JoinHandle<anyhow::Result<()>>,
    watcher: JoinHandle<()>,
}

impl Frontend {
    pub async fn start(env: &Environment, migration_limit: u32) -> Self {
        Self::start_with_capabilities(env, migration_limit, Vec::new()).await
    }

    pub async fn start_with_capabilities(
        env: &Environment,
        migration_limit: u32,
        capabilities: Vec<&'static str>,
    ) -> Self {
        use opentelemetry::trace::TracerProvider;
        use tracing_subscriber::prelude::*;
        static TRACING: std::sync::Once = std::sync::Once::new();
        TRACING.call_once(|| {
            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
            let tracer = provider.tracer("sidecar-testkit");
            opentelemetry::global::set_tracer_provider(provider);
            tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .with(dynamo_runtime::logging::DistributedTraceIdLayer)
                .try_init()
                .unwrap();
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let service = HttpService::builder()
            .host("127.0.0.1")
            .port(address.port())
            .enable_chat_endpoints(true)
            .enable_cmpl_endpoints(true)
            .enable_engine_apis(!capabilities.is_empty())
            .build()
            .unwrap();
        let metrics = service.state().metrics_clone();
        let manager = service.state().manager_clone();
        let mut watcher = ModelWatcher::new(
            env.runtime.clone(),
            manager.clone(),
            RouterConfig {
                router_mode: RouterMode::RoundRobin,
                ..Default::default()
            },
            migration_limit,
            None,
            None,
            None,
            metrics.clone(),
        );
        watcher.set_generate_engine_capabilities(capabilities);
        let watcher = Arc::new(watcher);
        let discovery = env
            .runtime
            .discovery()
            .list_and_watch(DiscoveryQuery::AllModels, Some(env.runtime.primary_token()))
            .await
            .unwrap();
        let watched = watcher.clone();
        let namespace = NamespaceFilter::Exact(env.namespace.clone());
        let watcher_task = tokio::spawn(async move { watched.watch(discovery, namespace).await });
        let cancel = CancellationToken::new();
        let service = service.spawn_with_listener(cancel.clone(), listener).await;
        bounded(
            "HTTP frontend model discovery",
            watcher.wait_for_chat_model(),
        )
        .await;
        Self {
            url: format!("http://{address}"),
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            metrics,
            manager,
            cancel,
            service,
            watcher: watcher_task,
        }
    }

    pub async fn workers(&self, model: &str, expected: usize) {
        bounded("HTTP frontend worker membership", async {
            loop {
                if self
                    .manager
                    .get_model(model)
                    .is_some_and(|model| model.total_workers() == expected)
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
    }

    pub fn generate(&self, model: &str, id: &str, max_tokens: u32) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}/v1/completions", self.url))
            .header("x-dynamo-request-id", id)
            .json(&serde_json::json!({
                "model": model, "prompt": "hello", "max_tokens": max_tokens,
                "stream": true, "temperature": 0, "ignore_eos": true,
            }))
    }

    pub fn chat(&self, model: &str, id: &str) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}/v1/chat/completions", self.url))
            .header("x-dynamo-request-id", id)
            .json(&serde_json::json!({
                "model": model, "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2, "stream": true, "temperature": 0,
            }))
    }

    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        bounded("HTTP frontend shutdown", &mut self.service)
            .await
            .unwrap()
            .unwrap();
        self.watcher.abort();
        let _ = (&mut self.watcher).await;
    }
}

impl Drop for Frontend {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.service.abort();
        self.watcher.abort();
    }
}
