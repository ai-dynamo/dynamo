// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use dynamo_backend_common::{DisaggregationMode, LLMEngineOutput, PreprocessedRequest};
use dynamo_llm::model_card::ModelDeploymentCard;
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

pub trait ProcessFixture: crate::support::WireFixture {
    fn endpoint(&self) -> String;
    fn command() -> Command;
    fn configure_request(request: &mut PreprocessedRequest);
    fn assert_registration(card: &ModelDeploymentCard);
}

pub fn sidecar_command(binary_name: &str, override_env: &str) -> Command {
    let binary = std::env::var_os(override_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join(binary_name)
        });
    assert!(
        binary.is_file(),
        "build {binary_name} first or set {override_env}: {}",
        binary.display()
    );
    Command::new(binary)
}

pub struct Environment {
    root: TempDir,
    pub model: String,
    pub namespace: String,
    pub runtime: DistributedRuntime,
}

impl Environment {
    pub async fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model");
        std::fs::create_dir(&model).unwrap();
        std::fs::write(model.join("config.json"), r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"max_position_embeddings":4096,"vocab_size":256,"bos_token_id":1,"eos_token_id":2}"#).unwrap();
        std::fs::write(model.join("tokenizer.json"), r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#).unwrap();
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
        Process::spawn::<F>(self, endpoint, mode, deadline, false, 0)
    }

    pub fn spawn_with_grace<F: ProcessFixture>(
        &self,
        endpoint: &str,
        mode: DisaggregationMode,
    ) -> Process {
        Process::spawn::<F>(self, endpoint, mode, 5, false, 1)
    }

    pub fn spawn_env<F: ProcessFixture>(
        &self,
        endpoint: &str,
        mode: DisaggregationMode,
        deadline: u64,
    ) -> Process {
        Process::spawn::<F>(self, endpoint, mode, deadline, true, 0)
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
    ) -> Self {
        use std::os::unix::process::CommandExt;
        let role = match mode {
            DisaggregationMode::Aggregated => "agg",
            DisaggregationMode::Prefill => "prefill",
            DisaggregationMode::Decode => "decode",
            DisaggregationMode::Encode => "encode",
        };
        let stdout = env.root.path().join(format!("{role}.stdout"));
        let stderr = env.root.path().join(format!("{role}.stderr"));
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
        if from_env {
            command.env("DYN_SIDECAR_GRPC_ENDPOINT", endpoint);
        } else {
            command.args(["--grpc-endpoint", endpoint]);
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
