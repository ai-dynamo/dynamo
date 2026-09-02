// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashSet, VecDeque};
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dynamo_llm::entrypoint::{EngineConfig, HttpFrontend, RouterConfig};
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_mocker_backend::specdec::protocol::proposal_digest;
use dynamo_runtime::discovery::EventTransportKind;
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::storage::kv;
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::{FutureExt, StreamExt};
use reqwest::{Client, Method};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio::time::{Instant, timeout};
use uuid::Uuid;

const MODEL_NAME: &str = "mock-specdec-model";
const READY_TIMEOUT: Duration = Duration::from_secs(20);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_HTTP_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_WORKER_LOG_BYTES: usize = 1024 * 1024;
const TARGET_FAILURE_PROMPT_TOKEN: u32 = 128_125;
const ORPHAN_CLEANUP_TIMEOUT: Duration = Duration::from_millis(1_000);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn http_request_traverses_target_and_draft_processes() -> Result<()> {
    run_process(RouterMode::KV, true).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn http_request_uses_default_round_robin_for_target_and_draft() -> Result<()> {
    run_process(RouterMode::RoundRobin, false).await
}

async fn run_process(router_mode: RouterMode, full_matrix: bool) -> Result<()> {
    let mut harness = ProcessHarness::new(router_mode)?;
    let outcome = if full_matrix {
        AssertUnwindSafe(harness.exercise_matrix())
            .catch_unwind()
            .await
    } else {
        AssertUnwindSafe(harness.exercise_smoke())
            .catch_unwind()
            .await
    };
    let teardown = harness.shutdown().await;

    let completions = match outcome {
        Ok(Ok(completions)) => completions,
        Ok(Err(error)) => bail!("{error:#}\n{}", teardown.diagnostics()),
        Err(payload) => {
            eprintln!(
                "process E2E panicked before teardown:\n{}",
                teardown.diagnostics()
            );
            resume_unwind(payload);
        }
    };
    teardown.ensure_clean()?;
    if full_matrix {
        teardown.ensure_matrix(&completions)?;
    } else {
        teardown.ensure_smoke(&completions)?;
    }
    Ok(())
}

#[derive(Debug)]
struct Evidence {
    request_id: String,
    proposal_digest: String,
}

struct ProcessHarness {
    temp_dir: TempDir,
    discovery_root: PathBuf,
    tokenizer_path: PathBuf,
    namespace: String,
    http_address: SocketAddr,
    http_listener: Option<TcpListener>,
    http_client: Client,
    router_mode: RouterMode,
    draft: Option<ChildProcess>,
    target: Option<ChildProcess>,
    frontend: Option<FrontendProcess>,
}

impl ProcessHarness {
    fn new(router_mode: RouterMode) -> Result<Self> {
        let temp_dir = tempfile::tempdir().context("create process E2E temp directory")?;
        let discovery_root = temp_dir.path().join("discovery");
        std::fs::create_dir(&discovery_root).context("create file-discovery root")?;
        let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../llm/tests/data/sample-models/mock-llama-3.1-8b-instruct")
            .canonicalize()
            .context("resolve bundled tiny tokenizer fixture")?;
        let http_listener = reserve_loopback_listener()?;
        let http_address = http_listener
            .local_addr()
            .context("read reserved HTTP listener address")?;
        let http_client = Client::builder()
            .connect_timeout(Duration::from_secs(1))
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("build process E2E HTTP client")?;

        Ok(Self {
            temp_dir,
            discovery_root,
            tokenizer_path,
            namespace: format!("specdec-e2e-{}", Uuid::new_v4().simple()),
            http_address,
            http_listener: Some(http_listener),
            http_client,
            router_mode,
            draft: None,
            target: None,
            frontend: None,
        })
    }

    async fn exercise_matrix(&mut self) -> Result<Vec<Evidence>> {
        self.start_workers()?;
        self.start_frontend().await?;
        self.wait_until_ready().await?;

        let repeated_prompt = vec![128000, 128000, 128000];
        let first = completion(
            &self.http_client,
            self.http_address,
            repeated_prompt.clone(),
        )
        .await?;
        let repeated = completion(&self.http_client, self.http_address, repeated_prompt).await?;

        let concurrent = (0..32_u32).map(|index| {
            completion(
                &self.http_client,
                self.http_address,
                vec![1_000, 2_000, 3_000 + index],
            )
        });
        let concurrent = futures::future::try_join_all(concurrent)
            .await
            .context("32-request process matrix failed")?;
        ensure!(
            concurrent
                .iter()
                .map(|evidence| evidence.request_id.as_str())
                .collect::<HashSet<_>>()
                .len()
                == 32,
            "concurrent process requests reused a request ID"
        );
        ensure!(
            concurrent
                .iter()
                .map(|evidence| evidence.proposal_digest.as_str())
                .collect::<HashSet<_>>()
                .len()
                == 32,
            "concurrent process requests crossed proposal streams"
        );

        let target_failure = timeout(
            REQUEST_TIMEOUT,
            completion_response(
                &self.http_client,
                self.http_address,
                vec![128000, 128000, TARGET_FAILURE_PROMPT_TOKEN],
            ),
        )
        .await
        .context("target-failure request timed out")??;
        ensure!(
            target_failure.status != 200,
            "injected target failure unexpectedly returned HTTP {}: {}",
            target_failure.status,
            String::from_utf8_lossy(&target_failure.body)
        );

        let accepted_before = accepted_request_count(&self.draft_logs()?);
        let client = self.http_client.clone();
        let address = self.http_address;
        let cancelled = tokio::spawn(async move {
            completion_response_with_max_tokens(&client, address, vec![128000, 128000, 128127], 512)
                .await
        });
        let acceptance = self
            .wait_for_new_draft_acceptance(accepted_before, &cancelled)
            .await;
        cancelled.abort();
        let cancellation = cancelled.await;
        acceptance?;
        ensure!(
            cancellation.is_err_and(|error| error.is_cancelled()),
            "process cancellation request completed before the evidence-driven abort"
        );
        tokio::time::sleep(ORPHAN_CLEANUP_TIMEOUT + Duration::from_millis(500)).await;

        let after_cancel = completion(
            &self.http_client,
            self.http_address,
            vec![128000, 128000, 128126],
        )
        .await
        .context("completion after cancellation failed")?;

        let mut completions = vec![first, repeated];
        completions.extend(concurrent);
        completions.push(after_cancel);
        Ok(completions)
    }

    async fn exercise_smoke(&mut self) -> Result<Vec<Evidence>> {
        self.start_workers()?;
        self.start_frontend().await?;
        self.wait_until_ready().await?;
        completion(
            &self.http_client,
            self.http_address,
            vec![128000, 128000, 128000],
        )
        .await
        .map(|evidence| vec![evidence])
    }

    fn start_workers(&mut self) -> Result<()> {
        let draft_listener = StdTcpListener::bind("127.0.0.1:0")
            .context("reserve ephemeral draft transport port")?;
        let draft_address = format!(
            "tcp://{}",
            draft_listener
                .local_addr()
                .context("read reserved draft transport address")?
        );
        drop(draft_listener);

        self.draft = Some(ChildProcess::spawn(
            "draft",
            env!("CARGO_BIN_EXE_mock-draft-spec-dec-worker"),
            [
                "--namespace".into(),
                self.namespace.clone(),
                "--component".into(),
                "draft".into(),
                "--endpoint".into(),
                "generate".into(),
                "--model-name".into(),
                MODEL_NAME.into(),
                "--draft-bind-address".into(),
                draft_address.clone(),
                "--draft-advertise-address".into(),
                draft_address,
                "--draft-prefill-ms".into(),
                "20".into(),
                "--draft-token-interval-ms".into(),
                "1".into(),
                "--orphan-cleanup-timeout-ms".into(),
                ORPHAN_CLEANUP_TIMEOUT.as_millis().to_string(),
                "--transport-hwm".into(),
                "512".into(),
                "--transport-queue-capacity".into(),
                "512".into(),
            ],
            &self.discovery_root,
            self.temp_dir.path(),
        )?);

        self.target = Some(ChildProcess::spawn(
            "target",
            env!("CARGO_BIN_EXE_mock-target-specdec-worker"),
            [
                "--namespace".into(),
                self.namespace.clone(),
                "--component".into(),
                "target".into(),
                "--endpoint".into(),
                "generate".into(),
                "--model-name".into(),
                MODEL_NAME.into(),
                "--model-path".into(),
                self.tokenizer_path.display().to_string(),
                "--draft-endpoint".into(),
                format!("{}/draft/generate", self.namespace),
                "--target-prefill-ms".into(),
                "250".into(),
                "--target-token-interval-ms".into(),
                "1".into(),
                "--transport-hwm".into(),
                "512".into(),
                "--transport-queue-capacity".into(),
                "512".into(),
                "--fail-after-draft-start-prompt-token".into(),
                TARGET_FAILURE_PROMPT_TOKEN.to_string(),
            ],
            &self.discovery_root,
            self.temp_dir.path(),
        )?);
        Ok(())
    }

    async fn start_frontend(&mut self) -> Result<()> {
        let mut model_builder = LocalModelBuilder::default();
        let model = model_builder
            .model_path(self.tokenizer_path.clone())
            .model_name(Some(MODEL_NAME.to_string()))
            .namespace(Some(self.namespace.clone()))
            .router_config(Some(RouterConfig {
                router_mode: self.router_mode,
                ..RouterConfig::default()
            }))
            .http_host(Some("127.0.0.1".to_string()))
            .http_port(self.http_address.port())
            .build()
            .await
            .context("build frontend local model")?;
        let engine_config = EngineConfig::Dynamic {
            model: Box::new(model),
            chat_engine_factory: None,
            prefill_load_estimator: None,
        };
        let listener = self
            .http_listener
            .take()
            .context("HTTP listener reservation missing")?;

        let runtime = Runtime::from_current().context("create frontend runtime")?;
        let config = DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(
                self.discovery_root.clone(),
            )),
            nats_config: None,
            request_plane: RequestPlaneMode::Tcp,
            event_transport_kind: EventTransportKind::Zmq,
        };
        let distributed_runtime = DistributedRuntime::new(runtime, config)
            .await
            .context("create frontend distributed runtime")?;
        let task_runtime = distributed_runtime.clone();
        let task = tokio::spawn(async move {
            HttpFrontend::default()
                .run_with_listener(task_runtime, engine_config, listener)
                .await
        });
        self.frontend = Some(FrontendProcess {
            runtime: distributed_runtime,
            task: Some(task),
        });
        Ok(())
    }

    async fn wait_until_ready(&mut self) -> Result<()> {
        let deadline = Instant::now() + READY_TIMEOUT;
        let mut interval = tokio::time::interval(Duration::from_millis(25));
        loop {
            self.ensure_processes_running()?;
            if let Some(frontend) = self.frontend.as_ref()
                && frontend.is_finished()
            {
                bail!("frontend exited before becoming ready");
            }
            let last_error = match http_json(
                &self.http_client,
                self.http_address,
                Method::GET,
                "/v1/models",
                None,
            )
            .await
            {
                Ok(response) if response.status == 200 => {
                    let body: Value = serde_json::from_slice(&response.body)
                        .context("decode model-list response")?;
                    let model_ready =
                        body.get("data")
                            .and_then(Value::as_array)
                            .is_some_and(|models| {
                                models.iter().any(|model| {
                                    model.get("id").and_then(Value::as_str) == Some(MODEL_NAME)
                                })
                            });
                    if model_ready {
                        return Ok(());
                    }
                    format!("model list does not contain {MODEL_NAME}: {body}")
                }
                Ok(response) => format!(
                    "model-list HTTP {}: {}",
                    response.status,
                    String::from_utf8_lossy(&response.body)
                ),
                Err(error) => format!("{error:#}"),
            };
            ensure!(
                Instant::now() < deadline,
                "readiness timed out: {last_error}"
            );
            interval.tick().await;
        }
    }

    fn ensure_processes_running(&mut self) -> Result<()> {
        for process in [&mut self.draft, &mut self.target].into_iter().flatten() {
            process.ensure_running()?;
        }
        Ok(())
    }

    fn draft_logs(&self) -> Result<String> {
        self.draft
            .as_ref()
            .context("draft worker is not running")?
            .live_logs()
    }

    async fn wait_for_new_draft_acceptance(
        &mut self,
        accepted_before: usize,
        request: &JoinHandle<Result<HttpResponse>>,
    ) -> Result<()> {
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        let mut interval = tokio::time::interval(Duration::from_millis(10));
        loop {
            self.ensure_processes_running()?;
            if accepted_request_count(&self.draft_logs()?) > accepted_before {
                return Ok(());
            }
            ensure!(
                !request.is_finished(),
                "process cancellation request completed before draft START acceptance"
            );
            ensure!(
                Instant::now() < deadline,
                "draft did not accept the process cancellation request before the deadline"
            );
            interval.tick().await;
        }
    }

    async fn shutdown(mut self) -> Teardown {
        let mut frontend_error = None;
        if let Some(frontend) = self.frontend.take()
            && let Err(error) = frontend.shutdown().await
        {
            frontend_error = Some(format!("{error:#}"));
        }

        let mut reports = Vec::new();
        if let Some(target) = self.target.take() {
            reports.push(target.shutdown().await);
        }
        if let Some(draft) = self.draft.take() {
            reports.push(draft.shutdown().await);
        }
        Teardown {
            frontend_error,
            reports,
        }
    }
}

fn reserve_loopback_listener() -> Result<TcpListener> {
    let listener = StdTcpListener::bind("127.0.0.1:0").context("reserve loopback listener")?;
    listener
        .set_nonblocking(true)
        .context("make reserved loopback listener nonblocking")?;
    TcpListener::from_std(listener).context("adopt reserved loopback listener")
}

struct FrontendProcess {
    runtime: DistributedRuntime,
    task: Option<JoinHandle<Result<()>>>,
}

impl FrontendProcess {
    fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(JoinHandle::is_finished)
    }

    async fn shutdown(mut self) -> Result<()> {
        self.runtime.shutdown();
        let mut task = self.task.take().context("frontend task missing")?;
        match timeout(SHUTDOWN_TIMEOUT, &mut task).await {
            Ok(joined) => joined.context("join frontend task")?,
            Err(_) => {
                task.abort();
                let abort_outcome = match timeout(SHUTDOWN_TIMEOUT, &mut task).await {
                    Ok(Ok(Ok(()))) => "frontend completed after abort".to_string(),
                    Ok(Ok(Err(error))) => {
                        format!("frontend returned an error after abort: {error:#}")
                    }
                    Ok(Err(error)) => format!("frontend abort join result: {error}"),
                    Err(_) => "aborted frontend task did not join before the deadline".to_string(),
                };
                bail!("frontend did not stop before the shutdown deadline; {abort_outcome}");
            }
        }
    }
}

impl Drop for FrontendProcess {
    fn drop(&mut self) {
        self.runtime.shutdown();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

struct ChildProcess {
    name: &'static str,
    child: Child,
    stdout: Option<JoinHandle<std::io::Result<()>>>,
    stderr: Option<JoinHandle<std::io::Result<()>>>,
    stdout_tail: LogTail,
    stderr_tail: LogTail,
}

impl ChildProcess {
    fn spawn(
        name: &'static str,
        program: &str,
        args: impl IntoIterator<Item = String>,
        discovery_root: &Path,
        working_directory: &Path,
    ) -> Result<Self> {
        let mut command = Command::new(program);
        command
            .args(args)
            .env("DYN_DISCOVERY_BACKEND", "file")
            .env("DYN_FILE_KV", discovery_root)
            .env("DYN_REQUEST_PLANE", "tcp")
            .env("DYN_EVENT_PLANE", "zmq")
            .env("DYN_TCP_RPC_HOST", "127.0.0.1")
            .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
            .env("DYN_SYSTEM_PORT", "-1")
            .env("DYN_SELF_HOST_METADATA", "0")
            .env("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0")
            .env("DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS", "5")
            .env("DYN_LOGGING_CONSOLE_FORMAT", "jsonl")
            .env("RUST_LOG", "info")
            .current_dir(working_directory)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn {name} worker"))?;
        let stdout = child.stdout.take().context("capture worker stdout")?;
        let stderr = child.stderr.take().context("capture worker stderr")?;
        let stdout_tail = LogTail::default();
        let stderr_tail = LogTail::default();
        Ok(Self {
            name,
            child,
            stdout: Some(tokio::spawn(read_log(stdout, stdout_tail.clone()))),
            stderr: Some(tokio::spawn(read_log(stderr, stderr_tail.clone()))),
            stdout_tail,
            stderr_tail,
        })
    }

    fn ensure_running(&mut self) -> Result<()> {
        if let Some(status) = self.child.try_wait().context("poll worker process")? {
            bail!("{} worker exited before readiness with {status}", self.name);
        }
        Ok(())
    }

    fn live_logs(&self) -> Result<String> {
        Ok(format!(
            "{}\n{}",
            self.stdout_tail.snapshot()?,
            self.stderr_tail.snapshot()?
        ))
    }

    async fn shutdown(mut self) -> ProcessReport {
        let mut forced = false;
        let mut shutdown_errors = Vec::new();
        if let Some(pid) = self.child.id() {
            // SAFETY: pid belongs to the child process owned by this guard.
            let signal_result = unsafe { libc::kill(pid as libc::pid_t, libc::SIGINT) };
            if signal_result != 0 {
                forced = true;
                shutdown_errors.push(format!("send SIGINT: {}", std::io::Error::last_os_error()));
                if let Err(error) = self.child.start_kill() {
                    shutdown_errors.push(format!("start forced kill: {error}"));
                }
            }
        }
        let status = match timeout(SHUTDOWN_TIMEOUT, self.child.wait()).await {
            Ok(Ok(status)) => Some(status),
            Ok(Err(error)) => {
                shutdown_errors.push(format!("wait for child: {error}"));
                None
            }
            Err(_) => {
                forced = true;
                if let Err(error) = self.child.start_kill() {
                    shutdown_errors.push(format!("start forced kill: {error}"));
                }
                match timeout(SHUTDOWN_TIMEOUT, self.child.wait()).await {
                    Ok(Ok(status)) => Some(status),
                    Ok(Err(error)) => {
                        shutdown_errors.push(format!("wait after forced kill: {error}"));
                        None
                    }
                    Err(_) => {
                        shutdown_errors.push(
                            "child was not reaped before the forced-kill deadline".to_string(),
                        );
                        None
                    }
                }
            }
        };
        let stdout = collect_log(self.stdout.take(), &self.stdout_tail).await;
        let stderr = collect_log(self.stderr.take(), &self.stderr_tail).await;
        ProcessReport {
            name: self.name,
            status,
            forced,
            shutdown_error: (!shutdown_errors.is_empty()).then(|| shutdown_errors.join("; ")),
            stdout,
            stderr,
        }
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        if let Some(task) = self.stdout.take() {
            task.abort();
        }
        if let Some(task) = self.stderr.take() {
            task.abort();
        }
    }
}

#[derive(Clone, Default)]
struct LogTail(Arc<StdMutex<VecDeque<u8>>>);

impl LogTail {
    fn append(&self, bytes: &[u8]) -> std::io::Result<()> {
        let mut output = self
            .0
            .lock()
            .map_err(|_| std::io::Error::other("worker log tail lock was poisoned"))?;
        output.extend(bytes);
        let overflow = output.len().saturating_sub(MAX_WORKER_LOG_BYTES);
        output.drain(..overflow);
        Ok(())
    }

    fn snapshot(&self) -> Result<String> {
        let output = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("worker log tail lock was poisoned"))?;
        let bytes = output.iter().copied().collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

async fn read_log(mut reader: impl AsyncRead + Unpin, output: LogTail) -> std::io::Result<()> {
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut chunk).await?;
        if read == 0 {
            return Ok(());
        }
        output.append(&chunk[..read])?;
    }
}

async fn collect_log(task: Option<JoinHandle<std::io::Result<()>>>, output: &LogTail) -> String {
    let Some(mut task) = task else {
        return "<log task missing>".to_string();
    };
    match timeout(Duration::from_secs(2), &mut task).await {
        Ok(Ok(Ok(()))) => output
            .snapshot()
            .unwrap_or_else(|error| format!("<log snapshot failed: {error:#}>")),
        Ok(Ok(Err(error))) => format!("<log read failed: {error}>"),
        Ok(Err(error)) => format!("<log task failed: {error}>"),
        Err(_) => {
            task.abort();
            let _ = task.await;
            "<log collection timed out>".to_string()
        }
    }
}

struct ProcessReport {
    name: &'static str,
    status: Option<ExitStatus>,
    forced: bool,
    shutdown_error: Option<String>,
    stdout: String,
    stderr: String,
}

impl ProcessReport {
    fn combined_log(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

struct Teardown {
    frontend_error: Option<String>,
    reports: Vec<ProcessReport>,
}

impl Teardown {
    fn ensure_clean(&self) -> Result<()> {
        ensure!(
            self.frontend_error.is_none(),
            "frontend teardown failed: {}",
            self.frontend_error.as_deref().unwrap_or_default()
        );
        for report in &self.reports {
            ensure!(
                report.shutdown_error.is_none(),
                "{} worker teardown failed: {}",
                report.name,
                report.shutdown_error.as_deref().unwrap_or_default()
            );
            ensure!(
                !report.forced,
                "{} worker required a forced kill",
                report.name
            );
            ensure!(
                report.status.is_some_and(|status| status.success()),
                "{} worker exited unsuccessfully: {:?}\n{}",
                report.name,
                report.status,
                report.combined_log()
            );
        }
        Ok(())
    }

    fn ensure_matrix(&self, completions: &[Evidence]) -> Result<()> {
        for evidence in completions {
            self.ensure_correlated(evidence)?;
        }
        let target_logs = self.worker_logs("target")?;
        let draft_logs = self.worker_logs("draft")?;
        ensure_lifecycle_identity(
            &target_logs,
            &[
                "draft client queued START",
                "draft client reader received START_ACK",
                "draft client start waiter received response",
                "mock speculative target cancelled during setup",
                "mock speculative draft START preceded target prefill completion",
                "mock speculative target injected failure after draft START",
                "mock speculative target consumed draft proposal",
                "mock speculative target received draft cleanup acknowledgement",
            ],
        )?;
        ensure_lifecycle_identity(
            &draft_logs,
            &[
                "draft server queued START_ACK",
                "draft server wrote START_ACK",
                "mock speculative draft accepted request",
                "mock speculative draft completed proposal",
                "mock speculative draft cleaned request",
                "reaped orphaned mock draft session",
            ],
        )?;
        require_lifecycle_events_with_identity(
            &target_logs,
            &[
                "draft client queued START",
                "draft client reader received START_ACK",
                "draft client start waiter received response",
            ],
        )?;
        require_lifecycle_events_with_identity(
            &draft_logs,
            &[
                "draft server queued START_ACK",
                "draft server wrote START_ACK",
            ],
        )?;
        let overlap_request_id = &completions
            .first()
            .context("matrix did not record a successful completion")?
            .request_id;
        require_log_event(
            &target_logs,
            "target-prefill overlap",
            &[
                "mock speculative draft START preceded target prefill completion",
                overlap_request_id,
            ],
        )?;
        let failure_event = require_log_event(
            &target_logs,
            "injected target failure",
            &["mock speculative target injected failure after draft START"],
        )?;
        ensure!(
            log_field(failure_event, "draft_cleanup").as_deref() == Some("acknowledged"),
            "injected target-failure event does not prove acknowledged cleanup: {failure_event}"
        );
        let failed_request_id = log_field(failure_event, "request_id")
            .context("injected target-failure event omitted request_id")?;
        require_log_event(
            &draft_logs,
            "injected target-failure cleanup",
            &["mock speculative draft cleaned request", &failed_request_id],
        )?;

        let cancellation_event = require_log_event(
            &target_logs,
            "target request cancellation",
            &["mock speculative target cancelled during setup"],
        )?;
        let cancelled_request_id = log_field(cancellation_event, "request_id")
            .context("target cancellation event omitted request_id")?;
        let cleanup_event = if find_log_event(
            &target_logs,
            &[
                "mock speculative target received draft cleanup acknowledgement",
                &cancelled_request_id,
            ],
        )
        .is_some()
        {
            require_log_event(
                &draft_logs,
                "acknowledged cancelled-request cleanup",
                &[
                    "mock speculative draft cleaned request",
                    &cancelled_request_id,
                ],
            )?
        } else {
            require_log_event(
                &draft_logs,
                "cancelled draft-session reap",
                &["reaped orphaned mock draft session", &cancelled_request_id],
            )?
        };
        ensure!(
            log_field(cleanup_event, "active_sessions").as_deref() == Some("0"),
            "cancelled request did not leave zero active sessions before teardown: {cleanup_event}"
        );
        self.ensure_zero_sessions()
    }

    fn ensure_smoke(&self, completions: &[Evidence]) -> Result<()> {
        ensure!(
            completions.len() == 1,
            "process smoke test expected one completion, got {}",
            completions.len()
        );
        self.ensure_correlated(&completions[0])?;
        self.ensure_zero_sessions()
    }

    fn ensure_correlated(&self, evidence: &Evidence) -> Result<()> {
        let target_logs = self.worker_logs("target")?;
        let draft_logs = self.worker_logs("draft")?;
        let target_event = require_log_event(
            &target_logs,
            "target proposal consumption",
            &[
                "mock speculative target consumed draft proposal",
                &evidence.request_id,
                &evidence.proposal_digest,
            ],
        )?;
        ensure!(
            log_field(target_event, "draft_cleanup").as_deref() == Some("acknowledged"),
            "target completion does not prove acknowledged cleanup: {target_event}"
        );
        require_log_event(
            &draft_logs,
            "draft proposal completion",
            &[
                "mock speculative draft completed proposal",
                &evidence.request_id,
                &evidence.proposal_digest,
            ],
        )?;
        require_log_event(
            &draft_logs,
            "draft request cleanup",
            &[
                "mock speculative draft cleaned request",
                &evidence.request_id,
            ],
        )?;
        Ok(())
    }

    fn ensure_zero_sessions(&self) -> Result<()> {
        let draft_logs = self.worker_logs("draft")?;
        ensure!(
            draft_logs
                .contains("mock speculative draft transport stopped with zero active sessions"),
            "draft shutdown did not prove zero active sessions\n{draft_logs}"
        );
        Ok(())
    }

    fn worker_logs(&self, name: &str) -> Result<String> {
        self.reports
            .iter()
            .find(|report| report.name == name)
            .with_context(|| format!("missing {name} worker report"))
            .map(ProcessReport::combined_log)
    }

    fn diagnostics(&self) -> String {
        let mut output = format!("frontend teardown: {:?}\n", self.frontend_error);
        for report in &self.reports {
            output.push_str(&format!(
                "\n{} status={:?} forced={} shutdown_error={:?}\n{}",
                report.name,
                report.status,
                report.forced,
                report.shutdown_error,
                report.combined_log()
            ));
        }
        output
    }
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

fn require_log_event<'a>(logs: &'a str, description: &str, needles: &[&str]) -> Result<&'a str> {
    find_log_event(logs, needles).with_context(|| {
        format!("logs do not contain one {description} event with fields {needles:?}\n{logs}")
    })
}

fn find_log_event<'a>(logs: &'a str, needles: &[&str]) -> Option<&'a str> {
    logs.lines().find(|line| {
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            return false;
        };
        let Some(fields) = event.as_object() else {
            return false;
        };
        needles.iter().all(|needle| {
            fields.values().any(|value| match value {
                Value::String(value) => value == needle,
                Value::Number(value) => value.to_string() == *needle,
                _ => false,
            })
        })
    })
}

fn log_field(line: &str, name: &str) -> Option<String> {
    let event: Value = serde_json::from_str(line).ok()?;
    event.get(name).and_then(|value| match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn ensure_lifecycle_identity(logs: &str, lifecycle_messages: &[&str]) -> Result<()> {
    for event in logs
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| {
            event
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| lifecycle_messages.contains(&message))
        })
    {
        for field in ["worker_id", "dp_rank", "draft_incarnation", "request_id"] {
            ensure!(
                event.get(field).is_some(),
                "lifecycle event omitted {field}: {event}"
            );
        }
    }
    Ok(())
}

fn require_lifecycle_events_with_identity(logs: &str, lifecycle_messages: &[&str]) -> Result<()> {
    for message in lifecycle_messages {
        let event = logs
            .lines()
            .find_map(|line| {
                let event = serde_json::from_str::<Value>(line).ok()?;
                (event.get("message").and_then(Value::as_str) == Some(message)).then_some(event)
            })
            .with_context(|| format!("missing required lifecycle event {message}"))?;
        for field in ["worker_id", "dp_rank", "draft_incarnation", "request_id"] {
            ensure!(
                event.get(field).is_some(),
                "required lifecycle event omitted {field}: {event}"
            );
        }
    }
    Ok(())
}

fn accepted_request_count(logs: &str) -> usize {
    logs.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| {
            matches!(
                event.get("message").and_then(Value::as_str),
                Some(
                    "mock speculative draft accepted request"
                        | "mock speculative draft accepted authenticated cache match"
                )
            )
        })
        .count()
}

async fn completion(client: &Client, address: SocketAddr, prompt: Vec<u32>) -> Result<Evidence> {
    let expected_proposal_digest =
        proposal_digest(&prompt.iter().copied().cycle().take(4).collect::<Vec<_>>());
    let response = timeout(
        REQUEST_TIMEOUT,
        completion_response(client, address, prompt),
    )
    .await
    .context("completion request timed out")??;
    ensure!(
        response.status == 200,
        "completion returned HTTP {}: {}",
        response.status,
        String::from_utf8_lossy(&response.body)
    );

    let body: Value =
        serde_json::from_slice(&response.body).context("decode completion response as JSON")?;
    ensure!(
        body.pointer("/choices/0/finish_reason") == Some(&Value::String("length".into())),
        "completion did not terminate at the requested token bound: {body}"
    );
    ensure!(
        body.pointer("/nvext/engine_data/_dynamo_external_speculation_v1")
            .is_none(),
        "internal external-speculation lifecycle marker leaked into HTTP: {body}"
    );
    let mock = body
        .pointer("/nvext/engine_data/mock_specdec")
        .context("completion omitted mock speculative-decoding evidence")?;
    let request_id = mock
        .get("request_id")
        .and_then(Value::as_str)
        .context("completion omitted mock request ID")?
        .to_owned();
    let proposal_digest = mock
        .get("proposal_digest")
        .and_then(Value::as_str)
        .context("completion omitted mock proposal digest")?
        .to_owned();
    Uuid::parse_str(&request_id).context("completion returned an invalid request ID")?;
    ensure!(
        proposal_digest.len() == 64,
        "completion returned an invalid proposal digest"
    );
    ensure!(
        proposal_digest == expected_proposal_digest,
        "completion crossed proposal streams: expected {expected_proposal_digest}, got {proposal_digest}"
    );

    Ok(Evidence {
        request_id,
        proposal_digest,
    })
}

async fn completion_response(
    client: &Client,
    address: SocketAddr,
    prompt: Vec<u32>,
) -> Result<HttpResponse> {
    completion_response_with_max_tokens(client, address, prompt, 4).await
}

async fn completion_response_with_max_tokens(
    client: &Client,
    address: SocketAddr,
    prompt: Vec<u32>,
    max_tokens: u32,
) -> Result<HttpResponse> {
    http_json(
        client,
        address,
        Method::POST,
        "/v1/completions",
        Some(&json!({
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": false,
            "nvext": {"extra_fields": ["engine_data"]},
        })),
    )
    .await
}

async fn http_json(
    client: &Client,
    address: SocketAddr,
    method: Method,
    path: &str,
    body: Option<&Value>,
) -> Result<HttpResponse> {
    let url = format!("http://{address}{path}");
    let mut request = client.request(method, url);
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request.send().await.context("send frontend HTTP request")?;
    let status = response.status().as_u16();
    if let Some(content_length) = response.content_length() {
        ensure!(
            content_length <= MAX_HTTP_RESPONSE_BYTES as u64,
            "frontend response exceeded the harness limit"
        );
    }
    let mut body = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.context("read frontend response body")?;
        ensure!(
            body.len().saturating_add(chunk.len()) <= MAX_HTTP_RESPONSE_BYTES,
            "frontend response exceeded the harness limit"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(HttpResponse { status, body })
}
