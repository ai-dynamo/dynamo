// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "specdec-kind")]

use std::collections::{HashSet, VecDeque};
use std::net::{SocketAddr, TcpListener};
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dynamo_mocker_backend::specdec::protocol::proposal_digest;
use futures::{FutureExt, StreamExt};
use reqwest::{Client, Method};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio::time::{Instant, timeout};
use uuid::Uuid;

const MODEL_NAME: &str = "mock-specdec-model";
const COMMAND_OUTPUT_LIMIT: usize = 4 * 1024 * 1024;
const DIAGNOSTIC_OUTPUT_LIMIT: usize = 1024 * 1024;
const HTTP_RESPONSE_LIMIT: usize = 1024 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(180);
const BUILD_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const READY_TIMEOUT: Duration = Duration::from_secs(90);
const PROCESS_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const LOG_POLL_COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const MANIFEST: &str =
    include_str!("../../../../../examples/backends/mocker/deploy/specdec-kind.yaml");

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
#[ignore = "developer-run local Kind E2E; requires Docker, Kind, and kubectl"]
async fn cpu_only_kind_topology_survives_load_cancellation_and_draft_replacement() -> Result<()> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../..")
        .canonicalize()
        .context("resolve Dynamo workspace root")?;
    check_prerequisites(&repo).await?;

    let content_id = workspace_content_id(&repo).await?;
    let worker_image = format!("dynamo-specdec-worker:{content_id}");
    let frontend_image = format!("dynamo-specdec-frontend:{content_id}");
    eprintln!("Kind E2E image set: {worker_image}, {frontend_image}");
    build_images(&repo, &worker_image, &frontend_image).await?;

    let temp_dir = tempfile::tempdir().context("create Kind E2E temp directory")?;
    let suffix = Uuid::new_v4().simple().to_string();
    let cluster = KindCluster {
        name: format!("dynamo-specdec-{}", &suffix[..12]),
        namespace: format!("specdec-{}", &suffix[..12]),
        kubeconfig: temp_dir.path().join("kubeconfig"),
        repo,
    };

    let run_outcome = AssertUnwindSafe(run_kind_matrix(
        &cluster,
        &temp_dir,
        &worker_image,
        &frontend_image,
    ))
    .catch_unwind()
    .await;
    if !matches!(&run_outcome, Ok(Ok(()))) {
        let diagnostics = AssertUnwindSafe(cluster.diagnostics()).catch_unwind().await;
        match (&run_outcome, diagnostics) {
            (Ok(Err(error)), Ok(diagnostics)) => {
                eprintln!("Kind E2E failed: {error:#}\n{diagnostics}");
            }
            (Err(_), Ok(diagnostics)) => {
                eprintln!("Kind E2E panicked before cleanup\n{diagnostics}");
            }
            (Ok(Err(error)), Err(_)) => {
                eprintln!("Kind E2E failed: {error:#}\ndiagnostics panicked");
            }
            (Err(_), Err(_)) => eprintln!("Kind E2E and diagnostics both panicked"),
            (Ok(Ok(())), _) => unreachable!("successful matrix does not collect diagnostics"),
        }
    }
    let cleanup_result = cluster.delete().await;

    let run_result = match run_outcome {
        Ok(result) => result,
        Err(payload) => {
            if let Err(cleanup) = cleanup_result {
                eprintln!("owned Kind cluster cleanup failed after panic: {cleanup:#}");
            }
            resume_unwind(payload);
        }
    };
    match (run_result, cleanup_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(run), Ok(())) => Err(run),
        (Ok(()), Err(cleanup)) => Err(cleanup.context("delete owned Kind cluster")),
        (Err(run), Err(cleanup)) => {
            bail!("{run:#}\nowned Kind cluster cleanup also failed: {cleanup:#}")
        }
    }
}

async fn run_kind_matrix(
    cluster: &KindCluster,
    temp_dir: &TempDir,
    worker_image: &str,
    frontend_image: &str,
) -> Result<()> {
    cluster.create().await?;
    cluster.load_images(worker_image, frontend_image).await?;
    let etcd_image = cluster.preloaded_etcd_image().await?;
    let manifest_path =
        render_manifest(temp_dir, worker_image, frontend_image, &etcd_image).await?;
    cluster.deploy(&manifest_path).await?;

    let mut port_forward = cluster.start_port_forward()?;
    let address = port_forward.address;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("build Kind E2E HTTP client")?;
    let matrix_outcome = AssertUnwindSafe(async {
        wait_for_model(&client, address, &mut port_forward).await?;

        let happy = completion(&client, address, vec![128000, 128000, 128000]).await?;
        let concurrent = futures::future::try_join_all(
            (0..32_u32)
                .map(|index| completion(&client, address, vec![1_000, 2_000, 3_000 + index])),
        )
        .await
        .context("32-request Kind matrix failed")?;
        ensure!(
            concurrent
                .iter()
                .map(|evidence| evidence.request_id.as_str())
                .collect::<HashSet<_>>()
                .len()
                == 32,
            "Kind concurrent requests reused a request ID"
        );
        ensure!(
            concurrent
                .iter()
                .map(|evidence| evidence.proposal_digest.as_str())
                .collect::<HashSet<_>>()
                .len()
                == 32,
            "Kind concurrent requests crossed proposal streams"
        );

        let accepted_before = accepted_request_count(&cluster.logs("draft", 5_000).await?);
        let cancelled_client = client.clone();
        let cancelled = AbortOnDropTask::new(tokio::spawn(async move {
            completion_with_max_tokens(
                &cancelled_client,
                address,
                vec![128000, 128000, 128127],
                512,
            )
            .await
        }));
        let probe_result = wait_for_new_draft_request(cluster, accepted_before, &cancelled).await;
        cancelled.abort();
        let cancellation_result = cancelled.join().await;
        probe_result?;
        match cancellation_result {
            Err(error) if error.is_cancelled() => {}
            Err(error) => return Err(error).context("Kind cancellation probe task failed"),
            Ok(Ok(evidence)) => {
                bail!(
                    "Kind cancellation probe completed before abort: {}",
                    evidence.request_id
                )
            }
            Ok(Err(error)) => return Err(error).context("Kind cancellation probe failed"),
        }

        let (target_logs, initial_draft_logs) = wait_for_cancelled_session_reaped(cluster).await?;
        let mut initial_completions = vec![happy];
        initial_completions.extend(concurrent);
        assert_correlated(&target_logs, &initial_draft_logs, &initial_completions)?;
        assert_overlap(&target_logs, &initial_completions[0].request_id)?;
        assert_cancelled_session_reaped(&target_logs, &initial_draft_logs)?;

        let initial_incarnation = draft_incarnation(&initial_draft_logs)?;
        let old_pod = cluster.ready_draft_pod().await?;
        let new_pod = cluster.replace_draft_pod(&old_pod).await?;
        ensure!(
            old_pod != new_pod,
            "draft replacement reused the deleted pod"
        );

        let recovery =
            retry_completion_after_replacement(&client, address, vec![128010, 128011, 128012])
                .await?;
        let recovered_target_logs = cluster.logs("target", 5_000).await?;
        let replacement_draft_logs = cluster.logs("draft", 5_000).await?;
        let replacement_incarnation = draft_incarnation(&replacement_draft_logs)?;
        ensure!(
            initial_incarnation != replacement_incarnation,
            "replacement draft reused incarnation {initial_incarnation}"
        );
        assert_correlated(
            &recovered_target_logs,
            &replacement_draft_logs,
            std::slice::from_ref(&recovery),
        )?;
        let cleanup = matching_event(
            &replacement_draft_logs,
            &[
                ("message", "mock speculative draft cleaned request"),
                ("request_id", &recovery.request_id),
                ("active_sessions", "0"),
            ],
        );
        ensure!(
            cleanup.is_some(),
            "replacement draft did not finish the recovery request with zero active sessions"
        );
        Ok(())
    })
    .catch_unwind()
    .await;

    let shutdown_result = port_forward.shutdown().await;
    let matrix_result = match matrix_outcome {
        Ok(result) => result,
        Err(payload) => {
            if let Err(shutdown) = shutdown_result {
                eprintln!("kubectl port-forward cleanup failed after panic: {shutdown:#}");
            }
            resume_unwind(payload);
        }
    };
    match (matrix_result, shutdown_result) {
        (Ok(()), Ok(_)) => Ok(()),
        (Err(matrix), Ok(logs)) => {
            Err(matrix).with_context(|| format!("kubectl port-forward output:\n{logs}"))
        }
        (Ok(()), Err(shutdown)) => Err(shutdown),
        (Err(matrix), Err(shutdown)) => {
            bail!("{matrix:#}\nkubectl port-forward cleanup also failed: {shutdown:#}")
        }
    }
}

async fn check_prerequisites(repo: &Path) -> Result<()> {
    for (program, args) in [
        ("docker", vec!["info", "--format", "{{.ServerVersion}}"]),
        ("kind", vec!["version"]),
        ("kubectl", vec!["version", "--client=true", "--output=yaml"]),
    ] {
        let args = args.into_iter().map(str::to_string).collect::<Vec<_>>();
        run_checked(program, &args, repo, Duration::from_secs(20), COMMAND_OUTPUT_LIMIT)
            .await
            .with_context(|| {
                format!(
                    "missing or unusable local prerequisite `{program}`; install Docker, Kind, and kubectl before running this ignored test"
                )
            })?;
    }
    Ok(())
}

async fn workspace_content_id(repo: &Path) -> Result<String> {
    let head = run_checked(
        "git",
        &strings(&["rev-parse", "HEAD"]),
        repo,
        Duration::from_secs(20),
        COMMAND_OUTPUT_LIMIT,
    )
    .await?;
    let diff = run_checked(
        "git",
        &strings(&["diff", "--binary", "--no-ext-diff", "HEAD", "--"]),
        repo,
        Duration::from_secs(30),
        32 * 1024 * 1024,
    )
    .await?;
    let untracked = run_checked(
        "git",
        &strings(&["ls-files", "-z", "--others", "--exclude-standard"]),
        repo,
        Duration::from_secs(20),
        COMMAND_OUTPUT_LIMIT,
    )
    .await?;

    ensure!(
        !head.stdout_truncated,
        "git HEAD output exceeded the content-hash capture limit"
    );
    ensure!(
        !diff.stdout_truncated,
        "git diff exceeded the content-hash capture limit"
    );
    ensure!(
        !untracked.stdout_truncated,
        "git untracked-file inventory exceeded the content-hash capture limit"
    );

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"dynamo-kind-worktree-v1\0");
    hash_framed(&mut hasher, &head.stdout)?;
    hash_framed(&mut hasher, &diff.stdout)?;
    for raw_path in untracked
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        let path = std::str::from_utf8(raw_path).context("git returned a non-UTF-8 path")?;
        let (content_len, content_hash) = hash_file(&repo.join(path))
            .await
            .with_context(|| format!("hash untracked build input {path}"))?;
        hasher.update(b"untracked-file\0");
        hash_framed(&mut hasher, raw_path)?;
        hasher.update(&content_len.to_le_bytes());
        hasher.update(content_hash.as_bytes());
    }
    Ok(hasher.finalize().to_hex()[..16].to_string())
}

fn hash_framed(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<()> {
    let len = u64::try_from(bytes.len()).context("worktree hash input is too large")?;
    hasher.update(&len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

async fn hash_file(path: &Path) -> Result<(u64, blake3::Hash)> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = blake3::Hasher::new();
    let mut len = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        len = len
            .checked_add(u64::try_from(read).context("file chunk length exceeds u64")?)
            .context("untracked build input is too large")?;
        hasher.update(&buffer[..read]);
    }
    Ok((len, hasher.finalize()))
}

async fn build_images(repo: &Path, worker_image: &str, frontend_image: &str) -> Result<()> {
    if !image_exists(repo, worker_image).await? {
        run_checked(
            "docker",
            &[
                "build".into(),
                "--progress=plain".into(),
                "--file".into(),
                "lib/backend-common/examples/mocker/Dockerfile".into(),
                "--tag".into(),
                worker_image.into(),
                ".".into(),
            ],
            repo,
            BUILD_TIMEOUT,
            COMMAND_OUTPUT_LIMIT,
        )
        .await
        .context("build current-tree speculative worker image")?;
    }
    if !image_exists(repo, frontend_image).await? {
        run_checked(
            "docker",
            &[
                "build".into(),
                "--progress=plain".into(),
                "--file".into(),
                "lib/backend-common/examples/mocker/Dockerfile.frontend".into(),
                "--tag".into(),
                frontend_image.into(),
                ".".into(),
            ],
            repo,
            BUILD_TIMEOUT,
            COMMAND_OUTPUT_LIMIT,
        )
        .await
        .context("build current-tree frontend image")?;
    }
    Ok(())
}

async fn image_exists(repo: &Path, image: &str) -> Result<bool> {
    let output = run_capture(
        "docker",
        &["image".into(), "inspect".into(), image.into()],
        repo,
        Duration::from_secs(20),
        COMMAND_OUTPUT_LIMIT,
    )
    .await?;
    Ok(output.status.success())
}

async fn render_manifest(
    temp_dir: &TempDir,
    worker_image: &str,
    frontend_image: &str,
    etcd_image: &str,
) -> Result<PathBuf> {
    let rendered = MANIFEST
        .replace("DYNAMO_SPECDEC_WORKER_IMAGE", worker_image)
        .replace("DYNAMO_SPECDEC_FRONTEND_IMAGE", frontend_image)
        .replace("DYNAMO_SPECDEC_ETCD_IMAGE", etcd_image);
    ensure!(
        !rendered.contains("DYNAMO_SPECDEC_"),
        "Kind manifest contains an unresolved image placeholder"
    );
    let path = temp_dir.path().join("specdec-kind-rendered.yaml");
    tokio::fs::write(&path, rendered)
        .await
        .context("write rendered Kind manifest")?;
    Ok(path)
}

struct KindCluster {
    name: String,
    namespace: String,
    kubeconfig: PathBuf,
    repo: PathBuf,
}

impl KindCluster {
    fn context(&self) -> String {
        format!("kind-{}", self.name)
    }

    async fn create(&self) -> Result<()> {
        run_checked(
            "kind",
            &[
                "create".into(),
                "cluster".into(),
                "--name".into(),
                self.name.clone(),
                "--kubeconfig".into(),
                self.kubeconfig.display().to_string(),
                "--wait".into(),
                "180s".into(),
                "--retain".into(),
            ],
            &self.repo,
            Duration::from_secs(5 * 60),
            COMMAND_OUTPUT_LIMIT,
        )
        .await
        .with_context(|| format!("create uniquely owned Kind cluster {}", self.name))?;
        Ok(())
    }

    async fn load_images(&self, worker_image: &str, frontend_image: &str) -> Result<()> {
        let args = strings(&[
            "load",
            "docker-image",
            worker_image,
            frontend_image,
            "--name",
            self.name.as_str(),
        ]);
        run_checked(
            "kind",
            &args,
            &self.repo,
            Duration::from_secs(10 * 60),
            COMMAND_OUTPUT_LIMIT,
        )
        .await
        .context("side-load current-tree images into Kind")?;
        Ok(())
    }

    async fn preloaded_etcd_image(&self) -> Result<String> {
        let output = run_capture(
            "docker",
            &strings(&[
                "exec",
                &format!("{}-control-plane", self.name),
                "crictl",
                "images",
                "--output",
                "json",
            ]),
            &self.repo,
            COMMAND_TIMEOUT,
            COMMAND_OUTPUT_LIMIT,
        )
        .await
        .context("inspect images preloaded in the Kind node")?;
        ensure!(
            output.status.success(),
            "inspect Kind node images failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let images: Value =
            serde_json::from_slice(&output.stdout).context("decode Kind node image inventory")?;
        images
            .get("images")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .flat_map(|image| {
                image
                    .get("repoTags")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .filter_map(Value::as_str)
            .find(|tag| tag.starts_with("registry.k8s.io/etcd:"))
            .map(str::to_owned)
            .context("Kind node does not contain its bootstrap etcd image")
    }

    async fn deploy(&self, manifest: &Path) -> Result<()> {
        self.kubectl(
            false,
            &["create".into(), "namespace".into(), self.namespace.clone()],
            COMMAND_TIMEOUT,
        )
        .await
        .context("create unique Kind test namespace")?;
        self.kubectl(
            true,
            &[
                "apply".into(),
                "--filename".into(),
                manifest.display().to_string(),
            ],
            COMMAND_TIMEOUT,
        )
        .await
        .context("apply CPU-only speculative decoding topology")?;
        for deployment in ["etcd", "draft", "target", "frontend"] {
            self.kubectl(
                true,
                &[
                    "rollout".into(),
                    "status".into(),
                    format!("deployment/{deployment}"),
                    "--timeout=180s".into(),
                ],
                Duration::from_secs(190),
            )
            .await
            .with_context(|| format!("wait for {deployment} deployment"))?;
        }
        Ok(())
    }

    fn start_port_forward(&self) -> Result<PortForward> {
        let listener = TcpListener::bind("127.0.0.1:0").context("reserve local HTTP port")?;
        let address = listener.local_addr().context("read reserved HTTP port")?;
        drop(listener);

        let args = self.kubectl_args(
            true,
            &[
                "port-forward".into(),
                "--address".into(),
                "127.0.0.1".into(),
                "service/frontend".into(),
                format!("{}:8000", address.port()),
            ],
        );
        let mut command = Command::new("kubectl");
        command
            .args(&args)
            .current_dir(&self.repo)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let process =
            OwnedProcess::spawn(command, "start kubectl port-forward", COMMAND_OUTPUT_LIMIT)?;
        Ok(PortForward { address, process })
    }

    async fn logs(&self, deployment: &str, tail: usize) -> Result<String> {
        self.logs_with_timeout(deployment, tail, COMMAND_TIMEOUT)
            .await
    }

    async fn logs_with_timeout(
        &self,
        deployment: &str,
        tail: usize,
        command_timeout: Duration,
    ) -> Result<String> {
        let output = self
            .kubectl(
                true,
                &[
                    "logs".into(),
                    format!("deployment/{deployment}"),
                    "--all-containers=true".into(),
                    format!("--tail={tail}"),
                ],
                command_timeout,
            )
            .await
            .with_context(|| format!("collect {deployment} logs"))?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    async fn ready_draft_pod(&self) -> Result<String> {
        let output = self
            .kubectl(
                true,
                &[
                    "get".into(),
                    "pods".into(),
                    "--selector=app=specdec-draft".into(),
                    "--output=json".into(),
                ],
                COMMAND_TIMEOUT,
            )
            .await?;
        ready_pod_name(&output.stdout, None)?
            .context("draft deployment has no ready pod before replacement")
    }

    async fn replace_draft_pod(&self, old_pod: &str) -> Result<String> {
        self.kubectl(
            true,
            &[
                "delete".into(),
                "pod".into(),
                old_pod.into(),
                "--wait=true".into(),
                "--timeout=60s".into(),
            ],
            Duration::from_secs(70),
        )
        .await
        .context("delete the owned draft pod")?;

        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            let command_budget = remaining_budget(deadline, "wait for replacement draft pod")?;
            let output = self
                .kubectl(
                    true,
                    &[
                        "get".into(),
                        "pods".into(),
                        "--selector=app=specdec-draft".into(),
                        "--output=json".into(),
                    ],
                    command_budget,
                )
                .await?;
            let last_state = String::from_utf8_lossy(&output.stdout).into_owned();
            if let Some(name) = ready_pod_name(&output.stdout, Some(old_pod))? {
                return Ok(name);
            }
            ensure!(
                Instant::now() < deadline,
                "replacement draft pod did not become ready: {last_state}"
            );
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn kubectl(
        &self,
        namespaced: bool,
        args: &[String],
        deadline: Duration,
    ) -> Result<CapturedOutput> {
        run_checked(
            "kubectl",
            &self.kubectl_args(namespaced, args),
            &self.repo,
            deadline,
            COMMAND_OUTPUT_LIMIT,
        )
        .await
    }

    fn kubectl_args(&self, namespaced: bool, args: &[String]) -> Vec<String> {
        let mut command = vec![
            "--kubeconfig".into(),
            self.kubeconfig.display().to_string(),
            "--context".into(),
            self.context(),
        ];
        if namespaced {
            command.extend(["--namespace".into(), self.namespace.clone()]);
        }
        command.extend(args.iter().cloned());
        command
    }

    async fn diagnostics(&self) -> String {
        let commands = [
            ("all resources", strings(&["get", "all", "--output=wide"])),
            ("pod descriptions", strings(&["describe", "pods"])),
            (
                "events",
                strings(&["get", "events", "--sort-by=.metadata.creationTimestamp"]),
            ),
        ];
        let mut diagnostics = String::new();
        for (label, args) in commands {
            append_diagnostic(
                &mut diagnostics,
                label,
                run_capture(
                    "kubectl",
                    &self.kubectl_args(true, &args),
                    &self.repo,
                    Duration::from_secs(30),
                    DIAGNOSTIC_OUTPUT_LIMIT,
                )
                .await,
            );
        }
        for deployment in ["etcd", "frontend", "target", "draft"] {
            let args = vec![
                "logs".into(),
                format!("deployment/{deployment}"),
                "--all-containers=true".into(),
                "--prefix=true".into(),
                "--tail=1000".into(),
            ];
            append_diagnostic(
                &mut diagnostics,
                &format!("{deployment} logs"),
                run_capture(
                    "kubectl",
                    &self.kubectl_args(true, &args),
                    &self.repo,
                    Duration::from_secs(30),
                    DIAGNOSTIC_OUTPUT_LIMIT,
                )
                .await,
            );
        }
        diagnostics
    }

    async fn delete(&self) -> Result<()> {
        run_checked(
            "kind",
            &[
                "delete".into(),
                "cluster".into(),
                "--name".into(),
                self.name.clone(),
                "--kubeconfig".into(),
                self.kubeconfig.display().to_string(),
            ],
            &self.repo,
            Duration::from_secs(3 * 60),
            COMMAND_OUTPUT_LIMIT,
        )
        .await
        .with_context(|| format!("delete exact owned Kind cluster {}", self.name))?;
        Ok(())
    }
}

fn ready_pod_name(raw: &[u8], excluded: Option<&str>) -> Result<Option<String>> {
    let value: Value = serde_json::from_slice(raw).context("decode kubectl pod JSON")?;
    let items = value
        .get("items")
        .and_then(Value::as_array)
        .context("kubectl pod JSON omitted items")?;
    Ok(items.iter().find_map(|pod| {
        let name = pod.pointer("/metadata/name")?.as_str()?;
        if excluded == Some(name) || pod.pointer("/status/phase")?.as_str()? != "Running" {
            return None;
        }
        let ready = pod
            .pointer("/status/conditions")?
            .as_array()?
            .iter()
            .any(|condition| {
                condition.get("type").and_then(Value::as_str) == Some("Ready")
                    && condition.get("status").and_then(Value::as_str) == Some("True")
            });
        ready.then(|| name.to_string())
    }))
}

struct PortForward {
    address: SocketAddr,
    process: OwnedProcess,
}

impl PortForward {
    fn ensure_running(&mut self) -> Result<()> {
        if let Some(status) = self
            .process
            .child
            .try_wait()
            .context("poll kubectl port-forward")?
        {
            self.process.reaped = true;
            bail!("kubectl port-forward exited before frontend readiness with {status}");
        }
        Ok(())
    }

    async fn shutdown(mut self) -> Result<String> {
        let mut errors = Vec::new();
        match self.process.child.try_wait() {
            Ok(Some(status)) => {
                self.process.reaped = true;
                if !status.success() {
                    errors.push(format!("port-forward exited with {status}"));
                }
            }
            Ok(None) => {
                if let Err(error) = self.process.stop_and_reap(PROCESS_STOP_TIMEOUT).await {
                    errors.push(format!("stop and reap port-forward: {error:#}"));
                }
            }
            Err(error) => {
                errors.push(format!("poll port-forward during shutdown: {error}"));
                if let Err(error) = self.process.stop_and_reap(PROCESS_STOP_TIMEOUT).await {
                    errors.push(format!("stop and reap port-forward: {error:#}"));
                }
            }
        }
        let (stdout, stderr) = self.process.collect_readers(PROCESS_STOP_TIMEOUT).await;
        let stdout = unpack_reader("port-forward stdout", stdout, &mut errors);
        let stderr = unpack_reader("port-forward stderr", stderr, &mut errors);
        let diagnostic = format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&stdout.bytes),
            String::from_utf8_lossy(&stderr.bytes)
        );
        ensure!(
            errors.is_empty(),
            "kubectl port-forward cleanup failed: {}\n{diagnostic}",
            errors.join("; "),
        );
        Ok(diagnostic)
    }
}

async fn wait_for_model(
    client: &Client,
    address: SocketAddr,
    port_forward: &mut PortForward,
) -> Result<()> {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        port_forward.ensure_running()?;
        let last_error = match http_json(client, address, Method::GET, "/v1/models", None).await {
            Ok(response) if response.status == 200 => {
                let body: Value = serde_json::from_slice(&response.body)
                    .context("decode Kind frontend model list")?;
                let ready = body
                    .get("data")
                    .and_then(Value::as_array)
                    .is_some_and(|models| {
                        models.iter().any(|model| {
                            model.get("id").and_then(Value::as_str) == Some(MODEL_NAME)
                        })
                    });
                if ready {
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
            "Kind frontend readiness timed out: {last_error}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[derive(Debug)]
struct Evidence {
    request_id: String,
    proposal_digest: String,
}

async fn completion(client: &Client, address: SocketAddr, prompt: Vec<u32>) -> Result<Evidence> {
    completion_with_max_tokens(client, address, prompt, 4).await
}

async fn completion_with_max_tokens(
    client: &Client,
    address: SocketAddr,
    prompt: Vec<u32>,
    max_tokens: usize,
) -> Result<Evidence> {
    let expected_digest =
        proposal_digest(&prompt.iter().copied().cycle().take(4).collect::<Vec<_>>());
    let response = http_json(
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
    .context("Kind completion request failed")?;
    ensure!(
        response.status == 200,
        "Kind completion returned HTTP {}: {}",
        response.status,
        String::from_utf8_lossy(&response.body)
    );
    let body: Value =
        serde_json::from_slice(&response.body).context("decode Kind completion response")?;
    ensure!(
        body.pointer("/choices/0/finish_reason") == Some(&Value::String("length".into())),
        "Kind completion did not terminate at the requested bound: {body}"
    );
    ensure!(
        body.pointer("/nvext/engine_data/_dynamo_external_speculation_v1")
            .is_none(),
        "internal speculative lifecycle marker leaked into Kind HTTP response: {body}"
    );
    let evidence = body
        .pointer("/nvext/engine_data/mock_specdec")
        .context("Kind completion omitted mock speculative evidence")?;
    let request_id = evidence
        .get("request_id")
        .and_then(Value::as_str)
        .context("Kind completion omitted request ID")?
        .to_string();
    let digest = evidence
        .get("proposal_digest")
        .and_then(Value::as_str)
        .context("Kind completion omitted proposal digest")?
        .to_string();
    Uuid::parse_str(&request_id).context("Kind completion returned an invalid request ID")?;
    ensure!(
        digest == expected_digest,
        "Kind completion crossed proposal streams: expected {expected_digest}, got {digest}"
    );
    Ok(Evidence {
        request_id,
        proposal_digest: digest,
    })
}

async fn retry_completion_after_replacement(
    client: &Client,
    address: SocketAddr,
    prompt: Vec<u32>,
) -> Result<Evidence> {
    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        let last_error = match completion(client, address, prompt.clone()).await {
            Ok(evidence) => return Ok(evidence),
            Err(error) => format!("{error:#}"),
        };
        ensure!(
            Instant::now() < deadline,
            "frontend did not recover after draft replacement: {last_error}"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

async fn http_json(
    client: &Client,
    address: SocketAddr,
    method: Method,
    path: &str,
    body: Option<&Value>,
) -> Result<HttpResponse> {
    let mut request = client.request(method, format!("http://{address}{path}"));
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request.send().await.context("send Kind frontend request")?;
    let status = response.status().as_u16();
    if let Some(content_length) = response.content_length() {
        ensure!(
            content_length <= HTTP_RESPONSE_LIMIT as u64,
            "Kind frontend response exceeded the harness limit"
        );
    }
    let mut body = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.context("read Kind frontend response")?;
        ensure!(
            body.len().saturating_add(chunk.len()) <= HTTP_RESPONSE_LIMIT,
            "Kind frontend response exceeded the harness limit"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(HttpResponse { status, body })
}

fn assert_correlated(target_logs: &str, draft_logs: &str, evidence: &[Evidence]) -> Result<()> {
    for item in evidence {
        ensure!(
            matching_event(
                target_logs,
                &[
                    ("message", "mock speculative target consumed draft proposal"),
                    ("request_id", &item.request_id),
                    ("proposal_digest", &item.proposal_digest),
                ],
            )
            .is_some(),
            "target logs omitted correlated proposal evidence for {}",
            item.request_id
        );
        ensure!(
            matching_event(
                draft_logs,
                &[
                    ("message", "mock speculative draft completed proposal"),
                    ("request_id", &item.request_id),
                    ("proposal_digest", &item.proposal_digest),
                ],
            )
            .is_some(),
            "draft logs omitted correlated proposal evidence for {}",
            item.request_id
        );
        ensure!(
            matching_event(
                draft_logs,
                &[
                    ("message", "mock speculative draft cleaned request"),
                    ("request_id", &item.request_id),
                ],
            )
            .is_some(),
            "draft logs omitted cleanup evidence for {}",
            item.request_id
        );
    }
    Ok(())
}

fn assert_overlap(target_logs: &str, request_id: &str) -> Result<()> {
    ensure!(
        matching_event(
            target_logs,
            &[
                (
                    "message",
                    "mock speculative draft START preceded target prefill completion",
                ),
                ("request_id", request_id),
            ],
        )
        .is_some(),
        "Kind target did not record target-prefill/draft overlap for {request_id}"
    );
    Ok(())
}

fn assert_cancelled_session_reaped(target_logs: &str, draft_logs: &str) -> Result<()> {
    let cancellation = matching_event(
        target_logs,
        &[("message", "mock speculative target cancelled during setup")],
    )
    .context("Kind target did not record client cancellation")?;
    let request_id = field_string(&cancellation, "request_id")
        .context("Kind target cancellation omitted request ID")?;
    let cleaned = matching_event(
        draft_logs,
        &[
            ("message", "mock speculative draft cleaned request"),
            ("request_id", &request_id),
            ("active_sessions", "0"),
        ],
    )
    .or_else(|| {
        matching_event(
            draft_logs,
            &[
                ("message", "reaped orphaned mock draft session"),
                ("request_id", &request_id),
                ("active_sessions", "0"),
            ],
        )
    });
    ensure!(
        cleaned.is_some(),
        "cancelled Kind request {request_id} did not leave the draft at zero active sessions"
    );
    Ok(())
}

fn accepted_request_count(logs: &str) -> usize {
    logs.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|event| {
            field_string(event, "message").as_deref()
                == Some("mock speculative draft accepted request")
        })
        .count()
}

async fn wait_for_new_draft_request(
    cluster: &KindCluster,
    accepted_before: usize,
    request: &AbortOnDropTask<Result<Evidence>>,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        ensure!(
            !request.is_finished(),
            "Kind cancellation probe completed before it could be cancelled"
        );
        let logs = cluster
            .logs_with_timeout("draft", 5_000, LOG_POLL_COMMAND_TIMEOUT)
            .await?;
        if accepted_request_count(&logs) > accepted_before {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "Kind cancellation probe never reached the draft worker"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_cancelled_session_reaped(cluster: &KindCluster) -> Result<(String, String)> {
    let deadline = Instant::now() + Duration::from_secs(6);
    loop {
        let target_logs = cluster
            .logs_with_timeout("target", 5_000, LOG_POLL_COMMAND_TIMEOUT)
            .await?;
        let draft_logs = cluster
            .logs_with_timeout("draft", 5_000, LOG_POLL_COMMAND_TIMEOUT)
            .await?;
        match assert_cancelled_session_reaped(&target_logs, &draft_logs) {
            Ok(()) => return Ok((target_logs, draft_logs)),
            Err(error) if Instant::now() < deadline => {
                tracing::debug!(%error, "waiting for Kind cancellation cleanup evidence");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(error) => return Err(error),
        }
    }
}

fn remaining_budget(deadline: Instant, phase: &str) -> Result<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    ensure!(!remaining.is_zero(), "{phase} exceeded its deadline");
    Ok(remaining.min(COMMAND_TIMEOUT))
}

fn draft_incarnation(logs: &str) -> Result<String> {
    let startup = matching_event(logs, &[("message", "mock speculative draft started")])
        .context("draft logs omitted startup event")?;
    field_string(&startup, "draft_incarnation").context("draft startup event omitted incarnation")
}

fn matching_event(logs: &str, fields: &[(&str, &str)]) -> Option<Value> {
    logs.lines().find_map(|line| {
        let event: Value = serde_json::from_str(line).ok()?;
        fields
            .iter()
            .all(|(name, expected)| field_string(&event, name).as_deref() == Some(*expected))
            .then_some(event)
    })
}

fn field_string(event: &Value, name: &str) -> Option<String> {
    event.get(name).and_then(|value| match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

struct CapturedOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_truncated: bool,
    stderr_truncated: bool,
}

impl CapturedOutput {
    fn diagnostic(&self) -> String {
        format!(
            "status={}\nstdout{}:\n{}\nstderr{}:\n{}",
            self.status,
            if self.stdout_truncated {
                " (truncated to tail)"
            } else {
                ""
            },
            String::from_utf8_lossy(&self.stdout),
            if self.stderr_truncated {
                " (truncated to tail)"
            } else {
                ""
            },
            String::from_utf8_lossy(&self.stderr)
        )
    }
}

struct TailOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

struct AbortOnDropTask<T> {
    handle: Option<JoinHandle<T>>,
}

impl<T> AbortOnDropTask<T> {
    fn new(handle: JoinHandle<T>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    fn abort(&self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }

    fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }

    async fn join(mut self) -> std::result::Result<T, tokio::task::JoinError> {
        let result = self
            .handle
            .as_mut()
            .expect("abort-on-drop task has a join handle")
            .await;
        self.handle.take();
        result
    }

    async fn join_bounded(mut self, deadline: Duration) -> Result<T> {
        let handle = self
            .handle
            .as_mut()
            .context("abort-on-drop task has no join handle")?;
        match timeout(deadline, handle).await {
            Ok(result) => {
                self.handle.take();
                result.context("child output task failed")
            }
            Err(_) => {
                self.abort();
                if let Some(handle) = self.handle.as_mut() {
                    let _ = handle.await;
                }
                self.handle.take();
                bail!("child output collection exceeded {deadline:?}")
            }
        }
    }
}

impl<T> Drop for AbortOnDropTask<T> {
    fn drop(&mut self) {
        self.abort();
    }
}

type ReaderTask = AbortOnDropTask<std::io::Result<TailOutput>>;

struct OwnedProcess {
    child: Child,
    stdout: Option<ReaderTask>,
    stderr: Option<ReaderTask>,
    reaped: bool,
}

impl OwnedProcess {
    fn spawn(mut command: Command, context: &str, output_limit: usize) -> Result<Self> {
        let mut child = command.spawn().with_context(|| context.to_string())?;
        let stdout = child.stdout.take().context("capture command stdout")?;
        let stderr = child.stderr.take().context("capture command stderr")?;
        Ok(Self {
            child,
            stdout: Some(AbortOnDropTask::new(tokio::spawn(read_tail(
                stdout,
                output_limit,
            )))),
            stderr: Some(AbortOnDropTask::new(tokio::spawn(read_tail(
                stderr,
                output_limit,
            )))),
            reaped: false,
        })
    }

    async fn stop_and_reap(&mut self, deadline: Duration) -> Result<()> {
        if self.reaped {
            return Ok(());
        }
        let mut errors = Vec::new();
        match self.child.try_wait() {
            Ok(Some(_)) => {
                self.reaped = true;
                return Ok(());
            }
            Ok(None) => {}
            Err(error) => errors.push(format!("poll child before kill: {error}")),
        }
        if let Err(error) = self.child.start_kill() {
            errors.push(format!("request child kill: {error}"));
        }
        match timeout(deadline, self.child.wait()).await {
            Ok(Ok(_)) => self.reaped = true,
            Ok(Err(error)) => errors.push(format!("reap child: {error}")),
            Err(_) => errors.push(format!("child did not stop within {deadline:?}")),
        }
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }

    async fn collect_readers(
        &mut self,
        deadline: Duration,
    ) -> (Result<TailOutput>, Result<TailOutput>) {
        let (stdout, stderr) = self.take_readers();
        let stdout = collect_optional_reader(stdout, "stdout", deadline);
        let stderr = collect_optional_reader(stderr, "stderr", deadline);
        tokio::join!(stdout, stderr)
    }

    fn take_readers(&mut self) -> (Option<ReaderTask>, Option<ReaderTask>) {
        (self.stdout.take(), self.stderr.take())
    }
}

impl Drop for OwnedProcess {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.start_kill();
        }
        if let Some(task) = &self.stdout {
            task.abort();
        }
        if let Some(task) = &self.stderr {
            task.abort();
        }
    }
}

async fn run_checked(
    program: &str,
    args: &[String],
    cwd: &Path,
    deadline: Duration,
    output_limit: usize,
) -> Result<CapturedOutput> {
    let output = run_capture(program, args, cwd, deadline, output_limit).await?;
    ensure!(
        output.status.success(),
        "command `{program} {}` failed\n{}",
        args.join(" "),
        output.diagnostic()
    );
    Ok(output)
}

async fn run_capture(
    program: &str,
    args: &[String],
    cwd: &Path,
    deadline: Duration,
    output_limit: usize,
) -> Result<CapturedOutput> {
    let hard_deadline = Instant::now() + deadline;
    let cleanup_reserve = (deadline / 2).min(PROCESS_STOP_TIMEOUT);
    let execution_deadline = hard_deadline - cleanup_reserve;
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut process = OwnedProcess::spawn(
        command,
        &format!("spawn `{program} {}`", args.join(" ")),
        output_limit,
    )?;
    let mut errors = Vec::new();
    let status = match tokio::time::timeout_at(execution_deadline, process.child.wait()).await {
        Ok(Ok(status)) => {
            process.reaped = true;
            Some(status)
        }
        Ok(Err(error)) => {
            errors.push(format!("wait for child command: {error}"));
            None
        }
        Err(_) => {
            errors.push(format!("command exceeded {deadline:?}"));
            None
        }
    };
    let cleanup_budget = remaining_budget(hard_deadline, "clean up child command")?;
    let (stdout, stderr) = if status.is_none() {
        let (stdout, stderr) = process.take_readers();
        let stop = process.stop_and_reap(cleanup_budget);
        let stdout = collect_optional_reader(stdout, "stdout", cleanup_budget);
        let stderr = collect_optional_reader(stderr, "stderr", cleanup_budget);
        let (stop, stdout, stderr) = tokio::join!(stop, stdout, stderr);
        if let Err(error) = stop {
            errors.push(format!("stop and reap command: {error:#}"));
        }
        (stdout, stderr)
    } else {
        process.collect_readers(cleanup_budget).await
    };
    let stdout = unpack_reader("command stdout", stdout, &mut errors);
    let stderr = unpack_reader("command stderr", stderr, &mut errors);
    if !errors.is_empty() {
        bail!(
            "command `{program} {}` failed lifecycle cleanup: {}\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            errors.join("; "),
            String::from_utf8_lossy(&stdout.bytes),
            String::from_utf8_lossy(&stderr.bytes)
        );
    }
    Ok(CapturedOutput {
        status: status.context("command status missing after successful cleanup")?,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
    })
}

async fn read_tail(
    mut reader: impl AsyncRead + Unpin,
    limit: usize,
) -> std::io::Result<TailOutput> {
    let mut chunks = VecDeque::<Vec<u8>>::new();
    let mut retained = 0_usize;
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut chunk).await?;
        if read == 0 {
            let mut bytes = Vec::with_capacity(retained);
            for chunk in chunks {
                bytes.extend_from_slice(&chunk);
            }
            return Ok(TailOutput { bytes, truncated });
        }
        if limit == 0 {
            truncated = true;
            continue;
        }
        if read >= limit {
            chunks.clear();
            chunks.push_back(chunk[read - limit..read].to_vec());
            retained = limit;
            truncated = true;
            continue;
        }
        chunks.push_back(chunk[..read].to_vec());
        retained += read;
        while retained > limit {
            let overflow = retained - limit;
            let front_len = chunks.front().map_or(0, Vec::len);
            if front_len <= overflow {
                chunks.pop_front();
                retained -= front_len;
            } else {
                chunks
                    .front_mut()
                    .expect("tail buffer has a front chunk")
                    .drain(..overflow);
                retained -= overflow;
            }
            truncated = true;
        }
    }
}

async fn collect_optional_reader(
    task: Option<ReaderTask>,
    label: &str,
    deadline: Duration,
) -> Result<TailOutput> {
    let task = task.with_context(|| format!("{label} reader task is missing"))?;
    collect_reader(task, deadline).await
}

async fn collect_reader(task: ReaderTask, deadline: Duration) -> Result<TailOutput> {
    task.join_bounded(deadline)
        .await?
        .context("read child output")
}

fn unpack_reader(label: &str, result: Result<TailOutput>, errors: &mut Vec<String>) -> TailOutput {
    match result {
        Ok(output) => output,
        Err(error) => {
            errors.push(format!("{label}: {error:#}"));
            TailOutput {
                bytes: format!("<{label} unavailable: {error:#}>").into_bytes(),
                truncated: false,
            }
        }
    }
}

fn append_diagnostic(output: &mut String, label: &str, result: Result<CapturedOutput>) {
    output.push_str(&format!("\n===== {label} =====\n"));
    match result {
        Ok(captured) => output.push_str(&captured.diagnostic()),
        Err(error) => output.push_str(&format!("diagnostic command failed: {error:#}")),
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}
