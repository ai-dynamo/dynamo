use std::collections::{HashMap, HashSet};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, bail, ensure};
use async_stream::stream;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::LLMMetricAnnotation;
use dynamo_llm::protocols::Annotated;
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_runtime::CancellationToken;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use futures::future::try_join_all;
use futures::{StreamExt, TryStreamExt};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{MissedTickBehavior, timeout};

const MODEL: &str = "stats-e2e";
const STATS_CONSUMER_ID: &str = "stats-e2e";
const ACTUAL_INPUT_TOKENS: u64 = 12;
const POLL_INTERVAL: Duration = Duration::from_millis(25);
const READY_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const WHOLE_TEST_TIMEOUT: Duration = Duration::from_secs(120);
const CHUNK_DELAY: Duration = Duration::from_millis(100);
const METRIC_CONNECTED_REQUIRED: &str = r#"pylon_engine_stats_stream_connected{mode="required"}"#;
const METRIC_EVENT_PING: &str = r#"pylon_engine_stats_stream_events_total{type="ping"}"#;
const METRIC_EVENT_STATS: &str = r#"pylon_engine_stats_stream_events_total{type="stats"}"#;
const METRIC_RECONNECTS: &str = "pylon_engine_stats_stream_reconnects_total";
const METRIC_RECONNECT_CONNECT_ERROR: &str =
    r#"pylon_engine_stats_stream_reconnects_total{reason="connect_error"}"#;
const METRIC_INVALID: &str = "pylon_engine_stats_stream_invalid_events_total";
const METRIC_LIVE_ENGINE_STATS: &str =
    r#"pylon_engine_stats_live_requests{source="engine_stats_stream"}"#;
const METRIC_STATS_SOURCE_ENGINE: &str =
    r#"pylon_model_stats_source{model="stats-e2e",source="engine_stats_stream"}"#;
const METRIC_STATS_CAPABILITY_ENGINE: &str = r#"pylon_model_stats_capability{capability="model.throughput.engine_stream",model="stats-e2e"}"#;
const METRIC_INPUT_TPS: &str = r#"pylon_model_last_mean_input_tps{model="stats-e2e"}"#;
const METRIC_OUTPUT_TPS: &str = r#"pylon_model_output_tps{model="stats-e2e"}"#;
const METRIC_MAX_OUTPUT_TPS: &str = r#"pylon_model_max_output_tps{model="stats-e2e"}"#;
const METRIC_REGISTRATION_CONNECTED: &str = "pylon_registration_stream_connected";

#[derive(Debug)]
struct Args {
    stats_consumer_bin: PathBuf,
    stargate_probe_bin: PathBuf,
    artifact_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args_os().skip(1);
        let mut stats_consumer_bin = None;
        let mut stargate_probe_bin = None;
        let mut artifact_dir = None;
        while let Some(argument) = values.next() {
            let argument = argument.to_string_lossy();
            let mut value = || {
                values
                    .next()
                    .with_context(|| format!("missing value for {argument}"))
            };
            match argument.as_ref() {
                "--stats-consumer-bin" => stats_consumer_bin = Some(PathBuf::from(value()?)),
                "--stargate-probe-bin" => stargate_probe_bin = Some(PathBuf::from(value()?)),
                "--artifact-dir" => artifact_dir = Some(PathBuf::from(value()?)),
                other => bail!("unknown argument: {other}"),
            }
        }
        Ok(Self {
            stats_consumer_bin: stats_consumer_bin.context("--stats-consumer-bin is required")?,
            stargate_probe_bin: stargate_probe_bin.context("--stargate-probe-bin is required")?,
            artifact_dir: artifact_dir.context("--artifact-dir is required")?,
        })
    }
}

#[derive(Clone)]
struct DeterministicEngine;

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        anyhow::Error,
    > for DeterministicEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>> {
        let (request, context) = request.transfer(());
        let context = context.context();
        let mut generator = request.response_generator(context.id().to_string());
        let output = stream! {
            for (index, (output_tokens, chunk_tokens)) in
                [(2usize, 2usize), (5, 3), (9, 4)].into_iter().enumerate()
            {
                if index > 0 {
                    tokio::time::sleep(CHUNK_DELAY).await;
                }
                let mut response = generator.create_choice(
                    0,
                    Some(format!("e2e chunk {}", index + 1)),
                    None,
                    None,
                );
                response.llm_metrics = Some(LLMMetricAnnotation {
                    input_tokens: usize::try_from(ACTUAL_INPUT_TOKENS)
                        .expect("fixture input tokens should fit usize"),
                    output_tokens,
                    chunk_tokens,
                    ..Default::default()
                });
                yield Annotated::from_data(response);
            }
        };
        Ok(ResponseStream::new(Box::pin(output), context))
    }
}

struct PortReservation {
    socket: Socket,
    addr: SocketAddr,
}

impl PortReservation {
    fn ephemeral() -> Result<Self> {
        Self::bind(0)
    }

    fn bind(port: u16) -> Result<Self> {
        let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
        socket.set_reuse_address(true)?;
        socket.bind(&SockAddr::from(SocketAddrV4::new(
            Ipv4Addr::LOCALHOST,
            port,
        )))?;
        let addr = socket
            .local_addr()?
            .as_socket()
            .context("reserved socket has no IP address")?;
        Ok(Self { socket, addr })
    }

    fn port(&self) -> u16 {
        self.addr.port()
    }

    fn into_listener(self) -> Result<tokio::net::TcpListener> {
        self.socket.listen(128)?;
        self.socket.set_nonblocking(true)?;
        let listener: std::net::TcpListener = self.socket.into();
        Ok(tokio::net::TcpListener::from_std(listener)?)
    }
}

struct DynamoFixture {
    port: u16,
    cancel: CancellationToken,
    task: Option<JoinHandle<Result<()>>>,
}

impl DynamoFixture {
    async fn start(reservation: PortReservation) -> Result<Self> {
        let port = reservation.port();
        let listener = reservation
            .into_listener()
            .with_context(|| format!("failed to activate reserved Dynamo port {port}"))?;
        let service = HttpService::builder()
            .host(Ipv4Addr::LOCALHOST.to_string())
            .port(port)
            .enable_chat_endpoints(true)
            .enable_cmpl_endpoints(false)
            .enable_embeddings_endpoints(false)
            .enable_responses_endpoints(false)
            .build()?;
        let card = ModelDeploymentCard::with_name_only(MODEL);
        service.model_manager().add_chat_completions_model(
            MODEL,
            card.mdcsum(),
            Arc::new(DeterministicEngine),
        )?;
        let cancel = CancellationToken::new();
        let task = service.spawn_with_listener(cancel.clone(), listener).await;
        wait_http_ok(&format!("http://127.0.0.1:{port}/health"), READY_TIMEOUT).await?;
        Ok(Self {
            port,
            cancel,
            task: Some(task),
        })
    }

    async fn stop_and_reserve(mut self) -> Result<PortReservation> {
        self.stop().await?;
        reserve_existing_port(self.port, READY_TIMEOUT).await
    }

    async fn stop(&mut self) -> Result<()> {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            let result = timeout(SHUTDOWN_TIMEOUT, task)
                .await
                .context("Dynamo shutdown timed out")?
                .context("Dynamo task panicked")?;
            result.context("Dynamo service failed during shutdown")?;
        }
        Ok(())
    }
}

impl Drop for DynamoFixture {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn reserve_existing_port(port: u16, deadline: Duration) -> Result<PortReservation> {
    wait_until("reserve stopped Dynamo port", deadline, || async move {
        match PortReservation::bind(port) {
            Ok(reservation) => Ok(Some(reservation)),
            Err(error) if is_addr_in_use(&error) => Ok(None),
            Err(error) => Err(error),
        }
    })
    .await
}

fn is_addr_in_use(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|source| source.downcast_ref::<std::io::Error>())
        .any(|source| source.kind() == std::io::ErrorKind::AddrInUse)
}

struct StargateFixture {
    grpc_addr: SocketAddr,
    http_addr: SocketAddr,
    snapshot_path: PathBuf,
    log_path: PathBuf,
    stdin: Option<tokio::process::ChildStdin>,
    child: Child,
}

impl StargateFixture {
    async fn start(probe_bin: &Path, artifact_dir: &Path) -> Result<Self> {
        let ready_path = artifact_dir.join("stargate-ready.json");
        let snapshot_path = artifact_dir.join("stargate-candidates.json");
        let log_path = artifact_dir.join("stargate.log");
        remove_file_if_present(&ready_path).await?;
        remove_file_if_present(&snapshot_path).await?;

        let arguments = vec![
            "--ready-file".to_string(),
            ready_path.display().to_string(),
            "--snapshot-file".to_string(),
            snapshot_path.display().to_string(),
        ];
        tokio::fs::write(
            artifact_dir.join("stargate.command.txt"),
            format!(
                "{} {}\n",
                probe_bin.display(),
                arguments
                    .iter()
                    .map(|argument| shell_quote(argument))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        )
        .await?;
        let stdout = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)?;
        let stderr = stdout.try_clone()?;
        let mut child = Command::new(probe_bin)
            .args(&arguments)
            .env(
                "RUST_LOG",
                std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .kill_on_drop(true)
            .spawn()
            .context("failed to spawn Stargate state probe")?;
        let stdin = child
            .stdin
            .take()
            .context("failed to open Stargate probe stdin")?;

        let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
        let ready = loop {
            if let Some(status) = child.try_wait()? {
                let logs = tokio::fs::read_to_string(&log_path)
                    .await
                    .unwrap_or_default();
                bail!("Stargate state probe exited with {status}:\n{logs}");
            }
            match tokio::fs::read(&ready_path).await {
                Ok(bytes) => break serde_json::from_slice::<StargateReady>(&bytes)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            ensure!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for Stargate state probe"
            );
            tokio::time::sleep(POLL_INTERVAL).await;
        };
        let grpc_addr = ready.grpc_addr.parse()?;
        let http_addr = ready.http_addr.parse()?;
        wait_http_ok(&format!("http://{http_addr}/healthz"), READY_TIMEOUT).await?;
        let fixture = Self {
            grpc_addr,
            http_addr,
            snapshot_path,
            log_path,
            stdin: Some(stdin),
            child,
        };
        wait_until("initial Stargate state snapshot", READY_TIMEOUT, || async {
            if tokio::fs::try_exists(&fixture.snapshot_path).await? {
                fixture.candidates().await?;
                Ok(Some(()))
            } else {
                Ok(None)
            }
        })
        .await?;
        Ok(fixture)
    }

    async fn candidates(&self) -> Result<Vec<CandidateReport>> {
        let bytes = tokio::fs::read(&self.snapshot_path)
            .await
            .context("failed to read Stargate state snapshot")?;
        let snapshot: StargateSnapshot =
            serde_json::from_slice(&bytes).context("invalid Stargate state snapshot")?;
        let now = unix_time_ms()?;
        ensure!(
            snapshot.written_at_unix_ms <= now.saturating_add(1_000)
                && now.saturating_sub(snapshot.written_at_unix_ms) <= 1_000,
            "Stargate state snapshot is stale"
        );
        Ok(snapshot.candidates)
    }

    async fn shutdown(mut self) -> Result<()> {
        if self.child.try_wait()?.is_none() {
            if let Some(mut stdin) = self.stdin.take() {
                stdin
                    .write_all(b"\n")
                    .await
                    .context("failed to signal Stargate probe shutdown")?;
            }
        }
        let status = timeout(SHUTDOWN_TIMEOUT + Duration::from_secs(2), self.child.wait())
            .await
            .context("Stargate shutdown timed out")??;
        ensure!(
            status.success(),
            "Stargate state probe failed with {status}:\n{}",
            tokio::fs::read_to_string(&self.log_path)
                .await
                .unwrap_or_default()
        );
        Ok(())
    }
}

impl Drop for StargateFixture {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

#[derive(Deserialize)]
struct StargateReady {
    grpc_addr: String,
    http_addr: String,
}

#[derive(Deserialize)]
struct StargateSnapshot {
    written_at_unix_ms: u64,
    candidates: Vec<CandidateReport>,
}

struct StatsConsumerProcess {
    child: Child,
    metrics_addr: SocketAddr,
    log_path: PathBuf,
}

async fn wait_for_child_tcp_listener(child: &mut Child, deadline: Duration) -> Result<SocketAddr> {
    let pid = child.id().context("stats consumer has no process ID")?;
    let deadline = tokio::time::Instant::now() + deadline;
    loop {
        if let Some(status) = child.try_wait()? {
            bail!("stats consumer exited before opening its metrics listener with {status}");
        }
        if let Some(addr) = child_tcp_listener(pid).await? {
            return Ok(addr);
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out discovering stats consumer metrics listener"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn child_tcp_listener(pid: u32) -> Result<Option<SocketAddr>> {
    let mut socket_inodes = HashSet::new();
    let mut entries = tokio::fs::read_dir(format!("/proc/{pid}/fd"))
        .await
        .context("failed to inspect stats consumer file descriptors")?;
    while let Some(entry) = entries.next_entry().await? {
        let target = match tokio::fs::read_link(entry.path()).await {
            Ok(target) => target,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if let Some(inode) = target
            .to_str()
            .and_then(|target| target.strip_prefix("socket:["))
            .and_then(|target| target.strip_suffix(']'))
            .and_then(|inode| inode.parse::<u64>().ok())
        {
            socket_inodes.insert(inode);
        }
    }

    let tcp = tokio::fs::read_to_string(format!("/proc/{pid}/net/tcp"))
        .await
        .context("failed to inspect stats consumer TCP sockets")?;
    let mut listeners = Vec::new();
    for line in tcp.lines().skip(1) {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.get(3) != Some(&"0A") {
            continue;
        }
        let Some(inode) = fields.get(9).and_then(|inode| inode.parse::<u64>().ok()) else {
            continue;
        };
        if !socket_inodes.contains(&inode) {
            continue;
        }
        let local = fields
            .get(1)
            .context("stats consumer TCP listener has no local address")?;
        listeners.push(parse_proc_ipv4_addr(local)?);
    }
    ensure!(
        listeners.len() <= 1,
        "stats consumer opened multiple TCP listeners: {listeners:?}"
    );
    Ok(listeners.pop())
}

fn parse_proc_ipv4_addr(value: &str) -> Result<SocketAddr> {
    let (address, port) = value
        .split_once(':')
        .with_context(|| format!("invalid /proc TCP address: {value}"))?;
    let address = u32::from_str_radix(address, 16)
        .with_context(|| format!("invalid /proc IPv4 address: {address}"))?;
    let port =
        u16::from_str_radix(port, 16).with_context(|| format!("invalid /proc TCP port: {port}"))?;
    let address = Ipv4Addr::from(address.to_le_bytes());
    ensure!(
        address == Ipv4Addr::LOCALHOST && port != 0,
        "unexpected stats consumer metrics listener address: {address}:{port}"
    );
    Ok(SocketAddr::from((address, port)))
}

impl StatsConsumerProcess {
    async fn spawn(
        stats_consumer_bin: &Path,
        artifact_dir: &Path,
        ordinal: usize,
        upstream_port: u16,
        stargate_grpc_addr: SocketAddr,
    ) -> Result<Self> {
        let log_path = artifact_dir.join(format!("stats-consumer-{ordinal}.log"));
        let command_path = artifact_dir.join(format!("stats-consumer-{ordinal}.command.txt"));
        let metrics_addr_path =
            artifact_dir.join(format!("stats-consumer-{ordinal}.metrics-addr.txt"));
        let arguments = vec![
            "--upstream-http-base-url".to_string(),
            format!("http://127.0.0.1:{upstream_port}"),
            "--model-name".to_string(),
            MODEL.to_string(),
            "--stargate-address".to_string(),
            stargate_grpc_addr.to_string(),
            "--inference-server-id".to_string(),
            STATS_CONSUMER_ID.to_string(),
            "--cluster-id".to_string(),
            STATS_CONSUMER_ID.to_string(),
            "--quic-listen-addr".to_string(),
            "127.0.0.1:0".to_string(),
            "--backend-connectivity".to_string(),
            "direct".to_string(),
            "--disable-bringup".to_string(),
            "--initial-input-tps".to_string(),
            "1".to_string(),
            "--engine-stats-stream".to_string(),
            "required".to_string(),
            "--engine-stats-stream-path".to_string(),
            "/v1/stats/stream".to_string(),
            "--min-update-interval-ms".to_string(),
            "100".to_string(),
            "--pylon-queue-mismatch-retry-enabled".to_string(),
            "false".to_string(),
            "--metrics-host".to_string(),
            Ipv4Addr::LOCALHOST.to_string(),
            "--metrics-port".to_string(),
            "0".to_string(),
        ];
        tokio::fs::write(
            &command_path,
            format!(
                "{} {}\n",
                stats_consumer_bin.display(),
                arguments
                    .iter()
                    .map(|argument| shell_quote(argument))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        )
        .await?;
        let stdout = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&log_path)?;
        let stderr = stdout.try_clone()?;
        let mut command = Command::new(stats_consumer_bin);
        command
            .args(&arguments)
            .env(
                "RUST_LOG",
                std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            )
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .kill_on_drop(true);
        let child = command.spawn().context("failed to spawn stats consumer")?;
        let mut process = Self {
            child,
            metrics_addr: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            log_path,
        };
        let startup = async {
            process.metrics_addr =
                wait_for_child_tcp_listener(&mut process.child, READY_TIMEOUT).await?;
            tokio::fs::write(&metrics_addr_path, format!("{}\n", process.metrics_addr)).await?;
            process.wait_metrics_ready(READY_TIMEOUT).await
        }
        .await;
        match startup {
            Ok(()) => Ok(process),
            Err(error) => {
                process.force_stop().await;
                Err(error.context(format!(
                    "stats consumer startup logs:\n{}",
                    process.last_logs()
                )))
            }
        }
    }

    fn metrics_url(&self) -> String {
        format!("http://{}/metrics", self.metrics_addr)
    }

    fn ensure_alive(&mut self) -> Result<()> {
        if let Some(status) = self.child.try_wait()? {
            bail!(
                "stats consumer exited unexpectedly with {status}; last logs:\n{}",
                self.last_logs()
            );
        }
        Ok(())
    }

    async fn scrape(&mut self) -> Result<PrometheusSnapshot> {
        self.ensure_alive()?;
        let text = reqwest::get(self.metrics_url())
            .await
            .context("failed to scrape stats consumer metrics")?
            .error_for_status()
            .context("stats consumer metrics returned an error")?
            .text()
            .await?;
        PrometheusSnapshot::parse(text)
    }

    async fn wait_metrics_ready(&mut self, deadline: Duration) -> Result<()> {
        let url = self.metrics_url();
        let deadline = tokio::time::Instant::now() + deadline;
        loop {
            if let Some(status) = self.child.try_wait()? {
                bail!("stats consumer exited during startup with {status}");
            }
            match reqwest::get(&url).await {
                Ok(response) if response.status().is_success() => return Ok(()),
                Ok(_) | Err(_) => {}
            }
            ensure!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for stats consumer metrics server"
            );
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn stop(&mut self) -> Result<()> {
        if self.child.try_wait()?.is_none() {
            signal_interrupt(&mut self.child, "stats consumer").await?;
            if timeout(SHUTDOWN_TIMEOUT, self.child.wait()).await.is_err() {
                self.child.start_kill()?;
                let _ = self.child.wait().await;
                bail!("stats consumer did not stop within the shutdown timeout");
            }
        }
        Ok(())
    }

    async fn force_stop(&mut self) {
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
    }

    fn last_logs(&self) -> String {
        std::fs::read_to_string(&self.log_path).unwrap_or_default()
    }
}

impl Drop for StatsConsumerProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn shell_quote(value: &str) -> String {
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"-_./:".contains(&byte))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StatsEvent {
    v: u64,
    #[serde(rename = "type")]
    event_type: String,
    request_id: String,
    model: String,
    #[serde(default)]
    tokens_processed: Option<u64>,
    #[serde(default)]
    tokens_generated: Option<u64>,
    #[serde(default)]
    finished: bool,
}

enum DiagnosticMessage {
    Ping,
    Stats(StatsEvent),
    Eof,
    Error(String),
}

struct DiagnosticStream {
    receiver: mpsc::UnboundedReceiver<DiagnosticMessage>,
    task: JoinHandle<()>,
}

impl DiagnosticStream {
    async fn connect(port: u16, raw_path: &Path) -> Result<Self> {
        let response = reqwest::Client::new()
            .get(format!("http://127.0.0.1:{port}/v1/stats/stream"))
            .header(ACCEPT, "application/x-ndjson")
            .send()
            .await?
            .error_for_status()?;
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        ensure!(
            content_type.starts_with("application/x-ndjson"),
            "unexpected stats content type: {content_type}"
        );
        let (sender, receiver) = mpsc::unbounded_channel();
        let raw_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(raw_path)
            .await?;
        let task = tokio::spawn(read_diagnostic_stream(response, raw_file, sender));
        let mut stream = Self { receiver, task };
        let first = timeout(READY_TIMEOUT, stream.receiver.recv())
            .await
            .context("timed out waiting for immediate Dynamo ping")?
            .context("Dynamo diagnostic stream closed before its ping")?;
        ensure!(
            matches!(first, DiagnosticMessage::Ping),
            "first Dynamo diagnostic event was not a ping"
        );
        Ok(stream)
    }

    async fn collect_batch(
        &mut self,
        requests: &[RequestSpec],
        success_batch: bool,
    ) -> Result<CollectedBatch> {
        let gateway_request_ids = requests
            .iter()
            .map(|request| request.id.as_str())
            .collect::<HashSet<_>>();
        let mut events: HashMap<String, Vec<StatsEvent>> = HashMap::new();
        let mut finished = HashSet::new();
        let collect = async {
            while finished.len() < requests.len() {
                match self
                    .receiver
                    .recv()
                    .await
                    .context("Dynamo diagnostic stream closed during batch")?
                {
                    DiagnosticMessage::Ping => {}
                    DiagnosticMessage::Stats(event) => {
                        ensure!(
                            gateway_request_ids.contains(event.request_id.as_str()),
                            "Dynamo stats emitted an unknown request ID {}",
                            event.request_id
                        );
                        if event.finished {
                            ensure!(
                                finished.insert(event.request_id.clone()),
                                "duplicate terminal event for {}",
                                event.request_id
                            );
                        } else {
                            ensure!(
                                !finished.contains(&event.request_id),
                                "post-terminal event for {}",
                                event.request_id
                            );
                        }
                        events
                            .entry(event.request_id.clone())
                            .or_default()
                            .push(event);
                        ensure!(
                            events.len() <= requests.len(),
                            "Dynamo emitted stats for more requests than the batch: {:?}",
                            events.keys().collect::<Vec<_>>()
                        );
                    }
                    DiagnosticMessage::Eof => bail!("Dynamo diagnostic stream ended during batch"),
                    DiagnosticMessage::Error(error) => {
                        bail!("Dynamo diagnostic stream failed: {error}")
                    }
                }
            }
            Result::<()>::Ok(())
        };
        timeout(REQUEST_TIMEOUT, collect)
            .await
            .context("timed out collecting Dynamo stats events")??;

        while let Ok(message) = self.receiver.try_recv() {
            match message {
                DiagnosticMessage::Ping => {}
                DiagnosticMessage::Stats(event) => {
                    bail!("event arrived after batch terminal: {event:?}")
                }
                DiagnosticMessage::Eof => {
                    bail!("Dynamo diagnostic stream ended after batch terminal")
                }
                DiagnosticMessage::Error(error) => {
                    bail!("Dynamo diagnostic stream failed after batch terminal: {error}")
                }
            }
        }

        ensure!(
            events.len() == requests.len(),
            "expected stats for {} requests, got {}",
            requests.len(),
            events.len()
        );
        for request in requests {
            let request_events = events
                .get(&request.id)
                .with_context(|| format!("missing stats for {}", request.id))?;
            validate_request_events(
                &request.id,
                ACTUAL_INPUT_TOKENS,
                request_events,
                success_batch,
            )?;
        }
        let event_count = events.values().map(Vec::len).sum::<usize>();
        if success_batch {
            ensure!(
                event_count == requests.len() * 4,
                "expected {} stats events, got {event_count}",
                requests.len() * 4
            );
        }
        Ok(CollectedBatch { event_count })
    }
}

impl Drop for DiagnosticStream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn read_diagnostic_stream(
    response: reqwest::Response,
    mut raw_file: File,
    sender: mpsc::UnboundedSender<DiagnosticMessage>,
) {
    let result = async {
        let mut bytes = response.bytes_stream();
        let mut buffer = Vec::new();
        while let Some(chunk) = bytes.try_next().await? {
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let mut line = buffer.drain(..=newline).collect::<Vec<_>>();
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if line.is_empty() {
                    continue;
                }
                raw_file.write_all(&line).await?;
                raw_file.write_all(b"\n").await?;
                raw_file.flush().await?;
                let value: Value = serde_json::from_slice(&line)?;
                match value.get("type").and_then(Value::as_str) {
                    Some("ping") => {
                        let _ = sender.send(DiagnosticMessage::Ping);
                    }
                    Some("stats") => {
                        ensure!(
                            value.get("correlation_id").is_none(),
                            "Dynamo stats event retained correlation_id: {value}"
                        );
                        let event = serde_json::from_value::<StatsEvent>(value)?;
                        let _ = sender.send(DiagnosticMessage::Stats(event));
                    }
                    event_type => bail!("unexpected Dynamo stats event type: {event_type:?}"),
                }
            }
        }
        ensure!(
            buffer.is_empty(),
            "Dynamo diagnostic stream ended with an unterminated NDJSON event"
        );
        Result::<()>::Ok(())
    }
    .await;
    match result {
        Ok(()) => {
            let _ = sender.send(DiagnosticMessage::Eof);
        }
        Err(error) => {
            let _ = sender.send(DiagnosticMessage::Error(format!("{error:#}")));
        }
    }
}

fn validate_request_events(
    request_id: &str,
    input_tokens: u64,
    events: &[StatsEvent],
    success_batch: bool,
) -> Result<()> {
    ensure!(!events.is_empty(), "no events for {request_id}");
    ensure!(
        events.iter().all(|event| event.v == 1
            && event.event_type == "stats"
            && event.request_id == request_id
            && event.model == MODEL),
        "invalid identity or schema for {request_id}: {events:?}"
    );
    ensure!(
        events.first().and_then(|event| event.tokens_processed) == Some(input_tokens),
        "first event for {request_id} did not carry tokens_processed={input_tokens}: {events:?}"
    );
    ensure!(
        events.last().is_some_and(|event| event.finished),
        "last event for {request_id} was not terminal: {events:?}"
    );
    ensure!(
        events.iter().filter(|event| event.finished).count() == 1,
        "request {request_id} did not have exactly one terminal: {events:?}"
    );
    let generated = events
        .iter()
        .filter_map(|event| event.tokens_generated)
        .collect::<Vec<_>>();
    ensure!(
        generated.windows(2).all(|pair| pair[0] <= pair[1]),
        "request {request_id} counters regressed: {events:?}"
    );
    let terminal = events.last().expect("nonempty events");
    ensure!(
        terminal.tokens_processed == Some(input_tokens),
        "terminal event for {request_id} lost tokens_processed: {events:?}"
    );
    if success_batch {
        ensure!(
            events.len() == 4,
            "request {request_id} expected four events: {events:?}"
        );
        let expected_generated = [Some(2), Some(5), Some(9), Some(9)];
        ensure!(
            events
                .iter()
                .map(|event| event.tokens_generated)
                .eq(expected_generated),
            "request {request_id} had wrong cumulative generated counters: {events:?}"
        );
        ensure!(
            events[1].tokens_processed.is_none() && events[2].tokens_processed.is_none(),
            "request {request_id} repeated nonterminal input counters: {events:?}"
        );
    } else {
        ensure!(
            (2..=4).contains(&events.len()),
            "cancelled request {request_id} emitted an unexpected event count: {events:?}"
        );
        let last_nonterminal_generated = events[..events.len() - 1]
            .iter()
            .rev()
            .find_map(|event| event.tokens_generated);
        ensure!(
            terminal.tokens_generated == last_nonterminal_generated,
            "cancelled request {request_id} terminal lost generated total: {events:?}"
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct RequestSpec {
    id: String,
    estimated_input_tokens: u64,
    cancel_after_first_chunk: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CollectedBatch {
    event_count: usize,
}

#[derive(Clone, Debug)]
struct PrometheusSnapshot {
    text: String,
}

impl PrometheusSnapshot {
    fn parse(text: String) -> Result<Self> {
        for line in metric_lines(&text) {
            parse_metric_sample(line)?;
        }
        Ok(Self { text })
    }

    fn value(&self, descriptor: &str) -> Result<f64> {
        let mut found = None;
        for line in metric_lines(&self.text) {
            let (candidate, value) = parse_metric_sample(line)?;
            if candidate == descriptor {
                ensure!(
                    found.replace(value).is_none(),
                    "duplicate Prometheus sample: {descriptor}"
                );
            }
        }
        Ok(found.unwrap_or_default())
    }

    fn family_sum(&self, name: &str) -> Result<f64> {
        let mut total = 0.0;
        for line in metric_lines(&self.text) {
            let (descriptor, value) = parse_metric_sample(line)?;
            if descriptor == name
                || descriptor
                    .strip_prefix(name)
                    .is_some_and(|labels| labels.starts_with('{'))
            {
                total += value;
            }
        }
        Ok(total)
    }
}

fn metric_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

fn parse_metric_sample(line: &str) -> Result<(&str, f64)> {
    let (descriptor, value) = line
        .split_once(char::is_whitespace)
        .with_context(|| format!("malformed Prometheus line: {line}"))?;
    if descriptor.contains('{') {
        ensure!(
            descriptor.ends_with('}'),
            "malformed Prometheus descriptor: {descriptor}"
        );
    }
    let value = value
        .split_whitespace()
        .next()
        .context("missing Prometheus value")?
        .parse::<f64>()
        .with_context(|| format!("invalid Prometheus value in {line}"))?;
    Ok((descriptor, value))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct CandidateReport {
    inference_server_id: String,
    stats_observed_at_unix_ms: u64,
    last_mean_input_tps: f64,
    output_tps: f64,
    max_output_tps: f64,
    num_running_queries: u64,
    input_processing_queries: u64,
    output_generation_queries: u64,
    stats_sources: Vec<String>,
    stats_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BatchReport {
    name: String,
    duration_ms: u128,
    request_count: usize,
    event_count: usize,
    stats_event_total: u64,
    invalid_event_total: u64,
    candidate: CandidateReport,
}

#[derive(Debug, Serialize)]
struct TestReport {
    total_duration_ms: u128,
    phases: Vec<PhaseReport>,
    batches: Vec<BatchReport>,
    reconnect_count: u64,
}

#[derive(Debug, Serialize)]
struct PhaseReport {
    name: String,
    duration_ms: u128,
}

struct ReportBuilder {
    started_at: Instant,
    phases: Vec<PhaseReport>,
    batches: Vec<BatchReport>,
}

impl ReportBuilder {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            phases: Vec::new(),
            batches: Vec::new(),
        }
    }

    async fn phase<T, F>(&mut self, name: &str, future: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let started = Instant::now();
        println!("PHASE {name}");
        let value = future.await?;
        self.phases.push(PhaseReport {
            name: name.to_string(),
            duration_ms: started.elapsed().as_millis(),
        });
        Ok(value)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .compact()
        .try_init()
        .ok();
    let args = Args::parse()?;
    tokio::fs::create_dir_all(&args.artifact_dir).await?;
    let failure_path = args.artifact_dir.join("failure.txt");
    let result = timeout(WHOLE_TEST_TIMEOUT, run(&args)).await;
    match result {
        Ok(Ok(report)) => {
            let result_path = args.artifact_dir.join("result.json");
            tokio::fs::write(&result_path, serde_json::to_vec_pretty(&report)?).await?;
            println!("PASS {}", result_path.display());
            Ok(())
        }
        Ok(Err(error)) => {
            let message = format!("{error:#}\n");
            tokio::fs::write(&failure_path, &message).await?;
            Err(error)
        }
        Err(_) => {
            let error = anyhow::anyhow!("whole E2E test exceeded {WHOLE_TEST_TIMEOUT:?}");
            tokio::fs::write(&failure_path, format!("{error:#}\n")).await?;
            Err(error)
        }
    }
}

async fn run(args: &Args) -> Result<TestReport> {
    let mut report = ReportBuilder::new();
    let raw_path = args.artifact_dir.join("dynamo-stats.ndjson");
    let mut reservation = PortReservation::ephemeral()?;
    let dynamo_port = reservation.port();
    let stargate = report
        .phase(
            "start-stargate",
            StargateFixture::start(&args.stargate_probe_bin, &args.artifact_dir),
        )
        .await?;
    let mut consumer = report
        .phase("consumer-retries-before-dynamo", async {
            let mut consumer = StatsConsumerProcess::spawn(
                &args.stats_consumer_bin,
                &args.artifact_dir,
                1,
                dynamo_port,
                stargate.grpc_addr,
            )
            .await?;
            let snapshot = wait_consumer_metrics(
                &mut consumer,
                READY_TIMEOUT,
                |metrics| {
                    let reconnects = metric_counter(metrics, METRIC_RECONNECT_CONNECT_ERROR)?;
                    let connected = metrics.value(METRIC_CONNECTED_REQUIRED)?;
                    Ok((reconnects >= 2 && connected == 0.0).then_some(metrics.clone()))
                },
                "two stats consumer connect_error retries while disconnected",
            )
            .await?;
            ensure!(
                snapshot.value(METRIC_CONNECTED_REQUIRED)? == 0.0,
                "stats consumer unexpectedly connected before Dynamo started"
            );
            Ok(consumer)
        })
        .await?;

    let ping_before = metric_counter(&consumer.scrape().await?, METRIC_EVENT_PING)?;
    let dynamo = report
        .phase("late-dynamo-start", DynamoFixture::start(reservation))
        .await?;
    let mut diagnostic = DiagnosticStream::connect(dynamo.port, &raw_path).await?;
    wait_for_stats_stream(&mut consumer, ping_before).await?;
    wait_for_registration_and_route(&mut consumer, &stargate).await?;

    let mut previous_timestamp = 0;
    let phase_one = vec![RequestSpec {
        id: "e2e-happy-1".to_string(),
        estimated_input_tokens: 120,
        cancel_after_first_chunk: false,
    }];
    let batch = report
        .phase(
            "happy-path",
            run_batch(
                "happy-path",
                &mut consumer,
                &stargate,
                &mut diagnostic,
                &phase_one,
                true,
                previous_timestamp,
            ),
        )
        .await?;
    previous_timestamp = batch.candidate.stats_observed_at_unix_ms;
    report.batches.push(batch);

    let concurrency = (0..16)
        .map(|index| RequestSpec {
            id: format!("e2e-concurrent-{index:02}"),
            estimated_input_tokens: 20 + index,
            cancel_after_first_chunk: false,
        })
        .collect::<Vec<_>>();
    let batch = report
        .phase(
            "concurrency-and-identity",
            run_batch(
                "concurrency",
                &mut consumer,
                &stargate,
                &mut diagnostic,
                &concurrency,
                true,
                previous_timestamp,
            ),
        )
        .await?;
    previous_timestamp = batch.candidate.stats_observed_at_unix_ms;
    report.batches.push(batch);

    let cancellation = vec![RequestSpec {
        id: "e2e-cancelled".to_string(),
        estimated_input_tokens: 37,
        cancel_after_first_chunk: true,
    }];
    let batch = report
        .phase(
            "client-cancellation",
            run_batch(
                "cancellation",
                &mut consumer,
                &stargate,
                &mut diagnostic,
                &cancellation,
                false,
                previous_timestamp,
            ),
        )
        .await?;
    previous_timestamp = batch.candidate.stats_observed_at_unix_ms;
    report.batches.push(batch);

    let mut dynamo = dynamo;
    for cycle in 1..=3 {
        drop(diagnostic);
        let metrics_before_restart = consumer.scrape().await?;
        let reconnect_before =
            metric_counter(&metrics_before_restart, METRIC_RECONNECT_CONNECT_ERROR)?;
        let ping_before = metric_counter(&metrics_before_restart, METRIC_EVENT_PING)?;
        reservation = report
            .phase(&format!("dynamo-stop-{cycle}"), dynamo.stop_and_reserve())
            .await?;
        wait_consumer_metrics(
            &mut consumer,
            READY_TIMEOUT,
            |metrics| {
                let connected = metrics.value(METRIC_CONNECTED_REQUIRED)?;
                let reconnects = metric_counter(metrics, METRIC_RECONNECT_CONNECT_ERROR)?;
                Ok((connected == 0.0 && reconnects > reconnect_before).then_some(metrics.clone()))
            },
            "stats consumer disconnect and connect_error retry",
        )
        .await?;
        dynamo = report
            .phase(
                &format!("dynamo-restart-{cycle}"),
                DynamoFixture::start(reservation),
            )
            .await?;
        wait_for_stats_stream(&mut consumer, ping_before).await?;
        diagnostic = DiagnosticStream::connect(dynamo.port, &raw_path).await?;
        let requests = (0..2)
            .map(|index| RequestSpec {
                id: format!("e2e-restart-{cycle}-{index}"),
                estimated_input_tokens: 40 + cycle * 10 + index,
                cancel_after_first_chunk: false,
            })
            .collect::<Vec<_>>();
        let batch = run_batch(
            &format!("restart-{cycle}"),
            &mut consumer,
            &stargate,
            &mut diagnostic,
            &requests,
            true,
            previous_timestamp,
        )
        .await?;
        previous_timestamp = batch.candidate.stats_observed_at_unix_ms;
        report.batches.push(batch);
    }

    let reconnect_count_before_consumer_restart =
        metric_counter_sum(&consumer.scrape().await?, METRIC_RECONNECTS)?;
    report
        .phase("consumer-stop-and-route-removal", async {
            consumer.stop().await?;
            wait_for_route(&stargate, false).await
        })
        .await?;
    consumer = report
        .phase(
            "consumer-restart",
            StatsConsumerProcess::spawn(
                &args.stats_consumer_bin,
                &args.artifact_dir,
                2,
                dynamo.port,
                stargate.grpc_addr,
            ),
        )
        .await?;
    wait_for_stats_stream(&mut consumer, 0).await?;
    wait_for_registration_and_route(&mut consumer, &stargate).await?;
    let after_consumer_restart = vec![
        RequestSpec {
            id: "e2e-consumer-restart-0".to_string(),
            estimated_input_tokens: 81,
            cancel_after_first_chunk: false,
        },
        RequestSpec {
            id: "e2e-consumer-restart-1".to_string(),
            estimated_input_tokens: 82,
            cancel_after_first_chunk: false,
        },
    ];
    let batch = run_batch(
        "consumer-restart",
        &mut consumer,
        &stargate,
        &mut diagnostic,
        &after_consumer_restart,
        true,
        previous_timestamp,
    )
    .await?;
    report.batches.push(batch);

    let final_metrics = consumer.scrape().await?;
    tokio::fs::write(
        args.artifact_dir.join("final-stats-consumer.metrics"),
        &final_metrics.text,
    )
    .await?;
    let reconnect_count = reconnect_count_before_consumer_restart
        + metric_counter_sum(&final_metrics, METRIC_RECONNECTS)?;

    report
        .phase("cleanup", async {
            drop(diagnostic);
            consumer.stop().await?;
            dynamo.stop().await?;
            stargate.shutdown().await
        })
        .await?;

    Ok(TestReport {
        total_duration_ms: report.started_at.elapsed().as_millis(),
        phases: report.phases,
        batches: report.batches,
        reconnect_count,
    })
}

async fn run_batch(
    name: &str,
    consumer: &mut StatsConsumerProcess,
    stargate: &StargateFixture,
    diagnostic: &mut DiagnosticStream,
    requests: &[RequestSpec],
    success_batch: bool,
    previous_timestamp: u64,
) -> Result<BatchReport> {
    let started = Instant::now();
    let baseline = wait_consumer_live_zero(consumer).await?;
    let event_baseline = metric_counter(&baseline, METRIC_EVENT_STATS)?;
    let invalid_baseline = metric_counter_sum(&baseline, METRIC_INVALID)?;
    let model_stats_baseline = (
        baseline.value(METRIC_INPUT_TPS)?,
        baseline.value(METRIC_OUTPUT_TPS)?,
        baseline.value(METRIC_MAX_OUTPUT_TPS)?,
    );

    let client = reqwest::Client::new();
    try_join_all(
        requests
            .iter()
            .cloned()
            .map(|request| send_request(client.clone(), stargate.http_addr, request)),
    )
    .await?;
    let collected = diagnostic.collect_batch(requests, success_batch).await?;
    let expected_delta = u64::try_from(collected.event_count)?;

    let parsed = wait_consumer_metrics(
        consumer,
        READY_TIMEOUT,
        |metrics| {
            let current = metric_counter(metrics, METRIC_EVENT_STATS)?;
            let delta = current
                .checked_sub(event_baseline)
                .context("stats consumer stats counter regressed")?;
            // Both subscribers consume the same broadcast channel; overshoot cannot settle.
            ensure!(
                delta <= expected_delta,
                "stats consumer parsed {delta} events but Dynamo emitted {expected_delta}"
            );
            Ok((delta == expected_delta).then_some(metrics.clone()))
        },
        "stats consumer exact stats event count",
    )
    .await?;
    ensure!(
        metric_counter_sum(&parsed, METRIC_INVALID)? == invalid_baseline,
        "stats consumer invalid-event counter advanced"
    );
    let parsed = wait_consumer_live_zero(consumer).await?;
    let (candidate, stable_metrics) =
        wait_for_model_stats(consumer, stargate, previous_timestamp, model_stats_baseline).await?;
    ensure!(
        metric_counter_sum(&stable_metrics, METRIC_INVALID)? == invalid_baseline,
        "stats consumer invalid-event counter advanced while publishing model stats"
    );
    Ok(BatchReport {
        name: name.to_string(),
        duration_ms: started.elapsed().as_millis(),
        request_count: requests.len(),
        event_count: collected.event_count,
        stats_event_total: metric_counter(&parsed, METRIC_EVENT_STATS)?,
        invalid_event_total: metric_counter_sum(&parsed, METRIC_INVALID)?,
        candidate,
    })
}

async fn send_request(
    client: reqwest::Client,
    stargate_http_addr: SocketAddr,
    request: RequestSpec,
) -> Result<()> {
    let response = client
        .post(format!("http://{stargate_http_addr}/v1/chat/completions"))
        .header("x-request-id", &request.id)
        .header("x-model", MODEL)
        .header("x-input-tokens", request.estimated_input_tokens)
        .header("x-priority", 0)
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "e2e"}],
            "stream": true
        }))
        .send()
        .await
        .with_context(|| format!("request {} failed", request.id))?;
    let status = response.status();
    ensure!(
        status.is_success(),
        "request {} returned {}: {}",
        request.id,
        status,
        response.text().await.unwrap_or_default()
    );
    if request.cancel_after_first_chunk {
        let mut stream = response.bytes_stream();
        let first = timeout(REQUEST_TIMEOUT, stream.next())
            .await
            .with_context(|| format!("request {} produced no first chunk", request.id))?
            .with_context(|| format!("request {} stream ended before first chunk", request.id))??;
        ensure!(
            !first.is_empty(),
            "request {} first chunk was empty",
            request.id
        );
        drop(stream);
    } else {
        timeout(REQUEST_TIMEOUT, response.bytes())
            .await
            .with_context(|| format!("request {} body timed out", request.id))??;
    }
    Ok(())
}

async fn wait_for_stats_stream(
    consumer: &mut StatsConsumerProcess,
    ping_before: u64,
) -> Result<()> {
    wait_consumer_metrics(
        consumer,
        READY_TIMEOUT,
        |metrics| {
            let connected = metrics.value(METRIC_CONNECTED_REQUIRED)?;
            let ping = metric_counter(metrics, METRIC_EVENT_PING)?;
            Ok((connected == 1.0 && ping > ping_before).then_some(metrics.clone()))
        },
        "stats consumer stats stream connection and immediate ping",
    )
    .await?;
    Ok(())
}

async fn wait_for_registration_and_route(
    consumer: &mut StatsConsumerProcess,
    stargate: &StargateFixture,
) -> Result<()> {
    wait_consumer_metrics(
        consumer,
        READY_TIMEOUT,
        |metrics| {
            Ok((metrics.family_sum(METRIC_REGISTRATION_CONNECTED)? > 0.0)
                .then_some(metrics.clone()))
        },
        "stats consumer registration stream",
    )
    .await?;
    wait_for_route(stargate, true).await
}

async fn wait_for_route(stargate: &StargateFixture, present: bool) -> Result<()> {
    let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
    loop {
        let candidates = stargate.candidates().await?;
        let matched = if present {
            candidates.len() == 1 && candidates[0].inference_server_id == STATS_CONSUMER_ID
        } else {
            candidates.is_empty()
        };
        if matched {
            return Ok(());
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for Stargate route {}",
            if present { "registration" } else { "removal" }
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn wait_consumer_live_zero(
    consumer: &mut StatsConsumerProcess,
) -> Result<PrometheusSnapshot> {
    wait_consumer_metrics(
        consumer,
        READY_TIMEOUT,
        |metrics| {
            let live = metrics.value(METRIC_LIVE_ENGINE_STATS)?;
            Ok((live == 0.0).then_some(metrics.clone()))
        },
        "stats consumer live request count to reach zero",
    )
    .await
}

async fn wait_for_model_stats(
    consumer: &mut StatsConsumerProcess,
    stargate: &StargateFixture,
    previous_timestamp: u64,
    model_stats_baseline: (f64, f64, f64),
) -> Result<(CandidateReport, PrometheusSnapshot)> {
    let deadline = tokio::time::Instant::now() + READY_TIMEOUT;
    loop {
        let candidates_before = stargate.candidates().await?;
        let metrics = consumer.scrape().await?;
        let candidates = stargate.candidates().await?;
        if candidates_before != candidates {
            ensure!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for a stable Stargate stats snapshot"
            );
            tokio::time::sleep(POLL_INTERVAL).await;
            continue;
        }
        let source = metrics.value(METRIC_STATS_SOURCE_ENGINE)?;
        let capability = metrics.value(METRIC_STATS_CAPABILITY_ENGINE)?;
        let input_tps = metrics.value(METRIC_INPUT_TPS)?;
        let output_tps = metrics.value(METRIC_OUTPUT_TPS)?;
        let max_output_tps = metrics.value(METRIC_MAX_OUTPUT_TPS)?;
        let model_stats_changed = [
            (input_tps, model_stats_baseline.0),
            (output_tps, model_stats_baseline.1),
            (max_output_tps, model_stats_baseline.2),
        ]
        .into_iter()
        .any(|(current, baseline)| !approximately_equal(current, baseline));
        if source == 1.0
            && capability == 1.0
            && positive_finite(input_tps)
            && positive_finite(output_tps)
            && positive_finite(max_output_tps)
            && model_stats_changed
            && candidates.len() == 1
        {
            let candidate = &candidates[0];
            if candidate.inference_server_id == STATS_CONSUMER_ID
                && candidate.stats_observed_at_unix_ms > previous_timestamp
                && candidate
                    .stats_sources
                    .iter()
                    .any(|source| source == "engine_stats_stream")
                && candidate
                    .stats_capabilities
                    .iter()
                    .any(|capability| capability == "model.throughput.engine_stream")
                && candidate.num_running_queries == 0
                && candidate.input_processing_queries == 0
                && candidate.output_generation_queries == 0
                && approximately_equal(candidate.last_mean_input_tps, input_tps)
                && approximately_equal(candidate.output_tps, output_tps)
                && approximately_equal(candidate.max_output_tps, max_output_tps)
            {
                return Ok((candidate.clone(), metrics));
            }
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for stats consumer model stats to reach Stargate"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let tolerance = 1e-6f64.max(left.abs().max(right.abs()) * 1e-6);
    (left - right).abs() <= tolerance
}

async fn wait_consumer_metrics<F>(
    consumer: &mut StatsConsumerProcess,
    deadline: Duration,
    mut predicate: F,
    label: &str,
) -> Result<PrometheusSnapshot>
where
    F: FnMut(&PrometheusSnapshot) -> Result<Option<PrometheusSnapshot>>,
{
    let deadline = tokio::time::Instant::now() + deadline;
    loop {
        let metrics = consumer.scrape().await?;
        if let Some(snapshot) = predicate(&metrics)? {
            return Ok(snapshot);
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {label}"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn wait_until<T, F, Fut>(label: &str, deadline: Duration, mut check: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Option<T>>>,
{
    let deadline = tokio::time::Instant::now() + deadline;
    let mut interval = tokio::time::interval(POLL_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        if let Some(value) = check().await? {
            return Ok(value);
        }
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for {label}");
        }
        interval.tick().await;
    }
}

async fn wait_http_ok(url: &str, deadline: Duration) -> Result<()> {
    let client = reqwest::Client::new();
    wait_until(url, deadline, || async {
        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => Ok(Some(())),
            Ok(_) | Err(_) => Ok(None),
        }
    })
    .await
}

fn metric_counter(snapshot: &PrometheusSnapshot, descriptor: &str) -> Result<u64> {
    metric_u64(snapshot.value(descriptor)?, descriptor)
}

fn metric_counter_sum(snapshot: &PrometheusSnapshot, name: &str) -> Result<u64> {
    metric_u64(snapshot.family_sum(name)?, name)
}

fn metric_u64(value: f64, name: &str) -> Result<u64> {
    ensure!(
        value.is_finite() && value >= 0.0 && value.fract() == 0.0,
        "{name} is not a nonnegative integer: {value}"
    );
    u64::try_from(value as u128).with_context(|| format!("{name} is too large: {value}"))
}

async fn signal_interrupt(child: &mut Child, label: &str) -> Result<()> {
    let pid = child
        .id()
        .with_context(|| format!("{label} has no process ID"))?;
    let status = Command::new("kill")
        .arg("-INT")
        .arg(pid.to_string())
        .status()
        .await
        .with_context(|| format!("failed to signal {label}"))?;
    ensure!(
        status.success(),
        "kill -INT for {label} failed with {status}"
    );
    Ok(())
}

async fn remove_file_if_present(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn unix_time_ms() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis();
    u64::try_from(millis).context("Unix timestamp does not fit in u64")
}
