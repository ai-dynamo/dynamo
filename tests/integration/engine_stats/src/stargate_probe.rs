use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, bail, ensure};
use serde::Serialize;
use stargate::auth::OpenAuthenticator;
use stargate::discovery::SelfOnlyDiscovery;
use stargate::proxy::{ProxyTransportConfig, QuicTunnelConfig};
use stargate::routing::RoutingTargetKey;
use stargate::runtime::{BoundStargateListeners, StargateRuntime, StargateRuntimeConfig};
use stargate_tls::ServerTlsIdentity;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::time::MissedTickBehavior;

const MODEL: &str = "stats-e2e";
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(25);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

struct Args {
    ready_file: PathBuf,
    snapshot_file: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args_os().skip(1);
        let mut ready_file = None;
        let mut snapshot_file = None;
        while let Some(argument) = values.next() {
            let argument = argument.to_string_lossy();
            let mut value = || {
                values
                    .next()
                    .with_context(|| format!("missing value for {argument}"))
            };
            match argument.as_ref() {
                "--ready-file" => ready_file = Some(PathBuf::from(value()?)),
                "--snapshot-file" => snapshot_file = Some(PathBuf::from(value()?)),
                other => bail!("unknown argument: {other}"),
            }
        }
        Ok(Self {
            ready_file: ready_file.context("--ready-file is required")?,
            snapshot_file: snapshot_file.context("--snapshot-file is required")?,
        })
    }
}

#[derive(Serialize)]
struct Ready {
    grpc_addr: String,
    http_addr: String,
}

#[derive(Serialize)]
struct Snapshot {
    written_at_unix_ms: u64,
    candidates: Vec<Candidate>,
}

#[derive(Serialize)]
struct Candidate {
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
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    remove_stale(&args.ready_file).await?;
    remove_stale(&args.snapshot_file).await?;

    let loopback = |port| SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let mut config = StargateRuntimeConfig {
        stargate_id: "stargate-stats-e2e".to_string(),
        grpc_listen_addr: loopback(0),
        model_discovery_listen_addr: loopback(0),
        http_listen_addr: loopback(0),
        metrics_listen_addr: None,
        advertise_addr: loopback(0),
        stargate_discovery_dns_name: "localhost".to_string(),
        remote_watch_stargate_urls: Vec::new(),
        grpc_pylon_dial_addr: None,
        dns_poll_interval: Duration::from_secs(60),
        watch_heartbeat_interval: Duration::from_secs(1),
        registration_update_idle_timeout:
            stargate::registration::DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT,
        registration_update_max_idle_timeout:
            stargate::registration::DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT,
        proxy_transport: ProxyTransportConfig {
            quic: QuicTunnelConfig {
                connect_timeout: Duration::from_secs(5),
                request_timeout: REQUEST_TIMEOUT,
                direct_quic_connections: 1,
                tls_cert_pem: None,
                server_tls_identity: ServerTlsIdentity::SelfSigned,
                quic_insecure: true,
                tunnel_protocol: Default::default(),
            },
            retry: Default::default(),
        },
        lb_config_path: None,
        metrics_prefix: stargate::metrics::DEFAULT_PREFIX.to_string(),
        forwarding: None,
        authenticator: Arc::new(OpenAuthenticator),
    };
    let listeners = BoundStargateListeners::bind(&mut config)?;
    let grpc_addr = listeners.grpc_addr();
    let http_addr = listeners.http_addr();
    let discovery = Box::new(SelfOnlyDiscovery::new(
        grpc_addr,
        config.stargate_id.clone(),
        http_addr.port(),
    ));
    let handle = StargateRuntime::new(config, discovery, listeners, None)
        .start()
        .await?;
    let state = handle.state();

    write_json_atomic(
        &args.ready_file,
        &Ready {
            grpc_addr: grpc_addr.to_string(),
            http_addr: http_addr.to_string(),
        },
    )
    .await?;

    let target = RoutingTargetKey {
        routing_key: None,
        model_id: MODEL.to_string(),
    };
    let mut interval = tokio::time::interval(SNAPSHOT_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut shutdown_line = String::new();
    let mut stdin = BufReader::new(tokio::io::stdin());
    let stdin_shutdown = stdin.read_line(&mut shutdown_line);
    tokio::pin!(stdin_shutdown);
    loop {
        tokio::select! {
            result = &mut stdin_shutdown => {
                result.context("failed to read shutdown signal")?;
                break;
            }
            signal = tokio::signal::ctrl_c() => {
                signal.context("failed to listen for shutdown signal")?;
                break;
            }
            _ = interval.tick() => {
                let candidates = state
                    .candidates_for_target(&target)
                    .await
                    .into_iter()
                    .map(|candidate| {
                        let stats = candidate.stats;
                        Candidate {
                            inference_server_id: candidate.inference_server_id,
                            stats_observed_at_unix_ms: stats.stats_observed_at_unix_ms,
                            last_mean_input_tps: stats.last_mean_input_tps,
                            output_tps: stats.output_tps,
                            max_output_tps: stats.max_output_tps,
                            num_running_queries: stats.num_running_queries,
                            input_processing_queries: stats.input_processing_queries,
                            output_generation_queries: stats.output_generation_queries,
                            stats_sources: stats.stats_sources,
                            stats_capabilities: stats.stats_capabilities,
                        }
                    })
                    .collect();
                write_json_atomic(
                    &args.snapshot_file,
                    &Snapshot {
                        written_at_unix_ms: unix_time_ms()?,
                        candidates,
                    },
                )
                .await?;
            }
        }
    }

    handle.begin_shutdown();
    ensure!(
        handle.wait_for_shutdown(SHUTDOWN_TIMEOUT).await,
        "Stargate shutdown timed out"
    );
    Ok(())
}

async fn remove_stale(path: &Path) -> Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

async fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    tokio::fs::write(&temporary, serde_json::to_vec(value)?).await?;
    tokio::fs::rename(&temporary, path).await?;
    Ok(())
}

fn unix_time_ms() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis();
    u64::try_from(millis).context("Unix timestamp does not fit in u64")
}
