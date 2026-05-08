// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use kvbm_hub::config::HubConfig;
use velo::backend::tcp::TcpTransportBuilder;

/// Selection policy used by the hub's prefill dispatcher to pick a
/// prefill peer for each dequeued request.
#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
enum RouterMode {
    /// Cycle through registered prefill peers in registration order.
    /// Stateless; ignores per-peer load.
    RoundRobin,
    /// Pick the prefill peer with the fewest in-flight requests.
    /// Tracks load via per-peer atomic counters with RAII permits.
    LeastLoaded,
}

#[derive(Parser)]
#[command(name = "kvbm-hub", about = "KVBM coordination hub server")]
struct Cli {
    /// TOML or JSON config file (env: KVBM_HUB_CONFIG)
    #[arg(long, env = "KVBM_HUB_CONFIG")]
    config: Option<PathBuf>,

    /// Address to bind (overrides KVBM_HUB_BIND_ADDR)
    #[arg(long)]
    bind_addr: Option<IpAddr>,

    /// Discovery HTTP port (overrides KVBM_HUB_DISCOVERY_PORT)
    #[arg(long)]
    discovery_port: Option<u16>,

    /// Control-plane HTTP port (overrides KVBM_HUB_CONTROL_PORT)
    #[arg(long)]
    control_port: Option<u16>,

    /// Velo transport port. When set, the hub binds a TCP transport and
    /// participates in velo active messaging. When omitted the hub is
    /// discovery-only.
    #[arg(long)]
    velo_port: Option<u16>,

    /// Liveness TTL (seconds) for the in-memory registry
    /// (overrides KVBM_HUB_REGISTRATION_TTL_SECS).
    #[arg(long)]
    registration_ttl_secs: Option<u64>,

    /// Reaper tick interval (seconds) for the in-memory registry
    /// (overrides KVBM_HUB_PRUNE_INTERVAL_SECS).
    #[arg(long)]
    prune_interval_secs: Option<u64>,

    /// Hub-driven heartbeat probe interval (seconds).
    /// (overrides KVBM_HUB_HEARTBEAT_INTERVAL_SECS).
    #[arg(long)]
    heartbeat_interval_secs: Option<u64>,

    /// Consecutive probe failures before unregister.
    /// (overrides KVBM_HUB_HEARTBEAT_MAX_FAILURES).
    #[arg(long)]
    heartbeat_max_failures: Option<u32>,

    /// Enable hub-driven prefill dispatcher: when set, the hub spawns a
    /// background worker that drains the CD prefill queue and POSTs each
    /// dequeued request to this URL's `/v1/completions` endpoint
    /// (typically the prefill vLLM frontend). Single-prefill only;
    /// multi-prefill routing is a future enhancement.
    #[arg(long)]
    prefill_vllm_url: Option<String>,

    /// Model name passed in dispatched POST bodies. Must match the
    /// `--model` flag the prefill vLLM was started with. Required when
    /// `--prefill-vllm-url` is set.
    #[arg(long)]
    prefill_vllm_model: Option<String>,

    /// Selection policy for the prefill dispatcher. Defaults to
    /// `round-robin`. Ignored when the dispatch pipeline isn't enabled
    /// (i.e. `--prefill-vllm-model` not set).
    #[arg(long, value_enum, default_value_t = RouterMode::RoundRobin)]
    prefill_router_mode: RouterMode,
}

fn build_config(cli: &Cli) -> anyhow::Result<HubConfig> {
    let mut f = HubConfig::figment(cli.config.as_deref());
    if let Some(addr) = cli.bind_addr {
        f = f.merge(("bind_addr", addr.to_string()));
    }
    if let Some(port) = cli.discovery_port {
        f = f.merge(("discovery_port", port));
    }
    if let Some(port) = cli.control_port {
        f = f.merge(("control_port", port));
    }
    if let Some(port) = cli.velo_port {
        f = f.merge(("velo_port", port));
    }
    if let Some(secs) = cli.registration_ttl_secs {
        f = f.merge(("registration_ttl_secs", secs));
    }
    if let Some(secs) = cli.prune_interval_secs {
        f = f.merge(("prune_interval_secs", secs));
    }
    if let Some(secs) = cli.heartbeat_interval_secs {
        f = f.merge(("heartbeat_interval_secs", secs));
    }
    if let Some(n) = cli.heartbeat_max_failures {
        f = f.merge(("heartbeat_max_failures", n));
    }
    Ok(f.extract()?)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let config = build_config(&cli)?;

    tracing::info!(
        bind_addr = %config.bind_addr,
        discovery_port = config.discovery_port,
        control_port = config.control_port,
        velo_port = ?config.velo_port,
        registration_ttl_secs = config.registration_ttl_secs,
        prune_interval_secs = config.prune_interval_secs,
        "starting kvbm-hub"
    );

    // Build the CD peer registry. Shared between the manager (writes via
    // on_register/on_unregister) and the prefill drainer (reads via the
    // PrefillPeerSource trait).
    let cd_registry = Arc::new(kvbm_hub::CdPeerRegistry::new());

    // Build the CD manager, optionally with the HTTP dispatch pipeline.
    let cd_manager = match (&cli.prefill_vllm_url, &cli.prefill_vllm_model) {
        (Some(url), Some(model)) => {
            // TODO(piece-9): drop the --prefill-vllm-url CLI flag entirely.
            // After piece-8 (connector advertises engine_url at registration),
            // the dispatcher gets per-call URLs from the selector — no need
            // for a hub-side default. `url` is still logged below for now.
            let dispatcher: Arc<dyn kvbm_hub::PrefillRequestDispatcher> =
                kvbm_hub::HttpVllmDispatcher::new(model.clone())?;
            let selector: Arc<dyn kvbm_hub::PrefillWorkerSelector> = match cli.prefill_router_mode {
                RouterMode::RoundRobin => Arc::new(kvbm_hub::RoundRobinSelector::new()),
                RouterMode::LeastLoaded => Arc::new(kvbm_hub::LeastLoadedSelector::new()),
            };
            tracing::info!(
                prefill_url = %url,
                prefill_model = %model,
                router_mode = ?cli.prefill_router_mode,
                "CD prefill dispatcher enabled (HTTP → engine frontend)"
            );
            kvbm_hub::ConditionalDisaggManager::new(Arc::clone(&cd_registry))
                .with_dispatch_pipeline(dispatcher, selector)
        }
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("--prefill-vllm-url and --prefill-vllm-model must be specified together");
        }
        (None, None) => {
            tracing::info!(
                "CD prefill dispatcher disabled (set --prefill-vllm-url + --prefill-vllm-model to enable)"
            );
            kvbm_hub::ConditionalDisaggManager::new(Arc::clone(&cd_registry))
        }
    };

    let mut builder = kvbm_hub::create_server_builder()
        .bind_addr(config.bind_addr)
        .discovery_port(config.discovery_port)
        .control_port(config.control_port)
        .registration_ttl(Duration::from_secs(config.registration_ttl_secs))
        .prune_interval(Duration::from_secs(config.prune_interval_secs))
        .heartbeat_interval(Duration::from_secs(config.heartbeat_interval_secs))
        .heartbeat_max_failures(config.heartbeat_max_failures)
        .add_feature_manager(Arc::new(cd_manager))
        .add_feature_manager(Arc::new(kvbm_hub::ConnectorControlManager::new()));

    if let Some(velo_port) = config.velo_port {
        let bind = SocketAddr::new(config.bind_addr, velo_port);
        let listener = std::net::TcpListener::bind(bind)
            .map_err(|e| anyhow::anyhow!("binding velo port {bind}: {e}"))?;
        listener.set_nonblocking(true)?;
        let transport = TcpTransportBuilder::new()
            .from_listener(listener)
            .map_err(|e| anyhow::anyhow!("tcp transport from_listener: {e}"))?
            .build()
            .map_err(|e| anyhow::anyhow!("tcp transport build: {e}"))?;
        builder = builder.add_transport(Arc::new(transport) as Arc<dyn velo::Transport>);
    }

    let server = builder.serve().await?;

    tracing::info!(
        discovery = %server.discovery_addr(),
        control = %server.control_addr(),
        "kvbm-hub listening"
    );

    tokio::signal::ctrl_c().await?;
    tracing::info!("shutting down");
    server.shutdown().await?;
    Ok(())
}
