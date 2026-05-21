// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use kvbm_hub::config::HubConfig;
use kvbm_hub::{BlockLayoutMode, FeatureKey};
use velo::transports::tcp::TcpTransportBuilder;

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

    /// Maximum sequence length (tokens). Shared "primary" config: validated
    /// against every registrant and used to size the KV index. Required (may
    /// also come from a config file / env). Must be a non-zero multiple of
    /// `--block-size`.
    #[arg(long)]
    max_seq_len: Option<usize>,

    /// Block size (tokens per block). Shared "primary" config. Required.
    /// Power of two in `16..=512`.
    #[arg(long)]
    block_size: Option<usize>,

    /// Cross-leader block-layout mode: `operational` (default) or `universal`.
    /// Shared "primary" config; validated against every registrant.
    #[arg(long)]
    layout: Option<String>,

    /// Comma-separated feature set the hub serves — subset of
    /// `p2p,conditional_disagg,kv_indexer`. Omitted = all. Dependencies are
    /// auto-included (selecting `conditional_disagg` pulls in `p2p`).
    #[arg(long)]
    features: Option<String>,

    /// Advisory G2 (host) cache size in GiB, seeded into generated connector
    /// config. At least one of `--g2-memory` / `--g2-block` is required.
    #[arg(long)]
    g2_memory: Option<f64>,

    /// Advisory G2 (host) cache size in blocks, seeded into generated connector
    /// config. At least one of `--g2-memory` / `--g2-block` is required.
    #[arg(long)]
    g2_block: Option<usize>,

    /// Maximum sequence length (tokens) for the KV indexer. **Deprecated**:
    /// prefer `--max-seq-len`. When set and `--max-seq-len` is absent, seeds
    /// the primary value (back-compat).
    #[arg(long)]
    kv_index_max_seq_len: Option<usize>,

    /// Block size (tokens per block) for the KV indexer. **Deprecated**: prefer
    /// `--block-size`. When set and `--block-size` is absent, seeds the primary
    /// value (back-compat).
    #[arg(long)]
    kv_index_block_size: Option<usize>,

    /// ZMQ bind spec for the KV indexer ingest socket
    /// (default `tcp://0.0.0.0:0`, OS-assigned port).
    #[arg(long)]
    kv_index_zmq_bind: Option<String>,

    /// Host advertised to publishers in the KV indexer's `GET /config`
    /// (default `127.0.0.1`).
    #[arg(long)]
    kv_index_advertise_host: Option<String>,
}

/// All hub features a client can be granted via `--features` (in dependency
/// order). `ConnectorControl` is infrastructure, always attached, not listed.
const SELECTABLE_FEATURES: [FeatureKey; 3] = [
    FeatureKey::P2P,
    FeatureKey::ConditionalDisagg,
    FeatureKey::KvIndexer,
];

/// Parse `--layout` into a [`BlockLayoutMode`].
fn parse_layout(s: &str) -> anyhow::Result<BlockLayoutMode> {
    match s {
        "operational" => Ok(BlockLayoutMode::Operational),
        "universal" => Ok(BlockLayoutMode::Universal),
        other => anyhow::bail!("--layout must be `operational` or `universal`, got {other:?}"),
    }
}

/// Resolve the enabled feature set from `--features`, expanding dependencies.
/// `None` = all selectable features.
fn parse_features(spec: Option<&str>) -> anyhow::Result<HashSet<FeatureKey>> {
    let mut set: HashSet<FeatureKey> = match spec {
        None => SELECTABLE_FEATURES.into_iter().collect(),
        Some(csv) => {
            let mut s = HashSet::new();
            for raw in csv.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                let key = FeatureKey::from_label(raw)
                    .filter(|k| SELECTABLE_FEATURES.contains(k))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown/unsupported feature {raw:?} in --features; \
                             choose from {:?}",
                            SELECTABLE_FEATURES.map(|k| k.as_str())
                        )
                    })?;
                s.insert(key);
            }
            s
        }
    };
    // Dependency closure: CD requires P2P.
    if set.contains(&FeatureKey::ConditionalDisagg) {
        set.insert(FeatureKey::P2P);
    }
    Ok(set)
}

/// Resolved hub config plus the set of features the hub serves.
struct ResolvedConfig {
    config: HubConfig,
    enabled: HashSet<FeatureKey>,
}

fn build_config(cli: &Cli) -> anyhow::Result<ResolvedConfig> {
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
    let mut config: HubConfig = f.extract()?;

    // Primary config: CLI flags win over file/env when explicitly passed.
    if let Some(v) = cli.max_seq_len {
        config.primary.max_seq_len = Some(v);
    }
    if let Some(v) = cli.block_size {
        config.primary.block_size = Some(v);
    }
    if let Some(layout) = &cli.layout {
        config.primary.block_layout = parse_layout(layout)?;
    }
    if let Some(v) = cli.g2_memory {
        config.primary.g2_memory_gib = Some(v);
    }
    if let Some(v) = cli.g2_block {
        config.primary.g2_blocks = Some(v);
    }
    // Back-compat: deprecated --kv-index-* sizing seeds primary when the
    // canonical flags are absent.
    if config.primary.block_size.is_none() {
        config.primary.block_size = cli.kv_index_block_size;
    }
    if config.primary.max_seq_len.is_none() {
        config.primary.max_seq_len = cli.kv_index_max_seq_len;
    }

    // Validate the required primary subset.
    let block_size = config
        .primary
        .block_size
        .ok_or_else(|| anyhow::anyhow!("--block-size is required"))?;
    if !block_size.is_power_of_two() || !(16..=512).contains(&block_size) {
        anyhow::bail!("--block-size must be a power of two in 16..=512, got {block_size}");
    }
    // `max_seq_len` is optional (advisory; the KV index grows dynamically as
    // registrants report larger values). When set it seeds the initial index
    // capacity and must be a non-zero multiple of block_size.
    if let Some(max_seq_len) = config.primary.max_seq_len
        && (max_seq_len == 0 || max_seq_len % block_size != 0)
    {
        anyhow::bail!(
            "--max-seq-len ({max_seq_len}) must be a non-zero multiple of \
             --block-size ({block_size})"
        );
    }
    if config.primary.g2_memory_gib.is_none() && config.primary.g2_blocks.is_none() {
        anyhow::bail!("at least one of --g2-memory / --g2-block is required");
    }

    let enabled = parse_features(cli.features.as_deref())?;

    // Reconcile implicit enablers with the explicit feature set.
    if cli.prefill_vllm_url.is_some() && !enabled.contains(&FeatureKey::ConditionalDisagg) {
        anyhow::bail!(
            "--prefill-vllm-url enables the conditional_disagg dispatcher but \
             conditional_disagg is not in --features"
        );
    }

    // KV indexer: resolve sizing from primary; carry ZMQ / advertise overrides.
    // Enabled iff selected in the feature set.
    if enabled.contains(&FeatureKey::KvIndexer) {
        let existing = config.kv_indexer.take().unwrap_or_default();
        config.kv_indexer = Some(kvbm_hub::KvIndexerConfig {
            max_seq_len: config.primary.max_seq_len,
            block_size: Some(block_size),
            zmq_bind: cli.kv_index_zmq_bind.clone().or(existing.zmq_bind),
            advertise_host: cli
                .kv_index_advertise_host
                .clone()
                .or(existing.advertise_host)
                .or_else(|| config.primary.advertise_host.clone()),
        });
    } else {
        config.kv_indexer = None;
    }

    Ok(ResolvedConfig { config, enabled })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let ResolvedConfig { config, enabled } = build_config(&cli)?;

    tracing::info!(
        bind_addr = %config.bind_addr,
        discovery_port = config.discovery_port,
        control_port = config.control_port,
        velo_port = ?config.velo_port,
        registration_ttl_secs = config.registration_ttl_secs,
        prune_interval_secs = config.prune_interval_secs,
        block_size = ?config.primary.block_size,
        max_seq_len = ?config.primary.max_seq_len,
        block_layout = config.primary.block_layout.as_label(),
        features = ?enabled.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "starting kvbm-hub"
    );

    let mut builder = kvbm_hub::create_server_builder()
        .bind_addr(config.bind_addr)
        .discovery_port(config.discovery_port)
        .control_port(config.control_port)
        .registration_ttl(Duration::from_secs(config.registration_ttl_secs))
        .prune_interval(Duration::from_secs(config.prune_interval_secs))
        .heartbeat_interval(Duration::from_secs(config.heartbeat_interval_secs))
        .heartbeat_max_failures(config.heartbeat_max_failures)
        .primary_config(config.primary.clone());

    // ConnectorControl is infrastructure — always attached.
    let cpm = Arc::new(kvbm_hub::ControlPlaneManager::new());

    // P2P (gate-only). Construct first so CPM can be wired to it for
    // describe-push layout-compat validation (c5). CD depends on P2P, so
    // `parse_features` guarantees P2P is enabled whenever CD is.
    if enabled.contains(&FeatureKey::P2P) {
        let p2p_manager = Arc::new(kvbm_hub::P2pManager::new());
        cpm.set_p2p_manager(Arc::clone(&p2p_manager));
        builder = builder.add_feature_manager(p2p_manager as Arc<dyn kvbm_hub::FeatureManager>);
    }

    // ConditionalDisagg, optionally with the HTTP prefill dispatcher.
    if enabled.contains(&FeatureKey::ConditionalDisagg) {
        let cd_manager = match (&cli.prefill_vllm_url, &cli.prefill_vllm_model) {
            (Some(url), Some(model)) => {
                let dispatcher = kvbm_hub::HttpVllmDispatcher::new(url.clone(), model.clone())?;
                tracing::info!(
                    prefill_url = %url,
                    prefill_model = %model,
                    "CD prefill dispatcher enabled (HTTP → vLLM frontend)"
                );
                kvbm_hub::ConditionalDisaggManager::new()
                    .with_dispatcher(dispatcher as Arc<dyn kvbm_hub::PrefillRequestDispatcher>)
            }
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!(
                    "--prefill-vllm-url and --prefill-vllm-model must be specified together"
                );
            }
            (None, None) => {
                tracing::info!(
                    "CD prefill dispatcher disabled (set --prefill-vllm-url + --prefill-vllm-model to enable)"
                );
                kvbm_hub::ConditionalDisaggManager::new()
            }
        };
        builder =
            builder.add_feature_manager(Arc::new(cd_manager) as Arc<dyn kvbm_hub::FeatureManager>);
    }

    builder = builder.add_feature_manager(cpm as Arc<dyn kvbm_hub::FeatureManager>);

    // Optional KV indexer feature. `max_seq_len` is the *initial* index
    // capacity (0 = start empty); it grows as registrants report larger values.
    if let Some(kvi) = &config.kv_indexer {
        let max_seq_len = kvi.max_seq_len.unwrap_or(0);
        let block_size = kvi
            .block_size
            .expect("kv_indexer.block_size resolved by build_config");
        let manager = kvbm_hub::KvIndexerManager::new(
            max_seq_len,
            block_size,
            kvi.zmq_bind.clone(),
            kvi.advertise_host.clone(),
        )?;
        tracing::info!(max_seq_len, block_size, "KV indexer feature enabled");
        builder =
            builder.add_feature_manager(Arc::new(manager) as Arc<dyn kvbm_hub::FeatureManager>);
    }

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
