// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
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

    /// Enable the prefill-router feature. The hub spawns a load-aware
    /// dispatcher that pops requests off the CD prefill queue and routes
    /// them to workers that registered with the
    /// `Feature::PrefillRouter` payload. Both HTTP and velo backends
    /// are supported. Requires the `disagg` feature.
    #[arg(long)]
    prefill_router: bool,

    /// Per-worker in-flight concurrency cap used by the prefill router.
    /// The fleet-wide backpressure shape is preserved by per-worker
    /// semaphores; a worker at the cap is skipped during selection and
    /// the dispatch blocks until *some* worker frees a slot.
    #[arg(long, default_value_t = 4)]
    prefill_worker_concurrency: u32,

    /// Enable the CD prefill-overload circuit breaker (P2). OFF by default —
    /// when off the hub never constructs the breaker, decode stays Calm, and
    /// behavior is byte-identical to today. Requires `--prefill-router`. The
    /// watermarks below tune the hysteresis (defaults mirror `BreakerConfig`).
    #[arg(long)]
    cd_breaker: bool,

    /// Breaker: free-capacity fraction at/below which the breaker trips to WARM.
    #[arg(long, default_value_t = 0.5)]
    cd_breaker_warm_high: f64,

    /// Breaker: free-capacity fraction at/below which the breaker trips to HOT.
    /// Must be `< --cd-breaker-warm-high`.
    #[arg(long, default_value_t = 0.15)]
    cd_breaker_hot_high: f64,

    /// Breaker: free-capacity fraction at/above which it may descend a tier
    /// (debounced). Should be `> --cd-breaker-warm-high` for a no-flap gap.
    #[arg(long, default_value_t = 0.7)]
    cd_breaker_clear_low: f64,

    /// Breaker: consecutive ticks the clear condition must hold before
    /// descending one tier. Trip is immediate; clear is debounced.
    #[arg(long, default_value_t = 3)]
    cd_breaker_clear_debounce_ticks: u32,

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
    /// `p2p,disagg,indexer`. Omitted = all. Dependencies are
    /// auto-included (selecting `disagg` pulls in `p2p`).
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

    /// Override a base connector-config key: `--kvbm <dotted.path>=<value>`
    /// (repeatable). The first path segment may be a profile
    /// (`default`/`leader`/`worker`) or a flat key. Values are parsed as JSON
    /// (`2`, `true`, `{…}`) with a string fallback. Highest precedence among the
    /// two override flags. The merged result is round-trip-validated through the
    /// connector parser and served verbatim as `GET /v1/config`'s `base_config`.
    #[arg(long = "kvbm", value_name = "KEY.PATH=VALUE")]
    kvbm: Vec<String>,

    /// Deep-merge a full `kv_connector_extra_config` JSON object into the served
    /// base config (below individual `--kvbm` overrides).
    #[arg(long = "kvbm-config", value_name = "JSON")]
    kvbm_config: Option<String>,

    /// Deep-merge a `kv_connector_extra_config` document read from a file into
    /// the served base config (below `--kvbm-config` and `--kvbm`). The file
    /// is parsed as TOML when the path ends in `.toml`, otherwise JSON. The
    /// flat / role-overlay shapes accepted by `KvbmConfig::from_figment_with_json`
    /// are supported (see lib/kvbm-config). Lets the operator point the hub
    /// at the same ConfigMap the workers mount: the hub bridges
    /// `cache.host.cache_size_gb` / `num_blocks` into its own
    /// `primary.g2_memory_gib` / `primary.g2_blocks` so the `--g2-memory` /
    /// `--g2-block` validation gate passes without duplicating the value on
    /// the CLI.
    #[arg(long = "kvbm-config-file", value_name = "PATH")]
    kvbm_config_file: Option<PathBuf>,
}

/// All hub features a client can be granted via `--features` (in dependency
/// order). `ConnectorControl` is infrastructure, always attached, not listed.
/// `PrefillRouter` is not selectable here — it is gated by the
/// `--prefill-router` flag so it is always paired with the disagg dispatcher.
const SELECTABLE_FEATURES: [FeatureKey; 3] = [
    FeatureKey::P2P,
    FeatureKey::ConditionalDisagg,
    FeatureKey::Indexer,
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
    /// Operator-supplied default connector config (sparse
    /// `kv_connector_extra_config` JSON), validated; served as `base_config`.
    base_config: serde_json::Value,
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
    // --kvbm-config-file: load TOML or JSON from disk and stage it as a
    // pre-merged base layer before --kvbm-config / --kvbm overrides land.
    // Auto-detected by extension; default to JSON when the suffix is anything
    // else. Bridge `cache.host.cache_size_gb` / `num_blocks` into primary so
    // pointing the hub at the worker's ConfigMap satisfies the g2 gate.
    let kvbm_config_file_blob: Option<String> = match &cli.kvbm_config_file {
        None => None,
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("reading --kvbm-config-file {}", path.display()))?;
            let json_value: serde_json::Value = if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
            {
                let toml_value: toml::Value = toml::from_str(&raw).with_context(|| {
                    format!("parsing --kvbm-config-file {} as TOML", path.display())
                })?;
                serde_json::to_value(toml_value).with_context(|| {
                    format!(
                        "converting --kvbm-config-file {} TOML to JSON",
                        path.display()
                    )
                })?
            } else {
                serde_json::from_str(&raw).with_context(|| {
                    format!("parsing --kvbm-config-file {} as JSON", path.display())
                })?
            };
            if !json_value.is_object() {
                anyhow::bail!("--kvbm-config-file {} must be an object", path.display());
            }
            // Bridge flat-shape cache.host sizes into HubConfig.primary so the
            // gate check below sees what the worker's mount would have given
            // each worker. CLI flags retain priority — only fill if absent.
            if config.primary.g2_memory_gib.is_none()
                && let Some(gib) = json_value
                    .pointer("/cache/host/cache_size_gb")
                    .and_then(|v| v.as_f64())
            {
                config.primary.g2_memory_gib = Some(gib);
            }
            if config.primary.g2_blocks.is_none()
                && let Some(n) = json_value
                    .pointer("/cache/host/num_blocks")
                    .and_then(|v| v.as_u64())
            {
                config.primary.g2_blocks = Some(n as usize);
            }
            Some(json_value.to_string())
        }
    };

    if config.primary.g2_memory_gib.is_none() && config.primary.g2_blocks.is_none() {
        anyhow::bail!(
            "at least one of --g2-memory / --g2-block (or `cache.host.cache_size_gb` / \
             `cache.host.num_blocks` via --kvbm-config-file) is required"
        );
    }

    let enabled = parse_features(cli.features.as_deref())?;

    // Reconcile implicit enablers with the explicit feature set.
    if cli.prefill_router && !enabled.contains(&FeatureKey::ConditionalDisagg) {
        anyhow::bail!(
            "--prefill-router routes CD prefill traffic but \
             disagg is not in --features"
        );
    }
    if cli.prefill_worker_concurrency == 0 {
        anyhow::bail!("--prefill-worker-concurrency must be >= 1");
    }
    if cli.cd_breaker {
        if !cli.prefill_router {
            anyhow::bail!(
                "--cd-breaker (CD prefill-overload circuit breaker) requires --prefill-router \
                 (the breaker senses the router's free-capacity fraction)"
            );
        }
        if !(cli.cd_breaker_hot_high < cli.cd_breaker_warm_high
            && cli.cd_breaker_warm_high < cli.cd_breaker_clear_low)
        {
            anyhow::bail!(
                "--cd-breaker watermarks must satisfy hot_high < warm_high < clear_low \
                 (got hot_high={}, warm_high={}, clear_low={})",
                cli.cd_breaker_hot_high,
                cli.cd_breaker_warm_high,
                cli.cd_breaker_clear_low,
            );
        }
        if cli.cd_breaker_clear_debounce_ticks == 0 {
            anyhow::bail!("--cd-breaker-clear-debounce-ticks must be >= 1");
        }
    }

    // KV indexer: resolve sizing from primary; carry ZMQ / advertise overrides.
    // Enabled iff selected in the feature set.
    if enabled.contains(&FeatureKey::Indexer) {
        let existing = config.indexer.take().unwrap_or_default();
        config.indexer = Some(kvbm_hub::IndexerConfig {
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
        config.indexer = None;
    }

    // Build the operator's base connector config from `--kvbm-config-file` +
    // `--kvbm-config` + `--kvbm`, then round-trip-validate through the same
    // parser the connector uses (CUDA-free; kvbm-config carries no velo/CUDA).
    // Precedence (lowest → highest): file → inline `--kvbm-config` JSON →
    // individual `--kvbm` dotted overrides. Each layer is applied via
    // `apply_overrides` so a single bad value aborts startup with the
    // parser's error.
    let mut base_config = serde_json::json!({});
    if let Some(blob) = kvbm_config_file_blob.as_deref() {
        kvbm_config::overrides::apply_overrides(&mut base_config, Some(blob), &[])
            .context("applying --kvbm-config-file")?;
    }
    kvbm_config::overrides::apply_overrides(
        &mut base_config,
        cli.kvbm_config.as_deref(),
        &cli.kvbm,
    )?;
    kvbm_config::overrides::validate_extra_config(&base_config)
        .map_err(|e| anyhow::anyhow!("--kvbm/--kvbm-config rejected: {e}"))?;

    Ok(ResolvedConfig {
        config,
        enabled,
        base_config,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let ResolvedConfig {
        config,
        enabled,
        base_config,
    } = build_config(&cli)?;

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
        .primary_config(config.primary.clone())
        .base_kvbm_config(base_config);

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

    // ConditionalDisagg + optional prefill router. The router is its
    // own FeatureManager (it owns the fleet selector and the
    // /v1/features/prefill-router/ HTTP surface). The CD manager
    // continues to own the messenger queue; if `--prefill-router` is
    // set we late-bind the router as the CD manager's dispatcher after
    // both have attached.
    let mut cd_for_late_bind: Option<Arc<kvbm_hub::ConditionalDisaggManager>> = None;
    let mut router_for_late_bind: Option<Arc<kvbm_hub::PrefillRouterManager>> = None;
    if enabled.contains(&FeatureKey::ConditionalDisagg) {
        let cd_manager = Arc::new(kvbm_hub::ConditionalDisaggManager::new());
        if cli.prefill_router {
            cd_for_late_bind = Some(Arc::clone(&cd_manager));
        }
        builder = builder
            .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>);
    }

    if cli.prefill_router {
        let block_size = config
            .primary
            .block_size
            .expect("validated above: --block-size required");
        let selector_config = kvbm_hub::SelectorConfig {
            per_worker_concurrency: cli.prefill_worker_concurrency,
            block_size,
        };
        // CD circuit breaker: opt-in (`--cd-breaker`). OFF ⇒ plain `new`, no
        // breaker, no tick task, no tier push — byte-identical to today.
        let router_manager = if cli.cd_breaker {
            let breaker_config = kvbm_hub::BreakerConfig {
                warm_high: cli.cd_breaker_warm_high,
                hot_high: cli.cd_breaker_hot_high,
                clear_low: cli.cd_breaker_clear_low,
                queue_depth_warm: 0,
                queue_depth_hot: 0,
                clear_debounce_ticks: cli.cd_breaker_clear_debounce_ticks,
            };
            tracing::info!(
                warm_high = cli.cd_breaker_warm_high,
                hot_high = cli.cd_breaker_hot_high,
                clear_low = cli.cd_breaker_clear_low,
                clear_debounce_ticks = cli.cd_breaker_clear_debounce_ticks,
                "CD prefill-overload circuit breaker ENABLED"
            );
            kvbm_hub::PrefillRouterManager::with_breaker(selector_config, breaker_config)
        } else {
            kvbm_hub::PrefillRouterManager::new(selector_config)
        };
        tracing::info!(
            per_worker_concurrency = cli.prefill_worker_concurrency,
            block_size,
            cd_breaker = cli.cd_breaker,
            "prefill router feature enabled (continuous fleet membership)"
        );
        router_for_late_bind = Some(Arc::clone(&router_manager));
        builder = builder
            .add_feature_manager(Arc::clone(&router_manager) as Arc<dyn kvbm_hub::FeatureManager>);
    } else {
        tracing::info!(
            "prefill router disabled (pass --prefill-router to route CD prefill traffic)"
        );
    }

    builder = builder.add_feature_manager(cpm as Arc<dyn kvbm_hub::FeatureManager>);

    // Optional KV indexer feature. `max_seq_len` is the *initial* index
    // capacity (0 = start empty); it grows as registrants report larger values.
    if let Some(kvi) = &config.indexer {
        let max_seq_len = kvi.max_seq_len.unwrap_or(0);
        let block_size = kvi
            .block_size
            .expect("indexer.block_size resolved by build_config");
        let manager = kvbm_hub::IndexerManager::new(
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

    // Install the prefill router as the CD manager's dispatcher AFTER
    // both have attached. The router accepts dispatch calls right away
    // and queues internally until a worker registers — no startup
    // discovery window needed.
    if let (Some(cd), Some(router)) = (cd_for_late_bind, router_for_late_bind) {
        // P2 CD-breaker tier push: cross-wire the broadcaster (owned by the
        // router, sharing the breaker Arc) and the CD manager (the decode-set
        // provider). Done AFTER both have attached so the velo handle is bound.
        // Only present when `--cd-breaker` was set; otherwise this is skipped
        // and the breaker path stays inert.
        if let Some(broadcaster) = router.broadcaster() {
            broadcaster.set_decode_provider(
                Arc::downgrade(&cd) as std::sync::Weak<dyn kvbm_hub::DecodeSetProvider>
            );
            cd.set_tier_broadcaster(Arc::clone(broadcaster));
            tracing::info!("CD breaker tier-push broadcaster wired to the decode set");
        }
        cd.start_dispatcher(router.dispatcher())
            .map_err(|e| anyhow::anyhow!("install prefill router as CD dispatcher: {e}"))?;
        tracing::info!("prefill router installed as CD prefill dispatcher");
    }

    tokio::signal::ctrl_c().await?;
    tracing::info!("shutting down");
    server.shutdown().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Minimal valid arg vector; callers append `--kvbm*` flags.
    fn args(extra: &[&str]) -> Vec<String> {
        let mut v = vec![
            "kvbm-hub".to_string(),
            "--block-size".to_string(),
            "16".to_string(),
            "--g2-memory".to_string(),
            "4".to_string(),
        ];
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn no_kvbm_flags_yields_empty_base_config() {
        let cli = Cli::parse_from(args(&[]));
        let resolved = build_config(&cli).unwrap();
        assert_eq!(resolved.base_config, json!({}));
    }

    #[test]
    fn kvbm_override_is_applied_typed_and_validated() {
        let cli = Cli::parse_from(args(&["--kvbm", "leader.tokio.worker_threads=4"]));
        let resolved = build_config(&cli).unwrap();
        // Parsed as an integer and nested under the leader profile.
        assert_eq!(
            resolved.base_config["leader"]["tokio"]["worker_threads"],
            json!(4)
        );
    }

    #[test]
    fn kvbm_config_blob_then_override_precedence() {
        let cli = Cli::parse_from(args(&[
            "--kvbm-config",
            r#"{"leader":{"tokio":{"worker_threads":8}}}"#,
            "--kvbm",
            "leader.tokio.worker_threads=3",
        ]));
        let resolved = build_config(&cli).unwrap();
        assert_eq!(
            resolved.base_config["leader"]["tokio"]["worker_threads"],
            json!(3)
        );
    }

    #[test]
    fn invalid_kvbm_override_aborts_startup() {
        // worker_threads must be an integer; a string fails the connector parser.
        let cli = Cli::parse_from(args(&["--kvbm", "leader.tokio.worker_threads=notanint"]));
        assert!(build_config(&cli).is_err());
    }
}
