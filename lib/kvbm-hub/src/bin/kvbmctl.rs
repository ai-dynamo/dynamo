// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbmctl` — a thin read-only client over the same `GET /v1/config` aggregate
//! the connector consumes. Its one job today is to render a paste-ready vLLM
//! CLI fragment so operators stop hand-writing `--kv-transfer-config` blobs:
//!
//! ```text
//! kvbmctl config vllm --hub http://hub:1337 [--features indexer,p2p,disagg] \
//!         [--role prefill|decode] [--kvbm leader.tokio.worker_threads=2] [--kvbm-config '{…}']
//! ```
//!
//! Top-level subcommands are flat and consistent: `config` (render), the
//! per-feature query groups from [`feature_clis`] (e.g. `indexer`), and `p2p`
//! (transfer actions). There is no `get` wrapper.
//!
//! The hub fills in `block_size` / `max_seq_len` / `block_layout`, advisory
//! `cache.host` sizing, and `leader.hub.{url,features}`; the operator overrides
//! only free fields. Stdout is the CLI fragment; everything else goes to stderr.

use anyhow::{Context, Result, anyhow};
use clap::{Args, Command, FromArgMatches};

use kvbm_hub::render::{VllmRenderOptions, render_vllm_cli};
use kvbm_hub::{HubClientBuilder, feature_clis, hub_arg, p2p_command, run_p2p};

#[derive(Args)]
struct VllmArgs {
    /// Hub discovery base URL, e.g. `http://hub-host:1337`. Falls back to the
    /// `KVBMCTL_HUB` env var.
    #[arg(long, env = "KVBMCTL_HUB")]
    hub: String,

    /// Feature subset to participate in (comma-separated or repeated). Omitted
    /// → the hub's full enabled set. Non-empty → validated ⊆ enabled, with
    /// dependency closure (e.g. `disagg` pulls in `p2p`).
    #[arg(long, value_delimiter = ',')]
    features: Vec<String>,

    /// Conditional-disagg role. Required iff `disagg` is effective.
    #[arg(long)]
    role: Option<String>,

    /// Override a config key: `--kvbm <dotted.path>=<value>` (repeatable). The
    /// first path segment may be a profile (`default`/`leader`/`worker`) or a
    /// flat key. Values are parsed as JSON (`2`, `true`, `{…}`) with a string
    /// fallback. Highest precedence among free fields; hub-authoritative fields
    /// (`default.block_layout`, `leader.hub.{url,features}`, and
    /// `leader.disagg.role` when disagg is active) still always win.
    #[arg(long = "kvbm", value_name = "KEY.PATH=VALUE")]
    kvbm: Vec<String>,

    /// Deep-merge a full `kv_connector_extra_config` JSON object over the
    /// rendered config (below individual `--kvbm` overrides). Hub-authoritative
    /// fields (`default.block_layout`, `leader.hub.{url,features}`, and
    /// `leader.disagg.role` when disagg is active) are re-applied
    /// last and cannot be overridden; free fields can.
    #[arg(long = "kvbm-config", value_name = "JSON")]
    kvbm_config: Option<String>,

    /// vLLM `kv_connector` class name.
    #[arg(long, default_value = "DynamoConnector")]
    kv_connector: String,

    /// vLLM `kv_role`.
    #[arg(long, default_value = "kv_both")]
    kv_role: String,

    /// vLLM `kv_load_failure_policy`.
    #[arg(long, default_value = "recompute")]
    kv_load_failure_policy: String,

    /// vLLM `kv_connector_module_path`.
    #[arg(long, default_value = "kvbm.v2.vllm.connector")]
    kv_connector_module_path: String,
}

/// Build the `kvbmctl` command tree. Top-level subcommands are flat: the typed
/// `config vllm` leaf, one dynamic `<feature>` group per [`feature_clis`] (e.g.
/// `indexer`), and `p2p` — each with `--hub` injected. Hybrid derive
/// (`VllmArgs`) + builder (feature subcommands).
fn build_cli() -> Command {
    let vllm = VllmArgs::augment_args(Command::new("vllm").about(
        "Render a vLLM CLI fragment: --block-size / --max-model-len + --kv-transfer-config",
    ));
    let config = Command::new("config")
        .about("Render connector configuration for a launch target")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(vllm);

    let mut cmd = Command::new("kvbmctl")
        .about("KVBM hub client — render connector configuration / query features")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(config)
        .subcommand(p2p_command().arg(hub_arg()));
    // Per-feature query groups (e.g. `indexer`) sit at the top level alongside
    // `config` and `p2p` — no `get` wrapper.
    for fc in feature_clis() {
        cmd = cmd.subcommand(fc.command().arg(hub_arg()));
    }
    cmd
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let matches = build_cli().get_matches();
    match matches.subcommand() {
        Some(("config", config_m)) => {
            let vllm_m = config_m
                .subcommand_matches("vllm")
                .expect("subcommand_required guarantees `vllm`");
            let args = VllmArgs::from_arg_matches(vllm_m).map_err(|e| anyhow!(e))?;
            render_vllm(args).await
        }
        Some(("p2p", p2p_m)) => {
            let client = hub_client(p2p_m)?;
            let value = run_p2p(&client, p2p_m).await?;
            print_json(&value)
        }
        // Per-feature query groups from `feature_clis()` (e.g. `indexer`).
        Some((feature, feature_m)) => {
            let clis = feature_clis();
            let fc = clis
                .iter()
                .find(|c| c.command().get_name() == feature)
                .ok_or_else(|| anyhow!("unknown subcommand: {feature}"))?;
            let client = hub_client(feature_m)?;
            let value = fc.run(&client, feature_m).await?;
            print_json(&value)
        }
        _ => unreachable!("subcommand_required guarantees a subcommand"),
    }
}

/// Build a [`HubClient`] from the `--hub` (or `KVBMCTL_HUB`) on `matches`.
fn hub_client(matches: &clap::ArgMatches) -> Result<std::sync::Arc<kvbm_hub::HubClient>> {
    let hub_url = matches
        .get_one::<String>("hub")
        .ok_or_else(|| anyhow!("missing required --hub <URL> (or set KVBMCTL_HUB)"))?;
    HubClientBuilder::from_url(hub_url)?
        .build()
        .context("building hub client")
}

fn print_json(value: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

async fn render_vllm(args: VllmArgs) -> Result<()> {
    let client = HubClientBuilder::from_url(&args.hub)?
        .build()
        .context("building hub client")?;
    let aggregate = client
        .get_config()
        .await
        .with_context(|| format!("fetching GET /v1/config from {}", args.hub))?;

    let opts = VllmRenderOptions {
        features: args.features,
        role: args.role,
        kvbm_overrides: args.kvbm,
        kvbm_config: args.kvbm_config,
        kv_connector: args.kv_connector,
        kv_role: args.kv_role,
        kv_load_failure_policy: args.kv_load_failure_policy,
        kv_connector_module_path: args.kv_connector_module_path,
    };

    let cli = render_vllm_cli(&aggregate, &args.hub, &opts)?;
    println!("{cli}");
    Ok(())
}
