// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake G4 transfer benchmark with model-shaped KV blocks.
//!
//! Drives [`MooncakeObjectBlockClient`] put / has / get for KV blocks whose
//! geometry is read from a HuggingFace `config.json` (default: Qwen3-30B-A3B,
//! 48 layers x 4 KV heads x 128 head_dim, bf16 = 96 KiB per token).
//!
//! Run one `holder` process that contributes the global segment and one
//! `client` process with no segment of its own, so every byte crosses the
//! configured transport instead of landing in the client's local segment:
//!
//! ```bash
//! mooncake_master --enable_http_metadata_server=true \
//!     --http_metadata_server_port=8080 --rpc_port=50051 &
//! bench_mooncake --role holder --segment-gib 64 &
//! bench_mooncake --model-dir /path/to/Qwen3-30B-A3B --layouts fc,lw
//! bench_mooncake --model-dir /path/to/Qwen3-30B-A3B --layouts fc --zero-copy
//! ```
//!
//! Each batch moves `prefix_tokens / page_size` blocks, so a row reads as
//! "time to offload / onboard an N-token prefix for one TP rank".

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};

use kvbm_config::{MooncakeObjectConfig, MooncakeProtocol};
use kvbm_engine::object::mooncake::client::MooncakeObjectBlockClient;
use kvbm_engine::object::{LayoutConfigExt, ObjectBlockOps};
use kvbm_engine::{BlockId, SequenceHash};
use kvbm_physical::layout::{LayoutConfig, PhysicalLayout};
use kvbm_physical::testing::{
    create_fc_layout_system_with_config, create_lw_layout_system_with_config,
};
use kvbm_physical::transfer::{FillPattern, compute_block_checksums, fill_blocks};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Role {
    /// Contribute a global segment and idle until killed.
    Holder,
    /// Run the benchmark without contributing memory.
    Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Protocol {
    Tcp,
    Rdma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LayoutKind {
    /// Fully contiguous: one region per block.
    Fc,
    /// Layer separate: one region per (layer, K/V) per block.
    Lw,
}

#[derive(Parser, Debug)]
#[command(about = "Mooncake G4 transfer benchmark with model-shaped KV blocks")]
struct Args {
    #[arg(long, value_enum, default_value_t = Role::Client)]
    role: Role,

    #[arg(long, default_value = "http://127.0.0.1:8080/metadata")]
    metadata_server: String,

    #[arg(long, default_value = "127.0.0.1:50051")]
    master_server: String,

    /// Address other Mooncake peers use to reach this process
    /// (use the RDMA NIC's IP for `--protocol rdma`).
    #[arg(long, default_value = "localhost")]
    hostname: String,

    #[arg(long, value_enum, default_value_t = Protocol::Tcp)]
    protocol: Protocol,

    /// RDMA device list, e.g. `mlx5_1` or `mlx5_1,mlx5_2` (empty = auto).
    #[arg(long, default_value = "")]
    device: String,

    /// Global segment contributed by this process (required for `holder`).
    #[arg(long, default_value_t = 0)]
    segment_gib: u64,

    #[arg(long, default_value_t = 1024)]
    local_buffer_mib: u64,

    /// HuggingFace model directory; KV geometry is read from its config.json.
    /// Defaults to the Qwen3-30B-A3B geometry when omitted.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Tensor-parallel degree; KV heads are split across ranks.
    #[arg(long, default_value_t = 1)]
    tp: usize,

    #[arg(long, value_delimiter = ',', default_value = "16,32,64")]
    page_sizes: Vec<usize>,

    /// Prefix lengths in tokens; each batch moves `tokens / page_size` blocks.
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384,32768")]
    prefix_tokens: Vec<usize>,

    #[arg(long, value_enum, value_delimiter = ',', default_value = "fc,lw")]
    layouts: Vec<LayoutKind>,

    /// Use put_from/get_into on a registered layout (fully contiguous only).
    #[arg(long)]
    zero_copy: bool,

    #[arg(long, default_value_t = 16)]
    concurrency: usize,

    #[arg(long, default_value_t = 5)]
    iters: usize,

    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

/// Per-rank KV cache geometry.
#[derive(Debug)]
struct KvGeometry {
    name: String,
    num_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    dtype_bytes: usize,
}

impl KvGeometry {
    fn qwen3_30b_a3b() -> Self {
        Self {
            name: "Qwen3-30B-A3B (built-in)".to_string(),
            num_layers: 48,
            kv_heads: 4,
            head_dim: 128,
            dtype_bytes: 2,
        }
    }

    fn from_model_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)?;
        let get = |k: &str| cfg.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);

        let num_layers = get("num_hidden_layers").context("num_hidden_layers missing")?;
        let heads = get("num_attention_heads").context("num_attention_heads missing")?;
        let kv_heads = get("num_key_value_heads").unwrap_or(heads);
        let head_dim = get("head_dim")
            .or_else(|| get("hidden_size").map(|h| h / heads))
            .context("head_dim / hidden_size missing")?;
        let dtype_bytes = match cfg.get("torch_dtype").and_then(|v| v.as_str()) {
            Some("float32") => 4,
            Some("float16") | Some("bfloat16") | None => 2,
            Some(other) => bail!("unsupported torch_dtype {other}"),
        };
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string());

        Ok(Self {
            name,
            num_layers,
            kv_heads,
            head_dim,
            dtype_bytes,
        })
    }

    /// KV heads held by one rank (heads are replicated when tp > kv_heads).
    fn kv_heads_per_rank(&self, tp: usize) -> usize {
        (self.kv_heads / tp).max(1)
    }

    fn inner_dim(&self, tp: usize) -> usize {
        self.kv_heads_per_rank(tp) * self.head_dim
    }

    fn bytes_per_token(&self, tp: usize) -> usize {
        self.num_layers * 2 * self.inner_dim(tp) * self.dtype_bytes
    }
}

struct Stats {
    p50: Duration,
    p99: Duration,
    mean: Duration,
}

impl Stats {
    fn from(mut v: Vec<Duration>) -> Self {
        v.sort();
        let n = v.len();
        Self {
            p50: v[n / 2],
            p99: v[((n as f64 * 0.99) as usize).min(n - 1)],
            mean: v.iter().sum::<Duration>() / n as u32,
        }
    }
}

struct Row {
    layout: LayoutKind,
    page_size: usize,
    blocks: usize,
    bytes: usize,
    put: Stats,
    get: Stats,
    has: Stats,
}

fn gbps(bytes: usize, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / 1e9
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    ensure!(args.iters > 0, "--iters must be > 0");
    ensure!(args.tp > 0, "--tp must be > 0");

    let config = MooncakeObjectConfig {
        metadata_server: args.metadata_server.clone(),
        master_server_addr: args.master_server.clone(),
        local_hostname: args.hostname.clone(),
        protocol: match args.protocol {
            Protocol::Tcp => MooncakeProtocol::Tcp,
            Protocol::Rdma => MooncakeProtocol::Rdma,
        },
        device_name: args.device.clone(),
        global_segment_size: args.segment_gib << 30,
        local_buffer_size: args.local_buffer_mib << 20,
        namespace: Some(format!("bench-{}", std::process::id())),
        max_concurrent_requests: args.concurrency,
    };

    match args.role {
        Role::Holder => run_holder(&args, &config),
        Role::Client => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(run_client(&args, &config)),
    }
}

fn run_holder(args: &Args, config: &MooncakeObjectConfig) -> Result<()> {
    ensure!(args.segment_gib > 0, "--segment-gib must be > 0 for --role holder");
    let _client = MooncakeObjectBlockClient::new(config)?;
    println!(
        "holder ready: {} GiB segment, protocol={:?} device={:?} hostname={}",
        args.segment_gib, args.protocol, args.device, args.hostname
    );
    loop {
        std::thread::park();
    }
}

async fn run_client(args: &Args, config: &MooncakeObjectConfig) -> Result<()> {
    let geo = match &args.model_dir {
        Some(dir) => KvGeometry::from_model_dir(dir)?,
        None => KvGeometry::qwen3_30b_a3b(),
    };
    let client = MooncakeObjectBlockClient::new(config)?.with_zero_copy(args.zero_copy);

    println!(
        "model={} layers={} kv_heads/rank={} head_dim={} dtype_bytes={} tp={} -> {} KiB/token",
        geo.name,
        geo.num_layers,
        geo.kv_heads_per_rank(args.tp),
        geo.head_dim,
        geo.dtype_bytes,
        args.tp,
        geo.bytes_per_token(args.tp) / 1024
    );
    println!(
        "protocol={:?} device={:?} zero_copy={} concurrency={} iters={} warmup={}",
        args.protocol, args.device, args.zero_copy, args.concurrency, args.iters, args.warmup
    );

    // Unique per-run key space so repeated runs never collide.
    let mut next_key: u64 = (std::process::id() as u64) << 32;
    print_header();

    for &kind in &args.layouts {
        if args.zero_copy && kind == LayoutKind::Lw {
            eprintln!("skipping lw: zero-copy only applies to fully-contiguous layouts");
            continue;
        }
        for &page_size in &args.page_sizes {
            let batches: Vec<usize> = args
                .prefix_tokens
                .iter()
                .map(|t| t.div_ceil(page_size))
                .collect();
            let max_blocks = batches.iter().copied().max().context("no prefix tokens")?;

            let layout_config = LayoutConfig::builder()
                .num_blocks(max_blocks)
                .num_layers(geo.num_layers)
                .outer_dim(2)
                .page_size(page_size)
                .inner_dim(geo.inner_dim(args.tp))
                .dtype_width_bytes(geo.dtype_bytes)
                .build()?;
            let layout = match kind {
                LayoutKind::Fc => create_fc_layout_system_with_config(layout_config),
                LayoutKind::Lw => create_lw_layout_system_with_config(layout_config),
            };
            let all_ids: Vec<BlockId> = (0..max_blocks).collect();
            fill_blocks(&layout, &all_ids, FillPattern::Sequential)?;

            if args.zero_copy {
                client.register_layout(&layout)?;
            }
            for &blocks in &batches {
                let row = bench_case(&client, &layout, kind, blocks, args, &mut next_key).await?;
                print_row(&row);
            }
            if args.zero_copy {
                client.unregister_layout(&layout)?;
            }
        }
    }
    Ok(())
}

async fn bench_case(
    client: &MooncakeObjectBlockClient,
    layout: &PhysicalLayout,
    kind: LayoutKind,
    blocks: usize,
    args: &Args,
    next_key: &mut u64,
) -> Result<Row> {
    let config = layout.layout().config();
    let block_bytes = config.block_size_bytes();
    let block_ids: Vec<BlockId> = (0..blocks).collect();
    let mut put_lat = Vec::with_capacity(args.iters);
    let mut get_lat = Vec::with_capacity(args.iters);
    let mut has_lat = Vec::with_capacity(args.iters);

    for iter in 0..args.warmup + args.iters {
        let keys: Vec<SequenceHash> = (0..blocks)
            .map(|pos| {
                *next_key += 1;
                SequenceHash::new(*next_key, None, pos as u64)
            })
            .collect();

        // Verify the first round trip end to end; later iterations are timed only.
        let expected = if iter == 0 {
            Some(compute_block_checksums(layout, &block_ids)?)
        } else {
            None
        };

        let t = Instant::now();
        let put = client
            .put_blocks_with_layout(keys.clone(), layout.clone(), block_ids.clone())
            .await;
        let put_dt = t.elapsed();
        let failed = put.iter().filter(|r| r.is_err()).count();
        ensure!(failed == 0, "{failed}/{blocks} puts failed");

        let t = Instant::now();
        let has = client.has_blocks(keys.clone()).await;
        let has_dt = t.elapsed();
        ensure!(
            has.iter().all(|(_, size)| *size == Some(block_bytes)),
            "has_blocks did not report every block with size {block_bytes}"
        );

        if expected.is_some() {
            fill_blocks(layout, &block_ids, FillPattern::Constant(0))?;
        }

        let t = Instant::now();
        let get = client
            .get_blocks_with_layout(keys, layout.clone(), block_ids.clone())
            .await;
        let get_dt = t.elapsed();
        let failed = get.iter().filter(|r| r.is_err()).count();
        ensure!(failed == 0, "{failed}/{blocks} gets failed");

        if let Some(expected) = expected {
            let actual = compute_block_checksums(layout, &block_ids)?;
            ensure!(expected == actual, "data mismatch after put/get round trip");
        }

        if iter >= args.warmup {
            put_lat.push(put_dt);
            get_lat.push(get_dt);
            has_lat.push(has_dt);
        }
    }

    Ok(Row {
        layout: kind,
        page_size: config.page_size,
        blocks,
        bytes: blocks * block_bytes,
        put: Stats::from(put_lat),
        get: Stats::from(get_lat),
        has: Stats::from(has_lat),
    })
}

fn print_header() {
    println!(
        "| layout | page | blocks | tokens | MiB | put p50 ms | put p99 ms | put GB/s | get p50 ms | get p99 ms | get GB/s | has p50 ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
}

fn print_row(r: &Row) {
    println!(
        "| {:?} | {} | {} | {} | {:.0} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
        r.layout,
        r.page_size,
        r.blocks,
        r.blocks * r.page_size,
        r.bytes as f64 / (1 << 20) as f64,
        ms(r.put.p50),
        ms(r.put.p99),
        gbps(r.bytes, r.put.mean),
        ms(r.get.p50),
        ms(r.get.p99),
        gbps(r.bytes, r.get.mean),
        ms(r.has.p50),
    );
}
