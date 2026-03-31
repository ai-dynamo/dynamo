// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configurable sweep benchmark for the distributed registry.
//!
//! Run with:
//! ```
//! cargo run -p kvbm-registry --bin bench --features bench -- [OPTIONS]
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use kvbm_bench::{BenchTable, LatencyStats, OutputFormat, SweepRunner};
use kvbm_registry::{
    BinaryCodec, HashMapStorage, InProcessHub, NoMetadata, OffloadStatus, QueryType, RegistryCodec,
    ResponseType, Storage, UdsClientTransport, UdsHubTransport, VeloClientTransport,
    VeloHubTransport, hub,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

// ── Transport kind ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TransportKind {
    Tcp,
    Uds,
    Inprocess,
}

impl std::fmt::Display for TransportKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportKind::Tcp => write!(f, "tcp"),
            TransportKind::Uds => write!(f, "uds"),
            TransportKind::Inprocess => write!(f, "inprocess"),
        }
    }
}

// ── Bench mode ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchMode {
    Query,
    Register,
    Mixed,
}

impl std::fmt::Display for BenchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BenchMode::Query => write!(f, "query"),
            BenchMode::Register => write!(f, "register"),
            BenchMode::Mixed => write!(f, "mixed"),
        }
    }
}

// ── Sweep params ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryBenchParams {
    pub transport: TransportKind,
    pub threads: usize,
    pub clients: usize,
    pub storage_size: u64,
    pub batch_size: usize,
    pub query_size: usize,
    pub duration_secs: u64,
    pub mode: BenchMode,
}

// ── Bench result ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct RegistryBenchResult {
    pub params: RegistryBenchParams,
    pub query_rps: f64,
    pub register_rps: f64,
    pub query_latency: Option<LatencyStats>,
    pub register_latency: Option<LatencyStats>,
    pub rss_delta_mb: f64,
    pub cpu_efficiency: f64,
    pub errors: u64,
}

// ── CLI args ───────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "kvbm-registry benchmark")]
struct Args {
    #[arg(long, default_value = "tcp")]
    transport: TransportKind,

    #[arg(long, default_value_t = 4)]
    threads: usize,

    #[arg(long, default_value_t = 1)]
    clients: usize,

    #[arg(long, default_value_t = 100_000)]
    storage_size: u64,

    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    #[arg(long, default_value_t = 16)]
    query_size: usize,

    #[arg(long, default_value_t = 10)]
    duration: u64,

    #[arg(long, default_value = "mixed")]
    mode: BenchMode,

    #[arg(long, default_value = "table")]
    output: OutputFormat,

    #[arg(long)]
    sweep: Option<String>,

    #[arg(long, default_value_t = 2)]
    warmup: u64,
}

// ── Uniform client handle ──────────────────────────────────────────────────

enum ClientHandle {
    Tcp(tokio::sync::Mutex<VeloClientTransport>),
    Uds(tokio::sync::Mutex<UdsClientTransport>),
    InProcess(Arc<InProcessHub>),
}

impl ClientHandle {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self {
            ClientHandle::Tcp(m) => m.lock().await.request(data).await,
            ClientHandle::Uds(m) => m.lock().await.request(data).await,
            ClientHandle::InProcess(hub) => Ok(hub.handle(data)),
        }
    }

    async fn publish(&self, data: &[u8]) -> Result<()> {
        match self {
            ClientHandle::Tcp(m) => m.lock().await.publish(data).await,
            ClientHandle::Uds(m) => m.lock().await.publish(data).await,
            ClientHandle::InProcess(hub) => {
                hub.handle(data);
                Ok(())
            }
        }
    }
}

// ── Pre-populate helper ────────────────────────────────────────────────────

fn make_storage(size: u64) -> HashMapStorage<u64, u64> {
    let s = HashMapStorage::with_capacity(size as usize);
    for i in 0..size {
        s.insert(i, i * 100);
    }
    s
}

// ── Bench runner ───────────────────────────────────────────────────────────

async fn run_single(params: RegistryBenchParams, warmup_secs: u64) -> Result<RegistryBenchResult> {
    let rss_before = kvbm_bench::sysinfo::rss_snapshot();
    let cpu_before = kvbm_bench::sysinfo::cpu_time();

    // Build transport and client handles based on kind
    let (clients_vec, abort_hub): (Vec<Arc<ClientHandle>>, Option<tokio::task::AbortHandle>) =
        match params.transport {
            TransportKind::Tcp => {
                let storage = make_storage(params.storage_size);
                let registry_hub = hub::<u64, u64, NoMetadata, _>(storage).build();

                // Use port 0 → OS picks free port
                let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
                let port = listener.local_addr()?.port();
                drop(listener);

                let mut hub_transport = VeloHubTransport::bind_local(port).await?;
                let jh = tokio::spawn(async move {
                    loop {
                        if registry_hub.process_one(&mut hub_transport).await.is_err() {
                            break;
                        }
                    }
                });

                // Brief wait for hub to accept connections
                tokio::time::sleep(Duration::from_millis(20)).await;

                let mut handles = Vec::new();
                for _ in 0..params.clients {
                    let c = VeloClientTransport::connect_local(port).await?;
                    handles.push(Arc::new(ClientHandle::Tcp(tokio::sync::Mutex::new(c))));
                }
                (handles, Some(jh.abort_handle()))
            }

            TransportKind::Uds => {
                let storage = make_storage(params.storage_size);
                let registry_hub = hub::<u64, u64, NoMetadata, _>(storage).build();

                let (mut hub_transport, sock_path) = UdsHubTransport::bind_temp().await?;
                let jh = tokio::spawn(async move {
                    loop {
                        if registry_hub.process_one(&mut hub_transport).await.is_err() {
                            break;
                        }
                    }
                });

                tokio::time::sleep(Duration::from_millis(20)).await;

                let mut handles = Vec::new();
                for _ in 0..params.clients {
                    let c = UdsClientTransport::connect(&sock_path).await?;
                    handles.push(Arc::new(ClientHandle::Uds(tokio::sync::Mutex::new(c))));
                }
                (handles, Some(jh.abort_handle()))
            }

            TransportKind::Inprocess => {
                let hub_obj = Arc::new(InProcessHub::new());

                // Pre-populate a DashMap for the handler (avoids needing Arc<Storage>)
                let store: Arc<dashmap::DashMap<u64, u64>> = Arc::new(
                    dashmap::DashMap::with_capacity(params.storage_size as usize),
                );
                for i in 0..params.storage_size {
                    store.insert(i, i * 100);
                }

                {
                    let store2 = Arc::clone(&store);
                    let codec = BinaryCodec::<u64, u64, NoMetadata>::new();
                    hub_obj.set_handler(move |data: &[u8]| {
                        if let Some(query) = codec.decode_query(data) {
                            let mut buf = Vec::new();
                            match query {
                                QueryType::CanOffload(keys) => {
                                    let statuses: Vec<_> = keys
                                        .iter()
                                        .map(|k| {
                                            if store2.contains_key(k) {
                                                OffloadStatus::AlreadyStored
                                            } else {
                                                OffloadStatus::Granted
                                            }
                                        })
                                        .collect();
                                    codec
                                        .encode_response(
                                            &ResponseType::CanOffload(statuses),
                                            &mut buf,
                                        )
                                        .ok();
                                }
                                QueryType::Match(keys) => {
                                    let entries: Vec<_> = keys
                                        .iter()
                                        .filter_map(|k| store2.get(k).map(|v| (*k, *v, NoMetadata)))
                                        .collect();
                                    codec
                                        .encode_response(&ResponseType::Match(entries), &mut buf)
                                        .ok();
                                }
                                QueryType::Remove(keys) => {
                                    let count =
                                        keys.iter().filter(|k| store2.contains_key(k)).count();
                                    codec
                                        .encode_response(&ResponseType::Remove(count), &mut buf)
                                        .ok();
                                }
                                QueryType::Touch(keys) => {
                                    codec
                                        .encode_response(&ResponseType::Touch(keys.len()), &mut buf)
                                        .ok();
                                }
                            }
                            buf
                        } else {
                            Vec::new()
                        }
                    });
                }

                let mut handles = Vec::new();
                for _ in 0..params.clients {
                    handles.push(Arc::new(ClientHandle::InProcess(Arc::clone(&hub_obj))));
                }
                (handles, None)
            }
        };

    // Keys for queries
    let query_keys: Vec<u64> = (0..params.query_size as u64).collect();

    // ── Warmup ────────────────────────────────────────────────────────────
    if warmup_secs > 0 {
        let stop_warmup = Arc::new(AtomicBool::new(false));
        let sw2 = Arc::clone(&stop_warmup);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(warmup_secs)).await;
            sw2.store(true, Ordering::Relaxed);
        });

        let mut warmup_set = JoinSet::new();
        for client in &clients_vec {
            let client = Arc::clone(client);
            let stop = Arc::clone(&stop_warmup);
            let keys = query_keys.clone();
            warmup_set.spawn(async move {
                let codec = BinaryCodec::<u64, u64, NoMetadata>::new();
                let mut warmup_idx = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    if warmup_idx & 255 == 0 {
                        tokio::task::yield_now().await;
                    }
                    warmup_idx += 1;
                    let mut buf = Vec::new();
                    codec
                        .encode_query(&QueryType::CanOffload(keys.clone()), &mut buf)
                        .ok();
                    client.request(&buf).await.ok();
                }
            });
        }
        // Wait for warmup tasks to exit cleanly (they exit at top of loop after
        // stop_warmup fires). Do NOT abort_all() — cancelling a task mid-request
        // leaves the shared transport stream in a corrupted state for measurement.
        while warmup_set.join_next().await.is_some() {}
    }

    // ── Measurement ───────────────────────────────────────────────────────
    let stop = Arc::new(AtomicBool::new(false));
    let wall_start = Instant::now();

    {
        let stop2 = Arc::clone(&stop);
        let dur = params.duration_secs;
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(dur)).await;
            stop2.store(true, Ordering::Relaxed);
        });
    }

    let mut task_set = JoinSet::new();
    for client in &clients_vec {
        let client = Arc::clone(client);
        let stop = Arc::clone(&stop);
        let mode = params.mode;
        let batch = params.batch_size;
        let keys = query_keys.clone();

        task_set.spawn(async move {
            let codec = BinaryCodec::<u64, u64, NoMetadata>::new();
            let mut query_samples: Vec<u64> = Vec::new();
            let mut register_samples: Vec<u64> = Vec::new();
            let mut errors = 0u64;
            let mut op_idx = 0u64;

            while !stop.load(Ordering::Relaxed) {
                // Yield every 256 ops so timer/control tasks can run.
                // Necessary for InProcess mode which is pure-CPU with no I/O yields.
                if op_idx & 255 == 0 {
                    tokio::task::yield_now().await;
                }

                let do_query = match mode {
                    BenchMode::Query => true,
                    BenchMode::Register => false,
                    BenchMode::Mixed => op_idx % 2 == 0,
                };
                op_idx += 1;

                if do_query {
                    let t0 = Instant::now();
                    let mut buf = Vec::new();
                    codec
                        .encode_query(&QueryType::CanOffload(keys.clone()), &mut buf)
                        .ok();
                    match client.request(&buf).await {
                        Ok(_) => query_samples.push(t0.elapsed().as_micros() as u64),
                        Err(_) => errors += 1,
                    }
                } else {
                    let t0 = Instant::now();
                    let entries: Vec<(u64, u64, NoMetadata)> =
                        (0..batch as u64).map(|i| (i, i * 10, NoMetadata)).collect();
                    let mut buf = Vec::new();
                    codec.encode_register(&entries, &mut buf).ok();
                    match client.publish(&buf).await {
                        Ok(_) => register_samples.push(t0.elapsed().as_micros() as u64),
                        Err(_) => errors += 1,
                    }
                }
            }

            (query_samples, register_samples, errors)
        });
    }

    let elapsed = wall_start.elapsed();
    let mut all_query: Vec<u64> = Vec::new();
    let mut all_register: Vec<u64> = Vec::new();
    let mut total_errors = 0u64;

    while let Some(res) = task_set.join_next().await {
        match res {
            Ok((q, r, e)) => {
                all_query.extend(q);
                all_register.extend(r);
                total_errors += e;
            }
            Err(_) => total_errors += 1,
        }
    }

    if let Some(ah) = abort_hub {
        ah.abort();
    }

    let elapsed_secs = elapsed.as_secs_f64();
    let query_rps = all_query.len() as f64 / elapsed_secs;
    let register_rps = all_register.len() as f64 / elapsed_secs;
    let query_latency = LatencyStats::from_micros(&all_query);
    let register_latency = LatencyStats::from_micros(&all_register);

    let rss_after = kvbm_bench::sysinfo::rss_snapshot();
    let rss_delta_mb = (rss_after.rss_kb as f64 - rss_before.rss_kb as f64) / 1024.0;

    let cpu_after = kvbm_bench::sysinfo::cpu_time();
    let cpu_us =
        (cpu_after.user_us + cpu_after.system_us) - (cpu_before.user_us + cpu_before.system_us);
    let cpu_efficiency = cpu_us as f64 / (elapsed_secs * 1_000_000.0);

    Ok(RegistryBenchResult {
        params,
        query_rps,
        register_rps,
        query_latency,
        register_latency,
        rss_delta_mb,
        cpu_efficiency,
        errors: total_errors,
    })
}

// ── Output ─────────────────────────────────────────────────────────────────

fn make_table_row(r: &RegistryBenchResult) -> Vec<String> {
    vec![
        r.params.transport.to_string(),
        r.params.threads.to_string(),
        r.params.clients.to_string(),
        r.params.storage_size.to_string(),
        r.params.batch_size.to_string(),
        r.params.query_size.to_string(),
        r.params.mode.to_string(),
        format!("{:.0}", r.query_rps),
        format!("{:.0}", r.register_rps),
        r.query_latency
            .as_ref()
            .map_or("-".into(), |l| l.p50_us.to_string()),
        r.query_latency
            .as_ref()
            .map_or("-".into(), |l| l.p99_us.to_string()),
        r.query_latency
            .as_ref()
            .map_or("-".into(), |l| l.p999_us.to_string()),
        format!("{:.1}", r.rss_delta_mb),
        format!("{:.2}", r.cpu_efficiency),
    ]
}

const TABLE_HEADERS: &[&str] = &[
    "transport",
    "threads",
    "clients",
    "storage",
    "batch",
    "query_sz",
    "mode",
    "q_rps",
    "r_rps",
    "q_p50µs",
    "q_p99µs",
    "q_p999µs",
    "rss_δMB",
    "cpu_eff",
];

fn print_results(results: &[RegistryBenchResult], format: &OutputFormat) {
    match format {
        OutputFormat::Table => {
            let mut table = BenchTable::new(TABLE_HEADERS);
            for r in results {
                table.add_row(&make_table_row(r));
            }
            table.print();
        }
        OutputFormat::Csv => {
            let mut table = BenchTable::new(TABLE_HEADERS);
            for r in results {
                table.add_row(&make_table_row(r));
            }
            print!("{}", table.to_csv());
        }
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(results).unwrap_or_default()
            );
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // Print system info header
    let sysinfo = kvbm_bench::sysinfo::collect();
    eprintln!("{}", sysinfo);

    // Determine sweep points
    let sweep_points: Vec<RegistryBenchParams> = if let Some(ref sweep_path) = args.sweep {
        let runner = SweepRunner::<RegistryBenchParams>::from_yaml_file(sweep_path)?;
        eprintln!("Sweep: {} points", runner.len());
        runner.points
    } else {
        vec![RegistryBenchParams {
            transport: args.transport,
            threads: args.threads,
            clients: args.clients,
            storage_size: args.storage_size,
            batch_size: args.batch_size,
            query_size: args.query_size,
            duration_secs: args.duration,
            mode: args.mode,
        }]
    };

    let warmup = args.warmup;
    let output = args.output.clone();
    let mut all_results = Vec::new();

    for params in sweep_points {
        eprintln!(
            "Running: transport={} threads={} clients={} storage={} batch={} query_sz={} mode={} duration={}s",
            params.transport,
            params.threads,
            params.clients,
            params.storage_size,
            params.batch_size,
            params.query_size,
            params.mode,
            params.duration_secs
        );

        let threads = params.threads;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .enable_all()
            .build()?;

        let result = runtime.block_on(run_single(params, warmup))?;
        all_results.push(result);
    }

    print_results(&all_results, &output);
    Ok(())
}
