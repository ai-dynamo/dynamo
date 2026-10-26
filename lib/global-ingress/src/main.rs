// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_ingress`.
//!
//! ```bash
//! # Pools from the Global Discovery Service:
//! cargo run -p dynamo-global-ingress -- --port 8000 \
//!     --discovery-url http://global-discovery:8080/
//! # Or pools given by hand:
//! cargo run -p dynamo-global-ingress -- --port 8000 \
//!     --pool ohio/dynamo/mocker-1=Qwen/Qwen3-0.6B@http://10.42.0.5:8000
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser, ValueEnum};
use dynamo_global_discovery_client::GlobalDiscoveryClient;
use dynamo_global_ingress::{ModelSync, PoolForwardingEngine};
use dynamo_global_router::{
    DiscoveryPoolCatalog, InFlightTracker, LeastInFlightSelector, Pool, PoolCatalog, PoolSelector,
    RoundRobinSelector, StaticPoolCatalog,
};
use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use dynamo_llm::http::service::service_v2::HttpService;
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Parser)]
#[command(about = "Serve the OpenAI API and forward each request to a pool frontend")]
#[command(group(ArgGroup::new("pool_source").required(true).args(["discovery_url", "pools"])))]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Root of the Global Discovery Service to read pools from, for example
    /// `http://global-discovery:8080/`.
    #[arg(long)]
    discovery_url: Option<Url>,
    /// How often to read pools from the Global Discovery Service. A read
    /// that takes longer is abandoned.
    #[arg(long, default_value_t = 5)]
    discovery_interval_secs: u64,
    /// A pool as `<site>/<namespace>/<DGD name>=<model>@<frontend URL>`, in
    /// place of `--discovery-url`. Separate several frontend URLs with commas,
    /// and repeat the flag for each pool. The site is also the pool's region.
    #[arg(long = "pool", value_parser = parse_pool)]
    pools: Vec<Pool>,
    /// How long to wait for a pool frontend to accept a connection.
    #[arg(long, default_value_t = 5)]
    connect_timeout_secs: u64,
    /// How to choose the pool for each request.
    #[arg(long, value_enum, default_value_t = RouterPolicy::RoundRobin)]
    router_policy: RouterPolicy,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RouterPolicy {
    /// Rotate through the pools serving the request's model.
    RoundRobin,
    /// The pool with the fewest requests in flight from this process.
    LeastInFlight,
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape =
        || format!("expected <site>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}");
    let (key, served) = value.split_once('=').ok_or_else(shape)?;
    let [site, namespace, dgd] = key.split('/').collect::<Vec<_>>()[..] else {
        return Err(shape());
    };
    let (model, frontends) = served.split_once('@').ok_or_else(shape)?;
    let key = PoolKey::new(site, namespace, dgd).map_err(|error| error.to_string())?;
    let frontends = frontends
        .split(',')
        .map(|url| {
            url.parse::<Url>()
                .map_err(|error| format!("{url}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Pool {
        id: V1PoolIdDeriver.derive(&key),
        model: model.to_owned(),
        location: PoolLocation {
            region: site.to_owned(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        frontends,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let mut ids = HashSet::new();
    if let Some(repeated) = args.pools.iter().find(|pool| !ids.insert(&pool.id)) {
        bail!("--pool {} is given twice", repeated.id);
    }

    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal.cancel();
    });

    let catalog: Box<dyn PoolCatalog> = match args.discovery_url {
        Some(url) => {
            let interval = Duration::from_secs(args.discovery_interval_secs);
            let client = GlobalDiscoveryClient::new(url, interval)
                .context("build the Global Discovery client")?;
            Box::new(DiscoveryPoolCatalog::start(client, interval, shutdown.child_token()).await)
        }
        None => Box::new(StaticPoolCatalog::new(args.pools)),
    };
    let service = HttpService::builder()
        .host(args.host.clone())
        .port(args.port)
        .cancel_token(Some(shutdown.clone()))
        .build()
        .context("build the HTTP service")?;
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .build()
        .context("build the pool HTTP client")?;
    let in_flight = Arc::new(InFlightTracker::new());
    let selector: Arc<dyn PoolSelector> = match args.router_policy {
        RouterPolicy::RoundRobin => Arc::new(RoundRobinSelector::new()),
        RouterPolicy::LeastInFlight => Arc::new(LeastInFlightSelector::new(in_flight.clone())),
    };
    let engine = Arc::new(PoolForwardingEngine::new(
        catalog.subscribe(),
        selector,
        in_flight,
        http,
    ));

    // Register the models before serving, so the first request finds them.
    let pools = catalog.subscribe();
    let mut models = ModelSync::new(service.clone(), engine);
    models.apply(&pools.borrow());
    tokio::spawn(models.follow(pools, shutdown.child_token()));

    tracing::info!(
        host = %args.host,
        port = args.port,
        router_policy = ?args.router_policy,
        "Global Frontend listening"
    );
    service
        .run(shutdown)
        .await
        .context("serve the Global Frontend")
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}
