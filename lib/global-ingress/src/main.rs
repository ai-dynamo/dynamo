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
use dynamo_global_ingress::{
    ModelSync, PoolForwardingEngine, RoutingStats, follow_relays, measure_round_trips,
};
use dynamo_global_router::{
    AffinitySelector, DiscoveryPoolCatalog, InFlightTracker, LeastInFlightSelector, Overload, Pool,
    PoolCatalog, PoolSelector, PoolSignals, RendezvousSelector, RoundRobinSelector,
    StaticPoolCatalog,
};
use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use dynamo_llm::http::service::axum;
use dynamo_llm::http::service::frontend_extension::FrontendRouteSet;
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
    /// This process's region, as pools' locations name regions. Requests
    /// enter here; the session-aware policies prefer its pools. Required
    /// for `rendezvous` and `affinity`.
    #[arg(long)]
    region: Option<String>,
    /// Port of each pool's Relay, whose address the catalog gives, for its
    /// serving-load stream.
    #[arg(long, default_value_t = 50051)]
    relay_port: u16,
    /// A pool's Relay load older than this is not used.
    #[arg(long, default_value_t = 3000)]
    load_max_age_ms: u64,
    /// A pool is overloaded when its requests in flight per worker exceed
    /// this multiple of the fleet's...
    #[arg(long, default_value_t = Overload::default().ratio)]
    overload_ratio: f64,
    /// ...and exceed the fleet's by at least this many per worker.
    #[arg(long, default_value_t = Overload::default().floor)]
    overload_floor: f64,
    /// `affinity`: forget a session's pool after this long without a request.
    #[arg(long, default_value_t = 1800)]
    pin_idle_secs: u64,
    /// How often to measure the round trip to each pool.
    #[arg(long, default_value_t = 30)]
    round_trip_interval_secs: u64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RouterPolicy {
    /// Rotate through the pools serving the request's model.
    RoundRobin,
    /// The pool with the fewest requests in flight from this process.
    LeastInFlight,
    /// Each session's home pool by hashing, nearest region first; leave
    /// home only while it is overloaded.
    Rendezvous,
    /// Pin each session to a pool; move the pin to the nearest pool with
    /// room when the pinned pool is overloaded.
    Affinity,
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape =
        || format!("expected <site>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}");
    let (key_text, served) = value.split_once('=').ok_or_else(shape)?;
    let [site, namespace, dgd] = key_text.split('/').collect::<Vec<_>>()[..] else {
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
        name: key_text.to_owned(),
        model: model.to_owned(),
        location: PoolLocation {
            region: site.to_owned(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        frontends,
        relay: None,
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
    let in_flight = Arc::new(InFlightTracker::new());
    let signals = Arc::new(PoolSignals::new(
        in_flight.clone(),
        Duration::from_millis(args.load_max_age_ms),
    ));
    let stats = Arc::new(RoutingStats::new(args.region.clone()));
    let overload = Overload {
        ratio: args.overload_ratio,
        floor: args.overload_floor,
    };
    let region = || {
        args.region
            .clone()
            .with_context(|| format!("--router-policy {:?} needs --region", args.router_policy))
    };
    let selector: Arc<dyn PoolSelector> = match args.router_policy {
        RouterPolicy::RoundRobin => Arc::new(RoundRobinSelector::new()),
        RouterPolicy::LeastInFlight => Arc::new(LeastInFlightSelector::new(in_flight.clone())),
        RouterPolicy::Rendezvous => Arc::new(RendezvousSelector::new(
            signals.clone(),
            region()?,
            overload,
        )),
        RouterPolicy::Affinity => Arc::new(AffinitySelector::new(
            signals.clone(),
            region()?,
            overload,
            Duration::from_secs(args.pin_idle_secs),
        )),
    };
    // Every policy follows pool load and round trips, so the stats report
    // them whichever policy runs.
    tokio::spawn(follow_relays(
        catalog.subscribe(),
        signals.clone(),
        args.relay_port,
        shutdown.child_token(),
    ));
    tokio::spawn(measure_round_trips(
        catalog.subscribe(),
        signals.clone(),
        Duration::from_secs(args.round_trip_interval_secs),
        shutdown.child_token(),
    ));

    let report = {
        let (pools, stats, signals) = (catalog.subscribe(), stats.clone(), signals.clone());
        move || {
            let report = stats.report(&pools.borrow(), &signals);
            async move { axum::Json(report) }
        }
    };
    let service = HttpService::builder()
        .host(args.host.clone())
        .port(args.port)
        .cancel_token(Some(shutdown.clone()))
        .add_frontend_route_extension(move |_| {
            Ok(FrontendRouteSet::builder()
                .get("/global-router/stats", report.clone())?
                .build())
        })
        .build()
        .context("build the HTTP service")?;
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .build()
        .context("build the pool HTTP client")?;
    let engine = Arc::new(PoolForwardingEngine::new(
        catalog.subscribe(),
        selector,
        in_flight,
        stats,
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
        region = ?args.region,
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
