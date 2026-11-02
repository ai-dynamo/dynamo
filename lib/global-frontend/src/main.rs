// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_frontend` and the crate's README.
//!
//! With `--use-selection-core --replica-sync`, the Global Frontend shares its
//! requests in flight and session pins with the other Global Frontends over
//! the gRPC `ReplicaSync` stream, served on the API port. Peers come from
//! Global Discovery (`--global-discovery`, with `--replica-url` naming this
//! frontend) or `--peer`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
use dynamo_global_discovery_client::GlobalDiscoveryClient;
use dynamo_global_frontend::{
    Metrics, PoolSnapshot, RelayGatedPoolCatalog, app, app_with_selection_core, discover_pools,
    follow_relays, measure_round_trips, poll_replica_peers, serve, serve_relay_views,
};
use dynamo_global_router::{
    AffinityConfig, AffinitySelector, GrpcReplicaSync, InFlightTracker, LeastInFlightConfig,
    LeastInFlightSelector, Pool, PoolSelector, PoolSignals, RoundRobinSelector,
    SelectionCorePolicy, SelectionCoreSelector,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

/// How often the round trip to each pool is measured.
const ROUND_TRIP_INTERVAL: Duration = Duration::from_secs(30);
/// How often expired session pins are dropped.
const PIN_SWEEP_INTERVAL: Duration = Duration::from_secs(60);
/// How often the replicas are listed from Global Discovery: the cadence of
/// the frontend's other periodic reads of the fleet (`ROUND_TRIP_INTERVAL`).
const DISCOVERY_POLL_INTERVAL: Duration = ROUND_TRIP_INTERVAL;
const DISCOVERY_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// How each request picks a pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RouterPolicy {
    /// Every pool serving the model in turn.
    RoundRobin,
    /// The nearest pool with the fewest requests in flight over its capacity
    /// (from `--pool-relay`), moving only for much more room.
    LeastInFlight,
    /// Keep each session on one pool, nearest region first, and move it only
    /// for a pool with much more room (see `--headroom-factor`). Reads each
    /// pool's load from its Relay (`--pool-relay`).
    Affinity,
}

#[derive(Parser)]
#[command(about = "Serve the OpenAI API and forward each request to a pool frontend")]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Accept Relay-initiated presence streams on this port and route only
    /// to pools with a live Relay view. Uses a trusted network, without mTLS.
    #[arg(long)]
    relay_view_port: Option<u16>,
    /// Read candidate pools from Global Discovery and require live Relay views.
    /// Router discovery must use a probe independent of pool availability (such as /live).
    #[arg(long, conflicts_with = "pools", requires = "relay_view_port")]
    global_discovery_url: Option<Url>,
    /// A pool as `<cluster>/<namespace>/<DGD name>=<model>@<frontend URL>`,
    /// where the cluster is the ID the pool's PoolRelay runs with. Separate
    /// several frontend URLs with commas, and repeat the flag for each pool.
    /// The cluster also names the pool's region.
    #[arg(long = "pool", value_parser = parse_pool, required_unless_present = "global_discovery_url")]
    pools: Vec<Pool>,
    /// How long to wait for a pool frontend to accept a connection.
    #[arg(long, default_value_t = 5)]
    connect_timeout_secs: u64,
    #[arg(long, value_enum, default_value_t = RouterPolicy::RoundRobin)]
    router_policy: RouterPolicy,
    /// Select pools through the embedded router core.
    #[arg(long)]
    use_selection_core: bool,
    /// The region this process runs in. `affinity` and `least-in-flight`
    /// prefer it, and metrics distinguish local requests from requests sent
    /// to other regions.
    #[arg(
        long,
        required_if_eq_any([("router_policy", "affinity"), ("router_policy", "least-in-flight")])
    )]
    region: Option<String>,
    /// A pool's PoolRelay, which streams the pool's load, as
    /// `<cluster>/<namespace>/<DGD name>=<URL>`. Repeat the flag for each pool;
    /// each pool needs a Relay of its own, since a Relay reports the load of
    /// everything it watches as one.
    #[arg(long = "pool-relay", value_parser = parse_pool_relay)]
    pool_relays: Vec<(PoolKey, Url)>,
    /// `affinity` and `least-in-flight`: a request leaves its pool only for
    /// one with more than this many times its spare capacity, or when its
    /// pool is full.
    #[arg(long, default_value_t = 2.0)]
    headroom_factor: f64,
    /// `affinity`: how long a session keeps its pool without a request.
    /// On the router core, at most 65,536 sessions per model are pinned.
    #[arg(long, default_value_t = 1800)]
    session_idle_secs: u64,
    /// Share this replica's requests in flight and session pins with the
    /// other Global Frontends. Needs `--use-selection-core`.
    #[arg(long, requires = "use_selection_core")]
    replica_sync: bool,
    /// `--replica-sync`: another Global Frontend to share requests in
    /// flight and session pins with, as its API address `http://host:port`.
    /// Repeat the flag for each peer. Static peers stay whatever Global
    /// Discovery lists.
    #[arg(long = "peer", value_parser = parse_api_url, requires = "replica_sync")]
    peers: Vec<Url>,
    /// `--replica-sync`: the Global Discovery service's base URL.
    /// Its `/routers` lists every Global Frontend's API URL; every entry but
    /// `--replica-url` becomes a peer, and the list is followed as it
    /// changes.
    #[arg(
        long,
        value_parser = parse_discovery_url,
        requires_all = ["replica_sync", "replica_url"]
    )]
    global_discovery: Option<Url>,
    /// This frontend's own API URL as Global Discovery lists it, so it does
    /// not dial itself.
    #[arg(long, value_parser = parse_api_url, requires = "global_discovery")]
    replica_url: Option<Url>,
}

fn parse_key(key: &str) -> Result<PoolKey, String> {
    let [cluster, namespace, dgd] = key.split('/').collect::<Vec<_>>()[..] else {
        return Err(format!(
            "expected <cluster>/<namespace>/<DGD name>, got {key}"
        ));
    };
    PoolKey::new(cluster, namespace, dgd).map_err(|error| error.to_string())
}

fn parse_url(url: &str, what: &str) -> Result<Url, String> {
    let parsed = url
        .parse::<Url>()
        .map_err(|error| format!("{url}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{what} must be an HTTP(S) URL: {url}"));
    }
    Ok(parsed)
}

fn parse_pool_relay(value: &str) -> Result<(PoolKey, Url), String> {
    let (key, relay) = value
        .split_once('=')
        .ok_or_else(|| format!("expected <cluster>/<namespace>/<DGD name>=<URL>, got {value}"))?;
    let relay = parse_url(relay, "pool relay")?;
    // The Relay client is built without TLS.
    if relay.scheme() != "http" {
        return Err(format!("pool relay must be an http:// URL: {relay}"));
    }
    Ok((parse_key(key)?, relay))
}

fn parse_api_url(value: &str) -> Result<Url, String> {
    dynamo_global_router::replica_sync::peer_url(value).map_err(|error| error.to_string())
}

fn parse_discovery_url(value: &str) -> Result<Url, String> {
    parse_url(value, "global discovery")
}

/// Gives each pool its Relay. Every Relay must name a configured pool, and
/// no two pools may share one.
fn attach_relays(pools: &mut [Pool], relays: &[(PoolKey, Url)]) -> Result<()> {
    for (key, relay) in relays {
        if pools.iter().any(|pool| pool.relay.as_ref() == Some(relay)) {
            bail!("--pool-relay {relay} is given for two pools; each pool needs its own Relay");
        }
        let id = V1PoolIdDeriver.derive(key);
        let Some(pool) = pools.iter_mut().find(|pool| pool.id == id) else {
            bail!("--pool-relay {relay} names a pool no --pool gives");
        };
        if pool.relay.replace(relay.clone()).is_some() {
            bail!("--pool-relay is given twice for pool {id}");
        }
    }
    Ok(())
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape = || {
        format!("expected <cluster>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}")
    };
    let (key, served) = value.split_once('=').ok_or_else(shape)?;
    let (model, frontends) = served.split_once('@').ok_or_else(shape)?;
    if model.trim().is_empty() {
        return Err("pool model must not be empty".to_owned());
    }
    let key = parse_key(key)?;
    let frontends = frontends
        .split(',')
        .map(|url| parse_url(url, "pool frontend"))
        .collect::<Result<Vec<_>, _>>()?;
    // Until pools come from Global Discovery, which reports each pool's
    // region, the cluster names it.
    Ok(Pool::new(&key, model, key.cluster_id(), frontends))
}

fn replica_transport(args: &Args) -> Result<Option<GrpcReplicaSync>> {
    Ok(args.replica_sync.then(|| GrpcReplicaSync {
        peers: args.peers.clone(),
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let mut args = Args::parse();
    let mut ids = HashSet::new();
    if let Some(repeated) = args.pools.iter().find(|pool| !ids.insert(&pool.id)) {
        bail!("--pool {} is given twice", repeated.id);
    }
    attach_relays(&mut args.pools, &args.pool_relays)?;
    if args.relay_view_port.is_some() && !args.pool_relays.is_empty() {
        bail!("--pool-relay cannot be combined with Relay-initiated receiving");
    }
    if !(args.headroom_factor >= 1.0 && args.headroom_factor.is_finite()) {
        bail!("--headroom-factor must be at least 1");
    }
    let replica_transport = replica_transport(&args)?;

    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal.cancel();
    });

    // Keep the explicit catalog sender alive for the static-pool mode.
    let (_catalog, pools) =
        watch::channel(Arc::new(PoolSnapshot::new(std::mem::take(&mut args.pools))));
    let pools = if let Some(url) = args.global_discovery_url.take() {
        let client = dynamo_global_discovery_client::GlobalDiscoveryClient::new(
            url,
            Duration::from_secs(5),
        )?;
        discover_pools(client, Duration::from_secs(5), shutdown.child_token())
    } else {
        pools
    };
    let in_flight = Arc::new(InFlightTracker::new());
    let signals = Arc::new(PoolSignals::new(in_flight.clone()));
    let relay_catalog = if args.relay_view_port.is_some() {
        Some(Arc::new(RelayGatedPoolCatalog::with_signals(
            pools.clone(),
            shutdown.child_token(),
            Some(signals.clone()),
        )))
    } else if !args.pool_relays.is_empty() {
        Some(Arc::new(RelayGatedPoolCatalog::for_subscriptions(
            pools.clone(),
            shutdown.child_token(),
            signals.clone(),
        )))
    } else {
        None
    };
    let receiving = if let Some(port) = args.relay_view_port {
        let listener = tokio::net::TcpListener::bind((args.host.as_str(), port))
            .await
            .context("listen for Relay views")?;
        Some((
            listener,
            Arc::clone(relay_catalog.as_ref().context("missing Relay catalog")?),
        ))
    } else {
        if let Some(catalog) = &relay_catalog {
            tokio::spawn(follow_relays(
                pools.clone(),
                Arc::clone(catalog),
                shutdown.child_token(),
            ));
        }
        None
    };
    let pools = relay_catalog
        .as_ref()
        .map_or(pools, |catalog| catalog.subscribe());
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .build()
        .context("build the pool HTTP client")?;
    let metrics = Metrics::new(args.region.clone(), in_flight.clone());
    let load_signals = || load_signals(&args, &pools, &signals, &shutdown);
    let app = if args.use_selection_core {
        let policy = selection_core_policy(&args, load_signals)?;
        let selection_core = Arc::new(
            SelectionCoreSelector::new(policy, in_flight.clone(), replica_transport).await?,
        );
        metrics.watch_selection_core(selection_core.clone());
        // Wait for the first pool snapshot, or early requests get a 503.
        let (first_snapshot, reconcile) =
            selection_core.reconcile_catalog(pools.clone(), shutdown.child_token());
        tokio::spawn(reconcile);
        first_snapshot
            .await
            .context("the router core stopped before taking the first pool snapshot")?;
        let replica_sync_routes = selection_core.replica_sync_routes();
        // Ending the long-lived replica streams as shutdown begins lets the
        // graceful drain finish with the requests.
        let replication = selection_core.clone();
        let stopping = shutdown.clone();
        tokio::spawn(async move {
            stopping.cancelled().await;
            replication.stop_replication();
        });
        if let Some(discovery) = args.global_discovery.clone() {
            let client = GlobalDiscoveryClient::new(discovery, DISCOVERY_REQUEST_TIMEOUT)
                .context("build the Global Discovery client")?;
            let (listed, peers) = watch::channel(Vec::new());
            tokio::spawn(poll_replica_peers(
                client,
                args.replica_url.clone(),
                DISCOVERY_POLL_INTERVAL,
                listed,
                shutdown.child_token(),
            ));
            let reconcile_peers = selection_core
                .reconcile_peers(peers, shutdown.child_token())
                .context("--global-discovery needs the gRPC replica transport")?;
            tokio::spawn(reconcile_peers);
        }
        let app = app_with_selection_core(pools, selection_core, in_flight, http, metrics);
        match replica_sync_routes {
            // The gRPC stream shares the listener, beside `/v1/...`; the
            // listener speaks HTTP/1 and h2c alike.
            Some(routes) => app.merge(routes),
            None => app,
        }
    } else {
        let selector: Arc<dyn PoolSelector> = match args.router_policy {
            RouterPolicy::RoundRobin => Arc::new(RoundRobinSelector::new()),
            RouterPolicy::LeastInFlight => Arc::new(LeastInFlightSelector::new(
                in_flight.clone(),
                load_signals(),
                args.region.clone().context("--region is required")?,
                args.headroom_factor,
            )),
            RouterPolicy::Affinity => {
                let region = args.region.clone().context("--region is required")?;
                let signals = load_signals();
                let idle = Duration::from_secs(args.session_idle_secs);
                let selector = Arc::new(AffinitySelector::new(
                    signals,
                    region,
                    args.headroom_factor,
                    idle,
                ));
                tokio::spawn(sweep_pins(selector.clone(), shutdown.child_token()));
                selector
            }
        };
        app(pools, selector, in_flight, http, metrics)
    };
    let address = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .with_context(|| format!("listen on {address}"))?;
    tracing::info!(%address, "Global Frontend listening");
    if let Some((relay_listener, catalog)) = receiving {
        let http = async {
            serve(listener, app, shutdown.clone()).await;
            shutdown.cancel();
            Ok::<(), anyhow::Error>(())
        };
        let relay = async {
            let result = serve_relay_views(relay_listener, catalog, shutdown.clone()).await;
            shutdown.cancel();
            result.context("serve Relay views")
        };
        tokio::try_join!(http, relay)?;
        Ok(())
    } else {
        serve(listener, app, shutdown).await;
        Ok(())
    }
}

/// `signals` starts the load signals the policy weighs, if any.
fn selection_core_policy(
    args: &Args,
    signals: impl FnOnce() -> Arc<PoolSignals>,
) -> Result<SelectionCorePolicy> {
    match args.router_policy {
        RouterPolicy::RoundRobin => Ok(SelectionCorePolicy::RoundRobin),
        RouterPolicy::LeastInFlight => {
            Ok(SelectionCorePolicy::LeastInFlight(LeastInFlightConfig {
                signals: signals(),
                region: args.region.clone().context("--region is required")?,
                headroom_factor: args.headroom_factor,
            }))
        }
        RouterPolicy::Affinity => Ok(SelectionCorePolicy::Affinity(AffinityConfig {
            signals: signals(),
            region: args.region.clone().context("--region is required")?,
            headroom_factor: args.headroom_factor,
            idle: Duration::from_secs(args.session_idle_secs),
        })),
    }
}

/// Starts the round-trip probes; the Relay catalog fills the rest.
fn load_signals(
    args: &Args,
    pools: &watch::Receiver<Arc<PoolSnapshot>>,
    signals: &Arc<PoolSignals>,
    shutdown: &CancellationToken,
) -> Arc<PoolSignals> {
    if args.pool_relays.is_empty() && args.relay_view_port.is_none() {
        tracing::warn!(
            "no --pool-relay is given, so no pool's load is known: requests stay on \
             their nearest pool"
        );
    }
    tokio::spawn(measure_round_trips(
        pools.clone(),
        signals.clone(),
        ROUND_TRIP_INTERVAL,
        shutdown.child_token(),
    ));
    signals.clone()
}

/// Drops expired session pins once a `PIN_SWEEP_INTERVAL`, off the request
/// path, until `cancel` fires.
async fn sweep_pins(selector: Arc<AffinitySelector>, cancel: CancellationToken) {
    let mut ticks = tokio::time::interval(PIN_SWEEP_INTERVAL);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticks.tick() => selector.sweep(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_paths_and_multiple_frontends() {
        let pool =
            parse_pool("ohio/dynamo/blue=Qwen/Qwen3@http://blue:8000,https://green:8000").unwrap();
        assert_eq!(pool.model, "Qwen/Qwen3");
        assert_eq!(pool.frontends.len(), 2);
    }

    #[test]
    fn rejects_missing_models_and_non_http_frontends() {
        for value in [
            "ohio/ns/pool=@http://pool:8000",
            "ohio/ns/pool=m@file:///tmp/pool",
            "ohio/ns/pool=m@mailto:user@example.com",
        ] {
            assert!(parse_pool(value).is_err(), "{value}");
        }
    }

    #[test]
    fn parses_pool_relays_and_requires_a_region_for_load_aware_policies() {
        let (key, relay) = parse_pool_relay("ohio/dynamo/blue=http://relay:50051").unwrap();
        assert_eq!(key.dgd_name(), "blue");
        assert_eq!(relay.as_str(), "http://relay:50051/");
        assert!(parse_pool_relay("ohio/dynamo=http://relay:50051").is_err());
        assert!(parse_pool_relay("ohio/dynamo/blue=https://relay:50051").is_err());

        // Each pool needs its own Relay.
        let mut pools = ["blue", "green"]
            .map(|dgd| parse_pool(&format!("ohio/dynamo/{dgd}=qwen@http://{dgd}:8000")).unwrap());
        let relays = ["blue", "green"]
            .map(|dgd| parse_pool_relay(&format!("ohio/dynamo/{dgd}=http://relay:50051")).unwrap());
        assert!(attach_relays(&mut pools, &relays).is_err());
        assert!(attach_relays(&mut pools[1..], &relays[1..]).is_ok());

        let pool = "--pool=ohio/dynamo/blue=qwen@http://blue:8000";
        let affinity = ["global-frontend", pool, "--router-policy", "affinity"];
        assert!(Args::try_parse_from(affinity).is_err());
        let args = Args::try_parse_from([&affinity[..], &["--region", "ohio"]].concat()).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::Affinity);
        let least = [
            "global-frontend",
            pool,
            "--router-policy",
            "least-in-flight",
        ];
        assert!(Args::try_parse_from(least).is_err());
        let args = Args::try_parse_from(["global-frontend", pool]).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::RoundRobin);
    }

    const POOL: &str = "--pool=ohio/dynamo/blue=qwen@http://blue:8000";

    fn transport(rest: &[&str]) -> Result<Option<GrpcReplicaSync>> {
        let args = Args::try_parse_from([&["global-frontend", POOL], rest].concat())?;
        replica_transport(&args)
    }

    const SYNC: [&str; 2] = ["--use-selection-core", "--replica-sync"];

    #[test]
    fn replica_sync_dials_api_addresses_over_grpc() {
        let peers = [
            "--peer",
            "http://10.0.0.2:8000",
            "--peer",
            "http://gf-b:8000",
        ];
        let expected = Some(GrpcReplicaSync {
            peers: vec![
                "http://10.0.0.2:8000".parse().unwrap(),
                "http://gf-b:8000".parse().unwrap(),
            ],
        });
        assert_eq!(
            transport(&[&SYNC[..], &peers[..]].concat()).unwrap(),
            expected
        );
        // Served from startup, with or without peers.
        assert_eq!(
            transport(&SYNC).unwrap(),
            Some(GrpcReplicaSync { peers: Vec::new() })
        );
        // A peer is an API address: tcp:// is refused.
        assert!(transport(&[&SYNC[..], &["--peer", "tcp://10.0.0.2:5600"]].concat()).is_err());
    }

    const DISCOVERY: [&str; 4] = [
        "--global-discovery",
        "http://discovery:8080",
        "--replica-url",
        "http://gf-a:8000",
    ];

    #[test]
    fn discovery_and_replica_flags_need_each_other_and_replica_sync() {
        let args =
            Args::try_parse_from([&["global-frontend", POOL], &SYNC[..], &DISCOVERY[..]].concat())
                .unwrap();
        assert_eq!(
            replica_transport(&args).unwrap(),
            Some(GrpcReplicaSync { peers: Vec::new() })
        );
        assert!(transport(&[&SYNC[..], &DISCOVERY[..2]].concat()).is_err());
        assert!(transport(&[&SYNC[..], &DISCOVERY[2..]].concat()).is_err());
        assert!(transport(&[&["--use-selection-core"][..], &DISCOVERY[..]].concat()).is_err());
        for rejected in [
            &["--replica-sync"][..],
            &["--use-selection-core", "--peer", "http://10.0.0.2:8000"],
            &["--peer", "http://10.0.0.2:8000"],
        ] {
            assert!(transport(rejected).is_err(), "{rejected:?}");
        }
    }

    #[test]
    fn parses_peers_as_api_addresses() {
        assert!(parse_api_url("http://peer:8000").is_ok());
        // TLS peers wait for the transport to carry TLS.
        assert!(parse_api_url("https://peer:8000").is_err());
        assert!(parse_api_url("tcp://peer:5600").is_err());
        assert!(parse_api_url("peer:8000").is_err());
        assert!(parse_discovery_url("https://discovery:8080").is_ok());
        assert!(parse_discovery_url("tcp://discovery:8080").is_err());
    }

    #[test]
    fn requires_a_pool() {
        assert!(Args::try_parse_from(["global-frontend"]).is_err());
        let discovery = [
            "global-frontend",
            "--global-discovery-url",
            "http://discovery:8080",
        ];
        assert!(Args::try_parse_from(discovery).is_err());
        assert!(
            Args::try_parse_from([&discovery[..], &["--relay-view-port", "50051"]].concat())
                .is_ok()
        );
        assert!(
            Args::try_parse_from(
                [
                    &discovery[..],
                    &[
                        "--relay-view-port",
                        "50051",
                        "--pool",
                        "east/dynamo/blue=qwen@http://pool:8000"
                    ],
                ]
                .concat()
            )
            .is_err()
        );
    }
}
